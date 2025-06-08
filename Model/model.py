import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass

# Creating a dataclass to store the configurations of the model.

@dataclass
class ModelConfig:
    vocab_size: int = 18000
    d_model: int = 128
    num_heads: int = 8
    batch_size: int = 32
    context_length: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class VectorEmbeddings(nn.Module):
    """
    Returns learnable token embeddings based on input token ids.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the embedding vectors.
        embeddings (nn.Embedding): Embedding layer mapping token ids to embedding vectors.
    """
    def __init__(self, config):
        """
        Args:
            config: A configuration object with attributes:
                - vocab_size (int): Number of tokens.
                - d_model (int): Dimension of the embedding space.
        """
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.embeddings = nn.Embedding(self.vocab_size, self.d_model)


    def forward(self, x): # x: (batch_size, context_length)
        """
        Performs the embedding lookup.
        
        Args:
            x (torch.LongTensor): Input tensor of token indices with shape (batch_size, context_length)

        Returns:
            torch.FloatTensor: Output tensor of embeddings with shape (batch_size, context_length, d_model)
        """
        emb = self.embeddings(x) # (batch_size, context_length, d_model)
        return emb


class PositionalEmbeddings(nn.Module):
    """
    Adds unlearnable sinusoidal positional encodings to input embeddings.

    Attributes:
        pos_emb (Tensor): Precomputed sinusoidal positional embeddings of shape (1, context_length, d_model).
    """
    def __init__(self, config):
        """
        Args:
            config: A configuration object with attributes:
                - context_length (int): Maximum sequence length.
                - d_model (int): Dimension of the embedding space.
        """
        super().__init__()
        
        pe = torch.zeros(config.context_length, config.d_model)  # (context_length, d_model)
        
        position = torch.arange(0, config.context_length, dtype=torch.float).unsqueeze(1)  # (context_length, 1)
        
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / config.d_model)
        )  # (d_model//2,) — used for sine and cosine scaling

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position / div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)  # (1, context_length, d_model)

        # Register as buffer (not a trainable parameter)
        self.register_buffer('pos_emb', pe)
    

    def forward(self, x):
        """
        Adds positional encodings to input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, context_length, d_model).

        Returns:
            Tensor: Output tensor of the same shape with positional encodings added.
        """
        return x + self.pos_emb  # (batch_size, context_length, d_model) + (1, context_length, d_model)


class MLP(nn.Module):
    """
    A simple feedforward MLP block.

    Attributes:
        lin1 (nn.Linear): Linear layer projecting from d_model to 4 * d_model.
        relu (nn.ReLU): ReLU activation function.
        lin2 (nn.Linear): Linear layer projecting from 4 * d_model back to d_model.
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.d_model  # Intermediate hidden layer size
        self.lin1 = nn.Linear(config.d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, config.d_model)


    def forward(self, x):
        """
        Applies the MLP to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, context_length, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, context_length, d_model)
        """
        x = self.lin1(x)  # (batch_size, context_length, 4 * d_model)
        x = self.relu(x)  # (batch_size, context_length, 4 * d_model)
        x = self.lin2(x)  # (batch_size, context_length, d_model)
        return x
    

class AddNorm(nn.Module):
    """
    Residual connection followed by layer normalization.

    Attributes:
        shape (int or tuple): The shape of the input to be normalized. Typically the last dimension.
        eps (float): A small value to avoid division by zero in LayerNorm.
        LayerNorm (nn.LayerNorm): The layer normalization module.
    """
    def __init__(self, shape, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps
        self.LayerNorm = nn.LayerNorm(normalized_shape=self.shape, eps=self.eps)


    def forward(self, sublayer_out, sublayer_in):
        """
        Applies residual connection followed by layer normalization.

        Args:
            x_attn (torch.Tensor): Output from attention mechanism with same shape as x.
            x (torch.Tensor): Original input tensor of shape (batch_size, context_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, context_length, d_model), normalized.
        """
        add_norm = self.LayerNorm(sublayer_out + sublayer_in)  # (batch_size, context_length, d_model)
        return add_norm


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention using parallel computation.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each head.
        d_model (int): Dimensionality of the whole multi-head attention block.
        batch_size (int): Size of the batch.
        context_length (int): Number of tokens in input.
        qkv_proj (nn.Linear): Projects input to queries, keys, and values.
        out_proj (nn.Linear): Final linear projection after attention.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.d_model = config.d_model
        self.batch_size = config.batch_size
        self.context_length = config.context_length

        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)


    def forward(self, x, mask=False):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, context_length, d_model)
            mask (bool or torch.Tensor): Causal mask flag or tensor.

        Returns:
            torch.Tensor: Output of shape (batch_size, context_length, d_model)
        """
        qkv = self.qkv_proj(x)                                                                  # (batch_size, context_length, 3 * d_model)
        qkv = qkv.view(-1, self.context_length, self.num_heads, 3 * self.head_dim)              # (batch_size, context_length, num_heads, 3 * head_dim)
        qkv = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)                                          # (batch_size, num_heads, context_length, head_dim)
        Q, K, V = [t.contiguous() for t in qkv]

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, context_length, context_length)

        if mask:
            mask_tensor = torch.triu(torch.ones(self.context_length, self.context_length, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask_tensor.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, context_length, context_length)
        attn_output = torch.matmul(attn_weights, V)    # (batch_size, num_heads, context_length, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, self.context_length, self.d_model)  # (batch_size, context_length, d_model)

        return self.out_proj(attn_output)  # (batch_size, context_length, d_model)


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention using parallel computation.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each head.
        q_proj (nn.Linear): Projects query input to queries.
        kv_proj (nn.Linear): Projects context input to keys and values.
        out_proj (nn.Linear): Final linear projection after attention.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.d_model = config.d_model

        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.kv_proj = nn.Linear(self.d_model, 2 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    
    def forward(self, q_embeddings, kv_embeddings, mask=None):
        """
        Args:
            q_embeddings (torch.Tensor): Query input of shape (batch_size, target_length, d_model)
            kv_embeddings (torch.Tensor): Key-value input of shape (batch_size, source_length, d_model)
            mask (torch.Tensor or None): Optional mask of shape (batch_size, num_heads, target_length, source_length)

        Returns:
            torch.Tensor: Output of shape (batch_size, target_length, d_model)
        """
        batch_size, target_length, _ = q_embeddings.size()
        source_length = kv_embeddings.size(1)

        # 1. Project queries, keys, and values
        Q = self.q_proj(q_embeddings).view(batch_size, target_length, self.num_heads, self.head_dim).transpose(1, 2)         # (batch_size, num_heads, target_length, head_dim)
        kv = self.kv_proj(kv_embeddings).view(batch_size, source_length, self.num_heads, 2 * self.head_dim).transpose(1, 2)  # (batch_size, num_heads, source_length, 2 * head_dim)
        K, V = kv.chunk(2, dim=-1)

        # 2. Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, target_length, source_length)

        if mask:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, target_length, source_length)
        attn_output = torch.matmul(attn_weights, V)    # (batch_size, num_heads, target_length, head_dim)

        # 3. Recombine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, target_length, self.d_model)

        return self.out_proj(attn_output)  # (batch_size, target_length, d_model)


class Encoder(nn.Module):
    """
    Transformer Encoder block.

    Attributes:
        attn_block (MultiHeadAttention): Multi-head attention mechanism.
        l_norm1 (AddNorm): Layer normalization after attention block.
        lin_layer (MLP): Feed-forward neural network.
        l_norm2 (AddNorm): Layer normalization after MLP block.
    """
    def __init__(self, config):
        """
        Initializes the Encoder module.

        Args:
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.attn_block = MultiHeadAttention(config)
        self.l_norm1 = AddNorm(config.d_model)
        self.lin_layer = MLP(config)
        self.l_norm2 = AddNorm(config.d_model)


    def forward(self, x):  # x: (batch_size, context_length, d_model)
        """
        Forward pass of the Encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, context_length, d_model)

        Returns:
            Tensor: Output tensor of shape (batch_size, context_length, d_model)
        """
        attn_out = self.attn_block(x)                     # (batch_size, context_length, d_model)
        l_norm1_out = self.l_norm1(attn_out, x)           # (batch_size, context_length, d_model)
        lin_out = self.lin_layer(l_norm1_out)             # (batch_size, context_length, d_model)
        l_norm2_out = self.l_norm2(lin_out, l_norm1_out)  # (batch_size, context_length, d_model)
        return l_norm2_out


class Decoder(nn.Module):
    """
    Transformer decoder block consisting of self-attention, cross-attention, 
    and feed-forward layers, each followed by residual and layer normalization.

    Attributes:
        attn_block (MultiHeadAttention): Multi-head self-attention layer.
        l_norm1 (AddNorm): Layer normalization after self-attention.
        cross_attn_block (MultiHeadCrossAttention): Multi-head cross-attention with encoder output.
        l_norm2 (AddNorm): Layer normalization after cross-attention.
        lin_layer (MLP): Feed-forward network.
        l_norm3 (AddNorm): Layer normalization after the feed-forward layer.
    """
    def __init__(self, config):
        """
        Initialize the Decoder with given model configuration.

        Args:
            config (ModelConfig): Configuration with model hyperparameters.
        """
        super().__init__()
        self.attn_block = MultiHeadAttention(config)
        self.l_norm1 = AddNorm(config.d_model)
        self.cross_attn_block = MultiHeadCrossAttention(config)
        self.l_norm2 = AddNorm(config.d_model)
        self.lin_layer = MLP(config)
        self.l_norm3 = AddNorm(config.d_model)


    def forward(self, x, encoder_out): # (batch_size, context_length, d_model)
        """
        Forward pass of the decoder block.

        Args:
            x (torch.Tensor): Decoder input tensor of shape 
                (batch_size, context_length, d_model).
            encoder_out (torch.Tensor): Output from the encoder of shape 
                (batch_size, context_length, d_model).

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers,
                of shape (batch_size, context_length, d_model).
        """
        attn_out = self.attn_block(x, mask=True)  # (batch_size, context_length, d_model)
        l_norm1_out = self.l_norm1(attn_out, x)   # (batch_size, context_length, d_model)

        cross_attn_out = self.cross_attn_block(l_norm1_out, encoder_out)  # (batch_size, context_length, d_model)
        l_norm2_out = self.l_norm2(cross_attn_out, l_norm1_out)           # (batch_size, context_length, d_model)

        lin_out = self.lin_layer(l_norm2_out)             # (batch_size, context_length, d_model)
        l_norm3_out = self.l_norm3(lin_out, l_norm2_out)  # (batch_size, context_length, d_model)

        return l_norm3_out  # (batch_size, context_length, d_model)


class Transformer(nn.Module):
    """
    A Transformer model composed of an encoder and decoder module, followed by a linear projection and a softmax.

    Attributes:
        encoder (nn.Module): The encoder module for input sequence processing.
        decoder (nn.Module): The decoder module that attends to encoder outputs.
        lin (nn.Linear): Linear layer mapping decoder output to logits.
    """
    def __init__(self, config):
        """
        Args:
            config (ModelConfig): Configuration object with model hyperparameters.
        """
        super().__init__()
        self.batch_size = config.batch_size
        self.emb = VectorEmbeddings(config)
        self.pos_emb = PositionalEmbeddings(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.proj = nn.Linear(config.d_model, config.vocab_size)  # (batch_size, context_length, vocab_size)


    def forward(self, x, y):
        """
        Perform a forward pass through the Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, context_length).

        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, context_length, vocab_size).

        Raises:
            RuntimeError: If tensor dimensions are incompatible with expected input.
        """
        emb_outx = self.emb(x)
        pe_outx = self.pos_emb(emb_outx)
        emb_outy = self.emb(y)
        pe_outy = self.pos_emb(emb_outy)
        
        encoder_out = self.encoder(pe_outx)               # (batch_size, context_length, d_model)
        decoder_out = self.decoder(pe_outy, encoder_out)  # (batch_size, context_length, d_model)
        lin_out = self.proj(decoder_out)                  # (batch_size, context_length, vocab_size)
        return lin_out
    

