#Importing the dataset loader from pytorch and the required Tokenizer modules from Hugging Face
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Loading the dataset
training_set, test_set, validation_set = load_dataset("wmt14", "de-en", split=["train", "test", 'validation'])

#---------------------------------------------------------------------------------------------------------------------------------------------
# Helper function to combine German and English text for joint tokenization
def get_training_corpus():
    for example in training_set:
        yield example['translation']['de']  # Lazily producing one value at a time with Yield
        yield example['translation']['en']

#---------------------------------------------------------------------------------------------------------------------------------------------
# Initialize a tokenizer with BPE model using the Hugging Face Tokenizer API
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Instantiating the the BPE tokenizer for learning the tokens
trainer = BpeTrainer(vocab_size=18000, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Adding [BOS] and [EOS] at the start and end of the sentences respectively.
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    pair="[BOS] $A [EOS] $B:1 [EOS]:1",
    special_tokens=[
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ],
)
