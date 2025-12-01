from transformers import RobertaTokenizerFast
from src.config import config


class Tokenizer:
    def __init__(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained(config.base_model)

    def tokenize(self, batch):
        inputs = self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return inputs


tokenizer = Tokenizer()
