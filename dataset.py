from typing import Iterator

import torch
from transformers import BertTokenizer, CamembertTokenizer

TXT_FILE_PATH = "data/eng-fra.txt"
MAX_SEQ_LEN = 30
EN_TOKENIZER_ID = "bert-base-uncased"
FR_TOKENIZER_ID = "camembert-base"

def get_en_tokenizer():
    return BertTokenizer.from_pretrained(
        EN_TOKENIZER_ID,
        model_max_length=MAX_SEQ_LEN
    )

def get_fr_tokenizer():
    return CamembertTokenizer.from_pretrained(
        FR_TOKENIZER_ID,
        model_max_length=MAX_SEQ_LEN
    )

def tokenize(text, tokenizer):
    tokens = tokenizer.encode(
        text,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return torch.squeeze(tokens)

class En2FrDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.en_tokenizer = get_en_tokenizer()
        self.fr_tokenizer = get_fr_tokenizer()

    @property
    def input_vocab_size(self):
        return self.en_tokenizer.vocab_size
    
    @property
    def output_vocab_size(self):
        return self.fr_tokenizer.vocab_size

    def __iter__(dataset):
        class En2FrIterator(Iterator):
            def __init__(self):
                self.txt_file = open(TXT_FILE_PATH, "r")
            def __next__(self):
                next_line = next(self.txt_file)
                next_line = next_line.strip()
                en_text, fr_text = next_line.split("\t")
                en_tokens = tokenize(en_text, dataset.en_tokenizer)
                fr_tokens = tokenize(fr_text, dataset.fr_tokenizer)
                return en_tokens, fr_tokens
        return En2FrIterator()
    
    def get_dataloader(self, shuffle=True, batch_size=32):
        if shuffle:
            dataset = self.shuffle()
        else:
            dataset = self
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
