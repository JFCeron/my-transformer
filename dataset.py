from typing import Iterator

import torch
from transformers import BertTokenizer, CamembertTokenizer

TXT_FILE_PATH = "data/eng-fra.txt"
MAX_SEQ_LEN = 20
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
                en_tokens = dataset.en_tokenizer.encode(
                    en_text,
                    padding="max_length",
                    return_tensors="pt"
                )
                en_tokens = torch.squeeze(en_tokens)
                fr_tokens = dataset.fr_tokenizer.encode(
                    fr_text,
                    padding="max_length",
                    return_tensors="pt"
                )
                fr_tokens = torch.squeeze(fr_tokens)
                return en_tokens, fr_tokens
        return En2FrIterator()
    
    def get_dataloader(self, shuffle=True, batch_size=32):
        if shuffle:
            dataset = self.shuffle()
        else:
            dataset = self
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
