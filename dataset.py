from typing import Iterator

import torch
import transformers

TXT_FILE_PATH = "data/eng-fra.txt"

class En2FrDataset(torch.utils.data.IterableDataset):
    def __init__(self, max_seq_len):
        self.en_tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=max_seq_len
        )
        self.fr_tokenizer = transformers.CamembertTokenizer.from_pretrained(
            "camembert-base",
            model_max_length=max_seq_len
        )

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
