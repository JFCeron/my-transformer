import os
from datetime import datetime
from typing import OrderedDict

import torch

from dataset import get_en_tokenizer, get_fr_tokenizer

class Transformer(torch.nn.Module):
    def __init__(
        self, n_encoder_layers, n_decoder_layers, d_model,
        num_heads, dropout, n_en_words, n_fr_words):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        # Source and target language embeddings
        self.en_embedding = torch.nn.Embedding(n_en_words, self.d_model)
        self.fr_embedding = torch.nn.Embedding(n_fr_words, self.d_model)
        # Encoder
        encoder_layers = OrderedDict()
        for i in range(n_encoder_layers):
            encoder_layers.update({
                f"enc{i}": torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dropout=dropout
                )
            })
        self.encoder = torch.nn.Sequential(encoder_layers)
        # Decoder layers
        self.decoder_layers = torch.nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decoder_layers.append(
                torch.nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dropout=dropout
                )
            )
        self.linear = torch.nn.Linear(d_model, n_fr_words)
        # Tokenizers for decoding
        self._en_tokenizer = None
        self._fr_tokenizer = None

    def forward(self, x, y):
        x = self.en_embedding(x)
        encoded_x = self.encoder(x)
        encoded_y = self.fr_embedding(y)
        for decoder_layer in self.decoder_layers:
            encoded_y = decoder_layer(encoded_y, encoded_x)
        pred = self.linear(encoded_y)
        pred = pred.permute(0, 2, 1)
        return pred

    def greedy_decode(self, str_x, device):
        import pdb; pdb.set_trace()
        x = self.en_tokenizer.encode(str_x)
        return 0

    @property
    def en_tokenizer(self):
        if self._en_tokenizer is None:
            self._en_tokenizer = get_en_tokenizer()
        return self._en_tokenizer

    @property
    def fr_tokenizer(self):
        if self._fr_tokenizer is None:
            self._fr_tokenizer = get_fr_tokenizer()
        return self._fr_tokenizer

    def save(self, path):
        parent_dir = os.path.dirname(path)
        os.makedirs(parent_dir, exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, pt_file):
        transformer = torch.load(pt_file)
        return transformer