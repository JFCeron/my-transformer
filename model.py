import os
import json
from datetime import datetime
from typing import OrderedDict

import torch

from dataset import get_en_tokenizer, get_fr_tokenizer, tokenize

class Transformer(torch.nn.Module):
    MODEL_DIR = "trained-models/{exp_name}/{model_id}"
    MODEL_ARGS_PATH = os.path.join(MODEL_DIR, "model_args.json")
    WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.pt")

    def __init__(
        self, exp_name, n_encoder_layers, n_decoder_layers, d_model, num_heads,
        dropout, n_en_words, n_fr_words):
        super().__init__()
        self.exp_name = exp_name
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
        x = tokenize(self.en_tokenizer, str_x)
        import pdb; pdb.set_trace()
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

    def save(self):
        new_model_id = datetime.now().strftime("%Y-%m-%d_%M-%S.pt")
        model_dir_args = {"exp_name": self.exp_name, "model_id": new_model_id}
        model_args_path = self.MODEL_ARGS_PATH.format(**model_dir_args)
        weights_path = self.WEIGHTS_PATH.format(**model_dir_args)
        parent_dir = os.path.dirname(model_args_path)
        os.makedirs(parent_dir)
        torch.save(self, weights_path)
        with open(model_args_path, "w") as model_args_file:
            model_args = {
                "exp_name": self.exp_name,
                "n_encoder_layers": self.n_encoder_layers,
                "n_decoder_layers": self.n_decoder_layers,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "n_en_words": self.n_en_words,
                "n_fr_words": self.n_fr_words   
            }
            model_args_str = json.dumps(model_args, indent=2)
            model_args_file.write(model_args_str)

    @classmethod
    def load(cls, exp_name, model_id):
        # Initialize model
        model_dir_args = {"exp_name": exp_name, "model_id": model_id}
        model_args_path = cls.MODEL_ARGS_PATH.format(**model_dir_args)
        with open(model_args_path, "r") as model_args_file:
            model_args_str = model_args_file.read()
            model_args = json.loads(model_args_str)
            transformer = Transformer(**model_args)
        # Get weights from .pt file
        weights_path = cls.WEIGHTS_PATH.format(**model_dir_args)
        loaded = torch.load(weights_path)
        transformer.load_state_dict(loaded.state_dict())
        return transformer