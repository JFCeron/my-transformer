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
        self.n_en_words = n_en_words
        self.n_fr_words = n_fr_words
        # Source and target language embeddings
        self.en_embedding = torch.nn.Embedding(n_en_words, self.d_model)
        self.fr_embedding = torch.nn.Embedding(n_fr_words, self.d_model)
        # Encoder
        self.encoder_layers = torch.nn.ModuleList()
        for _ in range(n_encoder_layers):
            self.encoder_layers.append(
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
        # Decoder layers
        self.decoder_layers = torch.nn.ModuleList()
        for _ in range(n_decoder_layers):
            self.decoder_layers.append(
                torch.nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
        self.linear = torch.nn.Linear(d_model, n_fr_words)
        # Tokenizers for decoding
        self._en_tokenizer = None
        self._fr_tokenizer = None

    def forward(self, src, tgt):
        # Define masks
        src_padding_mask, tgt_padding_mask, tgt_mask = self.get_masks(
            src, tgt, src.device)
        # Encode source sentence
        encoded_src = self.en_embedding(src)
        for encoder_layer in self.encoder_layers:
            encoded_src = encoder_layer(
                src=encoded_src,
                src_key_padding_mask=src_padding_mask
            )
        # Encode target sentence
        encoded_tgt = self.fr_embedding(tgt)
        for decoder_layer in self.decoder_layers:
            encoded_tgt = decoder_layer(
                tgt=encoded_tgt,
                memory=encoded_src,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
        # Logit output
        pred = self.linear(encoded_tgt)
        pred = pred.permute(0, 2, 1)
        return pred

    def get_masks(self, src, tgt, device):
        src_padding_mask = (src == self.en_tokenizer.pad_token_id)
        src_padding_mask = src_padding_mask.to(device)
        tgt_padding_mask = (tgt == self.fr_tokenizer.pad_token_id)
        tgt_padding_mask = tgt_padding_mask.to(device)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        tgt_mask = tgt_mask.to(device)
        return src_padding_mask, tgt_padding_mask, tgt_mask

    def greedy_decode(self, str_src, device):
        src = tokenize(str_src, self.en_tokenizer)
        src = src.view(1, -1)
        src = src.to(device)
        tgt = tokenize("", self.fr_tokenizer)
        tgt = tgt.view(1, -1)
        tgt = tgt.to(device)
        n_decoded_tokens = 0
        while True:
            logits = self.forward(src, tgt)
            next_step_logits = logits[0,:,n_decoded_tokens]
            next_token = torch.argmax(next_step_logits)
            n_decoded_tokens += 1
            tgt[0][n_decoded_tokens] = next_token
            if next_token == self.fr_tokenizer.eos_token_id \
               or n_decoded_tokens + 1 >= self.fr_tokenizer.model_max_length:
               break
        return self.fr_tokenizer.decode(tgt)

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
        new_model_id = datetime.now().strftime("%Y-%m-%d_%H-%M")
        model_dir_args = {"exp_name": self.exp_name, "model_id": new_model_id}
        model_args_path = self.MODEL_ARGS_PATH.format(**model_dir_args)
        weights_path = self.WEIGHTS_PATH.format(**model_dir_args)
        parent_dir = os.path.dirname(model_args_path)
        os.makedirs(parent_dir)
        torch.save(self.state_dict(), weights_path)
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
        # Load weights
        weights_path = cls.WEIGHTS_PATH.format(**model_dir_args)
        transformer.load_state_dict(torch.load(weights_path))
        return transformer