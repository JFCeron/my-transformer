import os
from datetime import datetime
from typing import OrderedDict

from torch import nn

class Transformer(nn.Module):
    def __init__(
        self, experiment, n_encoder_layers, n_decoder_layers, d_model,
        num_heads, dropout, n_en_words, n_fr_words):
        super().__init__()
        self.experiment = experiment
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        # Source and target language embeddings
        self.en_embedding = nn.Embedding(n_en_words, self.d_model)
        self.fr_embedding = nn.Embedding(n_fr_words, self.d_model)
        # Encoder
        encoder_layers = OrderedDict()
        for i in range(n_encoder_layers):
            encoder_layers.update({
                f"enc{i}": nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dropout=dropout
                )
            })
        self.encoder = nn.Sequential(encoder_layers)
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decoder_layers.append(
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dropout=dropout
                )
            )
        self.linear = nn.Linear(d_model, n_fr_words)

    def forward(self, x, y):
        x = self.en_embedding(x)
        encoded_x = self.encoder(x)
        encoded_y = self.fr_embedding(y)
        for decoder_layer in self.decoder_layers:
            encoded_y = decoder_layer(encoded_y, encoded_x)
        pred = self.linear(encoded_y)
        pred = pred.permute(0, 2, 1)
        return pred

    def save(self):
        trained_models_dir = self.experiment.trained_models_dir
        os.makedirs(trained_models_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y-%m-%d_%M-%S.pt")
        filename = os.path.join(trained_models_dir, filename)
        torch.save(self, filename)

    @classmethod
    def load(pt_file):
        return torch.load(pt_file)