import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from model import Transformer
from dataset import En2FrDataset

def main():
    parser = argparse.ArgumentParser("Train a Transformer")
    parser.add_argument(
        "--exp_name",
        required=True,
        help="Experiment name"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        required=False,
        default=512,
        help="Dimension of the feature vector at each sequence step"
    )
    parser.add_argument(
        "--n_encoder_layers",
        type=int,
        required=False,
        default=6,
        help="Number of Transformer encoder layers"
    )
    parser.add_argument(
        "--n_decoder_layers",
        type=int,
        required=False,
        default=6,
        help="Number of Transformer decoder layers"
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        required=False,
        default=8,
        help="Number of attention heads per encoder/decoder layer"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        required=False,
        default=0.1
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=1
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate"
    )
    args = parser.parse_args()
    dataset = En2FrDataset()
    model = Transformer(
        exp_name=args.exp_name,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_model=args.d_model,
        num_heads=args.n_heads,
        dropout=args.dropout,
        n_en_words=dataset.input_vocab_size,
        n_fr_words=dataset.output_vocab_size
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataloader = dataset.get_dataloader()
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        for n_steps, (x, y) in tqdm(enumerate(dataloader)):
            x = x.to(device)
            y = y.to(device)
            y = torch.roll(y, shifts=-1, dims=1)
            y[:,-1] = model.fr_tokenizer.pad_token_id
            y_one_hot = F.one_hot(y, num_classes=dataset.output_vocab_size)
            y_one_hot = y_one_hot.permute(0, 2, 1)
            y_one_hot = y_one_hot.float()
            logits = model(x, y)
            loss = loss_fn(logits, y_one_hot)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if n_steps % 50 == 0:
                print(f"Step = {n_steps}. Loss = {loss.item()}")
            if n_steps % 10000 == 0:
                model.save()

if __name__ == "__main__":
    main()