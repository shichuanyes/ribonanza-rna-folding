import argparse

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RNADataset
from model import RNAModel
from utils import nucleotides, mae


def train(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: callable,
        dataloader: DataLoader,
        num_epochs: int,
        device: torch.device
):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, labels, is_dmp in dataloader:
            inputs, labels, is_dmp = inputs.to(device), labels.to(device), is_dmp.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[torch.arange(outputs.size(0)), :, is_dmp], labels)
            loss.backward()
            optimizer.step()


def validate(
        model: nn.Module,
        criterion: callable,
        dataloader: DataLoader,
        device: torch.device
) -> float:
    model.eval()
    loss = 0.0
    for inputs, labels, is_dmp in tqdm(dataloader):
        inputs, labels, is_dmp = inputs.to(device), labels.to(device), is_dmp.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss += criterion(outputs[torch.arange(outputs.size(0)), :, is_dmp], labels).item()
    return loss / len(dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='RNATraining',
        description="Train a transformer model to predict RNA sequence reactivity"
    )

    parser.add_argument('train_path')
    parser.add_argument('save_path', nargs='?', default='model.pt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)

    args = parser.parse_args()

    print("Reading training set...")
    df = pd.read_csv(args.train_path)
    print()

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=283)

    train_loader = DataLoader(RNADataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RNADataset(val_df), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RNAModel(
        embed_dim=len(nucleotides),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    print(f"Started training on {device}. Current parameters:")
    print(args)
    train(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr),
        dataloader=train_loader,
        num_epochs=args.num_epochs,
        device=device
    )
    print()

    print("Running on validation set...")
    score = validate(
        model=model,
        criterion=mae,
        dataloader=train_loader,
        device=device
    )
    print(f"Validation score: {score}")

    print("Saving model...")
    torch.save(model, args.save_path)
    print(f"Model saved to {args.save_path}")
