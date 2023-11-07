import argparse

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RNADataset
from model import RNAModel
from utils import nucleotides, train_test_split


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
        for sequences, reactivities, experiment_types in dataloader:
            sequences, reactivities, experiment_types = sequences.to(device), reactivities.to(device), experiment_types.to(device)

            mask = sequences.sum(dim=-1) == 0

            optimizer.zero_grad()

            outputs = model(sequences, mask)
            outputs = outputs[torch.arange(outputs.size(0)), :, experiment_types]

            loss = criterion(outputs, reactivities)
            loss = torch.mean(loss[~ mask])
            # loss = torch.sum(loss * (~ mask)) / torch.sum(~ mask)
            loss.backward()
            optimizer.step()


def validate(
        model: nn.Module,
        criterion: callable,
        dataloader: DataLoader,
        device: torch.device
) -> float:
    model.eval()
    with torch.inference_mode():
        loss = 0.0
        count = 0
        for sequences, reactivities, experiment_types in tqdm(dataloader):
            sequences, reactivities, experiment_types = sequences.to(device), reactivities.to(device), experiment_types.to(device)

            mask = sequences.sum(dim=-1) == 0

            count += torch.sum(~ mask)

            outputs = model(sequences, mask)
            outputs = outputs[torch.arange(outputs.size(0)), :, experiment_types]
            outputs = torch.clamp(outputs, min=0.0, max=1.0)

            # loss += torch.sum(criterion(outputs, reactivities) * (~ mask))
            loss += torch.sum(criterion(outputs, reactivities)[~ mask])

    return loss / count


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
        criterion=nn.MSELoss(reduction='none'),
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr),
        dataloader=train_loader,
        num_epochs=args.num_epochs,
        device=device
    )

    print("Saving model...")
    torch.save(model, args.save_path)
    print(f"Model saved to {args.save_path}")

    print("Running on validation set...")
    score = validate(
        model=model,
        criterion=nn.L1Loss(reduction='none'),
        dataloader=train_loader,
        device=device
    )
    print(f"Validation score: {score}")
