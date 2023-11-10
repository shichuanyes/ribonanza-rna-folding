import argparse

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RNADataset
from model import RNAModel
from utils import nucleotides


def train_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: callable,

        sequences: torch.Tensor,
        reactivities: torch.Tensor,
        experiment_types: torch.Tensor,
):
    mask = sequences.sum(dim=-1) == 0

    optimizer.zero_grad()

    outputs = model(sequences, mask)
    outputs = outputs[torch.arange(outputs.size(0)), :, experiment_types]

    loss = criterion(outputs, reactivities)
    loss = torch.mean(loss[~ mask])
    loss.backward()
    optimizer.step()

    return loss.item()



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
        for sequences, reactivities, experiment_types in tqdm(dataloader, desc='Validation', leave=False):
            sequences, reactivities, experiment_types = sequences.to(device), reactivities.to(device), experiment_types.to(device)

            mask = sequences.sum(dim=-1) == 0

            count += torch.sum(~ mask)

            outputs = model(sequences, mask)
            outputs = outputs[torch.arange(outputs.size(0)), :, experiment_types]
            outputs = torch.clamp(outputs, min=0.0, max=1.0)

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
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)

    args = parser.parse_args()

    assert args.kernel_size % 2 == 1
    assert args.d_model % args.nhead == 0
    assert 0.0 <= args.dropout < 1.0

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
        kernel_size=args.kernel_size,
        dropout=args.dropout
    ).to(device)

    print(f"Started training on {device}. Current parameters:")
    print(args)

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dataloader = train_loader

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        for sequences, reactivities, experiment_types in tqdm(dataloader, desc='Train', leave=False):
            sequences, reactivities, experiment_types = sequences.to(device), reactivities.to(device), experiment_types.to(device)
            train_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                sequences=sequences,
                reactivities=reactivities,
                experiment_types=experiment_types
            )

        score = validate(
            model=model,
            criterion=nn.L1Loss(reduction='none'),
            dataloader=val_loader,
            device=device
        )
        print()
        print(f"Epoch: {epoch + 1} of {args.num_epochs}: Validation MAE={score}")

    print()
    print("Saving model...")
    torch.save(model, args.save_path)
    print(f"Model saved to {args.save_path}")
