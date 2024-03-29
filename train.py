import argparse

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RNADataset
from model import RNAModel
from utils import nucleotides


def train(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: callable,
        dataloader: DataLoader,
        device: torch.device,
        scaler
):
    model.train()
    for batch in tqdm(dataloader, desc='Train', leave=False):
        sequences = batch['seq'].to(device)
        reactivities = batch['react'].to(device)
        mask = batch['mask'].to(device)

        max_len = (~ mask).sum(-1).max()
        sequences = sequences[:, :max_len]
        reactivities = reactivities[:, :max_len]
        mask = mask[:, :max_len]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(sequences, mask)
            loss = criterion(outputs[~ mask], reactivities[~ mask].clip(0, 1))
            loss = torch.mean(loss[~ torch.isnan(loss)])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def validate(
        model: nn.Module,
        criterion: callable,
        dataloader: DataLoader,
        device: torch.device
) -> float:
    model.eval()
    with torch.inference_mode():
        total_loss = 0.0
        count = 0
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            sequences = batch['seq'].to(device)
            reactivities = batch['react'].to(device)
            mask = batch['mask'].to(device)

            max_len = (~ mask).sum(-1).max()
            sequences = sequences[:, :max_len]
            reactivities = reactivities[:, :max_len]
            mask = mask[:, :max_len]

            count += torch.sum(~ mask)

            with torch.cuda.amp.autocast():
                outputs = model(sequences, mask)
                outputs = torch.clamp(outputs, min=0.0, max=1.0)
                loss = criterion(outputs[~ mask], reactivities[~ mask])
                total_loss += torch.sum(loss[~ torch.isnan(loss)])

    return total_loss / (count * 2)


class WarmUpLR(StepLR):
    def __init__(self, optimizer: Optimizer, warm_up_epochs: int, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.warm_up_epochs = warm_up_epochs
        super(WarmUpLR, self).__init__(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_up_epochs:
            # Warm-up phase
            alpha = (self.last_epoch + 1) / self.warm_up_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Original scheduler phase
            return super(WarmUpLR, self).get_lr()


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
    parser.add_argument('--perturb', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--sn_threshold', type=float, default=0.5)
    parser.add_argument('--warm_up_epochs', type=int, default=3)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5)

    args = parser.parse_args()

    assert args.kernel_size % 2 == 1
    assert args.d_model % args.nhead == 0
    assert 0.0 <= args.dropout < 1.0
    assert 0.0 <= args.sn_threshold < 1.0

    print("Reading training set...")
    df = pd.read_csv(args.train_path)
    print()

    train_ds = RNADataset(df, sn_threshold=args.sn_threshold)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RNADataset(df, mode='test', sn_threshold=0.5), batch_size=args.batch_size, shuffle=False)

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

    criterion = nn.L1Loss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = WarmUpLR(optimizer, warm_up_epochs=args.warm_up_epochs, step_size=args.step_size, gamma=args.gamma)

    scaler = torch.cuda.amp.GradScaler()

    best = 1.0

    for epoch in tqdm(range(args.num_epochs)):
        train_ds.perturb(args.perturb)

        train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            scaler=scaler
        )

        score = validate(
            model=model,
            criterion=nn.L1Loss(reduction='none'),
            dataloader=val_loader,
            device=device
        )
        print()
        print(f"Epoch: {epoch + 1} of {args.num_epochs}: Validation MAE={score}")

        scheduler.step()

        if score < best:
            best = score
            torch.save(model, args.save_path)

        print(f"Best val score={best}")
