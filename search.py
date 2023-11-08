import argparse

import optuna
import pandas as pd
import torch
from optuna.trial import TrialState
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RNADataset
from model import RNAModel
from utils import nucleotides, train_test_split


def train_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: callable,

        sequences: torch.Tensor,
        reactivities: torch.Tensor,
        experiment_types: torch.Tensor
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


def define_model(trial) -> nn.Module:
    d_model = 2 ** trial.suggest_int('log_d_model', 7, 9)
    nhead = 2 ** trial.suggest_int('log_nhead', 2, 4)
    num_layers = trial.suggest_int('num_layers', 4, 12)

    return RNAModel(
        embed_dim=len(nucleotides),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=0.1
    )


def objective(
        trial,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
):
    model = define_model(trial).to(device)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    score = 0.0

    for epoch in tqdm(range(args.num_epochs), desc='Epochs', position=0):
        # Training starts here
        model.train()
        for sequences, reactivities, experiment_types in tqdm(train_loader, desc='Train batches', position=1, leave=False):
            sequences, reactivities, experiment_types = sequences.to(device), reactivities.to(device), experiment_types.to(device)

            train_step(model, optimizer, nn.MSELoss(reduction='none'), sequences, reactivities, experiment_types)

        # Validation starts here
        model.eval()
        with torch.inference_mode():
            loss = 0.0
            count = 0
            for sequences, reactivities, experiment_types in tqdm(val_loader, desc='Val batches', position=1, leave=False):
                sequences, reactivities, experiment_types = sequences.to(device), reactivities.to(device), experiment_types.to(device)

                mask = sequences.sum(dim=-1) == 0

                count += torch.sum(~ mask)

                outputs = model(sequences, mask)
                # outputs = F.pad(outputs, (0, 0, 0, mask.size(1) - outputs.size(1)))  # Because PyTorch 1.13 is stupid
                outputs = outputs[torch.arange(outputs.size(0)), :, experiment_types]
                outputs = torch.clamp(outputs, min=0.0, max=1.0)

                loss += torch.sum(nn.L1Loss(reduction='none')(outputs, reactivities)[~ mask])

        score = loss / count

        trial.report(score, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='RNASearching',
        description="Train a transformer model to predict RNA sequence reactivity"
    )

    parser.add_argument('train_path')
    parser.add_argument('save_path', nargs='?', default='model.pt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--n_trials', type=int, default=10)

    args = parser.parse_args()

    print("Reading training set...")
    df = pd.read_csv(args.train_path)
    print()

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=283)

    train_loader = DataLoader(RNADataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RNADataset(val_df), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, device, train_loader, val_loader), n_trials=args.n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
