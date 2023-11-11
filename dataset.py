from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import str_to_tensor


def flip(
        df: pd.DataFrame,
        columns: List[List],
        flip_ratio: float,
        seed: Optional[int]
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    mask = rng.random(size=len(df)) > flip_ratio
    idx = df.index[mask]

    df.loc[idx, 'sequence'] = df.loc[idx, 'sequence'].str[::-1]

    for cols in columns:
        lengths = df.loc[idx, 'sequence'].str.len().to_numpy()
        reactivities = df.loc[idx, cols].to_numpy()
        for i in range(reactivities.shape[0]):
            reactivities[i, :lengths[i]] = reactivities[i, :lengths[i]][::-1]
        df.loc[idx, cols] = reactivities

    return mask


class RNADataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            flip_ratio: float = 0.5,
            fill_na: bool = False,
            seed: int = 283
    ):
        self.df = df
        self.reactivity_columns = [
            column for column in self.df.columns
            if not column.startswith('reactivity_error') and column.startswith('reactivity')
        ]
        self.experiment_mapping = {
            'DMS_MaP': 0,
            '2A3_MaP': 1
        }

        if fill_na:
            for reactivity_column in self.reactivity_columns:
                error_column = reactivity_column.replace('reactivity', 'reactivity_error')
                self.df.loc[self.df[reactivity_column] < self.df[error_column] * 1.5, reactivity_column] = np.nan
        flip(df, [self.reactivity_columns], flip_ratio, seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sequence = self.df['sequence'].iloc[idx]
        sequence = str_to_tensor(sequence)
        sequence = F.pad(sequence, (0, 0, 0, len(self.reactivity_columns) - sequence.size(0)))

        reactivity = self.df[self.reactivity_columns].iloc[idx].to_numpy()
        reactivity = torch.FloatTensor(reactivity)
        # reactivity = torch.nan_to_num(reactivity)

        experiment_type = self.df['experiment_type'].iloc[idx]
        experiment_type = self.experiment_mapping[experiment_type]

        return sequence, reactivity, experiment_type

    def __getitems__(self, indices: List[int]):
        sequences = self.df['sequence'].iloc[indices]
        sequences = [str_to_tensor(sequence) for sequence in sequences]
        sequences = pad_sequence(sequences, batch_first=True)
        sequences = F.pad(sequences, (0, 0, 0, len(self.reactivity_columns) - sequences.size(1)))

        reactivities = self.df[self.reactivity_columns].iloc[indices].to_numpy()
        reactivities = torch.FloatTensor(reactivities)
        # reactivities = torch.nan_to_num(reactivities)

        experiment_types = self.df['experiment_type'].iloc[indices]
        experiment_types = experiment_types.map(self.experiment_mapping).to_numpy()

        return list(zip(sequences, reactivities, experiment_types))


class RNAPredictDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            flip_ratio: float = 0.5,
            seed: int = 283
    ):
        self.df = df
        self.max_seq_length = df['sequence'].str.len().max()
        self.flip = flip(df, [], flip_ratio, seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sequence = self.df['sequence'].iloc[idx]
        sequence = str_to_tensor(sequence)
        sequence = F.pad(sequence, (0, 0, 0, self.max_seq_length - sequence.size(0)))

        return sequence, self.flip[idx]

    def __getitems__(self, indices: List[int]):
        sequences = self.df['sequence'].iloc[indices]
        sequences = [str_to_tensor(sequence) for sequence in sequences]
        sequences = pad_sequence(sequences, batch_first=True)
        sequences = F.pad(sequences, (0, 0, 0, self.max_seq_length - sequences.size(1)))

        return list(zip(sequences, self.flip[indices]))
