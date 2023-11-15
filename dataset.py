import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from utils import str_to_seq


class RNADataset(Dataset):

    def __init__(
            self,
            df: pd.DataFrame,
            mode: str = 'train',
            seed: int = 283,
            fold: int = 0,
            n_splits: int = 4
    ):
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        if mode == 'predict':
            self.seq = df['sequence'].values
            self.len = df['sequence'].str.len().values
            self.max_len = max(self.len)
            return

        df_DMS = df.loc[df['experiment_type'] == 'DMS_MaP']
        df_2A3 = df.loc[df['experiment_type'] == '2A3_MaP']

        split = list(
            KFold(n_splits=n_splits, random_state=seed, shuffle=True).split(df_DMS)
        )[fold][0 if mode == 'train' else 1]
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)

        if 'signal_to_noise' in df.columns:
            mask = (df_DMS['signal_to_noise'].values > 0.5) & (df_2A3['signal_to_noise'].values > 0.5)
            df_DMS = df_DMS.loc[mask].reset_index(drop=True)
            df_2A3 = df_2A3.loc[mask].reset_index(drop=True)

        self.seq = df_DMS['sequence'].values
        self.len = df_DMS['sequence'].str.len().values
        self.max_len = max(self.len)

        react_cols = [
            col for col in df.columns if not col.startswith('reactivity_error') and col.startswith('reactivity')
        ]
        error_cols = [
            col for col in df.columns if col.startswith('reactivity_error')
        ]

        self.react_DMS = df_DMS[react_cols].values
        self.react_2A3 = df_2A3[react_cols].values

        self.error_DMS = df_DMS[error_cols].values
        self.error_2A3 = df_2A3[error_cols].values

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx: int):
        seq = self.seq[idx]
        seq = str_to_seq(seq)
        seq = np.pad(seq, (0, self.max_len - len(seq)))
        seq = torch.from_numpy(seq)

        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[self.len[idx]:] = True

        if self.mode == 'predict':
            return {
                'seq': seq,
                'mask': mask
            }

        react = torch.from_numpy(np.stack(
            [
                self.react_DMS[idx],
                self.react_2A3[idx]
            ],
            axis=-1
        ))

        return {
            'seq': seq,
            'react': react,
            'mask': mask
        }

    def perturb(self, perturb: float):
        self.react_DMS = self.rng.normal(loc=self.react_DMS, scale=perturb * self.error_DMS)
        self.react_2A3 = self.rng.normal(loc=self.react_2A3, scale=perturb * self.error_2A3)
