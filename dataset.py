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
            n_splits: int = 4,
            seq_only: bool = False
    ):
        self.seq_only = seq_only

        df_DMS = df.loc[df['experiment_type'] == 'DMS_MaP']
        df_2A3 = df.loc[df['experiment_type'] == '2A3_MaP']

        split = list(
            KFold(n_splits=n_splits, random_state=seed, shuffle=True).split(df_DMS)
        )[fold][0 if mode == 'train' else 1]
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)

        if 'SN_filter' in df.columns:
            mask = (df_DMS['SN_filter'].values > 0) & (df_2A3['SN_filter'].values > 0)
            df_DMS = df_DMS.loc[mask].reset_index(drop=True)
            df_2A3 = df_2A3.loc[mask].reset_index(drop=True)

        self.seq = df_DMS['sequence'].values
        self.len = df_DMS['sequence'].str.len().values

        self.max_len = max(self.len)

        if not self.seq_only:
            react_cols = [
                col for col in df.columns if not col.startswith('reactivity_error') and col.startswith('reactivity')
            ]

            self.react_DMS = df_DMS[react_cols].values
            self.react_2A3 = df_2A3[react_cols].values

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx: int):
        seq = self.seq[idx]
        seq = str_to_seq(seq)
        seq = np.pad(seq, (0, self.max_len - len(seq)))
        seq = torch.from_numpy(seq)

        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[self.len[idx]:] = True

        if self.seq_only:
            return {
                'seq': seq,
                'mask': mask
            }

        react = torch.from_numpy(np.stack(
            [self.react_DMS[idx], self.react_2A3[idx]],
            axis=-1
        ))

        return {
            'seq': seq,
            'react': react,
            'mask': mask
        }
