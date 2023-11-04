import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import str_to_seq, nucleotides, max_seq_length


def extract_input_seq(df: pd.DataFrame, idx):
    input_seq = df['sequence'].iloc[idx]
    input_seq = str_to_seq(input_seq)
    input_seq = torch.LongTensor(input_seq)
    input_seq = F.one_hot(input_seq, num_classes=len(nucleotides))
    input_seq = input_seq.float()
    input_seq = F.pad(input_seq, pad=(0, 0, 0, max_seq_length - input_seq.size(0)))

    return input_seq


def extract_label_seq(df: pd.DataFrame, label_idx, idx):
    label_seq = df.iloc[idx, label_idx]
    label_seq = torch.FloatTensor(label_seq.to_numpy(dtype=float))
    label_seq = torch.nan_to_num(label_seq)
    label_seq = F.pad(label_seq, pad=(0, max_seq_length - label_seq.size(0)))

    return label_seq


class RNADataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.label_idx = [idx for idx, column in enumerate(self.df.columns) if
                          not column.startswith('reactivity_error') and column.startswith('reactivity')]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_seq = extract_input_seq(self.df, idx)
        label_seq = extract_label_seq(self.df, self.label_idx, idx)
        return input_seq, label_seq, int(self.df['experiment_type'].iloc[idx] == 'DMS_MaP')


class RNAPredictDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_seq = extract_input_seq(self.df, idx)

        return input_seq
