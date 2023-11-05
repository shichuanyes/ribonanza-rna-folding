import math

import numpy as np
import pandas as pd
import torch
from torch import nn

max_seq_length = 206
nucleotides = 'ACGU'


def str_to_seq(s):
    mapping = {nucleotide: idx for idx, nucleotide in enumerate(nucleotides)}
    return [mapping[c] for c in s]


def mae(outputs, labels, seq_lengths):
    loss = 0.0
    se = torch.abs(outputs - labels)
    for i in range(outputs.size(0)):
        loss += torch.sum(se[i, :seq_lengths[i]])
    return loss / seq_lengths.sum()


def mse(outputs, labels, seq_lengths):
    loss = 0.0
    se = (outputs - labels) ** 2
    for i in range(outputs.size(0)):
        loss += torch.sum(se[i, :seq_lengths[i]])
    return loss / seq_lengths.sum()


def train_test_split(
        df: pd.DataFrame,
        test_size: float,
        random_state: int
):
    n = len(df)
    rng = np.random.default_rng(seed=random_state)
    idx = rng.permutation(n)
    split = math.floor(n * test_size)
    return df.iloc[idx[split:]], df.iloc[idx[:split]]
