import math

import numpy as np
import pandas as pd
import torch

max_seq_length = 206
nucleotides = 'ACGU'


def str_to_seq(s):
    mapping = {nucleotide: idx for idx, nucleotide in enumerate(nucleotides)}
    return [mapping[c] for c in s]


def mae(outputs, labels):
    return torch.mean(torch.abs(outputs - labels))


def train_test_split(
        df: pd.DataFrame,
        test_size: float,
        random_state: int
):
    n = len(df)
    rng = np.random.default_rng(seed=random_state)
    idx = rng.permutation(n)
    split = math.floor(n * test_size)
    return df.iloc[idx[:split]], df.iloc[idx[split:]]
