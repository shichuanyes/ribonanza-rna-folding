import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

nucleotides = 'ACGU'


def str_to_seq(s):
    mapping = {nucleotide: idx for idx, nucleotide in enumerate(nucleotides)}
    return [mapping[c] for c in s]


def str_to_tensor(s: str):
    sequence = str_to_seq(s)
    sequence = torch.LongTensor(sequence)
    sequence = F.one_hot(sequence, num_classes=len(nucleotides))

    return sequence.float()


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
