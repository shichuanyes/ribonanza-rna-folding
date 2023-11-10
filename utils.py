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
