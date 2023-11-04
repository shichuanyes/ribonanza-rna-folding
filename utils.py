import torch

max_seq_length = 206
nucleotides = 'ACGU'


def str_to_seq(s):
    mapping = {nucleotide: idx for idx, nucleotide in enumerate(nucleotides)}
    return [mapping[c] for c in s]


def mae(outputs, labels):
    return torch.mean(torch.abs(outputs - labels))
