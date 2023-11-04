# https://nlp.seas.harvard.edu/2018/04/03/attention.html
import math

import torch
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class RNAModel(nn.Module):
    def __init__(self, embed_dim, d_model, nhead, num_layers, dropout):
        super().__init__()

        self.conv = nn.Conv1d(embed_dim, d_model, kernel_size=3, padding=1)
        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.linear = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.pe(x)
        x = self.te(x)
        x = self.linear(x)

        return x.squeeze()
