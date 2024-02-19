import math

import numpy as np
import torch
import torch.nn as nn


def gen_mask(length):
    mask = np.triu(np.full([length, length], -np.inf), 1)
    mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0).unsqueeze(0)

    return mask


def gen_signal(length, channels):
    position = torch.arange(length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, channels, 2) * (-math.log(10000) / channels))

    signal = torch.zeros(length, channels)
    signal[:, 0::2] = torch.sin(position * div_term)
    signal[:, 1::2] = torch.cos(position * div_term)

    signal = torch.tensor(signal).unsqueeze(0)

    return signal


class MultiHeadAttn(nn.Module):
    def __init__(self, hidden_size, dim_k, dim_v, n_heads, mask, dropout):
        super(MultiHeadAttn, self).__init__()
        self.n_heads = n_heads
        self.mask = mask
        self.query_scale = (dim_k // n_heads) ** -0.5

        self.linear_q = nn.Linear(hidden_size, dim_k, bias=False)
        self.linear_k = nn.Linear(hidden_size, dim_k, bias=False)
        self.linear_v = nn.Linear(hidden_size, dim_v, bias=False)
        self.linear_out = nn.Linear(dim_v, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        return x.view(x.shape[0], x.shape[1], self.n_heads, x.shape[2] // self.n_heads).permute(0, 2, 1, 3)

    def merge_heads(self, x):
        return x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[2], x.shape[3] * self.n_heads)

    def forward(self, queries, keys, values):
        queries = self.linear_q(queries)
        keys = self.linear_k(keys)
        values = self.linear_v(values)

        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)

        queries *= self.query_scale

        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        logits += self.mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)

        weights = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(weights)

        contexts = torch.matmul(weights, values)
        contexts = self.merge_heads(contexts)

        outputs = self.linear_out(contexts)

        return outputs


class PositionwiseFF(nn.Module):
    def __init__(self, dim_in, dim_out, dropout):
        super(PositionwiseFF, self).__init__()
        self.pad = nn.ConstantPad1d((1, 1), 0)
        self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size=3, padding='valid')
        self.conv2 = nn.Conv1d(dim_in, dim_out, kernel_size=3, padding='valid')

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.pad(inputs.permute(0, 2, 1))
        x = self.conv1(x).permute(0, 2, 1)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.pad(x.permute(0, 2, 1))
        x = self.conv2(x).permute(0, 2, 1)
        x = self.relu(x)
        outputs = self.dropout(x)

        return outputs
