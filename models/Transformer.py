import torch
from torch import nn
from torch.nn import functional as F
from models.ContraNorm import ContraNorm  # 假设 ContraNorm 已正确实现并导入
# import numpy as np
# import os
# import torch_geometric.nn as pygnn
# import math


def padding_mask(seq):
    length = seq.shape[1]
    mask = seq.eq(0)
    mask = mask.unsqueeze(1).expand(-1, length, -1)
    return mask

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=None, dropout=0.1,
                 activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = ContraNorm(d_model)  # 使用 ContraNorm
        self.norm2 = ContraNorm(d_model)  # 使用 ContraNorm
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, attn_mask=None, src_mask=None, length_mask=None):
        N, L, F = x.shape  # N: batch_size, L: seq_len, F: feature_dim

        # 注意力机制
        if src_mask is not None:
            attn_mask = src_mask
        attn_output, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = x + self.dropout(attn_output)

        # 保证 x 是 3D 张量
        x = self.norm1(x)  # 直接通过 ContraNorm 归一化

        # 前馈网络
        y = self.dropout(self.activation(self.linear1(x)))
        y = self.dropout(self.linear2(y))

        # 使用 ContraNorm 进行第二次归一化
        x = (x + y)
        x = self.norm2(x)  # 保证是 3D 张量

        return x


class Encoder_Layer(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=64, num_heads=8, dropout=0):
        super(Encoder_Layer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.calculate_q = nn.Linear(self.embedding_dim, self.embedding_dim)
        torch.nn.init.xavier_normal(self.calculate_q.weight, gain=1)
        self.calculate_k = nn.Linear(self.embedding_dim, self.embedding_dim)
        torch.nn.init.xavier_normal(self.calculate_k.weight, gain=1)
        self.calculate_v = nn.Linear(self.embedding_dim, self.embedding_dim)
        torch.nn.init.xavier_normal(self.calculate_v.weight, gain=1)
        self.MultiheadAttention = nn.MultiheadAttention(self.embedding_dim, num_heads, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = ContraNorm(embedding_dim)  # 替换 LayerNorm 为 ContraNorm

        self.fc = nn.Linear(embedding_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_normal(self.fc1.weight, gain=1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_normal(self.fc2.weight, gain=1)

        self.dropout2 = nn.Dropout(p=dropout)

        self.layer_norm2 = ContraNorm(hidden_dim)  # 替换 LayerNorm 为 ContraNorm
        self.relu2 = nn.LeakyReLU()

    def forward(self, querys, keys, values, mask=None):
        query = self.calculate_q(querys).transpose(0, 1).contiguous()
        key = self.calculate_k(keys).transpose(0, 1).contiguous()
        value = self.calculate_v(values).transpose(0, 1).contiguous()

        output, _ = self.MultiheadAttention(query, key, value, key_padding_mask=mask)
        output = output.transpose(0, 1).contiguous()

        output = querys + self.dropout(output)
        output = self.layer_norm(output.unsqueeze(0)).squeeze(0)  # ContraNorm 处理

        output = self.fc(output)
        tmp = output

        output = self.fc2(self.dropout2(self.relu2(self.fc1(output))))
        output = tmp + self.dropout2(output)
        output = self.layer_norm2(output.unsqueeze(0)).squeeze(0)  # ContraNorm 处理

        return output

