# import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class TransformerEncoderLayer(nn.Module):
    """
    自定义 Transformer 编码器层，支持 attn_mask 和 src_key_padding_mask。
    """
    def __init__(self, d_model, num_heads=8, d_ff=None, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, attn_mask=None, src_key_padding_mask=None):
        # 自注意力机制
        attn_output, _ = self.attention(
            x, x, x, key_padding_mask=src_key_padding_mask, attn_mask=attn_mask
        )
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.norm1(x)

        # 前馈网络
        y = self.dropout(self.activation(self.linear1(x)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y)  # 残差连接


class GCNWithTransformer(nn.Module):
    """
    将 GCN 和 Transformer 层结合的模型
    """
    def __init__(self, feature, out_channel, gcn_hidden_dim=1024, transformer_dim=512, num_heads=4, num_transformer_layers=2, dropout=0.1):
        super(GCNWithTransformer, self).__init__()

        # GCN 层
        self.GConv1 = GCNConv(feature, gcn_hidden_dim)
        self.bn1 = BatchNorm(gcn_hidden_dim)
        self.GConv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.bn2 = BatchNorm(gcn_hidden_dim)
        # 投影层：将 GCN 的输出维度调整到 Transformer 的输入维度
        self.projection = nn.Linear(gcn_hidden_dim, transformer_dim)
        # Transformer 编码器
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=transformer_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_transformer_layers)
        ])

        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(transformer_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.fc1 = nn.Linear(512, out_channel)

    def forward(self, data):
        # 获取图数据中的节点特征和边信息
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # 1. 使用 GCN 提取节点特征
        x = F.relu(self.GConv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.relu(self.GConv2(x, edge_index, edge_weight))
        x = self.bn2(x)
        # 2. 投影到 Transformer 的输入维度
        x = self.projection(x)
        # 2. 传入 Transformer 层建模全局信息
        # Transformer 需要 (seq_len, batch_size, feature_dim)
        x = x.unsqueeze(1).permute(1, 0, 2)  # 调整为 Transformer 的输入格式

        for layer in self.transformer_layers:
            x = layer(x)

        x = x.squeeze(0)  # 恢复维度为 (batch_size, feature_dim)

        # 3. 分类
        x = self.fc(x)
        x = self.fc1(x)

        return x
