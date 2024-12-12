import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from models.Transformer import TransformerEncoderLayer  # 假设之前定义的 Transformer 层
from torch_geometric.data import Data

class GCNTransformerGAT(nn.Module):
    def __init__(self, feature, out_channel, gcn_hidden_dim=1024, transformer_dim=512, num_heads=4, num_transformer_layers=2, dropout=0.1):
        super(GCNTransformerGAT, self).__init__()
        # GAT 层
        self.gat = GATConv(gcn_hidden_dim, gcn_hidden_dim, heads=num_heads, concat=True)  # GAT 层
        self.gat_bn = BatchNorm(gcn_hidden_dim * num_heads)  # GAT 的 BatchNorm

        # GCN 层
        self.GConv1 = GCNConv(feature, gcn_hidden_dim)
        self.bn1 = BatchNorm(gcn_hidden_dim)
        self.GConv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.bn2 = BatchNorm(gcn_hidden_dim)

        # GAT 层
        self.gat = GATConv(gcn_hidden_dim, gcn_hidden_dim, heads=num_heads, concat=True)  # GAT 层
        self.gat_bn = BatchNorm(gcn_hidden_dim * num_heads)  # GAT 的 BatchNorm

        # 投影层：将 GCN 和 GAT 输出的维度调整到 Transformer 的输入维度
        self.projection = nn.Linear(gcn_hidden_dim * num_heads, transformer_dim)

        # Transformer 层
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

        # 2. 使用 GAT 层增强节点特征
        x = F.relu(self.gat(x, edge_index))
        x = self.gat_bn(x)

        # 3. 投影到 Transformer 的输入维度
        x = self.projection(x)

        # 4. 传入 Transformer 层进行全局建模
        x = x.unsqueeze(1).permute(1, 0, 2)  # 调整为 Transformer 的输入格式
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.squeeze(0)  # 恢复维度为 (num_nodes, feature_dim)

        # 5. 分类
        x = self.fc(x)  # [num_nodes, 512]
        x = self.fc1(x)  # [num_nodes, num_classes]

        return x  # 返回 [num_nodes, num_classes]
