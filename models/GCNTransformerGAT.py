import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from models.Transformer import TransformerEncoderLayer


class GCNTransformerGAT(nn.Module):
    def __init__(self, feature, out_channel, gcn_hidden_dim=1024, transformer_dim=512, num_heads=4,
                 num_gcn_layers=4, num_transformer_layers=2, dropout=0.1):
        super(GCNTransformerGAT, self).__init__()

        # GAT 层 1
        self.gat1 = GATConv(feature, gcn_hidden_dim, heads=num_heads, concat=True)
        self.gat_bn1 = BatchNorm(gcn_hidden_dim * num_heads)

        # 动态添加多层 GCN
        self.gcn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            in_dim = gcn_hidden_dim * num_heads if i == 0 else gcn_hidden_dim  # 第一层的输入是 GAT 的输出
            self.gcn_layers.append(GCNConv(in_dim, gcn_hidden_dim))
            self.bn_layers.append(BatchNorm(gcn_hidden_dim))

        # GAT 层 2
        self.gat2 = GATConv(gcn_hidden_dim, gcn_hidden_dim, heads=num_heads, concat=True)
        self.gat_bn2 = BatchNorm(gcn_hidden_dim * num_heads)

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

    def forward(self, data, return_features=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # 1. GAT 层 1 提取节点特征
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat_bn1(x)

        # 2. 多层 GCN 提取节点特征
        for gcn, bn in zip(self.gcn_layers, self.bn_layers):
            x = F.relu(gcn(x, edge_index, edge_weight))
            x = bn(x)

        # 3. GAT 层 2 增强节点特征
        x = F.relu(self.gat2(x, edge_index))
        x = self.gat_bn2(x)

        # 4. 投影到 Transformer 的输入维度
        x = self.projection(x)

        # 5. 传入 Transformer 层进行全局建模
        x = x.unsqueeze(1).permute(1, 0, 2)  # 调整为 Transformer 的输入格式
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.squeeze(0)  # 恢复维度为 (num_nodes, feature_dim)

        if return_features:
            return x  # 返回特征

        # 6. 分类
        x = self.fc(x)  # [num_nodes, 512]
        x = self.fc1(x)  # [num_nodes, num_classes]

        return x  # 返回 [num_nodes, num_classes]
