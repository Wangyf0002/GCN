import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm  # 引入图卷积层和批量归一化层

class GCN(torch.nn.Module):
    def __init__(self, feature, out_channel):
        """
        初始化图卷积网络模型 (GCN)
        Args:
            feature: 输入节点特征的维度
            out_channel: 输出的维度（通常是类别数量）
        """
        super(GCN, self).__init__()

        # 第一层图卷积，将输入的特征维度映射到 1024 维
        self.GConv1 = GCNConv(feature, 1024)
        # 对第一层卷积的输出进行批量归一化，稳定训练过程
        self.bn1 = BatchNorm(1024)

        # 第二层图卷积，将 1024 维特征继续映射到 1024 维
        self.GConv2 = GCNConv(1024, 1024)
        # 对第二层卷积的输出进行批量归一化
        self.bn2 = BatchNorm(1024)

        # 全连接层，将图卷积后的 1024 维特征降维到 512 维，并激活
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),  # 线性变换，1024 -> 512
            nn.ReLU(inplace=True)  # 激活函数
        )
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(0.2)

        # 最后一层全连接层，将 512 维特征降维到输出的维度（类别数量）
        self.fc1 = nn.Sequential(
            nn.Linear(512, out_channel)  # 线性变换，512 -> out_channel
        )

    def forward(self, data):
        """
        定义前向传播过程
        Args:
            data: 输入的图数据，包含以下属性：
                  - data.x: 节点特征矩阵，形状为 [节点数, 特征维度]
                  - data.edge_index: 边索引矩阵，形状为 [2, 边数]
                  - data.edge_attr: 边特征矩阵，形状为 [边数]
        Returns:
            x: 模型的输出，形状为 [节点数, 输出维度]
        """
        # 提取输入的节点特征、边索引和边权重
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # 第一层图卷积 + 批量归一化 + ReLU激活
        x = self.GConv1(x, edge_index, edge_weight)  # 图卷积操作
        x = self.bn1(x)  # 批量归一化
        x = F.relu(x)  # ReLU激活

        # 第二层图卷积 + 批量归一化 + ReLU激活
        x = self.GConv2(x, edge_index, edge_weight)  # 图卷积操作
        x = self.bn2(x)  # 批量归一化
        x = F.relu(x)  # ReLU激活

        # 全连接层 + Dropout
        x = self.fc(x)  # 将图卷积后的特征传入全连接层，降维到 512
        x = self.dropout(x)  # 随机丢弃部分神经元，防止过拟合
        x = self.fc1(x)  # 最后一层全连接，得到输出

        return x  # 返回最终输出
