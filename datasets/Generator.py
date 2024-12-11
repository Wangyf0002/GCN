import torch
from math import sqrt
import numpy as np
from torch_geometric.data import Data  # 图数据结构
from scipy.spatial.distance import pdist  # 计算距离矩阵
import copy


def KNN_classify(k, X_set, x):
    """
    K近邻分类算法，用于找到与节点x最近的k个邻居。
    参数:
    - k: 最近邻的数量
    - X_set: 所有节点的特征数据集
    - x: 目标节点
    返回:
    - node_index: 最近邻的索引
    - topK_x: 最近邻的特征
    """
    # 计算 x 与所有节点的欧几里得距离
    distances = [sqrt(np.sum((x_compare - x) ** 2)) for x_compare in X_set]
    nearest = np.argsort(distances)  # 按距离升序排序
    node_index = [i for i in nearest[1:k + 1]]  # 跳过自身节点
    topK_x = [X_set[i] for i in nearest[1:k + 1]]
    return node_index, topK_x


def KNN_weigt(x, topK_x):
    """
    计算节点x与其最近邻的边权重（基于高斯核）。
    参数:
    - x: 当前节点特征
    - topK_x: 最近邻节点的特征
    返回:
    - w: 边权重
    """
    distance = []
    for v_2 in topK_x:
        combine = np.vstack([x, v_2])  # 将当前节点与邻居堆叠
        likely = pdist(combine, 'euclidean')  # 计算欧几里得距离
        distance.append(likely[0])
    beta = np.mean(distance)  # 使用距离均值计算高斯核
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beta ** 2)))
    return w


def KNN_attr(data):
    """
    基于K近邻生成图的边和边权重。
    参数:
    - data: 节点特征数据
    返回:
    - edge_index: 边索引
    - edge_fea: 边权重
    """
    edge_raw0, edge_raw1, edge_fea = [], [], []
    for i, x in enumerate(data):
        node_index, topK_x = KNN_classify(5, data, x)  # 找到5个最近邻
        local_weight = KNN_weigt(x, topK_x)  # 计算边权重
        local_index = np.zeros(5) + i  # 当前节点索引

        edge_raw0 = np.hstack((edge_raw0, local_index))
        edge_raw1 = np.hstack((edge_raw1, node_index))
        edge_fea = np.hstack((edge_fea, local_weight))

    edge_index = [edge_raw0, edge_raw1]
    return edge_index, edge_fea


def cal_sim(data, s1, s2):
    """
    计算两个节点的相似性（余弦相似度）。
    参数:
    - data: 节点特征数据
    - s1, s2: 两个节点索引
    返回:
    - edge_index: 边索引
    - edge_feature: 边权重
    """
    edge_index = [[], []]
    edge_feature = []
    if s1 != s2:
        combine = np.vstack([data[s1], data[s2]])
        likely = 1 - pdist(combine, 'cosine')  # 余弦相似度
        if likely.item() >= 0:  # 只保留相似度为正的边
            edge_index[0].append(s1)
            edge_index[1].append(s2)
            edge_feature.append(1)  # 固定权重为1
    return edge_index, edge_feature


def Radius_attr(data):
    """
    基于半径的方式生成图的边和边权重。
    参数:
    - data: 节点特征数据
    返回:
    - edge_index: 边索引
    - edge_fe: 边权重
    """
    edge_index = np.array([[], []])
    edge_fe = []
    for i in range(len(data)):
        for j in range(len(data)):
            local_edge, w = cal_sim(data, i, j)
            edge_index = np.hstack((edge_index, local_edge))
            if any(w):
                edge_fe.append(w[0])
    return edge_index, edge_fe


def Path_attr(data):
    """
    基于路径生成图的边和边权重。
    参数:
    - data: 节点特征数据
    返回:
    - node_edge: 边索引
    - w: 边权重
    """
    node_edge = [[], []]
    for i in range(len(data) - 1):
        node_edge[0].append(i)
        node_edge[1].append(i + 1)

    distance = []
    for j in range(len(data) - 1):
        combine = np.vstack([data[j], data[j + 1]])
        likely = pdist(combine, 'euclidean')  # 计算欧几里得距离
        distance.append(likely[0])

    beta = np.mean(distance)  # 高斯核参数
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beta ** 2)))  # 高斯核计算边权重
    return node_edge, w


def Gen_graph(graphType, data, label, task):
    """
    生成图数据集。
    参数:
    - graphType: 图的类型（KNNGraph, RadiusGraph, PathGraph）
    - data: 输入数据
    - label: 图的标签
    - task: 任务类型（Node或Graph）
    返回:
    - data_list: 图数据集
    """
    data_list = []
    for graph_feature in data:
        if task == 'Node':
            labels = np.zeros(len(graph_feature)) + label
        elif task == 'Graph':
            labels = [label]
        else:
            print("There is no such task!!")

        if graphType == 'KNNGraph':
            node_edge, w = KNN_attr(graph_feature)
        elif graphType == 'RadiusGraph':
            node_edge, w = Radius_attr(graph_feature)
        elif graphType == 'PathGraph':
            node_edge, w = Path_attr(graph_feature)
        else:
            print("This GraphType is not included!")
            continue

        # 构建图数据
        node_features = torch.tensor(graph_feature, dtype=torch.float)
        graph_label = torch.tensor(labels, dtype=torch.long)
        edge_index = torch.tensor(node_edge, dtype=torch.long)
        edge_features = torch.tensor(w, dtype=torch.float)
        graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
        data_list.append(graph)

    return data_list

# import torch
# from math import sqrt
# import numpy as np
# from torch_geometric.data import Data
# from scipy.spatial.distance import pdist
# import copy
#
# def KNN_classify(k,X_set,x):
#     """
#     k:number of neighbours
#     X_set: the datset of x
#     x: to find the nearest neighbor of data x
#     """
#
#     distances = [sqrt(np.sum((x_compare-x)**2)) for x_compare in X_set]
#     nearest = np.argsort(distances)
#     node_index  = [i for i in nearest[1:k+1]]
#     topK_x = [X_set[i] for i in nearest[1:k+1]]
#     return  node_index,topK_x
#
#
# def KNN_weigt(x,topK_x):
#     distance = []
#     v_1 = x
#     data_2 = topK_x
#     for i in range(len(data_2)):
#         v_2 = data_2[i]
#         combine = np.vstack([v_1, v_2])
#         likely = pdist(combine, 'euclidean')
#         distance.append(likely[0])
#     beata = np.mean(distance)
#     w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))
#     return w
#
#
# def KNN_attr(data):
#     '''
#     for KNNgraph
#     :param data:
#     :return:
#     '''
#     edge_raw0 = []
#     edge_raw1 = []
#     edge_fea = []
#     for i in range(len(data)):
#         x = data[i]
#         node_index, topK_x= KNN_classify(5,data,x)
#         loal_weigt = KNN_weigt(x,topK_x)
#         local_index = np.zeros(5)+i
#
#         edge_raw0 = np.hstack((edge_raw0,local_index))
#         edge_raw1 = np.hstack((edge_raw1,node_index))
#         edge_fea = np.hstack((edge_fea,loal_weigt))
#
#     edge_index = [edge_raw0, edge_raw1]
#
#     return edge_index, edge_fea
#
#
#
# def cal_sim(data,s1,s2):
#     edge_index = [[],[]]
#     edge_feature = []
#     if s1 != s2:
#         v_1 = data[s1]
#         v_2 = data[s2]
#         combine = np.vstack([v_1, v_2])
#         likely = 1- pdist(combine, 'cosine')
# #         w = np.exp((-(likely[0]) ** 2) / 30)
#         if likely.item() >= 0:
#             w = 1
#             edge_index[0].append(s1)
#             edge_index[1].append(s2)
#             edge_feature.append(w)
#     return edge_index,edge_feature
#
#
#
# def Radius_attr(data):
#     '''
#     for RadiusGraph
#     :param feature:
#     :return:
#     '''
#     s1 = range(len(data))
#     s2 = copy.deepcopy(s1)
#     edge_index = np.array([[], []])  # 一个故障样本与其他故障样本匹配生成一次图
#     edge_fe = []
#     for i in s1:
#         for j in s2:
#             local_edge, w = cal_sim(data, i, j)
#             edge_index = np.hstack((edge_index, local_edge))
#             if any(w):
#                 edge_fe.append(w[0])
#     return edge_index,edge_fe
#
#
# def Path_attr(data):
#
#     node_edge = [[], []]
#
#     for i in range(len(data) - 1):
#         node_edge[0].append(i)
#         node_edge[1].append(i + 1)
#
#     distance = []
#     for j in range(len(data) - 1):
#         v_1 = data[j]
#         v_2 = data[j + 1]
#         combine = np.vstack([v_1, v_2])
#         likely = pdist(combine, 'euclidean')
#         distance.append(likely[0])
#
#     beata = np.mean(distance)
#     w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))  #Gussion kernel高斯核
#
#     return node_edge, w
#
#
# def Gen_graph(graphType, data, label,task):
#     data_list = []
#     if graphType == 'KNNGraph':
#         for i in range(len(data)):
#             graph_feature = data[i]
#             if task == 'Node':
#                 labels = np.zeros(len(graph_feature)) + label
#             elif task == 'Graph':
#                 labels = [label]
#             else:
#                 print("There is no such task!!")
#             node_edge, w = KNN_attr(data[i])
#             node_features = torch.tensor(graph_feature, dtype=torch.float)
#             graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
#             edge_index = torch.tensor(node_edge, dtype=torch.long)
#             edge_features = torch.tensor(w, dtype=torch.float)
#             graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
#             data_list.append(graph)
#
#     elif graphType == 'RadiusGraph':
#         for i in range(len(data)):
#             graph_feature = data[i]
#             if task == 'Node':
#                 labels = np.zeros(len(graph_feature)) + label
#             elif task == 'Graph':
#                 labels = [label]
#             else:
#                 print("There is no such task!!")
#             node_edge, w = Radius_attr(graph_feature)
#             node_features = torch.tensor(graph_feature, dtype=torch.float)
#             graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
#             edge_index = torch.tensor(node_edge, dtype=torch.long)
#             edge_features = torch.tensor(w, dtype=torch.float)
#             graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
#             data_list.append(graph)
#
#     elif graphType == 'PathGraph':
#         for i in range(len(data)):
#             graph_feature = data[i]
#             if task == 'Node':
#                 labels = np.zeros(len(graph_feature)) + label
#             elif task == 'Graph':
#                 labels = [label]
#             else:
#                 print("There is no such task!!")
#             node_edge, w = Path_attr(graph_feature)
#             node_features = torch.tensor(graph_feature, dtype=torch.float)
#             graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
#             edge_index = torch.tensor(node_edge, dtype=torch.long)
#             edge_features = torch.tensor(w, dtype=torch.float)
#             graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
#             data_list.append(graph)
#
#     else:
#         print("This GraphType is not included!")
#     return data_list
