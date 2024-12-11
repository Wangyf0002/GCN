import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.PathGraph import PathGraph  # 导入自定义的图生成工具
from datasets.AuxFunction import FFT  # 导入自定义的快速傅里叶变换函数
import pickle

# -------------------------------------------------------------
# 定义信号截取窗口大小
signal_size = 1024

# 故障类别名称列表，每种类型对应一个子文件夹名称
fault_name = [
    '1ndBearing_ball',
    '1ndBearing_inner',
    '1ndBearing_mix(inner+outer+ball)',
    '1ndBearing_outer',
    '2ndPlanetary_brokentooth',
    '2ndPlanetary_missingtooth',
    '2ndPlanetary_normalstate',
    '2ndPlanetary_rootcracks',
    '2ndPlanetary_toothwear'
]

# 类别标签，与故障类别一一对应
label = [i for i in range(9)]


# ---------------------- 数据生成函数 --------------------------

def get_files(sample_length, root, InputType, task, test=False):

    # 生成最终的训练集和测试集
    # 参数:
    # - sample_length: 信号截取窗口大小
    # - root: 数据集根目录
    # - InputType: 输入类型，'TD' 为时间域，'FD' 为频域
    # - task: 数据任务标签
    # - test: 是否为测试数据（默认 False）
    # 返回:
    # - 图数据列表，数据格式为 [Data(edge_attr, edge_index, x, y), ...]

    data = []

    for i in tqdm(range(len(fault_name))):  # 遍历所有故障类别
        data_name = 'Data_Chan1.txt'  # 每个类别目录下的数据文件名
        path2 = os.path.join('/tmp', root, fault_name[i], data_name)  # 生成完整路径
        # 加载并处理当前类别的数据
        data1 = data_load(sample_length, filename=path2, label=label[i], InputType=InputType, task=task)
        data += data1  # 合并所有类别数据

    return data


def data_load(signal_size, filename, label, InputType, task):
    # 加载信号数据并生成对应的图数据集
    # 参数:
    # - signal_size: 每段信号的长度
    # - filename: 数据文件路径
    # - label: 类别标签
    # - InputType: 输入类型，时间域 'TD' 或频域 'FD'
    # - task: 数据任务标签
    # 返回:
    # - 图数据列表

    # 加载信号数据，跳过前 14 行（可能是元信息），无列名
    fl = pd.read_csv(filename, skiprows=range(14), header=None)
    # 对信号进行 Min-Max 归一化处理
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    # 转换为一维 numpy 数组
    fl = fl.values.reshape(-1, )

    data = []  # 存储分段后的信号
    start, end = 0, signal_size  # 定义滑窗起点和终点
    while end <= fl[:signal_size * 1000].shape[0]:  # 按窗口大小滑动，生成信号段
        if InputType == "TD":  # 如果输入为时间域信号
            x = fl[start:end]
        elif InputType == "FD":  # 如果输入为频域信号，先进行 FFT 转换
            x = fl[start:end]
            x = FFT(x)
        else:  # 输入类型错误
            print("The InputType is wrong!!")

        data.append(x)  # 将分段信号加入列表
        start += signal_size  # 窗口右移
        end += signal_size

    # 将信号段转化为图数据
    graphset = PathGraph(10, data, label, task)  # 10 表示每张图包含的节点数
    return graphset


# -------------------- 数据集准备类 ---------------------------

class XJTUGearboxPath(object):
    num_classes = 9  # 故障分类数

    def __init__(self, sample_length, data_dir, InputType, task):
        # 初始化类参数
        # 参数:
        # - sample_length: 信号截取窗口大小
        # - data_dir: 数据集根目录
        # - InputType: 输入类型，时间域 'TD' 或频域 'FD'
        # - task: 数据任务标签

        self.sample_length = sample_length
        self.data_dir = data_dir
        self.InputType = InputType
        self.task = task

    def data_preprare(self, test=False):
        # 准备数据集，支持训练和测试模式
        # 参数:
        # - test: 是否生成测试数据集（默认 False）
        # 返回:
        # - 训练模式: 返回训练集和验证集
        # - 测试模式: 返回测试数据集

        # 如果目录下存在 .pkl 文件，直接加载
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:  # 否则生成图数据集并保存为 .pkl 文件
            list_data = get_files(self.sample_length, self.data_dir, self.InputType, self.task, test)
            with open(os.path.join(self.data_dir, "XJTUGearboxPath.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)

        if test:  # 如果是测试模式，返回测试数据集
            test_dataset = list_data
            return test_dataset
        else:  # 否则划分训练集和验证集
            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)
            return train_dataset, val_dataset

# 原代码：
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# from datasets.PathGraph import PathGraph
# from datasets.AuxFunction import FFT
# import pickle
#
# # -------------------------------------------------------------
# signal_size = 1024
#
# fault_name = ['1ndBearing_ball', '1ndBearing_inner', '1ndBearing_mix(inner+outer+ball)', '1ndBearing_outer',
#               '2ndPlanetary_brokentooth', '2ndPlanetary_missingtooth', '2ndPlanetary_normalstate',
#               '2ndPlanetary_rootcracks', '2ndPlanetary_toothwear']
# # label
# label = [i for i in range(9)]
#
#
# # generate Training Dataset and Testing Dataset
# def get_files(sample_length, root, InputType, task, test=False):
#     '''
#     This function is used to generate the final training set and test set.
#     root:The location of the data set
#     normalname:List of normal data
#     dataname:List of failure data
#     '''
#     data = []
#
#     for i in tqdm(range(len(fault_name))):
#         data_name = 'Data_Chan1.txt'
#         path2 = os.path.join('/tmp', root, fault_name[i], data_name)
#         data1 = data_load(sample_length, filename=path2, label=label[i], InputType=InputType, task=task)
#         data += data1
#
#     return data
#
#
# # 读取原始数据
# def data_load(signal_size, filename, label, InputType, task):
#     """
#     This function is mainly used to generate test data and training data.
#     filename:Data location
#     axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
#     signal_size: 每次截取的信号长度。
#     filename: 数据文件路径（如 Data_Chan1.txt）。
#     label: 当前数据类别标签。
#     InputType: 数据输入类型，可能是时间域 (TD) 或频域 (FD)。
#     task: 数据任务信息。
#     """
#     fl = pd.read_csv(filename, skiprows=range(14), header=None)
#     fl = (fl - fl.min()) / (fl.max() - fl.min())
#     fl = fl.values
#     fl = fl.reshape(-1, )
#     data = []
#     start, end = 0, signal_size
#     while end <= fl[:signal_size * 1000].shape[0]:
#         if InputType == "TD":
#             x = fl[start:end]
#         elif InputType == "FD":
#             x = fl[start:end]
#             x = FFT(x)
#         else:
#             print("The InputType is wrong!!")
#
#         data.append(x)
#         start += signal_size
#         end += signal_size
#
#     graphset = PathGraph(10, data, label, task)
#
#     return graphset
#
#
# class XJTUGearboxPath(object):
#     num_classes = 9
#
#     def __init__(self, sample_length, data_dir, InputType, task):
#         self.sample_length = sample_length
#         self.data_dir = data_dir
#         self.InputType = InputType
#         self.task = task
#
#     def data_preprare(self, test=False):
#         if len(os.path.basename(self.data_dir).split('.')) == 2:
#             with open(self.data_dir, 'rb') as fo:
#                 list_data = pickle.load(fo, encoding='bytes')
#         else:
#             list_data = get_files(self.sample_length, self.data_dir, self.InputType, self.task, test)
#             with open(os.path.join(self.data_dir, "XJTUGearboxPath.pkl"), 'wb') as fo:
#                 pickle.dump(list_data, fo)
#
#         if test:
#             test_dataset = list_data
#             return test_dataset
#         else:
#
#             train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)
#
#             return train_dataset, val_dataset