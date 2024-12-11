import os
import numpy as np
from itertools import islice  # 用于跳过文件的前几个行
from sklearn.model_selection import train_test_split  # 用于数据集划分
from tqdm import tqdm  # 进度条库
from datasets.PathGraph import PathGraph  # 导入PathGraph类，用于生成图数据
from datasets.AuxFunction import FFT  # 导入FFT类，用于快速傅里叶变换
import pickle  # 用于保存和加载数据

# -------------------------------------------------------------
signal_size = 1024  # 每个信号片段的长度为1024

# Data names of 5 bearing fault types under two working conditions
Bdata = ["ball_20_0.csv", "comb_20_0.csv", "health_20_0.csv", "inner_20_0.csv", "outer_20_0.csv",
         "ball_30_2.csv", "comb_30_2.csv", "health_30_2.csv", "inner_30_2.csv", "outer_30_2.csv"]
label1 = [i for i in range(0, 10)]  # Bearing fault types对应的标签（0到9）

# Data names of 5 gear fault types under two working conditions
Gdata = ["Chipped_20_0.csv", "Health_20_0.csv", "Miss_20_0.csv", "Root_20_0.csv", "Surface_20_0.csv",
         "Chipped_30_2.csv", "Health_30_2.csv", "Miss_30_2.csv", "Root_30_2.csv", "Surface_30_2.csv"]
labe12 = [i for i in range(10, 20)]  # Gear fault types对应的标签（10到19）


# generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task, test=False):
    '''
    该函数用于生成最终的训练集和测试集。
    root: 数据集的根目录
    datasetname: 包含数据集的文件夹
    '''
    # 获取 bearing 数据集和 gear 数据集的路径
    datasetname = os.listdir(os.path.join(root, os.listdir(root)[2]))  # 获取数据集的子文件夹（bearing和gear）
    # root1 = os.path.join("/tmp", root, os.listdir(root)[2], datasetname[0])  # Bearing 数据集路径
    # root2 = os.path.join("/tmp", root, os.listdir(root)[2], datasetname[2])  # Gear 数据集路径
    root1 = os.path.join("/tmp", root, os.listdir(root)[0])  # Bearing 数据集路径
    root2 = os.path.join("/tmp", root, os.listdir(root)[2])  # Gear 数据集路径
    print(root)
    print(root1)
    print(root2)
    data = []  # 用于存储所有数据

    # 处理 Bearing 数据集
    for i in tqdm(range(len(Bdata))):  # 遍历每个 bearing 数据文件
        path1 = os.path.join('/tmp', root1, Bdata[i])  # 拼接数据文件路径
        data1 = data_load(sample_length, path1, dataname=Bdata[i], label=label1[i], InputType=InputType, task=task)
        data += data1  # 将加载的数据添加到总数据列表

    # 处理 Gear 数据集
    for j in tqdm(range(len(Gdata))):  # 遍历每个 gear 数据文件
        path2 = os.path.join('/tmp', root2, Gdata[j])  # 拼接数据文件路径
        data2 = data_load(sample_length, path2, dataname=Gdata[j], label=labe12[j], InputType=InputType, task=task)
        data += data2  # 将加载的数据添加到总数据列表

    return data  # 返回最终的训练和测试数据


def data_load(signal_size, filename, dataname, label, InputType, task):
    '''
    该函数用于加载和处理数据。
    filename: 数据文件路径
    '''
    # 打开文件，读取数据（支持两种不同的分隔符）
    f = open(filename, "r", encoding='gb18030', errors='ignore')  # 使用GB18030编码打开文件
    fl = []
    if dataname == "ball_20_0.csv":  # 如果是特定的数据文件，按逗号分隔
        for line in islice(f, 16, None):  # 跳过前16行
            line = line.rstrip()  # 去除行尾的空白字符
            word = line.split(",", 8)  # 按逗号分隔
            fl.append(eval(word[1]))  # 提取振动信号（x方向）并加入列表
    else:  # 其他文件按制表符分隔
        for line in islice(f, 16, None):  # 跳过前16行
            line = line.rstrip()  # 去除行尾的空白字符
            word = line.split("\t", 8)  # 按制表符分隔
            fl.append(eval(word[1]))  # 提取振动信号（x方向）并加入列表

    fl = np.array(fl)  # 转换为numpy数组
    fl = (fl - fl.min()) / (fl.max() - fl.min())  # 数据归一化
    fl = fl.reshape(-1, )  # 展平数据
    data = []  # 存储信号片段数据
    start, end = 0, signal_size  # 信号片段的起始和结束位置
    while end <= fl[:signal_size * 1000].shape[0]:  # 读取每个信号片段
        if InputType == "TD":  # 如果输入类型是时间域
            x = fl[start:end]  # 取时间域数据
        elif InputType == "FD":  # 如果输入类型是频域
            x = fl[start:end]  # 取时间域数据
            x = FFT(x)  # 对数据进行FFT转换
        else:
            print("The InputType is wrong!!")  # 输入类型错误时提示

        data.append(x)  # 将数据片段加入到数据列表中
        start += signal_size  # 更新起始位置
        end += signal_size  # 更新结束位置

    # 创建图数据集（调用PathGraph类）
    graphset = PathGraph(10, data, label, task)

    return graphset  # 返回图数据集


# --------------------------------------------------------------------------------------------------------------------
class SEUPath(object):
    num_classes = 20  # 数据集的类别数（包括bearing和gear）

    def __init__(self, sample_length, data_dir, InputType, task):
        '''
        构造函数，初始化数据集的信息
        sample_length: 每个样本的长度
        data_dir: 数据集目录路径
        InputType: 输入数据类型（时间域TD或频域FD）
        task: 任务类型（如分类任务）
        '''
        self.sample_length = sample_length  # 保存样本长度
        self.data_dir = data_dir  # 保存数据目录路径
        self.InputType = InputType  # 保存输入数据类型
        self.task = task  # 保存任务类型

    def data_preprare(self, test=False):
        '''
        该函数用于准备数据集，如果数据已处理则直接加载，否则生成新的数据
        test: 是否只返回测试集
        '''
        if len(os.path.basename(self.data_dir).split('.')) == 2:  # 检查数据集文件是否存在
            # 如果文件存在，直接加载数据
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            # 如果文件不存在，生成数据并保存
            list_data = get_files(self.sample_length, self.data_dir, self.InputType, self.task, test)
            with open(os.path.join(self.data_dir, "SEUPath.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)

        if test:
            # 如果是测试集，直接返回
            test_dataset = list_data
            return test_dataset
        else:
            # 否则划分训练集和验证集
            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)
            return train_dataset, val_dataset  # 返回训练集和验证集
