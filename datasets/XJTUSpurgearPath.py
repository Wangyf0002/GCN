import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.PathGraph import PathGraph  # 导入PathGraph类，用于生成图数据
from datasets.AuxFunction import FFT  # 导入FFT类，用于进行快速傅里叶变换
from tqdm import tqdm  # 导入进度条库
import pickle  # 导入pickle，用于保存和加载数据
# ------------------------------------------------------------
signal_size = 1024  # 每个信号片段的长度为1024

# label: 定义数据的标签类别
label = [i for i in range(10)]  # 假设有10个类别（0到9）

# generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task, test=False):
    '''
    该函数用于生成最终的训练集和测试集
    root: 数据集所在路径
    normalname: 正常数据列表
    dataname: 故障数据列表
    '''
    file = ['15Hz', '20Hz']  # 假设数据集包含15Hz和20Hz两种频率的数据

    Subdir = []  # 存储每个子目录的路径
    for i in file:
        sub_root = os.path.join('/tmp', root, i)  # 构建每个频率的子目录路径
        file_name = os.listdir(sub_root)  # 获取子目录中的文件列表
        for j in file_name:
            Subdir.append(os.path.join('/tmp', sub_root, j))  # 将文件路径添加到Subdir列表

    data = []  # 存储所有的数据

    # 遍历所有的子目录，加载数据
    for i in tqdm(range(len(Subdir))):  # 使用tqdm显示进度条
        # 调用data_load函数来加载每个文件的数据
        data1 = data_load(sample_length, Subdir[i], label=label[i], InputType=InputType, task=task)
        data += data1  # 将加载的数据添加到data列表中

    return data  # 返回最终的训练和测试数据


def data_load(signal_size, root, label, InputType, task):
    '''
    该函数主要用于生成测试数据和训练数据
    root: 数据所在路径
    '''

    # 读取数据文件，假设文件为csv格式，且数据在第二列
    fl = pd.read_csv(root, sep='\t', usecols=[1], header=None)
    fl = fl.values  # 转换为numpy数组
    fl = (fl - fl.min()) / (fl.max() - fl.min())  # 对数据进行归一化处理
    fl = fl.reshape(-1,)  # 将数据展平为一维数组
    data = []  # 存储每个信号片段的数据
    start, end = 0, signal_size  # 每个信号片段的起始和结束索引
    while end <= fl[:signal_size*1000].shape[0]:  # 遍历整个数据
        # 根据输入类型选择不同的数据处理方式
        if InputType == "TD":  # 时间域数据
            x = fl[start:end]
        elif InputType == "FD":  # 频域数据
            x = fl[start:end]
            x = FFT(x)  # 对数据进行FFT变换
        else:
            print("The InputType is wrong!!")

        data.append(x)  # 将处理后的数据添加到data列表中
        start += signal_size  # 更新起始索引
        end += signal_size  # 更新结束索引

    # 创建图数据集
    graphset = PathGraph(10, data, label, task)  # 使用PathGraph类生成图数据集
    return graphset  # 返回图数据集


class XJTUSpurgearPath(object):
    num_classes = 10  # 数据集的类别数

    def __init__(self, sample_length, data_dir, InputType, task):
        '''
        构造函数，初始化数据集的基本信息
        sample_length: 每个样本的长度
        data_dir: 数据所在目录
        InputType: 输入类型，时间域（TD）或频域（FD）
        task: 任务类型（如分类任务）
        '''
        self.sample_length = sample_length  # 存储样本长度
        self.data_dir = data_dir  # 存储数据路径
        self.InputType = InputType  # 存储输入数据类型
        self.task = task  # 存储任务类型

    def data_preprare(self, test=False):
        '''
        该函数用于准备数据集，如果数据已经处理过则直接加载，否则生成新的数据
        test: 是否仅返回测试集
        '''
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            # 如果数据集文件已经存在，直接加载
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            # 如果数据集文件不存在，生成新的数据
            list_data = get_files(self.sample_length, self.data_dir, self.InputType, self.task, test)
            # 将生成的数据保存为pickle文件
            with open(os.path.join(self.data_dir, "XJTUSpurgearPath.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)

        if test:
            # 如果是测试集，直接返回
            test_dataset = list_data
            return test_dataset
        else:
            # 否则将数据集划分为训练集和验证集
            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)
            return train_dataset, val_dataset  # 返回训练集和验证集
