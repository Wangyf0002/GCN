#!/home/wang/anaconda3/envs/pytorch11/bin/python
# -*- coding:utf-8 -*-

import argparse  # 用于处理命令行参数
import os  # 用于文件路径和目录操作
from datetime import datetime  # 用于获取当前时间
from utils.logger import setlogger  # 用于设置日志记录器
import logging  # 用于记录日志
from utils.train_graph_utils import train_utils  # 导入训练工具类

# 初始化 args 变量
args = None

def parse_args():
    """
    解析命令行参数，返回一个包含所有参数值的对象
    """
    parser = argparse.ArgumentParser(description='Train')  # 创建一个命令行参数解析器
    # basic parameters: 基本参数，控制模型和训练过程
    parser.add_argument('--model_name', type=str, default='ChebyNet', help='the name of the model')  # 模型名称
    parser.add_argument('--sample_length', type=int, default=1024, help='batchsize of the training process')  # 样本长度，通常与输入数据相关
    parser.add_argument('--data_name', type=str, default='XJTUGearboxKnn', help='the name of the data')  # 数据集名称
    parser.add_argument('--Input_type', choices=['TD', 'FD','other'], type=str, default='TD', help='the input type decides the length of input')  # 输入数据类型
    parser.add_argument('--data_dir', type=str, default= "./data/XJTUGearbox/XJTUGearboxKnn.pkl", help='the directory of the data')  # 数据文件路径
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')  # 指定使用的GPU设备
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')  # 模型保存目录
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')  # 每个批次的大小
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')  # 训练过程中的工作线程数

    # Define the tasks: 定义任务类型
    parser.add_argument('--task', choices=['Node', 'Graph'], type=str, default='Node', help='Node classification or Graph classification')  # 任务类型：节点分类或图分类
    parser.add_argument('--pooltype', choices=['TopKPool', 'EdgePool', 'ASAPool', 'SAGPool'], type=str, default='EdgePool', help='For the Graph classification task')  # 图分类任务的池化方法

    # optimization information: 优化相关的参数
    parser.add_argument('--layer_num_last', type=int, default=0, help='the number of last layers which unfreeze')  # 冻结的最后层数量
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')  # 优化器类型：SGD或Adam
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')  # 初始学习率
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')  # SGD优化器的动量
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay')  # 权重衰减
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')  # 学习率调度方式
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')  # 学习率调度参数
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')  # 学习率衰减步数

    # save, load and display information: 保存、加载和显示相关的信息
    parser.add_argument('--resume', type=str, default='', help='the directory of the resume training model')  # 恢复训练模型的目录
    parser.add_argument('--max_model_num', type=int, default=1, help='the number of most recent models to save')  # 最大保存模型数量
    parser.add_argument('--max_epoch', type=int, default=30, help='max number of epoch')  # 最大训练轮数
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')  # 打印训练日志的间隔

    args = parser.parse_args()  # 解析命令行参数
    return args

if __name__ == '__main__':
    # import pdb;pdb.set_trace()  # 可以在此添加调试断点
    args = parse_args()  # 解析命令行参数
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()  # 设置使用的CUDA设备

    # Prepare the saving path for the model: 准备模型保存路径
    if args.task == 'Node':
        # 节点分类任务的保存路径
        sub_dir = args.task + '_' +args.model_name+'_'+args.data_name + '_' + args.Input_type +'_'+datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    else:
        # 图分类任务的保存路径
        sub_dir = args.task + '_' +  args.model_name + '_' + args.pooltype + '_' + args.data_name + '_' + args.Input_type + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)  # 完整的保存路径
    if not os.path.exists(save_dir):  # 如果保存路径不存在，则创建该目录
        os.makedirs(save_dir)

    # set the logger: 设置日志记录器
    setlogger(os.path.join(save_dir, 'train.log'))  # 日志文件保存路径

    # save the args: 保存命令行参数到日志
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))  # 打印每个参数及其对应值到日志文件

    trainer = train_utils(args, save_dir)  # 初始化训练工具类
    trainer.setup()  # 设置训练过程的配置
    trainer.train()  # 开始训练过程
