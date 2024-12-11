#!/home/wang/anaconda3/envs/pytorch11/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
from torch_geometric.data import DataLoader
import models
import models2
import datasets
from utils.save import Save_Tool
from utils.freeze import set_freeze_by_id
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

class train_utils(object):
    def __init__(self, args, save_dir):
        """
        初始化训练器
        :param args: 命令行参数，包括模型配置、数据集信息等
        :param save_dir: 模型保存目录
        """
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        初始化数据集、模型、损失函数和优化器
        """
        args = self.args

        # 检查是否使用 GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # 使用GPU
            self.device_count = torch.cuda.device_count()  # GPU数量
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")  # 没有GPU时使用CPU
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # 加载数据集
        Dataset = getattr(datasets, args.data_name)  # 从datasets模块中动态加载指定数据集类
        self.datasets = {}

        # 准备训练和验证数据集
        self.datasets['train'], self.datasets['val'] = Dataset(args.sample_length, args.data_dir, args.Input_type, args.task).data_preprare()

        # 定义训练和验证数据加载器
        self.dataloaders = {x: DataLoader(self.datasets[x], batch_size=args.batch_size,
                                           shuffle=(True if x == 'train' else False),
                                           num_workers=args.num_workers,
                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}

        # 根据输入类型定义特征的维度
        InputType = args.Input_type
        if InputType == "TD":
            feature = args.sample_length
        elif InputType == "FD":
            feature = int(args.sample_length / 2)
        elif InputType == "other":
            feature = 1
        else:
            print("The InputType is wrong!!")

        # 加载模型
        if args.task == 'Node':
            self.model = getattr(models, args.model_name)(feature=feature, out_channel=Dataset.num_classes)
        elif args.task == 'Graph':
            self.model = getattr(models2, args.model_name)(feature=feature, out_channel=Dataset.num_classes, pooltype=args.pooltype)
        else:
            print('The task is wrong!')

        # 如果冻结了某些层，则设置冻结层
        if args.layer_num_last != 0:
            set_freeze_by_id(self.model, args.layer_num_last)

        # 如果有多台GPU，使用DataParallel来加速训练
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # 定义优化器
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # 设置学习率衰减
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # 加载模型检查点
        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':  # 加载 .tar 格式的检查点
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':  # 加载 .pth 格式的检查点
                self.model.load_state_dict(torch.load(args.resume, map_location=self.device))

        # 将模型放到指定设备（GPU或CPU）
        self.model.to(self.device)
        # 定义损失函数（这里使用交叉熵损失）
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.MSELoss()  # 如果是回归任务，使用MSELoss

    def train(self):
        """
        训练过程
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        # 用于保存模型的工具
        save_list = Save_Tool(max_num=args.max_model_num)

        # 进行每个epoch的训练
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)

            # 更新学习率
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # 每个epoch分为训练和验证两个阶段
            for phase in ['train', 'val']:
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_precision = 0.0
                epoch_recall = 0.0
                epoch_f1 = 0.0

                # 设置模型为训练模式或验证模式
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                sample_num = 0
                true_labels = []
                pred_labels = []

                for data in self.dataloaders[phase]:
                    inputs = data.to(self.device)  # 将数据送到GPU或CPU
                    labels = inputs.y  # 获取标签

                    # 根据任务类型，计算正确的样本数量
                    if args.task == 'Node':
                        bacth_num = inputs.num_nodes
                        sample_num += len(labels)
                    elif args.task == 'Graph':
                        bacth_num = inputs.num_graphs
                        sample_num += len(labels)
                    else:
                        print("There is no such task!!")

                    # 在训练阶段计算梯度，在验证阶段不计算梯度
                    with torch.set_grad_enabled(phase == 'train'):
                        # 前向传播
                        if args.task == 'Node':
                            logits = self.model(inputs)
                        elif args.task == 'Graph':
                            logits = self.model(inputs, args.pooltype)
                        else:
                            print("There is no such task!!")

                        # 计算损失和准确度
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * bacth_num
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # 将标签和预测结果加入列表用于计算精确率、召回率和F1分数
                        true_labels.extend(labels.cpu().numpy())
                        pred_labels.extend(pred.cpu().numpy())

                        # 在训练阶段反向传播和优化
                        if phase == 'train':
                            self.optimizer.zero_grad()  # 清除梯度
                            loss.backward()  # 反向传播
                            self.optimizer.step()  # 更新优化器

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += bacth_num

                            # 每隔一定步数打印训练信息
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {}, Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_loss, batch_acc, sample_per_sec, batch_time
                                ))

                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # 计算精确率、召回率和F1分数
                epoch_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=1)
                epoch_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=1)
                epoch_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=1)

                # 如果有学习率调度器，更新学习率
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # 打印每个epoch的训练和验证结果
                epoch_loss = epoch_loss / sample_num
                epoch_acc = epoch_acc / sample_num

                logging.info(
                    'Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}, Cost {:.4f} sec'.format(
                        epoch, phase, epoch_loss, phase, epoch_acc, epoch_precision, epoch_recall, epoch_f1,
                        time.time() - epoch_start
                    ))

                # 在验证阶段保存模型
                if phase == 'val':
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)

                    # 保存最佳模型（根据验证准确率）
                    if epoch_acc > best_acc or epoch > args.max_epoch - 2:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

        # 训练完成后，计算最终模型的混淆矩阵
        logging.info("Training completed. Now calculating confusion matrix on the validation set.")
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        logging.info(f"Final Confusion Matrix:\n{conf_matrix}")


    # def train(self):
    #     """
    #     训练过程
    #     """
    #
    #
    #     args = self.args
    #
    #     step = 0
    #     best_acc = 0.0
    #     batch_count = 0
    #     batch_loss = 0.0
    #     batch_acc = 0
    #     step_start = time.time()
    #
    #     # 用于保存模型的工具
    #     save_list = Save_Tool(max_num=args.max_model_num)
    #
    #     # 进行每个epoch的训练
    #     for epoch in range(self.start_epoch, args.max_epoch):
    #         logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
    #
    #         # 更新学习率
    #         if self.lr_scheduler is not None:
    #             logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
    #         else:
    #             logging.info('current lr: {}'.format(args.lr))
    #
    #         # 每个epoch分为训练和验证两个阶段
    #         for phase in ['train', 'val']:
    #             epoch_start = time.time()
    #             epoch_acc = 0
    #             epoch_loss = 0.0
    #
    #             # 设置模型为训练模式或验证模式
    #             if phase == 'train':
    #                 self.model.train()
    #             else:
    #                 self.model.eval()
    #
    #             sample_num = 0
    #             for data in self.dataloaders[phase]:
    #                 inputs = data.to(self.device)  # 将数据送到GPU或CPU
    #                 labels = inputs.y  # 获取标签
    #
    #                 # 根据任务类型，计算正确的样本数量
    #                 if args.task == 'Node':
    #                     bacth_num = inputs.num_nodes
    #                     sample_num += len(labels)
    #                 elif args.task == 'Graph':
    #                     bacth_num = inputs.num_graphs
    #                     sample_num += len(labels)
    #                 else:
    #                     print("There is no such task!!")
    #
    #                 # 在训练阶段计算梯度，在验证阶段不计算梯度
    #                 with torch.set_grad_enabled(phase == 'train'):
    #                     # 前向传播
    #                     if args.task == 'Node':
    #                         logits = self.model(inputs)
    #                     elif args.task == 'Graph':
    #                         logits = self.model(inputs, args.pooltype)
    #                     else:
    #                         print("There is no such task!!")
    #
    #                     # 计算损失和准确度
    #                     loss = self.criterion(logits, labels)
    #                     pred = logits.argmax(dim=1)
    #                     correct = torch.eq(pred, labels).float().sum().item()
    #                     loss_temp = loss.item() * bacth_num
    #                     epoch_loss += loss_temp
    #                     epoch_acc += correct
    #
    #                     # 在训练阶段反向传播和优化
    #                     if phase == 'train':
    #                         self.optimizer.zero_grad()  # 清除梯度
    #                         loss.backward()  # 反向传播
    #                         self.optimizer.step()  # 更新优化器
    #
    #                         batch_loss += loss_temp
    #                         batch_acc += correct
    #                         batch_count += bacth_num
    #
    #                         # 每隔一定步数打印训练信息
    #                         if step % args.print_step == 0:
    #                             batch_loss = batch_loss / batch_count
    #                             batch_acc = batch_acc / batch_count
    #                             temp_time = time.time()
    #                             train_time = temp_time - step_start
    #                             step_start = temp_time
    #                             batch_time = train_time / args.print_step if step != 0 else train_time
    #                             sample_per_sec = 1.0 * batch_count / train_time
    #                             logging.info('Epoch: {}, Train Loss: {:.4f} Train Acc: {:.4f},'
    #                                          '{:.1f} examples/sec {:.2f} sec/batch'.format(
    #                                 epoch, batch_loss, batch_acc, sample_per_sec, batch_time
    #                             ))
    #
    #                             batch_acc = 0
    #                             batch_loss = 0.0
    #                             batch_count = 0
    #                         step += 1
    #
    #             # 如果有学习率调度器，更新学习率
    #             if self.lr_scheduler is not None:
    #                 self.lr_scheduler.step()
    #
    #             # 打印每个epoch的训练和验证结果
    #             epoch_loss = epoch_loss / sample_num
    #             epoch_acc = epoch_acc / sample_num
    #
    #             logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
    #                 epoch, phase, epoch_loss, phase, epoch_acc, time.time()-epoch_start
    #             ))
    #
    #             # 在验证阶段保存模型
    #             if phase == 'val':
    #                 model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
    #                 save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
    #                 torch.save({
    #                     'epoch': epoch,
    #                     'optimizer_state_dict': self.optimizer.state_dict(),
    #                     'model_state_dict': model_state_dic
    #                 }, save_path)
    #                 save_list.update(save_path)
    #
    #                 # 保存最佳模型（根据验证准确率）
    #                 if epoch_acc > best_acc or epoch > args.max_epoch - 2:
    #                     best_acc = epoch_acc
    #                     logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
    #                     torch.save(model_state_dic,
    #                                os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))
