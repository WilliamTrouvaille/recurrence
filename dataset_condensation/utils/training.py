#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025/10/30
@version : 1.0.0
@author  : William_Trouvaille
@function: 训练与评估模块 - 负责模型训练、epoch循环和性能评估
"""

import time
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

# 导入自定义模块
from .data import TensorDataset


def epoch(mode, dataloader, net, optimizer, criterion, device, aug, dc_aug_param=None):
    """
    执行一个训练或评估 epoch。

    参数:
        mode (str): 'train' 或 'test'/'eval'
        dataloader (DataLoader): 数据加载器
        net (nn.Module): 神经网络模型
        optimizer (Optimizer): 优化器 (仅在 mode='train' 时需要)
        criterion (Loss): 损失函数
        device (torch.device or str): 目标设备 ('cuda', 'cpu')
        aug (bool): 是否应用数据增强
        dc_aug_param (dict, optional): augment 函数所需的参数字典。默认为 None。
    """
    # 1. 初始化损失、准确率和样本计数器
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(device)
    criterion = criterion.to(device)

    # 2. 设置模型模式 (训练 or 评估)
    if mode == 'train':
        net.train()  # 启用 Dropout, BatchNorm 更新等
    else:
        net.eval()  # 禁用 Dropout, 固定 BatchNorm 统计量

    pbar = tqdm(dataloader, desc=f'{mode.capitalize()}', leave=False, ncols=100)

    # 3. 遍历数据加载器中的所有批次
    for i_batch, datum in enumerate(dataloader):
        # 3a. 获取数据和标签，并移到设备
        img = datum[0].float().to(device)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]  # 当前批次大小

        # 3b. (可选) 数据增强
        if aug:
            # 统一使用 augment 函数进行数据增强
            # 注意：这移除了对 args.dsa 和 DiffAugment 的依赖，
            #      仅适用于复现 DC (Gradient Matching) 论文。
            if dc_aug_param is None:
                # 如果没有提供增强参数，可以抛出错误或跳过增强
                logger.warning("Warning: aug is True but dc_aug_param is None. Skipping augmentation.")
            else:
                from .augmentation import augment
                img = augment(img, dc_aug_param, device=device)  # augment 函数本身也接收 device 参数

        # 3c. 前向传播
        output = net(img)
        # 3d. 计算损失
        loss = criterion(output, lab)
        # 3e. 计算准确率 (这部分假设是分类任务)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        # 3f. 累加损失和准确率
        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        # 3g. (仅训练模式) 反向传播和优化器步骤
        if mode == 'train':
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

        # 更新进度条显示的信息
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc / n_b:.4f}'
        })

    # 4. 计算整个 epoch 的平均损失和准确率
    loss_avg /= num_exp
    acc_avg /= num_exp

    # 5. 返回结果
    return loss_avg, acc_avg


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    """
    使用合成数据集评估模型性能。

    参数:
        it_eval (int): 当前评估迭代次数
        net (nn.Module): 要评估的神经网络模型
        images_train (Tensor): 训练图像数据
        labels_train (Tensor): 训练标签数据
        testloader (DataLoader): 测试数据加载器
        args (ConfigNamespace): 配置参数

    返回:
        tuple: (net, acc_train, acc_test)
            - net: 训练后的模型
            - acc_train: 训练准确率
            - acc_test: 测试准确率
    """
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)

    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)  # 从 args 获取 epoch 数
    lr_schedule = [Epoch // 2 + 1]

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch + 1):
        loss_train, acc_train = epoch(
            'train', trainloader, net, optimizer, criterion,
            args.device,  # 传入正确的 device
            aug=True,  # 评估训练时启用增强
            dc_aug_param=args.dc_aug_param)  # 传入增强参数字典

        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start

    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion,
                                args.device,  # 传入正确的 device
                                aug=False,  # 测试时不使用增强
                                dc_aug_param=None)  # 测试时不使用增强参数

    logger.info(
        'Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (
            it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test)
    )

    return net, acc_train, acc_test


def get_optimizer(model, optimizer_name, learning_rate, epochs):
    """
    获取优化器实例。

    参数:
        model (nn.Module): 模型
        optimizer_name (str): 优化器名称
        learning_rate (float): 学习率
        epochs (int): 训练轮数

    返回:
        torch.optim.Optimizer: 优化器实例
    """
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=5e-4
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

    return optimizer


def eval_model(model, dataloader, device):
    """
    评估模型在给定数据加载器上的性能。

    参数:
        model (nn.Module): 要评估的模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 计算设备

    返回:
        float: 准确率百分比
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def get_eval_pool(eval_mode, model, model_eval):
    """
    获取评估模型池，支持多种评估模式。

    支持的评估模式：
        - 'M': 多架构模式 (Multiple architectures)
        - 'B': BatchNorm 模式 (用于 DM 实验)
        - 'W': 宽度消融研究 (Width ablation study)
        - 'D': 深度消融研究 (Depth ablation study)
        - 'A': 激活函数消融研究 (Activation function ablation study)
        - 'P': 池化层消融研究 (Pooling layer ablation study)
        - 'N': 归一化层消融研究 (Normalization layer ablation study)
        - 'S': 自身 (Self)
        - 'SS': 自身 (Self, 严格模式)

    参数:
        eval_mode (str): 评估模式
        model (str): 基础模型名称
        model_eval (str): 评估模型名称（当 eval_mode 不是 'S' 或 'SS' 时使用）

    返回:
        list: 评估模型名称列表
    """
    if eval_mode == 'M':  # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W':  # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D':  # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A':  # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P':  # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N':  # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S':  # itself
        if 'BN' in model:
            logger.warning(
                'Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


def train_model(model, trainloader, testloader, args):
    """
    完整的模型训练流程。

    参数:
        model (nn.Module): 要训练的模型
        trainloader (DataLoader): 训练数据加载器
        testloader (DataLoader): 测试数据加载器
        args (ConfigNamespace): 配置参数

    返回:
        tuple: (最佳模型, 训练历史)
    """
    model = model.to(args.device)

    # 优化器和损失函数
    optimizer = get_optimizer(model, args.optimizer, args.lr_net, args.epochs)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    train_history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 学习率调度
        scheduler.step()

        # 计算训练准确率
        train_acc = 100. * correct / total
        avg_loss = train_loss / len(trainloader)

        # 测试阶段
        test_acc = eval_model(model, testloader, args.device)

        # 记录历史
        train_history['train_loss'].append(avg_loss)
        train_history['train_acc'].append(train_acc)
        train_history['test_acc'].append(test_acc)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict()

        # 日志输出
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch + 1}/{args.epochs}] - "
                        f"Train Loss: {avg_loss:.4f}, "
                        f"Train Acc: {train_acc:.2f}%, "
                        f"Test Acc: {test_acc:.2f}%")

    # 恢复最佳参数
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    logger.info(f"训练完成！最佳测试准确率: {best_acc:.2f}%")

    return model, train_history
