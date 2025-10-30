#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/10/30
@version : 1.0.0
@author  : William_Trouvaille
@function: 模型配置模块 - 负责网络实例化和相关配置
"""
import time

import torch
import torch.nn as nn

# 从上级目录导入网络定义
# 注意: 此处假设 networks.py 在 dataset_condensation 目录下
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import ConvNet, LeNet


# -----------------------------------------------------------------
# MARK: - 默认配置
# -----------------------------------------------------------------

def get_default_convnet_setting():
    """
    返回 ConvNet 的默认超参数配置。

    这些超参数定义了卷积神经网络的基本架构：
        - net_width: 卷积层的通道数（特征图宽度）
        - net_depth: 网络深度（卷积层数量）
        - net_act: 激活函数类型
        - net_norm: 归一化层类型
        - net_pooling: 池化层类型

    返回:
        tuple: (net_width, net_depth, net_act, net_norm, net_pooling)
            - net_width (int): 卷积层通道数，默认 128
            - net_depth (int): 网络深度，默认 3 层
            - net_act (str): 激活函数，默认 'relu'
            - net_norm (str): 归一化方法，默认 'instancenorm'
            - net_pooling (str): 池化方法，默认 'avgpooling'
    """
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


# -----------------------------------------------------------------
# MARK: - 网络实例化
# -----------------------------------------------------------------

def get_network(model, channel, num_classes, im_size=(32, 32)):
    """
    根据模型名称实例化神经网络，并自动配置设备（CPU/GPU）。

    支持的模型类型：
        1. 基础模型:
            - ConvNet: 标准卷积网络
            - LeNet: 经典的 LeNet 网络

        2. 深度变体 (ConvNetD*):
            - ConvNetD1, ConvNetD2, ConvNetD3, ConvNetD4: 不同深度的 ConvNet

        3. 宽度变体 (ConvNetW*):
            - ConvNetW32, ConvNetW64, ConvNetW128, ConvNetW256: 不同通道数的 ConvNet

        4. 激活函数变体 (ConvNetA*):
            - ConvNetAS: Sigmoid 激活
            - ConvNetAR: ReLU 激活
            - ConvNetAL: LeakyReLU 激活
            - ConvNetASwish: Swish 激活
            - ConvNetASwishBN: Swish + BatchNorm

        5. 归一化变体 (ConvNet**):
            - ConvNetNN: 无归一化
            - ConvNetBN: BatchNorm
            - ConvNetLN: LayerNorm
            - ConvNetIN: InstanceNorm
            - ConvNetGN: GroupNorm

        6. 池化变体 (ConvNet*P):
            - ConvNetNP: 无池化
            - ConvNetMP: MaxPooling
            - ConvNetAP: AvgPooling

    参数:
        model (str): 模型名称（见上方支持的模型类型）
        channel (int): 输入图像的通道数 (1 for 灰度, 3 for RGB)
        num_classes (int): 分类任务的类别数量
        im_size (tuple): 输入图像尺寸 (height, width)，默认 (32, 32)

    返回:
        nn.Module: 实例化的神经网络模型，已移动到合适的设备 (CPU/CUDA)
                   如果有多个 GPU，会自动包装为 DataParallel
    """
    # 设置随机种子以增加模型初始化的多样性
    # 使用当前时间的毫秒数取模，避免每次实例化都使用相同的初始权重
    torch.random.manual_seed(int(time.time() * 1000) % 100000)

    # 获取默认的 ConvNet 配置
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    # -----------------------------------------------------------------
    # 根据模型名称实例化网络
    # -----------------------------------------------------------------

    if model == 'ConvNet':
        # 标准 ConvNet，使用默认配置
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        # 经典 LeNet 网络
        net = LeNet(channel=channel, num_classes=num_classes)

    # --- 深度变体 (Depth Variants) ---
    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    # --- 宽度变体 (Width Variants) ---
    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    # --- 激活函数变体 (Activation Variants) ---
    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwish':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='swish', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwishBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='swish', net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)

    # --- 归一化变体 (Normalization Variants) ---
    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=im_size)

    # --- 池化变体 (Pooling Variants) ---
    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=im_size)
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=im_size)
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)

    else:
        # 未知模型类型，退出程序
        net = None
        exit('unknown model: %s' % model)

    # -----------------------------------------------------------------
    # 设备配置：自动选择 GPU 或 CPU
    # -----------------------------------------------------------------

    gpu_num = torch.cuda.device_count()
    if gpu_num > 0:
        device = 'cuda'
        if gpu_num > 1:
            # 如果有多个 GPU，使用 DataParallel 进行并行计算
            net = nn.DataParallel(net)
    else:
        device = 'cpu'

    # 将模型移动到目标设备
    net = net.to(device)

    return net


# -----------------------------------------------------------------
# MARK: - 循环参数配置
# -----------------------------------------------------------------

def get_loops(ipc):
    """
    根据 IPC (Images Per Class, 每类图像数) 返回训练的 outer_loop 和 inner_loop 超参数。

    这些参数控制梯度匹配算法的迭代次数：
        - outer_loop: 外层循环次数，通常对应合成图像的更新轮数
        - inner_loop: 内层循环次数，通常对应每轮中模型参数的更新次数

    说明：
        不同的 IPC 需要不同的迭代策略。IPC 越大，outer_loop 越大，inner_loop 相对减小，
        以平衡计算效率和优化效果。

    参数:
        ipc (int): 每类的合成图像数量

    返回:
        tuple: (outer_loop, inner_loop)
            - outer_loop (int): 外层循环次数
            - inner_loop (int): 内层循环次数

    异常:
        如果 ipc 不在预定义范围内，程序会退出并提示错误
    """
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        # 未定义的 IPC 值
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc' % ipc)

    return outer_loop, inner_loop
