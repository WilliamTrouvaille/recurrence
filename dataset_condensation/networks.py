#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/10/18 13:05
@version : 1.0.0
@author  : William_Trouvaille
@function: 模型定义
"""
# Acknowledgement to
# https://github.com/VICO-UoE/DatasetCondensation

import torch
from torch import nn
import torch.nn.functional as F


''' Swish activation '''
class Swish(nn.Module): # Swish(x) = x * sigmoid(x)
    def __init__(self):
        """
        初始化 Swish 激活函数模块。
        """
        super().__init__()

    def forward(self, input):
        """
        定义前向传播。
        参数:
            input: 输入张量
        返回:
            input * torch.sigmoid(input): 按照 Swish 定义计算的结果
        """
        return input * torch.sigmoid(input)

''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        """
        ConvNet 构造函数。
        参数:
            channel (int): 输入图像的通道数 (例如 MNIST=1, CIFAR10=3)
            num_classes (int): 分类的类别数
            net_width (int): 每个卷积层的滤波器（通道）数量
            net_depth (int): 卷积块的深度（数量）
            net_act (str): 激活函数的名称 (如 'relu')
            net_norm (str): 归一化层的名称 (如 'instancenorm')
            net_pooling (str): 池化层的名称 (如 'avgpooling')
            im_size (tuple): 输入图像的尺寸
        """
        super(ConvNet, self).__init__()

        # _make_layers 方法会动态构建特征提取网络 (self.features)
        # 并返回该网络以及输出特征图的形状 (shape_feat)
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)

        # 计算特征图展平后的总维度
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]

        # 定义最后的分类器层 (全连接层)
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        """
        定义模型的主前向传播（用于训练和推理）。
        参数:
            x: 输入的图像批次张量
        返回:
            out: 最终的分类 logits
        """
        # 1. 通过特征提取层
        out = self.features(x)
        # 2. 将特征图展平 (view(out.size(0), -1) 表示保持 batch_size，自动计算后续维度)
        out = out.view(out.size(0), -1)
        # 3. 通过分类器层
        out = self.classifier(out)
        return out

    def embed(self, x):
        """
        定义一个用于提取特征（嵌入）的前向传播方法。
        这在 Distribution Matching (DM) 等后续工作中会用到。
        参数:
            x: 输入的图像批次张量
        返回:
            out: 展平后的特征向量（分类器之前）
        """
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        """
        辅助函数：根据字符串名称返回激活函数模块。
        """
        if net_act == 'sigmoid':    # Sigmoid 激活函数
            return nn.Sigmoid()
        elif net_act == 'relu':     # ReLU 激活函数
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu': # LeakyReLU 激活函数
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':    # Swish 激活函数 (调用上面定义的 Swish 类)
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        """
        辅助函数：根据字符串名称返回池化层模块。
        """
        if net_pooling == 'maxpooling': # 最大池化
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling': # 平均池化
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':       # 不使用池化
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        """
        辅助函数：根据字符串名称和特征形状返回归一化层模块。
        参数:
            net_norm (str): 归一化名称
            shape_feat (list): 特征图的形状 [C, H, W]
        """
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':     # 批量归一化 (BN)
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':     # 层归一化 (LN)
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':  # 实例归一化 (IN)，论文默认使用
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True) # 用 GroupNorm 实现 IN
        elif net_norm == 'groupnorm':     # 组归一化 (GN)
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':          # 不使用归一化
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        """
        辅助函数：动态构建所有特征提取层。
        """
        layers = []
        in_channels = channel
        if im_size[0] == 28: # 对 MNIST (28x28) 进行特殊处理
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]] # 追踪特征图的形状 [C, H, W]

        # 循环构建 net_depth 个卷积块
        for d in range(net_depth):
            # 1. 添加卷积层 (kernel_size=3, padding=1 保持尺寸)
            # 特别注意：对 MNIST 且是第一个卷积层时，padding=3（为了适应32x32的处理）
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width # 更新 C

            # 2. 添加归一化层 (如果需要)
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]

            # 3. 添加激活函数
            layers += [self._get_activation(net_act)]

            # 4. 更新输入通道数 (下一层的输入是这一层的输出)
            in_channels = net_width

            # 5. 添加池化层 (如果需要)
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2 # 更新 H
                shape_feat[2] //= 2 # 更新 W

        return nn.Sequential(*layers), shape_feat


''' LeNet '''
class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        """
        LeNet 构造函数。
        参数:
            channel (int): 输入图像的通道数 (例如 MNIST=1)
            num_classes (int): 分类的类别数 (例如 10)
        """
        super(LeNet, self).__init__()

        # 'features' 模块定义了 LeNet 的卷积和池化部分
        self.features = nn.Sequential(
            # C1: 卷积层。输入通道=channel, 输出通道=6, 卷积核=5x5
            # padding=2 仅用于 MNIST (channel=1)，保持 (28+4-5)/1 + 1 = 28
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel==1 else 0),
            nn.ReLU(inplace=True),
            # S2: 池化层。最大池化, 2x2, 步长=2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # C3: 卷积层。输入通道=6, 输出通道=16, 卷积核=5x5
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            # S4: 池化层。最大池化, 2x2, 步长=2
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 'classifier' 部分 (论文中未显式命名，但逻辑如此)
        # F5: 全连接层。输入维度 16*5*5 (S4输出 16x5x5), 输出 120
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        # F6: 全连接层。输入 120, 输出 84
        self.fc_2 = nn.Linear(120, 84)
        # Output: 全连接层。输入 84, 输出 10 (num_classes)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        定义 LeNet 的前向传播。
        参数:
            x: 输入的图像批次张量 (例如 [batch_size, 1, 28, 28])
        """
        # 1. 通过 C1, S2, C3, S4 卷积和池化层
        x = self.features(x)

        # 2. 将特征图展平，为全连接层做准备
        # x 的形状此时为 [batch_size, 16, 5, 5]
        # x.view(x.size(0), -1) 将其变为 [batch_size, 16*5*5]
        x = x.view(x.size(0), -1)

        # 3. 通过 F5 (全连接层 + ReLU)
        x = F.relu(self.fc_1(x))

        # 4. 通过 F6 (全连接层 + ReLU)
        x = F.relu(self.fc_2(x))

        # 5. 通过 Output 层 (输出 logits，不激活)
        x = self.fc_3(x)
        return x
