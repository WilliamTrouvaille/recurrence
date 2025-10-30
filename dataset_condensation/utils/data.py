#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/10/30
@version : 1.0.0
@author  : William_Trouvaille
@function: 数据处理模块 - 负责数据集加载和自定义数据集类
"""
import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# -----------------------------------------------------------------
# MARK: - 数据集加载
# -----------------------------------------------------------------

def get_dataset(dataset, data_path):
    """
    加载指定的数据集并返回相关配置信息。

    支持的数据集：
        - MNIST: 手写数字识别数据集 (28x28 灰度图)
        - FashionMNIST: 时尚物品分类数据集 (28x28 灰度图)
        - SVHN: 街景门牌号数据集 (32x32 彩色图)
        - CIFAR10: 通用图像分类数据集 (32x32 彩色图, 10类)
        - CIFAR100: 通用图像分类数据集 (32x32 彩色图, 100类)
        - TinyImageNet: 小型 ImageNet 数据集 (64x64 彩色图, 200类)

    参数:
        dataset (str): 数据集名称
        data_path (str): 数据集存储路径

    返回:
        tuple: 包含以下元素
            - channel (int): 图像通道数 (灰度图为1, RGB为3)
            - im_size (tuple): 图像尺寸 (height, width)
            - num_classes (int): 类别数量
            - class_names (list): 类别名称列表
            - mean (list): 数据集的均值（用于归一化），长度为 channel
            - std (list): 数据集的标准差（用于归一化），长度为 channel
            - dst_train (Dataset): 训练集 Dataset 对象
            - dst_test (Dataset): 测试集 Dataset 对象
            - testloader (DataLoader): 测试集 DataLoader（batch_size=256）
    """
    if dataset == 'MNIST':
        # 1. 设置数据集特定参数
        channel = 1  # 通道数：MNIST 是灰度图，所以是 1
        im_size = (28, 28)  # 图像尺寸：28x28 像素
        num_classes = 10  # 类别数：0-9 共 10 个数字
        mean = [0.1307]  # MNIST 数据集在所有像素上的标准均值
        std = [0.3081]  # MNIST 数据集在所有像素上的标准差

        # 2. 定义数据变换
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将 PIL 图像或 numpy 数组转换为 FloatTensor，并把像素范围从 [0, 255] 缩放到 [0.0, 1.0]
            transforms.Normalize(mean=mean, std=std)
            # 用上面定义的均值和标准差对 Tensor 进行 Z-Score 归一化 (image = (image - mean) / std)
        ])

        # 3. 加载数据集
        # 从 torchvision.datasets 中加载 MNIST 训练集
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)  # 不包含随机裁剪、翻转等增强
        # 从 torchvision.datasets 中加载 MNIST 测试集
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        # 定义类名（用于日志或可视化）
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        # 1. 设置数据集特定参数
        channel = 3  # 通道数：CIFAR-10 是 RGB 彩色图，所以是 3
        im_size = (32, 32)  # 图像尺寸：32x32 像素
        num_classes = 10  # 类别数：10 个类别（飞机、汽车、鸟等）
        mean = [0.4914, 0.4822, 0.4465]  # CIFAR-10 在 R, G, B 三个通道上的标准均值
        std = [0.2023, 0.1994, 0.2010]  # CIFAR-10 在 R, G, B 三个通道上的标准差

        # 2. 定义数据变换 (同 MNIST)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # 3. 加载数据集
        # 从 torchvision.datasets 中加载 CIFAR-10 训练集
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)  # 同样，不包含数据增强
        # 从 torchvision.datasets 中加载 CIFAR-10 测试集
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        # 从数据集对象中直接获取类名
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:, c] = (images_train[:, c] - mean[c]) / std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation

    else:
        exit('unknown dataset: %s' % dataset)

    # 创建测试集 DataLoader
    # batch_size=256: 每批次处理 256 个样本
    # shuffle=False: 测试集不需要打乱顺序
    # num_workers=0: 不使用多进程加载数据（避免潜在的兼容性问题）
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader


# -----------------------------------------------------------------
# MARK: - 自定义数据集类
# -----------------------------------------------------------------

class TensorDataset(Dataset):
    """
    自定义的 PyTorch Dataset 类，用于将内存中的张量数据封装为 Dataset。

    注意：
        torchvision.datasets 并没有像 MNIST 或 CIFAR10 那样内置对 TinyImageNet 的原生支持，
        此处自定义类，用于将内存中已有的张量数据转换成 PyTorch 的 Dataset 格式。

    参数:
        images (Tensor): 图像张量，形状为 (N, C, H, W)
            - N: 样本数量
            - C: 通道数
            - H: 图像高度
            - W: 图像宽度
        labels (Tensor): 标签张量，形状为 (N,)
    """

    def __init__(self, images, labels):
        """
        初始化数据集。

        参数:
            images (Tensor): 图像张量，形状 (N, C, H, W)
            labels (Tensor): 标签张量，形状 (N,)
        """
        # detach() 用于从计算图中分离张量，避免梯度传播问题
        # float() 确保图像数据为浮点类型
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        """
        获取指定索引的样本。

        参数:
            index (int): 样本索引

        返回:
            tuple: (image, label) 图像张量和对应的标签
        """
        return self.images[index], self.labels[index]

    def __len__(self):
        """
        返回数据集中的样本数量。

        返回:
            int: 样本总数
        """
        return self.images.shape[0]
