#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/10/18 13:05
@version : 1.0.0
@author  : William_Trouvaille
@function: 工具类
"""
import os
import sys
import time
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.ndimage.interpolation import rotate as scipyrotate
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from networks import ConvNet, LeNet


# -----------------------------------------------------------------
# MARK: - 数据处理
# -----------------------------------------------------------------

def get_dataset(dataset, data_path):
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

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader


# torchvision.datasets 并没有像 MNIST 或 CIFAR10 那样内置对 TinyImageNet 的原生支持
# 此处自定义类，用于将内存中已有的张量数据转换成 PyTorch 的 Dataset 格式
class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


# -----------------------------------------------------------------
# MARK: - 模型与配置
# -----------------------------------------------------------------

# 返回 ConvNet 的默认 net_width, net_depth 等超参数。
def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


# 实例化模型
def get_network(model, channel, num_classes, im_size=(32, 32)):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)

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
        net = None
        exit('unknown model: %s' % model)

    gpu_num = torch.cuda.device_count()
    if gpu_num > 0:
        device = 'cuda'
        if gpu_num > 1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net


def get_loops(ipc):
    # 根据 ipc（每类图像数） 返回 outer_loop 和 inner_loop 这两个超参数。main.py 在启动时会调用它。
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
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc' % ipc)
    return outer_loop, inner_loop


# -----------------------------------------------------------------
# MARK: - 核心算法
# -----------------------------------------------------------------

# 这是最核心的算法实现。它实现了论文中的公式 (10)
# 即按输出节点（shape[0]）计算两个梯度张量（gwr, gws） 之间的余弦距离之和。
def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)
    # 论文提出的、按输出节点计算并加总的梯度余弦距离度量。它是梯度匹配算法的核心计算步骤。
    dis_weight = torch.sum(
        1 -
        torch.sum(gwr * gws, dim=-1) /
        (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
    )
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    # 均方误差（Mean Square Error,MSE）是回归损失函数中最常用的误差
    # 它是预测值f(x)与目标值y之间差值平方和的均值
    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s' % args.dis_metric)

    return dis


# -----------------------------------------------------------------
# MARK: - 训练与评估
# -----------------------------------------------------------------
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

    # 3. 遍历数据加载器中的所有批次
    for i_batch, datum in enumerate(dataloader):
        # 3a. 获取数据和标签，并移到设备
        img = datum[0].float().to(device)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]  # 当前批次大小

        # 3b. (可选) 数据增强
        # if aug:
        #     if args.dsa: # 忽略 DSA
        #         img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
        #     else: # DC 的传统增强
        #         img = augment(img, args.dc_aug_param, device=args.device)
        # (可选) 数据增强
        if aug:
            # 统一使用 augment 函数进行数据增强
            # 注意：这移除了对 args.dsa 和 DiffAugment 的依赖，
            #      仅适用于复现 DC (Gradient Matching) 论文。
            if dc_aug_param is None:
                # 如果没有提供增强参数，可以抛出错误或跳过增强
                logger.warning("Warning: aug is True but dc_aug_param is None. Skipping augmentation.")
            else:
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

    # 4. 计算整个 epoch 的平均损失和准确率
    loss_avg /= num_exp
    acc_avg /= num_exp

    # 5. 返回结果
    return loss_avg, acc_avg


def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:, c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1], shape[2] + crop * 2, shape[3] + crop * 2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop + shape[2], crop:crop + shape[3]] = images[i]
            r, c = np.random.permutation(crop * 2)[0], np.random.permutation(crop * 2)[0]
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1),
                              cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)

        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0]  # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch + 1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    logger.success('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (
        get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test


def get_eval_pool(eval_mode, model, model_eval):
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


# -----------------------------------------------------------------
# MARK: - 辅助函数
# -----------------------------------------------------------------

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


# --- Loguru 日志配置函数 ---

# 定义一个全局变量来跟踪日志是否已被配置
_logger_configured = False


def setup_logger(log_dir="logs", log_level="INFO"):
    """
    配置 loguru 日志记录器，使其输出到控制台和文件。
    这个函数应该只在程序入口处被调用一次。

    Args:
        log_dir (str): 保存日志文件的目录。默认为 "logs"。
        log_level (str): 输出的最低日志级别。默认为 "INFO"。
                      可选值: "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"。
    """
    global _logger_configured
    # 检查是否已经配置过，防止重复添加 handlers
    if _logger_configured:
        logger.warning("日志记录器已经被配置过，跳过重复配置。")
        return

    # 1. (可选但推荐) 移除默认的 loguru handler
    try:
        logger.remove(0)
    except ValueError:
        pass  # Handler 0 不存在，没关系

    # 2. 定义期望的日志格式
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # 3. 添加控制台 (stderr) 输出 sink
    logger.add(
        sys.stderr,  # 输出到标准错误流
        level=log_level.upper(),  # 控制台输出的最低级别 (确保是大写)
        format=log_format,  # 使用定义的格式
        colorize=True  # 在控制台中启用彩色输出
    )

    # 4. 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 5. 定义日志文件路径 (使用 rotation 来创建新文件)
    log_file_path = os.path.join(log_dir, "{time:YYYYMMDD_HHmmss}.log")

    # 6. 添加文件输出 sink
    logger.add(
        log_file_path,  # 文件路径模式
        level=log_level.upper(),  # 文件输出的最低级别 (确保是大写)
        format=log_format,  # 使用相同的定义格式
        rotation="10 MB",  # 当日志文件达到 10 MB 时进行轮转
        retention="10 days",  # 最多保留最近 10 天的日志文件
        encoding="utf-8",  # 指定文件编码
        enqueue=True,  # 启用异步日志记录
        backtrace=True,  # 显示完整的堆栈跟踪
        diagnose=True  # 提供更详细的错误诊断信息
    )

    _logger_configured = True  # 标记为已配置
    logger.info(f"Loguru 日志记录器配置完成。日志级别: {log_level.upper()}。将输出到控制台和 '{log_dir}' 目录下的文件。")
# --- 日志配置函数结束 ---

# --- 配置加载部分开始 ---
def load_config(config_path="config.yaml"):
    """
    加载 YAML 配置文件。

    Args:
        config_path (str): YAML 配置文件的路径。默认为当前目录下的 "config.yaml"。

    Returns:
        dict: 包含配置参数的字典。如果文件不存在或解析失败，则返回 None。
    """
    resolved_path = os.path.abspath(config_path) # 获取绝对路径，方便日志记录
    logger.info(f"尝试从 '{resolved_path}' 加载配置...")

    if not os.path.exists(resolved_path):
        logger.error(f"配置文件未找到: {resolved_path}")
        return None

    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) # 使用 safe_load 防止执行任意代码
        logger.success(f"成功加载配置文件: {resolved_path}")
        # 使用 DEBUG 级别选择性地打印加载的配置内容
        logger.debug(f"加载的配置内容: {config}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"解析 YAML 文件时出错: {resolved_path}\n错误详情: {e}")
        return None
    except Exception as e:
        logger.error(f"加载配置文件时发生未知错误: {resolved_path}\n错误详情: {e}")
        return None

class ConfigNamespace:
    def __init__(self, config_dict):
        # 将字典的键值对设置为对象的属性
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        # 定义对象的打印形式，方便日志记录
        return str(vars(self))

# --- 配置加载部分结束 ---