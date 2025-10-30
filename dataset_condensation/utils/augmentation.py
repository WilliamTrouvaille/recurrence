#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/10/30
@author  : William_Trouvaille
@function: 数据增强模块 - 负责图像数据的增强操作
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.interpolation import rotate as scipyrotate


# -----------------------------------------------------------------
# MARK: - 数据增强函数
# -----------------------------------------------------------------

def augment(images, dc_aug_param, device):
    """
    对图像批次应用数据增强操作。

    支持的增强策略：
        - crop: 随机裁剪
        - scale: 随机缩放
        - rotate: 随机旋转
        - noise: 添加随机噪声

    增强策略可以组合使用，例如 'crop_scale_rotate' 会随机选择其中一种增强方式。

    参数:
        images (Tensor): 图像张量，shape = (N, C, H, W)
            - N: batch size
            - C: 通道数
            - H: 图像高度
            - W: 图像宽度
        dc_aug_param (dict or None): 数据增强参数字典，包含以下键：
            - 'strategy' (str): 增强策略，多个策略用下划线连接，如 'crop_scale_rotate'
                               'none' 表示不进行增强
            - 'crop' (int): 裁剪时的填充像素数
            - 'scale' (float): 缩放范围，如 0.2 表示缩放到 [0.8, 1.2] 倍
            - 'rotate' (int): 旋转角度范围，如 45 表示 [-45°, 45°]
            - 'noise' (float): 噪声强度
        device (str or torch.device): 计算设备 ('cuda' or 'cpu')

    返回:
        Tensor: 增强后的图像张量，shape 与输入相同

    注意:
        - 此函数会直接修改输入的 images 张量
        - 每个图像从策略列表中随机选择一种增强方式应用
        - 如果 dc_aug_param 为 None 或 strategy 为 'none'，则不进行增强
    """

    # 如果没有提供增强参数或策略为 'none'，直接返回原图像
    if dc_aug_param is None or dc_aug_param['strategy'] == 'none':
        return images

    # -----------------------------------------------------------------
    # 提取增强参数
    # -----------------------------------------------------------------

    scale = dc_aug_param['scale']  # 缩放范围
    crop = dc_aug_param['crop']  # 裁剪填充
    rotate = dc_aug_param['rotate']  # 旋转角度范围
    noise = dc_aug_param['noise']  # 噪声强度
    strategy = dc_aug_param['strategy']  # 增强策略

    # -----------------------------------------------------------------
    # 计算图像的均值（用于填充背景）
    # -----------------------------------------------------------------

    shape = images.shape  # (N, C, H, W)
    mean = []
    for c in range(shape[1]):  # 对每个通道计算均值
        mean.append(float(torch.mean(images[:, c])))

    # -----------------------------------------------------------------
    # 定义各种增强操作的闭包函数
    # -----------------------------------------------------------------

    def cropfun(i):
        """
        随机裁剪函数。

        步骤:
            1. 创建一个更大的画布（周围填充 crop 个像素），用均值填充
            2. 将原图像放在画布中央
            3. 随机选择一个 (H, W) 大小的区域进行裁剪

        参数:
            i (int): 图像索引
        """
        # 创建扩展画布，尺寸 = (C, H + 2*crop, W + 2*crop)
        im_ = torch.zeros(shape[1], shape[2] + crop * 2, shape[3] + crop * 2, dtype=torch.float, device=device)

        # 用各通道的均值填充画布
        for c in range(shape[1]):
            im_[c] = mean[c]

        # 将原图像放在画布中央
        im_[:, crop:crop + shape[2], crop:crop + shape[3]] = images[i]

        # 随机选择裁剪起点
        r, c = np.random.permutation(crop * 2)[0], np.random.permutation(crop * 2)[0]

        # 裁剪出 (H, W) 大小的区域
        images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

    def scalefun(i):
        """
        随机缩放函数。

        步骤:
            1. 随机生成缩放后的高度和宽度
            2. 使用插值将图像缩放到新尺寸
            3. 创建一个更大的画布，将缩放后的图像放在中央
            4. 从中央裁剪出原始尺寸

        参数:
            i (int): 图像索引
        """
        # 随机生成缩放因子，范围 [1-scale, 1+scale]
        h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
        w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])

        # 使用双线性插值进行缩放
        tmp = F.interpolate(images[i:i + 1], [h, w], )[0]

        # 创建一个足够大的画布
        mhw = max(h, w, shape[2], shape[3])
        im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)

        # 将缩放后的图像放在画布中央
        r = int((mhw - h) / 2)
        c = int((mhw - w) / 2)
        im_[:, r:r + h, c:c + w] = tmp

        # 从中央裁剪出原始尺寸
        r = int((mhw - shape[2]) / 2)
        c = int((mhw - shape[3]) / 2)
        images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

    def rotatefun(i):
        """
        随机旋转函数。

        步骤:
            1. 将图像转移到 CPU（scipy.ndimage 不支持 GPU）
            2. 随机生成旋转角度
            3. 使用 scipy 的 rotate 函数进行旋转
            4. 裁剪出中央区域以保持原始尺寸

        参数:
            i (int): 图像索引
        """
        # 随机生成旋转角度，范围 [-rotate, rotate]
        angle = np.random.randint(-rotate, rotate)

        # 使用 scipy 进行旋转，填充值为各通道均值的平均
        im_ = scipyrotate(
            images[i].cpu().data.numpy(),  # 转为 numpy 数组
            angle=angle,  # 旋转角度
            axes=(-2, -1),  # 在 H 和 W 维度上旋转
            cval=np.mean(mean)  # 填充值
        )

        # 旋转后的图像可能比原图大，裁剪出中央区域
        r = int((im_.shape[-2] - shape[-2]) / 2)
        c = int((im_.shape[-1] - shape[-1]) / 2)

        # 转回 Tensor 并移到设备
        images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

    def noisefun(i):
        """
        添加随机噪声函数。

        步骤:
            1. 生成与图像相同尺寸的高斯噪声
            2. 将噪声乘以强度系数后加到图像上

        参数:
            i (int): 图像索引
        """
        # 生成标准正态分布噪声，shape = (C, H, W)
        # 乘以 noise 强度系数后加到图像上
        images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)

    # -----------------------------------------------------------------
    # 应用增强策略
    # -----------------------------------------------------------------

    # 将策略字符串拆分成列表（例如 'crop_scale_rotate' -> ['crop', 'scale', 'rotate']）
    augs = strategy.split('_')

    # 对批次中的每个图像应用随机选择的增强操作
    for i in range(shape[0]):
        # 从增强策略列表中随机选择一种
        choice = np.random.permutation(augs)[0]

        # 根据选择应用相应的增强函数
        if choice == 'crop':
            cropfun(i)
        elif choice == 'scale':
            scalefun(i)
        elif choice == 'rotate':
            rotatefun(i)
        elif choice == 'noise':
            noisefun(i)

    return images


# -----------------------------------------------------------------
# MARK: - 增强参数配置
# -----------------------------------------------------------------

def get_daparam(dataset, model, model_eval, ipc):
    """
    根据数据集和模型类型返回合适的数据增强参数。

    注意：
        论文发现数据增强并不总是有益于性能，因此针对不同的设置使用不同的增强策略。

    参数:
        dataset (str): 数据集名称（如 'MNIST', 'CIFAR10'）
        model (str): 训练模型名称
        model_eval (str): 评估模型名称
        ipc (int): Images Per Class（每类图像数）

    返回:
        dict: 数据增强参数字典，包含以下键：
            - 'crop' (int): 裁剪填充像素数，默认 4
            - 'scale' (float): 缩放范围，默认 0.2
            - 'rotate' (int): 旋转角度范围，默认 45
            - 'noise' (float): 噪声强度，默认 0.001
            - 'strategy' (str): 增强策略，默认 'none'
    """
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
