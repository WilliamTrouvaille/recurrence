#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/10/30
@version : 1.0.0
@author  : William_Trouvaille
@function: 核心算法模块 - 实现梯度匹配（Gradient Matching）的核心算法

论文参考：Dataset Condensation with Gradient Matching
         https://arxiv.org/abs/2006.05929
"""
import torch


# -----------------------------------------------------------------
# MARK: - 梯度距离计算
# -----------------------------------------------------------------

def distance_wb(gwr, gws):
    """
    计算两个梯度张量之间的加权余弦距离（Weighted Cosine Distance）。

    这是论文中公式 (10) 的核心实现，用于衡量真实数据梯度和合成数据梯度的相似度。

    算法原理：
        1. 对于不同形状的梯度张量（卷积层、全连接层、归一化层等），
           先将其 reshape 为二维矩阵，其中第一维是输出节点数。
        2. 对每个输出节点，计算两个梯度向量的余弦相似度。
        3. 将所有输出节点的余弦距离（1 - 余弦相似度）求和。

    数学表达：
        对于每个输出节点 i：
            cos_sim_i = (gwr_i · gws_i) / (||gwr_i|| * ||gws_i||)
            cos_dist_i = 1 - cos_sim_i

        总距离 = Σ cos_dist_i

    参数:
        gwr (Tensor): 真实数据的梯度张量
            - 卷积层: shape = (out_channels, in_channels, kernel_h, kernel_w)
            - 全连接层: shape = (out_features, in_features)
            - 归一化层: shape = (num_features,) 或 (C, H, W)
        gws (Tensor): 合成数据的梯度张量，shape 必须与 gwr 相同

    返回:
        Tensor: 标量，表示两个梯度之间的总余弦距离（越小表示越相似）

    注意:
        - 对于 BatchNorm/InstanceNorm 的 bias 参数（shape=(C,)），函数返回 0，
          因为这些参数对梯度匹配的贡献较小。
        - 添加了 epsilon=0.000001 防止除零错误。
    """
    shape = gwr.shape

    # -----------------------------------------------------------------
    # 根据梯度张量的维度进行 reshape
    # -----------------------------------------------------------------

    if len(shape) == 4:
        # 卷积层梯度: (out_channels, in_channels, kernel_h, kernel_w)
        # reshape 为: (out_channels, in_channels * kernel_h * kernel_w)
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])

    elif len(shape) == 3:
        # LayerNorm 梯度: (C, H, W)
        # reshape 为: (C, H * W)
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])

    elif len(shape) == 2:
        # 全连接层梯度: (out_features, in_features)
        # 已经是二维，无需 reshape
        tmp = 'do nothing'

    elif len(shape) == 1:
        # BatchNorm/InstanceNorm/GroupNorm 的参数或 bias
        # shape = (num_features,)
        # 对于这些参数，梯度匹配的贡献较小，直接返回 0
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    # -----------------------------------------------------------------
    # 计算按输出节点加总的余弦距离
    # -----------------------------------------------------------------

    # 余弦相似度计算：
    # cos_sim = (gwr · gws) / (||gwr|| * ||gws||)
    #
    # 具体步骤：
    # 1. torch.sum(gwr * gws, dim=-1): 计算每个输出节点的内积
    # 2. torch.norm(gwr, dim=-1): 计算每个输出节点梯度的 L2 范数
    # 3. torch.norm(gws, dim=-1): 同上
    # 4. 1 - cos_sim: 余弦距离
    # 5. torch.sum(...): 对所有输出节点的余弦距离求和

    dis_weight = torch.sum(
        1 -
        torch.sum(gwr * gws, dim=-1) /
        (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)  # epsilon 防止除零
    )

    dis = dis_weight
    return dis


# -----------------------------------------------------------------
# MARK: - 梯度匹配损失
# -----------------------------------------------------------------

def match_loss(gw_syn, gw_real, args):
    """
    计算合成数据梯度和真实数据梯度之间的匹配损失。

    这是梯度匹配算法的核心损失函数，用于优化合成数据集。
    目标是让合成数据产生的梯度尽可能接近真实数据的梯度。

    支持三种距离度量方式：
        1. 'ours': 论文提出的方法，使用 distance_wb 按输出节点计算余弦距离
        2. 'mse': 均方误差（Mean Square Error），直接计算梯度向量的 L2 距离
        3. 'cos': 全局余弦距离，将所有参数的梯度展平后计算余弦距离

    参数:
        gw_syn (list of Tensor): 合成数据产生的梯度列表
            - 列表长度 = 网络参数数量
            - 每个元素是一个参数的梯度张量
        gw_real (list of Tensor): 真实数据产生的梯度列表
            - 结构与 gw_syn 相同
        args (ConfigNamespace): 配置对象，必须包含以下属性：
            - args.dis_metric (str): 距离度量方式 ('ours', 'mse', 'cos')
            - args.device (str): 计算设备 ('cuda' or 'cpu')

    返回:
        Tensor: 标量损失值，表示两个梯度集合的总距离

    异常:
        如果 args.dis_metric 不是 'ours', 'mse', 'cos' 之一，程序会退出
    """
    # 初始化损失为 0
    dis = torch.tensor(0.0).to(args.device)

    # -----------------------------------------------------------------
    # 方法 1: 'ours' - 论文提出的方法（推荐）
    # -----------------------------------------------------------------
    if args.dis_metric == 'ours':
        # 遍历所有参数的梯度
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]  # 真实数据梯度
            gws = gw_syn[ig]  # 合成数据梯度
            # 累加每个参数的梯度距离
            dis += distance_wb(gwr, gws)

    # -----------------------------------------------------------------
    # 方法 2: 'mse' - 均方误差
    # -----------------------------------------------------------------
    elif args.dis_metric == 'mse':
        # 均方误差（Mean Square Error, MSE）是回归损失函数中最常用的误差度量
        # 它计算预测值 f(x) 与目标值 y 之间差值平方和的均值

        gw_real_vec = []
        gw_syn_vec = []

        # 将所有参数的梯度展平并拼接成一个长向量
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))  # 展平为一维
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))

        # 拼接所有参数的梯度
        gw_real_vec = torch.cat(gw_real_vec, dim=0)  # shape: (total_params,)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)

        # 计算 L2 距离的平方和
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    # -----------------------------------------------------------------
    # 方法 3: 'cos' - 全局余弦距离
    # -----------------------------------------------------------------
    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []

        # 同样将所有参数的梯度展平并拼接
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))

        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)

        # 计算全局余弦距离（不按输出节点分解）
        # cos_dist = 1 - cos_sim
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001
        )

    else:
        # 未知的距离度量方式
        exit('unknown distance function: %s' % args.dis_metric)

    return dis
