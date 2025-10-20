#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/10/18 13:05
@version : 1.0.0
@author  : William_Trouvaille
@function: 主要训练流程
"""
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from utils import (
    load_config, setup_logger, get_loops, get_dataset,
    get_eval_pool, get_time, ConfigNamespace, get_network, evaluate_synset, match_loss, epoch, TensorDataset,
    load_checkpoint, save_checkpoint
)
from loguru import logger

def main():
    # ==========================================================
    # 第 1 段：导入与参数解析 (Imports and Argument Parsing)
    # ==========================================================

    # === 1. 配置日志记录器 ===
    # 在程序最开始处调用日志配置函数
    setup_logger(log_dir="logs", log_level="INFO")
    # setup_logger(log_dir="logs", log_level="DEBUG")

    # === 2. 使用 argparse 仅解析配置文件路径 ===
    parser = argparse.ArgumentParser(description='数据集压缩实验 (使用配置文件)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件的路径 (YAML 格式)')
    # 只解析 --config 参数
    cmd_args, unknown = parser.parse_known_args()

    # === 3. 加载配置文件 ===
    config_dict = load_config(cmd_args.config)
    # logger.info(f"配置文件中的配置: {config_dict}")
    if config_dict is None:
        logger.error("无法加载配置，程序退出。")
        return

    # === 4. 将配置字典转换为 Namespace 对象并设置派生参数 ===
    try:
        # 使用 ConfigNamespace 将字典转换为对象 (假设 ConfigNamespace 在 utils.py 中定义)
        args = ConfigNamespace(config_dict)

        # 确保 method 存在
        if not hasattr(args, 'method') or args.method != 'DC':
            logger.warning(f"配置文件中 method 不是 'DC' 或未定义，强制设为 'DC'。")
            args.method = 'DC'
        args.dsa = False  # 禁用 DSA

        # 从配置的 ipc 计算循环次数
        args.outer_loop, args.inner_loop = get_loops(args.ipc)

        # 自动检测设备 ('cuda' 或 'cpu')
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"检测到设备: {args.device}")

        # --------------------------- 处理数据增强参数 ---------------------------
        # 读取 DC (不可微) 增强参数 (从配置文件读取)
        default_aug_params = {'strategy': 'none', 'crop': 0, 'scale': 0, 'rotate': 0, 'noise': 0}
        args.dc_aug_param = config_dict.get('dc_augmentation', default_aug_params)
        logger.info(f"DC 增强参数 (用于评估): {args.dc_aug_param}")
        # ----------------------------------------------------------------------

        # 检查必要的路径是否存在
        args.data_path = args.data_path if hasattr(args, 'data_path') else 'data'
        args.save_path = args.save_path if hasattr(args, 'save_path') else 'result'
        os.makedirs(args.data_path, exist_ok=True)  #
        os.makedirs(args.save_path, exist_ok=True)  #

        logger.info(f"最终使用的配置: {args}")

    except AttributeError as e:
        # 捕获因配置文件缺少必要键而导致的属性访问错误
        logger.error(f"配置文件 '{cmd_args.config}' 中缺少必要的参数: {e}")
        return  # 退出程序
    except KeyError as e:
        # 捕获其他可能的字典键错误 (例如 get_loops 需要 ipc)
        logger.error(f"配置文件 '{cmd_args.config}' 中缺少参数或结构错误: {e}")
        return
    except Exception as e:
        # 捕获其他未知错误
        logger.error(f"处理配置时发生错误: {e}")
        return

    # ==========================================================
    # 第 2 段：数据集加载 (Dataset Loading)
    # ==========================================================

    logger.info("开始加载数据集...")

    # 调用 utils 中的 get_dataset 函数加载数据
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,  args.data_path)
    logger.info(f"数据集 {args.dataset} 加载完成。通道: {channel}, 尺寸: {im_size}, 类别数: {num_classes}")

    # 定义评估迭代点列表
    if args.eval_mode == 'S' or args.eval_mode == 'SS':
        # 每 500 次迭代评估一次
        eval_it_pool = np.arange(0, args.Iteration + 1, 500).tolist()
    else:
        # 其他模式（如 'M'）只在最后一次迭代评估
        eval_it_pool = [args.Iteration]
    logger.info(f"将在以下迭代次数进行评估: {eval_it_pool}")

    # 获取需要评估的模型名称列表
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    logger.info(f"评估模式: {args.eval_mode}, 评估模型池: {model_eval_pool}")

    # ====================================================================
    # 第 3 段：实验与结果存储初始化 (Experiment and Result Storage Initialization)
    # ====================================================================

    # 初始化用于存储所有实验结果的字典
    accs_all_exps = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = []

    # ====================================================================
    # 第 4 段：主实验循环 (Main Experiment Loop)
    # ====================================================================

    # 初始化用于保存最终合成数据的列表
    data_save = []
    for exp in range(args.num_exp):  # 外循环，进行 num_exp 次独立实验

        # ==========================================================
        # 第 4.1 段：单次实验设置与数据初始化
        # ==========================================================

        logger.info(f"\n================== 开始实验 {exp} / {args.num_exp - 1} ==================\n ")
        logger.info(f"当前实验使用的超参数: \n {args}")
        logger.info(f"评估模型池: {model_eval_pool}")

        ''' organize the real dataset (整理真实数据集) '''
        # 将整个训练数据集加载到内存/显存中，方便快速随机采样
        images_all = []
        labels_all = []
        # 创建一个列表，长度为类别数，每个元素是一个列表，用于存储该类别下所有样本在 images_all 中的索引
        indices_class = [[] for c in range(num_classes)]

        # 遍历原始训练数据集 dst_train
        # 将图像 Tensor 增加一个维度 (unsqueeze) 并添加到列表
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        # 将标签添加到列表
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        # 再次遍历标签列表，填充 indices_class
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)  # 将样本索引 i 添加到其对应类别 lab 的列表中
        # 将图像列表拼接成一个大的 Tensor (N, C, H, W)，并移动到指定设备 (GPU 或 CPU)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        # 将标签列表也转换为 Tensor
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        # 记录每个类别的真实图像数量
        for c in range(num_classes):
            logger.info(f'类别 c = {c}: 包含 {len(indices_class[c])} 张真实图像')

        # 定义一个辅助函数，用于从指定类别 c 中随机采样 n 张图像
        def get_images(c, n):  # get random n images from class c
            # 对该类别的索引进行随机排列，并取前 n 个
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            # 从 images_all 中取出对应索引的图像
            return images_all[idx_shuffle]

        # （调试信息）打印真实图像每个通道的均值和标准差
        for ch in range(channel):
            logger.debug(
                f'真实图像通道 {ch}, mean = {torch.mean(images_all[:, ch]):.4f},'
                f' std = {torch.std(images_all[:, ch]):.4f}'
            )

        # 尝试加载检查点
        checkpoint_path_dir = args.checkpoint_path if hasattr(args, 'checkpoint_path') else 'checkpoints'
        checkpoint = load_checkpoint(checkpoint_path_dir, args)

        if checkpoint is not None and checkpoint['exp'] == exp:
            # 恢复训练状态
            logger.info(f"从检查点恢复训练: 实验 {checkpoint['exp']}, 迭代 {checkpoint['iteration']}")
            start_iteration = checkpoint['iteration'] + 1
            image_syn = checkpoint['image_syn'].to(args.device).requires_grad_(True)
            label_syn = checkpoint['label_syn'].to(args.device)
            accs_all_exps = checkpoint['accs_all_exps']
            data_save = checkpoint['data_save']
        else:
            # 从头开始训练
            start_iteration = 0

        ''' initialize the synthetic data (初始化合成数据) '''
        if checkpoint is None or checkpoint['exp'] != exp:# 仅在没有检查点或实验不匹配时初始化

            # 创建合成图像张量
            # 大小为 (类别数 * 每类图像数, 通道数, 高, 宽)
            # 初始化为标准正态分布的随机噪声
            # dtype=torch.float 指定数据类型
            # requires_grad=True 表明这个张量是需要计算梯度的，是我们要优化的目标
            # device=args.device 指定存储在哪个设备上
            image_syn = torch.randn(
                size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
                dtype=torch.float,
                requires_grad=True,
                device=args.device
            )

            # 创建合成图像对应的标签张量
            # 例如 ipc=10, num_classes=10 -> 生成 [[0,0,...,0], [1,1,...,1], ..., [9,9,...,9]]
            # 先创建一个 NumPy 数组的列表
            label_list_np = [np.ones(args.ipc) * i for i in range(num_classes)]
            # 将 NumPy 数组列表 转换为 单一的 NumPy 数组
            label_array_np = np.array(label_list_np, dtype=np.int64) # 指定 NumPy 的数据类型为整数
            # 从 单一的 NumPy 数组 创建 Tensor
            label_syn = torch.from_numpy(label_array_np).to(dtype=torch.long, device=args.device) # 使用 from_numpy 更高效，并指定最终类型和设备
            # .view(-1) 将形状从 [num_classes, ipc] 展平成 [num_classes * ipc]
            label_syn = label_syn.view(-1)
            # 确保标签不需要梯度
            label_syn.requires_grad_(False)

            if args.init == 'real':
                logger.info('从随机抽取的真实图像初始化合成数据')
                for c in range(num_classes):
                    start_idx = c * args.ipc
                    end_idx = (c + 1) * args.ipc
                    real_images_subset = get_images(c, args.ipc)
                    image_syn.data[start_idx:end_idx] = real_images_subset.detach().data
            else:
                logger.info('从随机高斯噪声初始化合成数据')


        # 创建合成图像对应的标签张量
        # 例如 ipc=3, num_classes=10 -> 生成 [[0,0,0], [1,1,1], ..., [9,9,9]]
        # dtype=torch.long 指定标签类型为长整型
        # requires_grad=False 表明标签是固定的，不需要优化
        # .view(-1) 将形状从 [num_classes, ipc] 展平成 [num_classes * ipc]
        # label_syn = torch.tensor(
        #     [np.ones(args.ipc) * i for i in range(num_classes)],
        #     dtype=torch.long,
        #     requires_grad=False,
        #     device=args.device
        # ).view(-1)

        # # 创建合成图像对应的标签张量
        # # 例如 ipc=10, num_classes=10 -> 生成 [[0,0,...,0], [1,1,...,1], ..., [9,9,...,9]]
        # # 先创建一个 NumPy 数组的列表
        # label_list_np = [np.ones(args.ipc) * i for i in range(num_classes)]
        # # 将 NumPy 数组列表 转换为 单一的 NumPy 数组
        # label_array_np = np.array(label_list_np, dtype=np.int64) # 指定 NumPy 的数据类型为整数
        # # 从 单一的 NumPy 数组 创建 Tensor
        # label_syn = torch.from_numpy(label_array_np).to(dtype=torch.long, device=args.device) # 使用 from_numpy 更高效，并指定最终类型和设备
        # # .view(-1) 将形状从 [num_classes, ipc] 展平成 [num_classes * ipc]
        # label_syn = label_syn.view(-1)
        # # 确保标签不需要梯度
        # label_syn.requires_grad_(False)

        # # 根据配置决定初始化方式 ('noise' 或 'real')
        # if args.init == 'real':
        #     logger.info('从随机抽取的真实图像初始化合成数据')
        #     # 遍历每个类别
        #     for c in range(num_classes):
        #         # 获取该类别在 image_syn 中的切片范围
        #         start_idx = c * args.ipc
        #         end_idx = (c + 1) * args.ipc
        #         # 从真实数据中随机抽取 ipc 张该类别的图像
        #         real_images_subset = get_images(c, args.ipc)
        #         # 将抽取的真实图像数据 (去除梯度信息 .detach()) 复制到 image_syn 对应位置
        #         image_syn.data[start_idx:end_idx] = real_images_subset.detach().data
        # else:  # 默认 'noise'
        #     logger.info('从随机高斯噪声初始化合成数据')

        ''' training setup (训练设置) '''
        # 定义用于优化合成图像 image_syn 的优化器
        # 使用带动量的 SGD，学习率由配置指定 (args.lr_img)
        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()  # 初始化时清空梯度

        # 定义用于训练网络（无论是蒸馏过程中的临时网络还是评估时的网络）的损失函数
        # 使用标准的交叉熵损失，并将其移动到指定设备
        criterion = nn.CrossEntropyLoss().to(args.device)

        # 使用 logger 记录蒸馏训练开始的时间戳
        logger.info(f'{get_time()} 蒸馏训练开始...')

        # ==========================================================
        # 第 4.2 段：迭代蒸馏与评估
        # ==========================================================

        # 主蒸馏循环，对应 Algorithm 1 的 t 循环
        for it in range(start_iteration,args.Iteration + 1): # 迭代次数由配置文件中的 training.iterations 定义

            # --- 4.2.1: 定期评估合成数据质量 ---
            # 检查当前迭代次数 it 是否在预定义的评估点列表 eval_it_pool 中
            if it in eval_it_pool:
                logger.info(f"\n===== 开始评估: 迭代 {it} / {args.Iteration} =====")
                # 遍历评估模型池中的每个模型名称
                for model_eval in model_eval_pool:
                    logger.info(f"--- 评估模型: {model_eval} ---")
                    # logger.info(f"蒸馏模型={args.model}, 评估模型={model_eval}, 迭代={it}") # 原打印信息

                    # 记录将使用的 DC 增强参数 (从配置文件加载)
                    logger.info(f"评估时使用的 DC 增强参数: {args.dc_aug_param}")

                    # # 根据是否使用增强，决定评估时的训练周期数
                    # # 这个逻辑可以保留，或者也将 epoch_eval_train 完全由配置文件控制
                    # current_eval_epochs = args.epoch_eval_train # 从配置读取基础值
                    #
                    # if args.dc_aug_param['strategy'] != 'none':
                    #     # 如果启用了增强，原代码可能会增加训练周期，这里我们只记录日志
                    #     logger.info(f"数据增强已启用，将使用 {current_eval_epochs} 个评估周期进行训练。")
                    #     # 可以取消下面这行的注释，如果想保持原代码增加周期的逻辑
                    #     # current_eval_epochs = 1000 # 例如固定为1000
                    # else: #
                    #     logger.info(f"数据增强未启用，将使用 {current_eval_epochs} 个评估周期进行训练。")
                    #     # 可以取消下面这行的注释，如果想保持原代码减少周期的逻辑
                    #     # current_eval_epochs = 300 # 例如固定为300

                    # 运行 num_eval 次独立的模型训练和评估
                    accs = []
                    # 这里的 num_eval 来自配置文件
                    for it_eval in range(args.num_eval):
                        # 获取一个随机初始化的评估网络实例
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                        # 深度拷贝当前的合成数据和标签，以防 evaluate_synset 内部意外修改
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())

                        # 调用 evaluate_synset 进行训练和测试
                        # 注意：需要确保 evaluate_synset 也被修改为接收独立的 device 参数，并正确使用 args.dc_aug_param
                        # 同时，需要传入 current_eval_epochs 替代 args.epoch_eval_train (如果需要动态调整)
                        # _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, current_eval_epochs) # 假设 evaluate_synset 接受 epoch 数
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args) # 保持原调用方式，依赖 args.epoch_eval_train

                        accs.append(acc_test)

                    # 记录本次评估结果的均值和标准差
                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs)
                    logger.info(f"评估 {args.num_eval} 个随机 {model_eval} 模型完成, 平均测试准确率 = {mean_acc:.4f} ± {std_acc:.4f}")

                    # 如果是最后一次迭代 (it == args.Iteration)，将结果存入全局字典 accs_all_exps
                    if it == args.Iteration:
                        accs_all_exps[model_eval] += accs
                        logger.debug(f"已记录模型 {model_eval} 的最终评估结果。")

                # --- 可视化与保存当前合成图像 ---
                # 构建保存文件名
                save_name = os.path.join(args.save_path, f'vis_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_exp{exp}_iter{it}.png')
                # 深度拷贝合成图像并移到 CPU
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                # 反归一化以便于人眼观察
                for ch_vis in range(channel):
                    # 确保 mean 和 std 是列表或 Tensor
                    image_syn_vis[:, ch_vis] = image_syn_vis[:, ch_vis] * std[ch_vis] + mean[ch_vis]
                # 将像素值裁剪到 [0, 1] 范围，防止因浮点误差超出范围
                image_syn_vis.clamp_(0, 1)
                # 使用 torchvision.utils.save_image 保存图像网格
                # nrow=args.ipc 表示每行显示 ipc 张图像
                save_image(image_syn_vis, save_name, nrow=args.ipc)
                logger.info(f"当前合成图像样本已可视化保存至: {save_name}")
                logger.info(f"===== 评估结束: 迭代 {it} / {args.Iteration} =====")
                # --- 评估部分结束 ---

            # --- 4.2.2: 初始化本轮蒸馏所需的网络 ---
            ''' Train synthetic data (蒸馏合成数据) '''
            # 获取一个新的随机初始化的网络实例，用于计算本轮梯度匹配
            # 这对应 Algorithm 1 的 line 3，每次 t 循环都重新初始化
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train() # 设置为训练模式
            # 获取网络的所有可训练参数，用于后续计算梯度
            net_parameters = list(net.parameters())
            # 定义用于更新该网络 (在 inner_loop 中) 的优化器
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad() # 清空梯度

            # 初始化用于记录本轮平均匹配损失的变量
            loss_avg = 0

            # 原注释：在学习合成数据时 (即梯度匹配阶段) 不使用 DC 的数据增强，以符合 DC 论文原文。
            # 这意味着下面第 4.3 段中的 epoch 调用，其 aug 参数应为 False。
            # args.dc_aug_param = None # 这行代码实际效果存疑，因为本参数已在外部设置，且此处未被使用
            # (我们已经在第 4.3 段的 epoch 调用中确保 aug=False)

            # ==========================================================
            # 第 4.3 段：核心梯度匹配与网络更新循环
            # ==========================================================

            # 这个循环对应原代码中的 outer_loop，但在 Algorithm 1 逻辑中
            # 更像是梯度匹配 (line 8) 和网络更新 (line 9) 的交替执行步骤
            for ol in range(args.outer_loop): # outer_loop 次数由 get_loops(args.ipc) 决定

                # --- BatchNorm 处理 ---
                # 如果网络包含 BatchNorm 层，需要特殊处理以稳定统计量
                BN_flag = False
                BNSizePC = 16  # 用于估计 BN 统计量的每类样本数
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): # 检查是否有 BN 层
                        BN_flag = True
                if BN_flag:
                    logger.debug(f"检测到 BatchNorm 层，使用真实数据估计并冻结统计量 (ol={ol})")
                    # 从每个类别抽取 BNSizePC 个真实图像
                    img_real_bn = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # 确保网络处于训练模式以更新 running_mean/var
                    _ = net(img_real_bn) # 进行一次前向传播以计算统计量
                    # 将所有 BN 层设置为评估模式，冻结统计量
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():
                            module.eval()
                # --- BatchNorm 处理结束 ---

                # --- 更新合成数据 (对应 Algorithm 1 line 5-8) ---
                loss = torch.tensor(0.0).to(args.device) # 初始化当前 outer_loop 的总匹配损失

                # 按类别计算梯度匹配损失
                for c in range(num_classes):
                    logger.debug(f"计算类别 {c} 的梯度匹配损失 (ol={ol})")
                    # 采样真实数据
                    img_real = get_images(c, args.batch_real)
                    # 准备真实数据的标签 (全为 c)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    # 获取当前类别的合成图像
                    # 注意 reshape 操作，确保维度正确
                    img_syn_c = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    # 获取对应的合成标签 (全为 c)
                    lab_syn_c = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    # (DSA 增强部分已移除)

                    # --- 计算真实梯度 gw_real ---
                    output_real = net(img_real) # 前向传播
                    loss_real = criterion(output_real, lab_real) # 计算损失
                    # 计算损失关于网络参数的梯度
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    # detach 并 clone 梯度，移除计算图依赖，作为匹配目标
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    # --- 真实梯度计算结束 ---

                    # --- 计算合成梯度 gw_syn ---
                    output_syn = net(img_syn_c) # 前向传播
                    loss_syn = criterion(output_syn, lab_syn_c) # 计算损失
                    # 计算损失关于网络参数的梯度
                    # !! 关键: create_graph=True 允许计算梯度的梯度 !!
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    # --- 合成梯度计算结束 ---

                    # 使用 match_loss (调用 distance_wb) 计算两个梯度列表的距离，并累加到总损失 loss 中
                    loss += match_loss(gw_syn, gw_real, args)

                # --- 梯度匹配损失计算完成 (所有类别) ---

                # 使用累加的匹配损失更新合成图像
                optimizer_img.zero_grad() # 清空 image_syn 的梯度
                # 反向传播，计算匹配损失 loss 关于 image_syn 的梯度
                loss.backward()
                # 使用 optimizer_img 更新 image_syn 的像素值
                optimizer_img.step()
                # 累加本轮 (ol) 的匹配损失值，用于后续计算平均损失
                loss_avg += loss.item()
                # logger.debug(f"Outer loop {ol} 完成梯度匹配与合成数据更新, loss = {loss.item():.4f}")

                # --- 合成数据更新结束 ---

                # 如果这是最后一次 outer loop (ol)，则不再需要更新网络 net，
                # 因为下一轮迭代 (it+1) 会重新初始化一个新的 net。直接跳出 ol 循环。
                if ol == args.outer_loop - 1:
                    break # 跳出 for ol ... 循环

                # --- 更新网络 (对应 Algorithm 1 line 9) ---
                # 使用刚刚更新过的合成数据训练当前网络 net 几步 (inner_loop 次)
                logger.debug(f"开始更新网络 (inner loop, {args.inner_loop} 步)... (ol={ol})")
                # 准备训练数据 (深度拷贝以防万一)
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                # 使用 TensorDataset 包装合成数据
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                # 创建 DataLoader
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                # 循环 inner_loop 次
                for il in range(args.inner_loop):
                    # 调用 epoch 函数训练网络 net 一轮
                    # !! 关键 !! aug=False，因为原注释说明 DC 蒸馏时不使用增强
                    # (假设 epoch 函数已被修改为接收 device 和 dc_aug_param)
                    epoch('train', trainloader, net, optimizer_net, criterion, args.device, aug=False, dc_aug_param=None)
                # --- 网络更新结束 ---

            # --- for ol in range(args.outer_loop) 循环结束 ---

            # ==========================================================
            # 第 4.4 段：迭代日志、最终结果保存
            # (原 main.py Line 243-250 对应内容)
            # ==========================================================

            # 计算本次迭代 (it) 中所有 outer_loop 的平均匹配损失
            loss_avg /= (num_classes * args.outer_loop)

            # 每 10 次迭代打印一次平均匹配损失日志
            if it % 10 == 0:
                # 使用 logger 记录损失信息，包含时间戳
                logger.info(f"{get_time()} iter = {it:04d}, loss = {loss_avg:.4f}")

            # 定期保存检查点
            checkpoint_interval = args.checkpoint_interval if hasattr(args, 'checkpoint_interval') else 100
            if it > 0 and it % checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_path_dir, exp, it, image_syn, label_syn,
                    optimizer_img, accs_all_exps, data_save, args
                )

            # --- 在最后一次迭代 (it == args.Iteration) 保存最终结果 ---
            if it == args.Iteration: # only record the final results
                logger.info(f"达到最终迭代次数 {args.Iteration}，保存结果...")
                # 保存最终的合成数据和所有评估结果
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                save_file_path = os.path.join(args.save_path, f'res_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc.pt')
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps}, save_file_path)
                logger.success(f"结果已保存至: {save_file_path}")
            # --- 保存逻辑结束 ---

        # --- for it in range(args.Iteration + 1) 循环结束 ---
    # --- for exp in range(args.num_exp) 循环结束 ---

    # ==========================================================
    # 第 5 段：最终结果汇总与打印
    # ==========================================================

    # 在所有 num_exp 次独立实验都完成后，打印最终的汇总结果
    logger.info('\n==================== 最终结果汇总 ====================\n')
    # 遍历在评估池中的每个模型名称
    for key in model_eval_pool: #
        # 获取该模型的所有评估准确率列表
        accs = accs_all_exps[key] #
        # 确保收集到了结果 (列表不为空)
        if len(accs) > 0:
            # 计算平均准确率和标准差，并使用 logger.info 打印
            # (原代码使用 print)
            logger.info(f'运行 {args.num_exp} 次实验, 蒸馏模型 {args.model}, 评估 {len(accs)} 个随机 {key} 模型, '
                        f'平均准确率 = {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%')
        else:
            # 如果某个模型没有收集到结果，打印警告信息
            logger.warning(f"模型 {key} 没有收集到评估结果。请检查配置或运行过程。")





if __name__ == '__main__':
    main()
