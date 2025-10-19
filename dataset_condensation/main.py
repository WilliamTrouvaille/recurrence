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

from loguru import logger
from utils import (
    load_config, setup_logger, get_loops, get_dataset,
    get_network, get_eval_pool, evaluate_synset,
    match_loss, get_time, TensorDataset, epoch, get_daparam,ConfigNamespace
)



def main():
    # ==========================================================
    # 第 1 段：导入与参数解析 (Imports and Argument Parsing)
    # ==========================================================

    # === 1. 配置日志记录器 ===
    # 在程序最开始处调用日志配置函数 (假设 setup_logger 在 utils.py 中定义)
    setup_logger(log_dir="logs", log_level="INFO")

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
        args.dsa = False # 禁用 DSA

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
        os.makedirs(args.data_path, exist_ok=True) #
        os.makedirs(args.save_path, exist_ok=True) #

        logger.info(f"最终使用的配置: {args}")

    except AttributeError as e:
        # 捕获因配置文件缺少必要键而导致的属性访问错误
        logger.error(f"配置文件 '{cmd_args.config}' 中缺少必要的参数: {e}")
        return # 退出程序
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
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
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

    # 初始化用于保存最终合成数据的列表
    data_save = []



if __name__ == '__main__':
    main()
