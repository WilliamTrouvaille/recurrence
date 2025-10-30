#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025/10/30
@version : 1.0.0
@author  : William_Trouvaille
@function: Dataset Condensation 工具包初始化模块
"""

# 导入 loguru 日志记录器
from loguru import logger

# 导入各个模块的主要功能
from .data import get_dataset, TensorDataset
from .model import get_network, get_default_convnet_setting, get_loops
from .algorithm import distance_wb, match_loss
from .training import epoch, evaluate_synset, get_eval_pool
from .augmentation import augment, get_daparam
from .config import load_config, ConfigNamespace
from .checkpoint import save_checkpoint, load_checkpoint
from .helpers import get_time
from .logger_config import setup_logger

# 版本信息
__version__ = "1.0.0"
__author__ = "William_Trouvaille"

# 导出主要接口（保持与原始 utils.py 的兼容性）
__all__ = [
    # 数据处理
    'get_dataset', 'TensorDataset',

    # 模型配置
    'get_network', 'get_default_convnet_setting', 'get_loops',

    # 核心算法
    'distance_wb', 'match_loss',

    # 训练与评估
    'epoch', 'evaluate_synset', 'get_eval_pool',

    # 数据增强
    'augment', 'get_daparam',

    # 配置管理
    'load_config', 'ConfigNamespace',

    # 检查点管理
    'save_checkpoint', 'load_checkpoint',

    # 辅助函数
    'get_time',

    # 日志配置
    'setup_logger',

    # 日志记录器
    'logger'
]