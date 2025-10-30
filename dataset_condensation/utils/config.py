# -*- coding: utf-8 -*-
"""
配置加载模块
包含 YAML 配置文件的加载、解析和管理功能
"""

import os

import yaml
from loguru import logger


class ConfigNamespace:
    """
    配置命名空间类，将字典转换为对象，以便通过属性访问配置项。
    """

    def __init__(self, config_dict):
        """
        初始化配置命名空间。

        参数:
            config_dict (dict): 配置字典
        """
        # 将字典的键值对设置为对象的属性
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归创建 ConfigNamespace 对象
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        """
        定义对象的打印形式，方便日志记录。

        返回:
            str: 对象的字符串表示
        """
        return str(vars(self))

    def to_dict(self):
        """
        将 ConfigNamespace 对象转换回字典。

        返回:
            dict: 配置字典
        """
        result = {}
        for key, value in vars(self).items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key, default=None):
        """
        安全地获取配置项，如果不存在则返回默认值。

        参数:
            key (str): 配置项键名
            default: 默认值

        返回:
            配置值或默认值
        """
        return getattr(self, key, default)

    def update(self, new_config):
        """
        更新配置项。

        参数:
            new_config (dict or ConfigNamespace): 新的配置

        返回:
            None
        """
        if isinstance(new_config, ConfigNamespace):
            new_config = new_config.to_dict()

        for key, value in new_config.items():
            if isinstance(value, dict) and hasattr(self, key):
                # 如果现有属性是 ConfigNamespace 对象，递归更新
                if isinstance(getattr(self, key), ConfigNamespace):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)


def load_config(config_path="config.yaml"):
    """
    加载 YAML 配置文件。

    参数:
        config_path (str): YAML 配置文件的路径。默认为当前目录下的 "config.yaml"。

    返回:
        dict: 包含配置参数的字典。如果文件不存在或解析失败，则返回 None。
    """
    resolved_path = os.path.abspath(config_path)  # 获取绝对路径，方便日志记录
    logger.info(f"尝试从 '{resolved_path}' 加载配置...")

    if not os.path.exists(resolved_path):
        logger.error(f"配置文件未找到: {resolved_path}")
        return None

    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)  # 使用 safe_load 防止执行任意代码
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


def load_config_namespace(config_path="config.yaml"):
    """
    加载 YAML 配置文件并返回 ConfigNamespace 对象。

    参数:
        config_path (str): YAML 配置文件的路径。默认为当前目录下的 "config.yaml"。

    返回:
        ConfigNamespace: 配置命名空间对象。如果加载失败则返回 None。
    """
    config_dict = load_config(config_path)
    if config_dict is None:
        return None

    return ConfigNamespace(config_dict)


def save_config(config_dict, config_path):
    """
    保存配置到 YAML 文件。

    参数:
        config_dict (dict or ConfigNamespace): 要保存的配置字典
        config_path (str): 保存路径

    返回:
        bool: 保存是否成功
    """
    if isinstance(config_dict, ConfigNamespace):
        config_dict = config_dict.to_dict()

    resolved_path = os.path.abspath(config_path)

    # 确保目录存在
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

    try:
        with open(resolved_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
        logger.success(f"配置文件已保存至: {resolved_path}")
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {resolved_path}, 错误: {e}")
        return False


def merge_configs(*configs):
    """
    合并多个配置字典，后面的配置会覆盖前面的配置。

    参数:
        *configs: 多个配置字典或 ConfigNamespace 对象

    返回:
        dict: 合并后的配置字典
    """
    merged = {}

    for config in configs:
        if isinstance(config, ConfigNamespace):
            config = config.to_dict()

        if isinstance(config, dict):
            merged.update(config)
        else:
            logger.warning(f"跳过无效的配置类型: {type(config)}")

    return merged


def get_default_config():
    """
    获取默认配置。

    返回:
        dict: 默认配置字典
    """
    default_config = {
        # 实验基本配置
        'experiment': {
            'name': 'dataset_condensation',
            'seed': 42,
            'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
        },

        # 数据集配置
        'dataset': {
            'name': 'CIFAR10',
            'data_path': './data',
            'ipc': 1,  # Images Per Class
            'batch_size': 128,
            'num_workers': 4
        },

        # 模型配置
        'model': {
            'name': 'ConvNet',
            'net_width': 128,
            'net_depth': 3,
            'net_act': 'relu',
            'net_norm': 'batchnorm',
            'net_pooling': 'avgpooling'
        },

        # 训练配置
        'training': {
            'epochs': 1000,
            'lr_img': 1.0,
            'lr_net': 0.01,
            'optimizer': 'adam',
            'epoch_eval_train': 300,
            'batch_real': 256,
            'dc_aug_param': None,
            'eval_mode': 'S'
        },

        # 评估配置
        'evaluation': {
            'eval_it': 1000,
            'eval_freq': 100,
            'save_it': 1000
        },

        # 日志配置
        'logging': {
            'log_dir': './logs',
            'log_level': 'INFO',
            'save_log': True
        },

        # 检查点配置
        'checkpoint': {
            'save_dir': './checkpoints',
            'save_freq': 100,
            'save_best': True,
            'max_checkpoints': 3
        }
    }

    return default_config


def validate_config(config_dict, required_keys=None):
    """
    验证配置文件的完整性。

    参数:
        config_dict (dict): 配置字典
        required_keys (list, optional): 必需的配置键列表

    返回:
        tuple: (is_valid, missing_keys)
            is_valid (bool): 配置是否有效
            missing_keys (list): 缺失的必需键列表
    """
    if required_keys is None:
        # 默认必需的配置键
        required_keys = [
            'dataset.name',
            'model.name',
            'training.epochs',
            'training.lr_img'
        ]

    missing_keys = []

    for key in required_keys:
        keys = key.split('.')
        current = config_dict

        try:
            for k in keys:
                current = current[k]
        except (KeyError, TypeError):
            missing_keys.append(key)

    is_valid = len(missing_keys) == 0

    if not is_valid:
        logger.error(f"配置验证失败，缺失必需的配置项: {missing_keys}")
    else:
        logger.info("配置验证通过")

    return is_valid, missing_keys


def create_sample_config(output_path="sample_config.yaml"):
    """
    创建示例配置文件。

    参数:
        output_path (str): 输出路径

    返回:
        bool: 创建是否成功
    """
    sample_config = get_default_config()

    # 添加注释说明
    sample_config['_comments'] = {
        'experiment': '实验基本配置，包括名称、随机种子等',
        'dataset': '数据集相关配置，包括名称、路径、每类样本数等',
        'model': '神经网络模型架构配置',
        'training': '训练过程相关参数',
        'evaluation': '模型评估相关配置',
        'logging': '日志记录配置',
        'checkpoint': '检查点保存配置'
    }

    return save_config(sample_config, output_path)


def update_config_from_args(config_dict, args_dict):
    """
    从命令行参数更新配置。

    参数:
        config_dict (dict): 原始配置字典
        args_dict (dict): 命令行参数字典

    返回:
        dict: 更新后的配置字典
    """
    updated_config = config_dict.copy()

    for key, value in args_dict.items():
        if value is not None:  # 只更新非 None 的值
            # 处理嵌套键，如 "dataset.name"
            if '.' in key:
                keys = key.split('.')
                current = updated_config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                updated_config[key] = value

    logger.info(f"已从命令行参数更新配置: {list(args_dict.keys())}")
    return updated_config


def print_config(config_dict, title="配置信息"):
    """
    打印配置信息到日志。

    参数:
        config_dict (dict or ConfigNamespace): 配置字典
        title (str): 标题

    返回:
        None
    """
    logger.info("=" * 60)
    logger.info(f"{title}")
    logger.info("=" * 60)

    def print_recursive(d, indent=0):
        for key, value in d.items():
            if key.startswith('_'):  # 跳过注释等特殊字段
                continue
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                print_recursive(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")

    if isinstance(config_dict, ConfigNamespace):
        config_dict = config_dict.to_dict()

    print_recursive(config_dict)
    logger.info("=" * 60)
