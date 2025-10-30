# -*- coding: utf-8 -*-
"""
辅助函数模块
包含各种通用的辅助函数和工具
"""

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def get_time(format_str: str = "[%Y-%m-%d %H:%M:%S]") -> str:
    """
    获取格式化的当前时间。

    参数:
        format_str (str): 时间格式字符串

    返回:
        str: 格式化的时间字符串
    """
    return str(time.strftime(format_str, time.localtime()))


def set_random_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保实验的可复现性。

    参数:
        seed (int): 随机种子值

    返回:
        None
    """
    # 设置 Python 内置随机数生成器种子
    random.seed(seed)

    # 设置 NumPy 随机数生成器种子
    np.random.seed(seed)

    # 设置 PyTorch 随机数生成器种子
    torch.manual_seed(seed)

    # 如果使用 CUDA，也设置 CUDA 随机数生成器种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 确保 cuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"已设置随机种子: {seed}")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    获取计算设备。

    参数:
        device (str, optional): 指定设备 ('cuda', 'cpu', 'auto')

    返回:
        torch.device: 计算设备
    """
    if device is None or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device_obj = torch.device(device)

    if device_obj.type == 'cuda':
        logger.info(f"使用 GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU 数量: {torch.cuda.device_count()}")
    else:
        logger.info("使用 CPU")

    return device_obj


def format_time(seconds: float) -> str:
    """
    格式化时间（秒）为可读的字符串。

    参数:
        seconds (float): 秒数

    返回:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def format_number(num: Union[int, float], precision: int = 2) -> str:
    """
    格式化数字，自动选择合适的单位（K, M, G等）。

    参数:
        num (int or float): 数字
        precision (int): 小数位数

    返回:
        str: 格式化的数字字符串
    """
    if isinstance(num, int) or num.is_integer():
        num = int(num)

        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.{precision}f}G"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.{precision}f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.{precision}f}K"
        else:
            return str(num)
    else:
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.{precision}f}G"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.{precision}f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.{precision}f}K"
        else:
            return f"{num:.{precision}f}"


def create_progress_bar(iterable, desc: str = "Processing", **kwargs) -> tqdm:
    """
    创建进度条。

    参数:
        iterable: 可迭代对象
        desc (str): 描述文本
        **kwargs: tqdm 的其他参数

    返回:
        tqdm: 进度条对象
    """
    default_kwargs = {
        'desc': desc,
        'leave': False,
        'ncols': 100,
        'ascii': True  # 确保 ASCII 兼容
    }
    default_kwargs.update(kwargs)

    return tqdm(iterable, **default_kwargs)


def ensure_dir(path: str) -> bool:
    """
    确保目录存在，如果不存在则创建。

    参数:
        path (str): 目录路径

    返回:
        bool: 操作是否成功
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"创建目录失败: {path}, 错误: {e}")
        return False


def get_file_size(file_path: str) -> str:
    """
    获取文件大小的可读格式。

    参数:
        file_path (str): 文件路径

    返回:
        str: 文件大小字符串
    """
    if not os.path.exists(file_path):
        return "File not found"

    size_bytes = os.path.getsize(file_path)
    return format_size(size_bytes)


def format_size(size_bytes: int) -> str:
    """
    格式化字节大小为可读格式。

    参数:
        size_bytes (int): 字节数

    返回:
        str: 格式化的大小字符串
    """
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = 1024 ** i
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"


def save_dict_to_json(data: Dict[str, Any], file_path: str) -> bool:
    """
    保存字典到 JSON 文件。

    参数:
        data (dict): 要保存的数据
        file_path (str): 文件路径

    返回:
        bool: 保存是否成功
    """
    try:
        import json
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"数据已保存到 JSON 文件: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存 JSON 文件失败: {e}")
        return False


def load_dict_from_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    从 JSON 文件加载字典。

    参数:
        file_path (str): 文件路径

    返回:
        dict or None: 加载的数据，失败时返回 None
    """
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"已从 JSON 文件加载数据: {file_path}")
        return data
    except Exception as e:
        logger.error(f"加载 JSON 文件失败: {e}")
        return None


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: int = 1) -> float:
    """
    计算分类准确率。

    参数:
        outputs (Tensor): 模型输出
        targets (Tensor): 目标标签
        topk (int): Top-K 准确率，默认为 1

    返回:
        float: 准确率
    """
    with torch.no_grad():
        batch_size = targets.size(0)

        if topk == 1:
            # Top-1 准确率
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
        else:
            # Top-K 准确率
            _, pred = outputs.topk(topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred)).contiguous().view(-1).eq(1).sum().item()

        accuracy = 100.0 * correct / batch_size
        return accuracy


def smooth_label(targets: torch.Tensor, num_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    """
    标签平滑。

    参数:
        targets (Tensor): 原始标签
        num_classes (int): 类别数
        smoothing (float): 平滑系数

    返回:
        Tensor: 平滑后的标签
    """
    assert 0 <= smoothing < 1

    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)

    one_hot = torch.zeros_like(targets).float()
    one_hot.scatter_(1, targets.unsqueeze(1), confidence)
    one_hot += smooth_value

    return one_hot


def get_model_size(model: torch.nn.Module) -> Dict[str, Union[int, str]]:
    """
    获取模型大小信息。

    参数:
        model (nn.Module): 模型

    返回:
        dict: 模型信息
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算模型大小（MB）
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024

    return {
        'total_params': param_count,
        'trainable_params': trainable_count,
        'non_trainable_params': param_count - trainable_count,
        'model_size_mb': f"{size_mb:.2f} MB",
        'model_size_bytes': param_size + buffer_size
    }


def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    验证张量是否有效（无 NaN 或 Inf）。

    参数:
        tensor (Tensor): 要验证的张量
        name (str): 张量名称

    返回:
        bool: 张量是否有效
    """
    if torch.isnan(tensor).any():
        logger.error(f"张量 {name} 包含 NaN 值")
        return False

    if torch.isinf(tensor).any():
        logger.error(f"张量 {name} 包含 Inf 值")
        return False

    return True


def clip_tensor(tensor: torch.Tensor, min_val: Optional[float] = None,
                max_val: Optional[float] = None) -> torch.Tensor:
    """
    裁剪张量值到指定范围。

    参数:
        tensor (Tensor): 输入张量
        min_val (float, optional): 最小值
        max_val (float, optional): 最大值

    返回:
        Tensor: 裁剪后的张量
    """
    if min_val is not None and max_val is not None:
        return torch.clamp(tensor, min=min_val, max=max_val)
    elif min_val is not None:
        return torch.clamp(tensor, min=min_val)
    elif max_val is not None:
        return torch.clamp(tensor, max=max_val)
    else:
        return tensor


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    计算模型参数数量。

    参数:
        model (nn.Module): 模型
        trainable_only (bool): 是否只计算可训练参数

    返回:
        int: 参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_memory_usage(device: torch.device) -> Dict[str, Union[float, str]]:
    """
    获取 GPU 内存使用情况。

    参数:
        device (torch.device): 设备

    返回:
        dict: 内存使用信息
    """
    if device.type != 'cuda':
        return {'error': 'Only available for CUDA devices'}

    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
    cached = torch.cuda.memory_reserved(device) / 1024 / 1024  # MB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB

    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024  # MB

    return {
        'allocated_mb': f"{allocated:.2f} MB",
        'cached_mb': f"{cached:.2f} MB",
        'max_allocated_mb': f"{max_allocated:.2f} MB",
        'total_memory_mb': f"{total_memory:.2f} MB",
        'utilization_percent': f"{(allocated / total_memory * 100):.1f}%"
    }


def clear_memory(device: torch.device) -> None:
    """
    清理 GPU 内存。

    参数:
        device (torch.device): 设备

    返回:
        None
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("已清理 GPU 缓存")


def log_memory_usage(device: torch.device, description: str = "Memory usage") -> None:
    """
    记录内存使用情况到日志。

    参数:
        device (torch.device): 设备
        description (str): 描述

    返回:
        None
    """
    memory_info = get_memory_usage(device)
    if 'error' not in memory_info:
        logger.info(f"{description}: {memory_info['utilization_percent']} "
                    f"({memory_info['allocated_mb']} / {memory_info['total_memory_mb']})")


def batch_tensor(tensor: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    """
    将张量分批。

    参数:
        tensor (Tensor): 输入张量
        batch_size (int): 批次大小

    返回:
        list: 批次张量列表
    """
    if tensor.size(0) <= batch_size:
        return [tensor]

    batches = []
    for i in range(0, tensor.size(0), batch_size):
        batch = tensor[i:i + batch_size]
        batches.append(batch)

    return batches
