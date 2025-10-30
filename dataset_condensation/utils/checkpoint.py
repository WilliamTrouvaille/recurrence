# -*- coding: utf-8 -*-
"""
检查点管理模块
包含训练检查点的保存、加载和管理功能
"""

import glob
import os
from typing import Dict, Any, Optional, List

import torch
from loguru import logger


def save_checkpoint(checkpoint_path: str, exp: int, it: int, image_syn: torch.Tensor,
                    label_syn: torch.Tensor, optimizer_img, accs_all_exps: Dict,
                    data_save: List, args, additional_data: Optional[Dict] = None) -> str:
    """
    保存训练检查点。

    参数:
        checkpoint_path (str): 检查点保存路径
        exp (int): 当前实验编号
        it (int): 当前迭代次数
        image_syn (Tensor): 当前合成图像
        label_syn (Tensor): 当前合成标签
        optimizer_img (Optimizer): 图像优化器
        accs_all_exps (dict): 所有实验的准确率记录
        data_save (list): 保存的数据列表
        args (ConfigNamespace): 配置参数
        additional_data (dict, optional): 额外要保存的数据

    返回:
        str: 保存的检查点文件路径
    """
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_file = os.path.join(
        checkpoint_path,
        f'checkpoint_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_exp{exp}_iter{it}.pt'
    )

    checkpoint = {
        'exp': exp,
        'iteration': it,
        'image_syn': image_syn.detach().cpu(),
        'label_syn': label_syn.detach().cpu(),
        'optimizer_img_state': optimizer_img.state_dict(),
        'accs_all_exps': accs_all_exps,
        'data_save': data_save,
        'args': vars(args) if hasattr(args, 'vars') else args
    }

    # 添加额外数据
    if additional_data:
        checkpoint.update(additional_data)

    torch.save(checkpoint, checkpoint_file)
    logger.info(f"检查点已保存至: {checkpoint_file}")

    return checkpoint_file


def load_checkpoint(checkpoint_path: str, args) -> Optional[Dict[str, Any]]:
    """
    加载最新的训练检查点。

    参数:
        checkpoint_path (str): 检查点保存路径
        args (ConfigNamespace): 配置参数

    返回:
        dict or None: 检查点字典，如果不存在则返回None
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f"检查点目录不存在: {checkpoint_path}")
        return None

    # 查找所有匹配的检查点文件
    pattern = f'checkpoint_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_*.pt'
    checkpoint_files = [f for f in os.listdir(checkpoint_path)
                        if f.startswith('checkpoint_') and f.endswith('.pt')]

    if not checkpoint_files:
        logger.info("未找到检查点文件，将从头开始训练")
        return None

    # 获取最新的检查点
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)), reverse=True)
    latest_checkpoint = os.path.join(checkpoint_path, checkpoint_files[0])

    logger.info(f"加载检查点: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')

    return checkpoint


def load_specific_checkpoint(checkpoint_file: str) -> Optional[Dict[str, Any]]:
    """
    加载指定的检查点文件。

    参数:
        checkpoint_file (str): 检查点文件路径

    返回:
        dict or None: 检查点字典，如果加载失败则返回None
    """
    if not os.path.exists(checkpoint_file):
        logger.error(f"检查点文件不存在: {checkpoint_file}")
        return None

    try:
        logger.info(f"加载检查点: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        return checkpoint
    except Exception as e:
        logger.error(f"加载检查点失败: {checkpoint_file}, 错误: {e}")
        return None


def save_best_model(model, optimizer, epoch, acc, config, save_path: str,
                    additional_data: Optional[Dict] = None) -> str:
    """
    保存最佳模型。

    参数:
        model (nn.Module): 模型
        optimizer (Optimizer): 优化器
        epoch (int): 当前轮次
        acc (float): 准确率
        config (dict): 配置信息
        save_path (str): 保存路径
        additional_data (dict, optional): 额外数据

    返回:
        str: 保存的文件路径
    """
    os.makedirs(save_path, exist_ok=True)

    best_model_file = os.path.join(save_path, 'best_model.pth')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc,
        'config': config
    }

    if additional_data:
        checkpoint.update(additional_data)

    torch.save(checkpoint, best_model_file)
    logger.info(f"最佳模型已保存至: {best_model_file} (准确率: {acc:.4f})")

    return best_model_file


def load_best_model(model, optimizer, save_path: str, device='cpu') -> bool:
    """
    加载最佳模型。

    参数:
        model (nn.Module): 模型
        optimizer (Optimizer): 优化器
        save_path (str): 保存路径
        device (str): 设备

    返回:
        bool: 加载是否成功
    """
    best_model_file = os.path.join(save_path, 'best_model.pth')

    if not os.path.exists(best_model_file):
        logger.warning(f"最佳模型文件不存在: {best_model_file}")
        return False

    try:
        checkpoint = torch.load(best_model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"已加载最佳模型 (轮次: {checkpoint['epoch']}, 准确率: {checkpoint['accuracy']:.4f})")
        return True
    except Exception as e:
        logger.error(f"加载最佳模型失败: {e}")
        return False


def save_epoch_checkpoint(model, optimizer, epoch, save_path: str,
                          additional_data: Optional[Dict] = None) -> str:
    """
    保存每个 epoch 的检查点。

    参数:
        model (nn.Module): 模型
        optimizer (Optimizer): 优化器
        epoch (int): 当前轮次
        save_path (str): 保存路径
        additional_data (dict, optional): 额外数据

    返回:
        str: 保存的文件路径
    """
    os.makedirs(save_path, exist_ok=True)

    checkpoint_file = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    if additional_data:
        checkpoint.update(additional_data)

    torch.save(checkpoint, checkpoint_file)
    logger.debug(f"检查点已保存: {checkpoint_file}")

    return checkpoint_file


def cleanup_checkpoints(save_path: str, max_checkpoints: int = 3) -> List[str]:
    """
    清理旧的检查点文件，保留最新的几个。

    参数:
        save_path (str): 保存路径
        max_checkpoints (int): 最大保留数量

    返回:
        list: 被删除的文件列表
    """
    if not os.path.exists(save_path):
        return []

    # 查找所有检查点文件
    pattern = os.path.join(save_path, 'checkpoint_epoch_*.pth')
    checkpoint_files = glob.glob(pattern)

    if len(checkpoint_files) <= max_checkpoints:
        return []

    # 按修改时间排序，删除最旧的文件
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    files_to_delete = checkpoint_files[max_checkpoints:]

    deleted_files = []
    for file in files_to_delete:
        try:
            os.remove(file)
            deleted_files.append(file)
            logger.info(f"已删除旧检查点: {file}")
        except Exception as e:
            logger.error(f"删除检查点失败: {file}, 错误: {e}")

    return deleted_files


def list_checkpoints(checkpoint_path: str, pattern: str = '*.pt') -> List[str]:
    """
    列出所有检查点文件。

    参数:
        checkpoint_path (str): 检查点路径
        pattern (str): 文件模式

    返回:
        list: 检查点文件列表
    """
    if not os.path.exists(checkpoint_path):
        return []

    search_pattern = os.path.join(checkpoint_path, pattern)
    checkpoint_files = glob.glob(search_pattern)

    # 按修改时间排序
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    return checkpoint_files


def get_checkpoint_info(checkpoint_file: str) -> Optional[Dict[str, Any]]:
    """
    获取检查点文件信息（不加载完整数据）。

    参数:
        checkpoint_file (str): 检查点文件路径

    返回:
        dict or None: 检查点信息
    """
    if not os.path.exists(checkpoint_file):
        return None

    try:
        # 只加载元数据，不加载张量数据
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        info = {
            'file_path': checkpoint_file,
            'file_size': os.path.getsize(checkpoint_file),
            'modified_time': os.path.getmtime(checkpoint_file)
        }

        # 提取关键信息
        if 'exp' in checkpoint:
            info['exp'] = checkpoint['exp']
        if 'iteration' in checkpoint:
            info['iteration'] = checkpoint['iteration']
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
        if 'accuracy' in checkpoint:
            info['accuracy'] = checkpoint['accuracy']
        if 'args' in checkpoint:
            args = checkpoint['args']
            if isinstance(args, dict):
                info['method'] = args.get('method')
                info['dataset'] = args.get('dataset')
                info['model'] = args.get('model')
                info['ipc'] = args.get('ipc')

        return info
    except Exception as e:
        logger.error(f"获取检查点信息失败: {checkpoint_file}, 错误: {e}")
        return None


def verify_checkpoint(checkpoint_file: str) -> bool:
    """
    验证检查点文件的完整性。

    参数:
        checkpoint_file (str): 检查点文件路径

    返回:
        bool: 文件是否完整
    """
    try:
        # 尝试加载检查点
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        # 检查必要的键
        required_keys = ['args']  # 至少应该有配置信息
        missing_keys = [key for key in required_keys if key not in checkpoint]

        if missing_keys:
            logger.warning(f"检查点缺少必要的键: {missing_keys}")
            return False

        logger.debug(f"检查点验证通过: {checkpoint_file}")
        return True
    except Exception as e:
        logger.error(f"检查点验证失败: {checkpoint_file}, 错误: {e}")
        return False


def backup_checkpoint(checkpoint_file: str, backup_dir: str) -> bool:
    """
    备份检查点文件。

    参数:
        checkpoint_file (str): 原检查点文件
        backup_dir (str): 备份目录

    返回:
        bool: 备份是否成功
    """
    if not os.path.exists(checkpoint_file):
        logger.error(f"源检查点文件不存在: {checkpoint_file}")
        return False

    try:
        os.makedirs(backup_dir, exist_ok=True)

        # 生成备份文件名
        import shutil
        filename = os.path.basename(checkpoint_file)
        backup_file = os.path.join(backup_dir, f"backup_{filename}")

        shutil.copy2(checkpoint_file, backup_file)
        logger.info(f"检查点已备份至: {backup_file}")
        return True
    except Exception as e:
        logger.error(f"备份检查点失败: {e}")
        return False


def restore_from_checkpoint(checkpoint_file: str, model, optimizer=None, device='cpu') -> bool:
    """
    从检查点恢复训练状态。

    参数:
        checkpoint_file (str): 检查点文件路径
        model (nn.Module): 模型
        optimizer (Optimizer, optional): 优化器
        device (str): 设备

    返回:
        bool: 恢复是否成功
    """
    checkpoint = load_specific_checkpoint(checkpoint_file)
    if checkpoint is None:
        return False

    try:
        # 加载模型状态
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'image_syn' in checkpoint:
            # 特殊处理 dataset condensation 的检查点
            pass  # 由调用者处理具体逻辑

        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif optimizer is not None and 'optimizer_img_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_img_state'])

        logger.info(f"已从检查点恢复训练状态: {checkpoint_file}")
        return True
    except Exception as e:
        logger.error(f"恢复训练状态失败: {e}")
        return False
