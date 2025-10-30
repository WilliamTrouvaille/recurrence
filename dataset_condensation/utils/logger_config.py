# -*- coding: utf-8 -*-
"""
日志配置模块
包含 loguru 日志记录器的配置和管理功能
"""

import os
import sys

from loguru import logger

# 全局变量，用于跟踪日志记录器是否已经配置过
_logger_configured = False


def setup_logger(log_dir="logs", log_level="INFO", log_format=None):
    """
    配置 loguru 日志记录器，使其输出到控制台和文件。
    这个函数应该只在程序入口处被调用一次。

    参数:
        log_dir (str): 保存日志文件的目录。默认为 "logs"。
        log_level (str): 输出的最低日志级别。默认为 "INFO"。
                      可选值: "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"。
        log_format (str, optional): 自定义日志格式。如果为 None，则使用默认格式。

    返回:
        bool: 配置是否成功
    """
    global _logger_configured

    # 检查是否已经配置过，防止重复添加 handlers
    if _logger_configured:
        logger.warning("日志记录器已经被配置过，跳过重复配置。")
        return False

    # 1. (可选但推荐) 移除默认的 loguru handler
    try:
        logger.remove(0)
    except ValueError:
        pass  # Handler 0 不存在，没关系

    # 2. 定义期望的日志格式
    if log_format is None:
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
    return True


def setup_simple_logger(log_file="app.log", log_level="INFO"):
    """
    设置简单的日志记录器，只输出到文件。

    参数:
        log_file (str): 日志文件路径
        log_level (str): 日志级别

    返回:
        bool: 配置是否成功
    """
    global _logger_configured

    if _logger_configured:
        logger.warning("日志记录器已经被配置过，跳过重复配置。")
        return False

    # 移除默认 handler
    try:
        logger.remove(0)
    except ValueError:
        pass

    # 简单格式
    simple_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

    # 添加文件输出
    logger.add(
        log_file,
        level=log_level.upper(),
        format=simple_format,
        rotation="5 MB",
        retention="7 days",
        encoding="utf-8"
    )

    _logger_configured = True
    logger.info(f"简单日志记录器配置完成。日志文件: {log_file}")
    return True


def setup_debug_logger(log_dir="debug_logs"):
    """
    设置调试模式的日志记录器，包含更详细的信息。

    参数:
        log_dir (str): 调试日志目录

    返回:
        bool: 配置是否成功
    """
    global _logger_configured

    if _logger_configured:
        logger.warning("日志记录器已经被配置过，跳过重复配置。")
        return False

    # 移除默认 handler
    try:
        logger.remove(0)
    except ValueError:
        pass

    # 调试格式，包含更多信息
    debug_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <9}</level> | "
        "<cyan>{process}</cyan>:<cyan>{thread}</cyan> | "
        "<yellow>{name}</yellow>:<yellow>{function}</yellow>:<yellow>{line}</yellow> | "
        "<level>{message}</level>"
    )

    # 控制台输出
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=debug_format,
        colorize=True
    )

    # 调试文件输出
    os.makedirs(log_dir, exist_ok=True)
    debug_log_file = os.path.join(log_dir, "debug_{time:YYYYMMDD_HHmmss}.log")

    logger.add(
        debug_log_file,
        level="TRACE",  # 最详细的级别
        format=debug_format,
        rotation="20 MB",
        retention="3 days",  # 调试日志保留时间较短
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )

    _logger_configured = True
    logger.info(f"调试日志记录器配置完成。日志目录: {log_dir}")
    return True


def add_file_sink(log_file, log_level="INFO", log_format=None, **kwargs):
    """
    添加额外的文件输出 sink。

    参数:
        log_file (str): 日志文件路径
        log_level (str): 日志级别
        log_format (str, optional): 日志格式
        **kwargs: 其他 logger.add 参数

    返回:
        int: 添加的 sink 的 ID，如果失败则返回 -1
    """
    if log_format is None:
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

    try:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # 默认参数
        default_kwargs = {
            'level': log_level.upper(),
            'format': log_format,
            'encoding': 'utf-8',
            'enqueue': True
        }
        default_kwargs.update(kwargs)

        sink_id = logger.add(log_file, **default_kwargs)
        logger.info(f"已添加文件输出 sink: {log_file} (ID: {sink_id})")
        return sink_id

    except Exception as e:
        logger.error(f"添加文件输出 sink 失败: {log_file}, 错误: {e}")
        return -1


def remove_sink(sink_id):
    """
    移除指定的 sink。

    参数:
        sink_id (int): sink 的 ID

    返回:
        bool: 移除是否成功
    """
    try:
        logger.remove(sink_id)
        logger.info(f"已移除 sink: {sink_id}")
        return True
    except Exception as e:
        logger.error(f"移除 sink 失败: {sink_id}, 错误: {e}")
        return False


def get_logger_configured():
    """
    获取日志记录器是否已配置的状态。

    返回:
        bool: 如果已配置返回 True，否则返回 False
    """
    return _logger_configured


def reset_logger_config():
    """
    重置日志记录器配置状态。
    这个函数主要用于测试或特殊场景。

    返回:
        bool: 重置是否成功
    """
    global _logger_configured
    try:
        # 移除所有 handlers
        logger.remove()
        _logger_configured = False
        return True
    except Exception as e:
        logger.error(f"重置日志记录器配置失败: {e}")
        return False


def log_system_info():
    """
    记录系统信息到日志中。

    返回:
        None
    """
    import platform
    import torch

    logger.info("=" * 50)
    logger.info("系统信息")
    logger.info("=" * 50)
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"Python 版本: {platform.python_version()}")
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA 版本: {torch.version.cuda}")
        logger.info(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info("=" * 50)


def log_experiment_start(exp_name, config_dict):
    """
    记录实验开始信息。

    参数:
        exp_name (str): 实验名称
        config_dict (dict): 实验配置字典

    返回:
        None
    """
    logger.info("=" * 60)
    logger.info(f"实验开始: {exp_name}")
    logger.info("=" * 60)

    # 记录配置参数
    for key, value in config_dict.items():
        logger.info(f"{key}: {value}")

    logger.info("=" * 60)


def log_experiment_end(exp_name, results_dict):
    """
    记录实验结束信息。

    参数:
        exp_name (str): 实验名称
        results_dict (dict): 实验结果字典

    返回:
        None
    """
    logger.info("=" * 60)
    logger.info(f"实验结束: {exp_name}")
    logger.info("=" * 60)

    # 记录结果
    for key, value in results_dict.items():
        logger.info(f"{key}: {value}")

    logger.info("=" * 60)
