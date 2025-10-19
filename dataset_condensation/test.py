#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/10/19 13:05
@version : 1.0.0
@author  : William_Trouvaille
@function: 测试类
"""
from utils import setup_logger, load_config
from loguru import logger

def main():
    setup_logger(log_level='DEBUG')
    config=load_config()


    logger.info("开始实验...")
    logger.debug(f"config==>{config},str==>{config['test']['str']},dist==>{config['test']['dist']['He']}")
    logger.warning("这是一条警告信息。")
    logger.success("实验完成")



if __name__ == '__main__':
    main()
