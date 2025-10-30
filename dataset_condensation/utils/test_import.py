#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试导入功能
验证 utils 包的所有模块和函数是否可以正常导入和使用
"""

import os
import sys

# 添加父目录到路径，以便能够导入 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_basic_import():
    """测试基本导入功能"""
    print("=" * 60)
    print("测试基本导入功能")
    print("=" * 60)

    try:
        # 测试导入整个 utils 包
        import utils
        print("✓ 成功导入 utils 包")
        print(f"✓ 版本: {utils.__version__}")
        print(f"✓ 作者: {utils.__author__}")

        # 测试导入 loguru
        from utils import logger
        print("✓ 成功导入 logger")

        return True
    except Exception as e:
        print(f"✗ 基本导入失败: {e}")
        return False


def test_module_imports():
    """测试各个模块的导入"""
    print("\n" + "=" * 60)
    print("测试各个模块的导入")
    print("=" * 60)

    modules_to_test = [
        'data', 'model', 'algorithm', 'training', 'augmentation',
        'logger_config', 'config', 'checkpoint', 'helpers'
    ]

    results = {}
    for module_name in modules_to_test:
        try:
            # 测试导入模块
            module = __import__(f'utils.{module_name}', fromlist=[module_name])
            print(f"✓ 成功导入 utils.{module_name}")

            # 测试模块中的一些函数
            if module_name == 'data':
                from utils.data import get_dataset
                print(f"  ✓ 成功导入 get_dataset")
            elif module_name == 'model':
                from utils.model import get_model
                print(f"  ✓ 成功导入 get_model")
            elif module_name == 'training':
                from utils.training import eval_model
                print(f"  ✓ 成功导入 eval_model")
            elif module_name == 'config':
                from utils.config import ConfigNamespace
                print(f"  ✓ 成功导入 ConfigNamespace")
            elif module_name == 'helpers':
                from utils.helpers import get_time, set_random_seed
                print(f"  ✓ 成功导入 get_time, set_random_seed")

            results[module_name] = True

        except Exception as e:
            print(f"✗ 导入 utils.{module_name} 失败: {e}")
            results[module_name] = False

    return results


def test_function_usage():
    """测试函数的基本使用"""
    print("\n" + "=" * 60)
    print("测试函数的基本使用")
    print("=" * 60)

    try:
        # 测试 helpers 模块的函数
        from utils.helpers import get_time, format_time, format_number
        current_time = get_time()
        print(f"✓ get_time(): {current_time}")

        formatted_time = format_time(3661.5)
        print(f"✓ format_time(3661.5): {formatted_time}")

        formatted_number = format_number(1500000)
        print(f"✓ format_number(1500000): {formatted_number}")

        # 测试 config 模块的函数
        from utils.config import get_default_config, ConfigNamespace
        default_config = get_default_config()
        print(f"✓ get_default_config(): 获得 {len(default_config)} 个配置项")

        config_namespace = ConfigNamespace({'test': 123})
        print(f"✓ ConfigNamespace: test = {config_namespace.test}")

        # 测试 logger_config 模块的函数
        from utils.logger_config import get_logger_configured
        configured = get_logger_configured()
        print(f"✓ get_logger_configured(): {configured}")

        return True

    except Exception as e:
        print(f"✗ 函数使用测试失败: {e}")
        return False


def test_quick_setup():
    """测试快速设置功能"""
    print("\n" + "=" * 60)
    print("测试快速设置功能")
    print("=" * 60)

    try:
        from utils import quick_setup
        success = quick_setup(log_level="INFO", log_dir="./test_logs")
        if success:
            print("✓ quick_setup() 执行成功")
        else:
            print("✗ quick_setup() 执行失败")
        return success
    except Exception as e:
        print(f"✗ quick_setup() 测试失败: {e}")
        return False


def test_module_utilities():
    """测试工具包实用函数"""
    print("\n" + "=" * 60)
    print("测试工具包实用函数")
    print("=" * 60)

    try:
        import utils

        # 测试列出所有模块
        all_modules = utils.list_all_modules()
        print(f"✓ list_all_modules(): {len(all_modules)} 个模块 - {', '.join(all_modules)}")

        # 测试列出模块函数
        data_functions = utils.list_module_functions('data')
        print(f"✓ list_module_functions('data'): {len(data_functions)} 个函数")

        # 测试获取版本信息
        version_info = utils.get_version_info()
        print(f"✓ get_version_info(): v{version_info['version']} by {version_info['author']}")

        # 测试获取模块函数
        get_dataset_func = utils.get_module_function('data', 'get_dataset')
        if get_dataset_func is not None:
            print("✓ get_module_function('data', 'get_dataset'): 成功获取函数")
        else:
            print("✗ get_module_function('data', 'get_dataset'): 获取失败")

        return True

    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始测试 utils 包的导入和基本功能")
    print("=" * 80)

    # 测试结果汇总
    test_results = {}

    # 执行各项测试
    test_results['basic_import'] = test_basic_import()
    test_results['module_imports'] = test_module_imports()
    test_results['function_usage'] = test_function_usage()
    test_results['quick_setup'] = test_quick_setup()
    test_results['module_utilities'] = test_module_utilities()

    # 汇总测试结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)

    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")

    print(f"\n总计: {passed_tests}/{total_tests} 项测试通过")

    if passed_tests == total_tests:
        print("🎉 所有测试通过！utils 包的导入和基本功能正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
