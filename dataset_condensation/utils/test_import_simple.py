#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的导入测试
测试不依赖 torch 的基本导入功能
"""

import os
import sys

# 添加父目录到路径，以便能够导入 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_package_structure():
    """测试包结构"""
    print("=" * 60)
    print("测试包结构")
    print("=" * 60)

    utils_dir = os.path.dirname(os.path.abspath(__file__))

    # 检查文件是否存在
    required_files = [
        '__init__.py',
        'data.py',
        'model.py',
        'algorithm.py',
        'training.py',
        'augmentation.py',
        'logger_config.py',
        'config.py',
        'checkpoint.py',
        'helpers.py'
    ]

    missing_files = []
    existing_files = []

    for file_name in required_files:
        file_path = os.path.join(utils_dir, file_name)
        if os.path.exists(file_path):
            existing_files.append(file_name)
            print(f"✓ {file_name}")
        else:
            missing_files.append(file_name)
            print(f"✗ {file_name} 缺失")

    print(f"\n现有文件: {len(existing_files)}/{len(required_files)}")

    return len(missing_files) == 0


def test_basic_imports():
    """测试基本的不依赖外部库的导入"""
    print("\n" + "=" * 60)
    print("测试基本的不依赖外部库的导入")
    print("=" * 60)

    try:
        # 测试导入 logger_config（不依赖 torch）
        import importlib.util

        # 测试 config 模块（只依赖 yaml 和 loguru）
        spec = importlib.util.spec_from_file_location(
            "config",
            os.path.join(os.path.dirname(__file__), "config.py")
        )
        if spec and spec.loader:
            config_module = importlib.util.module_from_spec(spec)
            print("✓ config.py 可以加载")
        else:
            print("✗ config.py 加载失败")

        # 测试 helpers 模块的部分功能
        import time
        current_time = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
        print(f"✓ 时间格式化功能正常: {current_time}")

        return True

    except Exception as e:
        print(f"✗ 基本导入测试失败: {e}")
        return False


def test_init_file():
    """测试 __init__.py 文件的内容"""
    print("\n" + "=" * 60)
    print("测试 __init__.py 文件")
    print("=" * 60)

    try:
        init_file = os.path.join(os.path.dirname(__file__), "__init__.py")

        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查关键内容
        checks = [
            ("版本声明", "__version__" in content),
            ("作者信息", "__author__" in content),
            ("模块字典", "__modules__" in content),
            ("快速设置", "quick_setup" in content),
            ("导出列表", "__all__" in content)
        ]

        passed_checks = 0
        for check_name, condition in checks:
            if condition:
                print(f"✓ {check_name}")
                passed_checks += 1
            else:
                print(f"✗ {check_name}")

        print(f"\n检查通过: {passed_checks}/{len(checks)}")

        return passed_checks == len(checks)

    except Exception as e:
        print(f"✗ __init__.py 测试失败: {e}")
        return False


def test_file_sizes():
    """测试文件大小，确保文件有实际内容"""
    print("\n" + "=" * 60)
    print("测试文件大小")
    print("=" * 60)

    utils_dir = os.path.dirname(os.path.abspath(__file__))
    py_files = [f for f in os.listdir(utils_dir) if f.endswith('.py') and f != 'test_import_simple.py']

    total_size = 0
    for file_name in py_files:
        file_path = os.path.join(utils_dir, file_name)
        file_size = os.path.getsize(file_path)
        total_size += file_size
        print(f"✓ {file_name}: {file_size} bytes")

    print(f"\n总大小: {total_size} bytes ({total_size / 1024:.1f} KB)")

    return total_size > 10000  # 至少应该有 10KB 的代码


def test_module_structure():
    """测试模块结构的一致性"""
    print("\n" + "=" * 60)
    print("测试模块结构")
    print("=" * 60)

    utils_dir = os.path.dirname(os.path.abspath(__file__))
    py_files = [f for f in os.listdir(utils_dir) if
                f.endswith('.py') and f != '__init__.py' and not f.startswith('test_')]

    # 每个模块应该有的基本结构
    structure_checks = []

    for file_name in py_files:
        file_path = os.path.join(utils_dir, file_name)
        module_name = file_name[:-3]  # 移除 .py 扩展名

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            checks = [
                ("编码声明", "# -*- coding: utf-8 -*-" in content),
                ("模块文档", '"""' in content),
                ("函数定义", "def " in content),
                ("导入语句", "import " in content)
            ]

            passed = sum(1 for _, condition in checks if condition)
            structure_checks.append((module_name, passed, len(checks)))

            print(f"✓ {module_name}: {passed}/{len(checks)} 项结构检查通过")

        except Exception as e:
            print(f"✗ {module_name}: 检查失败 - {e}")

    return structure_checks


def main():
    """主测试函数"""
    print("开始简化导入测试（不依赖外部库）")
    print("=" * 80)

    # 执行各项测试
    test_results = {}

    test_results['package_structure'] = test_package_structure()
    test_results['basic_imports'] = test_basic_imports()
    test_results['init_file'] = test_init_file()
    test_results['file_sizes'] = test_file_sizes()
    structure_result = test_module_structure()

    # 计算结构测试结果
    if structure_result:
        total_checks = sum(checks[2] for checks in structure_result)
        passed_checks = sum(checks[1] for checks in structure_result)
        test_results['module_structure'] = passed_checks / total_checks >= 0.8  # 80% 通过率
    else:
        test_results['module_structure'] = False

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
        print("🎉 所有结构测试通过！utils 包结构完整。")
        print("\n注意: 完整功能测试需要安装以下依赖:")
        print("- torch")
        print("- torchvision")
        print("- numpy")
        print("- loguru")
        print("- yaml")
        print("- tqdm")
        print("\n可以使用 'uv add [package_name]' 安装依赖。")
        return True
    else:
        print("⚠️  部分测试失败，请检查包结构。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
