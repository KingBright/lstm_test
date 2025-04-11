#!/usr/bin/env python3
"""
快速测试脚本 (quick_test.py)
用于快速验证LSTM动力学系统模拟项目的所有核心组件是否正常工作
使用最小数据量和最少训练周期运行整个流程以检查代码是否有错误
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import config
import importlib
import traceback
from pathlib import Path

# 添加颜色打印功能，使输出更易读
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """打印带颜色的标题"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}    {text}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.END}")

def print_step(text):
    """打印步骤信息"""
    print(f"\n{Colors.BOLD}[测试] {text}...{Colors.END}")

def print_success(text):
    """打印成功信息"""
    print(f"{Colors.GREEN}{Colors.BOLD}[通过] {text}{Colors.END}")

def print_error(text):
    """打印错误信息"""
    print(f"{Colors.RED}{Colors.BOLD}[错误] {text}{Colors.END}")

def print_warning(text):
    """打印警告信息"""
    print(f"{Colors.YELLOW}{Colors.BOLD}[警告] {text}{Colors.END}")

def print_info(text):
    """打印普通信息"""
    print(f"       {text}")

def backup_config():
    """备份原始配置值"""
    original_values = {}
    
    # 检查 NUM_ICS_TO_RUN 是否存在 (对于长轨迹策略)
    if hasattr(config, 'NUM_ICS_TO_RUN'):
        original_values['NUM_ICS_TO_RUN'] = config.NUM_ICS_TO_RUN
    
    # 检查 NUM_SIMULATIONS 是否存在 (对于多场景策略)
    if hasattr(config, 'NUM_SIMULATIONS'):
        original_values['NUM_SIMULATIONS'] = config.NUM_SIMULATIONS
    
    # 通用配置值
    original_values['NUM_EPOCHS'] = config.NUM_EPOCHS
    original_values['FORCE_REGENERATE_DATA'] = config.FORCE_REGENERATE_DATA
    original_values['BATCH_SIZE'] = config.BATCH_SIZE
    original_values['EARLY_STOPPING_PATIENCE'] = config.EARLY_STOPPING_PATIENCE
    original_values['SCHEDULER_PATIENCE'] = config.SCHEDULER_PATIENCE
    
    # 长轨迹策略的持续时间
    if hasattr(config, 'SIMULATION_DURATION_LONG'):
        original_values['SIMULATION_DURATION_LONG'] = config.SIMULATION_DURATION_LONG
    
    return original_values

def modify_config_for_quick_test(original_values):
    """修改配置以进行快速测试"""
    print_step("修改配置以加速测试")
    
    # 减少模拟数量
    if hasattr(config, 'NUM_ICS_TO_RUN'):
        config.NUM_ICS_TO_RUN = min(2, config.NUM_ICS_TO_RUN)  # 最多2个初始条件
        print_info(f"NUM_ICS_TO_RUN: {config.NUM_ICS_TO_RUN} (原值: {original_values['NUM_ICS_TO_RUN']})")
    
    if hasattr(config, 'NUM_SIMULATIONS'):
        config.NUM_SIMULATIONS = min(2, config.NUM_SIMULATIONS)  # 最多2个模拟
        print_info(f"NUM_SIMULATIONS: {config.NUM_SIMULATIONS} (原值: {original_values['NUM_SIMULATIONS']})")
    
    # 减少训练周期
    config.NUM_EPOCHS = 3  # 只运行3个周期
    print_info(f"NUM_EPOCHS: {config.NUM_EPOCHS} (原值: {original_values['NUM_EPOCHS']})")
    
    # 强制重新生成数据
    config.FORCE_REGENERATE_DATA = True
    print_info(f"FORCE_REGENERATE_DATA: {config.FORCE_REGENERATE_DATA}")
    
    # 缩短模拟持续时间
    if hasattr(config, 'SIMULATION_DURATION_LONG'):
        config.SIMULATION_DURATION_LONG = min(10.0, config.SIMULATION_DURATION_LONG)  # 最多10秒
        config.T_SPAN_LONG = (0, config.SIMULATION_DURATION_LONG)
        print_info(f"SIMULATION_DURATION_LONG: {config.SIMULATION_DURATION_LONG} (原值: {original_values['SIMULATION_DURATION_LONG']})")
    
    # 减小批量大小
    config.BATCH_SIZE = min(32, config.BATCH_SIZE)  # 较小的批量大小以减少内存使用
    print_info(f"BATCH_SIZE: {config.BATCH_SIZE} (原值: {original_values['BATCH_SIZE']})")
    
    # 调整早停和调度器参数
    config.EARLY_STOPPING_PATIENCE = 10  # 确保不会在测试过程中提前停止
    config.SCHEDULER_PATIENCE = 5
    print_info(f"EARLY_STOPPING_PATIENCE: {config.EARLY_STOPPING_PATIENCE} (原值: {original_values['EARLY_STOPPING_PATIENCE']})")
    
    print_success("配置已修改以加速测试")
    return True

def restore_config(original_values):
    """恢复原始配置值"""
    for key, value in original_values.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # 特殊处理T_SPAN_LONG
    if hasattr(config, 'SIMULATION_DURATION_LONG') and hasattr(config, 'T_SPAN_LONG'):
        config.T_SPAN_LONG = (0, config.SIMULATION_DURATION_LONG)
    
    print_info("配置已恢复到原始值")
    return True

def check_imports():
    """检查所有必要的导入"""
    print_step("检查必要模块导入")
    all_imports = True
    
    modules_to_check = [
        ('数据生成模块', 'data_generation'),
        ('数据预处理模块', 'data_preprocessing'),
        ('模型定义模块', 'model'),
        ('训练模块', 'training'),
        ('评估模块', 'evaluation'),
        ('工具模块', 'utils'),
        ('主实验模块', 'main_experiment')
    ]
    
    for desc, module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            print_info(f"✓ {desc} ({module_name})")
        except ImportError as e:
            print_error(f"无法导入 {desc} ({module_name}): {e}")
            all_imports = False
    
    # 检查PyTorch MPS或CUDA
    if torch.backends.mps.is_available():
        print_info(f"✓ PyTorch MPS加速可用 (Apple M系列芯片)")
    elif torch.cuda.is_available():
        print_info(f"✓ PyTorch CUDA加速可用 ({torch.cuda.get_device_name(0)})")
    else:
        print_info("✓ PyTorch 将使用CPU计算")
    
    if all_imports:
        print_success("所有模块导入成功")
    else:
        print_error("部分模块导入失败")
    
    return all_imports

def check_directory_structure():
    """检查并创建必要的目录"""
    print_step("检查目录结构")
    
    required_dirs = ['models', 'figures']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print_info(f"创建目录: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            print_info(f"✓ 目录已存在: {directory}")
    
    print_success("目录结构已就绪")
    return True

def verify_config_compatibility():
    """验证配置参数与main_experiment.py的兼容性"""
    print_step("验证配置兼容性")
    
    # 检查NUM_ICS_TO_RUN参数存在性
    has_num_ics = hasattr(config, 'NUM_ICS_TO_RUN')
    if has_num_ics:
        print_info(f"✓ 找到NUM_ICS_TO_RUN参数: {config.NUM_ICS_TO_RUN}")
    else:
        print_warning("× 未找到NUM_ICS_TO_RUN参数，这可能会导致main_experiment无法正确运行")
        
    # 检查必要的参数
    required_params = [
        'SIMULATION_DURATION_LONG', 'T_SPAN_LONG', 'DT', 
        'NUM_EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE',
        'EARLY_STOPPING_PATIENCE', 'SCHEDULER_PATIENCE',
        'FORCE_REGENERATE_DATA'
    ]
    
    all_params_exist = True
    for param in required_params:
        if hasattr(config, param):
            print_info(f"✓ 找到{param}参数: {getattr(config, param)}")
        else:
            print_warning(f"× 未找到{param}参数，这可能会导致问题")
            all_params_exist = False
    
    # 检查路径相关参数
    path_params = [
        'COMBINED_DATA_FILE', 'MODEL_BEST_PATH', 'MODEL_FINAL_PATH',
        'INPUT_SCALER_PATH', 'TARGET_SCALER_PATH'
    ]
    
    for param in path_params:
        if hasattr(config, param):
            param_value = getattr(config, param)
            print_info(f"✓ 找到{param}参数: {param_value}")
            
            # 检查路径是否包含文件扩展名
            if not os.path.splitext(param_value)[1]:
                print_warning(f"  警告: {param}路径缺少文件扩展名")
        else:
            print_warning(f"× 未找到{param}参数，这可能会导致问题")
            all_params_exist = False
    
    if all_params_exist:
        print_success("配置参数与main_experiment.py兼容")
    else:
        print_warning("配置参数可能与main_experiment.py不完全兼容，测试可能会失败")
    
    return all_params_exist

def clean_up_test_files():
    """清理快速测试生成的文件"""
    print_step("清理测试文件")
    
    # 确保output_files_to_check列表中的文件路径都是正确的
    output_files_to_check = [
        config.COMBINED_DATA_FILE,
        config.MODEL_BEST_PATH,
        config.MODEL_FINAL_PATH,
        config.INPUT_SCALER_PATH,
        config.TARGET_SCALER_PATH
    ]
    
    # 添加可能生成的特定文件
    if hasattr(config, 'OUTPUT_SCALER_PATH'):
        output_files_to_check.append(config.OUTPUT_SCALER_PATH)
    
    # 清理所有生成的文件
    for f_path in output_files_to_check:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                print_info(f"已删除: {f_path}")
            except OSError as e:
                print_warning(f"删除文件时出错 {f_path}: {e}")
    
    print_success("测试文件清理完成")
    return True

def run_main_experiment():
    """运行主实验流程"""
    print_step("运行主实验流程")
    
    try:
        import main_experiment
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行主实验
        main_experiment.run_experiment()
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        
        print_success(f"主实验流程完成! 耗时: {total_time:.2f} 秒")
        
        # 检查是否生成了模型文件
        if (os.path.exists(config.MODEL_BEST_PATH) or 
            os.path.exists(config.MODEL_FINAL_PATH)):
            print_info(f"✓ 成功生成模型文件")
        else:
            print_warning(f"未找到模型文件，但流程已完成")
        
        return True
    except Exception as e:
        print_error(f"主实验流程失败: {e}")
        traceback.print_exc()
        return False

def verify_output_files():
    """验证是否生成了所有必要的输出文件"""
    print_step("验证输出文件")
    
    # 检查模型文件
    model_files_exist = (os.path.exists(config.MODEL_BEST_PATH) or 
                         os.path.exists(config.MODEL_FINAL_PATH))
    
    # 检查缩放器文件
    scaler_files_exist = os.path.exists(config.INPUT_SCALER_PATH)
    
    # 检查数据文件
    data_file_exists = os.path.exists(config.COMBINED_DATA_FILE)
    
    # 检查图表文件
    figure_files = list(Path('figures').glob('*.png'))
    figure_files_exist = len(figure_files) > 0
    
    all_files_exist = model_files_exist and scaler_files_exist and data_file_exists and figure_files_exist
    
    if all_files_exist:
        print_success("所有必要的输出文件均已生成")
    else:
        print_warning("部分输出文件可能未生成:")
        if not model_files_exist:
            print_info("✗ 模型文件未找到")
        if not scaler_files_exist:
            print_info("✗ 缩放器文件未找到")
        if not data_file_exists:
            print_info("✗ 数据文件未找到")
        if not figure_files_exist:
            print_info("✗ 图表文件未找到")
    
    return all_files_exist

def main():
    """主函数"""
    print_header("LSTM动力学系统模拟 - 快速测试")
    print("此脚本将修改配置以进行快速测试（少量数据，少量训练周期）")
    print("目的是验证整个流程是否能够正常工作，而不是训练有效模型")
    
    # 记录总体开始时间
    overall_start_time = time.time()
    
    # 检查导入
    if not check_imports():
        print_error("无法继续：模块导入失败")
        return False
    
    # 验证配置参数兼容性
    verify_config_compatibility()
    
    # 检查目录结构
    check_directory_structure()
    
    # 备份原始配置
    original_values = backup_config()
    
    try:
        # 修改配置以加速测试
        modify_config_for_quick_test(original_values)
        
        # 执行主实验
        success = run_main_experiment()
        
        # 验证输出文件
        if success:
            verify_output_files()
        
    except Exception as e:
        print_error(f"测试过程中发生未捕获的错误: {e}")
        traceback.print_exc()
        success = False
    
    finally:
        # 恢复原始配置
        restore_config(original_values)
        
        # 根据命令行参数决定是否清理文件
        if "--keep-files" not in sys.argv:
            clean_up_test_files()
        else:
            print_info("保留测试文件以供检查 (使用了 --keep-files 参数)")
    
    # 计算总耗时
    overall_time = time.time() - overall_start_time
    
    # 打印最终结果
    print_header("快速测试结果")
    print(f"总耗时: {overall_time:.2f} 秒")
    
    if success:
        print_success("快速测试完成！整个流程顺利执行，可以运行完整实验")
    else:
        print_error("快速测试失败！请修复上述错误后再运行完整实验")
    
    return success

if __name__ == "__main__":
    # 命令行参数
    # --keep-files: 测试后保留生成的文件
    # --verbose: 启用详细输出
    if "--verbose" in sys.argv:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    success = main()
    sys.exit(0 if success else 1)

