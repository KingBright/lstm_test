#!/usr/bin/env python3
"""
M2 Max优化版LSTM动力学系统模拟启动器
此脚本设置针对M2 Max的优化环境变量并启动主实验
"""

import os
import sys
import time
import torch
import numpy as np
import subprocess
import multiprocessing
from pathlib import Path

def setup_m2_max_environment():
    """配置针对M2 Max的环境变量"""
    print("正在配置M2 Max优化环境...")
    
    # 检测MPS可用性
    if not torch.backends.mps.is_available():
        print("警告: 此设备不支持MPS加速，将使用CPU运行。")
        return False
        
    # 设置Metal性能相关环境变量
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # 设置线程数等于可用CPU核心数
    cpu_count = multiprocessing.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    
    # 禁用不必要的警告
    os.environ['TORCH_WARN_ALWAYS_UNSAFE_USAGE'] = '0'
    
    print(f"✓ MPS加速已启用")
    print(f"✓ 使用 {cpu_count} 个CPU核心进行并行计算")
    return True

def verify_prerequisites():
    """验证所需的依赖和文件是否存在"""
    print("检查项目先决条件...")
    
    # 检查主要脚本是否存在
    required_files = [
        'main_experiment.py',
        'config.py',
        'data_generation.py',
        'data_preprocessing.py',
        'model.py',
        'training.py',
        'evaluation.py',
        'utils.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"错误: 找不到以下必需文件: {', '.join(missing_files)}")
        return False
    
    # 检查目录是否存在，如果不存在则创建
    required_dirs = ['models', 'figures']
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # 验证PyTorch MPS是否可用
    try:
        import torch
        if torch.backends.mps.is_available():
            print(f"✓ PyTorch {torch.__version__} 已安装，MPS可用")
        else:
            print(f"✓ PyTorch {torch.__version__} 已安装，但MPS不可用")
            print("  将使用CPU进行计算")
    except ImportError:
        print("错误: 未安装PyTorch")
        return False
    
    # 验证其他关键依赖
    try:
        import numpy
        import pandas
        import matplotlib
        import scipy
        print(f"✓ NumPy {numpy.__version__}, Pandas {pandas.__version__}, Matplotlib {matplotlib.__version__}, SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"错误: 缺少依赖项: {e}")
        return False
    
    return True

def run_experiment_with_performance_tracking():
    """运行实验并跟踪性能指标"""
    print("\n====================================================")
    print("     启动M2 Max优化版LSTM动力学系统模拟")
    print("====================================================\n")
    
    # 记录开始时间
    start_time = time.time()
    
    # 记录CPU和内存使用情况的初始值
    import psutil
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 测试修复后的Latin Hypercube采样函数
    print("测试Latin Hypercube采样函数...")
    try:
        from improved_data_generation import generate_latin_hypercube_samples
        test_samples_count = 5
        test_param_ranges = {
            'theta': [-np.pi, np.pi],
            'theta_dot': [-3.0, 3.0]
        }
        print(f"测试参数: samples_count={test_samples_count}, param_ranges={test_param_ranges}")
        test_params = generate_latin_hypercube_samples(test_samples_count, test_param_ranges)
        print(f"测试成功! 生成了 {len(test_params['theta'])} 个样本")
        print(f"样本theta值: {test_params['theta']}")
        print(f"样本theta_dot值: {test_params['theta_dot']}")
        print("Latin Hypercube采样测试成功!")
    except Exception as e:
        print(f"测试Latin Hypercube采样函数失败: {e}")
        import traceback
        traceback.print_exc()
        # 继续尝试运行实验
    
    try:
        # 导入并运行主实验
        print("导入主实验模块...")
        import main_experiment
        
        print("\n开始运行实验...")
        main_experiment.run_experiment()
        
        # 计算结束时间和资源使用情况
        end_time = time.time()
        execution_time = end_time - start_time
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # 打印性能摘要
        print("\n====================================================")
        print("     实验执行完成 - 性能摘要")
        print("====================================================")
        print(f"总执行时间: {execution_time:.2f} 秒")
        print(f"内存使用增长: {memory_used:.2f} MB")
        
        # 保存性能数据到日志文件
        with open('performance_log.txt', 'a') as log_file:
            from datetime import datetime
            log_file.write(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_file.write(f"执行时间: {execution_time:.2f} 秒\n")
            log_file.write(f"内存使用增长: {memory_used:.2f} MB\n")
            log_file.write("------------------------------------\n")
        
        return True
        
    except Exception as e:
        print(f"\n错误: 实验执行失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # 写入错误日志到文件
        with open('error_log.txt', 'w') as error_file:
            error_file.write(f"错误: {str(e)}\n")
            error_file.write(f"错误类型: {type(e).__name__}\n")
            traceback.print_exc(file=error_file)
        
        return False

def main():
    """主函数"""
    print("\n===== M2 Max优化版LSTM动力学系统模拟启动器 =====\n")
    
    # 验证先决条件
    if not verify_prerequisites():
        print("错误: 缺少必要的文件或依赖项，无法继续。")
        sys.exit(1)
    
    # 设置M2 Max优化环境
    setup_m2_max_environment()
    
    # 打印系统信息
    print("\n系统信息:")
    import platform
    print(f"Python版本: {platform.python_version()}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"处理器: {platform.processor()}")
    
    # 打印PyTorch信息
    print("\nPyTorch设备信息:")
    if torch.backends.mps.is_available():
        print("- MPS加速可用，将使用Metal性能着色器进行加速")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("- CUDA加速可用，将使用GPU加速")
        device = torch.device("cuda")
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
    else:
        print("- 使用CPU计算")
        device = torch.device("cpu")
    
    # 运行优化后的实验
    print("\n准备开始实验...")
    time.sleep(1)  # 短暂停顿，使用户有时间阅读信息
    success = run_experiment_with_performance_tracking()
    
    if success:
        print("\n✅ 实验成功完成!")
    else:
        print("\n❌ 实验执行失败。")
        sys.exit(1)

if __name__ == "__main__":
    main()
