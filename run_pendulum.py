#!/usr/bin/env python3
"""
执行M2 Max优化版动力学系统神经网络模拟演示脚本
使用: python run_pendulum.py
"""

import os
import sys
import time
import torch

# 设置环境变量优化Metal性能
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# 设置权重加载安全模式，解决FutureWarning问题
os.environ['TORCH_WARN_ALWAYS_UNSAFE_USAGE'] = '0'

# 检测是否在Apple Silicon上运行
def is_apple_silicon():
    import platform
    return platform.processor() == 'arm'

def check_mps_available():
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()

if __name__ == "__main__":
    print("=" * 70)
    print("开始执行M2 Max优化版动力学系统神经网络模拟演示...")
    print("=" * 70)
    
    # 检测硬件类型
    if is_apple_silicon():
        print("✅ 已检测到Apple Silicon处理器")
        
        if check_mps_available():
            print("✅ Metal Performance Shaders (MPS) 可用，将使用GPU加速")
        else:
            print("❗ MPS不可用，将回退到CPU计算")
            print("  提示: 请确保安装了支持MPS的PyTorch版本(2.0+)")
    else:
        print("⚠️ 未检测到Apple Silicon处理器，将使用标准计算方式")
    
    # 检测PyTorch版本
    torch_version = torch.__version__
    print(f"PyTorch版本: {torch_version}")
    
    # 检测可用CPU核心数
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    print(f"可用CPU核心数: {num_cores}")
    
    # 检测可用内存
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"系统内存: 总计 {mem.total / (1024**3):.2f} GB, 可用 {mem.available / (1024**3):.2f} GB")
    except ImportError:
        print("无法检测系统内存 (psutil库未安装)")
    
    print("-" * 70)
    print("系统检查完成，开始导入优化后的模块...")
    
    try:
        # 加载模型时防止序列化警告
        try:
            torch.serialization.add_safe_globals('pickle')
            torch.serialization.add_safe_globals('copy_reg')
            torch.serialization.add_safe_globals('_codecs')
        except AttributeError:
            print("注意: 您的PyTorch版本不支持add_safe_globals，将使用兼容模式")
            
        # 导入优化后的模块
        from pendulum_lstm import main
        
        # 开始计时
        start_time = time.time()
        
        # 执行主程序
        main()
        
        # 计算总运行时间
        total_time = time.time() - start_time
        
        print("=" * 70)
        print(f"演示完成！总运行时间: {total_time:.2f}秒")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)