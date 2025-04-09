#!/usr/bin/env python3
"""
验证M2 Max深度学习环境
这个脚本检查所有必要的依赖和MPS加速是否正常工作
"""

import sys
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time

def check_component(name, module):
    print(f"✓ {name:<15}: {module.__version__}")

def test_mps_performance():
    """
    测试MPS加速性能
    """
    # 检查MPS是否可用
    if not torch.backends.mps.is_available():
        print("⚠️ MPS不可用，跳过性能测试")
        return False
    
    # 创建测试张量
    size = 5000  # 矩阵大小
    
    # 在CPU上测试
    print(f"\n测试矩阵乘法 ({size}x{size})...")
    
    # CPU测试
    a_cpu = torch.rand(size, size)
    b_cpu = torch.rand(size, size)
    
    start = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"CPU时间: {cpu_time:.4f}秒")
    
    # MPS测试
    device = torch.device("mps")
    a_mps = a_cpu.to(device)
    b_mps = b_cpu.to(device)
    
    # 预热
    _ = torch.matmul(a_mps, b_mps)
    torch.mps.synchronize()
    
    start = time.time()
    c_mps = torch.matmul(a_mps, b_mps)
    torch.mps.synchronize()  # 确保计算完成
    mps_time = time.time() - start
    print(f"MPS时间: {mps_time:.4f}秒")
    
    # 计算加速比
    speedup = cpu_time / mps_time
    print(f"加速比: {speedup:.2f}x")
    
    # 验证结果是否相同
    max_diff = torch.max(torch.abs(c_cpu - c_mps.cpu())).item()
    print(f"最大误差: {max_diff:.6f}")
    
    return speedup > 1.5  # 期望至少有1.5倍加速

def test_lstm_example():
    """
    测试一个简单的LSTM模型，确保一切正常工作
    """
    print("\n测试简单LSTM模型...")
    
    # 生成一些随机数据
    seq_length = 10
    batch_size = 32
    input_size = 3
    hidden_size = 64
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_length, input_size)
    
    # 创建LSTM模型
    lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
    
    # 转移到MPS（如果可用）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = x.to(device)
        lstm = lstm.to(device)
    
    # 前向传播
    start = time.time()
    output, (h_n, c_n) = lstm(x)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    lstm_time = time.time() - start
    
    print(f"LSTM运行时间: {lstm_time:.4f}秒")
    print(f"输出形状: {output.shape}")
    
    return True

def main():
    print("====================================================")
    print("     M2 Max深度学习环境验证")
    print("====================================================")
    
    # 系统信息
    print(f"Python版本: {platform.python_version()}")
    print(f"系统: {platform.system()} {platform.release()}")
    print(f"处理器: {platform.processor()}")
    
    # 检查关键库
    check_component("NumPy", np)
    check_component("Pandas", pd)
    check_component("Matplotlib", plt)
    check_component("PyTorch", torch)
    
    # PyTorch MPS信息
    print(f"\nPyTorch MPS加速:")
    print(f"- MPS可用: {torch.backends.mps.is_available()}")
    print(f"- MPS已构建: {torch.backends.mps.is_built()}")
    
    # 性能测试
    mps_performance_ok = test_mps_performance()
    lstm_ok = test_lstm_example()
    
    # 总结
    print("\n====================================================")
    if mps_performance_ok and lstm_ok:
        print("✅ 环境验证成功! M2 Max优化正常工作")
    elif not torch.backends.mps.is_available():
        print("⚠️ 环境正常但MPS加速不可用")
    else:
        print("⚠️ 环境验证出现问题，请检查上述输出")
    print("====================================================")

if __name__ == "__main__":
    main()