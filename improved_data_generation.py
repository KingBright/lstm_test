# improved_data_generation.py

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import qmc
import matplotlib.pyplot as plt
import os
import joblib
from collections import defaultdict
import time

# Import original data generation functions for compatibility
from data_generation import (
    PendulumSystem, set_global_torque_params, get_tau_at_time_global,
    run_simulation, rk4_step, generate_simulation_data
)
import config

# --- 优化的Latin Hypercube Sampling Function ---
def generate_latin_hypercube_samples(num_samples, param_ranges):
    """
    使用优化的拉丁超立方体抽样生成更均匀分布的参数组合
    
    Args:
        num_samples: 需要生成的样本数量
        param_ranges: 字典，包含每个参数的范围，例如 {'theta': [-np.pi, np.pi], ...}
    
    Returns:
        params_dict: 字典，包含每个参数的样本数组
    """
    # 获取参数数量
    n_dims = len(param_ranges)
    param_names = list(param_ranges.keys())
    
    # 创建优化的拉丁超立方体采样器 (使用 'lloyd' 或 'random-cd')
    # 仅在维度大于1时使用优化，因为Lloyd优化需要至少2维
    optimization_method = "lloyd" if n_dims > 1 else None
    sampler = qmc.LatinHypercube(d=n_dims, optimization=optimization_method, seed=config.SEED) # 添加seed确保可复现
    
    # 直接生成所需数量的样本
    samples = sampler.random(n=num_samples)
    best_quality = float('nan') # 初始化质量指标

    # (可选) 计算并打印离散度，但不用于选择样本
    if num_samples > 1: # 离散度至少需要2个点
        try:
            # 注意: 使用 samples 而不是 best_samples
            best_quality = qmc.discrepancy(samples)
        except Exception as e: # 更具体的异常捕获
            print(f"计算离散度时出错: {e}")
            best_quality = float('nan')
    
    # 确保samples是正确形状的2D数组
    # samples = np.array(samples) # samples已经是numpy数组
    if n_dims == 1:
        samples = np.atleast_2d(samples).T  # 转置确保形状是 (n, 1)
    
    print(f"拉丁超立方体样本离散度: {best_quality:.6f} (越低越均匀)")
    
    # 将样本映射到参数实际范围
    params_dict = {}
    for i, param_name in enumerate(param_names):
        low, high = param_ranges[param_name]
        
        if n_dims == 1:
            # 对于1维情况, samples已经确保是2D的
            params_dict[param_name] = qmc.scale(samples, low, high).flatten()
        else:
            # 对于多维情况，取出对应列并保持2D形状
            column = samples[:, i:i+1]  # 保持2D形状
            params_dict[param_name] = qmc.scale(column, low, high).flatten()
    
    return params_dict

# --- 力矩序列生成的改进方法 ---
# --- 优化的力矩序列生成方法 ---
def generate_diverse_torque_patterns(num_sequences, sequence_length, dt, torque_range=config.TORQUE_RANGE):
    """
    生成分布更均匀的多样化力矩模式
    
    Args:
        num_sequences: 需要生成的序列数量
        sequence_length: 每个序列的长度
        dt: 时间步长
        torque_range: 力矩范围 [min, max]
    
    Returns:
        torque_patterns: 数组，形状为 (num_sequences, sequence_length)
    """
    torque_patterns = []
    
    # 定义可能的模式类型
    pattern_types = [
        'constant',     # 恒定力矩
        'sinusoidal',   # 正弦力矩
        'step',         # 阶跃力矩
        'ramp',         # 斜坡力矩
        'oscillating',  # 振荡力矩（振幅变化）
        'random_walk',  # 随机游走力矩
        'mixed',        # 混合型
        'stratified'    # 分层均匀力矩
    ]
    
    # 更均衡的模式权重分配
    pattern_weights = [0.1, 0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.15]
    
    # 创建更均匀的力矩分布，确保不同力矩范围都有足够覆盖
    torque_min, torque_max = torque_range
    torque_range_size = torque_max - torque_min
    
    # 使用分层采样方法确保力矩值覆盖整个范围
    num_strata = 10  # 将力矩范围分成10个区间
    strata_size = torque_range_size / num_strata
    
    # 生成更多的候选模式，然后选择最均匀的子集
    oversample_factor = 2.0
    num_candidates = int(num_sequences * oversample_factor)
    candidate_patterns = []
    candidate_histograms = []
    
    # 用于分析均匀性的直方图箱体
    hist_bins = np.linspace(torque_min, torque_max, 20)
    
    # 准备频率和幅值的均匀分布
    frequencies = np.linspace(0.05, 2.5, 20)
    amplitudes = np.linspace(0.1, 0.8, 10)
    
    # 随机选择每个候选序列的模式类型
    pattern_choices = np.random.choice(
        pattern_types, 
        size=num_candidates, 
        p=pattern_weights
    )
    
    # 创建时间向量
    t = np.arange(sequence_length) * dt
    
    # 为每个候选序列生成力矩模式
    for idx, pattern_type in enumerate(pattern_choices):
        # 选择一个力矩区间，确保均匀覆盖
        stratum_idx = idx % num_strata
        stratum_min = torque_min + stratum_idx * strata_size
        stratum_max = stratum_min + strata_size
        
        if pattern_type == 'constant':
            # 使用分层均匀采样的恒定力矩
            # 为了更好的覆盖，每个区间取多个值
            sub_idx = (idx // num_strata) % 5
            sub_strata_size = strata_size / 5
            torque_value = stratum_min + (sub_idx + 0.5) * sub_strata_size
            torque = torque_value * np.ones(sequence_length)
            
        elif pattern_type == 'sinusoidal':
            # 优化的正弦波力矩 - 确保频率和幅值均匀分布
            freq_idx = idx % len(frequencies)
            amp_idx = (idx // len(frequencies)) % len(amplitudes)
            
            frequency = frequencies[freq_idx]
            # 振幅限制在区间范围内
            amplitude = min(amplitudes[amp_idx], strata_size * 0.8)
            phase = np.random.uniform(0, 2*np.pi)
            
            # 中心点定位在选定区间内
            center = stratum_min + strata_size/2
            torque = center + amplitude * np.sin(2*np.pi*frequency*t + phase)
            
        elif pattern_type == 'step':
            # 优化的阶跃力矩生成 - 更均匀的阶跃分布
            torque = np.zeros(sequence_length)
            
            # 创建均匀分布的阶跃点
            num_steps = min(4, max(2, int(sequence_length / 100)))
            step_positions = np.linspace(10, sequence_length-10, num_steps+1)[:-1]
            # 添加小随机扰动
            step_points = [int(pos + np.random.randint(-5, 6)) for pos in step_positions]
            step_points = sorted([p for p in step_points if 10 <= p < sequence_length-10])
            
            # 确保阶跃值覆盖整个力矩范围
            segments = len(step_points) + 1
            
            # 使用分层采样确保覆盖全部区间
            segment_values = [stratum_min + strata_size/2]
            strata_indices = [(stratum_idx + i*3) % num_strata for i in range(segments-1)]
            for s_idx in strata_indices:
                s_min = torque_min + s_idx * strata_size
                segment_values.append(s_min + strata_size/2)
            
            # 打乱顺序增加多样性
            np.random.shuffle(segment_values)
            
            # 应用阶跃序列
            last_idx = 0
            for i, step_point in enumerate(step_points):
                torque[last_idx:step_point] = segment_values[i]
                last_idx = step_point
            
            torque[last_idx:] = segment_values[-1]
            
        elif pattern_type == 'ramp':
            # 优化的斜坡力矩 - 确保起点和终点分布更均匀
            # 让斜坡跨越不同的区间以增加覆盖范围
            start_stratum = stratum_idx
            end_stratum = (start_stratum + 5) % num_strata
            
            start_value = torque_min + start_stratum * strata_size + strata_size/2
            end_value = torque_min + end_stratum * strata_size + strata_size/2
            
            # 如果是偶数索引，交换起点和终点以得到上升和下降的斜坡
            if idx % 2 == 0:
                start_value, end_value = end_value, start_value
                
            torque = np.linspace(start_value, end_value, sequence_length)
            
        elif pattern_type == 'oscillating':
            # 优化的振荡力矩模式
            amplitude_patterns = ['increasing', 'decreasing', 'peak', 'valley']
            amp_pattern = amplitude_patterns[idx % len(amplitude_patterns)]
            
            # 均匀采样频率
            freq_idx = idx % len(frequencies)
            base_freq = frequencies[freq_idx]
            
            # 振幅在区间内变化
            max_amplitude = min(strata_size * 0.8, torque_range_size * 0.3)
            min_amplitude = max_amplitude * 0.1
            
            # 创建不同的振幅变化模式
            if amp_pattern == 'increasing':
                amplitude = np.linspace(min_amplitude, max_amplitude, sequence_length)
            elif amp_pattern == 'decreasing':
                amplitude = np.linspace(max_amplitude, min_amplitude, sequence_length)
            elif amp_pattern == 'peak':
                half_seq = sequence_length // 2
                amplitude = np.concatenate([
                    np.linspace(min_amplitude, max_amplitude, half_seq),
                    np.linspace(max_amplitude, min_amplitude, sequence_length - half_seq)
                ])
            else:  # 'valley'
                half_seq = sequence_length // 2
                amplitude = np.concatenate([
                    np.linspace(max_amplitude, min_amplitude, half_seq),
                    np.linspace(min_amplitude, max_amplitude, sequence_length - half_seq)
                ])
            
            # 均匀覆盖区间中心
            center = stratum_min + strata_size/2
            torque = center + amplitude * np.sin(2*np.pi*base_freq*t)
            
        elif pattern_type == 'random_walk':
            # 优化的随机游走模式 - 受控布朗运动
            torque = np.zeros(sequence_length)
            
            # 从区间中点开始
            torque[0] = stratum_min + strata_size/2
            
            # 适当的步长
            step_size = strata_size * 0.05
            
            # 添加偏移以确保更好的覆盖
            if idx % 2 == 0:
                # 向上漂移
                drift = np.linspace(0, strata_size * 0.5, sequence_length)
            else:
                # 向下漂移
                drift = np.linspace(0, -strata_size * 0.5, sequence_length)
            
            # 生成随机游走
            for i in range(1, sequence_length):
                torque[i] = torque[i-1] + np.random.normal(0, step_size) + (drift[i] - drift[i-1])
            
            # 如果发现随机游走跑出范围太多，应用软约束
            torque = np.clip(torque, torque_min, torque_max)
            
        elif pattern_type == 'mixed':
            # 优化的混合模式 - 确保平滑过渡
            torque = np.zeros(sequence_length)
            
            # 将序列均匀分段
            num_segments = 3 + (idx % 3)  # 3-5段
            segment_length = sequence_length // num_segments
            segment_points = [i * segment_length for i in range(1, num_segments)]
            segment_points = [0] + segment_points + [sequence_length]
            
            # 确保每个混合段都覆盖不同区间
            segment_centers = []
            for i in range(num_segments):
                s_idx = (stratum_idx + i * 2) % num_strata
                center = torque_min + s_idx * strata_size + strata_size/2
                segment_centers.append(center)
            
            # 顺序使用不同模式类型
            segment_types = ['constant', 'sinusoidal', 'ramp', 'oscillating', 'random_walk']
            
            # 确保平滑过渡
            for i in range(num_segments):
                start_idx = segment_points[i]
                end_idx = segment_points[i+1]
                seg_length = end_idx - start_idx
                
                t_segment = np.arange(seg_length) * dt
                segment_type = segment_types[i % len(segment_types)]
                center = segment_centers[i]
                
                # 如果不是第一段，确保与前一段的结束值匹配
                if i > 0:
                    start_val = torque[start_idx-1]
                else:
                    start_val = center
                
                if segment_type == 'constant':
                    segment_torque = center * np.ones(seg_length)
                    # 创建从上一个值到目标值的平滑过渡
                    if i > 0:
                        transition_len = min(20, seg_length // 4)
                        transition = np.linspace(start_val, center, transition_len)
                        segment_torque[:transition_len] = transition
                
                elif segment_type == 'sinusoidal':
                    amp = min(strata_size * 0.4, 0.2)
                    freq = frequencies[i % len(frequencies)]
                    
                    # 确保起点与上一段终点匹配
                    if i > 0:
                        # 计算相位调整使起点值匹配
                        phase_adjust = np.arcsin((start_val - center) / (amp + 1e-10))
                        segment_torque = center + amp * np.sin(2*np.pi*freq*t_segment + phase_adjust)
                    else:
                        segment_torque = center + amp * np.sin(2*np.pi*freq*t_segment)
                
                elif segment_type == 'ramp':
                    end_val = segment_centers[(i+1) % num_segments]
                    segment_torque = np.linspace(start_val, end_val, seg_length)
                
                elif segment_type == 'oscillating':
                    amp = np.linspace(0.05, strata_size * 0.3, seg_length)
                    freq = frequencies[i % len(frequencies)]
                    
                    # 确保起点匹配
                    if i > 0:
                        phase_adjust = np.arcsin((start_val - center) / (amp[0] + 1e-10))
                        segment_torque = center + amp * np.sin(2*np.pi*freq*t_segment + phase_adjust)
                    else:
                        segment_torque = center + amp * np.sin(2*np.pi*freq*t_segment)
                
                elif segment_type == 'random_walk':
                    segment_torque = np.zeros(seg_length)
                    segment_torque[0] = start_val
                    
                    step_size = strata_size * 0.03
                    for j in range(1, seg_length):
                        segment_torque[j] = segment_torque[j-1] + np.random.normal(0, step_size)
                    
                    # 轻微中心化趋势
                    return_force = 0.02
                    for j in range(1, seg_length):
                        segment_torque[j] += return_force * (center - segment_torque[j-1])
                
                # 应用到主序列
                torque[start_idx:end_idx] = segment_torque
        
        elif pattern_type == 'stratified':
            # 专门设计的均匀覆盖模式
            torque = np.zeros(sequence_length)
            
            # 创建一个包含所有区间的序列
            strata_centers = np.linspace(
                torque_min + strata_size/2,
                torque_max - strata_size/2,
                num_strata
            )
            
            # 决定使用什么方式均匀覆盖
            coverage_method = idx % 3
            
            if coverage_method == 0:
                # 均匀时间分配法
                # 把时间序列平均分配到每个区间
                points_per_stratum = sequence_length // num_strata
                for i in range(num_strata):
                    start_idx = i * points_per_stratum
                    end_idx = start_idx + points_per_stratum if i < num_strata-1 else sequence_length
                    torque[start_idx:end_idx] = strata_centers[i]
                    
                # 添加随机序列确保平滑过渡
                for i in range(1, num_strata):
                    transition_idx = i * points_per_stratum
                    transition_len = min(10, points_per_stratum // 2)
                    if transition_idx + transition_len <= sequence_length:
                        transition = np.linspace(
                            strata_centers[i-1], strata_centers[i], transition_len
                        )
                        torque[transition_idx-transition_len//2:transition_idx+transition_len//2] = transition
            
            elif coverage_method == 1:
                # 正弦扫描法
                # 使用低频正弦波扫描整个力矩范围
                sweep_cycles = 1 + idx % 3  # 1-3个完整周期
                t_normalized = np.linspace(0, sweep_cycles, sequence_length)
                
                # 计算区间索引（连续）
                continuous_idx = (np.sin(2*np.pi*t_normalized) + 1) / 2 * (num_strata - 1)
                
                # 插值得到对应的力矩值
                for i in range(sequence_length):
                    idx_low = int(np.floor(continuous_idx[i]))
                    idx_high = int(np.ceil(continuous_idx[i]))
                    if idx_high >= num_strata:
                        idx_high = num_strata - 1
                    
                    # 线性插值
                    if idx_low == idx_high:
                        torque[i] = strata_centers[idx_low]
                    else:
                        fraction = continuous_idx[i] - idx_low
                        torque[i] = strata_centers[idx_low] * (1-fraction) + strata_centers[idx_high] * fraction
            
            else:
                # 分层重复序列法
                # 创建一个包含所有区间值的重复序列
                repeats = sequence_length // num_strata + 1
                expanded_centers = np.repeat(strata_centers, repeats)[:sequence_length]
                
                # 根据索引选择不同的排列方式
                arrangement = (idx // 3) % 4
                if arrangement == 0:
                    # 原始顺序
                    torque = expanded_centers
                elif arrangement == 1:
                    # 反向
                    torque = expanded_centers[::-1]
                elif arrangement == 2:
                    # 中间开始向两边
                    half_strata = num_strata // 2
                    middle_out = list(range(half_strata, num_strata)) + list(range(half_strata-1, -1, -1))
                    strata_arrangement = [strata_centers[i] for i in middle_out]
                    expanded = np.repeat(strata_arrangement, repeats)[:sequence_length]
                    torque = expanded
                else:
                    # 两边向中间
                    half_strata = num_strata // 2
                    edges_in = list(range(0, half_strata)) + list(range(num_strata-1, half_strata-1, -1))
                    strata_arrangement = [strata_centers[i] for i in edges_in]
                    expanded = np.repeat(strata_arrangement, repeats)[:sequence_length]
                    torque = expanded
                    
                # 使用Savgol滤波器对锯齿形波形进行平滑处理
                try:
                    from scipy.signal import savgol_filter
                    window_length = min(15, sequence_length//10*2+1) # 确保奇数
                    window_length = max(3, window_length) # 确保至少为3
                    torque = savgol_filter(torque, window_length=window_length, polyorder=3)
                except:
                    # 如果scipy不可用，使用简单的移动平均平滑
                    kernel_size = 5
                    kernel = np.ones(kernel_size) / kernel_size
                    # 使用numpy的卷积操作
                    padded = np.pad(torque, (kernel_size//2, kernel_size//2), mode='edge')
                    torque = np.convolve(padded, kernel, mode='valid')
        
        # 对所有力矩值应用小幅度随机噪声增加多样性
        # 使用更小的噪声幅度以保持区间特性
        noise_scale = strata_size * 0.1
        noise = np.random.normal(0, noise_scale, sequence_length)
        torque += noise
        
        # 确保所有值都在整体力矩范围内
        torque = np.clip(torque, torque_min, torque_max)
        
        # 保存候选模式
        candidate_patterns.append(torque)
        
        # 计算这个模式的直方图分布
        hist, _ = np.histogram(torque, bins=hist_bins)
        candidate_histograms.append(hist)
    
    # 如果生成了足够的候选模式，选择最均匀的子集
    if len(candidate_patterns) > num_sequences:
        # 计算当前的总体直方图
        combined_hist = np.zeros_like(candidate_histograms[0])
        selected_indices = []
        
        # 贪婪选择能最大化均匀性的模式
        while len(selected_indices) < num_sequences:
            best_cv = float('inf')
            best_idx = -1
            
            # 评估添加每个候选模式后的均匀性
            for i in range(len(candidate_patterns)):
                if i in selected_indices:
                    continue
                
                # 临时添加这个候选模式的直方图
                temp_hist = combined_hist + candidate_histograms[i]
                # 计算变异系数
                cv = np.std(temp_hist) / (np.mean(temp_hist) + 1e-10)
                
                if cv < best_cv:
                    best_cv = cv
                    best_idx = i
            
            # 添加最佳候选
            if best_idx != -1:
                selected_indices.append(best_idx)
                combined_hist += candidate_histograms[best_idx]
            else:
                break
        
        # 使用选择的模式
        torque_patterns = [candidate_patterns[i] for i in selected_indices]
    else:
        # 直接使用所有候选模式
        torque_patterns = candidate_patterns
    
    # 计算最终的力矩模式分布统计
    all_values = np.concatenate(torque_patterns)
    hist, _ = np.histogram(all_values, bins=hist_bins)
    cv = np.std(hist) / (np.mean(hist) + 1e-10)
    uniformity = 100 / (1 + cv)
    print(f"力矩模式均匀度: {uniformity:.2f}% (变异系数: {cv:.4f}, 越高越均匀)")
    
    return np.array(torque_patterns)


# --- 使用优化的物理能量级别采样改进的数据生成 ---
def generate_physics_informed_samples(num_samples, pendulum, energy_levels=None):
    theta_samples = []
    torque_samples = []
    """
    使用优化的物理能量级别生成更均匀分布的初始条件采样
    
    Args:
        num_samples: 需要生成的样本数量
        pendulum: PendulumSystem实例
        energy_levels: 可选，能量级别列表；如果为None则自动计算
        
    Returns:
        theta_samples, theta_dot_samples: 满足指定能量级别的角度和角速度采样
    """
    m = pendulum.m  # 质量
    L = pendulum.L  # 长度
    g = pendulum.g  # 重力加速度
    
    # 如果未指定能量级别，则计算更细化的范围
    if energy_levels is None:
        # 计算物理上有意义的能量范围
        min_energy = -m * g * L  # 底部位置的势能
        max_energy = 3 * m * g * L  # 允许足够的动能使摆能够旋转
        
        # 使用线性间隔生成能量级别，以获得更均匀的分布
        num_levels = 50  # 能量级别数量
        # 使用 linspace 替代 geomspace
        energy_levels_linear = np.linspace(min_energy, max_energy, num_levels)
        # 保留原始的 normalized_range 计算方式，但基于线性间隔 (虽然这里可以直接用 energy_levels_linear)
        # 为了最小化改动，我们暂时保留这种结构，但可以直接使用 energy_levels_linear
        # normalized_range = (energy_levels_linear - energy_levels_linear.min()) / (energy_levels_linear.max() - energy_levels_linear.min())
        # energy_levels = min_energy + normalized_range * (max_energy - min_energy)
        energy_levels = energy_levels_linear # 直接使用线性间隔的能量级别
        
        # 添加一些特殊能量级别，确保覆盖关键状态
        special_energies = [
            -0.8 * m * g * L,  # 接近底部静止
            -0.5 * m * g * L,  # 小振幅振荡
            0,                # 能够达到水平位置
            0.5 * m * g * L,   # 中等旋转能力
            1.5 * m * g * L,   # 可以完全旋转
            2.5 * m * g * L    # 高能旋转
        ]
        energy_levels = np.unique(np.concatenate([energy_levels, special_energies]))
    
    print(f"使用 {len(energy_levels)} 个能量级别进行物理采样")
    
    # 将样本均匀分配给每个能量级别
    num_levels = len(energy_levels)
    samples_per_level = num_samples // num_levels
    remainder_samples = num_samples % num_levels # 处理不能整除的情况

    print(f"每个能量级别分配 {samples_per_level} 个样本，剩余 {remainder_samples} 个样本随机分配")

    all_energy_indices = []
    all_theta_samples_for_levels = []

    # 为每个能量级别生成角度样本
    for level_idx in range(num_levels):
        num_level_samples = samples_per_level + (1 if level_idx < remainder_samples else 0)
        if num_level_samples == 0:
            continue
            
        # 为当前能量级别生成角度样本 (使用LHS保证在该级别内均匀)
        # 限制角度采样范围，避免采样物理上不可行的角度
        energy_ratio = energy_levels[level_idx] / (m * g * L)
        energy_ratio = np.clip(energy_ratio, -1, 1)
        max_theta = np.arccos(energy_ratio)  # 理论最大角度
        theta_range = min(max_theta, np.pi) # 确保不超过pi
        lower_bound = max(-theta_range, -np.pi/3)
        upper_bound = min(theta_range, np.pi/3)
        energy_ratio = energy_levels[level_idx] / (m * g * L)
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        lower_bound = min(lower_bound, upper_bound) # Ensure lower_bound <= upper_bound
        if lower_bound >= upper_bound:
            upper_bound = lower_bound + 1e-6  # 确保 upper_bound 稍微大于 lower_bound
        theta_level_lhs = generate_latin_hypercube_samples(num_level_samples, {'theta': [lower_bound, upper_bound]})['theta']
        all_theta_samples_for_levels.extend(theta_level_lhs)
        all_energy_indices.extend([level_idx] * num_level_samples)
        
    # 打乱顺序，避免样本按能量级别排序
    shuffle_indices = np.random.permutation(num_samples)
    energy_indices = np.array(all_energy_indices)[shuffle_indices]
    theta_lhs = np.array(all_theta_samples_for_levels)[shuffle_indices]
    
    # 按分配好的能量和角度计算角速度
    for i in range(num_samples):
        theta = theta_lhs[i] # 使用预先生成并打乱的LHS角度
        energy_idx = energy_indices[i] # 使用预先分配并打乱的能量索引
        energy = energy_levels[energy_idx]
        
        # 计算对应能量水平的角速度
        # E = 0.5*m*L^2*theta_dot^2 - m*g*L*cos(theta)
        # 解得: theta_dot = sqrt(2*(E + m*g*L*cos(theta))/(m*L^2))
        term = 2 * (energy + m * g * L * np.cos(theta)) / (m * L * L)
        
        # 如果能量-位置组合在物理上不可行，重新选择角度或能量
        attempts = 0
        while term < 0 and attempts < 10:
            # 尝试调整角度或能量
            if np.random.random() > 0.5:
                # 调整角度
                theta = np.random.uniform(-np.pi, np.pi)
            else:
                # 调整能量
                # 调整能量：在当前能量级别附近随机选择一个新级别
                # 避免完全随机跳跃，尝试保持一定的能量连续性
                new_energy_idx = np.random.choice(len(energy_levels)) # 简单起见，还是随机选一个
                energy = energy_levels[new_energy_idx]
            
            term = 2 * (energy + m * g * L * np.cos(theta)) / (m * L * L)
            attempts += 1
        
        # 如果还是不可行，使用随机值
        if term < 0:
            theta = np.random.uniform(-np.pi/3, np.pi/3)
            theta_dot = np.random.uniform(-0.1, 0.1)
        else:
            theta_dot = np.sqrt(term)
            # 随机决定速度方向
            if np.random.choice([-1, 1]) == -1:
                theta_dot = -theta_dot
        
    theta_dot_samples = []
    torque_samples = []
    
    torque_samples = np.array([torque_samples] * len(theta_samples))
    
    return np.array(theta_samples), np.array(theta_dot_samples), np.array(torque_samples)
    
    # 确保物理采样的相空间覆盖
    # 我们使用网格来检查覆盖情况
    grid_size = 10
    theta_bins = np.linspace(-np.pi, np.pi, grid_size+1)
    theta_dot_bins = np.linspace(-3.0, 3.0, grid_size+1)
    phase_space_grid = np.zeros((grid_size, grid_size))
    
    # 计算当前样本在相空间的分布
    for theta, theta_dot in zip(theta_samples, theta_dot_samples):
        i = min(grid_size-1, max(0, int((theta + np.pi) / (2*np.pi) * grid_size)))
        j = min(grid_size-1, max(0, int((theta_dot + 3.0) / 6.0 * grid_size)))
        phase_space_grid[i, j] += 1
    
    # 识别空白区域
    empty_cells = np.where(phase_space_grid == 0)
    if len(empty_cells[0]) > 0:
        print(f"检测到 {len(empty_cells[0])} 个空白区域，添加额外样本")
        
        # 添加额外样本到空白区域
        for i, j in zip(empty_cells[0], empty_cells[1]):
            theta = theta_bins[i] + (theta_bins[i+1] - theta_bins[i]) * np.random.random()
            theta = max(min(theta, np.pi/3), -np.pi/3) # 限制角度范围
            theta_dot = theta_dot_bins[j] + (theta_dot_bins[j+1] - theta_dot_bins[j]) * np.random.random()
            theta_dot = max(min(theta_dot, 0.1), -0.1) # 限制角速度范围
            
            # 确保不超过最大样本数
            if len(theta_samples) < num_samples * 1.2:
                
                torque_samples = np.append(torque_samples, np.random.uniform(config.TORQUE_RANGE[0], config.TORQUE_RANGE[1]))
                theta_samples.append(theta)
                theta_dot_samples.append(theta_dot)
    
    # 如果样本过多，随机采样到指定数量
    if len(theta_samples) > num_samples:
        indices = np.random.choice(len(theta_samples), num_samples, replace=False)
        theta_samples = [theta_samples[i] for i in indices]
        theta_dot_samples = [theta_dot_samples[i] for i in indices]
    
    # 如果样本太少，补充随机样本
    remaining = num_samples - len(theta_samples)
    if remaining > 0:
        print(f"通过随机采样添加 {remaining} 个额外样本")
        
        # 使用分层抽样确保均匀分布
        theta_grid = np.linspace(-np.pi/3, np.pi/3, int(np.sqrt(remaining))+1)
        theta_dot_grid = np.linspace(-0.1, 0.1, int(np.sqrt(remaining))+1)
        
        for i in range(remaining):
            # 使用网格坐标加随机扰动，确保覆盖全部区域
            grid_i = i % (len(theta_grid)-1)
            grid_j = i // (len(theta_grid)-1) % (len(theta_dot_grid)-1)
            
            theta = theta_grid[grid_i] + (theta_grid[grid_i+1] - theta_grid[grid_i]) * np.random.random()
            theta = max(min(theta, np.pi/3), -np.pi/3) # 限制角度范围
            theta_dot = theta_dot_grid[grid_j] + (theta_dot_grid[grid_j+1] - theta_dot_grid[grid_j]) * np.random.random()
            theta_dot = max(min(theta_dot, 0.1), -0.1) # 限制角速度范围
            
            theta_samples.append(theta)
            theta_dot_samples.append(theta_dot)
    
    return np.array(theta_samples), np.array(theta_dot_samples)


# --- 改进的仿真数据生成函数 ---
def generate_optimized_simulation_data(pendulum, initial_conditions,
                                      dt=config.DT, t_span=config.T_SPAN):
    """
    为多组初始条件和力矩模式高效并行生成仿真数据
    
    Args:
        pendulum: PendulumSystem实例
        initial_conditions: 列表，包含多组[theta, theta_dot, tau]初始条件
        dt: 时间步长
        t_span: 仿真时间范围 (start, end)
        
    Returns:
        all_dfs: 列表，包含所有仿真生成的DataFrame
        boundaries: 列表，包含合并后各仿真数据的边界索引
    """
    all_dfs = []
    boundaries = []
    total_rows = 0
    
    # 确认初始条件数量
    num_simulations = len(initial_conditions)
    if num_simulations == 0:
        print("错误: 没有可用的初始条件或力矩模式")
        return [], []
        
    print(f"开始生成 {num_simulations} 组仿真数据...")
    
    # 使用JobLib进行并行计算
    from joblib import Parallel, delayed
    
    def process_simulation(i, x0):
        """为单组初始条件生成仿真数据"""
        print(f"  [process_simulation] 开始处理仿真 {i+1}/{num_simulations}, 初始条件: {x0}")
        
        try:
            # Handle different IC formats
            if len(x0) == 2:
                theta0, theta_dot0 = x0
                tau_value = np.random.uniform(config.TORQUE_RANGE[0], config.TORQUE_RANGE[1])
                print(f"  [process_simulation] 生成模拟 {i+1}/{num_simulations} (IC: [{theta0:.3f}, {theta_dot0:.3f}, {tau_value:.3f}] - random tau)...")
            elif len(x0) == 3:
                theta0, theta_dot0, tau_value = x0
                print(f"  [process_simulation] 生成模拟 {i+1}/{num_simulations} (IC: [{theta0:.3f}, {theta_dot0:.3f}, {tau_value:.3f}])...")
            else:
                print(f"  [process_simulation] 生成模拟 {i+1}/{num_simulations} (IC: 初始条件数量错误，预期2个或3个值)...")
                return pd.DataFrame()
        except (TypeError, ValueError) as e:
            print(f"  [process_simulation] Error in simulation {i+1}: Initial conditions are missing, {e}")
            return pd.DataFrame()
        
        if len(x0) != 3:
             print(f"  [process_simulation] initial_conditions length error: {len(x0)}, expected 3.")

        if len(x0) < 2:
            print(f"  [process_simulation] Initial condition error, x0 is not valid.")
            return pd.DataFrame()
        
        if len(x0) == 2:
            theta0, theta_dot0 = x0
            tau_value = np.random.uniform(config.TORQUE_RANGE[0], config.TORQUE_RANGE[1])
            return pd.DataFrame()
        theta0, theta_dot0, tau_value = x0
        
        # 生成完整时间序列
        t_full = np.arange(t_span[0], t_span[1] + dt/2, dt)
        return pd.DataFrame()
            
        # 使用恒定的力矩值
        print(f"  [process_simulation] 调用 run_simulation: pendulum={pendulum}, t_span={t_span}, dt={dt}, [theta0, theta_dot0]=[{theta0}, {theta_dot0}], tau_values={tau_values}, t_eval=t_full")
        tau_values = np.full_like(t_full, tau_value)
        
        # 运行仿真
        time_points, theta_values, theta_dot_values = run_simulation(
            pendulum, t_span, dt, [theta0, theta_dot0], tau_values, t_eval=t_full
        )

        if len(time_points) == 0:
            print(f"  [process_simulation] run_simulation 返回空结果。")
            return pd.DataFrame()
        
        if np.isnan(time_points).any() or np.isnan(theta_values).any() or np.isnan(theta_dot_values).any() or np.isnan(tau_values).any() or np.isinf(time_points).any() or np.isinf(theta_values).any() or np.isinf(theta_dot_values).any() or np.isinf(tau_values).any():
            return pd.DataFrame()

        # 将力矩值与实际输出时间对齐
        set_global_torque_params(tau_values, t_span[0], dt)
        tau_at_output_times = [get_tau_at_time_global(t) for t in time_points]
        set_global_torque_params(None, 0, config.DT)

        # 创建数据框
        data = {
            'time': time_points, 
            'theta': theta_values, 
            'theta_dot': theta_dot_values, 
            'tau': tau_at_output_times
        }
        df = pd.DataFrame(data)
        
        # 检查数据质量
        if df.isnull().values.any() or np.isinf(df.drop('time', axis=1)).values.any(): 
            print(f"  [process_simulation] 警告: 模拟 {i+1} 生成的数据包含NaN或Inf值")
            print(df)
            return pd.DataFrame()
        
        
        print(f"  [process_simulation] 完成处理仿真 {i+1}/{num_simulations}, 返回数据帧")
        
        
        return df
        
    
    # 并行运行所有仿真
    print(f"使用并行处理生成 {num_simulations} 组仿真数据...")
    results = Parallel(n_jobs=-1)(
        delayed(process_simulation)(i, initial_conditions[i])
        for i in range(num_simulations)
    )
    
    # 收集结果并计算边界
    for df in results:
        if df.empty: print(f"Empty Dataframe found!")
        if not df.empty:
            pass
        else:
            print(f"  [generate_optimized_simulation_data] 警告: 收到空的数据帧")
            all_dfs.append(df)
            total_rows += len(df)
            boundaries.append(total_rows)
    
    # 移除最后一个边界（它等于总数据长度）
    if boundaries and boundaries[-1] == total_rows:
        boundaries = boundaries[:-1]
        
    return all_dfs, boundaries


# --- 主要改进的仿真数据生成工作流 ---
def generate_improved_dataset(target_sequences=config.TARGET_SEQUENCES, dt=config.DT, 
                             t_span=config.T_SPAN, output_file=config.COMBINED_DATA_FILE):
    """
    使用改进的均匀采样方法生成整个训练数据集
    
    Args:
        target_sequences: 目标序列数量
        dt: 时间步长
        t_span: 仿真时间范围
        output_file: 输出CSV文件路径
        
    Returns:
        df_all: 合并的DataFrame
        simulation_boundaries: 模拟边界列表
    """
    start_time = time.time()
    print(f"开始生成改进的数据集 (目标: {target_sequences} 序列)...")
    
    # 创建单摆系统实例
    pendulum = PendulumSystem(
        m=config.PENDULUM_MASS, 
        L=config.PENDULUM_LENGTH,
        g=config.GRAVITY, 
        c=config.DAMPING_COEFF
    )
    
    # 1. 估算每次模拟生成的序列数
    sequence_length = config.INPUT_SEQ_LEN + config.OUTPUT_SEQ_LEN
    points_per_simulation = int((t_span[1] - t_span[0]) / dt)
    sequences_per_simulation = max(1, points_per_simulation - sequence_length + 1)
    
    # 2. 计算需要的模拟数量 (增加20%余量)
    simulations_needed = int(target_sequences / sequences_per_simulation * 1.2)
    print(f"每次模拟约能生成 {sequences_per_simulation} 个序列，需要约 {simulations_needed} 次模拟")    
    
    # 3. 生成多样化的初始条件
    print("生成多样化的初始条件...")
    
    # 3.1 使用一些预定义的特定初始条件
    specific_ics = config.INITIAL_CONDITIONS_SPECIFIC.copy()
    
    # 3.2 使用拉丁超立方体采样生成部分均匀分布的随机初始条件
    # 初始采样使用更宽的范围，以便物理能量采样能找到更多有效的组合
    lhs_samples_count = simulations_needed // 3
    lhs_param_ranges = {
        'theta': [-np.pi/3, np.pi/3],
        'theta_dot': [-0.1, 0.1]
    }
    lhs_params = generate_latin_hypercube_samples(lhs_samples_count, lhs_param_ranges)
    lhs_ics = [[lhs_params['theta'][i], lhs_params['theta_dot'][i]] for i in range(lhs_samples_count)]
    
    # Add a random torque value to each of their entries
    for ic in specific_ics:
        if len(ic) == 2:
            ic.append(np.random.uniform(config.TORQUE_RANGE[0], config.TORQUE_RANGE[1]))
    
    # Add a random torque value to each of their entries
    for ic in lhs_ics:
        if len(ic) == 2:
            ic.append(np.random.uniform(config.TORQUE_RANGE[0], config.TORQUE_RANGE[1]))
    
    # 3.3 使用物理能量级别生成更多有意义的初始条件
    physics_samples_count = simulations_needed - len(specific_ics) - len(lhs_ics)
    if physics_samples_count > 0:
       theta_samples, theta_dot_samples, torque_samples = generate_physics_informed_samples(
           physics_samples_count, pendulum
       )
       physics_ics = [[theta_samples[i], theta_dot_samples[i], torque_samples[i]] for i in range(len(theta_samples))]
    else:      
        physics_ics = []
    
    # 3.4 合并所有初始条件
    all_ics = specific_ics + lhs_ics + physics_ics # all_ics 现在包含 theta, theta_dot, tau
    print(f"总共生成 {len(all_ics)} 组初始条件: {len(specific_ics)} 个特定, {len(lhs_ics)} 个LHS采样, {len(physics_ics)} 个物理采样")
    
    
    # 5. 并行生成所有模拟数据
    all_dfs, boundaries = generate_optimized_simulation_data(
        pendulum, all_ics, dt, t_span
    )
    
    # 6. 合并所有数据
    if not all_dfs:
        print("错误: 未能生成任何有效的模拟数据")
        return None, []
    
    print(f"合并 {len(all_dfs)} 组模拟数据...")
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    
    # 7. 保存合并的数据
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_all.to_csv(output_file, index=False)
        print(f"合并数据已保存到 {output_file}")
    except Exception as e:
        print(f"保存数据时出错: {e}")
    
    # 8. 分析数据集的覆盖情况
    total_points = len(df_all)
    estimated_sequences = max(0, total_points - sequence_length + 1)
    print(f"生成完成，总计 {total_points} 个数据点，估计可生成 {estimated_sequences} 个序列")
    
    elapsed_time = time.time() - start_time
    print(f"数据生成总耗时: {elapsed_time:.2f} 秒")
    


# --- 用于验证生成数据多样性和质量的函数 ---
def analyze_dataset_coverage(df, output_dir="figures"):
    """
    分析生成数据的状态空间覆盖情况，并生成可视化
    
    Args:
        df: 合并的DataFrame
        output_dir: 图表输出目录
    """
    if df is None or df.empty:
        print("错误: 没有有效的数据集可供分析")
        return
    
    print(f"分析数据集覆盖情况 (共 {len(df)} 个数据点)...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取状态变量
    theta = df['theta'].values
    theta_dot = df['theta_dot'].values
    tau = df['tau'].values
    
    # 创建3D散点图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 如果数据点太多，随机抽样减少
    max_points = 5000
    if len(theta) > max_points:
        indices = np.random.choice(len(theta), max_points, replace=False)
        theta_sample = theta[indices]
        theta_dot_sample = theta_dot[indices]
        tau_sample = tau[indices]
    else:
        theta_sample = theta
        theta_dot_sample = theta_dot
        tau_sample = tau
    
    # 计算伪能量（用于颜色编码）
    energy = np.abs(theta_dot_sample)**2 + config.GRAVITY * np.abs(np.sin(theta_sample))
    
    # 绘制3D散点图
    scatter = ax.scatter(
        theta_sample, 
        theta_dot_sample, 
        tau_sample,
        c=energy,
        cmap='viridis',
        alpha=0.6,
        s=5
    )
    
    ax.set_xlabel('角度 (θ)', fontsize=12)
    ax.set_ylabel('角速度 (dθ/dt)', fontsize=12)
    ax.set_zlabel('力矩 (τ)', fontsize=12)
    ax.set_title('单摆系统状态空间覆盖', fontsize=14)
    
    plt.colorbar(scatter, ax=ax, label='伪能量')
    
    # 保存3D图
    plt.tight_layout()
    plt.savefig(f"{output_dir}/state_space_coverage_3d.png", dpi=300)
    
    # 创建2D投影图 (相空间和参数覆盖)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 相空间 (theta vs theta_dot)
    h1 = axes[0].hexbin(theta, theta_dot, gridsize=50, cmap='Blues', mincnt=1)
    axes[0].set_xlabel('角度 (θ)', fontsize=12)
    axes[0].set_ylabel('角速度 (dθ/dt)', fontsize=12)
    axes[0].set_title('相空间覆盖', fontsize=14)
    plt.colorbar(h1, ax=axes[0])
    
    # 角度 vs 力矩
    h2 = axes[1].hexbin(theta, tau, gridsize=50, cmap='Greens', mincnt=1)
    axes[1].set_xlabel('角度 (θ)', fontsize=12)
    axes[1].set_ylabel('力矩 (τ)', fontsize=12)
    axes[1].set_title('角度-力矩覆盖', fontsize=14)
    plt.colorbar(h2, ax=axes[1])
    
    # 角速度 vs 力矩
    h3 = axes[2].hexbin(theta_dot, tau, gridsize=50, cmap='Reds', mincnt=1)
    axes[2].set_xlabel('角速度 (dθ/dt)', fontsize=12)
    axes[2].set_ylabel('力矩 (τ)', fontsize=12)
    axes[2].set_title('角速度-力矩覆盖', fontsize=14)
    plt.colorbar(h3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_coverage_2d.png", dpi=300)
    
    # 计算覆盖率指标
    theta_bins = np.linspace(-np.pi, np.pi, 20)
    theta_dot_bins = np.linspace(-3.0, 3.0, 20)
    tau_bins = np.linspace(config.TORQUE_RANGE[0], config.TORQUE_RANGE[1], 20)
    
    # 计算3D直方图
    hist, _ = np.histogramdd(
        np.column_stack([theta, theta_dot, tau]),
        bins=[theta_bins, theta_dot_bins, tau_bins]
    )
    
    # 计算非空箱子的比例作为覆盖度量
    coverage_percentage = 100 * np.count_nonzero(hist) / hist.size
    
    # 计算非空箱子中的变异系数(标准差/均值)作为均匀度量
    non_empty_values = hist[hist > 0]
    cv = 100 * np.std(non_empty_values) / np.mean(non_empty_values)
    
    # 计算更直观的均匀度指标 (0-100%, 越高越均匀)
    uniformity = 100 / (1 + cv/100)  # 当CV=0时，uniformity=100%；当CV很大时，uniformity接近0%
    
    print(f"状态空间覆盖率: {coverage_percentage:.2f}%")
    print(f"数据分散度 (变异系数): {cv:.2f}%")
    print(f"均匀度: {uniformity:.2f}% (越高越均匀)")

    # 增加详细诊断信息
    if len(non_empty_values) > 0:
        print("\n详细分布诊断:")
        
        # 获取非空箱子的索引
        non_empty_indices = np.argwhere(hist > 0)
        
        # 排序非空箱子 (按计数值)
        sorted_indices = np.argsort(non_empty_values)
        
        # 计算箱子中心
        theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
        theta_dot_centers = (theta_dot_bins[:-1] + theta_dot_bins[1:]) / 2
        tau_centers = (tau_bins[:-1] + tau_bins[1:]) / 2
        
        # 打印最稀疏的 N 个箱子
        n_report = 5
        print(f"\n最稀疏的 {min(n_report, len(non_empty_values))} 个非空箱子:")
        for i in range(min(n_report, len(non_empty_values))):
            idx = sorted_indices[i]
            bin_indices = non_empty_indices[idx]
            center = (theta_centers[bin_indices[0]],
                      theta_dot_centers[bin_indices[1]],
                      tau_centers[bin_indices[2]])
            count = non_empty_values[idx]
            print(f"  - 中心: (θ={center[0]:.2f}, θ'={center[1]:.2f}, τ={center[2]:.2f}), 计数值: {count:.0f}")

        # 打印最密集的 N 个箱子
        print(f"\n最密集的 {min(n_report, len(non_empty_values))} 个非空箱子:")
        for i in range(min(n_report, len(non_empty_values))):
            idx = sorted_indices[-(i+1)] # 从末尾取
            bin_indices = non_empty_indices[idx]
            center = (theta_centers[bin_indices[0]],
                      theta_dot_centers[bin_indices[1]],
                      tau_centers[bin_indices[2]])
            count = non_empty_values[idx]
            print(f"  - 中心: (θ={center[0]:.2f}, θ'={center[1]:.2f}, τ={center[2]:.2f}), 计数值: {count:.0f}")
    
    # 创建力矩值分布图
    plt.figure(figsize=(10, 6))
    plt.hist(tau, bins=30, color='teal', alpha=0.7)
    plt.xlabel('力矩 (τ)', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.title('力矩分布', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/torque_distribution.png", dpi=300)
    
    # 创建物理相平面轨迹
    plt.figure(figsize=(10, 8))
    plt.scatter(theta, theta_dot, c=tau, cmap='viridis', alpha=0.3, s=2)
    plt.colorbar(label='力矩 (τ)')
    plt.xlabel('角度 (θ)', fontsize=12)
    plt.ylabel('角速度 (dθ/dt)', fontsize=12)
    plt.title('相平面轨迹', fontsize=14)
    plt.grid(alpha=0.3)
    # 添加箭头标记方向
    for i in range(len(theta)-1):
        if i % 500 == 0:  # 减少箭头数量以避免过度绘制
            plt.arrow(theta[i], theta_dot[i],
                     (theta[i+1] - theta[i])*20, (theta_dot[i+1] - theta_dot[i])*20,
                     head_width=0.05, head_length=0.1, fc='red', ec='red', alpha=0.6)
    plt.savefig(f"{output_dir}/phase_plane_trajectories.png", dpi=300)
    
    # 返回覆盖率指标用于评估
    return coverage_percentage, cv


# --- 主函数 (用于测试) ---
if __name__ == "__main__":
    print("正在运行改进的数据生成模块...")
    
    # 设置随机种子以确保可重现性
    np.random.seed(config.SEED)
    
    # 生成改进的数据集
    force_regenerate = True  # 设置为True以重新生成数据，False以使用现有数据（如果有）
    
    if force_regenerate or not os.path.exists(config.COMBINED_DATA_FILE):
        df_all, boundaries = generate_improved_dataset(
            target_sequences=config.TARGET_SEQUENCES,
            dt=config.DT, 
            t_span=config.T_SPAN,
            output_file=config.COMBINED_DATA_FILE
        )
        
        # 分析数据集的覆盖情况
        if df_all is not None and not df_all.empty:
            analyze_dataset_coverage(df_all)
    else:
        print(f"已存在数据文件: {config.COMBINED_DATA_FILE}")
        try:
            df_all = pd.read_csv(config.COMBINED_DATA_FILE)
            print(f"已加载现有数据，共 {len(df_all)} 个数据点")
            analyze_dataset_coverage(df_all)
        except Exception as e:
            print(f"加载现有数据时出错: {e}")
    
    print("改进的数据生成模块运行完成!")