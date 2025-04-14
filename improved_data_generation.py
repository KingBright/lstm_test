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
    run_simulation, rk4_step, generate_simulation_data,
    generate_torque_sequence
)
import config
import utils

# --- Latin Hypercube Sampling ---
# (No changes from previous version)
def generate_latin_hypercube_samples(num_samples, param_ranges):
    """Generates parameter combinations using optimized Latin Hypercube Sampling."""
    n_dims = len(param_ranges)
    param_names = list(param_ranges.keys())
    optimization_method = "lloyd" if n_dims > 1 else None
    sampler = qmc.LatinHypercube(d=n_dims, optimization=optimization_method, seed=np.random.randint(0, 2**32 - 1))
    samples = sampler.random(n=num_samples)
    best_quality = float('nan')
    if num_samples > 1 and n_dims > 1: # Only calculate for >1D
        try:
            best_quality = qmc.discrepancy(samples)
        except Exception as e:
            print(f"计算离散度时出错: {e}")
            best_quality = float('nan')
    if n_dims == 1:
        samples = np.atleast_2d(samples).T
    if not np.isnan(best_quality):
        print(f"拉丁超立方体样本离散度: {best_quality:.6f} (越低越均匀)")
    params_dict = {}
    for i, param_name in enumerate(param_names):
        low, high = param_ranges[param_name]
        if n_dims == 1:
            params_dict[param_name] = qmc.scale(samples, low, high).flatten()
        else:
            column = samples[:, i:i+1]
            params_dict[param_name] = qmc.scale(column, low, high).flatten()
    return params_dict

# --- Physics-Informed Sampling ---
# (No changes from previous version)
def generate_physics_informed_samples(num_samples, pendulum):
    """Generates initial conditions (theta, theta_dot) based on physics energy levels."""
    m = pendulum.m; L = pendulum.L; g = pendulum.g
    min_energy = -m * g * L; max_energy = 3 * m * g * L
    num_levels = 35
    energy_levels_linear = np.linspace(min_energy, max_energy, num_levels)
    special_energies = [-0.8*m*g*L, -0.5*m*g*L, 0, 0.5*m*g*L, 1.5*m*g*L, 2.5*m*g*L]
    energy_levels = np.unique(np.concatenate([energy_levels_linear, special_energies]))
    actual_num_levels = len(energy_levels)
    print(f"使用 {actual_num_levels} 个能量级别进行物理采样")

    if num_samples < actual_num_levels:
        print(f"警告: 请求的物理样本数 ({num_samples}) 少于能量级别数 ({actual_num_levels})。")
        samples_per_level = 0; remainder_samples = num_samples
    else:
        samples_per_level = num_samples // actual_num_levels
        remainder_samples = num_samples % actual_num_levels
    print(f"每个能量级别分配 {samples_per_level} 个样本，剩余 {remainder_samples} 个样本随机分配")

    all_energy_indices = []; all_theta_samples_for_levels = []
    level_indices_to_sample = np.arange(actual_num_levels)
    if num_samples < actual_num_levels:
        level_indices_to_sample = np.random.choice(actual_num_levels, num_samples, replace=False)

    assigned_samples = 0
    for level_idx in level_indices_to_sample:
        num_level_samples = samples_per_level + (1 if assigned_samples < remainder_samples else 0)
        if num_level_samples == 0: continue
        energy_ratio = np.clip(energy_levels[level_idx] / (m * g * L), -1, 1)
        max_theta = np.arccos(energy_ratio); theta_range_limit = min(max_theta, np.pi)
        lower_bound = -theta_range_limit * 0.95; upper_bound = theta_range_limit * 0.95
        if lower_bound >= upper_bound: upper_bound = lower_bound + 1e-6
        if num_level_samples > 0:
             theta_level_lhs = generate_latin_hypercube_samples(num_level_samples, {'theta': [lower_bound, upper_bound]})['theta']
             all_theta_samples_for_levels.extend(theta_level_lhs)
             all_energy_indices.extend([level_idx] * num_level_samples)
             assigned_samples += num_level_samples

    current_total_samples = len(all_theta_samples_for_levels)
    if current_total_samples != num_samples:
         print(f"警告: 分配后的样本数 ({current_total_samples}) 与请求数 ({num_samples}) 不符。")
         if current_total_samples > num_samples:
             all_theta_samples_for_levels = all_theta_samples_for_levels[:num_samples]
             all_energy_indices = all_energy_indices[:num_samples]

    shuffle_indices = np.random.permutation(len(all_theta_samples_for_levels))
    energy_indices = np.array(all_energy_indices)[shuffle_indices]
    theta_samples_final = np.array(all_theta_samples_for_levels)[shuffle_indices]
    theta_dot_samples_final = np.zeros_like(theta_samples_final)

    for i in range(len(theta_samples_final)):
        theta = theta_samples_final[i]; energy_idx = energy_indices[i]; energy = energy_levels[energy_idx]
        term = 2 * (energy + m * g * L * np.cos(theta)) / (m * L * L)
        attempts = 0
        while term < -1e-9 and attempts < 10:
            theta = np.random.uniform(-np.pi, np.pi)
            term = 2 * (energy + m * g * L * np.cos(theta)) / (m * L * L)
            attempts += 1
        if term < -1e-9:
            print(f"警告: 无法为 E={energy:.2f} 找到有效角度，使用随机 theta_dot")
            theta_dot = np.random.uniform(*config.THETA_DOT_RANGE)
        else:
            term = max(0, term); theta_dot = np.sqrt(term)
            if np.random.choice([-1, 1]) == -1: theta_dot = -theta_dot
        theta_samples_final[i] = theta; theta_dot_samples_final[i] = theta_dot

    # Phase space coverage check and filling (remains the same)
    grid_size = 10
    theta_bins = np.linspace(-np.pi, np.pi, grid_size+1)
    theta_dot_bins = np.linspace(config.THETA_DOT_RANGE[0], config.THETA_DOT_RANGE[1], grid_size+1)
    phase_space_grid = np.zeros((grid_size, grid_size))
    theta_indices = np.clip(((theta_samples_final + np.pi) / (2*np.pi) * grid_size).astype(int), 0, grid_size-1)
    theta_dot_indices = np.clip(((theta_dot_samples_final - config.THETA_DOT_RANGE[0]) / (config.THETA_DOT_RANGE[1] - config.THETA_DOT_RANGE[0]) * grid_size).astype(int), 0, grid_size-1)
    np.add.at(phase_space_grid, (theta_indices, theta_dot_indices), 1)
    empty_cells = np.argwhere(phase_space_grid == 0)
    if len(empty_cells) > 0:
        print(f"检测到 {len(empty_cells)} 个空白区域，尝试添加额外样本")
        added_samples = 0; max_additional = int(len(theta_samples_final) * 0.1)
        for i, j in empty_cells:
            if added_samples >= max_additional: break
            theta = theta_bins[i] + (theta_bins[i+1] - theta_bins[i]) * np.random.random()
            theta_dot = theta_dot_bins[j] + (theta_dot_bins[j+1] - theta_dot_bins[j]) * np.random.random()
            current_energy = 0.5*m*L*L*theta_dot**2 - m*g*L*np.cos(theta)
            if current_energy <= max_energy + 1e-6:
                theta_samples_final = np.append(theta_samples_final, theta)
                theta_dot_samples_final = np.append(theta_dot_samples_final, theta_dot)
                added_samples += 1
        print(f"  实际添加了 {added_samples} 个额外样本到空白区域")

    return theta_samples_final, theta_dot_samples_final


# --- Optimized Simulation Data Generation ---
# (No changes needed here)
def generate_optimized_simulation_data(pendulum, initial_conditions_base,
                                      dt=config.DT, t_span=(0, config.SIMULATION_DURATION_MEDIUM)):
    """Generates simulation data in parallel using varied torque sequences for a given t_span."""
    all_dfs = []; boundaries = []; total_rows = 0
    num_simulations = len(initial_conditions_base)
    if num_simulations == 0: print("错误: 没有可用的初始条件"); return [], [], 0
    print(f"开始生成 {num_simulations} 组仿真数据 (时长: {t_span[1]:.1f}s, 使用变化力矩)...")

    from joblib import Parallel, delayed

    def process_simulation(i, x0):
        """Generates simulation data for a single IC with varied torque."""
        try:
            if len(x0) != 2: print(f"..."); return pd.DataFrame()
            theta0, theta_dot0 = x0
        except Exception as e: print(f"..."); return pd.DataFrame()

        t_full = np.arange(t_span[0], t_span[1] + dt/2, dt)
        if len(t_full) == 0: print(f"..."); return pd.DataFrame()

        available_torque_types = ["highly_random", "sine", "step", "ramp", "zero"]
        type_probabilities = [0.4, 0.15, 0.15, 0.15, 0.15]
        selected_torque_type = np.random.choice(available_torque_types, p=type_probabilities)

        try:
            tau_values = generate_torque_sequence(t_full, type=selected_torque_type, torque_change_steps=None)
            if len(tau_values) != len(t_full): print(f"..."); tau_values = np.zeros_like(t_full)
        except Exception as torque_e: print(f"..."); return pd.DataFrame()

        time_points, theta_values, theta_dot_values = run_simulation(pendulum, t_span, dt, [theta0, theta_dot0], tau_values, t_eval=t_full)
        if len(time_points) == 0: return pd.DataFrame()

        set_global_torque_params(tau_values, t_span[0], dt)
        tau_at_output_times = [get_tau_at_time_global(t) for t in time_points]
        set_global_torque_params(None, 0, config.DT)

        data = {'time': time_points, 'theta': theta_values, 'theta_dot': theta_dot_values, 'tau': tau_at_output_times}
        df = pd.DataFrame(data)

        if df.isnull().values.any() or np.isinf(df.drop('time', axis=1)).values.any():
            print(f"  [Process {os.getpid()}] 警告: 模拟 {i+1} ({selected_torque_type}) 生成的数据包含NaN或Inf值，将丢弃。")
            return pd.DataFrame()
        return df

    print(f"使用并行处理 ({joblib.cpu_count()} 核心) 生成 {num_simulations} 组仿真数据...")
    results = Parallel(n_jobs=-1, verbose=5)(delayed(process_simulation)(i, initial_conditions_base[i]) for i in range(num_simulations))

    valid_sim_count = 0; current_batch_rows = 0
    for i, df in enumerate(results):
        if df is not None and not df.empty:
            all_dfs.append(df)
            current_batch_rows += len(df)
            boundaries.append(current_batch_rows)
            valid_sim_count += 1
        else: print(f"  警告: 模拟 {i+1} 未能生成有效数据。")

    print(f"成功生成了 {valid_sim_count}/{num_simulations} 组有效仿真数据 (时长: {t_span[1]:.1f}s)。")
    if boundaries and current_batch_rows > 0 and boundaries[-1] == current_batch_rows:
        boundaries = boundaries[:-1]
    return all_dfs, boundaries, current_batch_rows


# --- Main Dataset Generation Workflow (MODIFIED FOR DOWNSAMPLING) ---
def generate_improved_dataset(target_sequences=config.TARGET_SEQUENCES, dt=config.DT,
                             output_file=config.COMBINED_DATA_FILE):
    """
    Generates the dataset using improved sampling, varied torque, mixed durations,
    and optional density-based downsampling.
    """
    start_time = time.time()
    print(f"开始生成改进的数据集 (目标: {target_sequences} 序列, 混合时长, 降采样: {config.USE_DOWNSAMPLING})...")
    pendulum = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH, g=config.GRAVITY, c=config.DAMPING_COEFF)

    # --- Calculate needed simulations (consider oversampling) ---
    sequence_length = config.INPUT_SEQ_LEN + config.OUTPUT_SEQ_LEN
    points_per_medium_sim = int(config.SIMULATION_DURATION_MEDIUM / dt) if dt > 0 else 0
    seq_per_medium_sim = max(1, points_per_medium_sim - sequence_length + 1) if points_per_medium_sim > 0 else 1
    base_simulations_needed = int(target_sequences / seq_per_medium_sim * 1.2) if seq_per_medium_sim > 0 else 100

    if config.USE_DOWNSAMPLING:
        simulations_to_generate = int(base_simulations_needed * config.OVERSAMPLING_FACTOR)
        print(f"启用降采样: 基础需求约 {base_simulations_needed} 次模拟, 过采样生成 {simulations_to_generate} 次模拟 (因子: {config.OVERSAMPLING_FACTOR:.1f})")
    else:
        simulations_to_generate = base_simulations_needed
        print(f"禁用降采样: 生成约 {simulations_to_generate} 次模拟")

    # --- Generate Initial Conditions ---
    print("生成多样化的初始条件...")
    specific_ics_base = config.INITIAL_CONDITIONS_SPECIFIC.copy()
    # Adjust sampling split based on total needed
    lhs_samples_count = simulations_to_generate // 3
    lhs_param_ranges = {'theta': config.THETA_RANGE, 'theta_dot': config.THETA_DOT_RANGE}
    lhs_params = generate_latin_hypercube_samples(lhs_samples_count, lhs_param_ranges)
    lhs_ics_base = [[lhs_params['theta'][i], lhs_params['theta_dot'][i]] for i in range(lhs_samples_count)]

    physics_samples_count = simulations_to_generate - len(specific_ics_base) - len(lhs_ics_base)
    if physics_samples_count > 0:
        theta_samples, theta_dot_samples = generate_physics_informed_samples(physics_samples_count, pendulum)
        physics_ics_base = [[theta_samples[i], theta_dot_samples[i]] for i in range(len(theta_samples))]
    else:
        physics_ics_base = []

    all_ics_base = specific_ics_base + lhs_ics_base + physics_ics_base
    actual_simulations_to_generate = len(all_ics_base) # Use the actual number generated
    print(f"总共生成 {actual_simulations_to_generate} 组基础初始条件: {len(specific_ics_base)} 特定, {len(lhs_ics_base)} LHS, {len(physics_ics_base)} 物理")

    np.random.shuffle(all_ics_base) # Shuffle ICs before splitting

    # --- Generate Data in Batches (Mixed Durations) ---
    durations = [config.SIMULATION_DURATION_SHORT, config.SIMULATION_DURATION_MEDIUM, config.SIMULATION_DURATION_LONG]
    t_spans = [(0, d) for d in durations]
    num_batches = len(t_spans)
    base_n = actual_simulations_to_generate // num_batches
    remainder = actual_simulations_to_generate % num_batches
    batch_sizes = [base_n + 1 if i < remainder else base_n for i in range(num_batches)]

    all_sim_dfs_oversampled = []
    start_idx = 0
    for i, t_span_batch in enumerate(t_spans):
        batch_size = batch_sizes[i]
        end_idx = start_idx + batch_size
        ics_batch = all_ics_base[start_idx:end_idx]
        start_idx = end_idx
        if not ics_batch: continue

        print(f"\n--- 生成批次 {i+1}/{num_batches} (时长: {t_span_batch[1]:.1f}s, 数量: {batch_size}) ---")
        batch_dfs, _, _ = generate_optimized_simulation_data( # Ignore boundaries and row count from this call now
            pendulum, ics_batch, dt, t_span_batch
        )
        if batch_dfs: all_sim_dfs_oversampled.extend(batch_dfs)

    if not all_sim_dfs_oversampled:
        print("错误: 未能生成任何有效的模拟数据")
        return None, None # Return None for boundaries

    print(f"\n合并所有批次的 {len(all_sim_dfs_oversampled)} 组有效模拟数据 (过采样)...")
    df_all_oversampled = pd.concat(all_sim_dfs_oversampled, ignore_index=True)
    print(f"过采样数据总点数: {len(df_all_oversampled)}")

    # --- Apply Density-Based Downsampling (if enabled) ---
    if config.USE_DOWNSAMPLING and not df_all_oversampled.empty:
        print("\n--- 开始基于密度的降采样 ---")
        downsampling_start_time = time.time()

        # Define bins based on the generated data range
        grid_size = config.DOWNSAMPLING_GRID_SIZE
        theta_min, theta_max = df_all_oversampled['theta'].min(), df_all_oversampled['theta'].max()
        theta_dot_min, theta_dot_max = df_all_oversampled['theta_dot'].min(), df_all_oversampled['theta_dot'].max()
        tau_min, tau_max = df_all_oversampled['tau'].min(), df_all_oversampled['tau'].max()

        # Add small epsilon to max range to include boundary points
        epsilon = 1e-6
        bins = [
            np.linspace(theta_min, theta_max + epsilon, grid_size + 1),
            np.linspace(theta_dot_min, theta_dot_max + epsilon, grid_size + 1),
            np.linspace(tau_min, tau_max + epsilon, grid_size + 1)
        ]
        print(f"降采样网格大小: {grid_size}x{grid_size}x{grid_size}")

        # Calculate histogram (density per bin)
        data_for_hist = df_all_oversampled[['theta', 'theta_dot', 'tau']].values
        hist, _ = np.histogramdd(data_for_hist, bins=bins)
        non_empty_counts = hist[hist > 0]
        total_non_empty_bins = len(non_empty_counts)
        print(f"总网格数: {hist.size}, 非空网格数: {total_non_empty_bins}")

        if total_non_empty_bins > 0:
            # Determine target count per bin based on percentile
            target_percentile = config.DOWNSAMPLING_TARGET_PERCENTILE
            target_count = int(np.percentile(non_empty_counts, target_percentile))
            # Ensure target count is at least 1 to avoid removing all points from sparse bins
            target_count = max(1, target_count)
            print(f"目标百分位数: {target_percentile}th => 每个非空网格最多保留 {target_count} 个点")

            # Assign each point to its bin index
            bin_indices_theta = np.digitize(df_all_oversampled['theta'].values, bins[0]) - 1
            bin_indices_theta_dot = np.digitize(df_all_oversampled['theta_dot'].values, bins[1]) - 1
            bin_indices_tau = np.digitize(df_all_oversampled['tau'].values, bins[2]) - 1
            # Clip indices to be within valid range [0, grid_size-1]
            bin_indices_theta = np.clip(bin_indices_theta, 0, grid_size - 1)
            bin_indices_theta_dot = np.clip(bin_indices_theta_dot, 0, grid_size - 1)
            bin_indices_tau = np.clip(bin_indices_tau, 0, grid_size - 1)

            # Add bin indices as temporary columns for grouping
            df_all_oversampled['_bin_idx_theta'] = bin_indices_theta
            df_all_oversampled['_bin_idx_theta_dot'] = bin_indices_theta_dot
            df_all_oversampled['_bin_idx_tau'] = bin_indices_tau

            # Group by bin and perform sampling
            grouped = df_all_oversampled.groupby(['_bin_idx_theta', '_bin_idx_theta_dot', '_bin_idx_tau'])
            final_indices = []

            for _, group in grouped:
                if len(group) <= target_count:
                    final_indices.extend(group.index.tolist()) # Keep all indices
                else:
                    # Sample target_count indices from this dense group
                    sampled_indices = np.random.choice(group.index, target_count, replace=False)
                    final_indices.extend(sampled_indices.tolist())

            # Create final dataframe
            df_final = df_all_oversampled.loc[final_indices].copy()
            # Remove temporary bin columns
            df_final = df_final.drop(columns=['_bin_idx_theta', '_bin_idx_theta_dot', '_bin_idx_tau'])
            # Shuffle the final dataset
            df_final = df_final.sample(frac=1, random_state=config.SEED).reset_index(drop=True)

            print(f"降采样完成，保留点数: {len(df_final)} (原点数: {len(df_all_oversampled)})")
            print(f"降采样耗时: {time.time() - downsampling_start_time:.2f} 秒")
            df_to_save = df_final
            # *** Boundaries are invalidated by downsampling ***
            final_boundaries = None
            print("注意: 降采样已执行，原始模拟边界信息已丢失。")

        else:
            print("警告: 未找到非空网格，无法执行降采样。")
            df_to_save = df_all_oversampled # Save the oversampled data
            final_boundaries = None # No boundaries applicable

    else:
        # Downsampling disabled or data empty
        print("降采样被禁用或无数据生成。")
        df_to_save = df_all_oversampled
        # Boundaries are not meaningful if downsampling isn't used with oversampling
        final_boundaries = None

    # --- Save Final Data ---
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True); print(f"创建目录: {output_dir}")
        df_to_save.to_csv(output_file, index=False)
        print(f"最终数据已保存到 {output_file}")
        # Do not save boundaries if downsampling was used or data was empty
        if final_boundaries is not None:
             boundaries_file = output_file.replace('.csv', '_boundaries.npy')
             np.save(boundaries_file, np.array(final_boundaries))
             print(f"模拟边界信息已保存到: {boundaries_file}")
        elif config.USE_DOWNSAMPLING:
             # Clean up any old boundary file if downsampling was used
             boundaries_file = output_file.replace('.csv', '_boundaries.npy')
             if os.path.exists(boundaries_file):
                 os.remove(boundaries_file)
                 print(f"已删除旧的边界文件 (因使用降采样): {boundaries_file}")

    except Exception as e:
        print(f"保存最终数据时出错: {e}")

    # --- Final Summary ---
    total_points = len(df_to_save)
    # Estimate sequences based on total points, assuming no boundaries if downsampled
    estimated_sequences = max(0, total_points - sequence_length + 1) if total_points > 0 else 0
    print(f"生成完成，最终数据集点数: {total_points}，估计可生成 {estimated_sequences} 个序列")
    elapsed_time = time.time() - start_time
    print(f"数据生成总耗时: {elapsed_time:.2f} 秒")

    # Return the final dataframe and None for boundaries if downsampled
    return df_to_save, final_boundaries


# --- Dataset Coverage Analysis ---
# (No changes needed here)
def analyze_dataset_coverage(df, output_dir="figures"):
    """Analyzes and visualizes the state space coverage of the generated dataset."""
    utils.setup_chinese_font()

    if df is None or df.empty:
        print("错误: 没有有效的数据集可供分析")
        return float('nan'), float('nan')

    print(f"分析数据集覆盖情况 (共 {len(df)} 个数据点)...")
    os.makedirs(output_dir, exist_ok=True)

    theta = df['theta'].values; theta_dot = df['theta_dot'].values; tau = df['tau'].values

    # 3D Scatter Plot
    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        max_points = 5000
        if len(theta) > max_points:
            indices = np.random.choice(len(theta), max_points, replace=False)
            theta_sample, theta_dot_sample, tau_sample = theta[indices], theta_dot[indices], tau[indices]
        else:
            theta_sample, theta_dot_sample, tau_sample = theta, theta_dot, tau
        energy = 0.5 * config.PENDULUM_MASS * config.PENDULUM_LENGTH**2 * theta_dot_sample**2 - config.PENDULUM_MASS * config.GRAVITY * config.PENDULUM_LENGTH * np.cos(theta_sample)
        scatter = ax.scatter(theta_sample, theta_dot_sample, tau_sample, c=energy, cmap='viridis', alpha=0.6, s=5)
        ax.set_xlabel(utils.safe_text('角度 (θ)')); ax.set_ylabel(utils.safe_text('角速度 (dθ/dt)')); ax.set_zlabel(utils.safe_text('力矩 (τ)'))
        ax.set_title(utils.safe_text('单摆系统状态空间覆盖')); plt.colorbar(scatter, ax=ax, label=utils.safe_text('能量'))
        plt.tight_layout(); plt.savefig(f"{output_dir}/state_space_coverage_3d.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"绘制 3D 散点图时出错: {e}"); plt.close()

    # 2D Projection Plots
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        h1 = axes[0].hexbin(theta, theta_dot, gridsize=40, cmap='Blues', mincnt=1)
        axes[0].set_xlabel(utils.safe_text('角度 (θ)')); axes[0].set_ylabel(utils.safe_text('角速度 (dθ/dt)')); axes[0].set_title(utils.safe_text('相空间覆盖')); plt.colorbar(h1, ax=axes[0])
        h2 = axes[1].hexbin(theta, tau, gridsize=40, cmap='Greens', mincnt=1)
        axes[1].set_xlabel(utils.safe_text('角度 (θ)')); axes[1].set_ylabel(utils.safe_text('力矩 (τ)')); axes[1].set_title(utils.safe_text('角度-力矩覆盖')); plt.colorbar(h2, ax=axes[1])
        h3 = axes[2].hexbin(theta_dot, tau, gridsize=40, cmap='Reds', mincnt=1)
        axes[2].set_xlabel(utils.safe_text('角速度 (dθ/dt)')); axes[2].set_ylabel(utils.safe_text('力矩 (τ)')); axes[2].set_title(utils.safe_text('角速度-力矩覆盖')); plt.colorbar(h3, ax=axes[2])
        plt.tight_layout(); plt.savefig(f"{output_dir}/parameter_coverage_2d.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"绘制 2D 投影图时出错: {e}"); plt.close()

    # Coverage Metrics
    coverage_percentage = float('nan'); cv = float('nan'); uniformity = float('nan')
    try:
        # Use the same grid size as used for downsampling if available
        grid_size_analysis = config.DOWNSAMPLING_GRID_SIZE if config.USE_DOWNSAMPLING else 20
        theta_bins = np.linspace(theta.min(), theta.max(), grid_size_analysis + 1)
        theta_dot_bins = np.linspace(theta_dot.min(), theta_dot.max(), grid_size_analysis + 1)
        tau_bins = np.linspace(tau.min(), tau.max(), grid_size_analysis + 1)
        hist, _ = np.histogramdd(np.column_stack([theta, theta_dot, tau]), bins=[theta_bins, theta_dot_bins, tau_bins])
        coverage_percentage = 100 * np.count_nonzero(hist) / hist.size
        non_empty_values = hist[hist > 0]
        if len(non_empty_values) > 0:
            mean_val = np.mean(non_empty_values); std_val = np.std(non_empty_values)
            cv = 100 * std_val / mean_val if mean_val > 1e-9 else 0
            uniformity = 100 / (1 + cv/100)
        else:
            cv = 0; uniformity = 0
        print(f"状态空间覆盖率: {coverage_percentage:.2f}%")
        print(f"数据分散度 (变异系数): {cv:.2f}%")
        print(f"均匀度: {uniformity:.2f}% (越高越均匀)")
    except Exception as e:
        print(f"计算覆盖率指标时出错: {e}")

    # Torque Distribution Plot
    try:
        plt.figure(figsize=(8, 5))
        plt.hist(tau, bins=30, color='teal', alpha=0.7)
        plt.xlabel(utils.safe_text('力矩 (τ)')); plt.ylabel(utils.safe_text('频率')); plt.title(utils.safe_text('力矩分布')); plt.grid(alpha=0.3);
        plt.savefig(f"{output_dir}/torque_distribution.png", dpi=150); plt.close()
    except Exception as e:
        print(f"绘制力矩分布图时出错: {e}"); plt.close()

    return coverage_percentage, uniformity


# --- Main function for testing ---
if __name__ == "__main__":
    print("正在运行改进的数据生成模块 (测试模式)...")
    np.random.seed(config.SEED)
    utils.setup_chinese_font()

    # Ensure test config reflects downsampling if needed
    config.USE_DOWNSAMPLING = True # Enable for testing this feature
    config.OVERSAMPLING_FACTOR = 2.0 # Lower factor for faster testing
    config.TARGET_SEQUENCES = 5000 # Smaller target for testing

    force_regenerate = True
    test_output_file = './test_downsampled_data.csv' # Changed test filename

    if force_regenerate or not os.path.exists(test_output_file):
        df_all, boundaries = generate_improved_dataset(
            target_sequences=config.TARGET_SEQUENCES, # Use smaller test target
            dt=config.DT,
            output_file=test_output_file
        )
        if df_all is not None and not df_all.empty:
            print("\n分析测试生成的数据集覆盖情况...")
            analyze_dataset_coverage(df_all, output_dir="figures_test_downsampled")
            # No boundaries to save after downsampling
        else:
            print("测试数据生成失败。")
    else:
        print(f"找到现有测试数据文件: {test_output_file}")
        try:
            df_all = pd.read_csv(test_output_file)
            print(f"已加载测试数据，共 {len(df_all)} 个数据点")
            analyze_dataset_coverage(df_all, output_dir="figures_test_downsampled")
        except Exception as e:
            print(f"加载测试数据时出错: {e}")

    print("\n改进的数据生成模块测试运行完成!")

