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
    run_simulation, rk4_step, generate_simulation_data, # Keep generate_simulation_data for eval compatibility
    generate_torque_sequence # Import the enhanced torque generator
)
import config
import utils # *** Import utils for font handling ***

# --- Latin Hypercube Sampling ---
def generate_latin_hypercube_samples(num_samples, param_ranges):
    """Generates parameter combinations using optimized Latin Hypercube Sampling."""
    n_dims = len(param_ranges)
    param_names = list(param_ranges.keys())
    optimization_method = "lloyd" if n_dims > 1 else None
    sampler = qmc.LatinHypercube(d=n_dims, optimization=optimization_method, seed=config.SEED)
    samples = sampler.random(n=num_samples)
    best_quality = float('nan')
    if num_samples > 1:
        try:
            best_quality = qmc.discrepancy(samples)
        except Exception as e:
            print(f"计算离散度时出错: {e}") # Error calculating discrepancy
            best_quality = float('nan')
    if n_dims == 1:
        samples = np.atleast_2d(samples).T
    print(f"拉丁超立方体样本离散度: {best_quality:.6f} (越低越均匀)") # LHS sample discrepancy (lower is more uniform)
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
def generate_physics_informed_samples(num_samples, pendulum):
    """Generates initial conditions (theta, theta_dot) based on physics energy levels."""
    m = pendulum.m; L = pendulum.L; g = pendulum.g
    min_energy = -m * g * L
    max_energy = 3 * m * g * L
    num_levels = 50
    energy_levels_linear = np.linspace(min_energy, max_energy, num_levels)
    special_energies = [-0.8 * m * g * L, -0.5 * m * g * L, 0, 0.5 * m * g * L, 1.5 * m * g * L, 2.5 * m * g * L]
    energy_levels = np.unique(np.concatenate([energy_levels_linear, special_energies]))
    print(f"使用 {len(energy_levels)} 个能量级别进行物理采样") # Using {len(energy_levels)} energy levels for physics sampling

    samples_per_level = num_samples // len(energy_levels)
    remainder_samples = num_samples % len(energy_levels)
    print(f"每个能量级别分配 {samples_per_level} 个样本，剩余 {remainder_samples} 个样本随机分配") # Assigning {samples_per_level} samples per level, {remainder_samples} remaining randomly assigned

    all_energy_indices = []
    all_theta_samples_for_levels = []
    for level_idx in range(len(energy_levels)):
        num_level_samples = samples_per_level + (1 if level_idx < remainder_samples else 0)
        if num_level_samples == 0: continue
        energy_ratio = np.clip(energy_levels[level_idx] / (m * g * L), -1, 1)
        max_theta = np.arccos(energy_ratio)
        theta_range_limit = min(max_theta, np.pi)
        lower_bound = -theta_range_limit * 0.95
        upper_bound = theta_range_limit * 0.95
        if lower_bound >= upper_bound: upper_bound = lower_bound + 1e-6
        theta_level_lhs = generate_latin_hypercube_samples(num_level_samples, {'theta': [lower_bound, upper_bound]})['theta']
        all_theta_samples_for_levels.extend(theta_level_lhs)
        all_energy_indices.extend([level_idx] * num_level_samples)

    shuffle_indices = np.random.permutation(num_samples)
    energy_indices = np.array(all_energy_indices)[shuffle_indices]
    theta_samples_final = np.array(all_theta_samples_for_levels)[shuffle_indices]
    theta_dot_samples_final = np.zeros_like(theta_samples_final)

    for i in range(num_samples):
        theta = theta_samples_final[i]
        energy_idx = energy_indices[i]
        energy = energy_levels[energy_idx]
        term = 2 * (energy + m * g * L * np.cos(theta)) / (m * L * L)
        attempts = 0
        while term < -1e-9 and attempts < 10:
            theta = np.random.uniform(-np.pi, np.pi)
            term = 2 * (energy + m * g * L * np.cos(theta)) / (m * L * L)
            attempts += 1
        if term < -1e-9:
            print(f"警告: 无法为 E={energy:.2f} 找到有效角度，使用随机 theta_dot") # Warning: Cannot find valid angle for E={energy:.2f}, using random theta_dot
            theta_dot = np.random.uniform(*config.THETA_DOT_RANGE)
        else:
            term = max(0, term)
            theta_dot = np.sqrt(term)
            if np.random.choice([-1, 1]) == -1: theta_dot = -theta_dot
        theta_samples_final[i] = theta
        theta_dot_samples_final[i] = theta_dot

    # Phase space coverage check and filling
    grid_size = 10
    theta_bins = np.linspace(-np.pi, np.pi, grid_size+1)
    theta_dot_bins = np.linspace(config.THETA_DOT_RANGE[0], config.THETA_DOT_RANGE[1], grid_size+1)
    phase_space_grid = np.zeros((grid_size, grid_size))
    theta_indices = np.clip(((theta_samples_final + np.pi) / (2*np.pi) * grid_size).astype(int), 0, grid_size-1)
    theta_dot_indices = np.clip(((theta_dot_samples_final - config.THETA_DOT_RANGE[0]) / (config.THETA_DOT_RANGE[1] - config.THETA_DOT_RANGE[0]) * grid_size).astype(int), 0, grid_size-1)
    np.add.at(phase_space_grid, (theta_indices, theta_dot_indices), 1)
    empty_cells = np.argwhere(phase_space_grid == 0)
    if len(empty_cells) > 0:
        print(f"检测到 {len(empty_cells)} 个空白区域，尝试添加额外样本") # Detected {len(empty_cells)} empty regions, attempting to add extra samples
        added_samples = 0
        max_additional = int(num_samples * 0.1)
        for i, j in empty_cells:
            if added_samples >= max_additional: break
            theta = theta_bins[i] + (theta_bins[i+1] - theta_bins[i]) * np.random.random()
            theta_dot = theta_dot_bins[j] + (theta_dot_bins[j+1] - theta_dot_bins[j]) * np.random.random()
            current_energy = 0.5*m*L*L*theta_dot**2 - m*g*L*np.cos(theta)
            if current_energy <= max_energy + 1e-6:
                theta_samples_final = np.append(theta_samples_final, theta)
                theta_dot_samples_final = np.append(theta_dot_samples_final, theta_dot)
                added_samples += 1
        print(f"  实际添加了 {added_samples} 个额外样本到空白区域") # Actually added {added_samples} extra samples to empty regions

    target_sample_size = num_samples
    if len(theta_samples_final) > target_sample_size * 1.05:
        print(f"样本数量 ({len(theta_samples_final)}) 过多，随机采样至约 {target_sample_size} 个") # Sample count ({len(theta_samples_final)}) too high, sampling down to ~{target_sample_size}
        indices = np.random.choice(len(theta_samples_final), target_sample_size, replace=False)
        theta_samples_final = theta_samples_final[indices]
        theta_dot_samples_final = theta_dot_samples_final[indices]

    return theta_samples_final, theta_dot_samples_final


# --- Optimized Simulation Data Generation (Calling Varied Torque) ---
def generate_optimized_simulation_data(pendulum, initial_conditions_base, # Now expects list of [theta, theta_dot]
                                      dt=config.DT, t_span=config.T_SPAN):
    """Generates simulation data in parallel using varied torque sequences."""
    all_dfs = []
    boundaries = []
    total_rows = 0
    num_simulations = len(initial_conditions_base)
    if num_simulations == 0:
        print("错误: 没有可用的初始条件") # Error: No initial conditions available
        return [], []
    print(f"开始生成 {num_simulations} 组仿真数据 (使用变化力矩)...") # Starting generation of {num_simulations} simulations (with varied torque)...

    from joblib import Parallel, delayed

    # --- Nested function for parallel execution ---
    def process_simulation(i, x0): # x0 is now [theta0, theta_dot0]
        """Generates simulation data for a single IC with varied torque."""
        try:
            if len(x0) != 2:
                 print(f"  [Process {os.getpid()}] 错误: 模拟 {i+1} 的初始条件长度应为2 (theta, theta_dot)，得到 {len(x0)}") # Error: Sim {i+1} IC length should be 2, got {len(x0)}
                 return pd.DataFrame()
            theta0, theta_dot0 = x0
        except (TypeError, ValueError) as e:
            print(f"  [Process {os.getpid()}] 错误: 模拟 {i+1} 的初始条件无效: {e}") # Error: Sim {i+1} IC invalid: {e}
            return pd.DataFrame()

        t_full = np.arange(t_span[0], t_span[1] + dt/2, dt)
        if len(t_full) == 0:
            print(f"  [Process {os.getpid()}] 错误: 模拟 {i+1} 生成的时间序列为空 (t_span={t_span}, dt={dt})") # Error: Sim {i+1} generated empty time series
            return pd.DataFrame()

        # *** MODIFICATION: Randomly select torque type for this simulation ***
        available_torque_types = ["highly_random", "sine", "step", "ramp", "zero"]
        # Give 'highly_random' a higher probability, others equal
        type_probabilities = [0.4, 0.15, 0.15, 0.15, 0.15]
        selected_torque_type = np.random.choice(available_torque_types, p=type_probabilities)
        # print(f"  [Process {os.getpid()}] Sim {i+1} using torque type: {selected_torque_type}") # Optional debug print

        try:
            # Generate torque using the randomly selected type
            tau_values = generate_torque_sequence(
                t_full,
                type=selected_torque_type, # Use the randomly selected type
                torque_change_steps=None # Let function use config range if needed
            )
            if len(tau_values) != len(t_full):
                 # This check might be redundant if generate_torque_sequence handles length correctly
                print(f"  [Process {os.getpid()}] 错误: 模拟 {i+1} ({selected_torque_type}) 生成的力矩序列长度 ({len(tau_values)}) 与时间序列长度 ({len(t_full)}) 不匹配") # Error: Sim {i+1} ({selected_torque_type}) generated torque sequence length mismatch
                tau_values = np.zeros_like(t_full) # Fallback
        except Exception as torque_e:
             print(f"  [Process {os.getpid()}] 错误: 模拟 {i+1} ({selected_torque_type}) 生成力矩序列时出错: {torque_e}") # Error: Sim {i+1} ({selected_torque_type}) error generating torque sequence
             return pd.DataFrame()

        # Run simulation with the generated torque
        time_points, theta_values, theta_dot_values = run_simulation(
            pendulum, t_span, dt, [theta0, theta_dot0], tau_values, t_eval=t_full
        )

        if len(time_points) == 0:
            return pd.DataFrame()

        # Align torque values with output times
        set_global_torque_params(tau_values, t_span[0], dt)
        tau_at_output_times = [get_tau_at_time_global(t) for t in time_points]
        set_global_torque_params(None, 0, config.DT)

        # Create DataFrame
        data = {'time': time_points, 'theta': theta_values, 'theta_dot': theta_dot_values, 'tau': tau_at_output_times}
        df = pd.DataFrame(data)

        # Check data quality
        if df.isnull().values.any() or np.isinf(df.drop('time', axis=1)).values.any():
            print(f"  [Process {os.getpid()}] 警告: 模拟 {i+1} ({selected_torque_type}) 生成的数据包含NaN或Inf值，将丢弃。") # Warning: Sim {i+1} ({selected_torque_type}) generated data contains NaN/Inf, discarding.
            return pd.DataFrame()
        return df
    # --- End of nested function ---

    # Run simulations in parallel
    print(f"使用并行处理 ({joblib.cpu_count()} 核心) 生成 {num_simulations} 组仿真数据...") # Using parallel processing ({joblib.cpu_count()} cores) to generate {num_simulations} simulations...
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_simulation)(i, initial_conditions_base[i]) # Pass [theta, theta_dot]
        for i in range(num_simulations)
    )

    # Collect results and calculate boundaries
    valid_sim_count = 0
    for i, df in enumerate(results):
        if df is not None and not df.empty:
            all_dfs.append(df)
            total_rows += len(df)
            boundaries.append(total_rows)
            valid_sim_count += 1
        else:
            print(f"  警告: 模拟 {i+1} 未能生成有效数据。") # Warning: Simulation {i+1} failed to generate valid data.

    print(f"成功生成了 {valid_sim_count}/{num_simulations} 组有效仿真数据。") # Successfully generated {valid_sim_count}/{num_simulations} valid simulations.
    if boundaries and total_rows > 0 and boundaries[-1] == total_rows:
        boundaries = boundaries[:-1]
    return all_dfs, boundaries


# --- Main Improved Dataset Generation Workflow (Calling Varied Torque) ---
def generate_improved_dataset(target_sequences=config.TARGET_SEQUENCES, dt=config.DT,
                             t_span=config.T_SPAN, output_file=config.COMBINED_DATA_FILE):
    """Generates the dataset using improved sampling and varied torque."""
    start_time = time.time()
    print(f"开始生成改进的数据集 (目标: {target_sequences} 序列)...") # Starting generation of improved dataset (Target: {target_sequences} sequences)...
    pendulum = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH, g=config.GRAVITY, c=config.DAMPING_COEFF)

    sequence_length = config.INPUT_SEQ_LEN + config.OUTPUT_SEQ_LEN
    points_per_simulation = int((t_span[1] - t_span[0]) / dt) if dt > 0 else 0
    sequences_per_simulation = max(1, points_per_simulation - sequence_length + 1) if points_per_simulation > 0 else 1
    simulations_needed = int(target_sequences / sequences_per_simulation * 1.2) if sequences_per_simulation > 0 else 100
    print(f"每次模拟约能生成 {sequences_per_simulation} 个序列，需要约 {simulations_needed} 次模拟") # Each sim generates ~{sequences_per_simulation} sequences, need ~{simulations_needed} simulations

    print("生成多样化的初始条件...") # Generating diverse initial conditions...
    specific_ics_base = config.INITIAL_CONDITIONS_SPECIFIC.copy()
    lhs_samples_count = simulations_needed // 3
    lhs_param_ranges = {'theta': config.THETA_RANGE, 'theta_dot': config.THETA_DOT_RANGE}
    lhs_params = generate_latin_hypercube_samples(lhs_samples_count, lhs_param_ranges)
    lhs_ics_base = [[lhs_params['theta'][i], lhs_params['theta_dot'][i]] for i in range(lhs_samples_count)]

    physics_samples_count = simulations_needed - len(specific_ics_base) - len(lhs_ics_base)
    if physics_samples_count > 0:
        theta_samples, theta_dot_samples = generate_physics_informed_samples(physics_samples_count, pendulum)
        physics_ics_base = [[theta_samples[i], theta_dot_samples[i]] for i in range(len(theta_samples))]
    else:
        physics_ics_base = []

    all_ics_base = specific_ics_base + lhs_ics_base + physics_ics_base
    print(f"总共生成 {len(all_ics_base)} 组基础初始条件: {len(specific_ics_base)} 特定, {len(lhs_ics_base)} LHS, {len(physics_ics_base)} 物理") # Generated {len(all_ics_base)} base ICs: {len(specific_ics_base)} specific, {len(lhs_ics_base)} LHS, {len(physics_ics_base)} physics

    all_dfs, boundaries = generate_optimized_simulation_data(
        pendulum, all_ics_base, dt, t_span
    )

    if not all_dfs:
        print("错误: 未能生成任何有效的模拟数据") # Error: Failed to generate any valid simulation data
        return None, []

    print(f"合并 {len(all_dfs)} 组有效的模拟数据...") # Concatenating {len(all_dfs)} valid simulation datasets...
    df_all = pd.concat(all_dfs, ignore_index=True)

    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True)
             print(f"创建目录: {output_dir}") # Created directory
        df_all.to_csv(output_file, index=False)
        print(f"合并数据已保存到 {output_file}") # Combined data saved to {output_file}
    except Exception as e:
        print(f"保存数据时出错: {e}") # Error saving data

    total_points = len(df_all)
    estimated_sequences = max(0, total_points - sequence_length + 1) if total_points > 0 else 0
    print(f"生成完成，总计 {total_points} 个数据点，估计可生成 {estimated_sequences} 个序列") # Generation complete, total {total_points} data points, estimated {estimated_sequences} sequences
    elapsed_time = time.time() - start_time
    print(f"数据生成总耗时: {elapsed_time:.2f} 秒") # Total data generation time

    return df_all, boundaries


# --- Dataset Coverage Analysis ---
def analyze_dataset_coverage(df, output_dir="figures"):
    """Analyzes and visualizes the state space coverage of the generated dataset."""
    utils.setup_chinese_font() # Setup font for plotting

    if df is None or df.empty:
        print("错误: 没有有效的数据集可供分析") # Error: No valid dataset to analyze
        return float('nan'), float('nan')

    print(f"分析数据集覆盖情况 (共 {len(df)} 个数据点)...") # Analyzing dataset coverage ({len(df)} points)...
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
        ax.set_title(utils.safe_text('单摆系统状态空间覆盖')); plt.colorbar(scatter, ax=ax, label=utils.safe_text('能量')) # Pendulum System State Space Coverage; Energy
        plt.tight_layout(); plt.savefig(f"{output_dir}/state_space_coverage_3d.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"绘制 3D 散点图时出错: {e}"); plt.close() # Error plotting 3D scatter

    # 2D Projection Plots
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        h1 = axes[0].hexbin(theta, theta_dot, gridsize=40, cmap='Blues', mincnt=1)
        axes[0].set_xlabel(utils.safe_text('角度 (θ)')); axes[0].set_ylabel(utils.safe_text('角速度 (dθ/dt)')); axes[0].set_title(utils.safe_text('相空间覆盖')); plt.colorbar(h1, ax=axes[0]) # Phase Space Coverage
        h2 = axes[1].hexbin(theta, tau, gridsize=40, cmap='Greens', mincnt=1)
        axes[1].set_xlabel(utils.safe_text('角度 (θ)')); axes[1].set_ylabel(utils.safe_text('力矩 (τ)')); axes[1].set_title(utils.safe_text('角度-力矩覆盖')); plt.colorbar(h2, ax=axes[1]) # Angle-Torque Coverage
        h3 = axes[2].hexbin(theta_dot, tau, gridsize=40, cmap='Reds', mincnt=1)
        axes[2].set_xlabel(utils.safe_text('角速度 (dθ/dt)')); axes[2].set_ylabel(utils.safe_text('力矩 (τ)')); axes[2].set_title(utils.safe_text('角速度-力矩覆盖')); plt.colorbar(h3, ax=axes[2]) # AngVel-Torque Coverage
        plt.tight_layout(); plt.savefig(f"{output_dir}/parameter_coverage_2d.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"绘制 2D 投影图时出错: {e}"); plt.close() # Error plotting 2D projections

    # Coverage Metrics
    coverage_percentage = float('nan'); cv = float('nan'); uniformity = float('nan')
    try:
        theta_bins = np.linspace(theta.min(), theta.max(), 20)
        theta_dot_bins = np.linspace(theta_dot.min(), theta_dot.max(), 20)
        tau_bins = np.linspace(tau.min(), tau.max(), 20)
        hist, _ = np.histogramdd(np.column_stack([theta, theta_dot, tau]), bins=[theta_bins, theta_dot_bins, tau_bins])
        coverage_percentage = 100 * np.count_nonzero(hist) / hist.size
        non_empty_values = hist[hist > 0]
        if len(non_empty_values) > 0:
            mean_val = np.mean(non_empty_values); std_val = np.std(non_empty_values)
            cv = 100 * std_val / mean_val if mean_val > 1e-9 else 0
            uniformity = 100 / (1 + cv/100)
        else:
            cv = 0; uniformity = 0
        print(f"状态空间覆盖率: {coverage_percentage:.2f}%") # State space coverage
        print(f"数据分散度 (变异系数): {cv:.2f}%") # Data dispersion (CV)
        print(f"均匀度: {uniformity:.2f}% (越高越均匀)") # Uniformity (higher is more uniform)
    except Exception as e:
        print(f"计算覆盖率指标时出错: {e}") # Error calculating coverage metrics

    # Torque Distribution Plot
    try:
        plt.figure(figsize=(8, 5))
        plt.hist(tau, bins=30, color='teal', alpha=0.7)
        plt.xlabel(utils.safe_text('力矩 (τ)')); plt.ylabel(utils.safe_text('频率')); plt.title(utils.safe_text('力矩分布')); plt.grid(alpha=0.3); # Torque Distribution; Frequency
        plt.savefig(f"{output_dir}/torque_distribution.png", dpi=150); plt.close()
    except Exception as e:
        print(f"绘制力矩分布图时出错: {e}"); plt.close() # Error plotting torque distribution

    return coverage_percentage, uniformity


# --- Main function for testing ---
if __name__ == "__main__":
    print("正在运行改进的数据生成模块 (测试模式)...") # Running improved data generation module (test mode)...
    np.random.seed(config.SEED)
    utils.setup_chinese_font() # Setup font for test mode plotting

    force_regenerate = True
    test_output_file = './test_varied_torque_data.csv' # Use different test filename

    if force_regenerate or not os.path.exists(test_output_file):
        df_all, boundaries = generate_improved_dataset(
            target_sequences=10000, # Lower sequence target for testing
            dt=config.DT,
            t_span=(0, 10.0), # Shorter simulation time for testing
            output_file=test_output_file
        )
        if df_all is not None and not df_all.empty:
            print("\n分析测试生成的数据集覆盖情况...") # Analyzing test generated dataset coverage...
            analyze_dataset_coverage(df_all, output_dir="figures_test_varied")
        else:
            print("测试数据生成失败。") # Test data generation failed.
    else:
        print(f"找到现有测试数据文件: {test_output_file}") # Found existing test data file
        try:
            df_all = pd.read_csv(test_output_file)
            print(f"已加载测试数据，共 {len(df_all)} 个数据点") # Loaded test data, {len(df_all)} points
            analyze_dataset_coverage(df_all, output_dir="figures_test_varied")
        except Exception as e:
            print(f"加载测试数据时出错: {e}") # Error loading test data

    print("\n改进的数据生成模块测试运行完成!") # Improved data generation module test run complete!

