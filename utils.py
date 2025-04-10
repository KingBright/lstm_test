# utils.py

import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import logging
import pandas as pd
import os
import numpy as np # Import numpy
import config # Import config for default parameters
# Import solve_ivp and PendulumSystem for validation
from scipy.integrate import solve_ivp
from data_generation import PendulumSystem

# --- setup_logging_and_warnings function (保持不变) ---
def setup_logging_and_warnings():
    # ... (内容不变) ...
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib"); logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR); warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization"); os.environ['TORCH_WARN_ALWAYS_UNSAFE_USAGE'] = '0'; # print("Logging and warnings configured.")

# --- safe_text function (保持不变) ---
def safe_text(text, fallback_text=None):
    # ... (内容不变) ...
    if fallback_text is None: fallback_text = text.replace('角度', 'Theta').replace('角速度', 'Angular Vel.').replace('预测', 'Pred').replace('真实', 'True').replace('误差', 'Error').replace('模型', 'Model').replace('累积', 'Cum.').replace('平均', 'Avg.').replace('纯物理', 'Physics').replace('轨迹', 'Traj.').replace('时间', 'Time').replace('步数', 'Steps').replace('范围', 'Range').replace('分析', 'Analysis').replace('相位图', 'Phase Plot').replace('vs', 'vs').replace('（', '(').replace('）', ')').replace('：', ':').replace(' ', '_').replace('场景', 'Scenario').replace('数据段', 'Data').replace('训练', 'Train').replace('验证', 'Val').replace('对比图', 'Comparison')
    try: _ = plt.figure(figsize=(0.1, 0.1)); plt.text(0.5, 0.5, text); plt.close(_); return text
    except Exception: return fallback_text

# --- setup_chinese_font function (保持不变) ---
def setup_chinese_font():
    # ... (内容不变) ...
    system = platform.system(); font_set = False; # print("Attempting to set Chinese font for Matplotlib...")
    try:
        if system == 'Darwin': font_list = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti'];
        elif system == 'Windows': font_list = ['SimHei', 'Microsoft YaHei', 'DengXian']
        elif system == 'Linux': font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Droid Sans Fallback']
        else: font_list = []
        for font in font_list:
            try: plt.rcParams['font.family'] = [font]; fig = plt.figure(figsize=(0.1, 0.1)); plt.text(0.5, 0.5, '测试'); plt.close(fig); # print(f"  Success: Set font to '{font}'"); font_set = True; break
            except Exception: continue
        # if not font_set and system in ['Darwin', 'Linux', 'Windows']: print(f"  Warning: Could not find suitable Chinese font for {system}.")
        if font_set: plt.rcParams['axes.unicode_minus'] = False; plt.rcParams['font.size'] = 10
        else: print("  Warning: Using default 'sans-serif' font."); plt.rcParams['font.family'] = ['sans-serif']; plt.rcParams['axes.unicode_minus'] = False; plt.rcParams['font.size'] = 10; plt.rcParams['axes.titlepad'] = 15
    except Exception as e: print(f"  Error setting Chinese font: {e}"); plt.rcParams['font.family'] = ['sans-serif']; plt.rcParams['axes.unicode_minus'] = False; plt.rcParams['font.size'] = 10

# --- plot_scenario_comparison function (保持不变) ---
def plot_scenario_comparison(scenario_name, train_dfs, val_dfs, save_dir=config.FIGURES_DIR):
    # ... (内容不变) ...
    if not train_dfs and not val_dfs: return
    # print(f"  Plotting scenario comparison for: {scenario_name}...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True); title = safe_text(f'场景数据对比 ({scenario_name}): 角度 θ', f'Scenario Data Comparison ({scenario_name}): Angle θ'); fig.suptitle(title, fontsize=14)
        axes[0].set_title(safe_text(f"训练数据段 (T={config.T_SPAN_TRAIN[1]}s)", f"Train Data (T={config.T_SPAN_TRAIN[1]}s)"));
        if train_dfs:
            for i, df_train in enumerate(train_dfs):
                if not df_train.empty and 'time' in df_train and 'theta' in df_train: axes[0].plot(df_train['time'], df_train['theta'], linewidth=1, alpha=0.7, label=f'IC {i+1}' if i < 5 else None)
            if len(train_dfs) > 0: axes[0].legend(fontsize=8, loc='upper right')
        else: axes[0].text(0.5, 0.5, 'No Training Data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_xlabel(safe_text('时间 (s)', 'Time (s)')); axes[0].set_ylabel(safe_text('角度 (rad)', 'Angle (rad)')); axes[0].grid(True)
        axes[1].set_title(safe_text(f"验证数据段 (T={config.T_SPAN_VAL[1]}s)", f"Validation Data (T={config.T_SPAN_VAL[1]}s)"));
        if val_dfs:
            for i, df_val in enumerate(val_dfs):
                 if not df_val.empty and 'time' in df_val and 'theta' in df_val: axes[1].plot(df_val['time'], df_val['theta'], linewidth=1, alpha=0.7, label=f'IC {i+1}' if i < 5 else None)
        else: axes[1].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_xlabel(safe_text('时间 (s)', 'Time (s)')); axes[1].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]); os.makedirs(save_dir, exist_ok=True); save_path = os.path.join(save_dir, f'scenario_comparison_{scenario_name}.png')
        plt.savefig(save_path, dpi=150); plt.close();
    except Exception as e: print(f"    Error plotting scenario comparison '{scenario_name}': {e}"); plt.close()


# +++ 更新后的物理校验函数 (基于 solve_ivp 步进比较) +++
def validate_physics(df, m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH,
                     g=config.GRAVITY, c=config.DAMPING_COEFF, dt=config.DT,
                     tolerance=config.PHYSICS_VALIDATION_TOLERANCE,
                     verbose=False):
    """
    通过将数据中的状态转移与 solve_ivp 单步预测进行比较来验证数据。
    检查 || state_data(t+1) - state_solve_ivp(t+1) || < tolerance 是否成立。
    注意：此函数会比较慢，因为它在循环中调用 solve_ivp。

    Args:
        df (pd.DataFrame): 包含 'time', 'theta', 'theta_dot', 'tau' 列的 DataFrame。
        m, L, g, c (float): 单摆参数。
        dt (float): 数据的时间步长 (用于设置 t_span)。
        tolerance (float): 允许的最大绝对状态差异。
        verbose (bool): 是否打印详细信息。

    Returns:
        bool: 如果数据看起来与 solve_ivp 预测一致则返回 True，否则返回 False。
    """
    required_cols = ['time', 'theta', 'theta_dot', 'tau']
    if not all(col in df.columns for col in required_cols): print("错误: DataFrame 缺少物理校验所需的列。"); return False
    if len(df) < 2: return True # 无法校验

    try: pendulum = PendulumSystem(m=m, L=L, g=g, c=c)
    except Exception as e: print(f"初始化 PendulumSystem 时出错: {e}"); return False

    states_data = df[['theta', 'theta_dot']].values
    tau_vals = df['tau'].values
    time_vals = df['time'].values

    if not np.all(np.isfinite(states_data)) or not np.all(np.isfinite(tau_vals)): print("警告: 数据中包含 NaN 或 Inf，物理校验失败。"); return False

    num_violations = 0
    violation_indices = []
    max_diff_observed = 0.0
    solver_failures = 0

    # --- 内部 ODE 函数，用于 solve_ivp ---
    # 这个函数需要能根据时间 t 获取对应的 tau 值
    # 为了避免全局变量，我们可以在每次调用 solve_ivp 时创建一个 lambda 函数捕获当前的 tau
    # 或者，如果 tau 在 dt 内变化不大，可以近似用 tau_t
    def ode_wrapper(t, x, current_tau):
        # 简单的 pendulum.ode 调用，传入捕获的 tau
        return pendulum.ode(t, x, current_tau=current_tau)

    # --- 迭代比较每一步 ---
    for i in range(len(df) - 1):
        state_t = states_data[i]
        tau_t = tau_vals[i] # 使用 i 时刻的 tau 作为这一步的近似恒定力矩
        time_t = time_vals[i]
        time_t_plus_1 = time_vals[i+1]
        state_data_t_plus_1 = states_data[i+1]

        # 检查时间步是否有效
        current_dt = time_t_plus_1 - time_t
        if current_dt <= 1e-9: continue # 跳过零或过小时间步

        # 使用 solve_ivp 进行单步预测
        try:
            sol = solve_ivp(
                lambda t, x: ode_wrapper(t, x, current_tau=tau_t), # 传递 tau_t
                (time_t, time_t_plus_1), # 时间区间
                state_t,                 # 初始状态
                method='RK45',           # 使用与生成数据相同的核心方法
                t_eval=[time_t_plus_1],  # 只在下一步结束时评估
                rtol=1e-6, atol=1e-9      # 使用较严格的容忍度进行校验
            )

            if sol.status == 0 and len(sol.y[0]) == 1:
                state_pred_t_plus_1 = sol.y[:, 0] # 获取预测结果
                if not np.all(np.isfinite(state_pred_t_plus_1)):
                     print(f"警告: 在索引 {i} 处 solve_ivp 预测失败 (NaN/Inf)。跳过。"); solver_failures += 1; continue

                # 计算差异
                diff = np.max(np.abs(state_data_t_plus_1 - state_pred_t_plus_1))
                max_diff_observed = max(max_diff_observed, diff)

                # 检查是否超差
                if diff > tolerance:
                    num_violations += 1
                    if verbose and len(violation_indices) < 5:
                        violation_indices.append({'index': i, 'diff': diff, 'data_next': state_data_t_plus_1, 'pred_next': state_pred_t_plus_1})
            else:
                # print(f"警告: 在索引 {i} 处 solve_ivp 求解失败 (status={sol.status}, len={len(sol.y[0])})。跳过。")
                solver_failures += 1
                continue # 跳过求解失败的步骤

        except Exception as e:
            print(f"警告: 在索引 {i} 处调用 solve_ivp 时出错: {e}。跳过。")
            solver_failures += 1
            continue

    # --- 总结结果 ---
    total_steps_checked = len(df) - 1 - solver_failures
    if total_steps_checked <= 0:
         print("  警告: 没有成功完成任何 solve_ivp 校验步骤。")
         return False # 或者 True，取决于如何定义？返回 False 表示校验有问题。

    if num_violations == 0:
        if verbose: print(f"  物理校验通过: 数据与 solve_ivp 步进预测一致 (容忍度={tolerance}, 最大差异={max_diff_observed:.2e})。")
        return True
    else:
        percentage_violation = (num_violations / total_steps_checked) * 100
        print(f"  警告: 物理校验失败! 在 {total_steps_checked} 个有效校验步中发现 {num_violations} ({percentage_violation:.2f}%) 个点"
              f" 状态转移与 solve_ivp 预测差异超过容忍度 ({tolerance})。")
        print(f"    观察到的最大差异: {max_diff_observed:.2e}")
        if solver_failures > 0: print(f"    另有 {solver_failures} 个步骤 solve_ivp 求解失败或返回无效结果。")
        if verbose:
            print(f"    前 {len(violation_indices)} 个违规示例 (索引, 差异, 数据值, 预测值):")
            for viol in violation_indices:
                 print(f"      {viol['index']}: diff={viol['diff']:.2e}, data={np.round(viol['data_next'], 6)}, pred={np.round(viol['pred_next'], 6)}")
        return False

