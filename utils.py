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
# Removed imports specific to validate_physics if they are no longer needed elsewhere
# from scipy.integrate import solve_ivp
# from data_generation import PendulumSystem

# --- M2 Max优化环境设置 (新函数) ---
def setup_m2_max_environment():
    """设置针对Apple M2 Max的PyTorch环境优化变量"""
    import torch
    if torch.backends.mps.is_available():
        try:
            # 设置环境变量优化Metal性能
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            # 禁用不必要的警告
            os.environ['TORCH_WARN_ALWAYS_UNSAFE_USAGE'] = '0'
            
            # 设置MPS缓存限制 (实验性设置，根据可用内存调整)
            # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 防止MPS后端过度分配内存
            
            # 设置并行工作线程数
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            os.environ['OMP_NUM_THREADS'] = str(min(num_cores, 6))  # 限制OpenMP线程数
            os.environ['MKL_NUM_THREADS'] = str(min(num_cores, 6))  # 限制MKL线程数
            
            print(f"✓ M2 Max优化环境已配置 (MPS加速已启用, {num_cores} 核心可用)")
            return True
        except Exception as e:
            print(f"! M2 Max环境设置时出错: {e}")
            return False
    return False

# --- setup_logging_and_warnings function (改进版) ---
def setup_logging_and_warnings():
    """配置日志和警告过滤，同时设置M2 Max优化环境"""
    # 设置警告过滤
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
    
    # 设置环境变量
    os.environ['TORCH_WARN_ALWAYS_UNSAFE_USAGE'] = '0'
    
    # 设置M2 Max优化环境
    setup_m2_max_environment()

# --- safe_text function (保持不变) ---
def safe_text(text, fallback_text=None):
    # ... (implementation remains the same) ...
    if fallback_text is None: fallback_text = text.replace('角度', 'Theta').replace('角速度', 'Angular Vel.').replace('预测', 'Pred').replace('真实', 'True').replace('误差', 'Error').replace('模型', 'Model').replace('累积', 'Cum.').replace('平均', 'Avg.').replace('纯物理', 'Physics').replace('轨迹', 'Traj.').replace('时间', 'Time').replace('步数', 'Steps').replace('范围', 'Range').replace('分析', 'Analysis').replace('相位图', 'Phase Plot').replace('vs', 'vs').replace('（', '(').replace('）', ')').replace('：', ':').replace(' ', '_').replace('场景', 'Scenario').replace('数据段', 'Data').replace('训练', 'Train').replace('验证', 'Val').replace('对比图', 'Comparison')
    try: _ = plt.figure(figsize=(0.1, 0.1)); plt.text(0.5, 0.5, text); plt.close(_); return text
    except Exception: return fallback_text

# --- setup_chinese_font function (保持不变) ---
def setup_chinese_font():
    # ... (implementation remains the same) ...
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

# --- plot_scenario_comparison function (保持不变, 但可能不再被调用) ---
# This function might be unused if main_experiment uses random data generation without scenarios
def plot_scenario_comparison(scenario_name, train_dfs, val_dfs, save_dir=config.FIGURES_DIR):
    # ... (implementation remains the same) ...
    if not train_dfs and not val_dfs: return
    # print(f"  Plotting scenario comparison for: {scenario_name}...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True); title = safe_text(f'场景数据对比 ({scenario_name}): 角度 θ', f'Scenario Data Comparison ({scenario_name}): Angle θ'); fig.suptitle(title, fontsize=14)
        axes[0].set_title(safe_text(f"训练数据段 (T={config.T_SPAN_TRAIN[1]}s)", f"Train Data (T={config.T_SPAN_TRAIN[1]}s)")); # Might error if T_SPAN_TRAIN removed
        if train_dfs:
            for i, df_train in enumerate(train_dfs):
                if not df_train.empty and 'time' in df_train and 'theta' in df_train: axes[0].plot(df_train['time'], df_train['theta'], linewidth=1, alpha=0.7, label=f'IC {i+1}' if i < 5 else None)
            if len(train_dfs) > 0: axes[0].legend(fontsize=8, loc='upper right')
        else: axes[0].text(0.5, 0.5, 'No Training Data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_xlabel(safe_text('时间 (s)', 'Time (s)')); axes[0].set_ylabel(safe_text('角度 (rad)', 'Angle (rad)')); axes[0].grid(True)
        axes[1].set_title(safe_text(f"验证数据段 (T={config.T_SPAN_VAL[1]}s)", f"Validation Data (T={config.T_SPAN_VAL[1]}s)")); # Might error if T_SPAN_VAL removed
        if val_dfs:
            for i, df_val in enumerate(val_dfs):
                 if not df_val.empty and 'time' in df_val and 'theta' in df_val: axes[1].plot(df_val['time'], df_val['theta'], linewidth=1, alpha=0.7, label=f'IC {i+1}' if i < 5 else None)
        else: axes[1].text(0.5, 0.5, 'No Validation Data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_xlabel(safe_text('时间 (s)', 'Time (s)')); axes[1].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]); os.makedirs(save_dir, exist_ok=True); save_path = os.path.join(save_dir, f'scenario_comparison_{scenario_name}.png')
        plt.savefig(save_path, dpi=150); plt.close();
    except Exception as e: print(f"    Error plotting scenario comparison '{scenario_name}': {e}"); plt.close()


# --- VVVVVV 物理校验函数已被移除 VVVVVV ---
# def validate_physics(...):
#     ... (旧代码) ...
# --- ^^^^^^ 物理校验函数已被移除 ^^^^^^ ---

