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

# --- setup_logging_and_warnings function (保持不变) ---
def setup_logging_and_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib"); logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR); warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization"); os.environ['TORCH_WARN_ALWAYS_UNSAFE_USAGE'] = '0'; # print("Logging and warnings configured.")

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

