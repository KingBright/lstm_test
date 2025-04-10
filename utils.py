# utils.py

import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import logging
import pandas as pd
import os
import numpy as np # Import numpy for isnan/isfinite checks
import config # Import config for FIGURE_DIR default

def setup_logging_and_warnings():
    """Configures logging levels and warning filters."""
    # Suppress Matplotlib font warnings and logs
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    # Suppress potential PyTorch serialization warnings (optional)
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
    os.environ['TORCH_WARN_ALWAYS_UNSAFE_USAGE'] = '0' # Alternative for some warnings
    print("Logging and warnings configured.")

def safe_text(text, fallback_text=None):
    """
    Attempts to use the provided text, falling back to a simplified version
    if rendering fails (e.g., due to font issues).
    """
    if fallback_text is None:
        # Basic fallback: remove potentially problematic characters or provide simple ASCII
        fallback_text = text.replace('角度', 'Theta').replace('角速度', 'Angular Vel.') \
                            .replace('预测', 'Pred').replace('真实', 'True') \
                            .replace('误差', 'Error').replace('模型', 'Model') \
                            .replace('累积', 'Cum.').replace('平均', 'Avg.') \
                            .replace('纯物理', 'Physics').replace('轨迹', 'Traj.') \
                            .replace('时间', 'Time').replace('步数', 'Steps') \
                            .replace('范围', 'Range').replace('分析', 'Analysis') \
                            .replace('相位图', 'Phase Plot').replace('vs', 'vs') \
                            .replace('（', '(').replace('）', ')') \
                            .replace('：', ':').replace(' ', '_') \
                            .replace('场景', 'Scenario').replace('数据段', 'Data') \
                            .replace('训练', 'Train').replace('验证', 'Val') \
                            .replace('对比图', 'Comparison')
        # Add more replacements as needed

    try:
        # Minimal check without plotting might be safer if display issues are frequent
        _ = plt.figure(figsize=(0.1, 0.1)) # Create a tiny temporary figure
        plt.text(0.5, 0.5, text)          # Try adding the text
        plt.close(_)                      # Close the figure immediately
        return text # If no error, assume it's okay
    except Exception:
        # print(f"Warning: Using fallback text for '{text}' due to potential rendering issues.")
        return fallback_text

def setup_chinese_font():
    """Sets up Matplotlib to support Chinese characters based on the OS."""
    system = platform.system()
    font_set = False
    print("Attempting to set Chinese font for Matplotlib...")
    try:
        if system == 'Darwin': # macOS
            font_list = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti']
            for font in font_list:
                try:
                    plt.rcParams['font.family'] = [font]
                    fig = plt.figure(figsize=(0.1, 0.1)); plt.text(0.5, 0.5, '测试'); plt.close(fig)
                    print(f"  Success: Set macOS font to '{font}'")
                    font_set = True; break
                except Exception: continue
            if not font_set: print("  Warning: Could not find suitable macOS Chinese font.")
        elif system == 'Windows':
            font_list = ['SimHei', 'Microsoft YaHei', 'DengXian']
            for font in font_list:
                 try:
                      plt.rcParams['font.family'] = [font]
                      fig = plt.figure(figsize=(0.1, 0.1)); plt.text(0.5, 0.5, '测试'); plt.close(fig)
                      print(f"  Success: Set Windows font to '{font}'")
                      font_set = True; break
                 except Exception: continue
            if not font_set: print("  Warning: Could not find suitable Windows Chinese font.")
        elif system == 'Linux':
            font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Droid Sans Fallback']
            for font in font_list:
                 try:
                     plt.rcParams['font.family'] = [font]
                     fig = plt.figure(figsize=(0.1, 0.1)); plt.text(0.5, 0.5, '测试'); plt.close(fig)
                     print(f"  Success: Set Linux font to '{font}'")
                     font_set = True; break
                 except Exception: continue
            if not font_set: print("  Warning: Could not find suitable Linux Chinese font.")

        if font_set:
             plt.rcParams['axes.unicode_minus'] = False
             plt.rcParams['font.size'] = 10 # Slightly smaller default size for potentially busy plots
        else:
             print("  Warning: Using default 'sans-serif' font.")
             plt.rcParams['font.family'] = ['sans-serif']
             plt.rcParams['axes.unicode_minus'] = False
             plt.rcParams['font.size'] = 10
             plt.rcParams['axes.titlepad'] = 15

    except Exception as e:
        print(f"  Error setting Chinese font: {e}")
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10

# --- NEW Plotting function for Scenario Comparison ---
def plot_scenario_comparison(scenario_name, train_dfs, val_dfs, save_dir=config.FIGURES_DIR):
    """
    Plots a comparison of training and validation data segments for a specific scenario.
    Overlays results from different initial conditions for theta only.

    Args:
        scenario_name (str): Name of the scenario.
        train_dfs (list): List of DataFrames, each generated using T_SPAN_TRAIN for one IC.
        val_dfs (list): List of DataFrames, each generated using T_SPAN_VAL for one IC.
        save_dir (str): Directory to save the plot.
    """
    if not train_dfs and not val_dfs:
        print(f"Warning: No train or validation data provided for scenario '{scenario_name}'. Skipping plot.")
        return

    print(f"  Plotting scenario comparison for: {scenario_name}...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True) # 1 row, 2 columns (Train, Val), shared y-axis
        title = safe_text(f'场景数据对比 ({scenario_name}): 角度 θ', f'Scenario Data Comparison ({scenario_name}): Angle θ')
        fig.suptitle(title, fontsize=14)

        # Plot Training Data Segment (Left)
        axes[0].set_title(safe_text(f"训练数据段 (T={config.T_SPAN_TRAIN[1]}s)", f"Train Data (T={config.T_SPAN_TRAIN[1]}s)"))
        if train_dfs:
            for i, df_train in enumerate(train_dfs):
                if not df_train.empty and 'time' in df_train and 'theta' in df_train:
                    # Plot each initial condition with slight transparency
                    axes[0].plot(df_train['time'], df_train['theta'], linewidth=1, alpha=0.7, label=f'IC {i+1}' if i < 5 else None) # Label first few
            axes[0].legend(fontsize=8, loc='upper right') # Add legend for ICs
        else:
            axes[0].text(0.5, 0.5, 'No Training Data', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
        axes[0].set_xlabel(safe_text('时间 (s)', 'Time (s)'))
        axes[0].set_ylabel(safe_text('角度 (rad)', 'Angle (rad)'))
        axes[0].grid(True)

        # Plot Validation Data Segment (Right)
        axes[1].set_title(safe_text(f"验证数据段 (T={config.T_SPAN_VAL[1]}s)", f"Validation Data (T={config.T_SPAN_VAL[1]}s)"))
        if val_dfs:
            for i, df_val in enumerate(val_dfs):
                 if not df_val.empty and 'time' in df_val and 'theta' in df_val:
                    axes[1].plot(df_val['time'], df_val['theta'], linewidth=1, alpha=0.7, label=f'IC {i+1}' if i < 5 else None)
            # axes[1].legend(fontsize=8, loc='upper right') # Legend might be redundant if ICs are same
        else:
             axes[1].text(0.5, 0.5, 'No Validation Data', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
        axes[1].set_xlabel(safe_text('时间 (s)', 'Time (s)'))
        # axes[1].set_ylabel(safe_text('角度 (rad)', 'Angle (rad)')) # Shared Y-axis
        axes[1].grid(True)

        # Save
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'scenario_comparison_{scenario_name}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"    -> Scenario comparison plot saved to: {save_path}")

    except Exception as e:
        print(f"    Error plotting scenario comparison '{scenario_name}': {e}")
        plt.close() # Ensure figure is closed even on error

