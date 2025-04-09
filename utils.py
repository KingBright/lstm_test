# utils.py

import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import logging
import pandas as pd
import os

def setup_logging_and_warnings():
    """Configures logging levels and warning filters."""
    # Suppress Matplotlib font warnings and logs
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    # Suppress potential PyTorch serialization warnings (optional)
    # warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
    os.environ['TORCH_WARN_ALWAYS_UNSAFE_USAGE'] = '0' # Alternative for some warnings

def setup_chinese_font():
    """Sets up Matplotlib to support Chinese characters based on the OS."""
    system = platform.system()
    font_set = False
    try:
        if system == 'Darwin': # macOS
            font_list = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti']
            for font in font_list:
                try:
                    plt.rcParams['font.family'] = [font]
                    # Test font validity briefly
                    fig = plt.figure(figsize=(0.1, 0.1))
                    plt.text(0.5, 0.5, '测试', ha='center', va='center')
                    plt.close(fig)
                    print(f"成功设置中文字体: {font}")
                    font_set = True
                    break
                except Exception:
                    continue
            if not font_set:
                 print("未找到合适的 macOS 中文字体 (Arial Unicode MS, PingFang SC, Heiti SC, STHeiti)。")
        elif system == 'Windows':
            plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei']
            # Test font validity (optional)
            font_set = True # Assume common Windows fonts exist
            print("设置Windows中文字体: Microsoft YaHei, SimHei")
        elif system == 'Linux':
            # Check common Linux fonts
            font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Droid Sans Fallback']
            for font in font_list:
                 try:
                     # A simple check without plotting might be less prone to display errors
                     # This doesn't guarantee rendering, but checks if Matplotlib knows the font
                     # mpl.font_manager.findfont(font, fallback_to_default=False)
                     # Or keep the plot test if preferred
                     plt.rcParams['font.family'] = [font]
                     fig = plt.figure(figsize=(0.1, 0.1)); plt.text(0.5, 0.5, '测试'); plt.close(fig)
                     print(f"成功设置Linux中文字体: {font}")
                     font_set = True
                     break
                 except Exception:
                     continue
            if not font_set:
                 print("未找到合适的 Linux 中文字体 (文泉驿微米黑, Noto Sans CJK SC等)。")

        # Apply general settings if a font was potentially set or use fallback
        if font_set or system not in ['Darwin', 'Linux']: # Apply even if Windows default is assumed
             plt.rcParams['axes.unicode_minus'] = False  # Solve minus sign display issue
             plt.rcParams['font.size'] = 12
        else:
             # Fallback if no specific font worked
             print("无法设置特定中文字体，将使用默认 sans-serif 字体。")
             plt.rcParams['font.family'] = ['sans-serif']
             plt.rcParams['axes.unicode_minus'] = False
             plt.rcParams['font.size'] = 12
             # Add padding for titles just in case
             plt.rcParams['axes.titlepad'] = 20

    except Exception as e:
        print(f"设置中文字体时发生未知错误: {e}")
        # Ensure basic fallback settings
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12

def plot_simulation_data_sample(df, save_dir='figures', num_points=5000):
     """Plots a sample of the simulation data."""
     if df.empty:
         print("Warning: DataFrame is empty, skipping plotting simulation sample.")
         return

     plot_points = min(num_points, len(df))
     if plot_points == 0:
         print("Warning: No data points to plot for simulation sample.")
         return

     try:
         plt.figure(figsize=(12, 6))

         plt.subplot(2, 1, 1)
         plt.plot(df['time'][:plot_points], df['theta'][:plot_points], linewidth=1)
         plt.title('部分仿真数据: 角度 (Theta)')
         plt.ylabel('角度 (rad)')
         plt.grid(True)

         plt.subplot(2, 1, 2)
         plt.plot(df['time'][:plot_points], df['theta_dot'][:plot_points], linewidth=1)
         plt.title('部分仿真数据: 角速度 (Theta_dot)')
         plt.ylabel('角速度 (rad/s)')
         plt.xlabel('时间 (s)')
         plt.grid(True)

         plt.tight_layout()
         save_path = os.path.join(save_dir, 'simulation_data_sample.png')
         plt.savefig(save_path, dpi=300)
         plt.close()
         print(f"仿真数据样本图已保存到: {save_path}")

     except Exception as e:
         print(f"绘制仿真数据样本时出错: {e}")
         
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
                            .replace('范围', 'Range')
        # Add more replacements as needed

    # Minimal check without plotting might be safer if display issues are frequent
    # For now, keep original intent if possible, but this is where you might simplify
    try:
        # Test if the text *might* render without crashing basic operations
        # This doesn't guarantee perfect display but avoids direct plot errors here
        _ = plt.figure(figsize=(0.1, 0.1)) # Create a tiny temporary figure
        plt.text(0.5, 0.5, text)          # Try adding the text
        plt.close(_)                      # Close the figure immediately
        return text # If no error, assume it's okay
    except Exception:
        print(f"Warning: Using fallback text for '{text}' due to potential rendering issues.")
        return fallback_text
# You can add other general utility functions here if needed.