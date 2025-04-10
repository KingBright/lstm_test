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

# --- setup_logging_and_warnings function ---
def setup_logging_and_warnings():
    """配置日志级别和警告过滤器。"""
    # 抑制 Matplotlib 字体警告和日志
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    # 抑制 PyTorch 序列化警告 (可选)
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
    # 尝试禁用 PyTorch 不安全用法警告 (适用于较新版本)
    os.environ['TORCH_WARN_ALWAYS_UNSAFE_USAGE'] = '0'
    print("日志和警告已配置。")

# --- safe_text function ---
def safe_text(text, fallback_text=None):
    """
    尝试使用提供的文本，如果渲染失败（例如由于字体问题），则回退到简化版本。
    """
    if fallback_text is None:
        # 基本回退：移除可能有问题的字符或提供简单的 ASCII
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
        # 根据需要添加更多替换
    try:
        # 最小化检查，无需实际绘图，减少潜在错误
        _ = plt.figure(figsize=(0.1, 0.1)) # 创建一个微小的临时图
        plt.text(0.5, 0.5, text)          # 尝试添加文本
        plt.close(_)                      # 立即关闭
        return text # 如果没有错误，假设文本可用
    except Exception:
        # print(f"警告: 因潜在渲染问题，为 '{text}' 使用回退文本。") # 可选的警告信息
        return fallback_text

# --- setup_chinese_font function ---
def setup_chinese_font():
    """根据操作系统设置 Matplotlib 以支持中文字符。"""
    system = platform.system()
    font_set = False
    print("正在尝试为 Matplotlib 设置中文字体...")
    try:
        if system == 'Darwin': # macOS
            font_list = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti']
            for font in font_list:
                try:
                    plt.rcParams['font.family'] = [font]
                    fig = plt.figure(figsize=(0.1, 0.1)); plt.text(0.5, 0.5, '测试'); plt.close(fig)
                    print(f"  成功: macOS 字体设置为 '{font}'")
                    font_set = True; break
                except Exception: continue
            if not font_set: print("  警告: 未找到合适的 macOS 中文字体。")
        elif system == 'Windows':
            font_list = ['SimHei', 'Microsoft YaHei', 'DengXian'] # 尝试常用 Windows 字体
            for font in font_list:
                 try:
                      plt.rcParams['font.family'] = [font]
                      fig = plt.figure(figsize=(0.1, 0.1)); plt.text(0.5, 0.5, '测试'); plt.close(fig)
                      print(f"  成功: Windows 字体设置为 '{font}'")
                      font_set = True; break
                 except Exception: continue
            if not font_set: print("  警告: 未找到合适的 Windows 中文字体。")
        elif system == 'Linux':
            font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Droid Sans Fallback'] # 检查常用 Linux 字体
            for font in font_list:
                 try:
                     plt.rcParams['font.family'] = [font]
                     fig = plt.figure(figsize=(0.1, 0.1)); plt.text(0.5, 0.5, '测试'); plt.close(fig)
                     print(f"  成功: Linux 字体设置为 '{font}'")
                     font_set = True; break
                 except Exception: continue
            if not font_set: print("  警告: 未找到合适的 Linux 中文字体。")

        # 应用通用设置或回退
        if font_set:
             plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
             plt.rcParams['font.size'] = 10 # 为可能的密集图表设置稍小的默认字体大小
        else:
             print("  警告: 使用默认 'sans-serif' 字体。中文字符可能无法正确显示。")
             plt.rcParams['font.family'] = ['sans-serif']
             plt.rcParams['axes.unicode_minus'] = False
             plt.rcParams['font.size'] = 10
             plt.rcParams['axes.titlepad'] = 15 # 以防万一增加标题边距

    except Exception as e:
        print(f"  设置中文字体时发生未知错误: {e}")
        # 确保基本的回退设置
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10

# --- plot_scenario_comparison function ---
def plot_scenario_comparison(scenario_name, train_dfs, val_dfs, save_dir=config.FIGURES_DIR):
    """
    为特定场景绘制训练和验证数据段的对比图（仅角度）。
    叠加显示不同初始条件的结果。

    Args:
        scenario_name (str): 场景名称 (例如 'sine', 'step')。
        train_dfs (list): 包含该场景下每个初始条件的训练数据 DataFrame 的列表。
        val_dfs (list): 包含该场景下每个初始条件的验证数据 DataFrame 的列表。
        save_dir (str): 保存绘图图像的目录。
    """
    if not train_dfs and not val_dfs:
        print(f"警告: 场景 '{scenario_name}' 没有训练或验证数据。跳过绘图。")
        return

    print(f"  正在为场景绘制对比图: {scenario_name}...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True) # 1 行, 2 列 (训练, 验证), 共享 Y 轴
        title = safe_text(f'场景数据对比 ({scenario_name}): 角度 θ', f'Scenario Data Comparison ({scenario_name}): Angle θ')
        fig.suptitle(title, fontsize=14)

        # 绘制训练数据段 (左侧)
        axes[0].set_title(safe_text(f"训练数据段 (T={config.T_SPAN_TRAIN[1]}s)", f"Train Data (T={config.T_SPAN_TRAIN[1]}s)"))
        if train_dfs:
            for i, df_train in enumerate(train_dfs):
                if not df_train.empty and 'time' in df_train and 'theta' in df_train:
                    # 用稍许透明度绘制每个初始条件
                    axes[0].plot(df_train['time'], df_train['theta'], linewidth=1, alpha=0.7, label=f'IC {i+1}' if i < 5 else None) # 仅标记前几个 IC
            if len(train_dfs) > 0: axes[0].legend(fontsize=8, loc='upper right') # 添加图例
        else:
            axes[0].text(0.5, 0.5, '无训练数据', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
        axes[0].set_xlabel(safe_text('时间 (s)', 'Time (s)'))
        axes[0].set_ylabel(safe_text('角度 (rad)', 'Angle (rad)'))
        axes[0].grid(True)

        # 绘制验证数据段 (右侧)
        axes[1].set_title(safe_text(f"验证数据段 (T={config.T_SPAN_VAL[1]}s)", f"Validation Data (T={config.T_SPAN_VAL[1]}s)"))
        if val_dfs:
            for i, df_val in enumerate(val_dfs):
                 if not df_val.empty and 'time' in df_val and 'theta' in df_val:
                    axes[1].plot(df_val['time'], df_val['theta'], linewidth=1, alpha=0.7, label=f'IC {i+1}' if i < 5 else None)
            # axes[1].legend(fontsize=8, loc='upper right') # 图例可能冗余
        else:
             axes[1].text(0.5, 0.5, '无验证数据', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
        axes[1].set_xlabel(safe_text('时间 (s)', 'Time (s)'))
        # axes[1].set_ylabel(safe_text('角度 (rad)', 'Angle (rad)')) # 共享 Y 轴
        axes[1].grid(True)

        # 保存
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # 调整布局
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'scenario_comparison_{scenario_name}.png')
        plt.savefig(save_path, dpi=150) # 对这些图使用稍低的分辨率
        plt.close()
        print(f"    -> 场景对比图已保存: {save_path}")

    except Exception as e:
        print(f"    绘制场景 '{scenario_name}' 对比图时出错: {e}")
        plt.close() # 确保即使出错也关闭图像


# +++ 物理校验函数 +++
def validate_physics(df, m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH,
                     g=config.GRAVITY, c=config.DAMPING_COEFF,
                     tolerance=config.PHYSICS_VALIDATION_TOLERANCE):
    """
    通过检查能量原理来验证 DataFrame 中的数据是否大致符合物理规律。
    检查 dE/dt <= P_input (来自力矩的功率) 是否在容忍度范围内成立。

    Args:
        df (pd.DataFrame): 包含 'time', 'theta', 'theta_dot', 'tau' 列的 DataFrame。
        m, L, g, c (float): 单摆参数。
        tolerance (float): 能量检查不等式允许的容忍度。

    Returns:
        bool: 如果数据看起来物理一致则返回 True，否则返回 False。
    """
    print("正在执行物理校验 (能量检查)...")
    required_cols = ['time', 'theta', 'theta_dot', 'tau']
    if not all(col in df.columns for col in required_cols):
        print("错误: DataFrame 缺少物理校验所需的列。")
        return False
    if len(df) < 2:
        print("警告: 数据点不足 (< 2)，无法进行物理校验。")
        return True # 无法校验，假设通过

    # 确保数据是 NumPy 数组以便计算
    time_vals = df['time'].values
    theta_vals = df['theta'].values
    theta_dot_vals = df['theta_dot'].values
    tau_vals = df['tau'].values

    # 检查数据中是否有 NaN 或 Inf
    if not np.all(np.isfinite(theta_vals)) or \
       not np.all(np.isfinite(theta_dot_vals)) or \
       not np.all(np.isfinite(tau_vals)):
        print("警告: 数据中包含 NaN 或 Inf，无法进行物理校验。")
        return False # 数据无效

    # 计算势能 (相对于最低点)
    potential_energy = m * g * L * (1 - np.cos(theta_vals))
    # 计算动能
    kinetic_energy = 0.5 * m * (L * theta_dot_vals)**2
    # 计算总机械能
    total_energy = potential_energy + kinetic_energy

    # 计算能量对时间的数值导数
    # np.gradient 使用中心差分（边界除外）
    time_diff = np.gradient(time_vals)
    # 避免除以零（如果时间步长相同）
    time_diff[time_diff == 0] = 1e-9 # 用一个很小的数代替零步长
    dE_dt_numerical = np.gradient(total_energy) / time_diff

    # 计算外部力矩输入的功率
    power_input = tau_vals * theta_dot_vals

    # 检查物理约束: dE/dt <= Power_input
    # 加入容忍度以考虑数值积分和微分带来的误差
    # 违反条件是 dE/dt 比 Power_input 大出 tolerance 以上
    violation_mask = dE_dt_numerical > (power_input + tolerance)
    num_violations = np.sum(violation_mask)

    if num_violations == 0:
        print(f"  物理校验通过: 能量变化与输入功率一致 (容忍度={tolerance})。")
        return True
    else:
        percentage_violation = (num_violations / len(df)) * 100
        print(f"  物理校验失败: 发现 {num_violations} ({percentage_violation:.2f}%) 个点"
              f" 满足 dE/dt > Power_input + tolerance。")
        # 可选：打印前几个违反点的详细信息
        violation_indices = np.where(violation_mask)[0]
        print(f"  违规示例 (dE/dt vs Power_in + tol) 位于索引:")
        for idx in violation_indices[:min(5, len(violation_indices))]: # 显示前 5 个
             # 检查索引是否有效
             if idx < len(dE_dt_numerical) and idx < len(power_input):
                 print(f"    索引 {idx}: dE/dt={dE_dt_numerical[idx]:.4f}, P_in+tol={power_input[idx] + tolerance:.4f}")
             else:
                 print(f"    索引 {idx} 无效 (超出范围)")
        return False

