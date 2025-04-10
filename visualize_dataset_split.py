# visualize_dataset_split.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import project modules for configuration and utilities
import config
import utils

def visualize_split(data_path=config.DATA_FILE_EXTENDED,
                      val_split=config.VALIDATION_SPLIT,
                      save_dir=config.FIGURES_DIR):
    """
    Loads the full dataset and plots the time series, visually separating
    the approximate time periods corresponding to the training and validation splits.

    Args:
        data_path (str): Path to the dataset CSV file.
        val_split (float): The proportion used for the validation set.
        save_dir (str): Directory to save the plot image.
    """
    print(f"Loading dataset from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            print("Error: Loaded DataFrame is empty.")
            return
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{data_path}'.")
        print("Please ensure 'main_experiment.py' has been run successfully to generate the data file.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Calculate the approximate split index in the DataFrame based on time
    # Note: This is a chronological split of the raw data points for visualization.
    # The actual train/val sequences might be different due to sequence creation
    # and stratified random sampling in create_stratified_train_val_data.
    split_df_index = int(len(df) * (1 - val_split))
    if split_df_index <= 0 or split_df_index >= len(df):
        print(f"Warning: Calculated split index ({split_df_index}) is invalid for DataFrame length ({len(df)}). "
              f"Check val_split value ({val_split}). Plotting entire dataset as training.")
        split_df_index = len(df) # Plot everything as training data

    print(f"Approximate split point for visualization: index {split_df_index} (Time ~ {df['time'].iloc[split_df_index-1]:.2f}s)")

    # Prepare data portions for plotting
    df_train_part = df.iloc[:split_df_index]
    df_val_part = df.iloc[split_df_index:]

    # Setup plotting
    utils.setup_chinese_font() # Ensure Chinese characters display correctly
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True) # 3 rows, 1 column, shared x-axis
    fig.suptitle(utils.safe_text(f'数据集可视化 (训练/验证分割点 ~{df_train_part["time"].iloc[-1]:.2f}s)',
                                f'Dataset Visualization (Train/Val Split Point ~{df_train_part["time"].iloc[-1]:.2f}s)'),
                 fontsize=16)

    # Plot Theta (Angle)
    axes[0].plot(df_train_part['time'], df_train_part['theta'], label=utils.safe_text('训练数据段', 'Train Period'), color='tab:blue', linewidth=1)
    if not df_val_part.empty:
        axes[0].plot(df_val_part['time'], df_val_part['theta'], label=utils.safe_text('验证数据段', 'Validation Period'), color='tab:orange', linewidth=1)
    axes[0].set_ylabel(utils.safe_text('角度 (rad)', 'Angle (rad)'))
    axes[0].set_title(utils.safe_text('角度 (θ) 时间序列', 'Angle (θ) Time Series'))
    axes[0].grid(True)
    axes[0].legend(loc='upper right')

    # Plot Theta_dot (Angular Velocity)
    axes[1].plot(df_train_part['time'], df_train_part['theta_dot'], label=utils.safe_text('训练数据段', 'Train Period'), color='tab:blue', linewidth=1)
    if not df_val_part.empty:
        axes[1].plot(df_val_part['time'], df_val_part['theta_dot'], label=utils.safe_text('验证数据段', 'Validation Period'), color='tab:orange', linewidth=1)
    axes[1].set_ylabel(utils.safe_text('角速度 (rad/s)', 'Angular Velocity (rad/s)'))
    axes[1].set_title(utils.safe_text('角速度 (θ̇) 时间序列', 'Angular Velocity (θ̇) Time Series'))
    axes[1].grid(True)
    axes[1].legend(loc='upper right')

    # Plot Tau (Torque)
    axes[2].plot(df_train_part['time'], df_train_part['tau'], label=utils.safe_text('训练数据段', 'Train Period'), color='tab:blue', linewidth=1)
    if not df_val_part.empty:
        axes[2].plot(df_val_part['time'], df_val_part['tau'], label=utils.safe_text('验证数据段', 'Validation Period'), color='tab:orange', linewidth=1)
    axes[2].set_ylabel(utils.safe_text('力矩 (Nm)', 'Torque (Nm)'))
    axes[2].set_title(utils.safe_text('输入力矩 (τ) 时间序列', 'Input Torque (τ) Time Series'))
    axes[2].set_xlabel(utils.safe_text('时间 (s)', 'Time (s)'))
    axes[2].grid(True)
    axes[2].legend(loc='upper right')

    # Save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'dataset_split_visualization.png')
    try:
        plt.savefig(save_path, dpi=200) # Use slightly lower DPI for potentially large plot
        plt.close()
        print(f"数据集可视化图表已保存到: {save_path}")
    except Exception as e:
        print(f"保存图表时出错: {e}")
        plt.close()


if __name__ == "__main__":
    # Ensure utilities like font setup run if script is executed directly
    utils.setup_logging_and_warnings()
    # Call the main visualization function
    visualize_split()
