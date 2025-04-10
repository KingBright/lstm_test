# evaluation.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Keep imports

# Import project modules
import config
import utils
# Import necessary functions if physics comparison is added back later
# from data_generation import PendulumSystem, run_simulation


# --- Plotting Training Curves (remains the same) ---
def plot_training_curves(train_losses, val_losses, save_dir=config.FIGURES_DIR):
    """绘制并保存训练和验证损失曲线。"""
    if not train_losses or not isinstance(train_losses, list) or \
       not val_losses or not isinstance(val_losses, list): print("Warning: Invalid loss lists."); return
    try:
        plt.figure(figsize=(10, 6)); epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=utils.safe_text('训练损失', 'Training Loss'))
        valid_epochs = [e for e, l in zip(epochs, val_losses) if l is not None and np.isfinite(l)]
        valid_val_losses = [l for l in val_losses if l is not None and np.isfinite(l)]
        if valid_val_losses: plt.plot(valid_epochs, valid_val_losses, label=utils.safe_text('验证损失', 'Validation Loss'))
        plt.title(utils.safe_text('模型训练过程中的损失', 'Model Loss During Training')); plt.xlabel(utils.safe_text('周期', 'Epoch')); plt.ylabel(utils.safe_text('均方误差 (MSE)', 'Mean Squared Error (MSE)'))
        plt.legend(); plt.grid(True)
        valid_train_losses = [l for l in train_losses if l is not None and np.isfinite(l) and l > 1e-9]
        if valid_train_losses and len(valid_train_losses) > 1 and (max(valid_train_losses) / max(1e-9, min(valid_train_losses))) > 100: plt.yscale('log'); print("Info: Using log scale for y-axis in training curves plot.")
        os.makedirs(save_dir, exist_ok=True); save_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300); plt.close()
        print(f"训练曲线图已保存到: {save_path}")
    except Exception as e: print(f"绘制训练曲线时出错: {e}"); plt.close()


# --- Evaluating Single-Step Loss (remains the same) ---
# Note: Criterion now compares sequences (Batch, K, Features)
def evaluate_model(model, data_loader, criterion, device=None):
    """在给定的数据加载器上评估模型并返回平均损失。"""
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    model.to(device); model.eval(); total_loss = 0.0; batches = 0; start_time = time.time()
    if data_loader is None: print("Warning: data_loader is None in evaluate_model."); return float('inf')
    try: loader_len = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else len(data_loader);
    except (TypeError, AttributeError): loader_len = 0
    if loader_len == 0 and hasattr(data_loader, 'dataset'): print("Warning: Evaluation dataset is empty."); return float('inf')
    with torch.no_grad():
        for inputs, targets in data_loader: # targets shape: (batch, K, output_size)
            inputs, targets = inputs.to(device), targets.to(device)
            try: outputs = model(inputs) # outputs shape: (batch, K, output_size)
            except Exception as e: print(f"Error during model evaluation forward pass: {e}"); return float('inf')
            loss = criterion(outputs, targets) # MSELoss compares element-wise
            if not torch.isfinite(loss): print("Warning: NaN or Inf loss detected."); return float('inf')
            total_loss += loss.item(); batches += 1
    avg_loss = total_loss / batches if batches > 0 else float('inf'); eval_time = time.time() - start_time
    if np.isfinite(avg_loss): print(f'Average Validation/Test Loss (MSE): {avg_loss:.6f}, Evaluation Time: {eval_time:.2f}s')
    else: print(f'Evaluation resulted in invalid average loss. Time: {eval_time:.2f}s')
    return avg_loss


# --- Multi-Step Prediction MODIFIED for Seq2Seq ---
def multi_step_prediction(model, initial_sequence, df_pred,
                          input_seq_len=config.INPUT_SEQ_LEN, # Use N from config
                          output_seq_len=config.OUTPUT_SEQ_LEN, # Use K from config
                          prediction_steps=None, # Total steps to generate
                          input_scaler=None, target_scaler=None, device=None):
    """
    使用训练好的 Seq2Seq 模型执行多步预测。
    每次预测 K 步，使用预测的第一步来前滚输入序列。

    Args:
        model (nn.Module): 训练好的 Seq2Seq 模型。
        initial_sequence (np.array): 起始的输入序列 (已缩放), shape (input_seq_len, features)。
        df_pred (pd.DataFrame): 包含未来真实值的 DataFrame (至少需要 tau, time, theta, theta_dot)
                                覆盖预测范围，索引应与预测步对齐 (0 to prediction_steps-1)。
        input_seq_len (int): 输入序列长度 (N)。
        output_seq_len (int): 模型输出序列的长度 (K)。
        prediction_steps (int): 要生成的总未来步数。如果为 None, 则预测 df_pred 的长度。
        input_scaler: 拟合好的输入缩放器。
        target_scaler: 拟合好的目标缩放器 (预测绝对状态)。
        device: Torch 设备。

    Returns:
        tuple: (predicted_states_original, true_states_original)
               predicted_states_original shape: (prediction_steps, output_size)
               true_states_original shape: (prediction_steps, output_size)
               如果预测失败则返回空数组。
    """
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    model.to(device); model.eval()

    # --- 输入验证 ---
    num_output_features = 2 # theta, dot
    if not isinstance(initial_sequence, np.ndarray) or initial_sequence.ndim != 2 or initial_sequence.shape[0] != input_seq_len:
         print(f"错误: 初始序列形状无效。期望 ({input_seq_len}, features), 得到 {initial_sequence.shape}")
         return np.empty((0, num_output_features)), np.empty((0, num_output_features))
    if prediction_steps is None: prediction_steps = len(df_pred)
    if df_pred.empty or len(df_pred) < prediction_steps:
         print(f"警告: df_pred 为空或短于请求步数 ({prediction_steps})。调整步数。")
         prediction_steps = len(df_pred)
    if prediction_steps <= 0: print("错误: 没有可预测的步数。"); return np.empty((0, num_output_features)), np.empty((0, num_output_features))
    if not hasattr(input_scaler, 'scale_') or not hasattr(target_scaler, 'scale_'): print("错误: 缩放器未拟合。"); return np.empty((0, num_output_features)), np.empty((0, num_output_features))
    # 假设 Seq2Seq 预测绝对状态
    is_delta_prediction = False

    current_sequence_np = initial_sequence.copy() # 缩放后的输入序列 [t-N+1, ..., t]
    predicted_states_list = [] # 存储 *反标准化后* 的预测状态 [t+1, t+2, ...]
    num_input_features = input_scaler.n_features_in_

    print(f"运行 Seq2Seq 多步预测，共 {prediction_steps} 步...")
    with torch.no_grad():
        for i in range(prediction_steps):
            current_sequence_tensor = torch.tensor(
                current_sequence_np.reshape(1, input_seq_len, num_input_features),
                dtype=torch.float32
            ).to(device)

            # --- 预测 (模型输出 K 步) ---
            try:
                # model_output_scaled shape: (1, K, output_size)
                model_output_scaled_sequence = model(current_sequence_tensor).cpu().numpy()[0]
                if not np.all(np.isfinite(model_output_scaled_sequence)): print(f"NaN/Inf detected at step {i}. Stopping."); break
                # 只取预测的 K 步中的第一步来前滚
                next_step_output_scaled = model_output_scaled_sequence[0, :] # Shape: (output_size,)
            except Exception as e: print(f"模型前向传播错误 step {i}: {e}"); break

            # --- 状态更新 (反标准化预测的第一步) ---
            try:
                current_state_unscaled = target_scaler.inverse_transform(next_step_output_scaled.reshape(1, -1))[0]
                predicted_states_list.append(current_state_unscaled) # 存储预测的 state(t+1)
            except Exception as e: print(f"状态反标准化错误 step {i}: {e}"); break

            # --- 准备下一步的输入 ---
            # 使用预测出的 state(t+1) 和已知的 tau(t+1) 来构建输入序列的最后一步
            try:
                # 需要 t+1 时刻的 tau (在 df_pred 的索引 i 处)
                next_tau_original = df_pred.iloc[i]['tau']
                # 处理 sin/cos 特征 (如果启用)
                pred_theta, pred_theta_dot = current_state_unscaled[0], current_state_unscaled[1]
                if config.USE_SINCOS_THETA:
                     next_input_features_unscaled = np.array([np.sin(pred_theta), np.cos(pred_theta), pred_theta_dot, next_tau_original])
                else:
                     next_input_features_unscaled = np.array([pred_theta, pred_theta_dot, next_tau_original])
                # 标准化这个新的特征向量
                next_step_features_scaled = input_scaler.transform(next_input_features_unscaled.reshape(1, -1))[0]
            except IndexError: print(f"警告: 在 df_pred 中找不到 tau 输入 step {i}. 停止。"); break
            except Exception as e: print(f"准备下一步输入时出错 step {i}: {e}"); break

            # 更新序列: 去掉最旧的一步, 加入最新的一步
            current_sequence_np = np.append(current_sequence_np[1:], next_step_features_scaled.reshape(1, -1), axis=0)
            # 循环结束

    # --- 处理结果 ---
    actual_prediction_steps = len(predicted_states_list)
    if actual_prediction_steps == 0: return np.empty((0, num_output_features)), np.empty((0, num_output_features))
    predicted_states_original = np.array(predicted_states_list)
    true_states_original = df_pred.iloc[:actual_prediction_steps][['theta', 'theta_dot']].values
    if len(predicted_states_original) != len(true_states_original): min_len = min(len(predicted_states_original), len(true_states_original)); predicted_states_original = predicted_states_original[:min_len]; true_states_original = true_states_original[:min_len]
    if len(predicted_states_original) > 0: mse = np.mean((predicted_states_original - true_states_original)**2); print(f"多步预测 MSE ({actual_prediction_steps} 步): {mse:.6f}")
    else: print("无法计算多步预测 MSE。")
    return predicted_states_original, true_states_original


# --- Plotting Multi-Step Prediction (保持不变, 绘制单次预测结果) ---
def plot_multi_step_prediction(time_vector, true_states, predicted_states,
                               physics_model_predictions=None, model_name="LSTM",
                               save_dir=config.FIGURES_DIR, filename_base="multi_step_prediction"):
    """绘制多步预测与真实轨迹的比较。"""
    # ... (绘图函数的实现保持不变，绘制6个子图) ...
    if not all(isinstance(arr, np.ndarray) for arr in [time_vector, true_states, predicted_states]) or \
       len(predicted_states) == 0 or len(true_states) == 0 or len(time_vector) == 0: print(f"Warning: Empty data for plotting '{filename_base}'."); return
    min_len = len(predicted_states); # ... (length adjustment logic) ...
    if len(true_states) != min_len or len(time_vector) != min_len: min_len = min(len(predicted_states), len(true_states), len(time_vector)); true_states = true_states[:min_len]; time_vector = time_vector[:min_len]; predicted_states = predicted_states[:min_len];
    if physics_model_predictions is not None and len(physics_model_predictions) != min_len: physics_model_predictions = physics_model_predictions[:min_len];
    if physics_model_predictions is not None and len(physics_model_predictions) != min_len: physics_model_predictions = None;
    if predicted_states.ndim == 1: predicted_states = predicted_states.reshape(-1, 1);
    if true_states.ndim == 1: true_states = true_states.reshape(-1, 1);
    if predicted_states.shape[1] < 2 or true_states.shape[1] < 2: print(f"Error plotting '{filename_base}': State arrays have < 2 columns."); return
    try:
        theta_error = np.abs(predicted_states[:, 0] - true_states[:, 0]); theta_dot_error = np.abs(predicted_states[:, 1] - true_states[:, 1]); mean_theta_error = np.mean(theta_error); mean_theta_dot_error = np.mean(theta_dot_error); mean_physics_theta_error, mean_physics_theta_dot_error = np.nan, np.nan; physics_theta_error, physics_theta_dot_error = None, None;
        if physics_model_predictions is not None:
            if physics_model_predictions.ndim == 1: physics_model_predictions = physics_model_predictions.reshape(-1, 1)
            if physics_model_predictions.shape[1] >= 2: physics_theta_error = np.abs(physics_model_predictions[:, 0] - true_states[:, 0]); physics_theta_dot_error = np.abs(physics_model_predictions[:, 1] - true_states[:, 1]); mean_physics_theta_error = np.mean(physics_theta_error); mean_physics_theta_dot_error = np.mean(physics_theta_dot_error)
            else: physics_model_predictions = None
    except Exception as calc_e: print(f"Error during error calculation for plot '{filename_base}': {calc_e}"); return
    try:
        fig = plt.figure(figsize=(15, 12), dpi=300); gs = plt.GridSpec(3, 2, figure=fig); fig.suptitle(utils.safe_text(f'多步预测分析: {model_name} 模型', f'Multi-step Prediction: {model_name} Model'), fontsize=16, fontweight='bold');
        ax1 = fig.add_subplot(gs[0, 0]); ax1.plot(time_vector, true_states[:, 0], 'g-', label=utils.safe_text('真实角度'), linewidth=1.5, alpha=0.8); ax1.plot(time_vector, predicted_states[:, 0], 'r--', label=utils.safe_text(f'预测角度 ({model_name})'), linewidth=1.5);
        if physics_model_predictions is not None: ax1.plot(time_vector, physics_model_predictions[:, 0], 'b-.', label=utils.safe_text('纯物理模型'), linewidth=1, alpha=0.6)
        ax1.set_title(utils.safe_text('角度 (θ) 预测'), fontsize=14); ax1.set_ylabel(utils.safe_text('角度 (rad)'), fontsize=12); ax1.text(0.02, 0.02, utils.safe_text(f'{model_name} Avg Err: {mean_theta_error:.4f} rad'), transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8)); ax1.legend(fontsize=9); ax1.grid(True)
        ax2 = fig.add_subplot(gs[0, 1]); ax2.plot(time_vector, true_states[:, 1], 'g-', label=utils.safe_text('真实角速度'), linewidth=1.5, alpha=0.8); ax2.plot(time_vector, predicted_states[:, 1], 'r--', label=utils.safe_text(f'预测角速度 ({model_name})'), linewidth=1.5);
        if physics_model_predictions is not None: ax2.plot(time_vector, physics_model_predictions[:, 1], 'b-.', label=utils.safe_text('纯物理模型'), linewidth=1, alpha=0.6)
        ax2.set_title(utils.safe_text('角速度 (θ̇) 预测'), fontsize=14); ax2.set_ylabel(utils.safe_text('角速度 (rad/s)'), fontsize=12); ax2.text(0.02, 0.02, utils.safe_text(f'{model_name} Avg Err: {mean_theta_dot_error:.4f} rad/s'), transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8)); ax2.legend(fontsize=9); ax2.grid(True)
        ax3 = fig.add_subplot(gs[1, 0]); ax3.plot(time_vector, theta_error, 'r-', label=utils.safe_text(f'{model_name} 角度误差'), linewidth=1.5);
        if physics_model_predictions is not None and physics_theta_error is not None: ax3.plot(time_vector, physics_theta_error, 'b-.', label=utils.safe_text('物理模型误差'), linewidth=1, alpha=0.7); ax3.axhline(y=mean_physics_theta_error, color='b', linestyle=':', alpha=0.6, label=utils.safe_text(f'物理AvgErr: {mean_physics_theta_error:.4f}'))
        ax3.set_title(utils.safe_text('角度预测误差'), fontsize=14); ax3.set_ylabel(utils.safe_text('|误差| (rad)'), fontsize=12); ax3.axhline(y=mean_theta_error, color='r', linestyle='--', alpha=0.7, label=utils.safe_text(f'{model_name} AvgErr: {mean_theta_error:.4f}')); ax3.legend(fontsize=9); ax3.grid(True)
        ax4 = fig.add_subplot(gs[1, 1]); ax4.plot(time_vector, theta_dot_error, 'r-', label=utils.safe_text(f'{model_name} 角速度误差'), linewidth=1.5);
        if physics_model_predictions is not None and physics_theta_dot_error is not None: ax4.plot(time_vector, physics_theta_dot_error, 'b-.', label=utils.safe_text('物理模型误差'), linewidth=1, alpha=0.7); ax4.axhline(y=mean_physics_theta_dot_error, color='b', linestyle=':', alpha=0.6, label=utils.safe_text(f'物理AvgErr: {mean_physics_theta_dot_error:.4f}'))
        ax4.set_title(utils.safe_text('角速度预测误差'), fontsize=14); ax4.set_ylabel(utils.safe_text('|误差| (rad/s)'), fontsize=12); ax4.axhline(y=mean_theta_dot_error, color='r', linestyle='--', alpha=0.7, label=utils.safe_text(f'{model_name} AvgErr: {mean_theta_dot_error:.4f}')); ax4.legend(fontsize=9); ax4.grid(True)
        ax5 = fig.add_subplot(gs[2, 0]); ax5.plot(true_states[:, 0], true_states[:, 1], 'g-', label=utils.safe_text('真实轨迹'), linewidth=1.5, alpha=0.7); ax5.plot(predicted_states[:, 0], predicted_states[:, 1], 'r--', label=utils.safe_text(f'预测轨迹 ({model_name})'), linewidth=1.5, alpha=0.9);
        if physics_model_predictions is not None: ax5.plot(physics_model_predictions[:, 0], physics_model_predictions[:, 1], 'b-.', label=utils.safe_text('纯物理轨迹'), linewidth=1, alpha=0.5)
        ax5.set_title(utils.safe_text('相位图: 角度 vs 角速度'), fontsize=14); ax5.set_xlabel(utils.safe_text('角度 (rad)'), fontsize=12); ax5.set_ylabel(utils.safe_text('角速度 (rad/s)'), fontsize=12); ax5.legend(fontsize=9); ax5.grid(True)
        ax6 = fig.add_subplot(gs[2, 1]); steps_axis = np.arange(1, len(theta_error) + 1); cum_error_theta = np.cumsum(theta_error) / steps_axis; cum_error_theta_dot = np.cumsum(theta_dot_error) / steps_axis;
        ax6.plot(time_vector, cum_error_theta, 'r-', label=utils.safe_text(f'累积角度误差 ({model_name})'), linewidth=1.5); ax6.plot(time_vector, cum_error_theta_dot, 'm-', label=utils.safe_text(f'累积角速度误差 ({model_name})'), linewidth=1.5);
        if physics_model_predictions is not None and physics_theta_error is not None and physics_theta_dot_error is not None: cum_error_physics_theta = np.cumsum(physics_theta_error) / steps_axis; cum_error_physics_theta_dot = np.cumsum(physics_theta_dot_error) / steps_axis; ax6.plot(time_vector, cum_error_physics_theta, 'b--', label=utils.safe_text('物理累积角度误差'), linewidth=1, alpha=0.7); ax6.plot(time_vector, cum_error_physics_theta_dot, 'c--', label=utils.safe_text('物理累积角速度误差'), linewidth=1, alpha=0.7)
        ax6.set_title(utils.safe_text('累积平均误差'), fontsize=14); ax6.set_xlabel(utils.safe_text('时间 (s)'), fontsize=12); ax6.set_ylabel(utils.safe_text('累积平均误差'), fontsize=12); ax6.legend(fontsize=9); ax6.grid(True); ax6.set_yscale('log')
        for ax in [ax1, ax2, ax3, ax4]: ax.set_xlabel(utils.safe_text('时间 (s)'), fontsize=12)
        fig.text(0.5, 0.01, utils.safe_text(f'预测步数: {len(time_vector)}   时间范围: {time_vector[0]:.2f}s - {time_vector[-1]:.2f}s', f'Steps: {len(time_vector)} Range: {time_vector[0]:.2f}s - {time_vector[-1]:.2f}s'), ha='center', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{filename_base}.png'); save_path_hq = os.path.join(save_dir, f'{filename_base}_hq.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e: print(f"Error plotting multi-step prediction for '{filename_base}': {e}"); import traceback; traceback.print_exc(); plt.close()

# --- Removed Consolidated Evaluation and Grid Plotting ---
# def run_multi_scenario_evaluation(...): ...
# def _plot_grid_internal(...): ...
# def plot_multi_scenario_grid_angle(...): ...
# def plot_multi_scenario_grid_velocity(...): ...

