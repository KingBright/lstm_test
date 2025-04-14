# evaluation.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import config
import utils
from data_generation import PendulumSystem, run_simulation # Keep for physics comparison

# --- Plotting Training Curves ---
# (No changes)
def plot_training_curves(train_losses, val_losses, save_dir=config.FIGURES_DIR):
    # ... (代码同上) ...
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


# --- Evaluating Single-Step Loss ---
# (No changes)
def evaluate_model(model, data_loader, criterion, device=None):
    # ... (代码同上) ...
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    model.to(device); model.eval(); total_loss = 0.0; batches = 0; start_time = time.time()
    if data_loader is None: print("警告: evaluate_model中data_loader为None。"); return float('inf')
    try: loader_len = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else len(data_loader)
    except (TypeError, AttributeError): loader_len = 0
    if loader_len == 0 and hasattr(data_loader, 'dataset'): print("警告: 评估数据集为空。"); return float('inf')

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True) # Targets shape (batch, K, output_features)
            try: outputs = model(inputs) # Outputs shape (batch, K, output_features)
            except Exception as e: print(f"模型评估前向传播时出错: {e}"); return float('inf')
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss): print("警告: 检测到NaN或Inf损失。"); return float('inf')
            total_loss += loss.item(); batches += 1
    if device.type == 'mps': torch.mps.synchronize()
    avg_loss = total_loss / batches if batches > 0 else float('inf')
    eval_time = time.time() - start_time
    if np.isfinite(avg_loss): print(f'Average Validation/Test Loss (MSE): {avg_loss:.6f}, Evaluation Time: {eval_time:.2f}s')
    else: print(f'Evaluation resulted in invalid average loss. Time: {eval_time:.2f}s')
    return avg_loss


# --- Multi-Step Prediction (MODIFIED FOR SIN/COS OUTPUT) ---
def multi_step_prediction(model, initial_sequence_scaled, df_pred,
                          input_seq_len=config.INPUT_SEQ_LEN,
                          output_seq_len=config.OUTPUT_SEQ_LEN,
                          prediction_steps=None,
                          input_scaler=None, target_scaler=None,
                          device=None,
                          predict_delta=config.PREDICT_DELTA, # Keep for consistency, should be False
                          predict_sincos_output=config.PREDICT_SINCOS_OUTPUT): # Use new flag
    """
    Performs multi-step prediction. Handles absolute, delta, or sin/cos output.
    """
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    model.to(device); model.eval()

    # Determine expected number of output features from model
    expected_output_features = 3 if predict_sincos_output else 2

    # Input Validation
    if not isinstance(initial_sequence_scaled, np.ndarray) or initial_sequence_scaled.ndim != 2 or initial_sequence_scaled.shape[0] != input_seq_len:
         print(f"Error: Invalid initial sequence shape."); return np.empty((0, 2)), np.empty((0, 2))
    if prediction_steps is None: prediction_steps = len(df_pred)
    if df_pred.empty or len(df_pred) < prediction_steps:
         print(f"Warning: df_pred empty or shorter than requested. Adjusting steps."); prediction_steps = len(df_pred)
    if prediction_steps <= 0: print("Error: No steps to predict."); return np.empty((0, 2)), np.empty((0, 2))
    if not hasattr(input_scaler, 'scale_') or not hasattr(target_scaler, 'scale_'): print("Error: Scalers not fitted."); return np.empty((0, 2)), np.empty((0, 2))

    current_sequence_np_scaled = initial_sequence_scaled.copy()
    predicted_states_absolute_list = [] # Store final [theta, theta_dot] predictions
    num_input_features = input_scaler.n_features_in_

    # Get initial state [theta, theta_dot] for accumulation/next input prep
    current_absolute_state_unscaled = np.zeros(2) # [theta, theta_dot]
    try:
        last_step_input_scaled = current_sequence_np_scaled[-1, :].reshape(1, -1)
        last_step_input_unscaled = input_scaler.inverse_transform(last_step_input_scaled)[0]
        if config.USE_SINCOS_THETA:
             last_theta = np.arctan2(last_step_input_unscaled[0], last_step_input_unscaled[1])
             last_theta_dot = last_step_input_unscaled[2]
        else:
             last_theta = last_step_input_unscaled[0]
             last_theta_dot = last_step_input_unscaled[1]
        current_absolute_state_unscaled = np.array([last_theta, last_theta_dot])
    except Exception as e:
        print(f"Error getting initial state: {e}"); return np.empty((0, 2)), np.empty((0, 2))

    print(f"Running multi-step prediction (Predict SinCos: {predict_sincos_output}, Predict Delta: {predict_delta})...")
    # --- Prediction Loop ---
    with torch.no_grad():
        for i in range(prediction_steps):
            current_sequence_tensor = torch.tensor(
                current_sequence_np_scaled.reshape(1, input_seq_len, num_input_features),
                dtype=torch.float32
            ).to(device, non_blocking=True)

            try:
                model_output = model(current_sequence_tensor) # Shape (1, K, output_features)
                # Output features = 3 if predict_sincos_output else 2
                predicted_sequence_scaled = model_output.cpu().numpy()[0] # Shape: (K, output_features)

                # *** DIAGNOSTIC PRINT ***
                # if i < 5: print(f"  Step {i+1} Raw Scaled Output (First Step): {predicted_sequence_scaled[0, :]}")

                if not np.all(np.isfinite(predicted_sequence_scaled)):
                    print(f"NaN/Inf detected in model output at step {i}. Stopping."); break
            except Exception as e: print(f"Model prediction error at step {i}: {e}"); break

            # --- Determine next absolute state [theta, theta_dot] ---
            next_absolute_state_unscaled = np.zeros(2) # Initialize [theta, theta_dot]

            if predict_sincos_output:
                # Model outputs scaled [sin, cos, dot] for K steps
                try:
                    predicted_output_unscaled = target_scaler.inverse_transform(predicted_sequence_scaled)
                    # Take the first predicted step [pred_sin, pred_cos, pred_thetadot]
                    pred_sin_k0, pred_cos_k0, pred_thetadot_k0 = predicted_output_unscaled[0, :]
                    # Reconstruct angle using arctan2
                    pred_theta_k0 = np.arctan2(pred_sin_k0, pred_cos_k0)
                    # This is the absolute state for the next step
                    next_absolute_state_unscaled = np.array([pred_theta_k0, pred_thetadot_k0])
                    # No need to wrap angle here, atan2 naturally returns in [-pi, pi]
                except Exception as e: print(f"Error processing sin/cos output at step {i}: {e}"); break

            elif predict_delta:
                # Model outputs scaled [delta_theta, delta_dot] for K steps
                try:
                    predicted_deltas_unscaled = target_scaler.inverse_transform(predicted_sequence_scaled)
                    # Accumulate the first delta
                    next_absolute_state_unscaled = current_absolute_state_unscaled + predicted_deltas_unscaled[0, :]
                    next_absolute_state_unscaled[0] = (next_absolute_state_unscaled[0] + np.pi) % (2 * np.pi) - np.pi # Wrap angle
                except Exception as e: print(f"Error processing delta at step {i}: {e}"); break
            else: # Predict absolute [theta, dot]
                try:
                    predicted_abs_unscaled = target_scaler.inverse_transform(predicted_sequence_scaled)
                    next_absolute_state_unscaled = predicted_abs_unscaled[0, :] # Take first step
                    next_absolute_state_unscaled[0] = (next_absolute_state_unscaled[0] + np.pi) % (2 * np.pi) - np.pi # Wrap angle
                except Exception as e: print(f"Error processing absolute state at step {i}: {e}"); break

            # *** DIAGNOSTIC PRINT ***
            # if i < 5: print(f"  Step {i+1} Predicted Unscaled State [Theta, ThetaDot]: {next_absolute_state_unscaled}")

            # Store the final absolute state prediction
            predicted_states_absolute_list.append(next_absolute_state_unscaled)
            # Update the current state for the next iteration's accumulation (if needed)
            current_absolute_state_unscaled = next_absolute_state_unscaled.copy()

            # --- Prepare Next Input ---
            try:
                next_tau_original = df_pred.iloc[i]['tau']
                pred_theta, pred_thetadot = next_absolute_state_unscaled[0], next_absolute_state_unscaled[1]
                # Construct features based on USE_SINCOS_THETA flag for input
                if config.USE_SINCOS_THETA:
                     next_input_features_unscaled = np.array([np.sin(pred_theta), np.cos(pred_theta), pred_thetadot, next_tau_original])
                else:
                     next_input_features_unscaled = np.array([pred_theta, pred_thetadot, next_tau_original])
                # Scale the features for the next input step
                next_step_features_scaled = input_scaler.transform(next_input_features_unscaled.reshape(1, -1))[0]
            except IndexError: print(f"Error: Cannot get tau for step {i+1}. Stopping."); break
            except Exception as e: print(f"Error preparing next input at step {i}: {e}"); break

            # Roll the sequence
            current_sequence_np_scaled = np.append(current_sequence_np_scaled[1:], next_step_features_scaled.reshape(1, -1), axis=0)

    # Process Results
    actual_prediction_steps = len(predicted_states_absolute_list)
    if actual_prediction_steps == 0: return np.empty((0, 2)), np.empty((0, 2))
    predicted_states_absolute = np.array(predicted_states_absolute_list)
    true_states_absolute = df_pred.iloc[:actual_prediction_steps][['theta', 'theta_dot']].values
    min_len = min(len(predicted_states_absolute), len(true_states_absolute))
    predicted_states_absolute = predicted_states_absolute[:min_len]
    true_states_absolute = true_states_absolute[:min_len]
    if min_len > 0:
        mse = np.mean((predicted_states_absolute - true_states_absolute)**2)
        print(f"Multi-step prediction finished. Steps: {min_len}, Final Abs State MSE: {mse:.6f}")
    else: print("Multi-step prediction yielded no comparable steps.")
    return predicted_states_absolute, true_states_absolute


# --- Plotting Multi-Step Prediction ---
# (No changes needed, plots absolute theta/theta_dot)
def plot_multi_step_prediction(time_vector, true_states, predicted_states,
                               physics_model_predictions=None, model_name="LSTM",
                               save_dir=config.FIGURES_DIR, filename_base="multi_step_prediction"):
    # ... (代码同上) ...
    utils.setup_chinese_font()
    if not all(isinstance(arr, np.ndarray) for arr in [time_vector, true_states, predicted_states]) or \
       len(predicted_states) == 0 or len(true_states) == 0 or len(time_vector) == 0: print(f"Warning: Empty data for plotting '{filename_base}'."); return
    min_len = len(predicted_states)
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
        fig = plt.figure(figsize=(15, 12), dpi=150); gs = plt.GridSpec(3, 2, figure=fig); fig.suptitle(utils.safe_text(f'多步预测分析: {model_name} 模型', f'Multi-step Prediction: {model_name} Model'), fontsize=16, fontweight='bold');
        ax1 = fig.add_subplot(gs[0, 0]); ax1.plot(time_vector, true_states[:, 0], 'g-', label=utils.safe_text('真实角度'), linewidth=1.5, alpha=0.8); ax1.plot(time_vector, predicted_states[:, 0], 'r--', label=utils.safe_text(f'预测角度 ({model_name})'), linewidth=1.5);
        if physics_model_predictions is not None: ax1.plot(time_vector, physics_model_predictions[:, 0], 'b-.', label=utils.safe_text('纯物理模型'), linewidth=1, alpha=0.6)
        ax1.set_title(utils.safe_text('角度 (θ) 预测')); ax1.set_ylabel(utils.safe_text('角度 (rad)')); ax1.text(0.02, 0.02, utils.safe_text(f'{model_name} Avg Err: {mean_theta_error:.4f} rad'), transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8)); ax1.legend(fontsize=9); ax1.grid(True)
        ax2 = fig.add_subplot(gs[0, 1]); ax2.plot(time_vector, true_states[:, 1], 'g-', label=utils.safe_text('真实角速度'), linewidth=1.5, alpha=0.8); ax2.plot(time_vector, predicted_states[:, 1], 'r--', label=utils.safe_text(f'预测角速度 ({model_name})'), linewidth=1.5);
        if physics_model_predictions is not None: ax2.plot(time_vector, physics_model_predictions[:, 1], 'b-.', label=utils.safe_text('纯物理模型'), linewidth=1, alpha=0.6)
        ax2.set_title(utils.safe_text('角速度 (θ̇) 预测')); ax2.set_ylabel(utils.safe_text('角速度 (rad/s)')); ax2.text(0.02, 0.02, utils.safe_text(f'{model_name} Avg Err: {mean_theta_dot_error:.4f} rad/s'), transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8)); ax2.legend(fontsize=9); ax2.grid(True)
        ax3 = fig.add_subplot(gs[1, 0]); ax3.plot(time_vector, theta_error, 'r-', label=utils.safe_text(f'{model_name} 角度误差'), linewidth=1.5);
        if physics_model_predictions is not None and physics_theta_error is not None: ax3.plot(time_vector, physics_theta_error, 'b-.', label=utils.safe_text('物理模型误差'), linewidth=1, alpha=0.7); ax3.axhline(y=mean_physics_theta_error, color='b', linestyle=':', alpha=0.6, label=utils.safe_text(f'物理AvgErr: {mean_physics_theta_error:.4f}'))
        ax3.set_title(utils.safe_text('角度预测误差')); ax3.set_ylabel(utils.safe_text('|误差| (rad)')); ax3.axhline(y=mean_theta_error, color='r', linestyle='--', alpha=0.7, label=utils.safe_text(f'{model_name} AvgErr: {mean_theta_error:.4f}')); ax3.legend(fontsize=9); ax3.grid(True)
        ax4 = fig.add_subplot(gs[1, 1]); ax4.plot(time_vector, theta_dot_error, 'r-', label=utils.safe_text(f'{model_name} 角速度误差'), linewidth=1.5);
        if physics_model_predictions is not None and physics_theta_dot_error is not None: ax4.plot(time_vector, physics_theta_dot_error, 'b-.', label=utils.safe_text('物理模型误差'), linewidth=1, alpha=0.7); ax4.axhline(y=mean_physics_theta_dot_error, color='b', linestyle=':', alpha=0.6, label=utils.safe_text(f'物理AvgErr: {mean_physics_theta_dot_error:.4f}'))
        ax4.set_title(utils.safe_text('角速度预测误差')); ax4.set_ylabel(utils.safe_text('|误差| (rad/s)')); ax4.axhline(y=mean_theta_dot_error, color='r', linestyle='--', alpha=0.7, label=utils.safe_text(f'{model_name} AvgErr: {mean_theta_dot_error:.4f}')); ax4.legend(fontsize=9); ax4.grid(True)
        ax5 = fig.add_subplot(gs[2, 0]); ax5.plot(true_states[:, 0], true_states[:, 1], 'g-', label=utils.safe_text('真实轨迹'), linewidth=1.5, alpha=0.7); ax5.plot(predicted_states[:, 0], predicted_states[:, 1], 'r--', label=utils.safe_text(f'预测轨迹 ({model_name})'), linewidth=1.5, alpha=0.9);
        if physics_model_predictions is not None: ax5.plot(physics_model_predictions[:, 0], physics_model_predictions[:, 1], 'b-.', label=utils.safe_text('纯物理轨迹'), linewidth=1, alpha=0.5)
        ax5.set_title(utils.safe_text('相位图: 角度 vs 角速度')); ax5.set_xlabel(utils.safe_text('角度 (rad)')); ax5.set_ylabel(utils.safe_text('角速度 (rad/s)')); ax5.legend(fontsize=9); ax5.grid(True)
        ax6 = fig.add_subplot(gs[2, 1]); steps_axis = np.arange(1, len(theta_error) + 1); cum_error_theta = np.cumsum(theta_error) / steps_axis; cum_error_theta_dot = np.cumsum(theta_dot_error) / steps_axis;
        ax6.plot(time_vector, cum_error_theta, 'r-', label=utils.safe_text(f'累积角度误差 ({model_name})'), linewidth=1.5); ax6.plot(time_vector, cum_error_theta_dot, 'm-', label=utils.safe_text(f'累积角速度误差 ({model_name})'), linewidth=1.5);
        if physics_model_predictions is not None and physics_theta_error is not None and physics_theta_dot_error is not None: cum_error_physics_theta = np.cumsum(physics_theta_error) / steps_axis; cum_error_physics_theta_dot = np.cumsum(physics_theta_dot_error) / steps_axis; ax6.plot(time_vector, cum_error_physics_theta, 'b--', label=utils.safe_text('物理累积角度误差'), linewidth=1, alpha=0.7); ax6.plot(time_vector, cum_error_physics_theta_dot, 'c--', label=utils.safe_text('物理累积角速度误差'), linewidth=1, alpha=0.7)
        ax6.set_title(utils.safe_text('累积平均误差')); ax6.set_xlabel(utils.safe_text('时间 (s)')); ax6.set_ylabel(utils.safe_text('累积平均误差')); ax6.legend(fontsize=9); ax6.grid(True); ax6.set_yscale('log')
        for ax in [ax1, ax2, ax3, ax4]: ax.set_xlabel(utils.safe_text('时间 (s)'))
        fig.text(0.5, 0.01, utils.safe_text(f'预测步数: {len(time_vector)}   时间范围: {time_vector[0]:.2f}s - {time_vector[-1]:.2f}s', f'Steps: {len(time_vector)} Range: {time_vector[0]:.2f}s - {time_vector[-1]:.2f}s'), ha='center', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{filename_base}.png');
        plt.savefig(save_path, dpi=150); plt.close()
    except Exception as e: print(f"Error plotting multi-step prediction for '{filename_base}': {e}"); import traceback; traceback.print_exc(); plt.close()

