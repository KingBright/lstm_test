# evaluation.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
# Import scalers used for checking instance type
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import config # Import config
import utils # For safe_text

# --- Plotting Training Curves ---
def plot_training_curves(train_losses, val_losses, save_dir=config.FIGURES_DIR):
    """
    Plots and saves the training and validation loss curves.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        save_dir (str): Directory to save the plot image.
    """
    # 检查损失列表是否有效
    if not train_losses or not isinstance(train_losses, list) or \
       not val_losses or not isinstance(val_losses, list):
        print("Warning: Invalid or empty loss lists provided. Skipping training curve plot.")
        return

    try:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1) # X轴为周期数

        # 绘制训练损失
        plt.plot(epochs, train_losses, label=utils.safe_text('训练损失', 'Training Loss'))

        # 筛选有效的验证损失点进行绘制 (忽略 NaN 或 Inf)
        valid_epochs = [e for e, l in zip(epochs, val_losses) if l is not None and np.isfinite(l)]
        valid_val_losses = [l for l in val_losses if l is not None and np.isfinite(l)]
        if valid_val_losses:
             plt.plot(valid_epochs, valid_val_losses, label=utils.safe_text('验证损失', 'Validation Loss'))

        plt.title(utils.safe_text('模型训练过程中的损失', 'Model Loss During Training'))
        plt.xlabel(utils.safe_text('周期', 'Epoch'))
        plt.ylabel(utils.safe_text('均方误差 (MSE)', 'Mean Squared Error (MSE)'))
        plt.legend()
        plt.grid(True)

        # 检查是否需要使用对数坐标轴
        valid_train_losses = [l for l in train_losses if l is not None and np.isfinite(l) and l > 1e-9]
        if valid_train_losses and len(valid_train_losses) > 1 and (max(valid_train_losses) / max(1e-9, min(valid_train_losses))) > 100:
             plt.yscale('log')
             print("Info: Using log scale for y-axis in training curves plot.")

        # 保存图像
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300)
        plt.close() # 关闭图像，释放内存
        print(f"训练曲线图已保存到: {save_path}")

    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")
        plt.close() # 确保即使出错也关闭图像

# --- Evaluating Single-Step Loss ---
def evaluate_model(model, data_loader, criterion, device=None):
    """
    Evaluates the model on a given dataset loader and returns the average loss.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader or None): DataLoader for the dataset to evaluate.
        criterion (nn.Module): The loss function (e.g., nn.MSELoss).
        device (torch.device): The device to run evaluation on.

    Returns:
        float: The average loss over the dataset. Returns float('inf') if evaluation fails.
    """
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    model.to(device)
    model.eval() # 设置为评估模式

    total_loss = 0.0
    batches = 0
    start_time = time.time()

    # 检查 data_loader 是否有效
    if data_loader is None:
        print("Warning: data_loader is None in evaluate_model. Cannot evaluate.")
        return float('inf')
    try:
         loader_len = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else len(data_loader)
         if loader_len == 0:
              print("Warning: Evaluation dataset is empty.")
              return float('inf')
    except (TypeError, AttributeError):
         print("Warning: Could not determine evaluation dataset length.")

    with torch.no_grad(): # 评估时不需要计算梯度
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            try:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if not torch.isfinite(loss):
                     print("Warning: NaN or Inf loss detected during evaluation.")
                     return float('inf')
                total_loss += loss.item()
                batches += 1
            except Exception as e:
                 print(f"Error during model evaluation forward pass: {e}")
                 return float('inf')

    avg_loss = total_loss / batches if batches > 0 else float('inf')
    eval_time = time.time() - start_time

    if np.isfinite(avg_loss):
        print(f'Average Validation/Test Loss (MSE): {avg_loss:.6f}, Evaluation Time: {eval_time:.2f}s')
    else:
        print(f'Evaluation resulted in invalid average loss. Time: {eval_time:.2f}s')

    return avg_loss


# --- Multi-Step Prediction (Handles Delta Prediction) ---
def multi_step_prediction(model, initial_sequence, df_pred, sequence_length, prediction_steps,
                         input_scaler, target_scaler, device=None):
    """
    Performs multi-step prediction using a trained model.
    Handles both absolute state prediction and delta state prediction based on target_scaler type.

    Args:
        model (nn.Module): Trained model.
        initial_sequence (np.array): Starting sequence (scaled), shape (seq_len, features).
        df_pred (pd.DataFrame): DataFrame with future ground truth (at least theta, theta_dot, tau, time)
                                covering the prediction horizon. Index should align with prediction steps (0 to N-1).
        sequence_length (int): Length of input sequence.
        prediction_steps (int): Number of steps to predict.
        input_scaler: Fitted scaler for input features.
        target_scaler: Fitted scaler for the target variable (state or delta).
        device: Torch device.

    Returns:
        tuple: (predicted_states_original, true_states_original)
               predicted_states_original shape: (prediction_steps, output_size)
               true_states_original shape: (prediction_steps, output_size)
               Returns empty arrays if prediction fails.
    """
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    model.to(device); model.eval()

    # --- Input Validation ---
    num_output_features = 2 # Assuming theta, theta_dot output or delta
    if not isinstance(initial_sequence, np.ndarray) or initial_sequence.ndim != 2 or initial_sequence.shape[0] != sequence_length:
         print(f"Error: Initial sequence shape invalid. Expected ({sequence_length}, features), got {initial_sequence.shape}")
         return np.empty((0, num_output_features)), np.empty((0, num_output_features))
    if df_pred.empty or len(df_pred) < prediction_steps:
         print(f"Warning: Prediction dataframe 'df_pred' empty or shorter ({len(df_pred)}) than steps ({prediction_steps}). Adjusting.")
         prediction_steps = len(df_pred)
         if prediction_steps <= 0: print("Error: No data available for prediction steps."); return np.empty((0, num_output_features)), np.empty((0, num_output_features))
    if not hasattr(input_scaler, 'scale_') or not hasattr(target_scaler, 'scale_'):
         print("Error: Input or target scaler is not fitted."); return np.empty((0, num_output_features)), np.empty((0, num_output_features))

    # Determine prediction mode based on scaler type
    is_delta_prediction = isinstance(target_scaler, StandardScaler)
    if is_delta_prediction: print("Running multi-step prediction in DELTA prediction mode.")
    else: print("Running multi-step prediction in ABSOLUTE state prediction mode.")

    current_sequence_np = initial_sequence.copy() # Scaled input sequence [t-N+1, ..., t]
    predicted_states_list = [] # Store *unscaled* predicted states [t+1, t+2, ...]

    num_input_features = input_scaler.n_features_in_

    # --- Initial State for Delta Reconstruction ---
    # Initialize last_unscaled_state. For delta prediction, this needs to be
    # the *true* state at the time step *before* the first prediction (time t).
    # We get this from the row in the *original* dataframe corresponding to the
    # end of the initial sequence. This requires passing the index or the state itself.
    # WORKAROUND: Use the first true state from df_pred (which is state t+1)
    # and use it to reconstruct state t+2 from delta(t+2). The first predicted point
    # will effectively be the true state t+1.
    last_unscaled_state = np.zeros(num_output_features) # Placeholder
    first_step_in_delta = True # Flag for delta mode initialization

    print(f"Running multi-step prediction loop for {prediction_steps} steps...")
    with torch.no_grad():
        for i in range(prediction_steps):
            current_sequence_tensor = torch.tensor(
                current_sequence_np.reshape(1, sequence_length, num_input_features),
                dtype=torch.float32
            ).to(device)

            # --- Prediction ---
            try:
                model_output_scaled = model(current_sequence_tensor).cpu().numpy()[0]
                if not np.all(np.isfinite(model_output_scaled)): print(f"NaN/Inf detected at step {i}. Stopping."); break
            except Exception as e: print(f"Error during model forward pass at step {i}: {e}"); break

            # --- State Reconstruction ---
            try:
                # Inverse transform the model output (state or delta)
                model_output_unscaled = target_scaler.inverse_transform(model_output_scaled.reshape(1, -1))[0]

                if is_delta_prediction:
                    if first_step_in_delta:
                        # Use the first TRUE state from df_pred (state at t+1) as the first point
                        current_state_unscaled = df_pred.iloc[0][['theta', 'theta_dot']].values
                        print(f"Delta Mode: Using true state at step 0 as starting point: {current_state_unscaled}")
                        first_step_in_delta = False
                    else:
                        # state(t+1) = state(t) + delta(t+1)
                        current_state_unscaled = last_unscaled_state + model_output_unscaled

                    # Store the reconstructed state
                    predicted_states_list.append(current_state_unscaled)
                    # Update the state for the next iteration's delta calculation
                    last_unscaled_state = current_state_unscaled.copy()

                else: # Absolute state prediction
                    current_state_unscaled = model_output_unscaled
                    predicted_states_list.append(current_state_unscaled)
                    # In absolute mode, last_unscaled_state isn't needed for reconstruction,
                    # but we use current_state_unscaled to build the next input.

            except Exception as e: print(f"Error during state reconstruction at step {i}: {e}"); break

            # --- Prepare Input for Next Step ---
            # Use the *reconstructed* or *predicted absolute* state for the next input
            try:
                # Need tau for time t+1 (which is index i in df_pred)
                next_tau_original = df_pred.iloc[i]['tau']
                next_input_features_unscaled = np.zeros(num_input_features)
                # Use the state just calculated/stored
                next_input_features_unscaled[0:num_output_features] = current_state_unscaled
                # Assume tau is the last feature
                if num_input_features > num_output_features: next_input_features_unscaled[-1] = next_tau_original
                # Scale the feature vector
                next_step_features_scaled = input_scaler.transform(next_input_features_unscaled.reshape(1, -1))[0]
            except IndexError: print(f"Warning: Reached end of df_pred for tau input at step {i}. Stopping."); break
            except Exception as e: print(f"Error preparing next input features at step {i}: {e}"); break

            # Update sequence buffer
            current_sequence_np = np.append(current_sequence_np[1:], next_step_features_scaled.reshape(1, -1), axis=0)
            # End of prediction loop

    # --- Process Results ---
    actual_prediction_steps = len(predicted_states_list)
    if actual_prediction_steps == 0: return np.empty((0, num_output_features)), np.empty((0, num_output_features))

    predicted_states_original = np.array(predicted_states_list)
    # Get true states corresponding to the predicted steps
    true_states_original = df_pred.iloc[:actual_prediction_steps][['theta', 'theta_dot']].values

    # Final check for length consistency
    if len(predicted_states_original) != len(true_states_original):
        print("Warning: Mismatch between final predicted and true state lengths. Trimming.")
        min_len = min(len(predicted_states_original), len(true_states_original))
        predicted_states_original = predicted_states_original[:min_len]
        true_states_original = true_states_original[:min_len]

    if len(predicted_states_original) > 0:
        mse = np.mean((predicted_states_original - true_states_original)**2)
        print(f"Multi-step Prediction MSE for {actual_prediction_steps} steps: {mse:.6f}")
    else: print("Could not calculate multi-step MSE.")

    return predicted_states_original, true_states_original


# --- Plotting Multi-Step Prediction ---
def plot_multi_step_prediction(time_vector, true_states, predicted_states,
                               physics_model_predictions=None, model_name="LSTM",
                               save_dir=config.FIGURES_DIR):
    """
    Plots the comparison of multi-step predictions against true values and
    optionally against a pure physics model prediction. Includes robustness checks.
    """
    # --- Input Data Validation ---
    if not all(isinstance(arr, np.ndarray) for arr in [time_vector, true_states, predicted_states]) or \
       len(predicted_states) == 0 or len(true_states) == 0 or len(time_vector) == 0:
        print("Warning: Empty or invalid data provided for multi-step plotting.")
        return

    # Ensure consistent lengths
    min_len = len(predicted_states)
    if len(true_states) != min_len or len(time_vector) != min_len:
         print(f"Warning: Length mismatch in plot data. Using shortest length: {min_len}")
         true_states = true_states[:min_len]; time_vector = time_vector[:min_len]
         if physics_model_predictions is not None: physics_model_predictions = physics_model_predictions[:min_len]
    if physics_model_predictions is not None and len(physics_model_predictions) != min_len:
         print("Warning: Physics prediction length mismatch. Disabling physics plot."); physics_model_predictions = None

    # Ensure state arrays have at least 2 columns (theta, theta_dot)
    if predicted_states.ndim == 1: predicted_states = predicted_states.reshape(-1, 1)
    if true_states.ndim == 1: true_states = true_states.reshape(-1, 1)
    if predicted_states.shape[1] < 2 or true_states.shape[1] < 2:
         print("Error: State arrays have fewer than 2 columns. Cannot plot theta and theta_dot separately.")
         return

    # --- Calculations ---
    # Use try-except for calculations in case of issues after length checks
    try:
        theta_error = np.abs(predicted_states[:, 0] - true_states[:, 0])
        theta_dot_error = np.abs(predicted_states[:, 1] - true_states[:, 1])
        mean_theta_error = np.mean(theta_error); mean_theta_dot_error = np.mean(theta_dot_error)
        mean_physics_theta_error, mean_physics_theta_dot_error = np.nan, np.nan
        physics_theta_error, physics_theta_dot_error = None, None # Initialize

        if physics_model_predictions is not None:
            if physics_model_predictions.ndim == 1: physics_model_predictions = physics_model_predictions.reshape(-1, 1)
            if physics_model_predictions.shape[1] >= 2:
                physics_theta_error = np.abs(physics_model_predictions[:, 0] - true_states[:, 0])
                physics_theta_dot_error = np.abs(physics_model_predictions[:, 1] - true_states[:, 1])
                mean_physics_theta_error = np.mean(physics_theta_error); mean_physics_theta_dot_error = np.mean(physics_theta_dot_error)
            else:
                 print("Warning: Physics predictions have fewer than 2 columns. Disabling physics error calculation.")
                 physics_model_predictions = None # Disable further use if shape is wrong
    except IndexError as ie:
         print(f"Error during error calculation (IndexError): {ie}. Check array shapes.")
         return
    except Exception as calc_e:
         print(f"Error during error calculation: {calc_e}")
         return


    # --- Plotting ---
    try:
        fig = plt.figure(figsize=(15, 12), dpi=300); gs = plt.GridSpec(3, 2, figure=fig)
        fig.suptitle(utils.safe_text(f'多步预测分析: {model_name} 模型', f'Multi-step Prediction: {model_name} Model'), fontsize=16, fontweight='bold')

        # Row 1: State Predictions
        ax1 = fig.add_subplot(gs[0, 0]); ax1.plot(time_vector, true_states[:, 0], 'g-', label=utils.safe_text('真实角度'), linewidth=1.5, alpha=0.8)
        ax1.plot(time_vector, predicted_states[:, 0], 'r--', label=utils.safe_text(f'预测角度 ({model_name})'), linewidth=1.5)
        if physics_model_predictions is not None: ax1.plot(time_vector, physics_model_predictions[:, 0], 'b-.', label=utils.safe_text('纯物理模型'), linewidth=1, alpha=0.6)
        ax1.set_title(utils.safe_text('角度 (θ) 预测'), fontsize=14); ax1.set_ylabel(utils.safe_text('角度 (rad)'), fontsize=12)
        ax1.text(0.02, 0.02, utils.safe_text(f'{model_name} Avg Err: {mean_theta_error:.4f} rad'), transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8)); ax1.legend(fontsize=9); ax1.grid(True)

        ax2 = fig.add_subplot(gs[0, 1]); ax2.plot(time_vector, true_states[:, 1], 'g-', label=utils.safe_text('真实角速度'), linewidth=1.5, alpha=0.8)
        ax2.plot(time_vector, predicted_states[:, 1], 'r--', label=utils.safe_text(f'预测角速度 ({model_name})'), linewidth=1.5)
        if physics_model_predictions is not None: ax2.plot(time_vector, physics_model_predictions[:, 1], 'b-.', label=utils.safe_text('纯物理模型'), linewidth=1, alpha=0.6)
        ax2.set_title(utils.safe_text('角速度 (θ̇) 预测'), fontsize=14); ax2.set_ylabel(utils.safe_text('角速度 (rad/s)'), fontsize=12)
        ax2.text(0.02, 0.02, utils.safe_text(f'{model_name} Avg Err: {mean_theta_dot_error:.4f} rad/s'), transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8)); ax2.legend(fontsize=9); ax2.grid(True)

        # Row 2: Error Analysis
        ax3 = fig.add_subplot(gs[1, 0]); ax3.plot(time_vector, theta_error, 'r-', label=utils.safe_text(f'{model_name} 角度误差'), linewidth=1.5)
        if physics_model_predictions is not None and physics_theta_error is not None: ax3.plot(time_vector, physics_theta_error, 'b-.', label=utils.safe_text('物理模型误差'), linewidth=1, alpha=0.7); ax3.axhline(y=mean_physics_theta_error, color='b', linestyle=':', alpha=0.6, label=utils.safe_text(f'物理AvgErr: {mean_physics_theta_error:.4f}'))
        ax3.set_title(utils.safe_text('角度预测误差'), fontsize=14); ax3.set_ylabel(utils.safe_text('|误差| (rad)'), fontsize=12); ax3.axhline(y=mean_theta_error, color='r', linestyle='--', alpha=0.7, label=utils.safe_text(f'{model_name} AvgErr: {mean_theta_error:.4f}')); ax3.legend(fontsize=9); ax3.grid(True)

        ax4 = fig.add_subplot(gs[1, 1]); ax4.plot(time_vector, theta_dot_error, 'r-', label=utils.safe_text(f'{model_name} 角速度误差'), linewidth=1.5)
        if physics_model_predictions is not None and physics_theta_dot_error is not None: ax4.plot(time_vector, physics_theta_dot_error, 'b-.', label=utils.safe_text('物理模型误差'), linewidth=1, alpha=0.7); ax4.axhline(y=mean_physics_theta_dot_error, color='b', linestyle=':', alpha=0.6, label=utils.safe_text(f'物理AvgErr: {mean_physics_theta_dot_error:.4f}'))
        ax4.set_title(utils.safe_text('角速度预测误差'), fontsize=14); ax4.set_ylabel(utils.safe_text('|误差| (rad/s)'), fontsize=12); ax4.axhline(y=mean_theta_dot_error, color='r', linestyle='--', alpha=0.7, label=utils.safe_text(f'{model_name} AvgErr: {mean_theta_dot_error:.4f}')); ax4.legend(fontsize=9); ax4.grid(True)

        # Row 3: Phase Plot and Cumulative Error
        ax5 = fig.add_subplot(gs[2, 0]); ax5.plot(true_states[:, 0], true_states[:, 1], 'g-', label=utils.safe_text('真实轨迹'), linewidth=1.5, alpha=0.7)
        ax5.plot(predicted_states[:, 0], predicted_states[:, 1], 'r--', label=utils.safe_text(f'预测轨迹 ({model_name})'), linewidth=1.5, alpha=0.9)
        if physics_model_predictions is not None: ax5.plot(physics_model_predictions[:, 0], physics_model_predictions[:, 1], 'b-.', label=utils.safe_text('纯物理轨迹'), linewidth=1, alpha=0.5)
        ax5.set_title(utils.safe_text('相位图: 角度 vs 角速度'), fontsize=14); ax5.set_xlabel(utils.safe_text('角度 (rad)'), fontsize=12); ax5.set_ylabel(utils.safe_text('角速度 (rad/s)'), fontsize=12); ax5.legend(fontsize=9); ax5.grid(True)

        ax6 = fig.add_subplot(gs[2, 1]); steps_axis = np.arange(1, len(theta_error) + 1)
        cum_error_theta = np.cumsum(theta_error) / steps_axis; cum_error_theta_dot = np.cumsum(theta_dot_error) / steps_axis
        ax6.plot(time_vector, cum_error_theta, 'r-', label=utils.safe_text(f'累积角度误差 ({model_name})'), linewidth=1.5)
        ax6.plot(time_vector, cum_error_theta_dot, 'm-', label=utils.safe_text(f'累积角速度误差 ({model_name})'), linewidth=1.5)
        if physics_model_predictions is not None and physics_theta_error is not None and physics_theta_dot_error is not None: cum_error_physics_theta = np.cumsum(physics_theta_error) / steps_axis; cum_error_physics_theta_dot = np.cumsum(physics_theta_dot_error) / steps_axis; ax6.plot(time_vector, cum_error_physics_theta, 'b--', label=utils.safe_text('物理累积角度误差'), linewidth=1, alpha=0.7); ax6.plot(time_vector, cum_error_physics_theta_dot, 'c--', label=utils.safe_text('物理累积角速度误差'), linewidth=1, alpha=0.7)
        ax6.set_title(utils.safe_text('累积平均误差'), fontsize=14); ax6.set_xlabel(utils.safe_text('时间 (s)'), fontsize=12); ax6.set_ylabel(utils.safe_text('累积平均误差'), fontsize=12); ax6.legend(fontsize=9); ax6.grid(True); ax6.set_yscale('log')

        # Add common x-label and footer text
        for ax in [ax1, ax2, ax3, ax4]: ax.set_xlabel(utils.safe_text('时间 (s)'), fontsize=12)
        fig.text(0.5, 0.01, utils.safe_text(f'预测步数: {len(time_vector)}   时间范围: {time_vector[0]:.2f}s - {time_vector[-1]:.2f}s', f'Steps: {len(time_vector)} Range: {time_vector[0]:.2f}s - {time_vector[-1]:.2f}s'), ha='center', fontsize=12)

        # Layout and Save
        plt.tight_layout(rect=[0, 0.02, 1, 0.95]); os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'multi_step_prediction.png'); save_path_hq = os.path.join(save_dir, 'multi_step_prediction_hq.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.savefig(save_path_hq, dpi=600, bbox_inches='tight'); plt.close()
        print(f"Multi-step prediction plot saved to {save_path}")

    except Exception as e: print(f"Error plotting multi-step prediction: {e}"); import traceback; traceback.print_exc(); plt.close()

