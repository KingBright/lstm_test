# evaluation.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict # For collecting results

# Import project modules
import config
import utils
# Import necessary functions from other modules
from data_generation import PendulumSystem, generate_simulation_data, run_simulation
from data_preprocessing import create_sequences


# --- Plotting Training Curves (remains the same) ---
def plot_training_curves(train_losses, val_losses, save_dir=config.FIGURES_DIR):
    # ... (implementation remains the same) ...
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
def evaluate_model(model, data_loader, criterion, device=None):
    # ... (implementation remains the same) ...
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    model.to(device); model.eval(); total_loss = 0.0; batches = 0; start_time = time.time()
    if data_loader is None: print("Warning: data_loader is None in evaluate_model."); return float('inf')
    try: loader_len = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else len(data_loader);
    except (TypeError, AttributeError): loader_len = 0 # Assume non-zero if length unknown
    if loader_len == 0 and hasattr(data_loader, 'dataset'): print("Warning: Evaluation dataset is empty."); return float('inf') # Check dataset length if possible
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            try: outputs = model(inputs); loss = criterion(outputs, targets)
            except Exception as e: print(f"Error during model evaluation forward pass: {e}"); return float('inf')
            if not torch.isfinite(loss): print("Warning: NaN or Inf loss detected."); return float('inf')
            total_loss += loss.item(); batches += 1
    avg_loss = total_loss / batches if batches > 0 else float('inf'); eval_time = time.time() - start_time
    if np.isfinite(avg_loss): print(f'Average Validation/Test Loss (MSE): {avg_loss:.6f}, Evaluation Time: {eval_time:.2f}s')
    else: print(f'Evaluation resulted in invalid average loss. Time: {eval_time:.2f}s')
    return avg_loss


# --- Multi-Step Prediction (Handles Delta Prediction - remains the same) ---
def multi_step_prediction(model, initial_sequence, df_pred, sequence_length, prediction_steps,
                         input_scaler, target_scaler, device=None):
    # ... (implementation remains the same as previous version) ...
    # ... (returns predicted_states_original, true_states_original) ...
    if device is None:
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
    model.to(device); model.eval()
    num_output_features = 2
    if not isinstance(initial_sequence, np.ndarray) or initial_sequence.ndim != 2 or initial_sequence.shape[0] != sequence_length: return np.empty((0, num_output_features)), np.empty((0, num_output_features))
    if df_pred.empty or len(df_pred) < prediction_steps: prediction_steps = len(df_pred)
    if prediction_steps <= 0: return np.empty((0, num_output_features)), np.empty((0, num_output_features))
    if not hasattr(input_scaler, 'scale_') or not hasattr(target_scaler, 'scale_'): return np.empty((0, num_output_features)), np.empty((0, num_output_features))
    is_delta_prediction = isinstance(target_scaler, StandardScaler)
    current_sequence_np = initial_sequence.copy(); predicted_states_list = []
    num_input_features = input_scaler.n_features_in_
    last_unscaled_state = np.zeros(num_output_features); first_step_in_delta = True
    with torch.no_grad():
        for i in range(prediction_steps):
            current_sequence_tensor = torch.tensor(current_sequence_np.reshape(1, sequence_length, num_input_features), dtype=torch.float32).to(device)
            try: model_output_scaled = model(current_sequence_tensor).cpu().numpy()[0];
            except Exception as e: print(f"Error forward pass step {i}: {e}"); break
            if not np.all(np.isfinite(model_output_scaled)): print(f"NaN/Inf detected step {i}."); break
            try:
                model_output_unscaled = target_scaler.inverse_transform(model_output_scaled.reshape(1, -1))[0]
                if is_delta_prediction:
                    if first_step_in_delta: current_state_unscaled = df_pred.iloc[0][['theta', 'theta_dot']].values; first_step_in_delta = False
                    else: current_state_unscaled = last_unscaled_state + model_output_unscaled
                    predicted_states_list.append(current_state_unscaled); last_unscaled_state = current_state_unscaled.copy()
                else: current_state_unscaled = model_output_unscaled; predicted_states_list.append(current_state_unscaled)
            except Exception as e: print(f"Error reconstruction step {i}: {e}"); break
            try:
                next_tau_original = df_pred.iloc[i]['tau']
                next_input_features_unscaled = np.zeros(num_input_features)
                next_input_features_unscaled[0:num_output_features] = current_state_unscaled
                if num_input_features > num_output_features: next_input_features_unscaled[-1] = next_tau_original
                next_step_features_scaled = input_scaler.transform(next_input_features_unscaled.reshape(1, -1))[0]
            except IndexError: break # End of df_pred
            except Exception as e: print(f"Error preparing next input step {i}: {e}"); break
            current_sequence_np = np.append(current_sequence_np[1:], next_step_features_scaled.reshape(1, -1), axis=0)
    actual_prediction_steps = len(predicted_states_list)
    if actual_prediction_steps == 0: return np.empty((0, num_output_features)), np.empty((0, num_output_features))
    predicted_states_original = np.array(predicted_states_list)
    true_states_original = df_pred.iloc[:actual_prediction_steps][['theta', 'theta_dot']].values
    if len(predicted_states_original) != len(true_states_original): min_len = min(len(predicted_states_original), len(true_states_original)); predicted_states_original = predicted_states_original[:min_len]; true_states_original = true_states_original[:min_len]
    return predicted_states_original, true_states_original


# --- Plotting Multi-Step Prediction (remains the same) ---
def plot_multi_step_prediction(time_vector, true_states, predicted_states,
                               physics_model_predictions=None, model_name="LSTM",
                               save_dir=config.FIGURES_DIR, filename_base="multi_step_prediction"):
    # ... (implementation remains the same as previous version) ...
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


# --- Internal Grid Plotting Function (remains the same) ---
def _plot_grid_internal(results_data, scenarios, ics, model_type_name, results_summary, save_dir,
                        variable_index, variable_label, variable_unit, plot_filename):
    # ... (implementation remains the same) ...
    num_rows = len(scenarios); num_cols = len(ics);
    if num_rows == 0 or num_cols == 0: print("Warning: No scenarios or ICs to plot."); return
    print(f"\nGenerating {num_rows}x{num_cols} grid plot for {variable_label}...")
    figsize_width = max(15, num_cols * 3); figsize_height = max(12, num_rows * 2.5)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize_width, figsize_height), sharex=True, sharey=True)
    if num_rows == 1 and num_cols == 1: axes = np.array([[axes]])
    elif num_rows == 1: axes = axes.reshape(1, -1)
    elif num_cols == 1: axes = axes.reshape(-1, 1)
    main_title = utils.safe_text(f'多场景多步预测 ({model_type_name}): {variable_label}', f'Multi-Scenario Multi-Step Prediction ({model_type_name}): {variable_label}')
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    for r, scenario_name in enumerate(scenarios):
        for c in range(num_cols):
            ax = axes[r, c]; run_id = f"{scenario_name}_ic{c+1}"; data = results_data.get(run_id)
            if data and data.get('time') is not None and len(data['time']) > 0:
                time_vector = data['time']; true_val = data['true'][:, variable_index] if data['true'].ndim > 1 and data['true'].shape[1] > variable_index else np.full_like(time_vector, np.nan); pred_val = data['pred'][:, variable_index] if data['pred'].ndim > 1 and data['pred'].shape[1] > variable_index else np.full_like(time_vector, np.nan); physics_val = data.get('physics')
                ax.plot(time_vector, true_val, 'g-', label=utils.safe_text('真实', 'True'), linewidth=1.5, alpha=0.8); ax.plot(time_vector, pred_val, 'r--', label=utils.safe_text('预测', 'Pred'), linewidth=1.5);
                if physics_val is not None and len(physics_val) == len(time_vector) and physics_val.ndim > 1 and physics_val.shape[1] > variable_index : ax.plot(time_vector, physics_val[:, variable_index], 'b-.', label=utils.safe_text('物理', 'Phys'), linewidth=1, alpha=0.6)
                mse_val = results_summary.get(run_id, {}).get('mse', np.nan); mse_str = f"MSE: {mse_val:.3f}" if np.isfinite(mse_val) else "MSE: N/A"; ax.set_title(utils.safe_text(f'{scenario_name} / IC {c+1}\n{mse_str}', f'{scenario_name}/IC{c+1}\n{mse_str}'), fontsize=9)
            else: ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes); ax.set_title(utils.safe_text(f'{scenario_name} / IC {c+1}\n(No Data)', f'{scenario_name}/IC{c+1}\n(No Data)'), fontsize=9)
            ax.grid(True);
            if r == num_rows - 1: ax.set_xlabel(utils.safe_text('时间 (s)', 'Time (s)'), fontsize=9)
            if c == 0: ax.set_ylabel(f"{variable_label} ({variable_unit})", fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc='upper right', fontsize=10)
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93]); os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, plot_filename)
    try: plt.savefig(save_path, dpi=200); plt.close(); print(f"Multi-scenario grid plot saved to: {save_path}")
    except Exception as e: print(f"Error saving grid plot '{plot_filename}': {e}"); plt.close()


# --- Wrapper function for Angle Grid Plot (remains the same) ---
def plot_multi_scenario_grid_angle(results_data, scenarios, ics, model_type_name, results_summary, save_dir=config.FIGURES_DIR):
    grid_filename = f'multistep_grid_angle_{model_type_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    _plot_grid_internal(results_data, scenarios, ics, model_type_name, results_summary, save_dir, variable_index=0, variable_label=utils.safe_text('角度 θ', 'Angle θ'), variable_unit='rad', plot_filename=grid_filename)

# --- Wrapper function for Velocity Grid Plot (remains the same) ---
def plot_multi_scenario_grid_velocity(results_data, scenarios, ics, model_type_name, results_summary, save_dir=config.FIGURES_DIR):
    grid_filename = f'multistep_grid_velocity_{model_type_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    _plot_grid_internal(results_data, scenarios, ics, model_type_name, results_summary, save_dir, variable_index=1, variable_label=utils.safe_text('角速度 θ̇', 'Angular Velocity θ̇'), variable_unit='rad/s', plot_filename=grid_filename)


# --- Consolidated Evaluation Function - MODIFIED ---
# Now calls the grid plotting functions at the end
def run_multi_scenario_evaluation(model, input_scaler, target_scaler, device,
                                  model_type_name, # Pass model type name for plotting labels
                                  scenarios_to_eval=config.SCENARIOS,
                                  ics_to_eval=config.INITIAL_CONDITIONS_SPECIFIC,
                                  limit_prediction_steps=None,
                                  save_dir=config.FIGURES_DIR):
    """
    Runs multi-step prediction evaluation for multiple scenarios and initial conditions.
    Generates data segments on-the-fly, collects results, and plots separate grids for angle and velocity.
    """
    eval_start_time = time.time()
    # Store detailed results for grid plotting
    results_data = defaultdict(dict)
    # Store summary results (MSE, steps)
    results_summary = defaultdict(lambda: {'mse': np.nan, 'steps': 0})

    print("\n--- Performing Multi-Scenario Multi-Step Prediction Evaluation ---")
    try: pendulum = PendulumSystem(m=config.PENDULUM_MASS, L=config.PENDULUM_LENGTH, g=config.GRAVITY, c=config.DAMPING_COEFF)
    except Exception as e: print(f"Error initializing PendulumSystem: {e}"); return results_summary

    model.eval(); model.to(device)

    for scenario_idx, scenario_type in enumerate(scenarios_to_eval):
        for ic_idx, x0 in enumerate(ics_to_eval):
            run_id = f"{scenario_type}_ic{ic_idx+1}"
            print(f"\nEvaluating Scenario: '{scenario_type}', Initial Condition {ic_idx+1}: {x0} ({run_id})")

            # 1. Generate data segment
            df_eval_segment = generate_simulation_data(pendulum, t_span=config.T_SPAN_VAL, dt=config.DT, x0=x0, torque_type=scenario_type)
            if df_eval_segment.empty or len(df_eval_segment) <= config.SEQUENCE_LENGTH: print(f"  Warning: Not enough data generated for {run_id}. Skipping."); continue

            # 2. Prepare sequences
            eval_data_values = df_eval_segment[['theta', 'theta_dot', 'tau']].values
            X_eval_segment, _ = create_sequences(eval_data_values, config.SEQUENCE_LENGTH, predict_delta=False)
            if len(X_eval_segment) == 0: print(f"  Warning: No sequences created for {run_id}. Skipping."); continue

            # 3. Scale sequences
            try: X_eval_scaled = input_scaler.transform(X_eval_segment.reshape(-1, X_eval_segment.shape[2])).reshape(X_eval_segment.shape)
            except Exception as e: print(f"  Error scaling sequences for {run_id}: {e}. Skipping."); continue

            initial_sequence = X_eval_scaled[0]
            df_for_pred_segment = df_eval_segment.iloc[config.SEQUENCE_LENGTH:].reset_index(drop=True)
            available_steps = len(df_for_pred_segment)
            prediction_steps = available_steps
            if limit_prediction_steps is not None and limit_prediction_steps > 0: prediction_steps = min(prediction_steps, limit_prediction_steps)
            if prediction_steps < config.MIN_PREDICTION_STEPS: print(f"  Warning: Insufficient steps ({prediction_steps}) for {run_id}. Skipping."); continue

            # 4. Run Multi-Step Prediction
            predicted_states, true_states = multi_step_prediction(model, initial_sequence, df_for_pred_segment, config.SEQUENCE_LENGTH, prediction_steps, input_scaler, target_scaler, device)

            # 5. Store results and calculate MSE
            if len(predicted_states) > 0 and len(true_states) == len(predicted_states):
                run_mse = np.mean((predicted_states - true_states)**2)
                results_summary[run_id]['mse'] = run_mse
                results_summary[run_id]['steps'] = len(predicted_states)
                print(f"  Multi-step MSE for {run_id} ({results_summary[run_id]['steps']} steps): {run_mse:.6f}")

                # 6. Generate Physics Comparison
                physics_predictions = None
                try:
                    physics_x0 = df_eval_segment.iloc[config.SEQUENCE_LENGTH][['theta', 'theta_dot']].values
                    physics_time_eval = df_for_pred_segment['time'].iloc[:prediction_steps].values
                    if len(physics_time_eval) > 0:
                         physics_t_span = (physics_time_eval[0], physics_time_eval[-1])
                         physics_tau_values = df_for_pred_segment['tau'].iloc[:prediction_steps].values
                         physics_dt = config.DT
                         physics_time, physics_theta, physics_theta_dot = run_simulation(pendulum, physics_t_span, physics_dt, physics_x0, physics_tau_values, t_eval=physics_time_eval)
                         if len(physics_time) == len(physics_time_eval): physics_predictions = np.stack([physics_theta, physics_theta_dot], axis=1)
                         else: print("  Warning: Physics sim length mismatch.")
                    else: print("  Warning: Cannot run physics sim (no time points).")
                except Exception as e: print(f"  Error generating physics comparison for {run_id}: {e}")

                # 7. Store detailed data needed for the grid plot
                results_data[run_id] = {
                    'time': physics_time_eval if 'physics_time_eval' in locals() and len(physics_time_eval)>0 else np.array([]),
                    'true': true_states,
                    'pred': predicted_states,
                    'physics': physics_predictions # Store physics results (can be None)
                }
            else: print(f"  Multi-step prediction failed for {run_id}.")
            # End of IC loop
        # End of Scenario loop

    # --- Plotting Grids (After all loops complete) ---
    if results_data: # Check if any results were collected
        # Prepare IC list (needed by plotting function for column count)
        # ics_to_eval was passed into this function
        # Plot Angle Grid
        plot_multi_scenario_grid_angle(results_data, scenarios_to_eval, ics_to_eval,
                                       model_type_name, results_summary, save_dir)
        # Plot Velocity Grid
        plot_multi_scenario_grid_velocity(results_data, scenarios_to_eval, ics_to_eval,
                                          model_type_name, results_summary, save_dir)
    else:
        print("No valid multi-step prediction results collected to generate grid plots.")


    # --- Summary Print ---
    print("\n--- Multi-Step Prediction MSE Summary ---")
    valid_mses = []
    for run_id, result in results_summary.items(): # Use summary dict
        if np.isfinite(result['mse']):
            print(f"  {run_id}: MSE = {result['mse']:.6f} ({result['steps']} steps)")
            valid_mses.append(result['mse'])
        else:
            print(f"  {run_id}: Failed or not enough steps.")
    if valid_mses: print(f"  Average MSE over successful runs: {np.mean(valid_mses):.6f}")
    else: print("  No successful multi-step prediction runs to average.")

    eval_total_time = time.time() - eval_start_time
    print(f"\nMulti-scenario evaluation finished in {eval_total_time:.2f} seconds.")
    return results_summary # Return the summary dictionary

