# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd # For results table
from tqdm import tqdm
import random # Ensure random is imported

# --- 从训练脚本复制必要的定义 ---

# 1. 模型定义 (PendulumRNN)
class PendulumRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, rnn_type='lstm'):
        """ Initialize the RNN model (LSTM or GRU). """
        super(PendulumRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size # Should be 3
        self.output_size = output_size # Should be 2
        self.rnn_type = rnn_type.lower()
        # print(f"Initializing {self.rnn_type.upper()} h={hidden_size}, l={num_layers}") # Less verbose for eval
        rnn_dropout = 0.0
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=rnn_dropout)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=rnn_dropout)
        else:
            raise ValueError("Unsupported RNN type. Choose 'lstm' or 'gru'.")
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """ Forward pass of the model. """
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 2. 修改后的 RK4 模拟函数 (使用随机外力)
def random_force_pendulum_dynamics(t, state, g, L, b, current_force):
    """Defines the differential equations using a provided force value."""
    theta, omega = state
    # Force is now an input, not calculated based on t
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta) - b * omega + current_force
    return np.array([dtheta_dt, domega_dt])

def simulate_random_force_pendulum_rk4(num_steps, dt=0.02, g=9.81, L=1.0, b=0.1,
                                       initial_theta=np.pi/4, initial_omega=0.0,
                                       force_alpha=0.9, force_beta=0.1): # Random force params
    """
    Simulate pendulum using RK4 method with a stochastic force:
    F[t] = force_alpha * F[t-1] + force_beta * randn()
    Returns states with 3 features: (angle, angular velocity, random_force).
    """
    states_theta_omega = np.zeros((num_steps, 2))
    forces = np.zeros(num_steps) # Store the generated random force
    times = np.arange(num_steps) * dt

    states_theta_omega[0] = [initial_theta, initial_omega]
    forces[0] = force_beta * random.gauss(0, 1) # Initial random force

    for i in range(num_steps - 1):
        t = times[i]
        y = states_theta_omega[i]
        current_force = forces[i] # Use the force calculated for this step

        # RK4 steps using the current force
        k1 = dt * random_force_pendulum_dynamics(t, y, g, L, b, current_force)
        k2 = dt * random_force_pendulum_dynamics(t + 0.5*dt, y + 0.5*k1, g, L, b, current_force) # Force assumed constant during step
        k3 = dt * random_force_pendulum_dynamics(t + 0.5*dt, y + 0.5*k2, g, L, b, current_force)
        k4 = dt * random_force_pendulum_dynamics(t + dt, y + k3, g, L, b, current_force)

        # Update state (theta, omega)
        states_theta_omega[i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        # Generate the force for the *next* time step
        forces[i+1] = force_alpha * forces[i] + force_beta * random.gauss(0, 1)

    # Combine states and the generated random forces
    states_with_force = np.hstack((
        states_theta_omega,
        forces.reshape(-1, 1)
    )) # Shape (steps, 3)
    return times, states_with_force

# --- 评估函数 (修正) ---
def calculate_rmse(predictions, targets):
    """Calculates Root Mean Squared Error, handles 1D or 2D input for masking."""
    if predictions.ndim != targets.ndim:
         print(f"Error: Predictions dim ({predictions.ndim}) != Targets dim ({targets.ndim}).")
         return np.nan
    if predictions.shape[0] != targets.shape[0]: # Check only first dimension (length)
        min_len = min(len(predictions), len(targets))
        print(f"Warning: Length mismatch in RMSE. Pred: {predictions.shape[0]}, Target: {targets.shape[0]}. Trimming to {min_len}.")
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        if predictions.shape[0] != targets.shape[0]:
             print("Error: Length mismatch persists after trimming.")
             return np.nan

    # --- Modification Start: Handle 1D or 2D for NaN check ---
    if predictions.ndim == 1:
        mask = ~np.isnan(predictions) & ~np.isnan(targets)
    elif predictions.ndim == 2:
        # If 2D, assume we check NaNs based on the first column (e.g., theta)
        # Or handle based on specific needs if necessary
        mask = ~np.isnan(predictions[:, 0]) & ~np.isnan(targets[:, 0])
    else:
        print("Error: Unsupported array dimension in calculate_rmse.")
        return np.nan
    # --- Modification End ---

    if np.sum(mask) == 0: return np.nan # Avoid division by zero if all are NaN

    # Calculate RMSE only on non-NaN values
    return np.sqrt(np.mean((predictions[mask] - targets[mask])**2))


# --- 主评估逻辑 ---
if __name__ == "__main__":
    # --- 1. 定义要评估的模型配置 ---
    # (确保与你训练的模型匹配)
    model_configs = [
        {'rnn_type': 'lstm', 'h': 32, 'l': 2, 'seq': 150}, {'rnn_type': 'lstm', 'h': 32, 'l': 2, 'seq': 200},
        {'rnn_type': 'lstm', 'h': 32, 'l': 3, 'seq': 150}, {'rnn_type': 'lstm', 'h': 32, 'l': 3, 'seq': 200},
        {'rnn_type': 'lstm', 'h': 48, 'l': 2, 'seq': 150}, {'rnn_type': 'lstm', 'h': 48, 'l': 2, 'seq': 200},
        {'rnn_type': 'lstm', 'h': 48, 'l': 3, 'seq': 200},
        {'rnn_type': 'lstm', 'h': 64, 'l': 2, 'seq': 150}, {'rnn_type': 'lstm', 'h': 64, 'l': 2, 'seq': 200},
        {'rnn_type': 'lstm', 'h': 64, 'l': 3, 'seq': 150}, {'rnn_type': 'lstm', 'h': 64, 'l': 3, 'seq': 200},
        # 可以添加更多配置
    ]

    # --- 2. 定义测试场景 (现在只需要初始条件) ---
    test_scenarios_initial = [
        {'name': 'InitCond_1', 'theta0': np.pi / 6, 'omega0': 0.5},
        {'name': 'InitCond_2', 'theta0': 0.1,       'omega0': 0.0},
        {'name': 'InitCond_3', 'theta0': -np.pi / 4, 'omega0': -0.5},
        {'name': 'InitCond_4', 'theta0': np.pi / 3, 'omega0': 1.0},
        {'name': 'InitCond_5', 'theta0': -0.2,      'omega0': 1.2},
    ]

    # --- 3. 设置评估参数 ---
    DT = 0.02
    EVAL_STEPS = 1000 # 在多少个单步上进行评估
    NUM_STEPS_FOR_GT = 3000 # 确保模拟足够长
    L = 1.0 # Pendulum length
    # --- Random Force Parameters ---
    FORCE_ALPHA = 0.95 # Autocorrelation factor (0 to 1)
    FORCE_BETA = 0.1   # Noise magnitude (std dev)


    # --- 4. 循环评估 ---
    results = []
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Evaluating with Random Force: alpha={FORCE_ALPHA}, beta={FORCE_BETA}")

    for config in tqdm(model_configs, desc="Evaluating Models"):
        rnn_type = config['rnn_type']
        h_size = config['h']
        n_layers = config['l']
        seq_len = config['seq'] # 使用配置中的 seq_len

        print(f"\n--- Evaluating Model: {rnn_type.upper()} H={h_size} L={n_layers} Seq={seq_len} (Random Force) ---")

        # 构建路径 (假设模型是基于正弦力训练的)
        model_dir = f"{rnn_type}_h{h_size}_l{n_layers}_seq{seq_len}_rk4_es" # Directory where the model was saved
        model_path = os.path.join(model_dir, f"best_model_{rnn_type}_h{h_size}_l{n_layers}_seq{seq_len}.pth")
        scaler_path = os.path.join(model_dir, f"scaler_{rnn_type}_h{h_size}_l{n_layers}_seq{seq_len}.joblib")

        # 加载 Scaler
        if not os.path.exists(scaler_path): print(f"Scaler not found: {scaler_path}. Skipping."); continue
        try:
            scaler: StandardScaler = joblib.load(scaler_path)
            if len(scaler.mean_) != 3: raise ValueError("Scaler feature mismatch (expected 3)")
        except Exception as e: print(f"Error loading scaler {scaler_path}: {e}. Skipping."); continue

        # 加载模型
        if not os.path.exists(model_path): print(f"Model not found: {model_path}. Skipping."); continue
        try:
            model = PendulumRNN(input_size=3, output_size=2, hidden_size=h_size, num_layers=n_layers, rnn_type=rnn_type)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e: print(f"Error loading model {model_path}: {e}. Skipping."); continue

        # 在所有测试初始条件下评估当前模型
        scenario_angle_rmses = []
        scenario_omega_rmses = []
        for scenario_init in tqdm(test_scenarios_initial, desc=f"Testing Init Conds for H={h_size} L={n_layers}", leave=False):
            # 模拟带随机力的基准轨迹
            sim_steps_needed = seq_len + EVAL_STEPS
            if sim_steps_needed > NUM_STEPS_FOR_GT: NUM_STEPS_FOR_GT = sim_steps_needed + 100

            test_times, test_states_orig = simulate_random_force_pendulum_rk4(
                NUM_STEPS_FOR_GT, dt=DT, initial_theta=scenario_init['theta0'], initial_omega=scenario_init['omega0'],
                force_alpha=FORCE_ALPHA, force_beta=FORCE_BETA, L=L
            )
            # --- 重要: 使用加载的 scaler (基于正弦力数据训练) ---
            test_states_scaled = scaler.transform(test_states_orig)

            # --- 单步预测评估 ---
            all_preds_scaled = []
            all_targets_scaled = []
            eval_start_idx = seq_len
            eval_end_idx = eval_start_idx + EVAL_STEPS
            if eval_end_idx > len(test_states_scaled):
                print(f"Warning: Sim too short for {scenario_init['name']}. Reducing EVAL_STEPS.")
                eval_end_idx = len(test_states_scaled)
                if eval_start_idx >= eval_end_idx: continue

            with torch.no_grad():
                for t in range(eval_start_idx, eval_end_idx):
                    input_sequence_scaled = test_states_scaled[t-seq_len : t]
                    input_tensor = torch.FloatTensor(input_sequence_scaled).unsqueeze(0).to(device) # (1, seq_len, 3)
                    pred_scaled = model(input_tensor) # Shape: (1, 2)
                    target_scaled = test_states_scaled[t, :2] # Shape: (2,)

                    if torch.isnan(pred_scaled).any() or torch.isinf(pred_scaled).any():
                         print(f"Warning: NaN/Inf predicted at step {t}. Appending NaN.")
                         all_preds_scaled.append(np.array([np.nan, np.nan]))
                    else:
                         all_preds_scaled.append(pred_scaled.cpu().numpy().flatten())
                    all_targets_scaled.append(target_scaled)

            if not all_preds_scaled: continue

            all_preds_scaled = np.array(all_preds_scaled)
            all_targets_scaled = np.array(all_targets_scaled)

            # 反向转换为原始尺度
            mean_output = scaler.mean_[:2]; scale_output = scaler.scale_[:2]
            if np.any(scale_output == 0): scale_output[scale_output == 0] = 1.0
            all_preds_orig = np.full_like(all_preds_scaled, np.nan)
            valid_pred_mask = ~np.isnan(all_preds_scaled[:, 0])
            all_preds_orig[valid_pred_mask] = all_preds_scaled[valid_pred_mask] * scale_output + mean_output
            all_targets_orig = all_targets_scaled * scale_output + mean_output

            # 计算这个场景下的 RMSE (ignoring NaNs)
            # --- 调用修正后的 calculate_rmse ---
            angle_rmse = calculate_rmse(all_preds_orig[:, 0], all_targets_orig[:, 0]) # Pass 1D array
            omega_rmse = calculate_rmse(all_preds_orig[:, 1], all_targets_orig[:, 1]) # Pass 1D array

            if not np.isnan(angle_rmse): scenario_angle_rmses.append(angle_rmse)
            if not np.isnan(omega_rmse): scenario_omega_rmses.append(omega_rmse)
            # --- 结束单步预测评估 ---

        # 计算该模型配置在所有场景下的平均 RMSE
        avg_angle_rmse = np.mean(scenario_angle_rmses) if scenario_angle_rmses else np.nan
        avg_omega_rmse = np.mean(scenario_omega_rmses) if scenario_omega_rmses else np.nan

        results.append({
            'RNN': rnn_type.upper(),
            'H': h_size,
            'L': n_layers,
            'Seq': seq_len,
            'AvgAngleRMSE_RandForce': avg_angle_rmse, # Renamed metric
            'AvgOmegaRMSE_RandForce': avg_omega_rmse  # Renamed metric
        })
        print(f"Finished. Avg Angle RMSE (Rand Force): {avg_angle_rmse:.6f}, Avg Omega RMSE (Rand Force): {avg_omega_rmse:.6f}")

    # --- 5. 结果汇总与展示 ---
    print("\n--- Random Force Evaluation Summary (Single-Step) ---")
    if not results:
        print("No models were successfully evaluated.")
    else:
        results_df = pd.DataFrame(results)
        results_df['CombinedRMSE_RandForce'] = np.sqrt(results_df['AvgAngleRMSE_RandForce']**2 + results_df['AvgOmegaRMSE_RandForce']**2)
        results_df = results_df.sort_values(by='CombinedRMSE_RandForce')
        print(results_df.to_string(index=False, float_format="%.6f"))

        best_model_config = results_df.iloc[0]
        print("\n--- Best Model Configuration (based on Random Force Single-Step RMSE) ---")
        print(f"RNN Type: {best_model_config['RNN']}")
        print(f"Hidden Size (H): {best_model_config['H']}")
        print(f"Num Layers (L): {best_model_config['L']}")
        print(f"Sequence Length (Seq): {best_model_config['Seq']}")
        print(f"Avg Angle RMSE (Rand Force): {best_model_config['AvgAngleRMSE_RandForce']:.6f}")
        print(f"Avg Omega RMSE (Rand Force): {best_model_config['AvgOmegaRMSE_RandForce']:.6f}")
        print(f"Combined RMSE (Rand Force): {best_model_config['CombinedRMSE_RandForce']:.6f}")

