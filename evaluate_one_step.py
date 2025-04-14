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

# 2. RK4 模拟函数 (simulate_forced_pendulum_rk4)
def pendulum_dynamics(t, state, g, L, b, force_amplitude, force_frequency):
    """Defines the differential equations for the forced pendulum."""
    theta, omega = state
    force = force_amplitude * np.sin(force_frequency * t)
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta) - b * omega + force
    return np.array([dtheta_dt, domega_dt]), force

def simulate_forced_pendulum_rk4(num_steps, dt=0.02, g=9.81, L=1.0, b=0.1,
                                 initial_theta=np.pi/4, initial_omega=0.0,
                                 force_amplitude=0.5, force_frequency=1.0):
    """ Simulate forced pendulum using RK4 method. """
    states_theta_omega = np.zeros((num_steps, 2))
    forces = np.zeros(num_steps)
    times = np.arange(num_steps) * dt
    states_theta_omega[0] = [initial_theta, initial_omega]
    _, forces[0] = pendulum_dynamics(times[0], states_theta_omega[0], g, L, b, force_amplitude, force_frequency)
    for i in range(num_steps - 1):
        t = times[i]; y = states_theta_omega[i]
        k1_deriv, _ = pendulum_dynamics(t, y, g, L, b, force_amplitude, force_frequency); k1 = dt * k1_deriv
        k2_deriv, _ = pendulum_dynamics(t + 0.5*dt, y + 0.5*k1, g, L, b, force_amplitude, force_frequency); k2 = dt * k2_deriv
        k3_deriv, _ = pendulum_dynamics(t + 0.5*dt, y + 0.5*k2, g, L, b, force_amplitude, force_frequency); k3 = dt * k3_deriv
        k4_deriv, _ = pendulum_dynamics(t + dt, y + k3, g, L, b, force_amplitude, force_frequency); k4 = dt * k4_deriv
        states_theta_omega[i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        _, forces[i+1] = pendulum_dynamics(times[i+1], states_theta_omega[i+1], g, L, b, force_amplitude, force_frequency)
    states_with_force = np.hstack((states_theta_omega, forces.reshape(-1, 1)))
    return times, states_with_force

# --- 评估函数 ---
def calculate_rmse(predictions, targets):
    """Calculates Root Mean Squared Error."""
    if predictions.shape != targets.shape:
        min_len = min(len(predictions), len(targets))
        print(f"Warning: Shape mismatch in RMSE. Pred: {predictions.shape}, Target: {targets.shape}. Trimming to {min_len}.")
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        if predictions.shape != targets.shape:
             print("Error: Shape mismatch persists after trimming.")
             return np.nan
    return np.sqrt(np.mean((predictions - targets)**2))

# --- 主评估逻辑 ---
if __name__ == "__main__":
    # --- 1. 定义要评估的模型配置 ---
    # (与上个脚本相同，确保与你训练的模型匹配)
    model_configs = [
        {'rnn_type': 'lstm', 'h': 32, 'l': 2, 'seq': 150}, {'rnn_type': 'lstm', 'h': 32, 'l': 2, 'seq': 200},
        {'rnn_type': 'lstm', 'h': 32, 'l': 3, 'seq': 150}, {'rnn_type': 'lstm', 'h': 32, 'l': 3, 'seq': 200},
        {'rnn_type': 'lstm', 'h': 48, 'l': 2, 'seq': 150}, {'rnn_type': 'lstm', 'h': 48, 'l': 2, 'seq': 200},
        {'rnn_type': 'lstm', 'h': 48, 'l': 3, 'seq': 200},
        {'rnn_type': 'lstm', 'h': 64, 'l': 2, 'seq': 150}, {'rnn_type': 'lstm', 'h': 64, 'l': 2, 'seq': 200},
        {'rnn_type': 'lstm', 'h': 64, 'l': 3, 'seq': 150}, {'rnn_type': 'lstm', 'h': 64, 'l': 3, 'seq': 200},
        # 可以添加更多配置
    ]

    # --- 2. 定义测试场景 ---
    # (与上个脚本相同)
    test_scenarios = [
        {'name': 'Scenario_1', 'theta0': np.pi / 6, 'omega0': 0.5, 'amp': 0.6, 'freq': 1.8},
        {'name': 'Scenario_2', 'theta0': 0.1,       'omega0': 0.0, 'amp': 0.2, 'freq': 0.7},
        {'name': 'Scenario_3', 'theta0': -np.pi / 4, 'omega0': -0.5,'amp': 0.9, 'freq': 1.5},
        {'name': 'Scenario_4', 'theta0': np.pi / 3, 'omega0': 1.0, 'amp': 0.5, 'freq': 2.3},
        {'name': 'Scenario_5', 'theta0': -0.2,      'omega0': 1.2, 'amp': 0.7, 'freq': 1.2},
    ]

    # --- 3. 设置评估参数 ---
    DT = 0.02
    EVAL_STEPS = 1000 # 在多少个单步上进行评估
    NUM_STEPS_FOR_GT = 2000 # 确保模拟足够长以覆盖 SEQ_LENGTH + EVAL_STEPS
    # 注意: START_INDEX_OFFSET 现在不需要了，因为我们直接用模拟数据构建序列

    L = 1.0 # Pendulum length

    # --- 4. 循环评估 ---
    results = []
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    for config in tqdm(model_configs, desc="Evaluating Models"):
        rnn_type = config['rnn_type']
        h_size = config['h']
        n_layers = config['l']
        seq_len = config['seq'] # 使用配置中的 seq_len

        print(f"\n--- Evaluating Model: {rnn_type.upper()} H={h_size} L={n_layers} Seq={seq_len} (Single-Step) ---")

        # 构建路径
        model_dir = f"{rnn_type}_h{h_size}_l{n_layers}_seq{seq_len}_rk4_es" # 假设目录名如此
        model_path = os.path.join(model_dir, f"best_model_{rnn_type}_h{h_size}_l{n_layers}_seq{seq_len}.pth")
        scaler_path = os.path.join(model_dir, f"scaler_{rnn_type}_h{h_size}_l{n_layers}_seq{seq_len}.joblib")

        # 加载 Scaler
        if not os.path.exists(scaler_path): print(f"Scaler not found: {scaler_path}. Skipping."); continue
        try:
            scaler: StandardScaler = joblib.load(scaler_path)
            if len(scaler.mean_) != 3: raise ValueError("Scaler feature mismatch")
        except Exception as e: print(f"Error loading scaler {scaler_path}: {e}. Skipping."); continue

        # 加载模型
        if not os.path.exists(model_path): print(f"Model not found: {model_path}. Skipping."); continue
        try:
            model = PendulumRNN(input_size=3, output_size=2, hidden_size=h_size, num_layers=n_layers, rnn_type=rnn_type)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e: print(f"Error loading model {model_path}: {e}. Skipping."); continue

        # 在所有测试场景上评估当前模型
        scenario_angle_rmses = []
        scenario_omega_rmses = []
        for scenario in tqdm(test_scenarios, desc=f"Testing Scenarios for H={h_size} L={n_layers}", leave=False):
            # 模拟基准轨迹
            sim_steps_needed = seq_len + EVAL_STEPS # 需要的总步数
            if sim_steps_needed > NUM_STEPS_FOR_GT: NUM_STEPS_FOR_GT = sim_steps_needed + 100 # 确保足够长

            test_times, test_states_orig = simulate_forced_pendulum_rk4(
                NUM_STEPS_FOR_GT, dt=DT, initial_theta=scenario['theta0'], initial_omega=scenario['omega0'],
                force_amplitude=scenario['amp'], force_frequency=scenario['freq'], L=L
            )
            test_states_scaled = scaler.transform(test_states_orig)

            # --- 单步预测评估 ---
            all_preds_scaled = []
            all_targets_scaled = []
            eval_start_idx = seq_len
            eval_end_idx = eval_start_idx + EVAL_STEPS
            if eval_end_idx > len(test_states_scaled):
                print(f"Warning: Simulation too short for scenario {scenario['name']}. Reducing EVAL_STEPS.")
                eval_end_idx = len(test_states_scaled)
                if eval_start_idx >= eval_end_idx: continue # Skip if no steps to evaluate

            with torch.no_grad():
                for t in range(eval_start_idx, eval_end_idx):
                    # 提取真实的输入序列 (t-seq_len 到 t-1)
                    input_sequence_scaled = test_states_scaled[t-seq_len : t]
                    input_tensor = torch.FloatTensor(input_sequence_scaled).unsqueeze(0).to(device) # (1, seq_len, 3)

                    # 模型进行单步预测 (预测 t 时刻的状态)
                    pred_scaled = model(input_tensor) # Shape: (1, 2)

                    # 获取真实的 t 时刻状态 (目标)
                    target_scaled = test_states_scaled[t, :2] # Shape: (2,)

                    all_preds_scaled.append(pred_scaled.cpu().numpy().flatten())
                    all_targets_scaled.append(target_scaled)

            if not all_preds_scaled: continue # Skip if no predictions were made

            # 转换为 Numpy 数组
            all_preds_scaled = np.array(all_preds_scaled)
            all_targets_scaled = np.array(all_targets_scaled)

            # 反向转换为原始尺度
            mean_output = scaler.mean_[:2]; scale_output = scaler.scale_[:2]
            if np.any(scale_output == 0): scale_output[scale_output == 0] = 1.0
            all_preds_orig = all_preds_scaled * scale_output + mean_output
            all_targets_orig = all_targets_scaled * scale_output + mean_output

            # 计算这个场景下的 RMSE
            angle_rmse = calculate_rmse(all_preds_orig[:, 0], all_targets_orig[:, 0])
            omega_rmse = calculate_rmse(all_preds_orig[:, 1], all_targets_orig[:, 1])

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
            'AvgAngleRMSE_1Step': avg_angle_rmse, # Renamed metric
            'AvgOmegaRMSE_1Step': avg_omega_rmse  # Renamed metric
        })
        print(f"Finished. Avg 1-Step Angle RMSE: {avg_angle_rmse:.6f}, Avg 1-Step Omega RMSE: {avg_omega_rmse:.6f}")

    # --- 5. 结果汇总与展示 ---
    print("\n--- Single-Step Prediction Evaluation Summary ---")
    if not results:
        print("No models were successfully evaluated.")
    else:
        results_df = pd.DataFrame(results)
        # 添加一个综合评分
        results_df['CombinedRMSE_1Step'] = np.sqrt(results_df['AvgAngleRMSE_1Step']**2 + results_df['AvgOmegaRMSE_1Step']**2)

        # 按综合评分排序
        results_df = results_df.sort_values(by='CombinedRMSE_1Step')

        print(results_df.to_string(index=False, float_format="%.6f"))

        # 找出最佳模型
        best_model_config = results_df.iloc[0]
        print("\n--- Best Model Configuration (based on Single-Step RMSE) ---")
        print(f"RNN Type: {best_model_config['RNN']}")
        print(f"Hidden Size (H): {best_model_config['H']}")
        print(f"Num Layers (L): {best_model_config['L']}")
        print(f"Sequence Length (Seq): {best_model_config['Seq']}")
        print(f"Avg Angle RMSE (1-Step): {best_model_config['AvgAngleRMSE_1Step']:.6f}")
        print(f"Avg Omega RMSE (1-Step): {best_model_config['AvgOmegaRMSE_1Step']:.6f}")
        print(f"Combined RMSE (1-Step): {best_model_config['CombinedRMSE_1Step']:.6f}")

