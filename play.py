# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import joblib # To load scaler
from sklearn.preprocessing import StandardScaler # To define scaler type hint
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# --- 必要时: 从训练脚本复制模型定义和RK4模拟函数 ---
# (确保这里的定义与训练时使用的完全一致)

# --- 3. 定义 RNN 模型 (输入3维, 输出2维) ---
class PendulumRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, rnn_type='lstm'):
        """ Initialize the RNN model (LSTM or GRU). """
        super(PendulumRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size # Should be 3
        self.output_size = output_size # Should be 2
        self.rnn_type = rnn_type.lower()
        print(f"Initializing {self.rnn_type.upper()} with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}")
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

# --- 1. 使用 RK4 模拟带外力的单摆数据 (返回 3 特征) ---
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

# --- 5. 评估和预测 (函数不变) ---
def predict_with_known_future_force(model, start_sequence_scaled, seq_length, future_steps, dt,
                                    predict_force_amp, predict_force_freq, # Force params for prediction
                                    scaler: StandardScaler, # Type hint for clarity
                                    start_time, device='cpu'):
    """ Perform multi-step prediction where future force is known/calculated. """
    model.eval()
    model.to(device)
    predictions_scaled_theta_omega = [] # Stores scaled [theta, omega]
    expected_input_size = 3
    if start_sequence_scaled.shape != (seq_length, expected_input_size):
         print(f"Error: Input data shape mismatch. Expected ({seq_length}, {expected_input_size}), got {start_sequence_scaled.shape}")
         return np.array([])
    current_sequence_scaled_np = start_sequence_scaled.copy()
    force_mean = scaler.mean_[2]
    force_scale = scaler.scale_[2] if scaler.scale_[2] != 0 else 1.0
    with torch.no_grad():
        for k in range(future_steps):
            current_sequence_tensor = torch.FloatTensor(current_sequence_scaled_np).unsqueeze(0).to(device)
            next_pred_state_scaled = model(current_sequence_tensor)
            if torch.isnan(next_pred_state_scaled).any() or torch.isinf(next_pred_state_scaled).any():
                print(f"Warning: NaN/Inf detected at step {k+1}. Stopping."); break
            predicted_theta_omega_scaled = next_pred_state_scaled.cpu().numpy().flatten()
            predictions_scaled_theta_omega.append(predicted_theta_omega_scaled)
            next_time = start_time + (k + 1) * dt
            next_force_original = predict_force_amp * np.sin(predict_force_freq * next_time)
            next_force_scaled = (next_force_original - force_mean) / force_scale
            next_full_state_scaled = np.array([predicted_theta_omega_scaled[0], predicted_theta_omega_scaled[1], next_force_scaled])
            current_sequence_scaled_np = np.vstack((current_sequence_scaled_np[1:], next_full_state_scaled))
    predictions_scaled_theta_omega = np.array(predictions_scaled_theta_omega)
    if predictions_scaled_theta_omega.size > 0 and predictions_scaled_theta_omega.shape[1] == 2:
        mean_output = scaler.mean_[:2]; scale_output = scaler.scale_[:2]
        if np.any(scale_output == 0): scale_output[scale_output == 0] = 1.0
        predictions_original = predictions_scaled_theta_omega * scale_output + mean_output
        return predictions_original
    else: return np.array([])

# --- 主程序 ---
if __name__ == "__main__":
    # --- Argument Parser for Simulator ---
    parser = argparse.ArgumentParser(description='Simulate Pendulum with Trained RNN Model')
    # Model parameters
    parser.add_argument('--hidden_size', type=int, required=True, help='Number of hidden units (must match trained model)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers (must match trained model)')
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], help='RNN type (must match trained model)')
    parser.add_argument('--seq_length', type=int, default=150, help='Sequence length used during training') # Match training
    # Initial conditions and force parameters for simulation
    parser.add_argument('--theta0', type=float, default=1.5, help='Initial angle (rad)')
    parser.add_argument('--omega0', type=float, default=0.2, help='Initial angular velocity (rad/s)')
    parser.add_argument('--amp', type=float, default=0.5, help='Force amplitude')
    parser.add_argument('--freq', type=float, default=1.0, help='Force frequency')
    # Other parameters
    parser.add_argument('--sim_steps', type=int, default=2000, help='Number of steps to simulate/predict for animation')
    parser.add_argument('--dt', type=float, default=0.02, help='Time step')
    parser.add_argument('--L', type=float, default=1.0, help='Pendulum length')

    args = parser.parse_args()
    # --- End Argument Parser ---

    # --- Construct paths based on args ---
    # Assuming the training script saved model/scaler in a dir named like this:
    MODEL_DIR = f"{args.rnn_type}_h{args.hidden_size}_l{args.num_layers}_seq{args.seq_length}_rk4_es" # Adjusted name based on last training script
    MODEL_PATH = os.path.join(MODEL_DIR, f"best_model_{args.rnn_type}_h{args.hidden_size}_l{args.num_layers}_seq{args.seq_length}.pth")
    SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_{args.rnn_type}_h{args.hidden_size}_l{args.num_layers}_seq{args.seq_length}.joblib")

    print(f"Attempting to load model from: {MODEL_PATH}")
    print(f"Attempting to load scaler from: {SCALER_PATH}")

    # --- Load Scaler ---
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler file not found at {SCALER_PATH}")
        exit()
    try:
        scaler: StandardScaler = joblib.load(SCALER_PATH)
        print("Scaler loaded successfully.")
        if len(scaler.mean_) != 3:
             print(f"Error: Loaded scaler has unexpected number of features ({len(scaler.mean_)}), expected 3.")
             exit()
    except Exception as e:
        print(f"Error loading scaler: {e}")
        exit()

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()
    input_size = 3; output_size = 2
    model = PendulumRNN(input_size, args.hidden_size, output_size, args.num_layers, rnn_type=args.rnn_type)
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    model.to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH} and set to eval mode.")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        exit()

    # --- Prepare Data for Prediction ---
    print("Preparing initial sequence...")
    init_steps = args.seq_length
    total_sim_steps = init_steps + args.sim_steps
    print(f"Simulating {total_sim_steps} steps for ground truth...")
    all_times, all_states_orig = simulate_forced_pendulum_rk4(
        total_sim_steps, dt=args.dt, initial_theta=args.theta0, initial_omega=args.omega0,
        force_amplitude=args.amp, force_frequency=args.freq, L=args.L
    )
    all_states_scaled = scaler.transform(all_states_orig)
    start_sequence_scaled = all_states_scaled[:init_steps]
    time_at_sequence_end = all_times[init_steps - 1]

    # --- Run Prediction ---
    print(f"Running prediction for {args.sim_steps} steps...")
    predictions_original = predict_with_known_future_force(
        model, start_sequence_scaled, args.seq_length, args.sim_steps, args.dt,
        predict_force_amp=args.amp, predict_force_freq=args.freq,
        scaler=scaler, start_time=time_at_sequence_end, device=device
    )
    if predictions_original.size == 0: print("Prediction failed."); exit()
    print("Prediction complete.")

    # --- Get Ground Truth for the Prediction Period ---
    actual_start_index = init_steps
    actual_end_index = actual_start_index + len(predictions_original)
    actual_future_states = all_states_orig[actual_start_index:actual_end_index, :2] # theta, omega
    actual_future_times = all_times[actual_start_index:actual_end_index]
    predicted_times = actual_future_times # Use same time axis

    # Ensure lengths match
    if len(predictions_original) != len(actual_future_states):
        print("Warning: Length mismatch. Trimming.")
        min_len = min(len(predictions_original), len(actual_future_states))
        predictions_original = predictions_original[:min_len]
        actual_future_states = actual_future_states[:min_len]
        actual_future_times = actual_future_times[:min_len]
        predicted_times = predicted_times[:min_len]

    # Calculate errors
    angle_error = predictions_original[:, 0] - actual_future_states[:, 0]
    velocity_error = predictions_original[:, 1] - actual_future_states[:, 1]

    # --- Create Animation with 4 subplots ---
    print("Creating animation...")
    # Adjust figsize and height_ratios for 4 plots
    fig, (ax1, ax4, ax2, ax3) = plt.subplots(4, 1, figsize=(8, 12),
                                             gridspec_kw={'height_ratios': [4, 2, 1, 1]}) # Renamed ax5->ax4, ax4->ax2, ax2->ax3 for consistency
    fig.suptitle(f'Pendulum Simulation\nAmp={args.amp:.2f}, Freq={args.freq:.2f} ({args.rnn_type.upper()} H={args.hidden_size} L={args.num_layers} Seq={args.seq_length})')
    pendulum_length_L = args.L

    # --- Plot 1: Overlayed Pendulum Animation ---
    ax1.set_title('Pendulum Motion')
    ax1.set_xlim(-pendulum_length_L*1.2, pendulum_length_L*1.2)
    ax1.set_ylim(-pendulum_length_L*1.2, pendulum_length_L*1.2)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True)
    line_actual, = ax1.plot([], [], 'o-', lw=3, color='blue', alpha=0.7, label='Actual (RK4)', markersize=10)
    line_pred, = ax1.plot([], [], 'o-', lw=2, color='red', alpha=0.7, label='Predicted', markersize=7)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    ax1.legend(loc='upper right')

    # --- Plot 4 (was ax4): Velocity Comparison Curve ---
    ax4.set_title('Velocity Comparison')
    ax4.set_ylabel('Velocity (rad/s)')
    ax4.grid(True)
    line_vel_actual, = ax4.plot([], [], lw=2, color='blue', alpha=0.7, label='Actual')
    line_vel_pred, = ax4.plot([], [], lw=1.5, color='red', linestyle='-.', label='Predicted')
    ax4.legend(loc='upper right')
    ax4.set_xlim(predicted_times[0], predicted_times[-1])

    # --- Plot 2 (was ax2): Angle Error Curve ---
    ax2.set_title('Angle Error (Pred - Actual)')
    ax2.set_ylabel('Angle Error (rad)')
    ax2.grid(True)
    line_angle_error, = ax2.plot([], [], lw=1, color='green')
    ax2.set_xlim(predicted_times[0], predicted_times[-1])
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1) # Zero error line

    # --- Plot 3 (was ax3): Velocity Error Curve ---
    ax3.set_title('Velocity Error (Pred - Actual)')
    ax3.set_xlabel('Time (s)') # Keep x-label only on bottom plot
    ax3.set_ylabel('Velocity Error (rad/s)')
    ax3.grid(True)
    line_velocity_error, = ax3.plot([], [], lw=1, color='purple')
    ax3.set_xlim(predicted_times[0], predicted_times[-1])
    ax3.axhline(0, color='gray', linestyle=':', linewidth=1) # Zero error line


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # Data for animation
    actual_theta_anim = actual_future_states[:, 0]
    actual_omega_anim = actual_future_states[:, 1]
    predicted_theta_anim = predictions_original[:, 0]
    predicted_omega_anim = predictions_original[:, 1]
    times_anim = actual_future_times

    # Initialization function
    def init():
        line_actual.set_data([], [])
        line_pred.set_data([], [])
        time_text.set_text('')
        line_angle_error.set_data([], [])
        line_velocity_error.set_data([], [])
        line_vel_actual.set_data([], [])
        line_vel_pred.set_data([], [])
        # Removed angle comparison lines from init
        return (line_actual, line_pred, time_text, line_angle_error,
                line_velocity_error, line_vel_actual, line_vel_pred) # Return 7 items

    # Update function
    def update(i):
        # Pendulum plot
        x_actual = pendulum_length_L * np.sin(actual_theta_anim[i])
        y_actual = -pendulum_length_L * np.cos(actual_theta_anim[i])
        x_pred = pendulum_length_L * np.sin(predicted_theta_anim[i])
        y_pred = -pendulum_length_L * np.cos(predicted_theta_anim[i])
        line_actual.set_data([0, x_actual], [0, y_actual])
        line_pred.set_data([0, x_pred], [0, y_pred])
        time_str = f'Time = {times_anim[i]:.2f}s'
        time_text.set_text(time_str)

        # Data up to current frame
        current_times = times_anim[:i+1]
        current_angle_error = angle_error[:i+1]
        current_velocity_error = velocity_error[:i+1]
        current_actual_omega = actual_omega_anim[:i+1]
        current_pred_omega = predicted_omega_anim[:i+1]
        # Removed current angle data fetch (not needed for plotting lines)

        # Update error plots
        line_angle_error.set_data(current_times, current_angle_error)
        line_velocity_error.set_data(current_times, current_velocity_error)

        # Update velocity comparison plot
        line_vel_actual.set_data(current_times, current_actual_omega)
        line_vel_pred.set_data(current_times, current_pred_omega)

        # Removed angle comparison plot update

        # Adjust y-limits dynamically
        if i > 0:
            # Angle Error Limits (ax2)
            min_ae = np.min(current_angle_error); max_ae = np.max(current_angle_error)
            margin_ae = max(0.05, (max_ae - min_ae) * 0.1); ax2.set_ylim(min_ae - margin_ae, max_ae + margin_ae)

            # Velocity Error Limits (ax3)
            min_ve = np.min(current_velocity_error); max_ve = np.max(current_velocity_error)
            margin_ve = max(0.1, (max_ve - min_ve) * 0.1); ax3.set_ylim(min_ve - margin_ve, max_ve + margin_ve)

            # Velocity Comparison Limits (ax4)
            min_vo = min(np.min(current_actual_omega), np.min(current_pred_omega))
            max_vo = max(np.max(current_actual_omega), np.max(current_pred_omega))
            margin_vo = max(0.1, (max_vo - min_vo) * 0.1); ax4.set_ylim(min_vo - margin_vo, max_vo + margin_vo)

            # Removed angle comparison limit adjustment

        # Return only the lines being updated
        return (line_actual, line_pred, time_text, line_angle_error,
                line_velocity_error, line_vel_actual, line_vel_pred) # Return 7 items

    # Create and display animation
    num_frames = len(times_anim)
    frame_interval = int(args.dt * 1000 * 1) # Adjust speed
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=frame_interval, blit=True, init_func=init)

    print("Displaying animation... Close the plot window to exit.")
    plt.show()

    print("\nSimulator finished.")

