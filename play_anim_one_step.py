# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from tqdm import tqdm # For single-step prediction progress

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

# --- 主程序 ---
if __name__ == "__main__":
    # --- Argument Parser for Single-Step Animation ---
    parser = argparse.ArgumentParser(description='Animate Single-Step Pendulum Prediction')
    # Model parameters
    parser.add_argument('--hidden_size', type=int, required=True, help='Number of hidden units (must match trained model)')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of layers (must match trained model)')
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], help='RNN type (must match trained model)')
    parser.add_argument('--seq_length', type=int, default=150, help='Sequence length used during training')
    # Initial conditions and force parameters for simulation scenario
    parser.add_argument('--theta0', type=float, default=np.pi/3, help='Initial angle (rad)')
    parser.add_argument('--omega0', type=float, default=1.0, help='Initial angular velocity (rad/s)')
    parser.add_argument('--amp', type=float, default=1.0, help='Force amplitude')
    parser.add_argument('--freq', type=float, default=10.0, help='Force frequency')
    # Animation parameters
    parser.add_argument('--anim_steps', type=int, default=300, help='Number of steps to animate')
    parser.add_argument('--dt', type=float, default=0.02, help='Time step')
    parser.add_argument('--L', type=float, default=1.0, help='Pendulum length')

    args = parser.parse_args()
    # --- End Argument Parser ---

    # --- Construct paths based on args ---
    MODEL_DIR = f"{args.rnn_type}_h{args.hidden_size}_l{args.num_layers}_seq{args.seq_length}_rk4_es" # Match training script's naming
    MODEL_PATH = os.path.join(MODEL_DIR, f"best_model_{args.rnn_type}_h{args.hidden_size}_l{args.num_layers}_seq{args.seq_length}.pth")
    SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_{args.rnn_type}_h{args.hidden_size}_l{args.num_layers}_seq{args.seq_length}.joblib")

    print(f"Attempting to load model from: {MODEL_PATH}")
    print(f"Attempting to load scaler from: {SCALER_PATH}")

    # --- Load Scaler ---
    if not os.path.exists(SCALER_PATH): print(f"Error: Scaler file not found at {SCALER_PATH}"); exit()
    try:
        scaler: StandardScaler = joblib.load(SCALER_PATH)
        print("Scaler loaded successfully.")
        if len(scaler.mean_) != 3: raise ValueError("Scaler feature mismatch")
    except Exception as e: print(f"Error loading scaler: {e}"); exit()

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH): print(f"Error: Model file not found at {MODEL_PATH}"); exit()
    input_size = 3; output_size = 2
    model = PendulumRNN(input_size, args.hidden_size, output_size, args.num_layers, rnn_type=args.rnn_type)
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    model.to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval(); print(f"Model loaded successfully from {MODEL_PATH} and set to eval mode.")
    except Exception as e: print(f"Error loading model state dict: {e}"); exit()

    # --- Simulate Ground Truth ---
    total_sim_steps = args.seq_length + args.anim_steps # Steps needed for history + animation
    print(f"Simulating {total_sim_steps} steps for ground truth using RK4...")
    all_times, all_states_orig = simulate_forced_pendulum_rk4(
        total_sim_steps, dt=args.dt, initial_theta=args.theta0, initial_omega=args.omega0,
        force_amplitude=args.amp, force_frequency=args.freq, L=args.L
    )
    all_states_scaled = scaler.transform(all_states_orig)

    # --- Perform Single-Step Prediction over the Animation Period ---
    print(f"Performing single-step prediction for {args.anim_steps} steps...")
    predictions_1step_scaled = []
    eval_start_idx = args.seq_length
    eval_end_idx = eval_start_idx + args.anim_steps

    with torch.no_grad():
        for t in tqdm(range(eval_start_idx, eval_end_idx), desc="Single-Step Predicting"):
            input_sequence_scaled = all_states_scaled[t - args.seq_length : t]
            input_tensor = torch.FloatTensor(input_sequence_scaled).unsqueeze(0).to(device)
            pred_scaled = model(input_tensor) # Shape: (1, 2)
            if torch.isnan(pred_scaled).any() or torch.isinf(pred_scaled).any():
                print(f"Warning: NaN/Inf detected at step {t}. Stopping prediction.")
                # Fill remaining predictions with NaN or stop? Let's fill.
                remaining_steps = eval_end_idx - t
                predictions_1step_scaled.extend([np.array([np.nan, np.nan])] * remaining_steps)
                break
            predictions_1step_scaled.append(pred_scaled.cpu().numpy().flatten())

    predictions_1step_scaled = np.array(predictions_1step_scaled)

    # Check if prediction generated expected number of steps
    if len(predictions_1step_scaled) != args.anim_steps:
         print(f"Warning: Only {len(predictions_1step_scaled)} steps predicted out of {args.anim_steps}.")
         # Adjust animation length if prediction stopped early
         args.anim_steps = len(predictions_1step_scaled)
         if args.anim_steps == 0:
             print("No valid prediction steps generated. Exiting.")
             exit()
         eval_end_idx = eval_start_idx + args.anim_steps


    # Inverse transform predictions
    mean_output = scaler.mean_[:2]; scale_output = scaler.scale_[:2]
    if np.any(scale_output == 0): scale_output[scale_output == 0] = 1.0
    predictions_1step_orig = predictions_1step_scaled * scale_output + mean_output

    # Get corresponding ground truth for the animation period
    actual_states_orig_anim = all_states_orig[eval_start_idx:eval_end_idx, :2] # theta, omega
    times_anim = all_times[eval_start_idx:eval_end_idx]

    # Calculate errors
    angle_error = predictions_1step_orig[:, 0] - actual_states_orig_anim[:, 0]
    velocity_error = predictions_1step_orig[:, 1] - actual_states_orig_anim[:, 1]

    # --- Create Animation ---
    print("Creating animation...")
    fig, (ax1, ax5, ax4, ax2, ax3) = plt.subplots(5, 1, figsize=(8, 14), # 5 subplots
                                                 gridspec_kw={'height_ratios': [5, 2, 2, 1, 1]})
    fig.suptitle(f'Single-Step Prediction\nAmp={args.amp:.2f}, Freq={args.freq:.2f} ({args.rnn_type.upper()} H={args.hidden_size} L={args.num_layers} Seq={args.seq_length})')
    pendulum_length_L = args.L

    # Plot 1: Overlayed Pendulum
    ax1.set_title('Pendulum Motion (Actual vs 1-Step Pred)')
    ax1.set_xlim(-pendulum_length_L*1.2, pendulum_length_L*1.2); ax1.set_ylim(-pendulum_length_L*1.2, pendulum_length_L*1.2)
    ax1.set_aspect('equal', adjustable='box'); ax1.grid(True)
    line_actual, = ax1.plot([], [], 'o-', lw=3, color='blue', alpha=0.7, label='Actual (RK4)', markersize=10)
    line_pred, = ax1.plot([], [], 'o-', lw=2, color='red', alpha=0.7, label='1-Step Pred', markersize=7)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes); ax1.legend(loc='upper right')

    # Plot 5: Angle Comparison Curve
    ax5.set_title('Angle Comparison (1-Step Pred)')
    ax5.set_ylabel('Angle (rad)'); ax5.grid(True)
    line_angle_actual, = ax5.plot([], [], lw=2, color='blue', alpha=0.7, label='Actual')
    line_angle_pred, = ax5.plot([], [], lw=1.5, color='red', linestyle='-.', label='1-Step Pred')
    ax5.legend(loc='upper right'); ax5.set_xlim(times_anim[0], times_anim[-1])

    # Plot 4: Velocity Comparison Curve
    ax4.set_title('Velocity Comparison (1-Step Pred)')
    ax4.set_ylabel('Velocity (rad/s)'); ax4.grid(True)
    line_vel_actual, = ax4.plot([], [], lw=2, color='blue', alpha=0.7, label='Actual')
    line_vel_pred, = ax4.plot([], [], lw=1.5, color='red', linestyle='-.', label='1-Step Pred')
    ax4.legend(loc='upper right'); ax4.set_xlim(times_anim[0], times_anim[-1])

    # Plot 2: Angle Error Curve
    ax2.set_title('Angle Error (1-Step Pred - Actual)')
    ax2.set_ylabel('Angle Error (rad)'); ax2.grid(True)
    line_angle_error, = ax2.plot([], [], lw=1, color='green')
    ax2.set_xlim(times_anim[0], times_anim[-1]); ax2.axhline(0, color='gray', linestyle=':', linewidth=1)

    # Plot 3: Velocity Error Curve
    ax3.set_title('Velocity Error (1-Step Pred - Actual)')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Velocity Error (rad/s)'); ax3.grid(True)
    line_velocity_error, = ax3.plot([], [], lw=1, color='purple')
    ax3.set_xlim(times_anim[0], times_anim[-1]); ax3.axhline(0, color='gray', linestyle=':', linewidth=1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Data for animation
    actual_theta_anim = actual_states_orig_anim[:, 0]
    actual_omega_anim = actual_states_orig_anim[:, 1]
    predicted_theta_anim = predictions_1step_orig[:, 0]
    predicted_omega_anim = predictions_1step_orig[:, 1]

    # Initialization function
    def init():
        # Pendulum lines
        line_actual.set_data([], [])
        line_pred.set_data([], [])
        time_text.set_text('')
        # Error lines
        line_angle_error.set_data([], [])
        line_velocity_error.set_data([], [])
        # Comparison lines
        line_vel_actual.set_data([], [])
        line_vel_pred.set_data([], [])
        line_angle_actual.set_data([], [])
        line_angle_pred.set_data([], [])
        return (line_actual, line_pred, time_text, line_angle_error, line_velocity_error,
                line_vel_actual, line_vel_pred, line_angle_actual, line_angle_pred)

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
        current_actual_theta = actual_theta_anim[:i+1]
        current_pred_theta = predicted_theta_anim[:i+1]

        # Update error plots
        line_angle_error.set_data(current_times, current_angle_error)
        line_velocity_error.set_data(current_times, current_velocity_error)

        # Update velocity comparison plot
        line_vel_actual.set_data(current_times, current_actual_omega)
        line_vel_pred.set_data(current_times, current_pred_omega)

        # Update angle comparison plot
        line_angle_actual.set_data(current_times, current_actual_theta)
        line_angle_pred.set_data(current_times, current_pred_theta)

        # Adjust y-limits dynamically for error and comparison plots
        if i > 0:
            min_ae = np.nanmin(current_angle_error); max_ae = np.nanmax(current_angle_error)
            margin_ae = max(0.05, (max_ae - min_ae) * 0.1); ax2.set_ylim(min_ae - margin_ae, max_ae + margin_ae)

            min_ve = np.nanmin(current_velocity_error); max_ve = np.nanmax(current_velocity_error)
            margin_ve = max(0.1, (max_ve - min_ve) * 0.1); ax3.set_ylim(min_ve - margin_ve, max_ve + margin_ve)

            min_vo = min(np.nanmin(current_actual_omega), np.nanmin(current_pred_omega))
            max_vo = max(np.nanmax(current_actual_omega), np.nanmax(current_pred_omega))
            margin_vo = max(0.1, (max_vo - min_vo) * 0.1); ax4.set_ylim(min_vo - margin_vo, max_vo + margin_vo)

            min_ao = min(np.nanmin(current_actual_theta), np.nanmin(current_pred_theta))
            max_ao = max(np.nanmax(current_actual_theta), np.nanmax(current_pred_theta))
            margin_ao = max(0.1, (max_ao - min_ao) * 0.1); ax5.set_ylim(min_ao - margin_ao, max_ao + margin_ao)

        return (line_actual, line_pred, time_text, line_angle_error, line_velocity_error,
                line_vel_actual, line_vel_pred, line_angle_actual, line_angle_pred)

    # Create and display animation
    num_frames = args.anim_steps
    frame_interval = int(args.dt * 1000 * 1) # Adjust speed
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=frame_interval, blit=True, init_func=init)

    print("Displaying animation... Close the plot window to exit.")
    # Note: Saving this animation might be slow due to many plot elements
    # To save: uncomment below (ensure SAVE_DIR is defined if needed)
    # anim_filename = f"animation_1step_{args.rnn_type}_h{args.hidden_size}_l{args.num_layers}_seq{args.seq_length}.gif"
    # print(f"Saving animation to {anim_filename}...")
    # ani.save(anim_filename, writer='pillow', fps=int(1000 / frame_interval))
    # print("Animation saved.")

    plt.show()

    print("\nSingle-step animation finished.")

