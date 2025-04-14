# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # 用于显示进度条
import os # 导入 os 模块用于创建文件夹
from sklearn.preprocessing import StandardScaler # --- Import StandardScaler ---
import random # For sampling parameters
import copy # For saving best model state
import argparse # --- Import argparse ---
import joblib # --- Import joblib to save scaler ---
from torch.optim.lr_scheduler import ReduceLROnPlateau # --- Import LR Scheduler ---

# --- 1. 使用 RK4 模拟带外力的单摆数据 (返回 3 特征) ---
# (Function simulate_forced_pendulum_rk4 remains the same)
def pendulum_dynamics(t, state, g, L, b, force_amplitude, force_frequency):
    """Defines the differential equations for the forced pendulum."""
    theta, omega = state
    force = force_amplitude * np.sin(force_frequency * t)
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta) - b * omega + force
    # Return derivatives and the force calculated at this time t
    return np.array([dtheta_dt, domega_dt]), force

def simulate_forced_pendulum_rk4(num_steps, dt=0.02, g=9.81, L=1.0, b=0.1,
                                 initial_theta=np.pi/4, initial_omega=0.0,
                                 force_amplitude=0.5, force_frequency=1.0):
    """
    Simulate forced pendulum using RK4 method.
    Returns states with 3 features: (angle, angular velocity, force).
    """
    states_theta_omega = np.zeros((num_steps, 2)) # Store theta, omega
    forces = np.zeros(num_steps)      # Store force
    times = np.arange(num_steps) * dt

    states_theta_omega[0] = [initial_theta, initial_omega]
    # Calculate initial force
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

# --- 2. 数据预处理 (目标y为2维) ---
# (Function create_sequences remains the same)
def create_sequences(data, seq_length):
    """
    Convert time series data into input sequences and targets for LSTM/GRU.
    Input: (theta, omega, force)
    Target: next (theta, omega)
    """
    X, y = [], []
    num_features = data.shape[1]
    num_output_features = 2 # Target is only theta and omega
    if num_features != 3:
         raise ValueError("Input data must have 3 features (theta, omega, force)")

    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length]) # Input sequence includes theta, omega, force
        y.append(data[i+seq_length, :num_output_features]) # Indices 0, 1 for theta, omega
    return np.array(X), np.array(y)

# --- 3. 定义 RNN 模型 (输入3维, 输出2维) ---
# (Class PendulumRNN remains the same)
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

# --- 4. 训练模型 (加入学习率调度器, 保存模型) ---
# (Function train_model remains the same)
def train_model(model, X_train, y_train, X_val, y_val,
                num_epochs=100, batch_size=32, learning_rate=0.001,
                patience=10, lr_scheduler_patience=5,
                model_save_path='best_model.pth', device='cpu'):
    """ Train the RNN model with validation, early stopping, and LR scheduling. """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=lr_scheduler_patience, verbose=True)
    print(f"Using Adam optimizer with learning rate: {learning_rate}")
    print(f"Using ReduceLROnPlateau scheduler with patience={lr_scheduler_patience}, factor=0.1")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print(f"Starting training on {device} for max {num_epochs} epochs (Early Stopping Patience={patience})...")
    for epoch in range(num_epochs):
        model.train()
        train_epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=False)
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
        avg_train_loss = train_epoch_loss / len(train_dataloader)
        loss_history.append(avg_train_loss)

        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_dataloader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                outputs_val = model(batch_X_val)
                val_loss = criterion(outputs_val, batch_y_val)
                val_epoch_loss += val_loss.item()

        if len(val_dataloader) > 0:
             avg_val_loss = val_epoch_loss / len(val_dataloader)
             val_loss_history.append(avg_val_loss)
             print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
             if avg_val_loss < best_val_loss:
                 best_val_loss = avg_val_loss; epochs_no_improve = 0
                 best_model_state = copy.deepcopy(model.state_dict())
                 print(f"Validation loss improved to {best_val_loss:.6f}.")
             else:
                 epochs_no_improve += 1
                 print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
             if epochs_no_improve >= patience:
                 print(f"Early stopping triggered after {epoch+1} epochs."); break
             scheduler.step(avg_val_loss)
        else:
             print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, No validation data.")

    if best_model_state:
        print(f"Saving best model state to {model_save_path} (Val Loss: {best_val_loss:.6f})")
        torch.save(best_model_state, model_save_path)
        model.load_state_dict(best_model_state)
    else:
         print(f"Warning: No best model state found/saved. Using model from last epoch. Model not saved to {model_save_path}.")

    print("Training finished.")
    return loss_history, val_loss_history

# --- 5. 评估和预测 (函数不变) ---
# (Function predict_with_known_future_force remains the same)
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
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description='Train RNN for Pendulum Prediction')
    parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN layer')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], help='Type of RNN cell (lstm or gru)')
    parser.add_argument('--seq_length', type=int, default=150, help='Input sequence length')
    parser.add_argument('--predict_steps', type=int, default=500, help='Number of steps to predict during evaluation') # --- Added predict_steps arg ---
    args = parser.parse_args()
    # --- End Argument Parser ---


    # --- Parameters ---
    # Data Generation Params
    NUM_SIMULATIONS = 50
    NUM_STEPS_PER_SIM = 2500 # Should be >= start_index + seq_length + predict_steps
    DT = 0.02
    THETA_RANGE = (-np.pi/2, np.pi/2)
    OMEGA_RANGE = (-1.5, 1.5)
    FORCE_AMP_RANGE = (0.1, 1.0)
    FORCE_FREQ_RANGE = (0.5, 2.5)

    # Model & Training Params - Now using args
    RNN_TYPE = args.rnn_type
    SEQ_LENGTH = args.seq_length
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = args.num_layers
    NUM_EPOCHS = 80
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 128
    TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)
    PATIENCE = 10 # Early stopping patience
    LR_SCHEDULER_PATIENCE = 5 # LR scheduler patience
    PREDICT_STEPS = args.predict_steps # Use command-line arg or default
    # --- Update SAVE_DIR based on runtime parameters ---
    # Note: SAVE_DIR name no longer includes 'lrsched' explicitly as it's now default part of training
    SAVE_DIR = f"{RNN_TYPE}_h{HIDDEN_SIZE}_l{NUM_LAYERS}_seq{SEQ_LENGTH}_rk4_es"

    # --- Create Save Directory ---
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    # --- Define paths for saving model and scaler ---
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_model_{RNN_TYPE}_h{HIDDEN_SIZE}_l{NUM_LAYERS}_seq{SEQ_LENGTH}.pth")
    SCALER_SAVE_PATH = os.path.join(SAVE_DIR, f"scaler_{RNN_TYPE}_h{HIDDEN_SIZE}_l{NUM_LAYERS}_seq{SEQ_LENGTH}.joblib")


    # --- Set Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device detected. Using MPS for training.")
    else:
        device = torch.device("cpu")
        print("No MPS or CUDA device detected. Using CPU for training.")

    # --- 1. Generate Diverse Data using RK4 (3 Features) ---
    # (Data generation code remains the same)
    print(f"Generating data (3 features) from {NUM_SIMULATIONS} simulations using RK4...")
    # (Code omitted for brevity - same as previous version)
    all_states_list = []
    pbar_sim = tqdm(range(NUM_SIMULATIONS), desc="Simulations")
    for i in pbar_sim:
        sim_theta0 = random.uniform(*THETA_RANGE); sim_omega0 = random.uniform(*OMEGA_RANGE)
        sim_amp = random.uniform(*FORCE_AMP_RANGE); sim_freq = random.uniform(*FORCE_FREQ_RANGE)
        pbar_sim.set_postfix(theta0=f"{sim_theta0:.2f}", amp=f"{sim_amp:.2f}", freq=f"{sim_freq:.2f}")
        times, states_3_features = simulate_forced_pendulum_rk4(
            NUM_STEPS_PER_SIM, dt=DT, initial_theta=sim_theta0, initial_omega=sim_omega0,
            force_amplitude=sim_amp, force_frequency=sim_freq)
        all_states_list.append(states_3_features)
    all_states = np.concatenate(all_states_list, axis=0)
    print(f"Total generated states shape: {all_states.shape}")


    # --- 2. Data Preprocessing ---
    # (Data preprocessing code remains the same)
    print("Scaling aggregated data (3 features) using StandardScaler...")
    scaler = StandardScaler()
    scaled_states = scaler.fit_transform(all_states) # Fit on 3 features
    print(f"Data scaled. Mean: {scaler.mean_}, Scale (Std Dev): {scaler.scale_}")
    # --- Save the fitted scaler ---
    try:
        joblib.dump(scaler, SCALER_SAVE_PATH)
        print(f"Scaler saved to {SCALER_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving scaler: {e}")
    # --- End saving scaler ---
    print("Creating sequences from aggregated data...")
    # (Sequence creation and splitting code remains the same)
    X, y = create_sequences(scaled_states, SEQ_LENGTH)
    print(f"Input sequences X shape: {X.shape}")
    print(f"Target sequences y shape: {y.shape}")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]; y = y[indices]
    n_total = X.shape[0]
    n_train = int(n_total * TRAIN_VAL_TEST_SPLIT[0])
    n_val = int(n_total * TRAIN_VAL_TEST_SPLIT[1])
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    print(f"Data split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    X_train = torch.FloatTensor(X_train); y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val); y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test); y_test = torch.FloatTensor(y_test)


    # --- 3. Define Model ---
    # (Model definition uses HIDDEN_SIZE and NUM_LAYERS from args)
    input_size = 3 # theta, omega, force
    output_size = 2 # predicting theta, omega
    print(f"Model Input Size: {input_size}, Output Size: {output_size}")
    model = PendulumRNN(input_size, HIDDEN_SIZE, output_size, NUM_LAYERS, rnn_type=RNN_TYPE)
    print("\nModel Structure:")
    print(model)


    # --- 4. Train Model ---
    # Pass the LR scheduler patience
    loss_history, val_loss_history = train_model(
                               model, X_train, y_train, X_val, y_val,
                               num_epochs=NUM_EPOCHS,
                               batch_size=BATCH_SIZE,
                               learning_rate=LEARNING_RATE,
                               patience=PATIENCE,
                               lr_scheduler_patience=LR_SCHEDULER_PATIENCE, # Pass scheduler patience
                               model_save_path=MODEL_SAVE_PATH,
                               device=device)


    # --- Visualize Training & Validation Loss ---
    # (Plotting uses dynamic title and filename)
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title(f"Loss ({RNN_TYPE.upper()} H={HIDDEN_SIZE} L={NUM_LAYERS} Seq={SEQ_LENGTH} LR Sched)") # Updated title
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error Loss")
    plt.legend()
    plt.grid(True)
    save_path_loss = os.path.join(SAVE_DIR, f"train_val_loss_{RNN_TYPE}_h{HIDDEN_SIZE}_l{NUM_LAYERS}_seq{SEQ_LENGTH}_lrsched.png") # Updated filename
    plt.savefig(save_path_loss)
    print(f"Saved train/val loss plot to: {save_path_loss}")
    plt.show()
    plt.close() # Close the loss plot figure


    # --- 5. Evaluate and Plot Multiple Test Scenarios (Static Plots) ---
    # (Evaluation loop uses PREDICT_STEPS from args)
    print(f"\n--- Evaluating on Multiple Test Scenarios ({RNN_TYPE.upper()} H={HIDDEN_SIZE} L={NUM_LAYERS} Seq={SEQ_LENGTH} LR Sched) ---")

    # --- Define Test Scenarios ---
    test_scenarios = [
        {'name': 'Scenario_1', 'theta0': np.pi / 6, 'omega0': 0.5, 'amp': 0.6, 'freq': 1.8},
        {'name': 'Scenario_2', 'theta0': 0.1,       'omega0': 0.0, 'amp': 0.2, 'freq': 0.7},
        {'name': 'Scenario_3', 'theta0': -np.pi / 4, 'omega0': -0.5,'amp': 0.9, 'freq': 1.5},
        {'name': 'Scenario_4', 'theta0': np.pi / 3, 'omega0': 1.0, 'amp': 0.5, 'freq': 2.3},
        {'name': 'Scenario_5', 'theta0': -0.2,      'omega0': 1.2, 'amp': 0.7, 'freq': 1.2},
    ]

    # --- Loop through scenarios ---
    for scenario in test_scenarios:
        test_name = scenario['name']
        test_theta0 = scenario['theta0']
        test_omega0 = scenario['omega0']
        test_amp = scenario['amp']
        test_freq = scenario['freq']

        print(f"\n--- Running Test {test_name} ---")
        print(f"Parameters: theta0={test_theta0:.2f}, omega0={test_omega0:.2f}, amp={test_amp:.2f}, freq={test_freq:.2f}")

        # Simulate ground truth using RK4 (3 features needed for eval)
        print("Simulating ground truth...")
        # Ensure simulation is long enough
        sim_steps_for_eval = SEQ_LENGTH + PREDICT_STEPS + 100 # Add buffer
        test_times, test_states_original_3_features = simulate_forced_pendulum_rk4(
            sim_steps_for_eval, dt=DT, initial_theta=test_theta0, initial_omega=test_omega0,
            force_amplitude=test_amp, force_frequency=test_freq
        )

        # Scale test data (3 features)
        test_states_scaled = scaler.transform(test_states_original_3_features)

        # Find starting sequence
        start_index_in_test_sim = 100
        if start_index_in_test_sim + SEQ_LENGTH >= len(test_states_scaled):
             print(f"Warning: Test simulation for {test_name} not long enough for start index {start_index_in_test_sim}. Skipping.")
             continue

        start_sequence_scaled = test_states_scaled[start_index_in_test_sim : start_index_in_test_sim + SEQ_LENGTH]
        time_at_sequence_end = test_times[start_index_in_test_sim + SEQ_LENGTH - 1]

        print(f"Predicting next {PREDICT_STEPS} steps...")
        predictions_original = predict_with_known_future_force(
            model, start_sequence_scaled, SEQ_LENGTH, PREDICT_STEPS, dt=DT,
            predict_force_amp=test_amp, predict_force_freq=test_freq,
            scaler=scaler, start_time=time_at_sequence_end, device=device
        )

        # Get corresponding actual values for the prediction period
        actual_start_index = start_index_in_test_sim + SEQ_LENGTH
        # Ensure prediction did not stop early due to NaN/Inf
        actual_pred_len = len(predictions_original)
        if actual_pred_len == 0:
             print(f"Warning: Prediction failed for {test_name}. Skipping plot.")
             continue
        actual_end_index = actual_start_index + actual_pred_len

        if actual_end_index > len(test_states_original_3_features):
            print(f"Warning: Actual data length ({len(test_states_original_3_features)}) insufficient for comparison up to prediction end index ({actual_end_index}). Skipping plot for {test_name}.")
            continue

        actual_future_states = test_states_original_3_features[actual_start_index:actual_end_index, :2] # theta, omega
        actual_future_times = test_times[actual_start_index:actual_end_index]
        predicted_times = time_at_sequence_end + np.arange(1, actual_pred_len + 1) * DT

        # Calculate errors
        angle_error = predictions_original[:, 0] - actual_future_states[:, 0]
        velocity_error = predictions_original[:, 1] - actual_future_states[:, 1]

        # --- Create Static Plot with 4 Subplots ---
        print(f"Creating static plot for {test_name}...")
        fig, axs = plt.subplots(2, 2, figsize=(14, 10)) # 2x2 layout
        fig.suptitle(f'Prediction vs Actual: {test_name} ({RNN_TYPE.upper()} H={HIDDEN_SIZE} L={NUM_LAYERS} Seq={SEQ_LENGTH} LR Sched)\nAmp={test_amp:.2f}, Freq={test_freq:.2f}')

        # Subplot 1: Angle Comparison
        axs[0, 0].plot(actual_future_times, actual_future_states[:, 0], label='Actual Angle', color='blue', linewidth=2)
        axs[0, 0].plot(predicted_times, predictions_original[:, 0], label='Predicted Angle', color='red', linestyle='--', linewidth=1.5)
        axs[0, 0].set_title('Angle Comparison')
        axs[0, 0].set_ylabel('Angle (rad)')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Subplot 2: Velocity Comparison
        axs[0, 1].plot(actual_future_times, actual_future_states[:, 1], label='Actual Velocity', color='blue', linewidth=2)
        axs[0, 1].plot(predicted_times, predictions_original[:, 1], label='Predicted Velocity', color='red', linestyle='--', linewidth=1.5)
        axs[0, 1].set_title('Velocity Comparison')
        axs[0, 1].set_ylabel('Velocity (rad/s)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Subplot 3: Angle Error
        axs[1, 0].plot(actual_future_times, angle_error, label='Angle Error', color='green', linewidth=1)
        axs[1, 0].axhline(0, color='gray', linestyle='--', linewidth=1) # Zero error line
        axs[1, 0].set_title('Angle Error (Pred - Actual)')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Error (rad)')
        axs[1, 0].grid(True)

        # Subplot 4: Velocity Error
        axs[1, 1].plot(actual_future_times, velocity_error, label='Velocity Error', color='purple', linewidth=1)
        axs[1, 1].axhline(0, color='gray', linestyle='--', linewidth=1) # Zero error line
        axs[1, 1].set_title('Velocity Error (Pred - Actual)')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Error (rad/s)')
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        # Save the figure
        plot_filename = os.path.join(SAVE_DIR, f"prediction_comparison_{RNN_TYPE}_h{HIDDEN_SIZE}_l{NUM_LAYERS}_seq{SEQ_LENGTH}_{test_name}.png")
        plt.savefig(plot_filename)
        print(f"Saved static plot to: {plot_filename}")
        plt.close(fig) # Close the figure

    print(f"\nMulti-scenario static plot generation finished for {RNN_TYPE.upper()} H={HIDDEN_SIZE} L={NUM_LAYERS} Seq={SEQ_LENGTH} LR Sched model.")

