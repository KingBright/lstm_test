# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # 用于显示进度条
import os # 导入 os 模块用于创建文件夹

# --- 1. 模拟带外力的单摆数据 ---
# 修改模拟器以包含外力项 F(t) = force_amplitude * sin(force_frequency * t)
# d(theta)/dt = omega
# d(omega)/dt = -(g/L) * sin(theta) - b * omega + F(t)

def simulate_forced_pendulum(num_steps, dt=0.02, g=9.81, L=1.0, b=0.1,
                             initial_theta=np.pi/4, initial_omega=0.0,
                             force_amplitude=0.5, force_frequency=1.0):
    """
    Simulate pendulum motion with an external sinusoidal force.

    Args:
        num_steps (int): Total number of simulation steps.
        dt (float): Time step.
        g (float): Acceleration due to gravity.
        L (float): Length of the pendulum.
        b (float): Damping coefficient.
        initial_theta (float): Initial angle (radians).
        initial_omega (float): Initial angular velocity.
        force_amplitude (float): Amplitude of the external force.
        force_frequency (float): Frequency (omega) of the external force.

    Returns:
        tuple: Contains time and states (angle, angular velocity, force) (times, states).
    """
    theta = np.zeros(num_steps)
    omega = np.zeros(num_steps)
    forces = np.zeros(num_steps) # Array to store the force at each step
    times = np.arange(num_steps) * dt

    theta[0] = initial_theta
    omega[0] = initial_omega
    forces[0] = force_amplitude * np.sin(force_frequency * times[0])

    for i in range(num_steps - 1):
        # Calculate external force at the current time step
        current_force = force_amplitude * np.sin(force_frequency * times[i])
        forces[i] = current_force # Store the force

        # Calculate angular acceleration including the external force
        omega_dot = -(g / L) * np.sin(theta[i]) - b * omega[i] + current_force
        omega[i+1] = omega[i] + omega_dot * dt # Update angular velocity
        theta[i+1] = theta[i] + omega[i+1] * dt # Update angle

    # Store the force for the last step
    forces[num_steps-1] = force_amplitude * np.sin(force_frequency * times[num_steps-1])

    # Combine angle, angular velocity, and force into states
    # Shape will be (num_steps, 3)
    states = np.stack((theta, omega, forces), axis=1)
    return times, states

# --- 2. 数据预处理 ---
# create_sequences function remains the same logically, but will handle 3 features now.
def create_sequences(data, seq_length):
    """
    Convert time series data into input sequences and targets for LSTM.
    The input sequence will contain all features, the target will be the non-force features.

    Args:
        data (np.ndarray): Time series data (num_steps, num_features).
        seq_length (int): Length of the input sequence.

    Returns:
        tuple: Input sequences (X) and target sequences (y).
               X shape: (num_samples, seq_length, num_features)
               y shape: (num_samples, num_output_features) # Output features = angle, omega
    """
    X, y = [], []
    num_features = data.shape[1]
    # Assuming the target variables (angle, omega) are the first two columns
    num_output_features = 2
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length]) # Input sequence includes all features (theta, omega, force)
        y.append(data[i+seq_length, :num_output_features]) # Target is the next state's theta and omega
    return np.array(X), np.array(y)

# --- 3. 定义 LSTM 模型 ---
# PendulumLSTM class remains the same, but will be initialized with input_size=3
class PendulumLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): Number of input features (3: angle, angular velocity, force).
            hidden_size (int): Size of the LSTM hidden layer.
            output_size (int): Number of output features (2: predicted angle, angular velocity).
            num_layers (int): Number of LSTM layers.
        """
        super(PendulumLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Define LSTM layer
        # batch_first=True means input and output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Define a fully connected layer to map LSTM output to final prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input sequence tensor (batch, seq_length, input_size).

        Returns:
            torch.Tensor: Predicted output tensor (batch, output_size).
        """
        # Initialize hidden state and cell state
        # h0 shape: (num_layers, batch, hidden_size)
        # c0 shape: (num_layers, batch, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        # out: Output of the last LSTM layer for all time steps (batch, seq_length, hidden_size)
        # hn: Hidden state at the last time step (num_layers, batch, hidden_size)
        # cn: Cell state at the last time step (num_layers, batch, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # We only need the output of the last time step for prediction
        # out[:, -1, :] has shape (batch, hidden_size)
        out = self.fc(out[:, -1, :])
        return out

# --- 4. 训练模型 ---
# train_model function remains the same.
def train_model(model, X_train, y_train, num_epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
    """
    Train the LSTM model.

    Args:
        model (nn.Module): The LSTM model to train.
        X_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        device (str): Training device ('cpu', 'cuda', 'mps').

    Returns:
        list: List of average loss per epoch.
    """
    model.to(device) # Move model to the specified device
    criterion = nn.MSELoss() # Use Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Use Adam optimizer

    # Create data loader
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        # Use tqdm for progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device) # Move data to device

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad() # Clear gradients
            loss.backward() # Compute gradients
            optimizer.step() # Update weights

            epoch_loss += loss.item()
            # Update progress bar with current batch loss
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.6f}")

    print("Training finished.")
    return loss_history

# --- 5. 评估和预测 ---
# Modify predict function to handle force input
def predict_forced(model, start_data, seq_length, future_steps, dt, force_amplitude, force_frequency, start_time, device='cpu'):
    """
    Perform multi-step prediction for the forced pendulum.
    Requires calculating the force for future steps.

    Args:
        model (nn.Module): Trained model.
        start_data (np.ndarray): Initial sequence data (seq_length, num_features=3). Includes theta, omega, force.
        seq_length (int): Input sequence length.
        future_steps (int): Number of future steps to predict.
        dt (float): Time step used in simulation.
        force_amplitude (float): Amplitude of the external force.
        force_frequency (float): Frequency of the external force.
        start_time (float): The time corresponding to the *end* of the start_data sequence.
        device (str): Prediction device.

    Returns:
        np.ndarray: Predicted future state sequence (angle, angular velocity) (future_steps, 2).
    """
    model.eval() # Set model to evaluation mode
    model.to(device)
    predictions = [] # Will store predicted [theta, omega] pairs

    # Ensure start_data is a numpy array
    if not isinstance(start_data, np.ndarray):
        print("Warning: Input start_data to predict function is not a numpy array. Attempting conversion.")
        try:
            start_data = np.array(start_data)
        except Exception as e:
            print(f"Error converting input data to numpy array: {e}")
            return np.array([])

    # Check shape of starting data
    if start_data.shape != (seq_length, model.lstm.input_size):
         print(f"Error: Input data shape mismatch in predict_forced. Expected ({seq_length}, {model.lstm.input_size}), got {start_data.shape}")
         return np.array([])

    current_sequence_np = start_data.copy() # Keep a numpy copy for easier manipulation

    with torch.no_grad(): # No need to compute gradients during prediction
        for k in range(future_steps):
            # Convert current numpy sequence to tensor for model input
            current_sequence_tensor = torch.FloatTensor(current_sequence_np).unsqueeze(0).to(device) # (1, seq_length, 3)

            # Get prediction (predicted theta, omega for the next step)
            next_pred_state = model(current_sequence_tensor) # Shape: (1, 2)

            # Check for NaN or Inf in predictions
            if torch.isnan(next_pred_state).any() or torch.isinf(next_pred_state).any():
                print(f"Warning: NaN or Inf detected in prediction at step {k+1}. Stopping prediction.")
                break

            predicted_theta_omega = next_pred_state.cpu().numpy().flatten() # Shape: (2,)
            predictions.append(predicted_theta_omega) # Store predicted [theta, omega]

            # Calculate the force for the next time step
            # Time for the predicted step = start_time + (k+1)*dt
            next_time = start_time + (k + 1) * dt
            next_force = force_amplitude * np.sin(force_frequency * next_time)

            # Create the full state for the next step [predicted_theta, predicted_omega, next_force]
            next_full_state = np.append(predicted_theta_omega, next_force) # Shape: (3,)

            # Update the sequence: remove the oldest step, append the new full state
            current_sequence_np = np.vstack((current_sequence_np[1:], next_full_state))

    return np.array(predictions) # Return array of [theta, omega] predictions

# --- 主程序 ---
if __name__ == "__main__":
    # --- Parameters ---
    NUM_STEPS = 10000     # Increased simulation steps for potentially more complex dynamics
    DT = 0.02             # Time step
    SEQ_LENGTH = 60       # Sequence length (might need adjustment)
    HIDDEN_SIZE = 128     # Increased hidden size (might need adjustment)
    NUM_LAYERS = 2        # Number of LSTM layers
    NUM_EPOCHS = 60       # Increased epochs (might need adjustment)
    BATCH_SIZE = 64       # Batch size
    LEARNING_RATE = 0.001 # Learning rate
    TRAIN_SPLIT = 0.8     # Training set ratio
    PREDICT_STEPS = 500   # Number of future steps to predict
    SAVE_DIR = "forced_pendulum_plots" # Directory to save plots

    # Force parameters
    FORCE_AMP = 0.8       # Amplitude of the external force
    FORCE_FREQ = 1.5      # Frequency of the external force

    # Create directory for saving plots if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    # --- Check for MPS device on M2 Mac ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device detected. Using MPS for training.")
    # elif torch.cuda.is_available(): # Uncomment if you might use NVIDIA GPU
    #     device = torch.device("cuda")
    #     print("CUDA device detected. Using CUDA for training.")
    else:
        device = torch.device("cpu")
        print("No MPS or CUDA device detected. Using CPU for training.")

    # --- 1. Generate Data ---
    print("Simulating forced pendulum data...")
    times, states = simulate_forced_pendulum(NUM_STEPS, dt=DT,
                                             force_amplitude=FORCE_AMP,
                                             force_frequency=FORCE_FREQ)
    print(f"Generated data shape: {states.shape}") # (NUM_STEPS, 3)

    # --- Visualize Raw Data ---
    plt.figure(figsize=(18, 5)) # Wider figure
    plt.subplot(1, 3, 1)
    plt.plot(times, states[:, 0])
    plt.title("Simulated Angle (theta)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(times, states[:, 1])
    plt.title("Simulated Angular Velocity (omega)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.grid(True)

    plt.subplot(1, 3, 3) # Add plot for the force
    plt.plot(times, states[:, 2])
    plt.title("External Force")
    plt.xlabel("Time (s)")
    plt.ylabel("Force")
    plt.grid(True)

    plt.tight_layout()
    # Save the figure
    save_path_sim = os.path.join(SAVE_DIR, "simulated_forced_data.png")
    plt.savefig(save_path_sim)
    print(f"Saved simulated data plot to: {save_path_sim}")
    plt.show()


    # --- 2. Data Preprocessing ---
    # Data normalization (scaling all 3 features)
    min_vals = states.min(axis=0)
    max_vals = states.max(axis=0)
    scale_range = max_vals - min_vals
    scale_range[scale_range == 0] = 1.0 # Avoid division by zero
    scaled_states = (states - min_vals) / scale_range


    print("Creating sequence data...")
    # X will have shape (samples, seq_length, 3)
    # y will have shape (samples, 2) -> predicting next theta and omega
    X, y = create_sequences(scaled_states, SEQ_LENGTH)
    print(f"Input sequences X shape: {X.shape}")
    print(f"Target sequences y shape: {y.shape}")

    # Split into training and testing sets
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # --- 3. Define Model ---
    input_size = X_train.shape[2] # Should be 3 now (theta, omega, force)
    output_size = y_train.shape[1] # Should be 2 (predicting theta, omega)
    print(f"Model Input Size: {input_size}, Output Size: {output_size}")
    model = PendulumLSTM(input_size, HIDDEN_SIZE, output_size, NUM_LAYERS)
    print("\nModel Structure:")
    print(model)

    # --- 4. Train Model ---
    loss_history = train_model(model, X_train, y_train,
                               num_epochs=NUM_EPOCHS,
                               batch_size=BATCH_SIZE,
                               learning_rate=LEARNING_RATE,
                               device=device)

    # --- Visualize Training Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.title("Training Loss Curve (Forced Pendulum)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error Loss")
    plt.grid(True)
    # Save the figure
    save_path_loss = os.path.join(SAVE_DIR, "training_loss_forced.png")
    plt.savefig(save_path_loss)
    print(f"Saved training loss plot to: {save_path_loss}")
    plt.show()


    # --- 5. Evaluate and Predict ---
    # Use the FIRST sequence from the test set as the starting point for prediction
    start_sequence_index = 0
    if len(X_test) == 0:
         print("Warning: Test set is empty. Attempting to use the last sequence from the training set for prediction.")
         start_sequence_index = len(X_train) - 1
         if start_sequence_index < 0:
              print("Error: Training data is also empty. Cannot make predictions.")
              exit()
         if len(X_train) > 0:
             start_sequence_data_scaled = X_train[start_sequence_index].cpu().numpy() # Shape (SEQ_LENGTH, 3)
             split_idx = 0
             print(f"Using training sequence index {start_sequence_index} for prediction.")
         else:
             print("Error: Training data is empty. Cannot make predictions.")
             exit()
    else:
        # Use the first sequence from the test set
        start_sequence_data_scaled = X_test[start_sequence_index].cpu().numpy() # Shape (SEQ_LENGTH, 3)
        print(f"Using test sequence index {start_sequence_index} for prediction.")


    # Calculate the time corresponding to the end of the starting sequence
    start_pred_time_index = split_idx + start_sequence_index + SEQ_LENGTH
    # Need the time at the *last* point of the input sequence to calculate future forces
    time_at_sequence_end = times[start_pred_time_index - 1]


    print(f"\nPredicting next {PREDICT_STEPS} steps starting after time index {start_pred_time_index - 1} (time {time_at_sequence_end:.2f}s)...")

    # Make predictions (predictions_scaled will contain only theta, omega)
    predictions_scaled = predict_forced(model, start_sequence_data_scaled, SEQ_LENGTH, PREDICT_STEPS,
                                        dt=DT, force_amplitude=FORCE_AMP, force_frequency=FORCE_FREQ,
                                        start_time=time_at_sequence_end, device=device)

    # Check if prediction was successful
    if predictions_scaled.size == 0 or predictions_scaled.shape[1] != 2:
        print("Error: Prediction failed or returned unexpected shape. Cannot plot predictions.")
        exit()

    # Inverse transform the predictions (only theta and omega)
    # We need the min/max values for theta and omega (first 2 features)
    min_vals_output = min_vals[:2]
    scale_range_output = scale_range[:2]
    predictions = predictions_scaled * scale_range_output + min_vals_output

    # Get the corresponding actual values for comparison (only theta and omega)
    actual_end_pred_time_index = start_pred_time_index + len(predictions)

    # Ensure indices are within bounds
    if start_pred_time_index >= len(states):
         print(f"Error: Calculated start index {start_pred_time_index} for actual data is out of bounds (total states: {len(states)}). Cannot plot comparison.")
         exit()
    if actual_end_pred_time_index > len(states):
        print(f"Warning: Prediction range ({len(predictions)} steps) exceeds available actual data ({len(states) - start_pred_time_index} steps remaining). Truncating comparison.")
        actual_end_pred_time_index = len(states)
        # Trim predictions array to match the available actual data length for plotting
        predictions = predictions[:(actual_end_pred_time_index - start_pred_time_index)]


    actual_future_states = states[start_pred_time_index:actual_end_pred_time_index, :2] # Get only theta, omega
    actual_future_times = times[start_pred_time_index:actual_end_pred_time_index]

    # Adjust the time axis for predictions
    if start_pred_time_index > 0:
        num_pred_points = len(predictions)
        predicted_times = times[start_pred_time_index-1] + np.arange(1, num_pred_points + 1) * DT
    else:
         num_pred_points = len(predictions)
         predicted_times = np.arange(1, num_pred_points + 1) * DT

    # --- Add Debugging Prints ---
    print("\n--- Debugging Prediction Data (Forced) ---")
    print(f"Split index (split_idx): {split_idx}")
    print(f"Start sequence index for prediction (relative to test/train set): {start_sequence_index}")
    print(f"Calculated start time index for actual data (start_pred_time_index): {start_pred_time_index}")
    print(f"Calculated end time index for actual data (actual_end_pred_time_index): {actual_end_pred_time_index}")
    print(f"Shape of actual_future_times: {actual_future_times.shape}")
    print(f"Shape of actual_future_states (theta, omega): {actual_future_states.shape}")
    print(f"Shape of predicted_times: {predicted_times.shape}")
    print(f"Shape of predictions (theta, omega): {predictions.shape}")
    if actual_future_times.size > 0:
        print(f"First few actual_future_times: {actual_future_times[:5]}")
        print(f"First few actual_future_states (angle): {actual_future_states[:5, 0]}")
    else:
        print("actual_future_times/states are empty!")
    if predicted_times.size > 0:
        print(f"First few predicted_times: {predicted_times[:5]}")
        print(f"First few predictions (angle): {predictions[:5, 0]}")
    else:
        print("predicted_times/predictions are empty!")
    print("--- End Debugging ---")


    # --- Visualize Prediction Results ---
    if actual_future_times.size == 0 or predicted_times.size == 0 or predictions.size == 0 or actual_future_states.size == 0:
        print("\nCannot generate prediction plot because some data arrays are empty. Check debugging output above.")
    elif len(actual_future_times) != len(predicted_times):
         print(f"\nWarning: Mismatch in length between actual_future_times ({len(actual_future_times)}) and predicted_times ({len(predicted_times)}). Skipping plot.")
    else:
        plt.figure(figsize=(14, 6))

        # Plot Angle Prediction
        plt.subplot(1, 2, 1)
        plt.plot(times, states[:, 0], label='Full Simulated Trajectory', color='gray', alpha=0.5)
        plt.plot(actual_future_times, actual_future_states[:, 0], label='Actual Future Trajectory', color='blue', linestyle='-', linewidth=2.5)
        plt.plot(predicted_times, predictions[:, 0], label='LSTM Predicted Trajectory', color='red', linestyle='-.', linewidth=1.5)
        plt.title("Angle (theta) Prediction vs Actual (Forced)")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad)")
        plt.legend()
        plt.grid(True)
        try: # Add try-except for min/max in case of empty arrays after slicing/prediction issues
            min_angle_plot = min(actual_future_states[:, 0].min(), predictions[:, 0].min()) - 0.2
            max_angle_plot = max(actual_future_states[:, 0].max(), predictions[:, 0].max()) + 0.2
            plt.ylim(min_angle_plot, max_angle_plot)
        except ValueError:
             print("Could not determine plot limits for angle.")
        if len(predicted_times) > 0:
          plt.xlim(predicted_times[0] - (predicted_times[-1]-predicted_times[0])*0.1, predicted_times[-1] * 1.05)


        # Plot Angular Velocity Prediction
        plt.subplot(1, 2, 2)
        plt.plot(times, states[:, 1], label='Full Simulated Trajectory', color='gray', alpha=0.5)
        plt.plot(actual_future_times, actual_future_states[:, 1], label='Actual Future Trajectory', color='blue', linestyle='-', linewidth=2.5)
        plt.plot(predicted_times, predictions[:, 1], label='LSTM Predicted Trajectory', color='red', linestyle='-.', linewidth=1.5)
        plt.title("Angular Velocity (omega) Prediction vs Actual (Forced)")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (rad/s)")
        plt.legend()
        plt.grid(True)
        try: # Add try-except for min/max
            min_omega_plot = min(actual_future_states[:, 1].min(), predictions[:, 1].min()) - 0.5
            max_omega_plot = max(actual_future_states[:, 1].max(), predictions[:, 1].max()) + 0.5
            plt.ylim(min_omega_plot, max_omega_plot)
        except ValueError:
             print("Could not determine plot limits for angular velocity.")

        if len(predicted_times) > 0:
            plt.xlim(predicted_times[0] - (predicted_times[-1]-predicted_times[0])*0.1, predicted_times[-1] * 1.05)

        plt.tight_layout()
        # Save the figure
        save_path_pred = os.path.join(SAVE_DIR, "prediction_vs_actual_forced.png")
        plt.savefig(save_path_pred)
        print(f"Saved prediction plot to: {save_path_pred}")
        plt.show()


    print("\nPrediction finished and plots saved/shown.")

