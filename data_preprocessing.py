# data_preprocessing.py

import numpy as np
import pandas as pd
# Import both scalers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split # Needed for shuffle and split
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import config # Import config

# --- Updated create_sequences for Seq2Seq ---
# Note: Removed predict_delta argument from definition as Seq2Seq predicts absolute states here
def create_sequences(data, input_seq_len=config.INPUT_SEQ_LEN, output_seq_len=config.OUTPUT_SEQ_LEN, use_sincos=config.USE_SINCOS_THETA):
    """
    Creates input sequences and corresponding multi-step target sequences (Seq2Seq).
    Input features can be [theta, theta_dot, tau] or [sin(theta), cos(theta), theta_dot, tau].
    Target 'y' will be absolute states for the next 'output_seq_len' steps.

    Args:
        data (np.array): Numpy array with columns [theta, theta_dot, tau].
        input_seq_len (int): The number of time steps for each input sequence (N).
        output_seq_len (int): The number of future time steps to predict (K).
        use_sincos (bool): If True, use sin/cos(theta) for input features.

    Returns:
        tuple: (X, y)
               X shape: (num_samples, input_seq_len, num_input_features) (3 or 4)
               y shape: (num_samples, output_seq_len, num_output_features) (always 2)
    """
    X, y = [], []
    if data.shape[1] < 3: print(f"Error: Data needs at least 3 features."); return np.empty((0, input_seq_len, data.shape[1])), np.empty((0, output_seq_len, 2))

    num_input_features = 4 if use_sincos else 3
    num_output_features = 2
    total_len_needed = input_seq_len + output_seq_len

    if len(data) < total_len_needed: print(f"Warning: Data length ({len(data)}) < {total_len_needed}."); return np.empty((0, input_seq_len, num_input_features)), np.empty((0, output_seq_len, num_output_features))

    # print(f"  Creating Seq2Seq sequences: Input={input_seq_len} steps, Output={output_seq_len} steps, Use SinCos={use_sincos}") # Less verbose
    for i in range(len(data) - total_len_needed + 1):
        input_window = data[i : i + input_seq_len]
        target_window = data[i + input_seq_len : i + input_seq_len + output_seq_len]

        theta_in = input_window[:, 0]; theta_dot_in = input_window[:, 1]; tau_in = input_window[:, 2]
        if use_sincos: input_seq = np.stack([np.sin(theta_in), np.cos(theta_in), theta_dot_in, tau_in], axis=1)
        else: input_seq = np.stack([theta_in, theta_dot_in, tau_in], axis=1)

        target_seq = target_window[:, 0:num_output_features] # Absolute states

        X.append(input_seq); y.append(target_seq)

    if not X: return np.empty((0, input_seq_len, num_input_features)), np.empty((0, output_seq_len, num_output_features))
    return np.array(X), np.array(y)


# --- NEW Core Function for Shuffle-then-Split Data Prep (for Seq2Seq) ---
def prepare_shuffled_train_val_data(df_all, sequence_length=config.INPUT_SEQ_LEN,
                                    output_sequence_length=config.OUTPUT_SEQ_LEN,
                                    val_fraction=config.VAL_SET_FRACTION,
                                    seed=config.SEED):
    """
    Prepares training and validation DataLoaders from a single combined DataFrame
    for a Seq2Seq task. Creates all sequences, shuffles them, then splits.
    Handles feature engineering and scaling based on config.

    Args:
        df_all (pd.DataFrame): DataFrame containing all combined simulation data.
        sequence_length (int): Length of input sequences (N).
        output_sequence_length (int): Length of output sequences (K).
        val_fraction (float): Proportion of total sequences for the validation set.
        seed (int): Random seed for shuffling and splitting.

    Returns:
        tuple: (train_loader, val_loader, input_scaler, target_scaler)
               Returns (None, None, None, None) if processing fails.
    """
    print(f"Preparing shuffled train/validation data (Val fraction: {val_fraction*100:.1f}%)...")
    required_cols = ['theta', 'theta_dot', 'tau']

    if df_all is None or df_all.empty or not all(col in df_all.columns for col in required_cols): print("Error: Input DataFrame invalid."); return (None,) * 4

    use_sincos = config.USE_SINCOS_THETA
    print(f"Using sin/cos features: {use_sincos}, Predicting absolute state sequence.")

    print("Creating all sequences from combined data...")
    data_values = df_all[required_cols].values
    # Call create_sequences for Seq2Seq (no predict_delta argument anymore)
    X_all, y_all = create_sequences(data_values, sequence_length, output_sequence_length, use_sincos=use_sincos)

    if X_all.shape[0] == 0 or y_all.shape[0] == 0: print("Error: No sequences created."); return (None,) * 4
    print(f"Total sequences created: {len(X_all)}")

    print("Shuffling and splitting sequences into train/validation sets...")
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=val_fraction, random_state=seed, shuffle=True)
    except ValueError as e: print(f"Error splitting data: {e}"); return (None,) * 4

    print(f"Train sequences: {len(X_train)}, Validation sequences: {len(X_val)}")
    if len(X_train) == 0: print("Error: Training set empty."); return (None,) * 4
    if len(X_val) == 0: print("Warning: Validation set empty.")

    # --- Scaling ---
    num_input_features = X_train.shape[2]; num_output_features = y_train.shape[2]
    X_train_reshaped = X_train.reshape(-1, num_input_features)
    X_val_reshaped = X_val.reshape(-1, num_input_features) if X_val.shape[0] > 0 else np.empty((0, num_input_features))
    y_train_reshaped = y_train.reshape(-1, num_output_features)
    y_val_reshaped = y_val.reshape(-1, num_output_features) if y_val.shape[0] > 0 else np.empty((0, num_output_features))

    # Initialize scalers based on config
    if config.INPUT_SCALER_TYPE.lower() == "standardscaler": input_scaler = StandardScaler()
    else: input_scaler = MinMaxScaler(feature_range=(-1, 1))
    print(f"Using {config.INPUT_SCALER_TYPE} for input features (X).")

    # For Seq2Seq absolute state prediction, use TARGET_SCALER_TYPE from config
    if config.TARGET_SCALER_TYPE == "StandardScaler": target_scaler = StandardScaler()
    else: target_scaler = MinMaxScaler(feature_range=(-1, 1))
    print(f"Using {config.TARGET_SCALER_TYPE} for target variable (y - absolute state sequence).")

    # Fit scalers ONLY on training data
    try:
        print("Fitting scalers on training data..."); start_fit_time = time.time()
        X_train_scaled_reshaped = input_scaler.fit_transform(X_train_reshaped)
        y_train_scaled_reshaped = target_scaler.fit_transform(y_train_reshaped)
        print(f"Scaler fitting took {time.time()-start_fit_time:.2f}s")
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        joblib.dump(input_scaler, config.INPUT_SCALER_PATH); joblib.dump(target_scaler, config.TARGET_SCALER_PATH)
        print(f"Input scaler saved to {config.INPUT_SCALER_PATH}"); print(f"Target scaler saved to {config.TARGET_SCALER_PATH}")
    except Exception as e: print(f"Error fitting or saving scalers: {e}"); return (None,) * 4

    # Scale validation data
    if X_val_reshaped.shape[0] > 0:
         try:
            X_val_scaled_reshaped = input_scaler.transform(X_val_reshaped)
            y_val_scaled_reshaped = target_scaler.transform(y_val_reshaped)
         except Exception as e: print(f"Error scaling validation data: {e}"); return (None, None, input_scaler, target_scaler)
    else: y_val_scaled_reshaped = np.empty((0, num_output_features))

    # Reshape back to original sequence shapes
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape) if X_val.shape[0] > 0 else np.empty((0, sequence_length, num_input_features))
    y_train_scaled = y_train_scaled_reshaped.reshape(y_train.shape)
    y_val_scaled = y_val_scaled_reshaped.reshape(y_val.shape) if y_val.shape[0] > 0 else np.empty((0, output_sequence_length, num_output_features))

    # --- Create Tensors and DataLoaders ---
    try:
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32); y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32); y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor) if X_val_tensor.shape[0] > 0 else None
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=False) if val_dataset else None
        print("DataLoaders created.")
        if val_loader is None: print("Warning: Validation DataLoader is None.")
    except Exception as e: print(f"Error creating Tensors or DataLoaders: {e}"); return (None, None, input_scaler, target_scaler)

    return train_loader, val_loader, input_scaler, target_scaler


# --- Remove or comment out old/unused functions ---
# def prepare_train_val_data_from_dfs(...): ...
# def get_test_data_and_scalers(...): ... # This is now less relevant for main training flow

