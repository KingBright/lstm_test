# data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# train_test_split is used here ONLY for shuffling the training set if needed
from sklearn.model_selection import train_test_split
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import config # Import config

# create_sequences remains the same (handles Seq2Seq output shape and SinCos features)
def create_sequences(data, input_seq_len=config.INPUT_SEQ_LEN, output_seq_len=config.OUTPUT_SEQ_LEN, use_sincos=config.USE_SINCOS_THETA):
    """
    Creates input sequences and corresponding multi-step target sequences (Seq2Seq).
    Input features can be [theta, theta_dot, tau] or [sin(theta), cos(theta), theta_dot, tau].
    Target 'y' will be absolute states for the next 'output_seq_len' steps.
    """
    X, y = [], []
    if data.shape[1] < 3: print(f"Error: Data needs at least 3 features."); return np.empty((0, input_seq_len, data.shape[1])), np.empty((0, output_seq_len, 2))
    num_input_features = 4 if use_sincos else 3; num_output_features = 2
    total_len_needed = input_seq_len + output_seq_len
    if len(data) < total_len_needed: print(f"Warning: Data length ({len(data)}) < {total_len_needed}."); return np.empty((0, input_seq_len, num_input_features)), np.empty((0, output_seq_len, num_output_features))

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


# --- NEW Core Function for Time-Split Data Prep (Seq2Seq) ---
def prepare_timesplit_seq2seq_data(df_all, input_sequence_length=config.INPUT_SEQ_LEN,
                                   output_sequence_length=config.OUTPUT_SEQ_LEN,
                                   val_split_ratio=config.VALIDATION_SPLIT,
                                   seed=config.SEED):
    """
    Prepares training and validation DataLoaders from a single combined DataFrame
    using a chronological split for train/validation sets. Handles Seq2Seq targets.

    Args:
        df_all (pd.DataFrame): DataFrame containing all combined simulation data.
        input_sequence_length (int): Length of input sequences (N).
        output_sequence_length (int): Length of output sequences (K).
        val_split_ratio (float): Proportion of total sequences for the validation set.
        seed (int): Random seed for shuffling the training set.

    Returns:
        tuple: (train_loader, val_loader, input_scaler, target_scaler)
               Returns (None, None, None, None) if processing fails.
    """
    print(f"Preparing time-split train/validation data (Val split ratio: {val_split_ratio*100:.1f}%)...")
    required_cols = ['theta', 'theta_dot', 'tau']

    if df_all is None or df_all.empty or not all(col in df_all.columns for col in required_cols):
        print("Error: Input DataFrame invalid."); return (None,) * 4

    use_sincos = config.USE_SINCOS_THETA
    predict_delta = False # Assuming Seq2Seq predicts absolute states
    print(f"Using sin/cos features: {use_sincos}, Predicting absolute state sequence.")

    # --- Create ALL Sequences ---
    print("Creating all sequences from combined data...")
    data_values = df_all[required_cols].values
    X_all, y_all = create_sequences(data_values, input_sequence_length, output_sequence_length, use_sincos=use_sincos)

    if X_all.shape[0] == 0 or y_all.shape[0] == 0:
        print("Error: No sequences created."); return (None,) * 4
    print(f"Total sequences created: {len(X_all)}")

    # --- Chronological Train/Validation Split ---
    print("Performing chronological split on sequences...")
    total_sequences = len(X_all)
    split_index = int(total_sequences * (1 - val_split_ratio))

    if split_index <= 0 or split_index >= total_sequences:
        print(f"Warning: Chronological split resulted in empty train ({split_index<=0}) or validation ({split_index>=total_sequences}) set.")
        if split_index <= 0: print("Error: Training set empty."); return (None,) * 4
        else: X_train, X_val, y_train, y_val = X_all, np.empty((0, *X_all.shape[1:])), y_all, np.empty((0, *y_all.shape[1:])); print("Warning: Validation set empty.")
    else:
        X_train, X_val = X_all[:split_index], X_all[split_index:]
        y_train, y_val = y_all[:split_index], y_all[split_index:]

    print(f"Train sequences: {len(X_train)}, Validation sequences: {len(X_val)}")

    # --- Shuffle ONLY Training Data ---
    indices = np.arange(len(X_train))
    np.random.seed(seed); np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]
    print("Training sequences shuffled.")

    # --- Scaling ---
    num_input_features = X_train.shape[2]; num_output_features = y_train.shape[2]
    X_train_reshaped = X_train.reshape(-1, num_input_features)
    X_val_reshaped = X_val.reshape(-1, num_input_features) if X_val.shape[0] > 0 else np.empty((0, num_input_features))
    y_train_reshaped = y_train.reshape(-1, num_output_features)
    y_val_reshaped = y_val.reshape(-1, num_output_features) if y_val.shape[0] > 0 else np.empty((0, num_output_features))

    if config.INPUT_SCALER_TYPE.lower() == "standardscaler": input_scaler = StandardScaler()
    else: input_scaler = MinMaxScaler(feature_range=(-1, 1))
    print(f"Using {config.INPUT_SCALER_TYPE} for input features (X).")
    if config.TARGET_SCALER_TYPE == "StandardScaler": target_scaler = StandardScaler()
    else: target_scaler = MinMaxScaler(feature_range=(-1, 1))
    print(f"Using {config.TARGET_SCALER_TYPE} for target variable (y).")

    try:
        print("Fitting scalers on training data..."); start_fit_time = time.time()
        X_train_scaled_reshaped = input_scaler.fit_transform(X_train_reshaped)
        y_train_scaled_reshaped = target_scaler.fit_transform(y_train_reshaped)
        print(f"Scaler fitting took {time.time()-start_fit_time:.2f}s")
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        joblib.dump(input_scaler, config.INPUT_SCALER_PATH); joblib.dump(target_scaler, config.TARGET_SCALER_PATH)
        print(f"Input scaler saved to {config.INPUT_SCALER_PATH}"); print(f"Target scaler saved to {config.TARGET_SCALER_PATH}")
    except Exception as e: print(f"Error fitting or saving scalers: {e}"); return (None,) * 4

    if X_val_reshaped.shape[0] > 0:
         try: X_val_scaled_reshaped = input_scaler.transform(X_val_reshaped); y_val_scaled_reshaped = target_scaler.transform(y_val_reshaped)
         except Exception as e: print(f"Error scaling validation data: {e}"); return (None, None, input_scaler, target_scaler)
    else: y_val_scaled_reshaped = np.empty((0, num_output_features))

    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape) if X_val.shape[0] > 0 else np.empty((0, input_sequence_length, num_input_features))
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


# --- Modified get_test_data_and_scalers for Chronological Split ---
def get_test_data_and_scalers(data_path=config.COMBINED_DATA_FILE, # Load the combined file
                              sequence_length=config.INPUT_SEQ_LEN,
                              output_sequence_length=config.OUTPUT_SEQ_LEN, # Needed for create_sequences
                              val_split_ratio=config.VALIDATION_SPLIT, # Use this to find the val part
                              input_scaler_path=config.INPUT_SCALER_PATH,
                              target_scaler_path=config.TARGET_SCALER_PATH):
    """
    Loads scalers and prepares the validation/test set sequences (X_val_scaled)
    and the corresponding future dataframe slice (df_for_pred) based on a
    chronological split of the data loaded from data_path.

    Args:
        data_path (str): Path to the combined dataset CSV file.
        sequence_length (int): Length of input sequences (N).
        output_sequence_length (int): Length of output sequences (K).
        val_split_ratio (float): Proportion used for validation set during training.
        input_scaler_path (str): Path to the fitted input scaler.
        target_scaler_path (str): Path to the fitted target scaler.

    Returns:
        tuple: (X_val_scaled, df_for_pred, input_scaler, target_scaler)
               Returns (None, None, None, None) if an error occurs.
    """
    print(f"Loading scalers and preparing evaluation data (time split) from: {data_path}")
    # --- Load Scalers ---
    try:
        if not os.path.exists(input_scaler_path) or not os.path.exists(target_scaler_path):
             raise FileNotFoundError(f"Scaler file(s) not found: {input_scaler_path}, {target_scaler_path}")
        input_scaler = joblib.load(input_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        print(f"Scalers loaded ({input_scaler_path}, {target_scaler_path}).")
        if not hasattr(input_scaler, 'scale_') and not hasattr(input_scaler, 'min_'): raise ValueError("Input scaler not fitted.")
        if not hasattr(target_scaler, 'scale_') and not hasattr(target_scaler, 'min_'): raise ValueError("Target scaler not fitted.")
    except Exception as e: print(f"Error loading scalers: {e}"); return None, None, None, None

    # --- Load Combined Data File ---
    try:
        df_all = pd.read_csv(data_path)
        if df_all.empty: raise ValueError("Evaluation data file is empty.")
    except FileNotFoundError: print(f"Error: Combined data file not found at {data_path}"); return None, None, input_scaler, target_scaler
    except Exception as e: print(f"Error loading combined data: {e}"); return None, None, input_scaler, target_scaler

    # --- Prepare Data for Evaluation ---
    required_cols = ['theta', 'theta_dot', 'tau', 'time']
    if not all(col in df_all.columns for col in required_cols):
        print("Error: Combined DataFrame missing required columns."); return None, None, input_scaler, target_scaler

    # Create ALL sequences from the combined data
    # Use config flag for feature engineering consistency
    use_sincos = config.USE_SINCOS_THETA
    # Use predict_delta=False because we need X sequences of absolute states
    data_values = df_all[required_cols[:3]].values
    print("Creating all sequences to determine validation split...")
    X_all, _ = create_sequences(data_values, sequence_length, output_sequence_length, use_sincos=use_sincos)

    if X_all.shape[0] == 0:
        print("Error: No sequences created from combined data file."); return None, None, input_scaler, target_scaler

    # Perform chronological split to get the validation sequences (X_val)
    total_sequences = len(X_all)
    split_index = int(total_sequences * (1 - val_split_ratio))
    if split_index >= total_sequences:
        print("Warning: Validation split is empty based on ratio and data length."); return np.empty((0, sequence_length, X_all.shape[2])), pd.DataFrame(), input_scaler, target_scaler
    X_val = X_all[split_index:]

    if X_val.shape[0] == 0:
        print("Warning: X_val is empty after chronological split."); return np.empty((0, sequence_length, X_all.shape[2])), pd.DataFrame(), input_scaler, target_scaler

    # Scale the validation input sequences (X_val) using the loaded input scaler
    try:
        # Check feature consistency
        if input_scaler.n_features_in_ != X_val.shape[2]:
             raise ValueError(f"Input scaler expects {input_scaler.n_features_in_} features, but data sequence has {X_val.shape[2]}. Check USE_SINCOS_THETA consistency.")
        X_val_reshaped = X_val.reshape(-1, X_val.shape[2])
        X_val_scaled = input_scaler.transform(X_val_reshaped).reshape(X_val.shape)
    except Exception as e: print(f"Error scaling validation data sequences: {e}"); return None, None, input_scaler, target_scaler

    # Prepare the dataframe slice needed for multi-step prediction ground truth
    # This slice corresponds to the time steps *after* the validation sequences start
    # The first validation sequence starts at index split_index in X_all
    # This sequence uses data from df_all index split_index to split_index + sequence_length - 1
    # The first prediction target corresponds to df_all index split_index + sequence_length
    start_index_in_df_for_pred = split_index + sequence_length
    if start_index_in_df_for_pred >= len(df_all):
        print("Error: Cannot get DataFrame slice for prediction (index out of bounds)."); return X_val_scaled, pd.DataFrame(), input_scaler, target_scaler
    df_for_pred = df_all.iloc[start_index_in_df_for_pred:].reset_index(drop=True)

    print(f"Validation sequences shape for evaluation: {X_val_scaled.shape}")
    print(f"DataFrame slice for prediction length: {len(df_for_pred)}")

    # Return scaled validation sequences, the prediction df slice, and both scalers
    return X_val_scaled, df_for_pred, input_scaler, target_scaler


# --- Comment out or remove old/unused functions ---
# def prepare_shuffled_train_val_data(...): ...
# def prepare_train_val_data_from_dfs(...): ...

