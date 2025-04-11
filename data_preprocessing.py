# data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import config

# create_sequences function remains the same
def create_sequences(data, input_seq_len=config.INPUT_SEQ_LEN, output_seq_len=config.OUTPUT_SEQ_LEN, use_sincos=config.USE_SINCOS_THETA):
    """
    Creates input sequences and corresponding multi-step target sequences (Seq2Seq).
    Input features can be [theta, theta_dot, tau] or [sin(theta), cos(theta), theta_dot, tau].
    Target 'y' will be absolute states for the next 'output_seq_len' steps.
    """
    X, y = [], []
    if data.shape[1] < 3: return np.empty((0, input_seq_len, data.shape[1])), np.empty((0, output_seq_len, 2))
    num_input_features = 4 if use_sincos else 3; num_output_features = 2
    total_len_needed = input_seq_len + output_seq_len
    if len(data) < total_len_needed: return np.empty((0, input_seq_len, num_input_features)), np.empty((0, output_seq_len, num_output_features))

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


# --- MODIFIED Core Function for Shuffle-then-Split Data Prep ---
def prepare_shuffled_train_val_data(list_of_dfs, # <<<--- 输入改为 DataFrame 列表
                                    sequence_length=config.INPUT_SEQ_LEN,
                                    output_sequence_length=config.OUTPUT_SEQ_LEN,
                                    val_fraction=config.VAL_SET_FRACTION,
                                    seed=config.SEED):
    """
    Prepares training and validation DataLoaders from a list of DataFrames (each from an independent simulation).
    Creates sequences WITHIN each DataFrame, combines all sequences, shuffles them, then splits.
    Handles feature engineering and scaling based on config.
    """
    print(f"Preparing shuffled train/validation data from list of DFs (Val fraction: {val_fraction*100:.1f}%)...")
    required_cols = ['theta', 'theta_dot', 'tau']
    X_all_list, y_all_list = [], [] # Store sequences from all segments

    # Determine feature/prediction modes from config
    use_sincos = config.USE_SINCOS_THETA
    predict_delta = False # Assuming Seq2Seq predicts absolute states
    print(f"Using sin/cos features: {use_sincos}, Predicting absolute state sequence.")

    # --- Step 1: Create sequences WITHIN each DataFrame segment ---
    print("Creating sequences from each simulation segment...")
    total_segments = len(list_of_dfs)
    processed_segments = 0
    for i, df_segment in enumerate(list_of_dfs):
        if df_segment is None or df_segment.empty or not all(col in df_segment.columns for col in required_cols):
            # print(f"  Warning: Skipping empty/invalid DataFrame segment {i+1}/{total_segments}.")
            continue

        data_values = df_segment[required_cols].values
        X_segment, y_segment = create_sequences(data_values, sequence_length, output_sequence_length, use_sincos=use_sincos)

        if X_segment.shape[0] > 0:
            X_all_list.append(X_segment)
            y_all_list.append(y_segment)
            processed_segments += 1
        # else:
            # print(f"  Warning: No sequences created from segment {i+1}/{total_segments}.")

    if not X_all_list: # Check if any sequences were created at all
        print("Error: No sequences created from any simulation segment."); return (None,) * 4

    # --- Step 2: Combine ALL sequences from all segments ---
    X_all = np.concatenate(X_all_list, axis=0)
    y_all = np.concatenate(y_all_list, axis=0)
    print(f"Total sequences combined from {processed_segments} segments: {len(X_all)}")

    # --- Step 3: Shuffle and Split ALL Sequences ---
    print("Shuffling and splitting all sequences into train/validation sets...")
    try:
        # Ensure enough samples for splitting
        if len(X_all) < 2 or (val_fraction > 0 and len(X_all) * val_fraction < 1) or ((1-val_fraction) > 0 and len(X_all) * (1-val_fraction) < 1):
             print("Warning: Not enough total sequences to perform train/val split. Using all data for training.")
             X_train, X_val, y_train, y_val = X_all, np.empty((0, *X_all.shape[1:])), y_all, np.empty((0, *y_all.shape[1:]))
        else:
             X_train, X_val, y_train, y_val = train_test_split(
                 X_all, y_all, test_size=val_fraction, random_state=seed, shuffle=True
             )
    except ValueError as e: print(f"Error splitting data: {e}"); return (None,) * 4

    print(f"Train sequences: {len(X_train)}, Validation sequences: {len(X_val)}")
    if len(X_train) == 0: print("Error: Training set empty."); return (None,) * 4
    if len(X_val) == 0: print("Warning: Validation set empty.")

    # --- Step 4: Scaling ---
    # (Scaling logic remains the same, fitting only on X_train_reshaped, y_train_reshaped)
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
    X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape) if X_val.shape[0] > 0 else np.empty((0, sequence_length, num_input_features))
    y_train_scaled = y_train_scaled_reshaped.reshape(y_train.shape)
    y_val_scaled = y_val_scaled_reshaped.reshape(y_val.shape) if y_val.shape[0] > 0 else np.empty((0, output_sequence_length, num_output_features))

    # --- Step 5: Create Tensors and DataLoaders ---
    # (DataLoader creation logic remains the same)
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
# def get_test_data_and_scalers(...): ...

