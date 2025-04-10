# data_preprocessing.py

import numpy as np
import pandas as pd
# Import both scalers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# train_test_split is no longer needed here as splitting happens during generation
# from sklearn.model_selection import train_test_split
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import config # Import config

# Updated create_sequences to handle delta calculation based on argument
def create_sequences(data, sequence_length, predict_delta=False):
    """
    Creates input sequences and corresponding target outputs (state or delta state).

    Args:
        data (np.array): Numpy array with columns [theta, theta_dot, tau].
        sequence_length (int): The number of time steps for each input sequence.
        predict_delta (bool): If True, target 'y' will be state difference y(t+1)-y(t).
                              If False, target 'y' will be absolute state y(t+1).

    Returns:
        tuple: (X, y) where X is the array of input sequences and y is the array of targets.
               X shape: (num_samples, sequence_length, num_features)
               y shape: (num_samples, num_output_features)
    """
    X, y = [], []
    num_features = data.shape[1]
    output_size = 2 # Predicting theta, theta_dot or their deltas
    if num_features < 3: # Need at least theta, theta_dot, tau
        print(f"Error in create_sequences: Data has only {num_features} features, expected at least 3.")
        return np.empty((0, sequence_length, num_features)), np.empty((0, output_size))

    min_len_needed = sequence_length + 1

    if len(data) < min_len_needed:
        # print(f"Warning in create_sequences: Data length ({len(data)}) < {min_len_needed} needed. Cannot create sequences.")
        return np.empty((0, sequence_length, num_features)), np.empty((0, output_size))

    for i in range(len(data) - sequence_length):
        input_seq = data[i : i + sequence_length] # Sequence from t-N+1 to t

        if predict_delta:
            # Calculate delta: y(t+1) - y(t)
            if i + sequence_length -1 < 0 or i + sequence_length >= len(data): continue
            current_state = input_seq[-1, 0:output_size]
            next_state = data[i + sequence_length, 0:output_size]
            target = next_state - current_state
        else:
            # Predict absolute state y(t+1)
            target = data[i + sequence_length, 0:output_size]

        X.append(input_seq)
        y.append(target)

    if not X: return np.empty((0, sequence_length, num_features)), np.empty((0, output_size))
    return np.array(X), np.array(y)


# --- NEW Core Function for Preparing Separately Generated Data ---
def prepare_train_val_data_from_dfs(df_train, df_val, sequence_length=config.SEQUENCE_LENGTH, seed=config.SEED):
    """
    Prepares training and validation DataLoaders from separately generated DataFrames.
    Handles sequence creation, shuffling, scaling (fit on train only), and DataLoader creation.
    Determines target type (absolute/delta) and scaler types from config.

    Args:
        df_train (pd.DataFrame): DataFrame containing the training data.
        df_val (pd.DataFrame): DataFrame containing the validation data.
        sequence_length (int): Length of input sequences.
        seed (int): Random seed for shuffling.

    Returns:
        tuple: (train_loader, val_loader, input_scaler, target_scaler)
               Returns (None, None, None, None) if processing fails.
    """
    print("Preparing train/validation data from separate DataFrames...")
    required_cols = ['theta', 'theta_dot', 'tau']

    # --- Validate Input DataFrames ---
    if df_train is None or df_train.empty or not all(col in df_train.columns for col in required_cols):
        print("Error: Training DataFrame is missing, empty, or lacks required columns."); return (None,) * 4
    if df_val is None or df_val.empty or not all(col in df_val.columns for col in required_cols):
        print("Warning: Validation DataFrame is missing, empty, or lacks required columns. Proceeding without validation loader.")
        # We can proceed to prepare training data, but val_loader will be None
        df_val = pd.DataFrame(columns=required_cols) # Create empty df to avoid errors later

    # Determine if delta prediction is needed
    predict_delta = config.MODEL_TYPE.lower().startswith("delta")
    print(f"Predicting delta state: {predict_delta}")

    # --- Create Sequences ---
    print("Creating sequences for training data...")
    X_train, y_train = create_sequences(df_train[required_cols].values, sequence_length, predict_delta)
    print("Creating sequences for validation data...")
    X_val, y_val = create_sequences(df_val[required_cols].values, sequence_length, predict_delta)

    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        print("Error: No sequences created from training data."); return (None,) * 4
    if X_val.shape[0] == 0 or y_val.shape[0] == 0:
        print("Warning: No sequences created from validation data. Validation loader will be None.")
        X_val = np.empty((0, sequence_length, X_train.shape[2]))
        y_val = np.empty((0, y_train.shape[1]))

    print(f"Total Train sequences: {len(X_train)}, Total Validation sequences: {len(X_val)}")

    # --- Shuffle Training Data ---
    indices = np.arange(len(X_train))
    np.random.seed(seed); np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]
    print("Training data shuffled.")

    # --- Scaling ---
    num_input_features = X_train.shape[2]; num_output_features = y_train.shape[1]
    X_train_reshaped = X_train.reshape(-1, num_input_features)
    X_val_reshaped = X_val.reshape(-1, num_input_features) if X_val.shape[0] > 0 else np.empty((0, num_input_features))

    # Initialize scalers based on config
    if config.INPUT_SCALER_TYPE.lower() == "standardscaler":
        input_scaler = StandardScaler()
        print("Using StandardScaler for input features (X).")
    else: # Default to MinMaxScaler
        input_scaler = MinMaxScaler(feature_range=(-1, 1))
        print("Using MinMaxScaler for input features (X).")

    if config.TARGET_SCALER_TYPE == "StandardScaler":
        target_scaler = StandardScaler()
        print(f"Using StandardScaler for target variable (y - {'delta' if predict_delta else 'state'}).")
    else: # Default to MinMaxScaler
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        print(f"Using MinMaxScaler for target variable (y - {'delta' if predict_delta else 'state'}).")

    # Fit scalers ONLY on training data
    try:
        print("Fitting scalers on training data..."); start_fit_time = time.time()
        X_train_scaled_reshaped = input_scaler.fit_transform(X_train_reshaped)
        y_train_scaled = target_scaler.fit_transform(y_train) # Fit target scaler on y_train
        print(f"Scaler fitting took {time.time()-start_fit_time:.2f}s")

        os.makedirs(config.MODELS_DIR, exist_ok=True)
        joblib.dump(input_scaler, config.INPUT_SCALER_PATH)
        joblib.dump(target_scaler, config.TARGET_SCALER_PATH)
        print(f"Input scaler ({config.INPUT_SCALER_TYPE}) saved to {config.INPUT_SCALER_PATH}")
        print(f"Target scaler ({config.TARGET_SCALER_TYPE}) saved to {config.TARGET_SCALER_PATH}")
    except Exception as e: print(f"Error fitting or saving scalers: {e}"); return (None,) * 4

    # Scale validation data (if it exists)
    if X_val_reshaped.shape[0] > 0:
         try:
            X_val_scaled_reshaped = input_scaler.transform(X_val_reshaped)
            y_val_scaled = target_scaler.transform(y_val) # Transform validation target
         except Exception as e: print(f"Error scaling validation data: {e}"); return (None, None, input_scaler, target_scaler)
    else: y_val_scaled = np.empty((0, num_output_features))

    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape) if X_val.shape[0] > 0 else np.empty((0, sequence_length, num_input_features))

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


# --- Modified get_test_data_and_scalers ---
def get_test_data_and_scalers(data_path=config.VAL_DATA_FILE, # Default to loading VAL data
                              sequence_length=config.SEQUENCE_LENGTH,
                              input_scaler_path=config.INPUT_SCALER_PATH,
                              target_scaler_path=config.TARGET_SCALER_PATH):
    """
    Loads scalers and prepares the validation/test set data (from data_path)
    and the corresponding dataframe slice needed for multi-step prediction evaluation.
    """
    print(f"Loading scalers and preparing evaluation data from: {data_path}")
    # --- Load Scalers ---
    try:
        if not os.path.exists(input_scaler_path) or not os.path.exists(target_scaler_path):
             raise FileNotFoundError(f"Scaler file(s) not found: {input_scaler_path}, {target_scaler_path}")
        input_scaler = joblib.load(input_scaler_path)
        target_scaler = joblib.load(target_scaler_path) # Load the specified target scaler
        print(f"Scalers loaded ({input_scaler_path}, {target_scaler_path}).")
        # Basic check if scalers seem fitted (more robust checks might be needed)
        if not hasattr(input_scaler, 'scale_') and not hasattr(input_scaler, 'min_'): # Check for attributes of either scaler type
             raise ValueError("Input scaler does not appear to be fitted.")
        if not hasattr(target_scaler, 'scale_') and not hasattr(target_scaler, 'min_'):
             raise ValueError("Target scaler does not appear to be fitted.")
    except Exception as e: print(f"Error loading or validating scalers: {e}"); return None, None, None, None

    # --- Load Specified Data File ---
    try:
        df_eval = pd.read_csv(data_path)
        if df_eval.empty: raise ValueError("Evaluation data file is empty.")
    except FileNotFoundError: print(f"Error: Evaluation data file not found at {data_path}"); return None, None, input_scaler, target_scaler
    except Exception as e: print(f"Error loading evaluation data: {e}"); return None, None, input_scaler, target_scaler

    # --- Prepare Data for Evaluation ---
    required_cols = ['theta', 'theta_dot', 'tau', 'time']
    if not all(col in df_eval.columns for col in required_cols):
        print("Error: Evaluation DataFrame missing required columns."); return None, None, input_scaler, target_scaler

    # Create sequences from the loaded evaluation data (use predict_delta=False for X)
    eval_data_values = df_eval[required_cols[:3]].values
    X_eval_all, _ = create_sequences(eval_data_values, sequence_length, predict_delta=False)

    if X_eval_all.shape[0] == 0:
        print("Error: No sequences created from evaluation data file."); return None, None, input_scaler, target_scaler

    # Scale the input sequences (X) using the loaded input scaler
    try:
        X_eval_reshaped = X_eval_all.reshape(-1, X_eval_all.shape[2])
        X_eval_scaled = input_scaler.transform(X_eval_reshaped).reshape(X_eval_all.shape)
    except Exception as e: print(f"Error scaling evaluation data sequences: {e}"); return None, None, input_scaler, target_scaler

    # Prepare the dataframe slice needed for multi-step prediction ground truth
    # This slice should start *after* the first sequence ends
    start_index_in_df_for_pred = sequence_length
    if start_index_in_df_for_pred >= len(df_eval):
        print("Error: Not enough data in evaluation file to create prediction slice."); return X_eval_scaled, pd.DataFrame(), input_scaler, target_scaler
    df_for_pred = df_eval.iloc[start_index_in_df_for_pred:].reset_index(drop=True)

    print(f"Evaluation sequences shape: {X_eval_scaled.shape}")
    print(f"DataFrame slice for prediction length: {len(df_for_pred)}")

    # Return scaled sequences, the prediction df slice, and both scalers
    return X_eval_scaled, df_for_pred, input_scaler, target_scaler

