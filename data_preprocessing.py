# data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split # Only for shuffling train set
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import config # Import config

# --- Sequence Creation (Handles Absolute/Delta/SinCos Prediction) ---
def create_sequences(data, simulation_boundaries=None,
                     input_seq_len=config.INPUT_SEQ_LEN,
                     output_seq_len=config.OUTPUT_SEQ_LEN,
                     use_sincos=config.USE_SINCOS_THETA,
                     predict_delta=config.PREDICT_DELTA,
                     predict_sincos_output=config.PREDICT_SINCOS_OUTPUT):
    """
    Creates input sequences and corresponding target sequences.
    Targets 'y' can be [theta, dot], [delta_theta, delta_dot], or [sin, cos, dot].
    """
    X, y = [], []
    required_cols = 3
    if predict_sincos_output: output_cols = 3
    elif predict_delta: output_cols = 2
    else: output_cols = 2

    if data.shape[1] < required_cols: print(f"Error: Data needs at least {required_cols} features."); return np.empty((0,)), np.empty((0,))
    num_input_features = 4 if use_sincos else 3
    total_len_needed = input_seq_len + output_seq_len

    if len(data) < total_len_needed: print(f"Warning: Data length ({len(data)}) < total steps needed ({total_len_needed})."); return np.empty((0,)), np.empty((0,))

    if simulation_boundaries is None:
        print("Warning: No simulation boundaries provided. Treating data as single sequence.")
        simulation_boundaries = [0, len(data)]
    else:
        if not simulation_boundaries or simulation_boundaries[0] != 0: simulation_boundaries = [0] + (simulation_boundaries if simulation_boundaries else [])
        if simulation_boundaries[-1] != len(data): simulation_boundaries.append(len(data))
    simulation_boundaries = sorted(list(set(simulation_boundaries)))

    # print(f"Creating sequences (Delta: {predict_delta}, SinCosOut: {predict_sincos_output})...") # Less verbose
    sequences_created = 0
    for sim_idx in range(len(simulation_boundaries) - 1):
        start_idx = simulation_boundaries[sim_idx]; end_idx = simulation_boundaries[sim_idx + 1]
        segment_len = end_idx - start_idx
        if segment_len < total_len_needed: continue

        for i in range(start_idx, end_idx - total_len_needed + 1):
            input_window = data[i : i + input_seq_len]
            target_window_abs = data[i + input_seq_len : i + input_seq_len + output_seq_len]

            theta_in = input_window[:, 0]; theta_dot_in = input_window[:, 1]; tau_in = input_window[:, 2]
            if use_sincos: input_seq = np.stack([np.sin(theta_in), np.cos(theta_in), theta_dot_in, tau_in], axis=1)
            else: input_seq = np.stack([theta_in, theta_dot_in, tau_in], axis=1)

            if predict_sincos_output:
                theta_target = target_window_abs[:, 0]; theta_dot_target = target_window_abs[:, 1]
                target_seq = np.stack([np.sin(theta_target), np.cos(theta_target), theta_dot_target], axis=1)
            elif predict_delta:
                last_input_state = input_window[-1, 0:2]; states_for_diff = np.vstack([last_input_state, target_window_abs[:, 0:2]])
                target_seq = np.diff(states_for_diff, axis=0)
            else: target_seq = target_window_abs[:, 0:2]

            X.append(input_seq); y.append(target_seq); sequences_created += 1

    # print(f"Total sequences created: {sequences_created}") # Less verbose
    if not X: return np.empty((0, input_seq_len, num_input_features)), np.empty((0, output_seq_len, output_cols))
    return np.array(X), np.array(y)


# --- Data Preparation (ADDED TARGET SCALED STATS PRINT) ---
def prepare_timesplit_seq2seq_data(df_all, simulation_boundaries=None,
                                   input_sequence_length=config.INPUT_SEQ_LEN,
                                   output_sequence_length=config.OUTPUT_SEQ_LEN,
                                   val_split_ratio=config.VALIDATION_SPLIT,
                                   seed=config.SEED):
    """
    Prepares training and validation DataLoaders using a chronological split.
    Handles scaling and prints statistics of scaled targets.
    """
    print(f"Preparing time-split data (Val split: {val_split_ratio*100:.1f}%, Predict Delta: {config.PREDICT_DELTA}, Predict SinCos: {config.PREDICT_SINCOS_OUTPUT})...")
    required_cols = ['theta', 'theta_dot', 'tau']
    if df_all is None or df_all.empty or not all(col in df_all.columns for col in required_cols):
        print("Error: Input DataFrame invalid."); return (None,) * 4

    print("Creating all sequences from combined data...")
    data_values = df_all[required_cols].values
    X_all, y_all = create_sequences(
        data_values, simulation_boundaries=simulation_boundaries,
        input_seq_len=input_sequence_length, output_seq_len=output_sequence_length,
        use_sincos=config.USE_SINCOS_THETA, predict_delta=config.PREDICT_DELTA,
        predict_sincos_output=config.PREDICT_SINCOS_OUTPUT
    )

    if X_all.shape[0] == 0 or y_all.shape[0] == 0: print("Error: No sequences created."); return (None,) * 4
    num_output_features = y_all.shape[2]
    print(f"Total sequences created: {len(X_all)}, Output features: {num_output_features}")

    print("Performing chronological split...")
    total_sequences = len(X_all); split_index = int(total_sequences * (1 - val_split_ratio))
    if split_index <= 0 or split_index >= total_sequences:
        print(f"Warning: Chronological split resulted in empty train or validation set.")
        if split_index <= 0: print("Error: Training set empty."); return (None,) * 4
        else: X_train, X_val, y_train, y_val = X_all, np.empty((0, *X_all.shape[1:])), y_all, np.empty((0, *y_all.shape[1:])); print("Warning: Validation set empty.")
    else:
        X_train, X_val = X_all[:split_index], X_all[split_index:]
        y_train, y_val = y_all[:split_index], y_all[split_index:]
    print(f"Train sequences: {len(X_train)}, Validation sequences: {len(X_val)}")

    indices = np.arange(len(X_train)); np.random.seed(seed); np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]
    print("Training sequences shuffled.")

    num_input_features = X_train.shape[2]
    X_train_reshaped = X_train.reshape(-1, num_input_features)
    X_val_reshaped = X_val.reshape(-1, num_input_features) if X_val.shape[0] > 0 else np.empty((0, num_input_features))
    y_train_reshaped = y_train.reshape(-1, num_output_features)
    y_val_reshaped = y_val.reshape(-1, num_output_features) if y_val.shape[0] > 0 else np.empty((0, num_output_features))

    if config.INPUT_SCALER_TYPE.lower() == "standardscaler": input_scaler = StandardScaler()
    else: input_scaler = MinMaxScaler(feature_range=(-1, 1))
    if config.TARGET_SCALER_TYPE.lower() == "standardscaler": target_scaler = StandardScaler()
    else: target_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_desc = "Sin/Cos/Dot" if config.PREDICT_SINCOS_OUTPUT else ("Delta States" if config.PREDICT_DELTA else "Absolute States")
    print(f"Using {config.INPUT_SCALER_TYPE} for input (X), {config.TARGET_SCALER_TYPE} for target (y - {target_desc}).")

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
         try:
             X_val_scaled_reshaped = input_scaler.transform(X_val_reshaped)
             y_val_scaled_reshaped = target_scaler.transform(y_val_reshaped)
         except Exception as e: print(f"Error scaling validation data: {e}"); return (None, None, input_scaler, target_scaler)
    else: y_val_scaled_reshaped = np.empty((0, num_output_features))

    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape) if X_val.shape[0] > 0 else np.empty((0, input_sequence_length, num_input_features))
    y_train_scaled = y_train_scaled_reshaped.reshape(y_train.shape)
    y_val_scaled = y_val_scaled_reshaped.reshape(y_val.shape) if y_val.shape[0] > 0 else np.empty((0, output_sequence_length, num_output_features))

    # *** ADDED: Print statistics of scaled targets ***
    print("\n--- Scaled Target Statistics ---")
    if len(y_train_scaled) > 0:
        print(f"Scaled Training Targets (y_train_scaled) Shape: {y_train_scaled.shape}")
        # Reshape to 2D for easier stats calculation per feature
        y_train_scaled_2d = y_train_scaled.reshape(-1, num_output_features)
        print(f"  Min per feature : {np.min(y_train_scaled_2d, axis=0)}")
        print(f"  Max per feature : {np.max(y_train_scaled_2d, axis=0)}")
        print(f"  Mean per feature: {np.mean(y_train_scaled_2d, axis=0)}")
        print(f"  Std per feature : {np.std(y_train_scaled_2d, axis=0)}")
    else:
        print("Scaled Training Targets: Empty")

    if len(y_val_scaled) > 0:
        print(f"Scaled Validation Targets (y_val_scaled) Shape: {y_val_scaled.shape}")
        y_val_scaled_2d = y_val_scaled.reshape(-1, num_output_features)
        print(f"  Min per feature : {np.min(y_val_scaled_2d, axis=0)}")
        print(f"  Max per feature : {np.max(y_val_scaled_2d, axis=0)}")
        print(f"  Mean per feature: {np.mean(y_val_scaled_2d, axis=0)}")
        print(f"  Std per feature : {np.std(y_val_scaled_2d, axis=0)}")
    else:
        print("Scaled Validation Targets: Empty")
    print("---------------------------------\n")
    # *** END OF ADDED PRINT ***

    # Create Tensors and DataLoaders
    try:
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor) if X_val_tensor.shape[0] > 0 else None

        import multiprocessing
        num_workers = min(4, max(2, multiprocessing.cpu_count() // 2))
        # print(f"Using {num_workers} worker processes for DataLoaders.") # Less verbose

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=2) if val_dataset else None
        print(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}")
        if val_loader is None: print("Warning: Validation DataLoader is None.")

    except Exception as e: print(f"Error creating Tensors or DataLoaders: {e}"); return (None, None, input_scaler, target_scaler)

    return train_loader, val_loader, input_scaler, target_scaler


# --- Function to load test data ---
# (No changes needed here)
def get_test_data_and_scalers(data_path=config.COMBINED_DATA_FILE, simulation_boundaries=None,
                              sequence_length=config.INPUT_SEQ_LEN, output_sequence_length=config.OUTPUT_SEQ_LEN,
                              val_split_ratio=config.VALIDATION_SPLIT, input_scaler_path=config.INPUT_SCALER_PATH,
                              target_scaler_path=config.TARGET_SCALER_PATH):
    """Loads scalers and prepares validation/test sequences and future dataframe slice."""
    print(f"Loading scalers and preparing evaluation data from: {data_path}")
    print(f"  Input Scaler: {input_scaler_path}")
    print(f"  Target Scaler: {target_scaler_path} (Type: {config.TARGET_SCALER_TYPE}, Predict SinCos: {config.PREDICT_SINCOS_OUTPUT})")

    try:
        if not os.path.exists(input_scaler_path) or not os.path.exists(target_scaler_path): raise FileNotFoundError(f"Scaler file(s) not found")
        input_scaler = joblib.load(input_scaler_path); target_scaler = joblib.load(target_scaler_path)
        print(f"Scalers loaded successfully.")
        if not hasattr(input_scaler, 'scale_') and not hasattr(input_scaler, 'min_'): raise ValueError("Input scaler not fitted.")
        if not hasattr(target_scaler, 'scale_'): raise ValueError("Target scaler does not appear to be fitted.")
    except Exception as e: print(f"Error loading scalers: {e}"); return None, None, None, None

    try:
        df_all = pd.read_csv(data_path)
        if df_all.empty: raise ValueError("Data file is empty.")
        boundaries_file = data_path.replace('.csv', '_boundaries.npy')
        if simulation_boundaries is None and not config.USE_DOWNSAMPLING and os.path.exists(boundaries_file):
             try: simulation_boundaries = list(np.load(boundaries_file)); print(f"Loaded boundaries")
             except Exception as be: print(f"Warning: Error loading boundaries file: {be}"); simulation_boundaries = None
        elif config.USE_DOWNSAMPLING: simulation_boundaries = None; print("Info: Downsampling used, no boundaries loaded.")
    except Exception as e: print(f"Error loading data: {e}"); return None, None, input_scaler, target_scaler

    required_cols = ['theta', 'theta_dot', 'tau', 'time']
    if not all(col in df_all.columns for col in required_cols): print("Error: Missing required columns."); return None, None, input_scaler, target_scaler

    use_sincos = config.USE_SINCOS_THETA
    predict_delta_eval = config.PREDICT_DELTA
    predict_sincos_eval = config.PREDICT_SINCOS_OUTPUT
    data_values = df_all[required_cols[:3]].values
    print(f"Creating sequences for validation split (Delta: {predict_delta_eval}, SinCosOut: {predict_sincos_eval})...")
    X_all, _ = create_sequences(
        data_values, simulation_boundaries=simulation_boundaries, input_seq_len=sequence_length,
        output_seq_len=output_sequence_length, use_sincos=use_sincos,
        predict_delta=predict_delta_eval, predict_sincos_output=predict_sincos_eval
    )

    if X_all.shape[0] == 0: print("Error: No sequences created."); return None, None, input_scaler, target_scaler

    total_sequences = len(X_all); split_index = int(total_sequences * (1 - val_split_ratio))
    if split_index >= total_sequences: print("Warning: Validation split empty."); return np.empty((0,)), pd.DataFrame(), input_scaler, target_scaler
    X_val = X_all[split_index:]
    if X_val.shape[0] == 0: print("Warning: X_val empty."); return np.empty((0,)), pd.DataFrame(), input_scaler, target_scaler

    try:
        if input_scaler.n_features_in_ != X_val.shape[2]: raise ValueError(f"Input scaler feature mismatch.")
        X_val_scaled = input_scaler.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)
    except Exception as e: print(f"Error scaling validation data: {e}"); return None, None, input_scaler, target_scaler

    start_index_in_df_for_pred = split_index + sequence_length
    if start_index_in_df_for_pred >= len(df_all): print("Error: Cannot get DataFrame slice for prediction."); return X_val_scaled, pd.DataFrame(), input_scaler, target_scaler
    df_for_pred = df_all.iloc[start_index_in_df_for_pred:].reset_index(drop=True)

    print(f"Validation sequences shape for evaluation: {X_val_scaled.shape}")
    print(f"DataFrame slice for prediction length: {len(df_for_pred)}")

    return X_val_scaled, df_for_pred, input_scaler, target_scaler

