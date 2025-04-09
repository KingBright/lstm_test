# data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

# Import necessary configurations
import config

def create_sequences(data, sequence_length):
    """
    Creates input sequences and corresponding target outputs from time series data.

    Args:
        data (np.array): Numpy array with columns [feature1, feature2, ..., target1, target2, ...]
                         Assumes target columns are the first 'output_size' columns.
        sequence_length (int): The number of time steps for each input sequence.

    Returns:
        tuple: (X, y) where X is the array of input sequences and y is the array of targets.
               X shape: (num_samples, sequence_length, num_features)
               y shape: (num_samples, num_output_features) - assumes output is first 2 cols
    """
    X, y = [], []
    num_features = data.shape[1]
    # Determine expected output size (assuming theta, theta_dot are the first two features)
    output_size = 2
    if num_features < output_size:
        print(f"Error: Data has only {num_features} features, expected at least {output_size} for target extraction.")
        # Return empty arrays with expected dimensions if possible
        return np.empty((0, sequence_length, num_features)), np.empty((0, output_size))

    if len(data) <= sequence_length:
        print(f"Warning: Data length ({len(data)}) is not greater than sequence length ({sequence_length}). Cannot create sequences.")
        return np.empty((0, sequence_length, num_features)), np.empty((0, output_size))

    for i in range(len(data) - sequence_length):
        input_seq = data[i : i + sequence_length]  # Input is the full sequence
        target = data[i + sequence_length, 0:output_size] # Target is next state's theta, theta_dot
        X.append(input_seq)
        y.append(target)

    if not X: # If loop didn't run or data was unsuitable
        return np.empty((0, sequence_length, num_features)), np.empty((0, output_size))

    return np.array(X), np.array(y)


def prepare_data_for_training(df, sequence_length=config.SEQUENCE_LENGTH, test_split=config.TEST_SPLIT):
    """
    Prepares the simulation data for training the LSTM model.
    Includes sequence creation, train/test splitting, scaling, and DataLoader creation.

    Args:
        df (pd.DataFrame): DataFrame containing simulation data ('theta', 'theta_dot', 'tau').
        sequence_length (int): Length of input sequences.
        test_split (float): Proportion of data for the test set.

    Returns:
        tuple: Contains scaled data arrays, tensors, DataLoaders, and scalers.
               (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
                X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
                train_loader, test_loader, input_scaler, output_scaler)
               Returns tuple of None values if processing fails.
    """
    # --- Initial Checks ---
    required_cols = ['theta', 'theta_dot', 'tau']
    if df.empty or not all(col in df.columns for col in required_cols):
        print("Error: DataFrame is empty or missing required columns. Cannot prepare data.")
        return (None,) * 12 # Return tuple of Nones matching expected output

    if not isinstance(sequence_length, int) or sequence_length <= 0:
        print("Error: sequence_length must be a positive integer.")
        return (None,) * 12

    if not (0 < test_split < 1):
         print("Error: test_split must be between 0 and 1 (exclusive).")
         return (None,) * 12

    # --- Feature Extraction and Sequence Creation ---
    data_values = df[required_cols].values
    X, y = create_sequences(data_values, sequence_length)

    if X.shape[0] == 0 or y.shape[0] == 0:
        print("Error: No sequences created. Check data length vs sequence length.")
        return (None,) * 12

    # --- Train/Test Split ---
    total_sequences = len(X)
    split_index = int(total_sequences * (1 - test_split))

    if split_index == 0 or split_index == total_sequences:
        print(f"Warning: Splitting resulted in empty train ({split_index==0}) or test ({split_index==total_sequences}) set. "
              f"Split index: {split_index}, Total sequences: {total_sequences}. Adjust test_split or data size.")
        # Allow proceeding but warn user results might be unreliable.
        # Adjust split to avoid index errors later if possible, e.g., ensure at least 1 sample if split is 0/total?
        # For now, just warn. Need robust handling based on use case.

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Train sequences: {len(X_train)}, Test sequences: {len(X_test)}")

    # --- Scaling ---
    # Reshape X for scaler (requires 2D: samples * features)
    num_input_features = X_train.shape[2]
    num_output_features = y_train.shape[1]

    # Handle potentially empty arrays after split before reshaping
    X_train_reshaped = X_train.reshape(-1, num_input_features) if X_train.shape[0] > 0 else np.empty((0, num_input_features))
    X_test_reshaped = X_test.reshape(-1, num_input_features) if X_test.shape[0] > 0 else np.empty((0, num_input_features))
    # y is already 2D (samples * features)

    input_scaler = MinMaxScaler(feature_range=(-1, 1))
    output_scaler = MinMaxScaler(feature_range=(-1, 1))

    # Fit scalers ONLY on training data (if it exists)
    if X_train_reshaped.shape[0] > 0 and y_train.shape[0] > 0:
        try:
            X_train_scaled_reshaped = input_scaler.fit_transform(X_train_reshaped)
            y_train_scaled = output_scaler.fit_transform(y_train)

            # Save the scalers
            os.makedirs(config.MODELS_DIR, exist_ok=True) # Ensure directory exists
            joblib.dump(input_scaler, config.INPUT_SCALER_PATH)
            joblib.dump(output_scaler, config.OUTPUT_SCALER_PATH)
            print(f"Scalers saved to {config.INPUT_SCALER_PATH} and {config.OUTPUT_SCALER_PATH}")

        except Exception as e:
             print(f"Error fitting or saving scalers: {e}")
             return (None,) * 12
    else:
        print("Warning: Training data is empty. Cannot fit scalers.")
        # Create empty scaled arrays to maintain structure
        X_train_scaled_reshaped = np.empty((0, num_input_features))
        y_train_scaled = np.empty((0, num_output_features))
        # Do not save unfitted scalers

    # Transform test data using fitted scalers (if test data exists and scalers are fitted)
    if X_test_reshaped.shape[0] > 0 and hasattr(input_scaler, 'scale_') and hasattr(output_scaler, 'scale_'):
         try:
            X_test_scaled_reshaped = input_scaler.transform(X_test_reshaped)
            y_test_scaled = output_scaler.transform(y_test)
         except Exception as e:
             print(f"Error transforming test data: {e}")
             return (None,) * 12
    else:
        # Create empty scaled arrays if no test data or scalers not fitted
        X_test_scaled_reshaped = np.empty((0, num_input_features))
        y_test_scaled = np.empty((0, num_output_features))

    # Reshape X back to 3D for RNN
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)

    # --- Create Tensors and DataLoaders ---
    try:
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

        # Create datasets (handle potentially empty tensors)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor) if X_train_tensor.shape[0] > 0 else TensorDataset(torch.empty(0, sequence_length, num_input_features), torch.empty(0, num_output_features))
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor) if X_test_tensor.shape[0] > 0 else TensorDataset(torch.empty(0, sequence_length, num_input_features), torch.empty(0, num_output_features))


        # Create DataLoaders
        # Shuffle only if train dataset is not empty
        shuffle_train = train_dataset.__len__() > 0
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle_train, pin_memory=True, num_workers=2, persistent_workers=(True if shuffle_train else False)) # persistent_workers require num_workers > 0
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=False)

    except Exception as e:
        print(f"Error creating Tensors or DataLoaders: {e}")
        return (None,) * 12

    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
            train_loader, test_loader, input_scaler, output_scaler)

def get_test_data_and_scalers(
    df, sequence_length=config.SEQUENCE_LENGTH, test_split=config.TEST_SPLIT,
    input_scaler_path=config.INPUT_SCALER_PATH,
    output_scaler_path=config.OUTPUT_SCALER_PATH
    ):
    """
    Loads scalers and prepares only the test set data and the corresponding
    dataframe slice needed for multi-step prediction evaluation.

    Returns:
        tuple: (X_test_scaled, df_for_pred, input_scaler, output_scaler)
               Returns (None, None, None, None) if an error occurs.
    """
    print("Loading scalers and preparing test data for evaluation...")
    # --- Load Scalers ---
    try:
        if not os.path.exists(input_scaler_path) or not os.path.exists(output_scaler_path):
             raise FileNotFoundError("Scaler file(s) not found.")
        input_scaler = joblib.load(input_scaler_path)
        output_scaler = joblib.load(output_scaler_path)
        print("Scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return None, None, None, None

    # --- Basic Data Validation ---
    required_cols = ['theta', 'theta_dot', 'tau', 'time'] # Need time for df_for_pred
    if df.empty or not all(col in df.columns for col in required_cols):
        print("Error: DataFrame is empty or missing required columns. Cannot prepare test data.")
        return None, None, input_scaler, output_scaler # Return scalers if loaded

    # --- Create Sequences ---
    data_values = df[['theta', 'theta_dot', 'tau']].values
    X, y = create_sequences(data_values, sequence_length)
    if X.shape[0] == 0:
        print("Error: No sequences created from data.")
        return None, None, input_scaler, output_scaler

    # --- Split Data (only need test part) ---
    total_sequences = len(X)
    split_index = int(total_sequences * (1 - test_split))
    if split_index == total_sequences: # Handle case where test set is empty
         print("Warning: Test split resulted in an empty test set.")
         return np.empty((0, sequence_length, X.shape[2])), pd.DataFrame(), input_scaler, output_scaler

    X_test = X[split_index:]
    # y_test = y[split_index:] # Not strictly needed for multi-step input prep

    # --- Scale Test Data ---
    try:
        num_input_features = X_test.shape[2]
        X_test_reshaped = X_test.reshape(-1, num_input_features)
        X_test_scaled_reshaped = input_scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)
    except Exception as e:
         print(f"Error scaling test data: {e}")
         return None, None, input_scaler, output_scaler


    # --- Get DataFrame Slice for Prediction ---
    # Index in original df corresponding to the start of test sequence predictions
    start_index_in_df_for_pred = split_index + sequence_length
    if start_index_in_df_for_pred >= len(df):
        print("Error: Cannot get DataFrame slice for prediction, index out of bounds.")
        return X_test_scaled, pd.DataFrame(), input_scaler, output_scaler

    df_for_pred = df.iloc[start_index_in_df_for_pred:].reset_index(drop=True)

    print(f"Test sequences shape: {X_test_scaled.shape}")
    print(f"DataFrame slice for prediction length: {len(df_for_pred)}")

    return X_test_scaled, df_for_pred, input_scaler, output_scaler