# config.py

"""
Configuration settings for the Pendulum LSTM/GRU experiment.
Using separate generation for Train and Validation sets.
"""

import numpy as np
import os

# --- Model Selection ---
# Options: "PureLSTM", "PureGRU", "DeltaLSTM", "DeltaGRU"
MODEL_TYPE = "DeltaGRU" # Keep DeltaGRU for now, can be changed

# --- Model Architecture Parameters ---
# Using dictionary structure
MODEL_PARAMS = {
    "defaults": { "dense_units": 64, "dropout_rate": 0.35 },
    "purelstm": { "hidden_size": 96, "num_layers": 2 },
    "puregru":  { "hidden_size": 96, "num_layers": 2 },
    "deltalstm":{ "hidden_size": 96, "num_layers": 2 },
    "deltagru": { "hidden_size": 96, "num_layers": 2 }
}
def get_current_model_params():
    model_type_key = MODEL_TYPE.lower()
    params = MODEL_PARAMS.get("defaults", {}).copy()
    base_type_key = model_type_key.replace("delta", "pure", 1) if model_type_key.startswith("delta") else model_type_key
    specific_params = MODEL_PARAMS.get(base_type_key)
    if specific_params is None: raise ValueError(f"Params for '{MODEL_TYPE}'/'{base_type_key}' not found.")
    params.update(specific_params)
    print(f"Info: Using parameters from '{base_type_key}' definition for model '{MODEL_TYPE}'.")
    return params

# --- Training Hyperparameters ---
NUM_EPOCHS = 150
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 8

# --- Simulation Parameters ---
PENDULUM_MASS = 1.0
PENDULUM_LENGTH = 1.0
GRAVITY = 9.81
DAMPING_COEFF = 0.5
DT = 0.02

# Define specific Initial Conditions (ICs)
INITIAL_CONDITIONS_SPECIFIC = [
    [0.0, 0.0],         # 1. Static (or near static)
    [0.3, -0.5],        # 2. Low Amplitude, Falling Right
    [-0.3, 0.5],        # 3. Low Amplitude, Rising Left
    [-1.0, -1.0],       # 4. High Amplitude, Falling Left
    [1.0, 1.0]          # 5. High Amplitude, Rising Right
]
# Define Torque Scenarios
SCENARIOS = ["zero", "step", "sine", "random", "mixed"]

# Define different simulation durations for train and validation
T_SPAN_TRAIN = (0, 40)  # Longer duration for training data (e.g., 40 seconds per segment)
T_SPAN_VAL = (0, 10)    # Shorter duration for validation data (e.g., 10 seconds per segment)

# --- Data Handling ---
# Define separate file paths for train and validation data
TRAIN_DATA_FILE = 'train_data_sep.csv'
VAL_DATA_FILE = 'validation_data_sep.csv'
# DATA_FILE_EXTENDED is no longer the primary source for training/validation split
FORCE_REGENERATE_DATA = False # Still useful to control regeneration
MODELS_DIR = 'models'
FIGURES_DIR = 'figures'
PLOT_SCENARIO_DATA = True # Flag to plot individual generated scenarios

# --- Preprocessing Parameters ---
SEQUENCE_LENGTH = 20
# VALIDATION_SPLIT is no longer needed as we generate sets separately
MIN_PREDICTION_STEPS = 50

# --- Paths ---
MODEL_BASENAME = f'pendulum_{MODEL_TYPE.lower()}'
MODEL_BEST_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_best.pth')
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_final.pth')

# Scaler paths and type determination remains the same
if MODEL_TYPE.lower().startswith("delta"):
    TARGET_SCALER_PATH = os.path.join(MODELS_DIR, 'delta_scaler.pkl')
    TARGET_SCALER_TYPE = "StandardScaler"
else:
    TARGET_SCALER_PATH = os.path.join(MODELS_DIR, 'output_scaler.pkl')
    TARGET_SCALER_TYPE = "MinMaxScaler"
INPUT_SCALER_PATH = os.path.join(MODELS_DIR, 'input_scaler.pkl')

# --- Misc ---
SEED = 42