# config.py

"""
Configuration settings for the Pendulum LSTM/GRU experiment.
Using separate generation for Train/Validation sets.
Configured for Delta Prediction with smaller model and StandardScaler for input.
"""

import numpy as np
import os

# --- Model Selection ---
# Options: "PureLSTM", "PureGRU", "DeltaLSTM", "DeltaGRU"
MODEL_TYPE = "DeltaGRU" # <<<--- Keep DeltaGRU (or change to DeltaLSTM if preferred)

# --- Model Architecture Parameters ---
# Use a dictionary to hold parameters specific to each model type
MODEL_PARAMS = {
    "defaults": {
        # Reduce dense units for smaller model
        "dense_units": 32, # Reduced from 64
        "dropout_rate": 0.30 # Can slightly adjust dropout if needed
    },
    "purelstm": {
        "hidden_size": 64, # Reduced from 96
        "num_layers": 2,   # Keep 2 layers (or try 1)
    },
    "puregru": {
        "hidden_size": 64, # Reduced from 96
        "num_layers": 2,   # Keep 2 layers (or try 1)
    },
    # Delta models use the base architecture params
    "deltalstm": {
        "hidden_size": 64,
        "num_layers": 2,
    },
    "deltagru": {
        "hidden_size": 64,
        "num_layers": 2,
    }
}

# Function to easily get parameters (remains the same)
def get_current_model_params():
    """Gets the parameter dictionary for the currently selected MODEL_TYPE."""
    model_type_key = MODEL_TYPE.lower()
    params = MODEL_PARAMS.get("defaults", {}).copy()
    base_type_key = model_type_key.replace("delta", "pure", 1) if model_type_key.startswith("delta") else model_type_key
    specific_params = MODEL_PARAMS.get(base_type_key)
    if specific_params is None: raise ValueError(f"Params for '{MODEL_TYPE}'/'{base_type_key}' not found.")
    params.update(specific_params)
    print(f"Info: Using parameters from '{base_type_key}' definition for model '{MODEL_TYPE}'.")
    # Manually add defaults if they weren't in the specific dict
    for key, value in MODEL_PARAMS.get("defaults", {}).items():
        if key not in params: params[key] = value
    return params

# --- Training Hyperparameters ---
NUM_EPOCHS = 150
LEARNING_RATE = 0.0002 # Keep lowered LR
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 8

# --- Simulation Parameters (remain the same) ---
PENDULUM_MASS = 1.0; PENDULUM_LENGTH = 1.0; GRAVITY = 9.81; DAMPING_COEFF = 0.5
DT = 0.02; T_SPAN_TRAIN = (0, 40); T_SPAN_VAL = (0, 10)
INITIAL_CONDITIONS_SPECIFIC = [
    [0.0, 0.0], [0.3, -0.5], [-0.3, 0.5], [-1.0, -1.0], [1.0, 1.0]
]
SCENARIOS = ["zero", "step", "sine", "random", "mixed"]

# --- Data Handling ---
TRAIN_DATA_FILE = 'train_data_sep.csv'
VAL_DATA_FILE = 'validation_data_sep.csv'
FORCE_REGENERATE_DATA = False
MODELS_DIR = 'models'
FIGURES_DIR = 'figures'
PLOT_SCENARIO_DATA = True # Keep plotting scenarios enabled

# --- Preprocessing Parameters ---
SEQUENCE_LENGTH = 20
MIN_PREDICTION_STEPS = 50
# Add Input Scaler Type configuration
INPUT_SCALER_TYPE = "StandardScaler" # <<<--- CHANGE TO StandardScaler (options: "MinMaxScaler", "StandardScaler")

# --- Paths ---
MODEL_BASENAME = f'pendulum_{MODEL_TYPE.lower()}'
MODEL_BEST_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_best.pth')
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_final.pth')

# Target Scaler paths and type determination remains the same
if MODEL_TYPE.lower().startswith("delta"):
    TARGET_SCALER_PATH = os.path.join(MODELS_DIR, 'delta_scaler.pkl')
    TARGET_SCALER_TYPE = "StandardScaler"
else:
    TARGET_SCALER_PATH = os.path.join(MODELS_DIR, 'output_scaler.pkl')
    TARGET_SCALER_TYPE = "MinMaxScaler"
print(f"Config: Set TARGET_SCALER_PATH to {TARGET_SCALER_PATH}")
print(f"Config: Set TARGET_SCALER_TYPE to {TARGET_SCALER_TYPE}")

# Input scaler path remains the same, but the type of scaler saved will change
INPUT_SCALER_PATH = os.path.join(MODELS_DIR, 'input_scaler.pkl')
print(f"Config: Set INPUT_SCALER_TYPE to {INPUT_SCALER_TYPE}")

# --- Misc ---
SEED = 42
