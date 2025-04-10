# config.py

"""
Configuration settings for the Pendulum LSTM/GRU experiment.
Using separate generation for Train and Validation sets.
"""

import numpy as np
import os

# --- Model Selection ---
MODEL_TYPE = "DeltaGRU" # Or "DeltaLSTM", "PureGRU", "PureLSTM"

# --- Model Architecture Parameters ---
MODEL_PARAMS = {
    "defaults": { "dense_units": 32, "dropout_rate": 0.35 },
    "purelstm": { "hidden_size": 64, "num_layers": 2 },
    "puregru":  { "hidden_size": 64, "num_layers": 2 },
    "deltalstm":{ "hidden_size": 64, "num_layers": 2 },
    "deltagru": { "hidden_size": 64, "num_layers": 2 }
}
def get_current_model_params():
    # ... (function remains the same) ...
    model_type_key = MODEL_TYPE.lower(); params = MODEL_PARAMS.get("defaults", {}).copy()
    base_type_key = model_type_key.replace("delta", "pure", 1) if model_type_key.startswith("delta") else model_type_key
    specific_params = MODEL_PARAMS.get(base_type_key)
    if specific_params is None: raise ValueError(f"Params for '{MODEL_TYPE}'/'{base_type_key}' not found.")
    params.update(specific_params)
    for key, value in MODEL_PARAMS.get("defaults", {}).items():
        if key not in params: params[key] = value
    print(f"Info: Using parameters from '{base_type_key}' definition for model '{MODEL_TYPE}'.")
    return params

# --- Training Hyperparameters ---
NUM_EPOCHS = 150; LEARNING_RATE = 0.0002; WEIGHT_DECAY = 1e-5; BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 20; SCHEDULER_FACTOR = 0.5; SCHEDULER_PATIENCE = 8

# --- Simulation Parameters ---
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
MODELS_DIR = 'models'; FIGURES_DIR = 'figures'
PLOT_SCENARIO_DATA = True

# --- Preprocessing Parameters ---
SEQUENCE_LENGTH = 20
MIN_PREDICTION_STEPS = 50
INPUT_SCALER_TYPE = "StandardScaler" # Or "MinMaxScaler"

# --- Paths ---
MODEL_BASENAME = f'pendulum_{MODEL_TYPE.lower()}'
MODEL_BEST_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_best.pth')
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_final.pth')
if MODEL_TYPE.lower().startswith("delta"):
    TARGET_SCALER_PATH = os.path.join(MODELS_DIR, 'delta_scaler.pkl')
    TARGET_SCALER_TYPE = "StandardScaler"
else:
    TARGET_SCALER_PATH = os.path.join(MODELS_DIR, 'output_scaler.pkl')
    TARGET_SCALER_TYPE = "MinMaxScaler"
# print(f"Config: Set TARGET_SCALER_PATH to {TARGET_SCALER_PATH}") # Removed repetitive print
# print(f"Config: Set TARGET_SCALER_TYPE to {TARGET_SCALER_TYPE}") # Removed repetitive print
INPUT_SCALER_PATH = os.path.join(MODELS_DIR, 'input_scaler.pkl')
# print(f"Config: Set INPUT_SCALER_TYPE to {INPUT_SCALER_TYPE}") # Removed repetitive print

# --- Evaluation Parameters ---
# Define a threshold for 'good' multi-step prediction MSE
# This is subjective and might need tuning based on expected performance
GOOD_MSE_THRESHOLD = 0.01 # <<<--- 新增：定义“好”结果的MSE阈值

# --- Misc ---
SEED = 42
