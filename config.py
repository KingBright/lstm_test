# config.py

"""
Configuration settings for the Pendulum experiment.
Strategy: Random ICs, Random Torque, Short Simulations, Shuffle+Split, Seq2Seq.
Physics validation removed.
"""

import numpy as np
import os

# --- Feature Engineering ---
USE_SINCOS_THETA = True

# --- Model Selection ---
MODEL_TYPE = "Seq2SeqGRU" # Or other types

# --- Model Architecture Parameters ---
MODEL_PARAMS = {
    "defaults": { "dense_units": 32, "dropout_rate": 0.30 },
    "purelstm": { "hidden_size": 32, "num_layers": 2 },
    "puregru":  { "hidden_size": 32, "num_layers": 2 },
    "deltalstm":{ "hidden_size": 32, "num_layers": 2 },
    "deltagru": { "hidden_size": 32, "num_layers": 2 },
    "seq2seqlstm":{ "hidden_size": 32, "num_layers": 2 },
    "seq2seqgru": { "hidden_size": 32, "num_layers": 2 }
}
def get_current_model_params():
    # ... (function remains the same) ...
    model_type_key = MODEL_TYPE.lower(); params = MODEL_PARAMS.get("defaults", {}).copy()
    base_type_key = model_type_key.replace("delta", "pure", 1).replace("seq2seq", "pure", 1)
    specific_params = MODEL_PARAMS.get(base_type_key)
    if specific_params is None: raise ValueError(f"Params for '{MODEL_TYPE}'/'{base_type_key}' not found.")
    params.update(specific_params)
    for key, value in MODEL_PARAMS.get("defaults", {}).items():
        if key not in params: params[key] = value
    # print(f"Info: Using parameters from '{base_type_key}' definition for model '{MODEL_TYPE}'.")
    return params

# --- Training Hyperparameters ---
NUM_EPOCHS = 150; LEARNING_RATE = 0.0005; WEIGHT_DECAY = 1e-5; BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 20; SCHEDULER_FACTOR = 0.5; SCHEDULER_PATIENCE = 8

# --- Simulation Parameters ---
PENDULUM_MASS = 1.0; PENDULUM_LENGTH = 1.0; GRAVITY = 9.81; DAMPING_COEFF = 0.5
DT = 0.02
THETA_RANGE = [-np.pi / 2, np.pi / 2]; THETA_DOT_RANGE = [-3.0, 3.0]
TORQUE_TYPE = "highly_random"; TORQUE_RANGE = [-1.0, 1.0]; TORQUE_CHANGE_STEPS = 10
NUM_SIMULATIONS = 7000; SIMULATION_DURATION = 1.0; T_SPAN_SHORT = (0, SIMULATION_DURATION)

# --- Data Handling ---
COMBINED_DATA_FILE = f'combined_data_{MODEL_TYPE.lower().replace("seq2seq","")}{"_sincos" if USE_SINCOS_THETA else ""}.csv'
FORCE_REGENERATE_DATA = False
MODELS_DIR = 'models'; FIGURES_DIR = 'figures'
PLOT_SCENARIO_DATA = False

# --- Preprocessing Parameters ---
INPUT_SEQ_LEN = 10; OUTPUT_SEQ_LEN = 3
VAL_SET_FRACTION = 0.2
MIN_PREDICTION_STEPS = 50
INPUT_SCALER_TYPE = "StandardScaler"
TARGET_SCALER_TYPE = "MinMaxScaler" # For Seq2Seq absolute state

# --- Paths ---
MODEL_BASENAME = f'pendulum_{MODEL_TYPE.lower()}'
if USE_SINCOS_THETA: MODEL_BASENAME += "_sincos"
MODEL_BEST_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_best.pth')
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_final.pth')
if MODEL_TYPE.lower().startswith("delta"): TARGET_SCALER_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_delta_scaler.pkl')
else: TARGET_SCALER_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_output_scaler.pkl')
INPUT_SCALER_FILENAME = f'{MODEL_BASENAME}_input_scaler.pkl'
INPUT_SCALER_PATH = os.path.join(MODELS_DIR, INPUT_SCALER_FILENAME)

# --- Evaluation Parameters ---
GOOD_MSE_THRESHOLD = 0.01
PHYSICS_VALIDATION_TOLERANCE = 0.02 # <<<--- 移除或注释掉这一行

# --- Misc ---
SEED = 42
