# config.py

"""
Configuration settings for the Pendulum experiment.
SANITY CHECK: N=5, K=1. No Damping, No Torque. Use Raw Theta Input.
Predict Absolute States. StandardScaler Target. Mixed Durations. Density Downsampling.
Increased Complexity. Disabled Early Stopping.
"""

import numpy as np
import os

# --- Strategy ---
PREDICT_DELTA = False
PREDICT_SINCOS_OUTPUT = False
USE_DOWNSAMPLING = True
OVERSAMPLING_FACTOR = 3.0
DOWNSAMPLING_GRID_SIZE = 30
DOWNSAMPLING_TARGET_PERCENTILE = 25

# --- Feature Engineering ---
# *** CHANGED: Disable Sin/Cos input features ***
USE_SINCOS_THETA = False

# --- Model Selection ---
MODEL_TYPE = "Seq2SeqLSTM"

# --- Model Architecture Parameters ---
MODEL_PARAMS = {
    "defaults": { "dense_units": 64, "dropout_rate": 0.20 },
    "purelstm": { "hidden_size": 32, "num_layers": 2 },
    "puregru":  { "hidden_size": 32, "num_layers": 2 },
    "seq2seqlstm":{ "hidden_size": 96, "num_layers": 3 }, # Keep increased complexity
    "seq2seqgru": { "hidden_size": 96, "num_layers": 3 }
}
def get_current_model_params():
    """Gets parameters for the currently selected MODEL_TYPE."""
    model_type_key = MODEL_TYPE.lower(); params = MODEL_PARAMS.get("defaults", {}).copy()
    specific_params = MODEL_PARAMS.get(model_type_key)
    if specific_params is None: raise ValueError(f"Params for '{MODEL_TYPE}' (key: '{model_type_key}') not found.")
    params.update(specific_params)
    for key, value in MODEL_PARAMS.get("defaults", {}).items():
        if key not in params: params[key] = value
    # print(f"Info: Using final parameters: {params}") # Optional debug
    return params

# --- Training Hyperparameters ---
NUM_EPOCHS = 150
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = NUM_EPOCHS # Keep early stopping disabled
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 8

# --- Simulation Parameters (Sanity Check Active) ---
PENDULUM_MASS = 1.0; PENDULUM_LENGTH = 1.0; GRAVITY = 9.81
DAMPING_COEFF = 0.0
TORQUE_RANGE = [0.0, 0.0]
DT = 0.02
INITIAL_CONDITIONS_SPECIFIC = [
    [0.0, 0.0], [0.3, 0.5], [0.3, -0.5], [-0.3, 0.5], [-0.3, -0.5],
    [np.pi/4, 0.0], [-np.pi/4, 0.0], [np.pi/2, 0.0], [-np.pi/2, 0.0],
    [0.0, 1.0], [0.0, -1.0], [0.0, 2.0], [0.0, -2.0]
]
TARGET_SEQUENCES = 100000
THETA_RANGE = [-np.pi/1.5, np.pi/1.5]
THETA_DOT_RANGE = [-3.0, 3.0]
TORQUE_TYPE = "zero"
TORQUE_CHANGE_STEPS_RANGE = [10, 30]
SIMULATION_DURATION_SHORT = 10.0
SIMULATION_DURATION_MEDIUM = 30.0
SIMULATION_DURATION_LONG = 60.0

# --- Data Handling ---
DATA_STRATEGY_SUFFIX = "_mixed_duration"
if USE_DOWNSAMPLING: DATA_STRATEGY_SUFFIX += "_ds"
SANITY_SUFFIX = "_simple_nodamp_notorque"
SEQ_LEN_SUFFIX = "_N5K1" # Keep N=5, K=1 for this test

# Define OUTPUT_SUFFIX based on flags (will be "" here)
if PREDICT_SINCOS_OUTPUT: OUTPUT_SUFFIX = "_sincos_out"
elif PREDICT_DELTA: OUTPUT_SUFFIX = "_delta"
else: OUTPUT_SUFFIX = ""

# Construct DATA_SUFFIX
DATA_SUFFIX = f"_opt_{int(TARGET_SEQUENCES/1000)}k{DATA_STRATEGY_SUFFIX}{SANITY_SUFFIX}{OUTPUT_SUFFIX}{SEQ_LEN_SUFFIX}"

# *** Filenames will NOT contain '_sincos' now ***
COMBINED_DATA_FILE = f'./combined_{MODEL_TYPE.lower().replace("seq2seq","")}{"_sincos" if USE_SINCOS_THETA else ""}{DATA_SUFFIX}.csv'
FORCE_REGENERATE_DATA = True # Force regenerate data with raw theta input
MODELS_DIR = 'models'; FIGURES_DIR = 'figures'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Preprocessing Parameters ---
INPUT_SEQ_LEN = 5
OUTPUT_SEQ_LEN = 1
VALIDATION_SPLIT = 0.2
MIN_PREDICTION_STEPS = 50
INPUT_SCALER_TYPE = "StandardScaler"
TARGET_SCALER_TYPE = "StandardScaler"

# --- Paths ---
MODEL_BASENAME_PREFIX = f'pendulum_{MODEL_TYPE.lower()}'
# *** _sincos will NOT be added to prefix now ***
if USE_SINCOS_THETA: MODEL_BASENAME_PREFIX += "_sincos"
actual_model_params = get_current_model_params()
MODEL_COMPLEXITY_SUFFIX = f"_h{actual_model_params['hidden_size']}_l{actual_model_params['num_layers']}"
MODEL_BASENAME = f"{MODEL_BASENAME_PREFIX}{MODEL_COMPLEXITY_SUFFIX}{DATA_SUFFIX}"

MODEL_BEST_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_best.pth')
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_final.pth')

# Scaler paths tied to data config (will not contain _sincos)
SCALER_BASENAME = f'pendulum_{MODEL_TYPE.lower().replace("seq2seq","")}{"_sincos" if USE_SINCOS_THETA else ""}{DATA_SUFFIX}'
TARGET_SCALER_PATH = os.path.join(MODELS_DIR, f'{SCALER_BASENAME}_output_scaler.pkl')
INPUT_SCALER_FILENAME = f'{SCALER_BASENAME}_input_scaler.pkl'
INPUT_SCALER_PATH = os.path.join(MODELS_DIR, INPUT_SCALER_FILENAME)

# --- Evaluation Parameters ---
GOOD_MSE_THRESHOLD = 0.01

# --- Misc ---
SEED = 42
