# config.py

"""
Configuration settings for the Pendulum experiment.
Strategy: Random ICs, Random Torque, Short Simulations, Shuffle+Split, Seq2Seq.
"""

import numpy as np
import os

# --- Feature Engineering ---
USE_SINCOS_THETA = True # Keep using sin/cos features for input

# --- Model Selection ---
# Options: "PureLSTM", "PureGRU", "DeltaLSTM", "DeltaGRU", "Seq2SeqLSTM", "Seq2SeqGRU"
# Let's define new types for Seq2Seq models
MODEL_TYPE = "Seq2SeqLSTM" # <<<--- NEW: Using GRU for Seq2Seq initially

# --- Model Architecture Parameters ---
# Parameters for the chosen MODEL_TYPE will be used.
# Input size will be 4 if USE_SINCOS_THETA is True. Output size is 2 (theta, dot).
# Output layer will be adapted for OUTPUT_SEQ_LEN in model.py.
MODEL_PARAMS = {
    "defaults": {
        "dense_units": 32,  # Keep smaller dense units
        "dropout_rate": 0.30 # Keep moderate dropout
    },
    # Define base params, Seq2Seq might use these or override
    "purelstm": { "hidden_size": 32, "num_layers": 2 }, # Keep model small
    "puregru":  { "hidden_size": 32, "num_layers": 2 }, # Keep model small
    # Delta models are not the focus now, but keep definitions
    "deltalstm":{ "hidden_size": 32, "num_layers": 2 },
    "deltagru": { "hidden_size": 32, "num_layers": 2 },
    # Add specific params for Seq2Seq if needed, otherwise they use base params
    "seq2seqlstm":{ "hidden_size": 32, "num_layers": 2 },
    "seq2seqgru": { "hidden_size": 10, "num_layers": 1 }
}
# Function to get parameters (remains the same, handles base types)
def get_current_model_params():
    model_type_key = MODEL_TYPE.lower(); params = MODEL_PARAMS.get("defaults", {}).copy()
    base_type_key = model_type_key.replace("delta", "pure", 1).replace("seq2seq", "pure", 1) # Find base type
    specific_params = MODEL_PARAMS.get(base_type_key)
    if specific_params is None: raise ValueError(f"Base params for '{MODEL_TYPE}'/'{base_type_key}' not found.")
    params.update(specific_params)
    for key, value in MODEL_PARAMS.get("defaults", {}).items():
        if key not in params: params[key] = value
    print(f"Info: Using parameters from '{base_type_key}' definition for model '{MODEL_TYPE}'.")
    return params

# --- Training Hyperparameters ---
NUM_EPOCHS = 150 # Keep epoch count, let early stopping decide
LEARNING_RATE = 0.0005 # May need adjustment for Seq2Seq, start slightly higher than last run
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 40
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 8

# --- Simulation Parameters ---
PENDULUM_MASS = 1.0; PENDULUM_LENGTH = 1.0; GRAVITY = 9.81; DAMPING_COEFF = 0.5
DT = 0.02

# Random Initial Conditions Ranges
THETA_RANGE = [-np.pi / 2, np.pi / 2] # -90 to +90 degrees
THETA_DOT_RANGE = [-3.0, 3.0]         # Safe velocity range (rad/s)

# Torque Scenario - Use only highly random
TORQUE_TYPE = "highly_random"
TORQUE_RANGE = [-1.0, 1.0] # Range for random torque
TORQUE_CHANGE_STEPS = 10   # How often random torque changes (in steps)

# Simulation Generation Method
# Target ~24k params * 10 = 240k sequences
# Input=10, Output=5 -> Min steps=15. Use 50 steps (1.0s) -> ~36 sequences/run
# Need ~6700 simulations. Let's use 7000.
NUM_SIMULATIONS = 7000
SIMULATION_DURATION = 1.0 # Duration of each short simulation in seconds
T_SPAN_SHORT = (0, SIMULATION_DURATION)

# --- Data Handling ---
# Single combined data file
COMBINED_DATA_FILE = f'combined_data_{MODEL_TYPE.lower()}.csv'
FORCE_REGENERATE_DATA = False # Set to True to force data regeneration
MODELS_DIR = 'models'; FIGURES_DIR = 'figures'
PLOT_SCENARIO_DATA = False # No separate scenarios to plot now

# --- Preprocessing Parameters ---
INPUT_SEQ_LEN = 10  # <<<--- NEW: Input sequence length (N)
OUTPUT_SEQ_LEN = 5   # <<<--- NEW: Output sequence length (K) for Seq2Seq
VAL_SET_FRACTION = 0.2 # <<<--- NEW: Fraction for validation set after shuffling all sequences
MIN_PREDICTION_STEPS = 50
INPUT_SCALER_TYPE = "StandardScaler"
# Target Scaler: Predicting absolute states again for Seq2Seq simplicity
TARGET_SCALER_TYPE = "MinMaxScaler" # <<<--- Back to MinMaxScaler for absolute state prediction

# --- Paths ---
MODEL_BASENAME = f'pendulum_{MODEL_TYPE.lower()}'
if USE_SINCOS_THETA: MODEL_BASENAME += "_sincos"
MODEL_BEST_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_best.pth')
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_final.pth')

# Determine scaler paths based on config
if MODEL_TYPE.lower().startswith("delta"): # Keep this logic, although we set Seq2Seq to absolute
    TARGET_SCALER_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_delta_scaler.pkl')
else: # For Pure... or Seq2Seq... predicting absolute state
    TARGET_SCALER_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_output_scaler.pkl')

INPUT_SCALER_FILENAME = f'{MODEL_BASENAME}_input_scaler{"_sincos" if USE_SINCOS_THETA else ""}.pkl'
INPUT_SCALER_PATH = os.path.join(MODELS_DIR, INPUT_SCALER_FILENAME)

# --- Evaluation Parameters ---
GOOD_MSE_THRESHOLD = 0.01 # Keep threshold, might need adjustment
PHYSICS_VALIDATION_TOLERANCE = 1e-3 # <<<--- 新增：物理校验的容忍度

# --- Misc ---
SEED = 42
