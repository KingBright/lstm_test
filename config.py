# config.py

"""
Configuration settings for the Pendulum LSTM experiment.
"""

import numpy as np

# --- Simulation Parameters ---
PENDULUM_MASS = 1.0       # kg
PENDULUM_LENGTH = 1.0     # m
GRAVITY = 9.81            # m/s^2
DAMPING_COEFF = 0.5       # N·m·s/rad
DT = 0.02                 # s (Simulation time step)
T_SPAN = (0, 100)         # (start_time, end_time) in seconds for simulation
# Define multiple initial conditions for diverse data
INITIAL_CONDITIONS = [
    [0.1, 0.0],
    [-0.2, 0.1],
    [0.5, -0.2],
    [1.0, 0.0],
    [-1.0, 0.5]
]
TORQUE_TYPE = "mixed"     # Type of torque sequence for simulation

# --- Data Handling ---
DATA_FILE_EXTENDED = 'simulation_data_extended.csv' # File to save/load generated data
FORCE_REGENERATE_DATA = False # Set to True to force data regeneration even if file exists
MODELS_DIR = 'models'     # Directory to save models and scalers
FIGURES_DIR = 'figures'   # Directory to save plots
PLOT_SAMPLE_DATA = True   # Whether to plot a sample of the generated data

# --- Preprocessing Parameters ---
SEQUENCE_LENGTH = 20      # Number of time steps in each input sequence
TEST_SPLIT = 0.2          # Proportion of data to use for the test set (0.0 to 1.0)

# --- Model Hyperparameters ---
HIDDEN_SIZE = 128         # Number of units in LSTM hidden layers
NUM_LAYERS = 3            # Number of LSTM layers
DENSE_UNITS = 96          # Number of units in the intermediate dense layer
DROPOUT_RATE = 0.25       # Dropout probability

# --- Training Hyperparameters ---
NUM_EPOCHS = 150          # Maximum number of training epochs
LEARNING_RATE = 0.0005    # Initial learning rate (Lowered based on previous results)
WEIGHT_DECAY = 1e-5       # L2 regularization strength
BATCH_SIZE = 128          # Number of samples per training batch (defined in preprocessing/training)
EARLY_STOPPING_PATIENCE = 15 # Number of epochs to wait for improvement before stopping
SCHEDULER_FACTOR = 0.4    # Factor by which the learning rate will be reduced
SCHEDULER_PATIENCE = 7    # Number of epochs with no improvement after which learning rate will be reduced

# --- Evaluation Parameters ---
MIN_PREDICTION_STEPS = 50 # Minimum number of steps required to run multi-step prediction plot

# --- Paths ---
MODEL_BEST_PATH = f'{MODELS_DIR}/pendulum_lstm_best.pth'
MODEL_FINAL_PATH = f'{MODELS_DIR}/pendulum_lstm_final.pth'
INPUT_SCALER_PATH = f'{MODELS_DIR}/input_scaler.pkl'
OUTPUT_SCALER_PATH = f'{MODELS_DIR}/output_scaler.pkl'

# --- Misc ---
SEED = 42                 # Random seed for reproducibility