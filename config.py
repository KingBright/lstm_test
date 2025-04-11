# config.py

"""
Configuration settings for the Pendulum experiment.
Strategy: Long Random Trajectories, Chronological Sequence Split, Seq2Seq.
"""

import numpy as np
import os

# --- Feature Engineering ---
USE_SINCOS_THETA = True

# --- Model Selection ---
MODEL_TYPE = "Seq2SeqGRU" # Or Seq2SeqLSTM

# --- Model Architecture Parameters ---
MODEL_PARAMS = {
    "defaults": { "dense_units": 32, "dropout_rate": 0.30 },
    "purelstm": { "hidden_size": 32, "num_layers": 2 },
    "puregru":  { "hidden_size": 32, "num_layers": 2 },
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
# 基础特定初始条件
INITIAL_CONDITIONS_SPECIFIC = [
    [0.0, 0.0], [0.3, -0.5], [-0.3, 0.5], [-1.0, -1.0], [1.0, 1.0]
]

# 优化的数据生成策略参数
# 目标序列总数约100,000
TARGET_SEQUENCES = 100000
# 角度和角速度范围参数 (用于随机初始条件)
THETA_RANGE = [-np.pi/2, np.pi/2]  # 角度范围 [-90°, 90°]
THETA_DOT_RANGE = [-2.0, 2.0]      # 角速度范围 [-2, 2] rad/s
# 随机初始条件数量 (将会根据目标序列数自动计算)
NUM_RANDOM_ICS = 500  # 默认值，会根据总序列需求自动调整

# Torque 参数设置
TORQUE_TYPE = "highly_random"
TORQUE_RANGE = [-0.7, 0.7]  # 力矩范围
# 力矩变化步长范围 (随机选择)
TORQUE_CHANGE_STEPS_RANGE = [10, 30]  # 在每次模拟时随机选择力矩变化步长

# 优化的模拟时长设置
SIMULATION_DURATION = 30.0  # 每个模拟的时长减少到30秒
T_SPAN = (0, SIMULATION_DURATION)  # 定义模拟时间段

# --- Data Handling ---
# Combined data file from optimized training data
COMBINED_DATA_FILE = f'combined_optimized_{MODEL_TYPE.lower().replace("seq2seq","")}{"_sincos" if USE_SINCOS_THETA else ""}.csv'
FORCE_REGENERATE_DATA = False
MODELS_DIR = 'models'; FIGURES_DIR = 'figures'
PLOT_SCENARIO_DATA = False # Plotting individual scenarios less relevant now

# --- Preprocessing Parameters ---
INPUT_SEQ_LEN = 10; OUTPUT_SEQ_LEN = 5
# VALIDATION_SPLIT for chronological splitting of sequences
VALIDATION_SPLIT = 0.2 # <<<--- 确认 VALIDATION_SPLIT 已定义 (用于时序分割)
MIN_PREDICTION_STEPS = 50
INPUT_SCALER_TYPE = "StandardScaler"
TARGET_SCALER_TYPE = "MinMaxScaler" # For Seq2Seq absolute state

# --- Paths ---
MODEL_BASENAME = f'pendulum_{MODEL_TYPE.lower()}'
if USE_SINCOS_THETA: MODEL_BASENAME += "_sincos"
MODEL_BASENAME += "_optimized" # 更改标识符以反映优化的数据生成策略
MODEL_BEST_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_best.pth')
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_final.pth')

if MODEL_TYPE.lower().startswith("delta"): TARGET_SCALER_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_delta_scaler.pkl')
else: TARGET_SCALER_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_output_scaler.pkl')
INPUT_SCALER_FILENAME = f'{MODEL_BASENAME}_input_scaler.pkl'
INPUT_SCALER_PATH = os.path.join(MODELS_DIR, INPUT_SCALER_FILENAME)

# --- Evaluation Parameters ---
GOOD_MSE_THRESHOLD = 0.01
# PHYSICS_VALIDATION_TOLERANCE removed

# --- Misc ---
SEED = 42
