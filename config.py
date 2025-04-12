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
MODEL_TYPE = "Seq2SeqLSTM" # Or Seq2SeqLSTM

# --- Model Architecture Parameters ---
MODEL_PARAMS = {
    "defaults": { "dense_units": 32, "dropout_rate": 0.20 },
    "purelstm": { "hidden_size": 32, "num_layers": 2 },
    "puregru":  { "hidden_size": 32, "num_layers": 2 },
    "seq2seqlstm":{ "hidden_size": 16, "num_layers": 3 },
    "seq2seqgru": { "hidden_size": 16, "num_layers": 3 }
}
def get_current_model_params():
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
    [0.0, 0.0], [0.3, 0.5], [0.3, -0.5], [-0.3, 0.5], [-0.3, -0.5], [-1.0, -0.25], [-1.0, 0.25], [1.0, 0.25], [1.0, -0.25]
]

# 优化的数据生成策略参数
TARGET_SEQUENCES = 100000 # 目标序列总数约1,000,000
THETA_RANGE = [-np.pi/3, np.pi/3]  # 角度范围 [-60°, 60°]
THETA_DOT_RANGE = [-0.5, 0.5]      # 角速度范围 [-0.5, 0.5] rad/s
# 随机初始条件数量 (将会根据目标序列数自动计算)
# NUM_RANDOM_ICS = 5000 # 不再需要，由 simulations_needed 决定

# Torque 参数设置
TORQUE_TYPE = "highly_random" # 这个参数当前在 generate_improved_dataset 中未直接使用，因为IC中包含了tau
TORQUE_RANGE = [-0.5, 0.5]  # 力矩范围 (用于为IC添加随机tau)
# 力矩变化步长范围 (如果使用 generate_torque_sequence 会用到)
TORQUE_CHANGE_STEPS_RANGE = [10, 30]

# 优化的模拟时长设置
SIMULATION_DURATION = 30.0  # 每个模拟的时长
T_SPAN = (0, SIMULATION_DURATION)  # 定义模拟时间段

# --- Data Handling ---
# *** FIX: Added './' to specify the current directory for the output file ***
COMBINED_DATA_FILE = f'./combined_optimized_{MODEL_TYPE.lower().replace("seq2seq","")}{"_sincos" if USE_SINCOS_THETA else ""}.csv'
FORCE_REGENERATE_DATA = True # 设为 True 会强制重新生成数据
MODELS_DIR = 'models'; FIGURES_DIR = 'figures'
# 创建必要的目录 (如果它们不存在)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
# 可选：为数据文件创建一个单独的目录
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)
COMBINED_DATA_FILE = os.path.join(DATA_DIR, f'combined_optimized_{MODEL_TYPE.lower().replace("seq2seq","")}{"_sincos" if USE_SINCOS_THETA else ""}.csv')


PLOT_SCENARIO_DATA = False # Plotting individual scenarios less relevant now

# --- Preprocessing Parameters ---
INPUT_SEQ_LEN = 20; OUTPUT_SEQ_LEN = 2
VALIDATION_SPLIT = 0.2 # 用于时序分割的验证集比例
MIN_PREDICTION_STEPS = 50 # 多步预测的最小步数
INPUT_SCALER_TYPE = "StandardScaler" # 输入特征缩放器类型
TARGET_SCALER_TYPE = "MinMaxScaler" # 目标特征缩放器类型 (Seq2Seq 预测绝对值)

# --- Paths ---
MODEL_BASENAME = f'pendulum_{MODEL_TYPE.lower()}'
if USE_SINCOS_THETA: MODEL_BASENAME += "_sincos"
MODEL_BASENAME += "_optimized" # 标识符反映优化的数据生成
MODEL_BEST_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_best.pth')
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_final.pth')

# Scaler 文件路径也应包含目录
if MODEL_TYPE.lower().startswith("delta"): TARGET_SCALER_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_delta_scaler.pkl')
else: TARGET_SCALER_PATH = os.path.join(MODELS_DIR, f'{MODEL_BASENAME}_output_scaler.pkl')
INPUT_SCALER_FILENAME = f'{MODEL_BASENAME}_input_scaler.pkl'
INPUT_SCALER_PATH = os.path.join(MODELS_DIR, INPUT_SCALER_FILENAME)

# --- Evaluation Parameters ---
GOOD_MSE_THRESHOLD = 0.01

# --- Misc ---
SEED = 42
