"""
使用神经网络模拟动力学系统：一个演示项目 (PyTorch & LSTM版)
针对Apple M2 Max优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import platform
import matplotlib as mpl
import warnings
import logging

# 抑制Matplotlib的字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# 抑制Matplotlib的字体日志
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 设置中文字体支持
def setup_chinese_font():
    system = platform.system()
    
    # 尝试设置适合当前系统的字体
    try:
        if system == 'Darwin':  # macOS
            font_list = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti']
            for font in font_list:
                try:
                    plt.rcParams['font.family'] = [font]
                    # 测试字体是否有效
                    fig = plt.figure(figsize=(1, 1))
                    plt.text(0.5, 0.5, '测试中文', ha='center', va='center')
                    plt.close(fig)
                    print(f"成功设置中文字体: {font}")
                    break
                except Exception:
                    continue
        elif system == 'Windows':
            plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei']
        elif system == 'Linux':
            plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        
        # 通用设置
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.rcParams['font.size'] = 12
    except Exception as e:
        print(f"设置中文字体时发生错误，将使用默认字体: {e}")
        # 在出错时使用默认设置
        plt.rcParams['font.family'] = ['sans-serif']
        # 设置中文标题的备用方案
        plt.rcParams['axes.titlepad'] = 20  # 增加标题间距以避免中文被截断

# 应用中文字体设置
setup_chinese_font()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# 设置环境变量，优化Metal性能
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def get_tau_at_time_global(time, tau_values, t_span_start, dt):
    """
    在任意时间点计算力矩值
    """
    # 使用t_span_start（单个浮点数）而不是尝试访问t_span[0]
    idx = int((time - t_span_start) / dt)
    if idx < 0:
        idx = 0
    elif idx >= len(tau_values):
        idx = len(tau_values) - 1
    return tau_values[idx]

# 创建目录保存模型和图表
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('figures'):
    os.makedirs('figures')

#############################################
# 步骤 1：定义动力学系统——受阻尼驱动的单摆
#############################################

class PendulumSystem:
    def __init__(self, m=1.0, L=1.0, g=9.81, c=0.5):
        """
        初始化单摆系统参数
        
        参数:
        m (float): 摆锤质量 (kg)
        L (float): 摆杆长度 (m)
        g (float): 重力加速度 (m/s²)
        c (float): 阻尼系数 (N·m·s/rad)
        """
        self.m = m
        self.L = L
        self.g = g
        self.c = c
        
        # 派生参数
        self.beta = self.c / (self.m * self.L**2)
        self.omega0_sq = self.g / self.L
        
    def ode(self, t, x, tau):
        """
        单摆的微分方程
        
        参数:
        t (float): 时间
        x (array): 状态向量 [theta, theta_dot]
        tau (float): 外部力矩
        
        返回:
        array: 状态导数 [dtheta_dt, dtheta_dot_dt]
        """
        theta, theta_dot = x
        
        # 角加速度方程
        dtheta_dt = theta_dot
        dtheta_dot_dt = (-self.beta * theta_dot - 
                          self.omega0_sq * np.sin(theta) + 
                          tau / (self.m * self.L**2))
        
        return np.array([dtheta_dt, dtheta_dot_dt])

#############################################
# 步骤 2：生成仿真数据
#############################################

def generate_torque_sequence(t, type="mixed"):
    """
    生成不同类型的力矩序列
    
    参数:
    t (array): 时间点数组
    type (str): 力矩类型 ('zero', 'step', 'sine', 'random', 'mixed')
    
    返回:
    array: 对应时间点的力矩值
    """
    if type == "zero":
        return np.zeros_like(t)
    
    elif type == "step":
        tau = np.zeros_like(t)
        step_time = t.max() * 0.3
        tau[t >= step_time] = 1.0
        return tau
    
    elif type == "sine":
        amplitude = 0.5
        frequency = 0.5  # Hz
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    elif type == "random":
        np.random.seed(42)  # 使结果可重现
        # 生成随机段
        segment_length = 20
        num_segments = len(t) // segment_length + 1
        segments = np.random.uniform(-0.8, 0.8, num_segments)
        
        # 将随机段扩展到与时间数组相同长度
        repeated_segments = np.repeat(segments, segment_length)
        return repeated_segments[:len(t)]
    
    elif type == "mixed":
        # 混合多种力矩类型
        tau = np.zeros_like(t)
        
        # 前1/4为零力矩
        quarter = len(t) // 4
        
        # 第二个1/4为阶跃力矩
        tau[quarter:2*quarter] = 0.8
        
        # 第三个1/4为正弦力矩
        sine_part = 0.5 * np.sin(2 * np.pi * 1.0 * t[2*quarter:3*quarter])
        tau[2*quarter:3*quarter] = sine_part
        
        # 最后1/4为随机力矩
        np.random.seed(42)
        segment_length = 10
        num_segments = quarter // segment_length + 1
        segments = np.random.uniform(-0.8, 0.8, num_segments)
        repeated_segments = np.repeat(segments, segment_length)
        tau[3*quarter:] = repeated_segments[:len(t)-3*quarter]
        
        return tau
    
    else:
        raise ValueError(f"未知的力矩类型: {type}")

def integrate_chunk(args):
    """
    并行集成一个时间块
    """
    pendulum, t_start, t_end, dt, x_start, tau_values, t_span_start = args
    
    # 创建时间点数组
    t_local = np.arange(t_start, t_end, dt)
    if len(t_local) == 0:
        return [], [], [], x_start
    
    # 存储结果
    time_points = []
    theta_values = []
    theta_dot_values = []
    
    # 当前状态
    x_current = np.array(x_start)
    
    # 对这个块进行积分
    for i in range(len(t_local) - 1):
        time_points.append(t_local[i])
        theta_values.append(x_current[0])
        theta_dot_values.append(x_current[1])
        
        # 获取当前时间的力矩
        current_tau = get_tau_at_time_global(t_local[i], tau_values, t_span_start, dt)
        
        # 定义当前时间步的ODE函数
        def ode_with_current_tau(t, x):
            return pendulum.ode(t, x, current_tau)
        
        # 求解从当前时间到下一时间步
        sol = solve_ivp(
            ode_with_current_tau, 
            [t_local[i], t_local[i+1]], 
            x_current, 
            method='RK45', 
            t_eval=[t_local[i+1]]
        )
        
        # 更新当前状态
        x_current = sol.y[:, -1]
    
    # 添加最后一个时间点
    if len(t_local) > 0:
        time_points.append(t_local[-1])
        theta_values.append(x_current[0])
        theta_dot_values.append(x_current[1])
    
    return time_points, theta_values, theta_dot_values, x_current

def generate_simulation_data(pendulum, t_span=(0, 10), dt=0.02, x0=None, torque_type="mixed", 
                             use_parallel=True, num_workers=None):
    """
    生成单摆仿真数据 (使用并行计算)
    
    参数:
    pendulum (PendulumSystem): 单摆系统实例
    t_span (tuple): 仿真时间范围 (start, end)
    dt (float): 时间步长
    x0 (array): 初始状态 [theta0, theta_dot0]
    torque_type (str): 力矩类型
    use_parallel (bool): 是否使用并行计算
    num_workers (int): 并行工作进程数，None表示使用CPU数量
    
    返回:
    DataFrame: 包含时间、状态和输入的数据
    """
    if x0 is None:
        x0 = [0.1, 0.0]  # 默认初始状态: 略微偏离平衡位置，无初始角速度
    
    # 创建时间点数组
    t = np.arange(t_span[0], t_span[1], dt)
    
    # 生成力矩序列
    tau_values = generate_torque_sequence(t, type=torque_type)
    
    # 注意：我们不再定义嵌套函数get_tau_at_time
    
    if use_parallel and len(t) > 100:
        # 确定CPU数量
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        
        # 分块，每个CPU处理一部分时间区间
        chunk_size = (t_span[1] - t_span[0]) / num_workers
        chunks = []
        
        current_state = np.array(x0)
        
        for i in range(num_workers):
            t_start = t_span[0] + i * chunk_size
            t_end = t_span[0] + (i+1) * chunk_size if i < num_workers-1 else t_span[1]
            
            # 修改参数，传递tau_values和t_span[0]而非函数
            chunks.append((pendulum, t_start, t_end, dt, current_state, tau_values, t_span[0]))
            
            # 使用上一块的最后状态作为下一块的初始状态
            if i < num_workers - 1:
                # 对第一个块进行积分以获取下一块的初始状态
                _, _, _, current_state = integrate_chunk(chunks[-1])
        
        # 使用进程池进行并行计算
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(integrate_chunk, chunks))
        
        # 合并结果
        time_points = []
        theta_values = []
        theta_dot_values = []
        
        for time_chunk, theta_chunk, theta_dot_chunk, _ in results:
            time_points.extend(time_chunk)
            theta_values.extend(theta_chunk)
            theta_dot_values.extend(theta_dot_chunk)
    else:
        # 使用单进程计算 - 修改为使用全局函数
        result = integrate_chunk((pendulum, t_span[0], t_span[1], dt, x0, tau_values, t_span[0]))
        time_points, theta_values, theta_dot_values, _ = result
    
    # 创建DataFrame
    data = {
        'time': time_points,
        'theta': theta_values,
        'theta_dot': theta_dot_values,
        'tau': [get_tau_at_time_global(t, tau_values, t_span[0], dt) for t in time_points]
    }
    
    return pd.DataFrame(data)

#############################################
# 步骤 3：准备数据以供RNN训练
#############################################

def create_sequences(data, sequence_length):
    """
    创建用于RNN训练的输入/输出序列
    
    参数:
    data (array): 原始数据数组，列为 [theta, theta_dot, tau]
    sequence_length (int): 序列长度 (窗口大小)
    
    返回:
    tuple: (X, y) 输入序列和目标输出
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        # 输入序列: t到t+sequence_length-1的状态和输入
        input_seq = data[i:i + sequence_length]
        
        # 输出目标: t+sequence_length时刻的状态
        target = data[i + sequence_length, 0:2]  # 只取theta和theta_dot
        
        X.append(input_seq)
        y.append(target)
    
    return np.array(X), np.array(y)

def prepare_data_for_training(df, sequence_length=10, test_split=0.2):
    """
    准备训练数据，包括创建序列、划分训练/测试集和归一化
    
    参数:
    df (DataFrame): 包含仿真数据的DataFrame
    sequence_length (int): 序列长度
    test_split (float): 测试集比例
    
    返回:
    tuple: 训练和测试用的张量、归一化器等
    """
    # 提取相关列的数值
    data_values = df[['theta', 'theta_dot', 'tau']].values
    
    # 创建序列
    X, y = create_sequences(data_values, sequence_length)
    
    # 按时间顺序划分训练集和测试集
    split_index = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # 重塑X以适应归一化器 (samples * features)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    # 初始化归一化器
    input_scaler = MinMaxScaler(feature_range=(-1, 1))
    output_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # 在训练数据上拟合并转换
    X_train_scaled_reshaped = input_scaler.fit_transform(X_train_reshaped)
    y_train_scaled = output_scaler.fit_transform(y_train)
    
    # 保存归一化器
    joblib.dump(input_scaler, 'models/input_scaler.pkl')
    joblib.dump(output_scaler, 'models/output_scaler.pkl')
    
    # 转换测试数据
    X_test_scaled_reshaped = input_scaler.transform(X_test_reshaped)
    y_test_scaled = output_scaler.transform(y_test)
    
    # 重塑回3D形状，用于RNN
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 为M2 Max优化批处理大小
    batch_size = 128  # 增大批次大小以充分利用M2 Max的并行能力
    
    # 使用pin_memory=True可以加速CPU到GPU的数据传输
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=2  # 使用多个工作进程加载数据
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )
    
    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
            train_loader, test_loader, input_scaler, output_scaler)

#############################################
# 步骤 4：构建循环神经网络模型
#############################################

class PhysicsAugmentedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dense_units, dropout=0.2, physics_enabled=True):
        """
        物理增强型LSTM模型，结合物理方程与神经网络
        
        参数:
        input_size (int): 输入特征数
        hidden_size (int): LSTM隐藏单元数
        num_layers (int): LSTM层数
        output_size (int): 输出特征数
        dense_units (int): 中间全连接层的单元数
        dropout (float): Dropout比率，用于防止过拟合
        physics_enabled (bool): 是否启用物理模型增强
        """
        super(PhysicsAugmentedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.physics_enabled = physics_enabled
        
        # 物理模型参数
        self.m = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # 质量参数可学习
        self.L = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # 长度参数可学习
        self.g = nn.Parameter(torch.tensor(9.81), requires_grad=False)  # 重力常数
        self.c = nn.Parameter(torch.tensor(0.5), requires_grad=True)   # 阻尼系数可学习
        self.dt = nn.Parameter(torch.tensor(0.02), requires_grad=False)  # 时间步长
        
        # 输入层的归一化
        self.input_norm = nn.LayerNorm(input_size)
        
        # 第一层LSTM - 处理输入序列
        self.lstm1 = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True  # 使用双向LSTM更好地捕捉上下文信息
        )
        
        # LSTM输出的归一化
        self.lstm1_norm = nn.LayerNorm(hidden_size * 2)  # 双向LSTM输出尺寸加倍
        
        # 第二层LSTM - 进一步处理序列
        self.lstm2 = nn.LSTM(
            hidden_size * 2,  # 双向LSTM的输出尺寸
            hidden_size,
            num_layers=num_layers-1 if num_layers > 1 else 1, 
            batch_first=True,
            dropout=dropout if num_layers > 2 else 0
        )
        
        # 注意力机制 - 帮助模型关注重要的时间步
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 神经网络修正器 - 学习物理模型的残差
        self.corrector_net = nn.Sequential(
            nn.BatchNorm1d(hidden_size + input_size),  # 接收LSTM特征和物理模型输入
            nn.Linear(hidden_size + input_size, dense_units),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units//2),
            nn.LeakyReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(dense_units//2, output_size)
        )
        
        # 物理/神经网络集成器 - 学习如何混合两个模型的输出
        self.integration_net = nn.Sequential(
            nn.Linear(output_size * 2, dense_units//2),  # 接收物理模型和神经网络的输出
            nn.ReLU(),
            nn.Linear(dense_units//2, 1),
            nn.Sigmoid()  # 输出一个0-1之间的权重
        )
        
        # 输出层的归一化
        self.output_norm = nn.LayerNorm(output_size)
        
    def physics_step(self, state, tau):
        """
        使用物理模型计算下一个状态
        
        参数:
        state (tensor): 当前状态 [theta, theta_dot]
        tau (tensor): 外部力矩
        
        返回:
        tensor: 下一个状态 [theta_next, theta_dot_next]
        """
        # 将输入解包
        theta, theta_dot = state[:, 0], state[:, 1]
        
        # 计算参数
        beta = self.c / (self.m * self.L**2)
        omega0_sq = self.g / self.L
        
        # 计算下一个状态
        theta_next = theta + self.dt * theta_dot
        theta_dot_next = theta_dot + self.dt * (-beta * theta_dot - 
                                               omega0_sq * torch.sin(theta) + 
                                               tau / (self.m * self.L**2))
        
        # 将结果重新打包
        return torch.stack([theta_next, theta_dot_next], dim=1)
        
    def attention_net(self, lstm_output):
        """
        注意力机制，计算各时间步的权重
        """
        attn_weights = self.attention(lstm_output)
        soft_attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * soft_attn_weights, dim=1)
        return context
        
    def forward(self, x):
        """
        前向传播
        
        参数:
        x (tensor): 输入序列，形状为 (batch_size, seq_len, input_size)
        
        返回:
        tensor: 预测输出，形状为 (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # 获取最后时间步的状态和力矩，用于物理模型
        last_step = x[:, -1, :]
        current_state = last_step[:, :2]  # theta, theta_dot
        current_tau = last_step[:, 2]     # tau
        
        # 输入归一化
        x = self.input_norm(x)
        
        # 第一层LSTM - 双向
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.lstm1_norm(lstm1_out)
        
        # 第二层LSTM
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # 应用注意力机制
        context = self.attention_net(lstm2_out)
        
        # 物理模型计算 (如果启用)
        if self.physics_enabled:
            physics_prediction = self.physics_step(current_state, current_tau)
            
            # 将LSTM特征和当前状态连接，用于修正物理模型
            combined_features = torch.cat([context, last_step], dim=1)
            
            # 神经网络修正
            nn_correction = self.corrector_net(combined_features)
            
            # 神经网络原始预测
            nn_prediction = nn_correction + current_state  # 残差学习
            
            # 计算混合权重 - 物理模型和神经网络预测的加权平均
            inputs_for_integration = torch.cat([physics_prediction, nn_prediction], dim=1)
            physics_weight = self.integration_net(inputs_for_integration)
            
            # 最终预测是物理模型和神经网络预测的加权平均
            final_output = physics_weight * physics_prediction + (1 - physics_weight) * nn_prediction
        else:
            # 不使用物理模型时，只依赖神经网络预测
            final_output = self.corrector_net(torch.cat([context, last_step], dim=1)) + current_state
        
        # 输出归一化
        final_output = self.output_norm(final_output)
        
        return final_output

#############################################
# 步骤 5：训练神经网络
#############################################

def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001, device=None,
                early_stopping_patience=5, scheduler_factor=0.5, scheduler_patience=3, weight_decay=1e-5):
    """
    训练LSTM模型 - 增强版
    
    参数:
    model (nn.Module): 要训练的模型
    train_loader (DataLoader): 训练数据加载器
    test_loader (DataLoader): 测试数据加载器
    num_epochs (int): 训练周期数
    learning_rate (float): 学习率
    device (torch.device): 训练设备 (CPU/GPU/MPS)
    early_stopping_patience (int): 早停耐心值
    scheduler_factor (float): 学习率衰减因子
    scheduler_patience (int): 学习率调度器耐心值
    weight_decay (float): 权重衰减系数，用于L2正则化
    
    返回:
    tuple: 训练和验证损失历史，best_epoch (最佳模型的epoch)
    """
    start_time = time.time()
    
    if device is None:
        # 检测并使用M2 Max上的MPS (Metal Performance Shaders)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"使用M2 Max MPS加速训练")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用CUDA加速训练")
        else:
            device = torch.device("cpu")
            print(f"使用CPU训练")
    
    model.to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params:,}")
    
    # 定义损失函数 - 使用更复杂的损失函数
    criterion = nn.MSELoss()
    
    # 创建自定义损失函数 - 包含物理约束
    def physics_informed_loss(outputs, targets, inputs):
        # 基本MSE损失
        mse_loss = criterion(outputs, targets)
        
        # 解包输入和输出，用于物理约束
        # 注意: 这里假设：
        # inputs 形状为 [batch_size, seq_len, features]，其中features包含 [theta, theta_dot, tau]
        # outputs 形状为 [batch_size, 2]，包含预测的 [theta, theta_dot]
        
        # 物理约束权重 - 增加物理约束的重要性
        physics_weight = 0.5  # 从0.1增加到0.5，让物理约束占更大比重
        
        # 获取上一个时间步的状态和输入力矩
        prev_theta = inputs[:, -1, 0]        # 上一时间步的theta
        prev_theta_dot = inputs[:, -1, 1]    # 上一时间步的theta_dot
        tau = inputs[:, -1, 2]               # 上一时间步的力矩
        
        # 预测的下一个状态
        pred_theta = outputs[:, 0]
        pred_theta_dot = outputs[:, 1]
        
        # 简化的物理模型约束（时间步长假设为0.02）
        dt = 0.02
        g = 9.81  # 重力加速度
        L = 1.0   # 摆长
        c = 0.5   # 阻尼系数
        m = 1.0   # 质量
        
        # 根据简化物理方程计算的下一个状态（欧拉方法）
        beta = c / (m * L**2)
        omega0_sq = g / L
        
        # 物理模型预测的theta
        physics_theta = prev_theta + dt * prev_theta_dot
        
        # 物理模型预测的theta_dot
        physics_theta_dot = prev_theta_dot + dt * (-beta * prev_theta_dot - 
                                                omega0_sq * torch.sin(prev_theta) + 
                                                tau / (m * L**2))
        
        # 物理约束损失 - 与物理模型预测的差异
        physics_loss = (torch.mean((pred_theta - physics_theta)**2) + 
                       torch.mean((pred_theta_dot - physics_theta_dot)**2))
        
        # 总损失 = MSE损失 + 物理约束权重 * 物理约束损失
        total_loss = mse_loss + physics_weight * physics_loss
        
        return total_loss
    
    # 优化器 - 使用AdamW，包含权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 使用更好的学习率调度器 - 余弦退火
    T_max = num_epochs  # 余弦周期
    eta_min = 1e-6      # 最小学习率
    
    # 创建一个余弦退火学习率调度器
    cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=T_max, 
        eta_min=eta_min
    )
    
    # 同时保持原有的ReduceLROnPlateau调度器
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=scheduler_factor, 
        patience=scheduler_patience, 
        verbose=True,
        min_lr=eta_min
    )
    
    # 早停设置
    best_val_loss = float('inf')
    best_epoch = 0
    early_stopping_counter = 0
    
    # 存储损失
    train_losses = []
    val_losses = []
    
    # 保存初始模型状态用于重新初始化
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()  # 设置为训练模式
        running_train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 使用带有物理约束的损失函数
            loss = physics_informed_loss(outputs, targets, inputs)
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_train_loss += loss.item()
        
        # 计算平均训练损失
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # 验证
        model.eval()  # 设置为评估模式
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # 使用相同的物理约束损失函数进行验证
                loss = physics_informed_loss(outputs, targets, inputs)
                running_val_loss += loss.item()
        
        # 计算平均验证损失
        epoch_val_loss = running_val_loss / len(test_loader)
        val_losses.append(epoch_val_loss)
        
        # 更新学习率调度器
        plateau_scheduler.step(epoch_val_loss)
        
        # 在余弦退火调度器上也步进一步
        cos_scheduler.step()
        
        # 打印进度
        epoch_time = time.time() - epoch_start_time
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch > num_epochs - 5:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_train_loss:.6f}, '
                  f'Val Loss: {epoch_val_loss:.6f}, '
                  f'Time: {epoch_time:.2f}s, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 检查是否需要早停和保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'models/pendulum_lstm_best.pth')
            early_stopping_counter = 0
            print(f"最佳模型更新: 验证损失 = {best_val_loss:.6f}，周期 = {best_epoch}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"早停触发，共训练 {epoch + 1} 个周期，最佳性能在第 {best_epoch} 个周期")
                break
        
        # 学习率退火的策略 - 如果接近收敛但仍在改善，可以尝试降低学习率
        # 这是除了自动调度器之外的一种手动控制策略
        if epoch > num_epochs * 0.6 and epoch % 10 == 0 and early_stopping_counter > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
                print(f"手动降低学习率至: {param_group['lr']:.6f}")
    
    # 加载最佳模型
    checkpoint = torch.load('models/pendulum_lstm_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 将最佳模型单独另存一份
    torch.save({
        'epoch': best_epoch,
        'state_dict': model.state_dict(),
        'best_val_loss': best_val_loss,
    }, 'models/pendulum_lstm_final.pth')
    
    total_time = time.time() - start_time
    print(f"训练完成，总用时: {total_time:.2f}s")
    print(f"最佳模型来自第 {best_epoch} 个周期，验证损失: {best_val_loss:.6f}")
    
    return train_losses, val_losses, best_epoch

#############################################
# 步骤 6：评估模型性能
#############################################

def plot_training_curves(train_losses, val_losses):
    """
    绘制训练和验证损失曲线
    
    参数:
    train_losses (list): 训练损失历史
    val_losses (list): 验证损失历史
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/training_curves.png', dpi=300)
    plt.close()

def evaluate_model(model, test_loader, criterion, device=None):
    """
    在测试集上评估模型
    
    参数:
    model (nn.Module): 要评估的模型
    test_loader (DataLoader): 测试数据加载器
    criterion: 损失函数
    device (torch.device): 计算设备
    
    返回:
    float: 平均测试损失
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    eval_time = time.time() - start_time
    
    print(f'Average Test Loss: {avg_loss:.6f}, Evaluation Time: {eval_time:.2f}s')
    return avg_loss

def multi_step_prediction(model, initial_sequence, df, sequence_length, prediction_steps, 
                         input_scaler, output_scaler, device=None, use_error_correction=True, 
                         teacher_forcing_ratio=0.0, pure_physics_ratio=0.5):
    """
    使用物理增强模型进行多步预测
    
    参数:
    model (nn.Module): 训练好的模型
    initial_sequence (array): 初始输入序列
    df (DataFrame): 原始数据DataFrame (用于获取真实力矩和真实状态)
    sequence_length (int): 序列长度
    prediction_steps (int): 预测步数
    input_scaler (MinMaxScaler): 输入归一化器
    output_scaler (MinMaxScaler): 输出归一化器
    device (torch.device): 计算设备
    use_error_correction (bool): 是否使用误差校正
    teacher_forcing_ratio (float): 教师强制比例，用于混合预测值和真实值
    pure_physics_ratio (float): 纯物理模型权重，0表示完全使用神经网络，1表示完全使用物理模型
    
    返回:
    tuple: 预测状态、真实状态和纯物理模型预测
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    model.eval()  # 设置为评估模式
    
    # 提取初始序列
    current_sequence = initial_sequence.copy()
    
    # 存储预测结果
    predicted_states_scaled = []     # 神经网络原始预测
    predicted_states_corrected = []  # 校正后的状态
    pure_physics_states = []         # 纯物理模型预测
    
    # 开始索引 (初始序列之后的第一个时间步)
    start_index = df.index[sequence_length]
    
    # 保存物理模型和神经网络的混合模式
    if hasattr(model, 'physics_enabled'):
        is_physics_model = model.physics_enabled
    else:
        is_physics_model = False
        
    print(f"多步预测模式: {'物理增强' if is_physics_model else '纯神经网络'}, "
          f"纯物理模型权重: {pure_physics_ratio:.2f}")
          
    # 启用评估模式下的梯度计算，以便可视化物理权重
    # 通常在eval模式下不需要梯度，但这里我们想了解模型的行为
    torch.set_grad_enabled(True)
    
    # 获取模型中学到的物理参数，如果是物理增强模型
    if hasattr(model, 'physics_enabled') and model.physics_enabled:
        # 使用模型学习的物理参数
        g = model.g.item()
        m = model.m.item()
        L = model.L.item()
        c = model.c.item()
        print(f"使用模型学习到的物理参数: m={m:.3f}, L={L:.3f}, c={c:.3f}")
    else:
        # 使用默认物理参数
        g = 9.81       # 重力加速度 (m/s²)
        m = 1.0        # 质量 (kg)
        L = 1.0        # 摆长 (m)
        c = 0.5        # 阻尼系数 (N·m·s/rad)
    
    dt = df['time'].iloc[1] - df['time'].iloc[0]  # 时间步长
    
    # 物理模型辅助函数
    def physics_step(theta, theta_dot, tau, dt):
        """使用物理方程计算下一个状态"""
        beta = c / (m * L**2)
        omega0_sq = g / L
        
        # 更新角度
        new_theta = theta + dt * theta_dot
        
        # 更新角速度
        new_theta_dot = theta_dot + dt * (-beta * theta_dot - 
                                        omega0_sq * np.sin(theta) + 
                                        tau / (m * L**2))
        return new_theta, new_theta_dot
    
    # 误差预估和校正函数
    def kalman_correct(pred_theta, pred_theta_dot, physics_theta, physics_theta_dot, prev_error_theta=0, prev_error_theta_dot=0):
        """简化的卡尔曼滤波风格校正"""
        # 卡尔曼增益 (Kalman gain) - 决定信任模型预测还是物理预测的程度
        # 这里使用简化版本，固定的增益系数
        k_theta = 0.7        # 角度的增益
        k_theta_dot = 0.6    # 角速度的增益
        
        # 误差估计
        error_theta = pred_theta - physics_theta
        error_theta_dot = pred_theta_dot - physics_theta_dot
        
        # 误差平滑 (考虑前一步的误差)
        smoothed_error_theta = 0.7 * error_theta + 0.3 * prev_error_theta
        smoothed_error_theta_dot = 0.7 * error_theta_dot + 0.3 * prev_error_theta_dot
        
        # 校正预测值
        corrected_theta = pred_theta - k_theta * smoothed_error_theta
        corrected_theta_dot = pred_theta_dot - k_theta_dot * smoothed_error_theta_dot
        
        return corrected_theta, corrected_theta_dot, error_theta, error_theta_dot
    
    # 教师强制函数 - 混合预测状态和真实状态
    def apply_teacher_forcing(pred_state, true_state, ratio):
        """应用教师强制策略"""
        if np.random.random() < ratio:
            # 使用一定比例的真实状态，而不是全部使用预测状态
            return pred_state * (1 - ratio) + true_state * ratio
        return pred_state
    
    # 初始化误差估计
    prev_error_theta = 0
    prev_error_theta_dot = 0
    
    with torch.set_grad_enabled(False):  # 关闭梯度计算以加速预测
        for i in range(prediction_steps):
            # 转换当前序列为tensor
            current_sequence_tensor = torch.tensor(
                current_sequence.reshape(1, sequence_length, -1), 
                dtype=torch.float32
            ).to(device)
            
            # 获取当前状态和力矩 (用于物理模型)
            current_state = output_scaler.inverse_transform(current_sequence[-1, :2].reshape(1, -1))[0]
            current_theta, current_theta_dot = current_state
            
            # 获取当前时间步的力矩
            next_tau_original = df.iloc[start_index + i]['tau']
            
            # 预测下一状态 - 使用神经网络
            next_state_scaled_tensor = model(current_sequence_tensor)
            next_state_scaled = next_state_scaled_tensor.cpu().numpy()
            
            # 存储神经网络原始预测
            predicted_states_scaled.append(next_state_scaled[0])
            
            # 逆归一化神经网络预测
            nn_prediction = output_scaler.inverse_transform(next_state_scaled)[0]
            nn_theta, nn_theta_dot = nn_prediction
            
            # 使用物理模型计算下一状态
            physics_theta, physics_theta_dot = physics_step(
                current_theta, current_theta_dot, next_tau_original, dt
            )
            
            # 存储纯物理模型预测
            pure_physics_states.append([physics_theta, physics_theta_dot])
            
            # 混合预测结果 - 结合物理模型和神经网络
            # pure_physics_ratio控制物理模型的权重
            hybrid_theta = pure_physics_ratio * physics_theta + (1 - pure_physics_ratio) * nn_theta
            hybrid_theta_dot = pure_physics_ratio * physics_theta_dot + (1 - pure_physics_ratio) * nn_theta_dot
            
            # 如果启用误差校正，进一步优化预测
            if use_error_correction:
                # 获取真实状态 (仅用于计算误差)
                true_row = df.iloc[start_index + i]
                true_theta = true_row['theta']
                true_theta_dot = true_row['theta_dot']
                
                # 自适应误差校正 - 根据历史预测准确度动态调整校正强度
                if i > 0:
                    # 获取真实状态 (仅用于计算误差)
                    prev_true_row = df.iloc[start_index + i - 1]
                    prev_true_state = np.array([prev_true_row['theta'], prev_true_row['theta_dot']])
                    
                    # 计算前一步预测的误差
                    prev_nn_error = np.abs(nn_prediction - prev_true_state).mean()
                    prev_physics_error = np.abs(pure_physics_states[i-1] - prev_true_state).mean()
                    
                    # 动态调整混合比例，更倾向于更准确的模型
                    # 使用softmax函数使两种模型的权重和为1
                    total_error = prev_nn_error + prev_physics_error
                    if total_error > 0:
                        # 误差越小，权重越大
                        physics_weight = prev_nn_error / total_error
                        nn_weight = prev_physics_error / total_error
                    else:
                        physics_weight = nn_weight = 0.5
                    
                    # 应用动态权重
                    corrected_theta = physics_weight * physics_theta + nn_weight * nn_theta
                    corrected_theta_dot = physics_weight * physics_theta_dot + nn_weight * nn_theta_dot
                else:
                    # 第一步使用预设混合比例
                    corrected_theta = hybrid_theta
                    corrected_theta_dot = hybrid_theta_dot
                
                # 可选：应用教师强制 (将预测值部分替换为真实值)
                if teacher_forcing_ratio > 0:
                    corrected_theta = (1 - teacher_forcing_ratio) * corrected_theta + teacher_forcing_ratio * true_theta
                    corrected_theta_dot = (1 - teacher_forcing_ratio) * corrected_theta_dot + teacher_forcing_ratio * true_theta_dot
            else:
                # 不使用误差校正，直接使用混合预测
                corrected_theta = hybrid_theta
                corrected_theta_dot = hybrid_theta_dot
            
            # 存储校正后的状态
            corrected_state = np.array([corrected_theta, corrected_theta_dot])
            predicted_states_corrected.append(corrected_state)
            
            # 归一化校正后的状态和下一个力矩，用于下一步预测
            dummy_input = np.zeros((1, 3))
            dummy_input[0, 0] = corrected_theta
            dummy_input[0, 1] = corrected_theta_dot
            dummy_input[0, 2] = next_tau_original
            next_features_scaled = input_scaler.transform(dummy_input)[0]
            
            # 更新序列，滚动窗口
            current_sequence = np.append(
                current_sequence[1:],
                next_features_scaled.reshape(1, -1),
                axis=0
            )
    
    # 转换为NumPy数组
    predicted_states_scaled = np.array(predicted_states_scaled)
    predicted_states_original = output_scaler.inverse_transform(predicted_states_scaled)
    predicted_states_corrected = np.array(predicted_states_corrected)
    pure_physics_states = np.array(pure_physics_states)
    
    # 获取真实状态进行比较
    true_states = df.iloc[start_index:start_index+prediction_steps][['theta', 'theta_dot']].values
    
    # 计算误差统计
    mse_original = np.mean((predicted_states_original - true_states)**2)
    mse_corrected = np.mean((predicted_states_corrected - true_states)**2)
    mse_physics = np.mean((pure_physics_states - true_states)**2)
    
    print(f"神经网络原始预测MSE: {mse_original:.6f}")
    print(f"混合预测(校正后)MSE: {mse_corrected:.6f}")
    print(f"纯物理模型预测MSE: {mse_physics:.6f}")
    
    # 计算相对性能比较
    if mse_physics > 0:
        nn_vs_physics = mse_physics / mse_original
        corrected_vs_physics = mse_physics / mse_corrected
        print(f"神经网络vs物理模型: {nn_vs_physics:.2f}x")
        print(f"校正后vs物理模型: {corrected_vs_physics:.2f}x")
    
    # 返回校正后的预测和纯物理模型的预测
    return predicted_states_corrected, true_states, pure_physics_states

def plot_multi_step_prediction(time_vector, true_states, predicted_states, physics_model_predictions=None, model_name="LSTM"):
    """
    绘制多步预测与真实轨迹的比较 - 增强版
    
    参数:
    time_vector (array): 时间向量
    true_states (array): 真实状态
    predicted_states (array): 预测状态
    physics_model_predictions (array, optional): 物理模型预测，用于比较
    model_name (str): 模型名称，用于标题
    """
    # 定义一个函数来安全地设置标题和标签，避免中文显示问题
    def safe_text(text, fallback_text=None):
        if fallback_text is None:
            fallback_text = text.replace('角度', 'Theta').replace('角速度', 'Theta_dot').replace('预测', 'Prediction')
        try:
            # 测试是否可以渲染文本
            fig = plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, text, ha='center', va='center')
            plt.close(fig)
            return text
        except Exception:
            return fallback_text
    
    # 为了更好的可视化，计算误差
    theta_error = np.abs(predicted_states[:, 0] - true_states[:, 0])
    theta_dot_error = np.abs(predicted_states[:, 1] - true_states[:, 1])
    
    # 计算平均误差
    mean_theta_error = np.mean(theta_error)
    mean_theta_dot_error = np.mean(theta_dot_error)
    
    # 如果有物理模型预测，计算其误差
    if physics_model_predictions is not None:
        physics_theta_error = np.abs(physics_model_predictions[:, 0] - true_states[:, 0])
        physics_theta_dot_error = np.abs(physics_model_predictions[:, 1] - true_states[:, 1])
        mean_physics_theta_error = np.mean(physics_theta_error)
        mean_physics_theta_dot_error = np.mean(physics_theta_dot_error)
    
    # 创建更详细的多面板图
    fig = plt.figure(figsize=(15, 12), dpi=300)
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # 顶部标题
    fig.suptitle(safe_text(f'多步预测分析: {model_name}模型\n', f'Multi-step Prediction: {model_name} Model\n'), fontsize=16, fontweight='bold')
    
    # 第一行：状态预测
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_vector, true_states[:, 0], 'g-', label=safe_text('真实角度', 'True Theta'), linewidth=2)
    ax1.plot(time_vector, predicted_states[:, 0], 'r--', label=safe_text('预测角度', 'Predicted Theta'), linewidth=2)
    if physics_model_predictions is not None:
        ax1.plot(time_vector, physics_model_predictions[:, 0], 'b-.', label=safe_text('物理模型', 'Physics Model'), linewidth=2)
    ax1.set_title(safe_text('角度 (θ) 预测', 'Theta (θ) Prediction'), fontsize=14)
    ax1.set_ylabel(safe_text('角度 (rad)', 'Angle (rad)'), fontsize=12)
    ax1.text(0.02, 0.02, safe_text(f'平均误差: {mean_theta_error:.4f} rad', f'Mean Error: {mean_theta_error:.4f} rad'), transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_vector, true_states[:, 1], 'g-', label=safe_text('真实角速度', 'True Angular Velocity'), linewidth=2)
    ax2.plot(time_vector, predicted_states[:, 1], 'r--', label=safe_text('预测角速度', 'Predicted Angular Velocity'), linewidth=2)
    if physics_model_predictions is not None:
        ax2.plot(time_vector, physics_model_predictions[:, 1], 'b-.', label=safe_text('物理模型', 'Physics Model'), linewidth=2)
    ax2.set_title(safe_text('角速度 (θ̇) 预测', 'Angular Velocity (θ̇) Prediction'), fontsize=14)
    ax2.set_ylabel(safe_text('角速度 (rad/s)', 'Angular Velocity (rad/s)'), fontsize=12)
    ax2.text(0.02, 0.02, safe_text(f'平均误差: {mean_theta_dot_error:.4f} rad/s', f'Mean Error: {mean_theta_dot_error:.4f} rad/s'), transform=ax2.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    ax2.legend(fontsize=10)
    ax2.grid(True)
    
    # 第二行：误差分析
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_vector, theta_error, 'r-', label=safe_text('角度误差', 'Theta Error'), linewidth=2)
    if physics_model_predictions is not None:
        ax3.plot(time_vector, physics_theta_error, 'b-.', label=safe_text('物理模型误差', 'Physics Model Error'), linewidth=2)
    ax3.set_title(safe_text('角度预测误差', 'Theta Prediction Error'), fontsize=14)
    ax3.set_ylabel(safe_text('|误差| (rad)', '|Error| (rad)'), fontsize=12)
    ax3.axhline(y=mean_theta_error, color='r', linestyle='--', alpha=0.5, label=safe_text(f'平均误差: {mean_theta_error:.4f}', f'Mean Error: {mean_theta_error:.4f}'))
    if physics_model_predictions is not None:
        ax3.axhline(y=mean_physics_theta_error, color='b', linestyle='--', alpha=0.5, 
                    label=safe_text(f'物理模型平均误差: {mean_physics_theta_error:.4f}', f'Physics Model Mean Error: {mean_physics_theta_error:.4f}'))
    ax3.legend(fontsize=10)
    ax3.grid(True)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_vector, theta_dot_error, 'r-', label=safe_text('角速度误差', 'Angular Velocity Error'), linewidth=2)
    if physics_model_predictions is not None:
        ax4.plot(time_vector, physics_theta_dot_error, 'b-.', label=safe_text('物理模型误差', 'Physics Model Error'), linewidth=2)
    ax4.set_title(safe_text('角速度预测误差', 'Angular Velocity Prediction Error'), fontsize=14)
    ax4.set_ylabel(safe_text('|误差| (rad/s)', '|Error| (rad/s)'), fontsize=12)
    ax4.axhline(y=mean_theta_dot_error, color='r', linestyle='--', alpha=0.5, label=safe_text(f'平均误差: {mean_theta_dot_error:.4f}', f'Mean Error: {mean_theta_dot_error:.4f}'))
    if physics_model_predictions is not None:
        ax4.axhline(y=mean_physics_theta_dot_error, color='b', linestyle='--', alpha=0.5, 
                    label=safe_text(f'物理模型平均误差: {mean_physics_theta_dot_error:.4f}', f'Physics Model Mean Error: {mean_physics_theta_dot_error:.4f}'))
    ax4.legend(fontsize=10)
    ax4.grid(True)
    
    # 第三行：相位图（角度 vs 角速度）和累积误差
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(true_states[:, 0], true_states[:, 1], 'g-', label=safe_text('真实轨迹', 'True Trajectory'), linewidth=2, alpha=0.8)
    ax5.plot(predicted_states[:, 0], predicted_states[:, 1], 'r--', label=safe_text('预测轨迹', 'Predicted Trajectory'), linewidth=2, alpha=0.8)
    if physics_model_predictions is not None:
        ax5.plot(physics_model_predictions[:, 0], physics_model_predictions[:, 1], 'b-.', 
                label=safe_text('物理模型轨迹', 'Physics Model Trajectory'), linewidth=2, alpha=0.8)
    ax5.set_title(safe_text('相位图：角度 vs 角速度', 'Phase Plot: Theta vs Angular Velocity'), fontsize=14)
    ax5.set_xlabel(safe_text('角度 (rad)', 'Angle (rad)'), fontsize=12)
    ax5.set_ylabel(safe_text('角速度 (rad/s)', 'Angular Velocity (rad/s)'), fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True)
    
    # 累积误差图
    ax6 = fig.add_subplot(gs[2, 1])
    cum_error_theta = np.cumsum(theta_error) / np.arange(1, len(theta_error) + 1)
    cum_error_theta_dot = np.cumsum(theta_dot_error) / np.arange(1, len(theta_dot_error) + 1)
    ax6.plot(time_vector, cum_error_theta, 'r-', label=safe_text('累积角度误差', 'Cumulative Theta Error'), linewidth=2)
    ax6.plot(time_vector, cum_error_theta_dot, 'b-', label=safe_text('累积角速度误差', 'Cumulative Angular Velocity Error'), linewidth=2)
    
    # 如果有物理模型，也画出其累积误差
    if physics_model_predictions is not None:
        cum_error_physics_theta = np.cumsum(physics_theta_error) / np.arange(1, len(physics_theta_error) + 1)
        cum_error_physics_theta_dot = np.cumsum(physics_theta_dot_error) / np.arange(1, len(physics_theta_dot_error) + 1)
        ax6.plot(time_vector, cum_error_physics_theta, 'r--', label=safe_text('物理模型累积角度误差', 'Physics Model Cumulative Theta Error'), linewidth=2)
        ax6.plot(time_vector, cum_error_physics_theta_dot, 'b--', label=safe_text('物理模型累积角速度误差', 'Physics Model Cumulative Angular Velocity Error'), linewidth=2)
    
    ax6.set_title(safe_text('累积平均误差', 'Cumulative Average Error'), fontsize=14)
    ax6.set_xlabel(safe_text('时间 (s)', 'Time (s)'), fontsize=12)
    ax6.set_ylabel(safe_text('累积平均误差', 'Cumulative Average Error'), fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True)
    
    # 为所有子图添加共享X轴标签
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel(safe_text('时间 (s)', 'Time (s)'), fontsize=12)
    
    # 额外信息
    fig.text(0.5, 0.01, 
             safe_text(f'预测步数: {len(time_vector)}   时间范围: {time_vector[0]:.2f}s - {time_vector[-1]:.2f}s', 
                      f'Prediction Steps: {len(time_vector)}   Time Range: {time_vector[0]:.2f}s - {time_vector[-1]:.2f}s'), 
             ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])  # 调整边距，为suptitle留出空间
    plt.savefig('figures/multi_step_prediction.png', dpi=300, bbox_inches='tight')
    
    # 额外保存一个高质量版本
    plt.savefig('figures/multi_step_prediction_hq.png', dpi=600, bbox_inches='tight')
    
    plt.close()

#############################################
# 主程序：运行完整模拟流程
#############################################

def main():
    """
    运行完整的动力学系统神经网络模拟流程 - 增强版
    """
    start_time = time.time()
    
    # 检测设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("已检测到M2 Max，将使用Metal Performance Shaders (MPS)加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("已检测到NVIDIA GPU，将使用CUDA加速")
    else:
        device = torch.device("cpu")
        print("未检测到GPU加速器，将使用CPU计算")
    
    # 设置随机种子，确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 步骤1: 创建单摆系统
    pendulum = PendulumSystem(m=1.0, L=1.0, g=9.81, c=0.5)
    print("步骤1: 已创建单摆系统")
    
    # 步骤2: 生成仿真数据
    print("步骤2: 开始生成仿真数据...")
    gen_start_time = time.time()
    
    # 检测CPU核心数以确定并行计算工作进程数
    num_workers = multiprocessing.cpu_count()
    print(f"检测到{num_workers}个CPU核心，将用于并行计算")
    
    # 检查是否已经存在仿真数据文件，避免重复生成
    if os.path.exists('simulation_data.csv'):
        print("正在加载已有的仿真数据...")
        df = pd.read_csv('simulation_data.csv')
        
        # 检查文件是否完整
        expected_columns = ['time', 'theta', 'theta_dot', 'tau']
        if all(col in df.columns for col in expected_columns):
            print("仿真数据已加载完成")
        else:
            print("已存在的仿真数据不完整，将重新生成")
            df = generate_simulation_data(
                pendulum, 
                t_span=(0, 30),  # 增加仿真时间至30秒，获取更多样本
                dt=0.02,         # 0.02秒的时间步长
                x0=[0.1, 0.0],   # 初始状态：theta=0.1, theta_dot=0
                torque_type="mixed",  # 混合力矩类型
                use_parallel=True,    # 使用并行计算
                num_workers=num_workers  # 工作进程数
            )
            # 保存仿真数据
            df.to_csv('simulation_data.csv', index=False)
    else:
        df = generate_simulation_data(
            pendulum, 
            t_span=(0, 30),  # 增加仿真时间至30秒，获取更多样本
            dt=0.02,         # 0.02秒的时间步长
            x0=[0.1, 0.0],   # 初始状态：theta=0.1, theta_dot=0
            torque_type="mixed",  # 混合力矩类型
            use_parallel=True,    # 使用并行计算
            num_workers=num_workers  # 工作进程数
        )
        # 保存仿真数据
        df.to_csv('simulation_data.csv', index=False)
    
    gen_time = time.time() - gen_start_time
    print(f"仿真数据处理完成，耗时: {gen_time:.2f}秒")
    
    # 绘制仿真数据
    plt.figure(figsize=(12, 10), dpi=300)
    
    plt.subplot(3, 1, 1)
    plt.plot(df['time'], df['theta'], linewidth=2)
    plt.title('单摆角度 (Theta)', fontsize=14)
    plt.ylabel('角度 (rad)', fontsize=12)
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(df['time'], df['theta_dot'], linewidth=2)
    plt.title('角速度 (Theta_dot)', fontsize=14)
    plt.ylabel('角速度 (rad/s)', fontsize=12)
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(df['time'], df['tau'], linewidth=2)
    plt.title('输入力矩 (Tau)', fontsize=14)
    plt.xlabel('时间 (s)', fontsize=12)
    plt.ylabel('力矩 (N·m)', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figures/simulation_data.png', dpi=300)
    plt.close()
    
    # 步骤3: 准备数据用于RNN训练
    print("步骤3: 开始准备训练数据...")
    data_prep_start_time = time.time()
    
    # 增加序列长度，捕捉更长时间的依赖关系
    sequence_length = 20  # 从10增加到20
    data_tuple = prepare_data_for_training(df, sequence_length=sequence_length, test_split=0.2)
    (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
     X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
     train_loader, test_loader, input_scaler, output_scaler) = data_tuple
    
    data_prep_time = time.time() - data_prep_start_time
    print(f"训练数据准备完成，耗时: {data_prep_time:.2f}秒")
    print(f"训练集形状: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
    print(f"测试集形状: X={X_test_scaled.shape}, y={y_test_scaled.shape}")
    
    # 步骤4: 构建物理增强LSTM模型
    # 定义模型参数
    input_size = X_train_scaled.shape[2]  # 输入特征数
    hidden_size = 128                     # LSTM隐藏单元数
    num_layers = 3                        # LSTM层数
    output_size = y_train_scaled.shape[1] # 输出特征数
    dense_units = 96                      # 全连接层单元数
    dropout_rate = 0.25                   # Dropout比率
    physics_enabled = True                # 启用物理模型增强
    
    # 创建物理增强模型
    model = PhysicsAugmentedLSTM(
        input_size, 
        hidden_size, 
        num_layers, 
        output_size, 
        dense_units, 
        dropout_rate,
        physics_enabled
    )
    model.to(device)
    print("步骤4: 已构建物理增强型LSTM模型")
    print(model)
    
    # 打印物理参数
    print(f"物理参数(可学习): 质量={model.m.item():.2f}kg, 长度={model.L.item():.2f}m, 阻尼={model.c.item():.2f}")
    
    # 步骤5: 训练模型
    print("步骤5: 开始训练模型...")
    train_start_time = time.time()
    
    # 增加训练参数
    num_epochs = 300              # 增加训练周期上限 (从200增加到300)
    learning_rate = 0.001         # 初始学习率保持不变
    weight_decay = 1e-5           # 添加权重衰减参数
    
    train_losses, val_losses, best_epoch = train_model(
        model, train_loader, test_loader, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        device=device,
        early_stopping_patience=20,   # 增加早停耐心参数 (从15增加到20)
        scheduler_factor=0.4,         # 增强学习率衰减 (从0.5减少到0.4)
        scheduler_patience=7,         # 增加学习率调度器耐心参数 (从5增加到7)
        weight_decay=weight_decay     # 添加权重衰减
    )
    
    train_time = time.time() - train_start_time
    print(f"模型训练完成，耗时: {train_time:.2f}秒")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses)
    print("训练曲线已绘制并保存")
    
    # 步骤6: 评估模型
    print("步骤6: 开始评估模型...")
    eval_start_time = time.time()
    
    # 加载保存的最佳模型
    checkpoint = torch.load('models/pendulum_lstm_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"已加载最佳模型：来自第 {checkpoint['epoch'] + 1} 个周期")
    
    criterion = nn.MSELoss()
    avg_test_loss = evaluate_model(model, test_loader, criterion, device)
    
    # 增强多步预测评估
    start_idx = 0  # 从测试集的开始
    initial_sequence = X_test_scaled[start_idx]
    prediction_steps = min(300, len(df) - sequence_length - start_idx)  # 预测步数
    
    print(f"执行多步预测，预测步数: {prediction_steps}")
    
    # 设置评估参数
    use_error_correction = True      # 启用误差校正
    teacher_forcing_ratio = 0.0      # 在测试时不使用教师强制
    pure_physics_ratio = 0.7         # 增加物理模型在混合中的权重(从0.5增加到0.7)
    
    # 生成多步预测
    predicted_states, true_states, physics_states = multi_step_prediction(
        model, initial_sequence, df, sequence_length, prediction_steps,
        input_scaler, output_scaler, device,
        use_error_correction=use_error_correction,
        teacher_forcing_ratio=teacher_forcing_ratio,
        pure_physics_ratio=pure_physics_ratio
    )
    
    # 生成纯物理模型预测作为参考
    print("生成物理模型预测作为参考...")
    # 物理模型参数
    m = 1.0        # 质量 (kg)
    L = 1.0        # 摆长 (m)
    g = 9.81       # 重力加速度 (m/s²)
    c = 0.5        # 阻尼系数 (N·m·s/rad)
    dt = df['time'].iloc[1] - df['time'].iloc[0]  # 时间步长
    
    # 初始状态 - 使用与神经网络相同的起点
    start_index = df.index[sequence_length]
    init_state = df.iloc[start_index-1][['theta', 'theta_dot']].values
    
    # 存储物理模型预测
    physics_predictions = []
    theta, theta_dot = init_state
    
    for i in range(prediction_steps):
        # 获取当前力矩
        tau = df.iloc[start_index + i]['tau']
        
        # 计算下一个状态
        beta = c / (m * L**2)
        omega0_sq = g / L
        
        # 更新角度
        theta_new = theta + dt * theta_dot
        
        # 更新角速度
        theta_dot_new = theta_dot + dt * (-beta * theta_dot - 
                                         omega0_sq * np.sin(theta) + 
                                         tau / (m * L**2))
        
        # 保存预测
        physics_predictions.append([theta_new, theta_dot_new])
        
        # 更新状态
        theta, theta_dot = theta_new, theta_dot_new
    
    physics_predictions = np.array(physics_predictions)
    
    # 绘制多步预测结果 - 增强版
    time_vector = df['time'].iloc[sequence_length:sequence_length+prediction_steps].values
    plot_multi_step_prediction(
        time_vector, 
        true_states, 
        predicted_states, 
        physics_model_predictions=physics_states,
        model_name="物理增强型神经网络"
    )
    
    eval_time = time.time() - eval_start_time
    print(f"模型评估完成，耗时: {eval_time:.2f}秒")
    print("多步预测结果已保存到figures/multi_step_prediction.png")
    
    # 总结执行时间
    total_time = time.time() - start_time
    print(f"项目执行完毕！总耗时: {total_time:.2f}秒")
    
    # 性能摘要
    print("\n===== 性能摘要 =====")
    print(f"数据生成时间: {gen_time:.2f}秒")
    print(f"数据准备时间: {data_prep_time:.2f}秒")
    print(f"模型训练时间: {train_time:.2f}秒")
    print(f"模型评估时间: {eval_time:.2f}秒")
    print(f"总执行时间: {total_time:.2f}秒")
    
    # 模型性能摘要
    print("\n===== 模型性能摘要 =====")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"序列长度: {sequence_length}")
    print(f"隐藏层大小: {hidden_size}")
    print(f"LSTM层数: {num_layers}")
    print(f"最佳验证损失: {min(val_losses):.6f} (第 {best_epoch} 个周期)")
    print(f"测试集损失: {avg_test_loss:.6f}")
    
    # 预测性能分析
    model_mse = np.mean((predicted_states - true_states) ** 2)
    physics_mse = np.mean((physics_states - true_states) ** 2)
    performance_ratio = physics_mse / model_mse if model_mse > 0 else float('inf')
    
    print(f"\n模型多步预测MSE: {model_mse:.6f}")
    print(f"物理模型多步预测MSE: {physics_mse:.6f}")
    print(f"性能比值 (物理模型MSE/神经网络MSE): {performance_ratio:.2f}x")
    
    if performance_ratio > 1.0:
        print(f"神经网络模型预测性能比纯物理模型好 {performance_ratio:.2f} 倍")
    else:
        print(f"神经网络模型预测性能不如纯物理模型")
        
    print("\n项目完成！")

if __name__ == "__main__":
    main()