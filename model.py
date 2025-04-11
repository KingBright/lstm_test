# model.py

import torch
import torch.nn as nn
import config # 导入配置

# ------------------------- PureLSTM (增强正则化) -------------------------
class PureLSTM(nn.Module):
    """
    优化版LSTM模型，添加了额外的正则化层和技术。
    用于Sequence-to-Sequence预测，输入N步，预测未来K步的状态。
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, # output_size 仍然是每步的输出维度 (例如 2)
                 dense_units, dropout):
        super(PureLSTM, self).__init__()

        if num_layers < 1: raise ValueError("num_layers 必须至少为 1。")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = config.OUTPUT_SEQ_LEN # 获取预测序列长度 K

        # 增强版dropout
        self.dropout_rate = dropout
        self.feature_dropout = nn.Dropout2d(dropout/2)  # 对特征通道随机归零，提供更强的正则化

        # 增强输入归一化
        self.input_norm = nn.LayerNorm(input_size)
        
        # 添加噪声注入层（训练时）
        self.noise_level = 0.01  # 轻微噪声，帮助模型对输入变化更加鲁棒

        # 第一个 LSTM 层 (双向)
        is_bidirectional = True
        lstm1_output_size = hidden_size * 2 if is_bidirectional else hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidirectional)
        self.lstm1_norm = nn.LayerNorm(lstm1_output_size)
        
        # 在LSTM层之间添加dropout
        self.inter_dropout = nn.Dropout(dropout)

        # 后续 LSTM 层 (单向)
        self.lstm2 = None
        last_lstm_feature_size = lstm1_output_size
        if self.num_layers > 1:
             lstm2_num_layers = self.num_layers - 1
             lstm2_dropout = dropout if lstm2_num_layers > 1 else 0.0
             self.lstm2 = nn.LSTM(lstm1_output_size, hidden_size, num_layers=lstm2_num_layers, batch_first=True, dropout=lstm2_dropout, bidirectional=False)
             self.lstm2_norm = nn.LayerNorm(hidden_size)  # 增加第二个LSTM的归一化
             last_lstm_feature_size = hidden_size

        # 增强版输出 MLP 网络
        self.output_net = nn.Sequential(
            nn.Linear(last_lstm_feature_size, dense_units),
            nn.LayerNorm(dense_units),  # 添加层归一化
            nn.LeakyReLU(),
            nn.Dropout(dropout * 1.2),  # 稍微增加dropout率
            nn.Linear(dense_units, dense_units // 2),
            nn.LayerNorm(dense_units // 2),  # 添加层归一化
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            # 最终线性层输出 K * output_size 个值
            nn.Linear(dense_units // 2, self.output_seq_len * output_size)
        )
        
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """使用改进的权重初始化策略"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # 输入到隐藏状态的权重
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # 隐藏状态到隐藏状态的权重
                nn.init.orthogonal_(param.data)  # 正交初始化对RNN特别有用
            elif 'bias' in name:  # 偏置项初始化为0
                param.data.fill_(0)
            elif 'weight' in name and param.dim() > 1:  # 线性层权重
                nn.init.kaiming_normal_(param.data, nonlinearity='relu')  # 适合ReLU激活函数

    def forward(self, x):
        """
        定义LSTM模型的前向传播路径，具有增强的正则化。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_seq_len, input_size)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_seq_len, output_size)。
        """
        # x shape: (batch, input_seq_len, input_size)
        batch_size = x.size(0)
        output_size = self.output_net[-1].out_features // self.output_seq_len

        # 输入归一化
        x_norm = self.input_norm(x)
        
        # 训练模式下添加噪声
        if self.training:
            noise = torch.randn_like(x_norm) * self.noise_level
            x_norm = x_norm + noise
            # 使用特征dropout进行通道级dropout
            x_norm = self.feature_dropout(x_norm.transpose(1, 2)).transpose(1, 2)

        # 第一个LSTM层
        lstm1_out, _ = self.lstm1(x_norm)
        lstm1_out = self.lstm1_norm(lstm1_out)
        
        # 在LSTM层之间应用dropout
        lstm1_out = self.inter_dropout(lstm1_out)
        
        # 第二个LSTM层
        if self.lstm2 is not None:
            lstm_final_output, _ = self.lstm2(lstm1_out)
            lstm_final_output = self.lstm2_norm(lstm_final_output)  # 应用第二层归一化
        else:
            lstm_final_output = lstm1_out

        # 获取最后一个时间步的隐藏状态
        last_step_output = lstm_final_output[:, -1, :] # Shape: (batch, last_lstm_feature_size)

        # 使用MLP网络生成输出
        mlp_output = self.output_net(last_step_output) # Shape: (batch, K * output_size)

        # 重塑输出以匹配 (batch_size, output_seq_len, output_size)
        final_output = mlp_output.view(batch_size, self.output_seq_len, output_size)

        return final_output

# ---------------------------- PureGRU (增强正则化) ----------------------------
class PureGRU(nn.Module):
    """
    优化版GRU模型，添加了额外的正则化层和技术。
    用于Sequence-to-Sequence预测，输入N步，预测未来K步的状态。
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, # output_size 仍然是每步的输出维度 (例如 2)
                 dense_units, dropout):
        super(PureGRU, self).__init__()

        if num_layers < 1: raise ValueError("num_layers 必须至少为 1。")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = config.OUTPUT_SEQ_LEN # 获取预测序列长度 K
        
        # 增强版dropout
        self.dropout_rate = dropout
        self.feature_dropout = nn.Dropout2d(dropout/2)  # 对特征通道随机归零，提供更强的正则化

        # 增强输入归一化
        self.input_norm = nn.LayerNorm(input_size)
        
        # 添加噪声注入层（训练时）
        self.noise_level = 0.01  # 轻微噪声，帮助模型对输入变化更加鲁棒
        
        # 第一个 GRU 层 (双向)
        is_bidirectional = True
        gru1_output_size = hidden_size * 2 if is_bidirectional else hidden_size
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidirectional)
        self.gru1_norm = nn.LayerNorm(gru1_output_size)
        
        # 在GRU层之间添加dropout
        self.inter_dropout = nn.Dropout(dropout)

        # 后续 GRU 层 (单向)
        self.gru2 = None
        last_gru_feature_size = gru1_output_size
        if self.num_layers > 1:
            gru2_num_layers = self.num_layers - 1
            gru2_dropout = dropout if gru2_num_layers > 1 else 0.0
            self.gru2 = nn.GRU(gru1_output_size, hidden_size, num_layers=gru2_num_layers, batch_first=True, dropout=gru2_dropout, bidirectional=False)
            self.gru2_norm = nn.LayerNorm(hidden_size)  # 增加第二个GRU的归一化
            last_gru_feature_size = hidden_size

        # 增强版输出 MLP 网络
        self.output_net = nn.Sequential(
            nn.Linear(last_gru_feature_size, dense_units),
            nn.LayerNorm(dense_units),  # 添加层归一化
            nn.LeakyReLU(),
            nn.Dropout(dropout * 1.2),  # 稍微增加dropout率
            nn.Linear(dense_units, dense_units // 2),
            nn.LayerNorm(dense_units // 2),  # 添加层归一化
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            # 最终线性层输出 K * output_size 个值
            nn.Linear(dense_units // 2, self.output_seq_len * output_size)
        )
        
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """使用改进的权重初始化策略"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # 输入到隐藏状态的权重
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # 隐藏状态到隐藏状态的权重
                nn.init.orthogonal_(param.data)  # 正交初始化对RNN特别有用
            elif 'bias' in name:  # 偏置项初始化为0
                param.data.fill_(0)
            elif 'weight' in name and param.dim() > 1:  # 线性层权重
                nn.init.kaiming_normal_(param.data, nonlinearity='relu')  # 适合ReLU激活函数

    def forward(self, x):
        """
        定义GRU模型的前向传播路径，具有增强的正则化。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_seq_len, input_size)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_seq_len, output_size)。
        """
        batch_size = x.size(0)
        # 确定每个步骤的基本输出大小（例如，theta和theta_dot为2）
        output_size_per_step = self.output_net[-1].out_features // self.output_seq_len

        # 输入归一化
        x_norm = self.input_norm(x)
        
        # 训练模式下添加噪声
        if self.training:
            noise = torch.randn_like(x_norm) * self.noise_level
            x_norm = x_norm + noise
            # 使用特征dropout进行通道级dropout
            x_norm = self.feature_dropout(x_norm.transpose(1, 2)).transpose(1, 2)

        # 第一个GRU层
        gru1_out, _ = self.gru1(x_norm)
        gru1_out = self.gru1_norm(gru1_out)
        
        # 在GRU层之间应用dropout
        gru1_out = self.inter_dropout(gru1_out)
        
        # 第二个GRU层
        if self.gru2 is not None:
            gru_final_output, _ = self.gru2(gru1_out)
            gru_final_output = self.gru2_norm(gru_final_output)  # 应用第二层归一化
        else:
            gru_final_output = gru1_out

        # 获取最后一个时间步的隐藏状态
        last_step_output = gru_final_output[:, -1, :]

        # 使用MLP网络生成输出
        mlp_output = self.output_net(last_step_output)  # 形状: (batch, K * output_size_per_step)

        # 重塑输出以匹配 (batch_size, output_seq_len, output_size_per_step)
        final_output = mlp_output.view(batch_size, self.output_seq_len, output_size_per_step)

        return final_output


# --- 模型工厂函数 (保持不变, 它传递基础的 output_size=2) ---
def get_model(model_type=config.MODEL_TYPE, input_size=3, output_size=2):
    """
    根据指定的类型和配置实例化相应的模型。
    注意: output_size 参数应为 *每个时间步* 的输出维度 (例如 2)。
          模型内部会使用 config.OUTPUT_SEQ_LEN 来调整最终输出层。
    """
    print(f"正在创建模型，类型: {model_type}")
    try:
        params = config.get_current_model_params()
        if model_type.lower() != config.MODEL_TYPE.lower():
             print(f"注意: 请求的模型类型 '{model_type}' 与 config.py 不同。")
             # ... (获取特定类型参数的逻辑保持不变) ...
             temp_config = config.MODEL_PARAMS.get("defaults", {}).copy(); base_type_key = model_type.lower().replace("delta", "pure", 1).replace("seq2seq", "pure", 1); specific_params = config.MODEL_PARAMS.get(base_type_key)
             if specific_params is None: raise ValueError(f"找不到 '{model_type}' 的参数。")
             temp_config.update(specific_params);
             for key, value in config.MODEL_PARAMS.get("defaults", {}).items():
                 if key not in temp_config: temp_config[key] = value
             params = temp_config
             print(f"信息: 使用为 '{model_type}' 定义的参数。")
        else: print(f"信息: 使用 config.py 中为 '{model_type}' 定义的参数。")
    except ValueError as e: print(f"获取模型参数时出错: {e}"); raise
    except Exception as e: print(f"获取模型参数时发生未知错误: {e}"); raise

    model_type_key = model_type.lower()

    # 根据类型确定模型类
    if model_type_key.startswith("purelstm") or model_type_key.startswith("deltalstm") or model_type_key.startswith("seq2seqlstm"):
        model_class = PureLSTM
    elif model_type_key.startswith("puregru") or model_type_key.startswith("deltagru") or model_type_key.startswith("seq2seqgru"):
        model_class = PureGRU
    else: raise ValueError(f"未知的模型类型: {model_type}")

    # 使用从字典中获取的参数实例化模型
    try:
        required_keys = ["hidden_size", "num_layers", "dense_units", "dropout_rate"]
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys: raise KeyError(f"模型类型 '{model_type}' 的配置参数缺失: {missing_keys}")

        print(f"  Instantiating {model_class.__name__} with (per-step output_size={output_size}):") # 打印基础 output_size
        print(f"    input_size={input_size}, hidden_size={params['hidden_size']}, layers={params['num_layers']}, dense_units={params['dense_units']}, dropout={params['dropout_rate']}")

        model = model_class(
            input_size=input_size,
            output_size=output_size, # 传递基础的 output_size
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dense_units=params["dense_units"],
            dropout=params["dropout_rate"]
        )
        return model
    except KeyError as e: raise KeyError(f"实例化模型 '{model_type}' 时缺少参数 {e}")
    except Exception as e: raise RuntimeError(f"实例化模型 '{model_type}' 时出错: {e}")

