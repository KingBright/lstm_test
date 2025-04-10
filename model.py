# model.py

import torch
import torch.nn as nn
import config # 导入配置

# ------------------------- PureLSTM (无 Attention, Seq2Seq 输出) -------------------------
class PureLSTM(nn.Module):
    """
    一个纯 LSTM 模型 (无 Attention)，用于 Sequence-to-Sequence 预测。
    输入 N 步，预测未来 K 步的状态。
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, # output_size 仍然是每步的输出维度 (例如 2)
                 dense_units, dropout):
        super(PureLSTM, self).__init__()

        if num_layers < 1: raise ValueError("num_layers 必须至少为 1。")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = config.OUTPUT_SEQ_LEN # 获取预测序列长度 K

        self.input_norm = nn.LayerNorm(input_size)

        # 第一个 LSTM 层 (双向)
        is_bidirectional = True
        lstm1_output_size = hidden_size * 2 if is_bidirectional else hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidirectional)
        self.lstm1_norm = nn.LayerNorm(lstm1_output_size)

        # 后续 LSTM 层 (单向)
        self.lstm2 = None
        last_lstm_feature_size = lstm1_output_size
        if self.num_layers > 1:
             lstm2_num_layers = self.num_layers - 1
             lstm2_dropout = dropout if lstm2_num_layers > 1 else 0.0
             self.lstm2 = nn.LSTM(lstm1_output_size, hidden_size, num_layers=lstm2_num_layers, batch_first=True, dropout=lstm2_dropout, bidirectional=False)
             last_lstm_feature_size = hidden_size

        # 输出 MLP 网络
        self.output_net = nn.Sequential(
            nn.Linear(last_lstm_feature_size, dense_units),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units // 2),
            nn.LeakyReLU(),
            # 最终线性层输出 K * output_size 个值
            nn.Linear(dense_units // 2, self.output_seq_len * output_size)
        )

    def forward(self, x):
        """
        定义 Seq2Seq LSTM 模型的前向传播路径。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_seq_len, input_size)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_seq_len, output_size)。
        """
        # x shape: (batch, input_seq_len, input_size)
        batch_size = x.size(0)
        output_size = config.MODEL_PARAMS.get(config.MODEL_TYPE.lower().replace("seq2seq","pure"), {}).get("output_size", 2) # Get base output size (2)

        x_norm = self.input_norm(x)

        # LSTM 层
        lstm1_out, _ = self.lstm1(x_norm)
        lstm1_out_norm = self.lstm1_norm(lstm1_out)
        if self.lstm2 is not None:
            lstm_final_layer_output, _ = self.lstm2(lstm1_out_norm)
        else:
            lstm_final_layer_output = lstm1_out_norm

        # 获取最后一个时间步的隐藏状态输出
        last_step_output = lstm_final_layer_output[:, -1, :] # Shape: (batch, last_lstm_feature_size)

        # MLP 输出
        # mlp_output shape: (batch, K * output_size)
        mlp_output = self.output_net(last_step_output)

        # Reshape 输出以匹配 (batch_size, output_seq_len, output_size)
        # output_size is the number of features per step (e.g., 2 for theta, dot)
        final_output = mlp_output.view(batch_size, self.output_seq_len, output_size)

        return final_output

# ---------------------------- PureGRU (无 Attention, Seq2Seq 输出) ----------------------------
class PureGRU(nn.Module):
    """
    一个纯 GRU 模型 (无 Attention)，用于 Sequence-to-Sequence 预测。
    输入 N 步，预测未来 K 步的状态。
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, # output_size 仍然是每步的输出维度 (例如 2)
                 dense_units, dropout):
        super(PureGRU, self).__init__()

        if num_layers < 1: raise ValueError("num_layers 必须至少为 1。")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = config.OUTPUT_SEQ_LEN # 获取预测序列长度 K

        self.input_norm = nn.LayerNorm(input_size)

        # 第一个 GRU 层 (双向)
        is_bidirectional = True
        gru1_output_size = hidden_size * 2 if is_bidirectional else hidden_size
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidirectional)
        self.gru1_norm = nn.LayerNorm(gru1_output_size)

        # 后续 GRU 层 (单向)
        self.gru2 = None
        last_gru_feature_size = gru1_output_size
        if self.num_layers > 1:
            gru2_num_layers = self.num_layers - 1
            gru2_dropout = dropout if gru2_num_layers > 1 else 0.0
            self.gru2 = nn.GRU(gru1_output_size, hidden_size, num_layers=gru2_num_layers, batch_first=True, dropout=gru2_dropout, bidirectional=False)
            last_gru_feature_size = hidden_size

        # 输出 MLP 网络
        self.output_net = nn.Sequential(
            nn.Linear(last_gru_feature_size, dense_units),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units // 2),
            nn.LeakyReLU(),
            # 最终线性层输出 K * output_size 个值
            nn.Linear(dense_units // 2, self.output_seq_len * output_size)
        )

    def forward(self, x):
        """
        定义 Seq2Seq GRU 模型的前向传播路径。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_seq_len, input_size)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_seq_len, output_size)。
        """
        batch_size = x.size(0)
        # Determine base output size per step (e.g., 2 for theta, dot)
        # This assumes the 'output_size' passed during construction is per-step size
        # Find the output_size from the last layer's expected total output / K
        output_size_per_step = self.output_net[-1].out_features // self.output_seq_len

        x_norm = self.input_norm(x)

        # GRU 层
        gru1_out, _ = self.gru1(x_norm)
        gru1_out_norm = self.gru1_norm(gru1_out)
        if self.gru2 is not None:
            gru_final_layer_output, _ = self.gru2(gru1_out_norm)
        else:
            gru_final_layer_output = gru1_out_norm

        # 获取最后一个时间步的隐藏状态输出
        last_step_output = gru_final_layer_output[:, -1, :]

        # MLP 输出
        mlp_output = self.output_net(last_step_output) # Shape: (batch, K * output_size_per_step)

        # Reshape 输出以匹配 (batch_size, output_seq_len, output_size_per_step)
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

