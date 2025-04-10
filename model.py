# model.py

import torch
import torch.nn as nn
import config # 导入配置以方便访问默认参数或类型信息

# ------------------------- PureLSTM (无 Attention) -------------------------
class PureLSTM(nn.Module):
    """
    一个纯 LSTM 模型 (无 Attention)，可选第一层双向。
    使用最后一个时间步的输出来进行预测。
    """
    # 参数在实例化时通过 get_model 工厂函数传递
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dense_units, dropout):
        super(PureLSTM, self).__init__()

        if num_layers < 1: raise ValueError("num_layers 必须至少为 1。")
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 输入层归一化 (Layer Normalization)
        self.input_norm = nn.LayerNorm(input_size)

        # 第一个 LSTM 层 (双向)
        is_bidirectional = True # 保持第一层双向
        lstm1_output_size = hidden_size * 2 if is_bidirectional else hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, # 第一层独立
                             batch_first=True, bidirectional=is_bidirectional)
        self.lstm1_norm = nn.LayerNorm(lstm1_output_size) # 对双向输出进行归一化

        # 后续 LSTM 层 (单向)
        self.lstm2 = None
        last_lstm_feature_size = lstm1_output_size # 如果只有一层，则以此为准
        if self.num_layers > 1:
             lstm2_num_layers = self.num_layers - 1
             # 仅当后续层数大于1时应用 dropout
             lstm2_dropout = dropout if lstm2_num_layers > 1 else 0.0
             self.lstm2 = nn.LSTM(
                 lstm1_output_size, # 输入来自上一层的输出
                 hidden_size,       # 输出是标准的隐藏层大小 (单向)
                 num_layers=lstm2_num_layers,
                 batch_first=True,
                 dropout=lstm2_dropout,
                 bidirectional=False # 后续层通常是单向的
             )
             last_lstm_feature_size = hidden_size # MLP 的输入大小是最后一个单向 LSTM 的输出大小

        # 输出 MLP 网络
        # 输入大小取决于最后一个 LSTM 层的特征大小
        self.output_net = nn.Sequential(
            nn.Linear(last_lstm_feature_size, dense_units),
            nn.LeakyReLU(),
            nn.Dropout(dropout), # 在 MLP 中也应用 Dropout
            nn.Linear(dense_units, dense_units // 2),
            nn.LeakyReLU(),
            nn.Linear(dense_units // 2, output_size) # 最终预测层
        )

    def forward(self, x):
        """
        定义 PureLSTM 模型的前向传播路径。
        """
        # x shape: (batch, seq_len, input_size)
        x_norm = self.input_norm(x)

        # 第一个 LSTM 层
        lstm1_out, _ = self.lstm1(x_norm) # lstm1_out shape: (batch, seq_len, hidden_size * 2)
        lstm1_out_norm = self.lstm1_norm(lstm1_out)

        # 后续 LSTM 层
        if self.lstm2 is not None:
            lstm_final_layer_output, _ = self.lstm2(lstm1_out_norm) # (batch, seq_len, hidden_size)
        else:
            # 如果总共只有一层，则使用第一层的输出
            lstm_final_layer_output = lstm1_out_norm # (batch, seq_len, hidden_size * 2)

        # 获取最后一个时间步的输出
        # shape: (batch, last_lstm_feature_size)
        last_step_output = lstm_final_layer_output[:, -1, :]

        # MLP 输出
        final_output = self.output_net(last_step_output) # (batch, output_size)

        return final_output

# ---------------------------- PureGRU (无 Attention) ----------------------------
class PureGRU(nn.Module):
    """
    一个纯 GRU 模型 (无 Attention)，结构与 PureLSTM 类似。
    使用最后一个时间步的输出来进行预测。
    """
    # 参数在实例化时通过 get_model 工厂函数传递
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dense_units, dropout):
        super(PureGRU, self).__init__()

        if num_layers < 1: raise ValueError("num_layers 必须至少为 1。")
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 输入层归一化
        self.input_norm = nn.LayerNorm(input_size)

        # 第一个 GRU 层 (双向)
        is_bidirectional = True
        gru1_output_size = hidden_size * 2 if is_bidirectional else hidden_size
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers=1, # 使用 GRU
                           batch_first=True, bidirectional=is_bidirectional)
        self.gru1_norm = nn.LayerNorm(gru1_output_size)

        # 后续 GRU 层 (单向)
        self.gru2 = None
        last_gru_feature_size = gru1_output_size
        if self.num_layers > 1:
            gru2_num_layers = self.num_layers - 1
            gru2_dropout = dropout if gru2_num_layers > 1 else 0.0
            self.gru2 = nn.GRU(gru1_output_size, hidden_size, # 使用 GRU
                               num_layers=gru2_num_layers, batch_first=True,
                               dropout=gru2_dropout, bidirectional=False)
            last_gru_feature_size = hidden_size

        # 输出 MLP 网络 (与 PureLSTM 的 output_net 结构相同)
        self.output_net = nn.Sequential(
            nn.Linear(last_gru_feature_size, dense_units),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units // 2),
            nn.LeakyReLU(),
            nn.Linear(dense_units // 2, output_size)
        )

    def forward(self, x):
        """
        定义 PureGRU 模型的前向传播路径。
        """
        x_norm = self.input_norm(x)

        # 第一个 GRU 层
        gru1_out, _ = self.gru1(x_norm) # GRU 返回 output, h_n
        gru1_out_norm = self.gru1_norm(gru1_out)

        # 后续 GRU 层
        if self.gru2 is not None:
            gru_final_layer_output, _ = self.gru2(gru1_out_norm)
        else:
            gru_final_layer_output = gru1_out_norm

        # 获取最后一个时间步的输出
        last_step_output = gru_final_layer_output[:, -1, :]

        # MLP 输出
        final_output = self.output_net(last_step_output)
        return final_output


# --- 模型工厂函数 ---
def get_model(model_type=config.MODEL_TYPE, input_size=3, output_size=2):
    """
    根据指定的类型和配置实例化相应的模型。

    Args:
        model_type (str): 模型的类型字符串 (例如 "PureLSTM", "DeltaGRU")。
        input_size (int): 模型的输入特征维度。
        output_size (int): 模型的输出维度。

    Returns:
        torch.nn.Module: 实例化后的模型对象。

    Raises:
        ValueError: 如果模型类型未知或配置参数缺失。
        RuntimeError: 如果模型实例化过程中发生错误。
    """
    print(f"正在创建模型，类型: {model_type}")
    # 从 config 获取当前选定类型的参数
    try:
        params = config.get_current_model_params() # 获取 config.MODEL_TYPE 对应的参数
        # 如果函数调用时显式传入了不同的 model_type，则获取该类型的参数
        if model_type.lower() != config.MODEL_TYPE.lower():
             print(f"注意: 请求的模型类型 '{model_type}' 与 config.py 中的 '{config.MODEL_TYPE}' 不同。")
             temp_config = config.MODEL_PARAMS.get("defaults", {}).copy()
             model_type_key = model_type.lower()
             base_type_key = model_type_key.replace("delta", "pure", 1) if model_type_key.startswith("delta") else model_type_key
             specific_params = config.MODEL_PARAMS.get(base_type_key)
             if specific_params is None:
                  raise ValueError(f"在 config.py 中找不到显式请求的模型类型 '{model_type}' (或其基础类型 '{base_type_key}') 的参数。")
             temp_config.update(specific_params)
             params = temp_config # 使用显式请求的参数
             print(f"信息: 使用为 '{model_type}' (基础类型 '{base_type_key}') 定义的参数进行实例化。")

    except ValueError as e:
        print(f"从配置中获取模型参数时出错: {e}")
        raise

    model_type_key = model_type.lower()

    # 根据类型确定模型类
    if model_type_key.startswith("purelstm") or model_type_key.startswith("deltalstm"):
        model_class = PureLSTM
    elif model_type_key.startswith("puregru") or model_type_key.startswith("deltagru"):
        model_class = PureGRU
    # 在此添加其他模型类型
    # elif model_type_key == "seq2seqlstm":
    #     model_class = Seq2SeqLSTM # 假设存在这个类
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    # 使用从字典中获取的参数实例化模型
    try:
        # 确保所有必需的参数都在字典中
        required_keys = ["hidden_size", "num_layers", "dense_units", "dropout_rate"]
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
             raise KeyError(f"模型类型 '{model_type}' 的配置参数缺失: {missing_keys}")

        model = model_class(
            input_size=input_size,
            output_size=output_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dense_units=params["dense_units"],
            dropout=params["dropout_rate"]
        )
        return model
    except KeyError as e:
        # 上面的检查应该能捕捉到，但为了安全起见保留
        raise KeyError(f"实例化模型 '{model_type}' 时缺少参数 {e}")
    except Exception as e:
        raise RuntimeError(f"实例化模型 '{model_type}' 时出错: {e}")

