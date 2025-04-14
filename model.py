# model.py

import torch
import torch.nn as nn
import config # Import config

# --- Residual Block Helper ---
# Optional: Define a reusable residual block class (or implement directly)
class SimpleResidualBlock(nn.Module):
    def __init__(self, feature_size, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.LeakyReLU(), # Or SiLU based on previous choices
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, feature_size) # Linear layer keeps size same
        )

    def forward(self, x):
        return x + self.block(x) # Add input to output of the block

# ------------------------- PureLSTM with Residual MLP -------------------------
class PureLSTM(nn.Module):
    """
    Optimized LSTM model with Residual Connections in the MLP head.
    Predicts K steps of [sin(theta), cos(theta), theta_dot] or other targets.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dense_units, dropout):
        super(PureLSTM, self).__init__()

        if num_layers < 1: raise ValueError("num_layers must be at least 1.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = config.OUTPUT_SEQ_LEN # K
        self.output_features_per_step = output_size # 2 or 3 based on config

        self.dropout_rate = dropout
        self.feature_dropout = nn.Dropout2d(dropout/2)
        self.input_norm = nn.LayerNorm(input_size)
        self.noise_level = 0.01

        # --- RNN Layers ---
        is_bidirectional = True # Keep bidirectional for first layer
        lstm1_output_size = hidden_size * 2 if is_bidirectional else hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidirectional)
        self.lstm1_norm = nn.LayerNorm(lstm1_output_size)
        self.inter_dropout = nn.Dropout(dropout)

        self.lstm2 = None
        last_lstm_feature_size = lstm1_output_size
        if self.num_layers > 1:
             lstm2_num_layers = self.num_layers - 1
             lstm2_dropout = dropout if lstm2_num_layers > 1 else 0.0
             # Input to lstm2 is output of lstm1
             self.lstm2 = nn.LSTM(lstm1_output_size, hidden_size, num_layers=lstm2_num_layers, batch_first=True, dropout=lstm2_dropout, bidirectional=False)
             self.lstm2_norm = nn.LayerNorm(hidden_size)
             last_lstm_feature_size = hidden_size # Output size of the final LSTM layer

        # --- MLP Head with Residual Blocks ---
        # Initial projection layer
        self.mlp_input_layer = nn.Linear(last_lstm_feature_size, dense_units)

        # Define one or more residual blocks
        self.res_block1 = SimpleResidualBlock(dense_units, dropout)
        self.res_block2 = SimpleResidualBlock(dense_units, dropout) # Add a second block

        # Final output layer
        self.mlp_output_layer = nn.Sequential(
            nn.LayerNorm(dense_units), # Normalize before final projection
            nn.LeakyReLU(),
            nn.Linear(dense_units, self.output_seq_len * self.output_features_per_step)
        )
        # --- End of MLP Head Definition ---

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights."""
        for name, param in self.named_parameters():
            if 'lstm' in name or 'gru' in name: # Initialize RNN weights
                 if 'weight_ih' in name: nn.init.xavier_uniform_(param.data)
                 elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
                 elif 'bias' in name: param.data.fill_(0)
            elif 'mlp' in name or 'res_block' in name: # Initialize MLP/ResBlock weights
                 if 'weight' in name and param.dim() > 1: nn.init.kaiming_normal_(param.data, nonlinearity='leaky_relu')
                 elif 'bias' in name: param.data.fill_(0)
            elif 'norm.weight' in name: nn.init.ones_(param.data) # Init norm weights to 1
            elif 'norm.bias' in name: nn.init.zeros_(param.data) # Init norm bias to 0


    def forward(self, x):
        """Forward pass with residual MLP head."""
        batch_size = x.size(0)
        x_norm = self.input_norm(x)
        if self.training:
            noise = torch.randn_like(x_norm) * self.noise_level
            x_norm = x_norm + noise
            x_norm = self.feature_dropout(x_norm.transpose(1, 2)).transpose(1, 2)

        # --- RNN Forward ---
        lstm1_out, _ = self.lstm1(x_norm)
        lstm1_out = self.lstm1_norm(lstm1_out)
        lstm1_out = self.inter_dropout(lstm1_out)

        if self.lstm2 is not None:
            lstm_final_output, _ = self.lstm2(lstm1_out)
            lstm_final_output = self.lstm2_norm(lstm_final_output)
        else:
            lstm_final_output = lstm1_out

        last_step_output = lstm_final_output[:, -1, :] # Get output of last time step

        # --- Residual MLP Forward ---
        x_mlp = self.mlp_input_layer(last_step_output) # Initial projection
        x_mlp = self.res_block1(x_mlp)                 # Apply first residual block
        x_mlp = self.res_block2(x_mlp)                 # Apply second residual block
        mlp_output = self.mlp_output_layer(x_mlp)      # Final projection

        # Reshape output
        final_output = mlp_output.view(batch_size, self.output_seq_len, self.output_features_per_step)
        return final_output

# ---------------------------- PureGRU with Residual MLP ----------------------------
class PureGRU(nn.Module):
    """
    Optimized GRU model with Residual Connections in the MLP head.
    Predicts K steps of [sin(theta), cos(theta), theta_dot] or other targets.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dense_units, dropout):
        super(PureGRU, self).__init__()
        if num_layers < 1: raise ValueError("num_layers must be at least 1.")
        self.hidden_size = hidden_size; self.num_layers = num_layers
        self.output_seq_len = config.OUTPUT_SEQ_LEN # K
        self.output_features_per_step = output_size # 2 or 3

        self.dropout_rate = dropout; self.feature_dropout = nn.Dropout2d(dropout/2)
        self.input_norm = nn.LayerNorm(input_size); self.noise_level = 0.01

        # --- RNN Layers ---
        is_bidirectional = True
        gru1_output_size = hidden_size * 2 if is_bidirectional else hidden_size
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=is_bidirectional)
        self.gru1_norm = nn.LayerNorm(gru1_output_size); self.inter_dropout = nn.Dropout(dropout)

        self.gru2 = None; last_gru_feature_size = gru1_output_size
        if self.num_layers > 1:
            gru2_num_layers = self.num_layers - 1
            gru2_dropout = dropout if gru2_num_layers > 1 else 0.0
            self.gru2 = nn.GRU(gru1_output_size, hidden_size, num_layers=gru2_num_layers, batch_first=True, dropout=gru2_dropout, bidirectional=False)
            self.gru2_norm = nn.LayerNorm(hidden_size)
            last_gru_feature_size = hidden_size

        # --- MLP Head with Residual Blocks ---
        self.mlp_input_layer = nn.Linear(last_gru_feature_size, dense_units)
        self.res_block1 = SimpleResidualBlock(dense_units, dropout)
        self.res_block2 = SimpleResidualBlock(dense_units, dropout)
        self.mlp_output_layer = nn.Sequential(
            nn.LayerNorm(dense_units),
            nn.LeakyReLU(),
            nn.Linear(dense_units, self.output_seq_len * self.output_features_per_step)
        )
        # --- End of MLP Head Definition ---

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights."""
        for name, param in self.named_parameters():
            if 'lstm' in name or 'gru' in name:
                 if 'weight_ih' in name: nn.init.xavier_uniform_(param.data)
                 elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
                 elif 'bias' in name: param.data.fill_(0)
            elif 'mlp' in name or 'res_block' in name:
                 if 'weight' in name and param.dim() > 1: nn.init.kaiming_normal_(param.data, nonlinearity='leaky_relu')
                 elif 'bias' in name: param.data.fill_(0)
            elif 'norm.weight' in name: nn.init.ones_(param.data)
            elif 'norm.bias' in name: nn.init.zeros_(param.data)

    def forward(self, x):
        """Forward pass with residual MLP head."""
        batch_size = x.size(0)
        x_norm = self.input_norm(x)
        if self.training:
            noise = torch.randn_like(x_norm) * self.noise_level
            x_norm = x_norm + noise
            x_norm = self.feature_dropout(x_norm.transpose(1, 2)).transpose(1, 2)

        # --- RNN Forward ---
        gru1_out, _ = self.gru1(x_norm)
        gru1_out = self.gru1_norm(gru1_out); gru1_out = self.inter_dropout(gru1_out)

        if self.gru2 is not None:
            gru_final_output, _ = self.gru2(gru1_out)
            gru_final_output = self.gru2_norm(gru_final_output)
        else: gru_final_output = gru1_out

        last_step_output = gru_final_output[:, -1, :]

        # --- Residual MLP Forward ---
        x_mlp = self.mlp_input_layer(last_step_output)
        x_mlp = self.res_block1(x_mlp)
        x_mlp = self.res_block2(x_mlp)
        mlp_output = self.mlp_output_layer(x_mlp)

        # Reshape output
        final_output = mlp_output.view(batch_size, self.output_seq_len, self.output_features_per_step)
        return final_output


# --- Model Factory Function ---
# (No changes needed here, uses updated classes)
def get_model(model_type=config.MODEL_TYPE, input_size=None, output_size=None):
    """Instantiates the appropriate model based on config settings."""
    print(f"正在创建模型，类型: {model_type}")
    try: params = config.get_current_model_params()
    except Exception as e: print(f"获取模型参数时出错: {e}"); raise

    if input_size is None: input_size = 4 if config.USE_SINCOS_THETA else 3; print(f"  自动检测输入大小: {input_size}")
    if output_size is None: output_size = 3 if config.PREDICT_SINCOS_OUTPUT else 2; print(f"  自动检测输出大小: {output_size}")

    model_type_key = model_type.lower()
    if model_type_key.startswith("purelstm") or model_type_key.startswith("deltalstm") or model_type_key.startswith("seq2seqlstm"): model_class = PureLSTM
    elif model_type_key.startswith("puregru") or model_type_key.startswith("deltagru") or model_type_key.startswith("seq2seqgru"): model_class = PureGRU
    else: raise ValueError(f"未知的模型类型: {model_type}")

    try:
        required_keys = ["hidden_size", "num_layers", "dense_units", "dropout_rate"]
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys: raise KeyError(f"模型类型 '{model_type}' 的配置参数缺失: {missing_keys}")

        print(f"  Instantiating {model_class.__name__} with (per-step output_size={output_size}):")
        print(f"    input_size={input_size}, hidden_size={params['hidden_size']}, layers={params['num_layers']}, dense_units={params['dense_units']}, dropout={params['dropout_rate']:.2f}")

        model = model_class(
            input_size=input_size, output_size=output_size,
            hidden_size=params["hidden_size"], num_layers=params["num_layers"],
            dense_units=params["dense_units"], dropout=params["dropout_rate"]
        )
        return model
    except KeyError as e: raise KeyError(f"实例化模型 '{model_type}' 时缺少参数 {e}")
    except Exception as e: raise RuntimeError(f"实例化模型 '{model_type}' 时出错: {e}")

