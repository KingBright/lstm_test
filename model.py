# model.py

import torch
import torch.nn as nn

class PureLSTM(nn.Module):
    """
    A pure LSTM model with optional bidirectional layers, attention, and LayerNorm.
    Designed to predict the next state [theta, theta_dot] based on input sequences.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dense_units, dropout=0.2):
        """
        Initializes the PureLSTM model.

        Args:
            input_size (int): Number of features in the input sequence.
            hidden_size (int): Number of features in the LSTM hidden state.
            num_layers (int): Number of recurrent layers. >= 1.
            output_size (int): Number of output features (e.g., 2 for theta, theta_dot).
            dense_units (int): Number of units in the intermediate dense layer.
            dropout (float): Dropout probability for LSTM layers (if num_layers > 1) and MLP.
        """
        super(PureLSTM, self).__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input Layer Normalization
        self.input_norm = nn.LayerNorm(input_size)

        # First LSTM Layer (potentially bidirectional)
        # We assume bidirectional is generally better for capturing context
        is_bidirectional = True # Hardcoded for now, could be a parameter
        lstm1_output_size = hidden_size * 2 if is_bidirectional else hidden_size
        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1, # First layer is separate
            batch_first=True,
            bidirectional=is_bidirectional
        )
        self.lstm1_norm = nn.LayerNorm(lstm1_output_size)

        # Subsequent LSTM Layers (if num_layers > 1)
        self.lstm2 = None
        if self.num_layers > 1:
             lstm2_num_layers = self.num_layers - 1
             # Apply dropout only if there are multiple subsequent layers
             lstm2_dropout = dropout if lstm2_num_layers > 1 else 0.0
             self.lstm2 = nn.LSTM(
                 lstm1_output_size, # Input comes from previous layer
                 hidden_size,       # Output is standard hidden size (unidirectional)
                 num_layers=lstm2_num_layers,
                 batch_first=True,
                 dropout=lstm2_dropout,
                 bidirectional=False # Subsequent layers typically unidirectional
             )
             self.lstm2_output_size = hidden_size # Output size for attention
        else:
             # If only 1 layer total, attention works on lstm1's output
             self.lstm2_output_size = lstm1_output_size


        # Attention Mechanism
        # Input size to attention depends on the output of the last LSTM layer used
        self.attention = nn.Sequential(
            nn.Linear(self.lstm2_output_size, self.lstm2_output_size // 2), # Reduce complexity slightly
            nn.Tanh(),
            nn.Linear(self.lstm2_output_size // 2, 1)
        )

        # Output MLP Network
        # Takes context from attention (size = self.lstm2_output_size)
        self.output_net = nn.Sequential(
            # Optional: Add LayerNorm before MLP
            # nn.LayerNorm(self.lstm2_output_size),
            nn.Linear(self.lstm2_output_size, dense_units),
            nn.LeakyReLU(),
            nn.Dropout(dropout), # Apply dropout here as well
            nn.Linear(dense_units, dense_units // 2),
            nn.LeakyReLU(),
            # Optional dropout before final layer
            # nn.Dropout(dropout / 2),
            nn.Linear(dense_units // 2, output_size) # Final prediction layer
        )

    def attention_net(self, lstm_output):
        """ Applies the attention mechanism. """
        # lstm_output shape: (batch, seq_len, feature_size)
        # feature_size depends on which LSTM layer's output is used

        # Pass sequence through attention layers
        # attn_weights shape: (batch, seq_len, 1)
        attn_weights = self.attention(lstm_output)

        # Softmax over the sequence length dimension (dim=1)
        # soft_attn_weights shape: (batch, seq_len, 1)
        soft_attn_weights = torch.softmax(attn_weights, dim=1)

        # Compute context vector by weighted sum
        # Multiply element-wise (broadcasts along feature dim)
        # weighted_output shape: (batch, seq_len, feature_size)
        weighted_output = lstm_output * soft_attn_weights

        # Sum across the sequence length dimension
        # context shape: (batch, feature_size)
        context = torch.sum(weighted_output, dim=1)
        return context

    def forward(self, x):
        """
        Defines the forward pass of the PureLSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # x shape: (batch, seq_len, input_size)

        # 1. Input Normalization
        x_norm = self.input_norm(x)

        # 2. First LSTM Layer
        lstm1_out, _ = self.lstm1(x_norm) # lstm1_out shape: (batch, seq_len, hidden_size * 2)
        lstm1_out_norm = self.lstm1_norm(lstm1_out)

        # 3. Subsequent LSTM Layers (if they exist)
        if self.lstm2 is not None:
            lstm2_out, _ = self.lstm2(lstm1_out_norm) # lstm2_out shape: (batch, seq_len, hidden_size)
            last_lstm_output = lstm2_out
        else:
            # If only one layer was specified, use the output of the first (bidirectional) layer
            last_lstm_output = lstm1_out_norm # Shape: (batch, seq_len, hidden_size * 2)

        # 4. Attention Mechanism
        # Input to attention should match the feature size of last_lstm_output
        context = self.attention_net(last_lstm_output) # context shape: (batch, hidden_size or hidden_size*2)

        # 5. Output MLP
        final_output = self.output_net(context) # final_output shape: (batch, output_size)

        return final_output