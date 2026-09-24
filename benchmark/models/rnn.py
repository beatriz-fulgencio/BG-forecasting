"""
Deep learning models for blood glucose forecasting.

Implementations of neural network approaches including:
- LSTM/GRU networks
- Transformer models (implemented in ``transformer.py``)
- CNN-based models TODO
- Attention mechanisms TODO
- Hybrid architectures TODO
"""

import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from .base_model import BasePyTorchBGModel

#pyTorch implementation
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
    
    class RNNBGModel(BasePyTorchBGModel):
        """Traditional RNN model for blood glucose forecasting"""
        
        def _build_model(self):
            hidden_size = self.hyperparameters.get("hidden_size", 64)
            num_layers = self.hyperparameters.get("num_layers", 2)
            dropout = self.hyperparameters.get("dropout", 0.2)
            batch_first = self.hyperparameters.get("batch_first", True)
            nonlinearity = self.hyperparameters.get("nonlinearity", "tanh")
            
            # Create Model
            class RNNModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, batch_first, nonlinearity):
                    super().__init__()
                    self.rnn = nn.RNN(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        batch_first=batch_first,
                        nonlinearity=nonlinearity
                    )
                    self.output_layer = nn.Linear(hidden_size, output_size)

                    self.dropout = nn.Dropout(dropout)

                def forward(self, x):
                    rnn_out, _ = self.rnn(x)
                    final_output = rnn_out[:, -1, :]
                    
                    final_output = self.dropout(final_output)
                    predictions = self.output_layer(final_output)
                    return predictions

            # assign RNN model
            self.model = RNNModel(input_size=self.feature_dim,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=dropout,
                                  output_size=self.prediction_horizon,
                                  batch_first=batch_first,
                                  nonlinearity=nonlinearity)

    class LSTMBGModel(BasePyTorchBGModel):
        """ LSTM model for blood glucose forecasting """
        
        def _build_model(self):
            hidden_size = self.hyperparameters.get("hidden_size", 64)
            num_layers = self.hyperparameters.get("num_layers", 2)
            dropout = self.hyperparameters.get("dropout", 0.2)
            batch_first = self.hyperparameters.get("batch_first", True)
            
            # Create Model
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, batch_first):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        batch_first=batch_first
                    )
                    self.output_layer = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    final_output = lstm_out[:, -1, :]
                    predictions = self.output_layer(final_output)
                    return predictions

            # assign LSTM model
            self.model = LSTMModel(input_size=self.feature_dim,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    output_size=self.prediction_horizon,
                                    batch_first=batch_first)
                        
    class GRUBGModel(BasePyTorchBGModel):
        """ GRU model for blood glucose forecasting """
        
        def _build_model(self):
            hidden_size = self.hyperparameters.get("hidden_size", 64)
            num_layers = self.hyperparameters.get("num_layers", 2)
            dropout = self.hyperparameters.get("dropout", 0.2)
            batch_first = self.hyperparameters.get("batch_first", True)
            
            # Create Model
            class GRUModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, batch_first):
                    super().__init__()
                    self.gru = nn.GRU(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        batch_first=batch_first
                    )
                    self.output_layer = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    gru_out, _ = self.gru(x)
                    final_output = gru_out[:, -1, :]
                    predictions = self.output_layer(final_output)
                    return predictions

            # assign GRU model
            self.model = GRUModel(input_size=self.feature_dim,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=dropout,
                                  output_size=self.prediction_horizon,
                                  batch_first=batch_first)

except ImportError:

     # Placeholder classes when PyTorch is not available
    class RNNBGModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for RNNBGModel")
     
    class LSTMBGModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for LSTMBGModel")

    class GRUBGModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for GRUBGModel")
