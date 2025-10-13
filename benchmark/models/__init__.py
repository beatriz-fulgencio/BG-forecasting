"""
BG-Forecasting Models Module

This module provides standardized model interfaces and implementations
for reproducible comparison of blood glucose prediction approaches.

Main Components:
- base_model: Abstract base classes for all BG forecasting models
- rnn: Recurrent neural network implementations (LSTM, GRU)

Example Usage:
    from benchmark.models import LSTMBGModel
    
    # Create model
    model = LSTMBGModel(
        model_name="lstm_patient_540",
        sequence_length=12,
        prediction_horizon=6,
        feature_dim=5,
        hyperparameters={'hidden_size': 64, 'num_layers': 2}
    )
    
    # Train model
    history = model.fit(train_loader, validation_loader, epochs=100)
    
    # Make predictions
    predictions = model.predict(test_loader)
"""

# Import main classes for easy access
try:
    from .base_model import (
        BaseBGModel,
        BasePyTorchBGModel
    )
    from .rnn import (
        RNNBGModel,
        LSTMBGModel,
        GRUBGModel
    )
    
    __all__ = [
        # Base models
        'BaseBGModel',
        'BasePyTorchBGModel',
        # RNN models
        'RNNBGModel',
        'LSTMBGModel',
        'GRUBGModel'
    ]
    
except ImportError as e:
    # If imports fail, provide informative error
    import warnings
    warnings.warn(f"Could not import all model modules: {e}. "
                 "Please ensure all dependencies are installed.")
    
    __all__ = []