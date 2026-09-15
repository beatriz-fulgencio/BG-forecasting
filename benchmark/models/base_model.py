"""
Base model interface for blood glucose forecasting.

This module defines the standardized interface that all benchmark models
must implement to ensure consistent evaluation and comparison.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
import pandas as pd  # type: ignore
from pathlib import Path
import pickle
import json
import copy
from datetime import datetime

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.base import BaseEstimator  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import evaluation module components
try:
    from ..evaluation.metrics import BGEvaluator
    EVALUATION_MODULE_AVAILABLE = True
except ImportError:
    EVALUATION_MODULE_AVAILABLE = False


class BaseBGModel(ABC):
    """
    Abstract base class for blood glucose forecasting models.
    
    This class defines the standard interface that all models must implement
    to ensure consistent evaluation and comparison across different approaches.
    
    Supports:
    - PyTorch neural networks
    - Scikit-learn models  
    - Custom implementations
    - Model serialization/deserialization
    - Hyperparameter management
    """
    
    def __init__(self, 
                 model_name: str,
                 sequence_length: int = 12,
                 prediction_horizon: int = 1,
                 feature_dim: int = 1,
                 hyperparameters: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        Initialize the base model.
        
        Args:
            model_name: Unique identifier for the model
            sequence_length: Input sequence length (e.g., 12 for 1 hour at 5min intervals)
            prediction_horizon: Number of steps to predict ahead
            feature_dim: Number of input features (1 for glucose-only, >1 for multivariate)
            hyperparameters: Model-specific hyperparameters
            device: PyTorch device ('cuda', 'cpu', 'mps' or None for auto-detection)
        """
        self.model_name = model_name
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_dim = feature_dim
        self.hyperparameters = hyperparameters or {}
        
        # Training state
        self.is_fitted = False
        self.training_history = {}
        self.model_metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': self.__class__.__name__,
            'version': '1.0'
        }
        
        # PyTorch specific
        if TORCH_AVAILABLE:
            self.device = self._get_device(device)
        else:
            self.device = None
            
        # Initialize model architecture
        self._build_model()
    
    @abstractmethod
    def _build_model(self) -> None:
        """
        Build the model architecture.
        
        This method should initialize the underlying model (PyTorch nn.Module,
        sklearn estimator, etc.) based on the specified hyperparameters.
        """
        pass
    
    @abstractmethod
    def fit(self, 
            train_data: Union[DataLoader, np.ndarray, pd.DataFrame],
            validation_data: Optional[Union[DataLoader, np.ndarray, pd.DataFrame]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            train_data: Training data (DataLoader for PyTorch, arrays for sklearn)
            validation_data: Optional validation data for early stopping
            **kwargs: Additional training parameters (epochs, batch_size, etc.)
            
        Returns:
            Dictionary containing training metrics and history
        """
        pass
    
    @abstractmethod
    def predict(self, 
                data: Union[DataLoader, np.ndarray, pd.DataFrame],
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on new data.
        
        Args:
            data: Input data for prediction
            return_uncertainty: Whether to return prediction uncertainty (if supported)
            
        Returns:
            Predictions array, optionally with uncertainty estimates
        """
        pass
    
    def validate(self, 
                 test_data: Union[DataLoader, np.ndarray, pd.DataFrame],
                 evaluator=None,
                 metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Validate model performance on test data using the evaluation module.
        
        Args:
            test_data: Test data for validation
            evaluator: Instance of evaluation.metrics.BGEvaluator (if None, creates default one)
            metrics: List of metrics to compute (passed to evaluator)
            
        Returns:
            Dictionary of computed metrics
        """
        # Get predictions
        if hasattr(self, 'predict_with_uncertainty'):
            predictions, uncertainty = self.predict(test_data, return_uncertainty=True)
        else:
            predictions = self.predict(test_data)
            uncertainty = None
        
        # Extract true values
        y_true = self._extract_targets(test_data)
        
        # Use evaluation module for metrics computation
        if evaluator is not None:
            # Use provided evaluator
            results = evaluator.compute_metrics(
                y_true=y_true,
                y_pred=predictions,
                uncertainty=uncertainty,
                metrics=metrics
            )
        elif EVALUATION_MODULE_AVAILABLE:
            # Create default evaluator from evaluation module
            default_evaluator = BGEvaluator()
            results = default_evaluator.compute_metrics(
                y_true=y_true,
                y_pred=predictions,
                uncertainty=uncertainty,
                metrics=metrics
            )
            
        return results

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model_name': self.model_name,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'feature_dim': self.feature_dim,
            'hyperparameters': self.hyperparameters,
            'training_history': self.training_history,
            'model_metadata': self.model_metadata,
            'is_fitted': self.is_fitted
        }
        
        # Save model-specific components
        if hasattr(self, 'model') and TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            # PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_data': model_data
            }, filepath)
            
        elif hasattr(self, 'model') and SKLEARN_AVAILABLE and hasattr(self.model, 'get_params'):
            # Scikit-learn model
            model_data['model'] = self.model
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        else:
            # Custom model - subclasses should override this method
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'BaseBGModel':
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        filepath = Path(filepath)
        
        if TORCH_AVAILABLE and filepath.suffix in ['.pt', '.pth']:
            # PyTorch model
            checkpoint = torch.load(filepath, map_location='cpu')
            model_data = checkpoint['model_data']
            
            # Recreate model instance
            instance = cls(
                model_name=model_data['model_name'],
                sequence_length=model_data['sequence_length'],
                prediction_horizon=model_data['prediction_horizon'],
                feature_dim=model_data['feature_dim'],
                hyperparameters=model_data['hyperparameters']
            )
            
            # Load state
            instance.model.load_state_dict(checkpoint['model_state_dict'])
            instance.training_history = model_data['training_history']
            instance.model_metadata = model_data['model_metadata']
            instance.is_fitted = model_data['is_fitted']
            
            return instance
        else:
            # Pickle-based loading
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Recreate instance and restore state
            instance = cls(
                model_name=model_data['model_name'],
                sequence_length=model_data['sequence_length'],
                prediction_horizon=model_data['prediction_horizon'],
                feature_dim=model_data['feature_dim'],
                hyperparameters=model_data['hyperparameters']
            )
            
            # Restore additional attributes
            for key, value in model_data.items():
                if key not in ['model_name', 'sequence_length', 'prediction_horizon', 
                              'feature_dim', 'hyperparameters']:
                    setattr(instance, key, value)
            
            return instance
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return self.hyperparameters.copy()
    
    def set_hyperparameters(self, **kwargs) -> None:
        """
        Update hyperparameters and rebuild model if necessary.
        
        Args:
            **kwargs: Hyperparameter updates
        """
        self.hyperparameters.update(kwargs)
        if self.is_fitted:
            print("Warning: Updating hyperparameters on fitted model. Consider retraining.")
    
    def _get_device(self, device: Optional[str] = None) -> str:
        """Get appropriate device for PyTorch models."""
        if not TORCH_AVAILABLE:
            return 'cpu'
        
        if device is not None:
            return device

        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _extract_targets(self, data: Union[DataLoader, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Extract target values from various data formats.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Target values as numpy array
        """
        if TORCH_AVAILABLE and isinstance(data, DataLoader):
            targets = []
            for batch in data:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    _, y = batch[0], batch[1]
                    targets.append(y.cpu().numpy() if hasattr(y, 'cpu') else y)
            return np.concatenate(targets) if targets else np.array([])
        
        elif isinstance(data, np.ndarray):
            # Assume last column(s) are targets
            if data.ndim == 2:
                return data[:, -self.prediction_horizon:]
            return data
        
        elif isinstance(data, pd.DataFrame):
            # Assume glucose column is target
            if 'glucose' in data.columns:
                return data['glucose'].values
            return data.iloc[:, -1].values  # Last column as fallback
        
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"{self.__class__.__name__}(name='{self.model_name}', "
                f"seq_len={self.sequence_length}, pred_horizon={self.prediction_horizon}, "
                f"features={self.feature_dim}, status={status})")


class BasePyTorchBGModel(BaseBGModel):
    """
    Base class specifically for PyTorch-based blood glucose models.
    
    Provides common PyTorch functionality like training loops, device management, and standard neural network operations.
    """
    
    def __init__(self, *args, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchBGModel. Install with: pip install torch")
        
        # Initialize PyTorch-specific attributes first
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        
        # Call parent constructor (which will call _build_model)
        super().__init__(*args, **kwargs)
        
    def fit(self, 
            train_loader: DataLoader,
            validation_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            learning_rate: float = 0.001,
            early_stopping_patience: int = 10,
            **kwargs) -> Dict[str, Any]:
        """
        Train the PyTorch model.
        
        Args:
            train_loader: Training data loader
            validation_loader: Optional validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Patience for early stopping
            **kwargs: Additional training parameters
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call _build_model() first.")
        
        # Setup training components
        self.model.to(self.device)

        # Use Adam optimizer as default
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Use MSE loss as default
        if self.criterion is None:
            self.criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(self.device) # Move input to device
                targets = targets.to(self.device) # Move target to device
                
                # Forward pass
                self.optimizer.zero_grad() # Clear gradients
                predictions = self.model(sequences) # Make predictions
                loss = self.criterion(predictions, targets) # Compute loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step() # Update model parameters

                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if validation_loader is not None:
                val_loss = self._validate_epoch(validation_loader)
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}")
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_completed': len(train_losses),
            'best_val_loss': best_val_loss
        }
        
        return self.training_history
    
    def predict(self, 
                data_loader: DataLoader,
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the trained model.
        
        Args:
            data_loader: Input data loader
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions array, optionally with uncertainty
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad(): # Disable gradient tracking
            for sequences, _ in data_loader:
                sequences = sequences.to(self.device)
                batch_pred = self.model(sequences)
                predictions.append(batch_pred.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        if return_uncertainty:
            # Placeholder for uncertainty - subclasses can override
            uncertainty = np.zeros_like(predictions)
            return predictions, uncertainty
        
        return predictions
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate model for one epoch."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad(): # Disable gradient tracking
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(sequences)
                loss = self.criterion(predictions, targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
