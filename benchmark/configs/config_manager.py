"""
Configuration management for benchmarking experiments.

This module provides functionality to:
- Load and validate configuration files
- Merge default and custom configurations
- Handle environment-specific settings
- Validate configuration parameters
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Manages configuration loading, validation, and access for benchmarking experiments.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_dir = Path(__file__).parent
        self.default_config_path = self.config_dir / "default.yaml"
        
        # Load default configuration
        self.config = self._load_default_config()
        
        # Load custom configuration if provided
        if config_path:
            custom_config = self._load_config_file(config_path)
            self.config = self._merge_configs(self.config, custom_config)
        
        # Validate configuration
        self._validate_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load the default configuration file."""
        if not self.default_config_path.exists():
            # Create default configuration if it doesn't exist
            default_config = self._get_default_config()
            self._save_config(default_config, self.default_config_path)
            return default_config
        
        return self._load_config_file(str(self.default_config_path))
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration dictionary."""
        return {
            'data': {
                'dataset': 'ohiot1dm',
                'version': '2018',
                'patient_ids': [540, 544, 552, 567, 584, 596],
                'sequence_length': 12,
                'prediction_horizon': 6,
                'batch_size': 32,
                'preprocessing': {
                    'interpolate_missing': True
                }
            },
            'models': {
                'rnn': {
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2
                },
                'lstm': {
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2
                },
                'gru': {
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2
                }
            },
            'training': {
                'epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'early_stopping': {
                    'patience': 10,
                    'min_delta': 0.001
                },
                'transfer_learning': {
                    'enabled': True,
                    'source_patients': [540, 544],
                    'target_patients': [552, 567, 584, 596],
                    'freeze_layers': ['embedding'],
                    'fine_tune_epochs': 25
                }
            },
            'evaluation': {
                'metrics': ['mae', 'rmse', 'mard', 'tir', 'clarke_error_grid', 'parkes_error_grid'],
                'cross_validation': {
                    'enabled': False,
                    'folds': 5
                }
            },
            'paths': {
                'data_root': 'data',
                'processed_data': 'processed_data',
                'results': 'results',
                'saved_models': 'saved_models',
                'logs': 'logs'
            },
            'experiment': {
                'name': 'default_experiment',
                'description': 'Default benchmarking experiment',
                'tags': [],
                'tracking': {
                    'enabled': True,
                    'save_models': True,
                    'save_predictions': True
                }
            },
            'system': {
                'device': 'auto',  # 'auto', 'cpu', 'cuda', 'mps'
                'random_seed': 42,
                'num_workers': 4,
                'precision': 'float32'
            }
        }
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration file {config_path}: {e}")
    
    def _merge_configs(self, base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge custom configuration with base configuration.
        
        Args:
            base_config: Base configuration dictionary
            custom_config: Custom configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in custom_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_sections = ['data', 'models', 'training', 'evaluation', 'paths']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        data_config = self.config['data']
        if 'patient_ids' not in data_config or not data_config['patient_ids']:
            raise ValueError("At least one patient ID must be specified in data.patient_ids")
        
        if 'sequence_length' not in data_config or data_config['sequence_length'] <= 0:
            raise ValueError("data.sequence_length must be a positive integer")
        
        if 'prediction_horizon' not in data_config or data_config['prediction_horizon'] <= 0:
            raise ValueError("data.prediction_horizon must be a positive integer")
        
        # Validate training configuration
        training_config = self.config['training']
        if 'epochs' not in training_config or training_config['epochs'] <= 0:
            raise ValueError("training.epochs must be a positive integer")
        
        
        if 'learning_rate' not in training_config or training_config['learning_rate'] <= 0:
            raise ValueError("training.learning_rate must be positive")
        
        # Validate model configurations
        models_config = self.config['models']
        for model_name, model_config in models_config.items():
            if 'hidden_size' not in model_config or model_config['hidden_size'] <= 0:
                raise ValueError(f"models.{model_name}.hidden_size must be positive")
            
            if 'num_layers' not in model_config or model_config['num_layers'] <= 0:
                raise ValueError(f"models.{model_name}.num_layers must be positive")
    
    def _save_config(self, config: Dict[str, Any], config_path: Path):
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.sequence_length')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.sequence_length')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        return self.config.get('models', {}).get(model_name, {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.config.get('paths', {})
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return self.config.get('experiment', {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self.config.get('system', {})
    
    def save_config(self, output_path: str):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        self._save_config(self.config, Path(output_path))
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config.copy()
    
    def update_from_args(self, args: Dict[str, Any]):
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Dictionary of command-line arguments
        """
        # Map common argument names to configuration keys
        arg_mapping = {
            'batch_size': 'data.batch_size',
            'epochs': 'training.epochs',
            'fine_tune_epochs': 'training.transfer_learning.fine_tune_epochs',
            'learning_rate': 'training.learning_rate',
            'hidden_size': 'models.*.hidden_size',  # Apply to all models
            'num_layers': 'models.*.num_layers',
            'dropout': 'models.*.dropout',
            'device': 'system.device',
            'random_seed': 'system.random_seed',
            'patient_ids': 'data.patient_ids',
            'sequence_length': 'data.sequence_length',
            'prediction_horizon': 'data.prediction_horizon'
        }
        
        for arg_name, value in args.items():
            if value is None:
                continue
                
            if arg_name in arg_mapping:
                config_key = arg_mapping[arg_name]
                
                if config_key.endswith('*.*'):
                    # Apply to all models
                    base_key = config_key.replace('*.', '').replace('.*', '')
                    model_key = config_key.split('*.')[1]
                    
                    for model_name in self.config.get('models', {}):
                        self.set(f'models.{model_name}.{model_key}', value)
                else:
                    self.set(config_key, value)
    
    def create_experiment_config(self, **kwargs) -> Dict[str, Any]:
        """
        Create experiment-specific configuration.
        
        Args:
            **kwargs: Additional experiment parameters
            
        Returns:
            Experiment configuration dictionary
        """
        exp_config = self.config.copy()
        
        # Update with experiment-specific parameters
        for key, value in kwargs.items():
            if '.' in key:
                self.set(key, value)
            else:
                exp_config[key] = value
        
        return exp_config


def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Load configuration manager.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ConfigManager instance
    """
    return ConfigManager(config_path)