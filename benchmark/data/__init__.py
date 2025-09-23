"""
BG-Forecasting Data Module

This module provides comprehensive data loading, preprocessing, and validation
utilities for blood glucose forecasting research.

Main Components:
- loaders: Data loading utilities for various BG datasets
- preprocessors: Data preprocessing and feature engineering
- torch_dataset: PyTorch dataset and dataloader utilities for BG forecasting

Supported Datasets:
- OhioT1DM (2018 and 2020 versions)

Example Usage:
    from benchmark.data import load_ohiot1dm_data, preprocess_ohiot1dm_data
    
    # Load data
    data = load_ohiot1dm_data("data", patient_ids=[540, 544])
    
    # Preprocess data
    processed_data = preprocess_ohiot1dm_data(data)
"""

# Import main classes and functions for easy access
try:
    from .loaders import (
        OhioT1DMDataLoader,
        load_ohiot1dm_data
        )
    from .preprocessors import (
        OhioBGDataPreprocessor,
        preprocess_ohiot1dm_data,
        extract_and_save_ohio_data
    )
    from .torch_dataset import (
        OhioDataset,
        prepare_patient_datasets,
        prepare_personal_data,
        prepare_multi_patient_dataset
    )
    
    __all__ = [
        # Loaders
        'OhioT1DMDataLoader',
        'load_ohiot1dm_data',        
        # Preprocessors
        'OhioBGDataPreprocessor',
        'preprocess_ohiot1dm_data',
        'extract_and_save_ohio_data',
        #torch data
        'OhioDataset',
        'prepare_patient_datasets',
        'prepare_personal_data',
        'prepare_multi_patient_dataset'
        
    ]
    
except ImportError as e:
    # If imports fail (e.g., missing dependencies), provide informative error
    import warnings
    warnings.warn(f"Could not import all data modules: {e}. "
                 "Please ensure all dependencies are installed.")
    
    __all__ = []