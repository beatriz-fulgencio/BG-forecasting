"""
PyTorch Dataset for OhioT1DM blood glucose forecasting.

Adapted from GluPred project for BG-forecasting research.
This module provides PyTorch-compatible dataset classes for training
deep learning models on blood glucose time series data.
"""

import datetime
import os
from typing import Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset


class OhioDataset(Dataset):
    """
    PyTorch Dataset for blood glucose forecasting from OhioT1DM data.
    
    This dataset handles:
    - Loading preprocessed DataFrames
    - Extracting contiguous sequences without missing values
    - Standardization/normalization
    - Feature selection (unimodal vs multimodal)
    - Sequence extraction for time series forecasting
    """

    def __init__(self, 
                 raw_df: pd.DataFrame, 
                 sequence_length: int,
                 prediction_horizon: int = 1,
                 external_mean: Optional[List[float]] = None, 
                 external_std: Optional[List[float]] = None, 
                 unimodal: bool = False,
                 feature_columns: Optional[List[str]] = None):
        """
        Initialize the PyTorch dataset.
        
        Args:
            raw_df: Preprocessed DataFrame with time series data
            sequence_length: Length of input sequences (history)
            prediction_horizon: Number of steps to predict (future)
            external_mean: Pre-computed mean for standardization (if None, compute from data)
            external_std: Pre-computed std for standardization (if None, compute from data)
            unimodal: If True, only use glucose values as features
            feature_columns: List of feature columns to use (if None, auto-detect)
        """
        self.df = raw_df
        
        # Replace missing value markers with NaN
        self.df.replace(to_replace=-1, value=np.nan, inplace=True)
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.unimodal = unimodal
        self.feature_columns = feature_columns
               
        # Extract and preprocess data
        self.data = self._extract_features()  # (len, n_features)
        self.valid_sequences = self._find_valid_sequences()
        self._standardize(external_mean, external_std)
        
        print(f"Dataset loaded: {len(self)} sequences of length {sequence_length}")
        print(f"Features: {self.feature_columns}")
        
        # Validate no NaN in final data
        self._validate_data()

    def _extract_features(self) -> np.ndarray:
        """Extract feature matrix from DataFrame."""
        feature_data = []
        glucose = self.df["glucose"].to_numpy(dtype=np.float32)
        
        if self.unimodal:
            return np.array([
                glucose
            ], dtype=np.float32).T
        
        if self.feature_columns is None:
            self.feature_columns = ['glucose', 'basal', 'bolus', 'carbs']
        
        for col in self.feature_columns:
            if col in self.df.columns:
                values = self.df[col].to_numpy(dtype=np.float32)
                
                # Handle special preprocessing for certain features
                if col in ['bolus', 'carbs']:
                    # Fill missing insulin/carbs with 0
                    values[np.isnan(values)] = 0.0
                
                if col == 'bolus' and 'bolus_dur' in self.df.columns:
                    # Smooth long-acting insulin like in original
                    values = self._smooth_bolus(values, self.df['bolus_dur'].to_numpy(dtype=np.float32))
                
                feature_data.append(values)
            else:
                # If column doesn't exist, create zeros
                print(f"Warning: Column '{col}' not found, using zeros")
                feature_data.append(np.zeros(len(self.df), dtype=np.float32))
        
        return np.array(feature_data, dtype=np.float32).T  # Shape: (time, features)

    @staticmethod
    def str2dt(s):
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    
    def timestamp2dt(self, timestamp: int) -> datetime.datetime:
        """Convert timestamp to datetime."""
        return self.str2dt(str(self.df.index[timestamp]))

    def _smooth_bolus(self, bolus: np.ndarray, bolus_dur: np.ndarray) -> np.ndarray:
        """
        Smooth long-acting insulin.
        Redistribute long-acting insulin over the duration.
        """
        times = [self.timestamp2dt(i) for i, _ in enumerate(self.df.index)]
        bolus = bolus.copy()
        total_len = len(times)
        
        i = 0
        while i < total_len:
            if bolus[i] > 0 and not np.isnan(bolus_dur[i]) and bolus_dur[i] > 0:
                # Found a non-instant bolus
                j = 1
                while i + j < total_len:
                    if bolus[i + j] == bolus[i]:
                        j += 1
                    else:
                        break
                # Distribute the bolus over the duration
                bolus[i: i + j] = bolus[i: i + j] / j
                i += j
            else:
                i += 1
        
        return bolus

    def _find_valid_sequences(self) -> List[Tuple[int, int]]:
        """
        Find all valid sequences without missing values.
        
        Returns:
            List of (start_idx, target_idx) tuples for valid sequences
        """
        valid_sequences = []
        total_len = self.data.shape[0]
        required_length = self.sequence_length + self.prediction_horizon - 1 # For multi-step prediction
        
        def check_contiguous_valid(start_idx: int) -> List[Tuple[int, int]]:
            """Check how many valid sequences we can extract starting from start_idx."""
            sequences = []
            end_idx = start_idx
            
            while end_idx < total_len:
                if np.any(np.isnan(self.data[end_idx, :])):
                    break
                
                # Check if we can create a full sequence
                if end_idx - start_idx + 1 >= required_length:
                    # We can create a sequence ending at this point
                    seq_start = end_idx - required_length + 1
                    target_idx = end_idx - self.prediction_horizon + 1
                    sequences.append((seq_start, target_idx))
                
                end_idx += 1
            
            return sequences, end_idx

        i = 0
        while i < total_len:
            if not np.any(np.isnan(self.data[i, :])):
                sequences, next_i = check_contiguous_valid(i)
                valid_sequences.extend(sequences)
                i = next_i + 1
            else:
                i += 1
        
        return valid_sequences

    def _standardize(self, external_mean: Optional[List[float]] = None, 
                    external_std: Optional[List[float]] = None):
        """Standardize features using z-score normalization."""
        if external_mean is None and external_std is None:
            # Compute statistics from this dataset
            self.mean = []
            self.std = []
            for i in range(self.data.shape[1]):
                self.mean.append(np.mean(self.data[:, i]))
                self.std.append(np.std(self.data[:, i]))
        else:
            self.mean = external_mean
            self.std = external_std
        
        # Apply standardization
        for i in range(self.data.shape[1]):
            if self.std[i] > 0:  # Avoid division by zero
                self.data[:, i] = (self.data[:, i] - self.mean[i]) / self.std[i]

    def _validate_data(self):
        """Ensure no NaN values in the final sequences."""
        for i in range(len(self)):
            sequence = self[i]
            if torch.isnan(sequence).any():
                raise ValueError(f"NaN detected in sequence {i}!")

    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self.valid_sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sequence for training.
        
        Args:
            idx: Sequence index
            
        Returns:
            Tensor of shape (sequence_length, n_features)
        """
        start_idx, target_idx = self.valid_sequences[idx]
        end_idx = start_idx + self.sequence_length
        
        sequence = self.data[start_idx:end_idx, :]
        return torch.from_numpy(sequence)

    def get_target(self, idx: int) -> torch.Tensor:
        """
        Get the target value(s) for a sequence.
        
        Args:
            idx: Sequence index
            
        Returns:
            Target tensor (single value or sequence)
        """
        start_idx, target_idx = self.valid_sequences[idx]
        
        # Find target column index
        target_col_idx = self.feature_columns.index('glucose')
        
        if self.prediction_horizon == 1:
            # Single-step prediction
            target_value = self.data[target_idx + self.sequence_length - 1, target_col_idx]
            return torch.tensor(target_value)
        else:
            # Multi-step prediction
            target_start = target_idx + self.sequence_length - 1
            target_end = target_start + self.prediction_horizon
            target_sequence = self.data[target_start:target_end, target_col_idx]
            return torch.from_numpy(target_sequence)

def prepare_patient_datasets(train_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           sequence_length: int = 12,
                           prediction_horizon: int = 1,
                           unimodal: bool = False,
                           feature_columns: Optional[List[str]] = None) -> Tuple[OhioDataset, OhioDataset]:
    """
    Prepare train and test datasets for a single patient.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame  
        sequence_length: Length of input sequences
        prediction_horizon: Number of steps to predict
        unimodal: Whether to use only glucose data
        feature_columns: List of feature columns to use
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Create training dataset first to get normalization parameters
    train_dataset = OhioDataset(
        train_df, 
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        unimodal=unimodal,
        feature_columns=feature_columns
    )
    
    # Create test dataset using training normalization
    test_dataset = OhioDataset(
        test_df,
        sequence_length=sequence_length, 
        prediction_horizon=prediction_horizon,
        external_mean=train_dataset.mean,
        external_std=train_dataset.std,
        unimodal=unimodal,
        feature_columns=feature_columns
    )
    
    return train_dataset, test_dataset


def prepare_personal_data(train_csv_path: str,
                                       test_csv_path: str,
                                       sequence_length: int = 12,
                                       prediction_horizon: int = 1,
                                       unimodal: bool = False) -> Tuple[OhioDataset, OhioDataset]:
    """
    Prepare datasets from CSV files.
    
    Args:
        train_csv_path: Path to training CSV
        test_csv_path: Path to test CSV
        sequence_length: Length of input sequences
        prediction_horizon: Number of steps to predict
        unimodal: Whether to use only glucose data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_df = pd.read_csv(train_csv_path, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_csv_path, index_col=0, parse_dates=True)
    
    return prepare_patient_datasets(
        train_df, test_df, sequence_length, prediction_horizon, unimodal
    )


def prepare_multi_patient_dataset(patient_data: dict,
                                 sequence_length: int = 12,
                                 prediction_horizon: int = 1,
                                 target_patient_id: Optional[int] = None,
                                 unimodal: bool = False) -> Union[ConcatDataset, Tuple[ConcatDataset, OhioDataset, OhioDataset]]:
    """
    Prepare dataset combining multiple patients for transfer learning.
    
    Args:
        patient_data: Dictionary mapping patient_id -> {'train': df, 'test': df}
        sequence_length: Length of input sequences
        prediction_horizon: Number of steps to predict  
        target_patient_id: If specified, return global dataset + target patient datasets
        unimodal: Whether to use only glucose data
        
    Returns:
        If target_patient_id is None: Combined dataset
        If target_patient_id is specified: (global_dataset, target_train, target_test)
    """
    if target_patient_id is not None:
        # Prepare target patient data first for normalization
        target_train_df = patient_data[target_patient_id]['train']
        target_test_df = patient_data[target_patient_id]['test']
        
        target_train_dataset, target_test_dataset = prepare_patient_datasets(
            target_train_df, target_test_df, sequence_length, prediction_horizon, unimodal
        )
        
        # Use target patient's normalization for all datasets
        mean, std = target_train_dataset.mean, target_train_dataset.std
        
        # Create datasets for all other patients
        global_datasets = []
        for patient_id, data in patient_data.items():
            if patient_id != target_patient_id:
                for split in ['train', 'test']:
                    if split in data:
                        dataset = OhioDataset(
                            data[split],
                            sequence_length=sequence_length,
                            prediction_horizon=prediction_horizon,
                            external_mean=mean,
                            external_std=std,
                            unimodal=unimodal
                        )
                        global_datasets.append(dataset)
        
        global_dataset = ConcatDataset(global_datasets)
        return global_dataset, target_train_dataset, target_test_dataset
    
    else:
        # Just combine all patient data
        datasets = []
        first_dataset = None
        
        for patient_id, data in patient_data.items():
            for split in ['train', 'test']:
                if split in data:
                    if first_dataset is None:
                        # Use first dataset for normalization
                        dataset = OhioDataset(
                            data[split],
                            sequence_length=sequence_length,
                            prediction_horizon=prediction_horizon,
                            unimodal=unimodal
                        )
                        first_dataset = dataset
                        mean, std = dataset.mean, dataset.std
                    else:
                        dataset = OhioDataset(
                            data[split],
                            sequence_length=sequence_length,
                            prediction_horizon=prediction_horizon,
                            external_mean=mean,
                            external_std=std,
                            unimodal=unimodal
                        )
                    datasets.append(dataset)
        
        return ConcatDataset(datasets)
