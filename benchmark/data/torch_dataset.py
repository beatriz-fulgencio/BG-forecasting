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
import pandas as pd # type: ignore
import torch # type: ignore
from torch.utils.data import Dataset, ConcatDataset  # type: ignore


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
        # Dataset construction must not mutate the caller's preprocessed frame.
        self.df = raw_df.copy()
        
        # Replace missing value markers with NaN
        self.df.replace(to_replace=-1, value=np.nan, inplace=True)
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.unimodal = unimodal
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
               
        # Extract and preprocess data
        self.data = self._extract_features()  # (len, n_features)
        self.valid_sequences = self._find_valid_sequences()
        self._standardize(external_mean, external_std)
        
        print(f"    Dataset loaded: {len(self)} sequences of length {sequence_length}")
        print(f"    Features: {self.feature_columns}")
        
        # Validate no NaN in final data
        self._validate_data()

    def _extract_features(self) -> np.ndarray:
        """Extract feature matrix from DataFrame."""
        feature_data = []
        print(self.df.columns)
        glucose = self.df["glucose"].to_numpy(dtype=np.float32)
        
        if self.unimodal:
            self.feature_columns = ["glucose"]
            return np.array([
                glucose
            ], dtype=np.float32).T
        
        if self.feature_columns is None:
            self.feature_columns = ['glucose', 'basal', 'bolus', 'carbs']
            # Add cyclical time features if present
            for cyc_col in ['hour_sin', 'hour_cos']:
                if cyc_col in self.df.columns:
                    self.feature_columns.append(cyc_col)

        for col in self.feature_columns:
            if col in self.df.columns:
                values = self.df[col].to_numpy(dtype=np.float32)
                
                # Handle special preprocessing for certain features
                if col in ['bolus', 'carbs']:
                    # Fill missing insulin/carbs with 0
                    values[np.isnan(values)] = 0.0
                
                feature_data.append(values)
            else:
                # If column doesn't exist, create zeros
                print(f"    Warning: Column '{col}' not found, using zeros")
                feature_data.append(np.zeros(len(self.df), dtype=np.float32))
        
        return np.array(feature_data, dtype=np.float32).T  # Shape: (time, features)

    @staticmethod
    def str2dt(s):
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    
    def timestamp2dt(self, timestamp: int) -> datetime.datetime:
        """Convert timestamp to datetime."""
        return self.str2dt(str(self.df.index[timestamp]))
    
    def _find_valid_sequences(self) -> List[Tuple[int, int]]:
        """
        Find all valid sequences without missing values.
        
        Returns:
            List of (start_idx, target_idx) tuples for valid sequences
        """
        valid_sequences = []
        total_len = self.data.shape[0]
        required_length = self.sequence_length + self.prediction_horizon  # Full sequence + full target

        print(f"    Finding valid sequences in data of min {required_length} length")

        def check_contiguous_valid(start_idx: int) -> List[Tuple[int, int]]:
            """Check how many valid sequences we can extract starting from start_idx."""
            sequences = []
            end_idx = start_idx
            
            while end_idx < total_len:
                if np.any(np.isnan(self.data[end_idx, :])):
                    break
                
                # Check if we can create a full sequence with full target
                if end_idx - start_idx + 1 >= required_length:
                    # We can create a sequence ending at this point
                    seq_start = end_idx - required_length + 1
                    target_idx = seq_start  # Target starts after sequence
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

        print(f"    Starting standardization...")

        if (external_mean is None) != (external_std is None):
            raise ValueError("external_mean and external_std must be supplied together")

        if external_mean is None and external_std is None:
            print(f"    Using dataset to get mean and std")
            # Valid sequences may coexist with missing values elsewhere in the
            # time series. Exclude those missing values from the normalization
            # statistics so they do not silently disable standardization.
            self.mean = []
            self.std = []
            for i in range(self.data.shape[1]):
                self.mean.append(np.nanmean(self.data[:, i]))
                self.std.append(np.nanstd(self.data[:, i]))
        else:
            print(f"    Using externally given mean and std")
            if len(external_mean) != self.data.shape[1] or len(external_std) != self.data.shape[1]:
                raise ValueError(
                    "external_mean and external_std must match the number of features "
                    f"({self.data.shape[1]})"
                )
            self.mean = list(external_mean)
            self.std = list(external_std)

        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)):
            raise ValueError("normalization mean and std must contain only finite values")
        if any(value < 0 for value in self.std):
            raise ValueError("normalization standard deviations must be non-negative")
        
        # Apply standardization
        for i in range(self.data.shape[1]):
            self.data[:, i] = self.data[:, i] - self.mean[i]
            if self.std[i] > 0:  # Avoid division by zero
                self.data[:, i] = self.data[:, i] / self.std[i]  # Z-score normalization

    def inverse_transform_feature(self, values, feature_name: str):
        """Convert standardized values for one feature to original units."""
        if feature_name not in self.feature_columns:
            raise ValueError(f"Unknown feature: {feature_name}")
        feature_index = self.feature_columns.index(feature_name)
        array = np.asarray(values, dtype=np.float64)
        return array * (self.std[feature_index] or 1.0) + self.mean[feature_index]

    def inverse_transform_target(self, values):
        """Convert standardized glucose targets back to mg/dL."""
        return self.inverse_transform_feature(values, "glucose")

    def inverse_transform_reference(self, values, tolerance: float = 1e-3):
        """Recover the original CGM readings behind standardized targets.
        """
        recovered = self.inverse_transform_target(values)
        nearest = np.round(recovered)
        return np.where(np.abs(recovered - nearest) <= tolerance, nearest, recovered)

    @property
    def target_units(self) -> str:
        """Units of the glucose target returned by ``inverse_transform_target``."""
        return "mg/dL"

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
            target_value = self.data[target_idx + self.sequence_length, target_col_idx]
            return torch.tensor([target_value])  # Keep as 1D tensor for consistency
        else:
            # Multi-step prediction
            target_start = target_idx + self.sequence_length
            target_end = target_start + self.prediction_horizon
            
            # Ensure we don't exceed data bounds
            if target_end > self.data.shape[0]:
                target_end = self.data.shape[0]
                
            target_sequence = self.data[target_start:target_end, target_col_idx]
            
            # Ensure consistent shape even if we're at the end of data
            if len(target_sequence) < self.prediction_horizon:
                # Pad with the last available value
                padding = np.full(self.prediction_horizon - len(target_sequence), target_sequence[-1])
                target_sequence = np.concatenate([target_sequence, padding])
                
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


def prepare_personal_data(patient_data: dict,
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
    train_df = patient_data['train']    
    test_df = patient_data['test']

    return prepare_patient_datasets(
        train_df, test_df, sequence_length, prediction_horizon, unimodal
    )


def prepare_multi_patient_dataset(patient_data: dict,
                                 sequence_length: int = 12,
                                 prediction_horizon: int = 6,
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
        If target_patient_id is None: Combined training-split dataset
        If target_patient_id is specified: (source-training dataset,
        target_train, target_test). No patient test split enters pretraining.
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
        
        # Pretrain only on the other patients' training splits. Their test
        # splits remain held out for their own target-patient evaluations.
        global_datasets = []
        for patient_id, data in patient_data.items():
            if patient_id != target_patient_id:
                if 'train' not in data:
                    raise KeyError(f"Patient {patient_id} has no training split for pretraining")
                dataset = OhioDataset(
                    data['train'],
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
        # Without a target, combine only training splits as well.
        datasets = []
        first_dataset = None
        
        for patient_id, data in patient_data.items():
            if 'train' not in data:
                raise KeyError(f"Patient {patient_id} has no training split for pretraining")
            if first_dataset is None:
                # Use first training dataset for normalization.
                dataset = OhioDataset(
                    data['train'],
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon,
                    unimodal=unimodal
                )
                first_dataset = dataset
                mean, std = dataset.mean, dataset.std
            else:
                dataset = OhioDataset(
                    data['train'],
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon,
                    external_mean=mean,
                    external_std=std,
                    unimodal=unimodal
                )
            datasets.append(dataset)
        
        return ConcatDataset(datasets)
