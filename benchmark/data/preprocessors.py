"""
Data preprocessing utilities for blood glucose forecasting.

This module provides standardized preprocessing functions including:
- Missing data handling
- Outlier detection and treatment
- Feature engineering
- Data normalization and scaling
- Time series windowing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import os

from benchmark.data.loaders import OhioT1DMDataLoader

warnings.simplefilter('ignore', Warning)


class OhioBGDataPreprocessor:
    """
    Comprehensive preprocessor for blood glucose time series data.
    
    This class provides standardized preprocessing steps for blood glucose
    forecasting tasks, ensuring consistent data quality and format across
    different experiments.
    """
    
    def __init__(self, 
                 target_column: str = 'glucose',
                 time_column: str = 'index',
                 sampling_rate: int = 5):
        """
        Initialize the preprocessor.
        
        Args:
            target_column: Name of the glucose column
            time_column: Name of the time index column
            sampling_rate: Sampling rate in minutes
        """
        self.target_column = target_column
        self.time_column = time_column
        self.sampling_rate = sampling_rate
        
        # Scalers for different feature types
        self.glucose_scaler = None
        self.physiological_scaler = None
        self.insulin_scaler = None
        
        # Data quality parameters
        self.glucose_range = (40, 400)  # Valid glucose range in mg/dL
        self.max_gap_minutes = 30  # Maximum acceptable gap in minutes
    
    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply basic preprocessing to raw loaded data.
        
        This method includes all the data alterations that were previously
        done in the loader, including filling missing values, applying temporal
        constraints, and data quality checks.
        
        Args:
            df: Raw loaded DataFrame
            
        Returns:
            Basic preprocessed DataFrame
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Fill missing values with appropriate defaults (same as original loader)
        # Only process columns that exist in the data
        if 'glucose' in df.columns:
            df['glucose'] = df['glucose'].fillna(-1)
        if 'gsr' in df.columns:
            df['gsr'] = df['gsr'].fillna(-1)  # Galvanic Skin Response
        if 'hr' in df.columns:
            df['hr'] = df['hr'].fillna(-1)  # Heart Rate
        if 'st' in df.columns:
            df['st'] = df['st'].fillna(-1)  # Skin Temperature
        if 'basal' in df.columns:
            df['basal'] = df['basal'].fillna(method='ffill')
        
        if 'bolus' in df.columns:
            df['bolus'] = df['bolus'].fillna(-1)
        if 'bolus_dur' in df.columns:
            df['bolus_dur'] = df['bolus_dur'].fillna(-1)
        if 'bolus_end' in df.columns:
            df['bolus_end'] = df['bolus_end'].fillna(-1)
        if 'temp_basal' in df.columns:
            df['temp_basal'] = df['temp_basal'].fillna(-1)
        if 'basal_end' in df.columns:
            df['basal_end'] = df['basal_end'].fillna(-1)
        if 'carbs' in df.columns:
            df['carbs'] = df['carbs'].fillna(-1)
        if 'meal_type' in df.columns:
            df['meal_type'] = df['meal_type'].fillna(-1)
        
        if 'sleep' in df.columns:
            df['sleep'] = df['sleep'].fillna(-1)
        if 'sleep_dur' in df.columns:
            df['sleep_dur'] = df['sleep_dur'].fillna(-1)
        if 'sleep_end' in df.columns:
            df['sleep_end'] = df['sleep_end'].fillna(-1)
        if 'work' in df.columns:
            df['work'] = df['work'].fillna(-1)
        if 'work_dur' in df.columns:
            df['work_dur'] = df['work_dur'].fillna(-1)
        if 'work_end' in df.columns:
            df['work_end'] = df['work_end'].fillna(-1)
        if 'exercise_intensity' in df.columns:
            df['exercise_intensity'] = df['exercise_intensity'].fillna(-1)
        if 'exer_dur' in df.columns:
            df['exer_dur'] = df['exer_dur'].fillna(-1)
        
        # Drop rows with all NaN values
        df = df.dropna(how='all')
        
        # Add helper columns
        df['index'] = df.index
        df['index_new'] = df.index
        
        # Only create temp columns if the original columns exist
        if 'bolus' in df.columns:
            df['temp_bolus'] = df['bolus']
        if 'sleep' in df.columns:
            df['temp_sleep'] = df['sleep']
        if 'work' in df.columns:
            df['temp_work'] = df['work']
        if 'exercise_intensity' in df.columns:
            df['temp_exercise_intensity'] = df['exercise_intensity']
        
        df['missing'] = -1
        
        # Remove duplicate rows by taking max values
        df = df.groupby(df['index_new']).max()
        
        # Apply temporal event processing
        df = self._apply_temporal_events(df)
        
        # Check for missing timesteps
        df = self._check_missing_timesteps(df)
        
        # Clean up temporary columns
        temp_columns = ['temp_basal', 'temp_bolus', 'temp_sleep', 'temp_work', 
                       'temp_exercise_intensity', 'basal_end', 'bolus_end', 
                       'sleep_end', 'work_end', 'exer_dur', 'missing']
        columns_to_drop = [col for col in temp_columns if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        return df
    
    def _apply_temporal_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal event durations to the time series."""
        
        # Apply temporary basal rates -> overwrite existing values
        if 'temp_basal' in df.columns and 'basal_end' in df.columns and 'basal' in df.columns:
            for i in range(len(df)):
                if df['temp_basal'].iloc[i] != -1:
                    basal_end_time = df['basal_end'].iloc[i]
                    mask = (df['index'] >= df['index'].iloc[i]) & (df['index'] <= basal_end_time)
                    df.loc[mask, 'basal'] = df['temp_basal'].iloc[i]
            #drop temporary columns
            df = df.drop(columns=['temp_basal', 'basal_end'], errors='ignore')
        # Apply bolus durations
        if 'temp_bolus' in df.columns and 'bolus_end' in df.columns and 'bolus' in df.columns:
            for i in range(len(df)):
                if df['temp_bolus'].iloc[i] != -1:
                    bolus_end_time = df['bolus_end'].iloc[i]
                    mask = (df['index'] >= df['index'].iloc[i]) & (df['index'] <= bolus_end_time)
                    df.loc[mask, 'bolus'] = df['bolus'].iloc[i]
            #drop temporary columns
            df = df.drop(columns=['temp_bolus', 'bolus_end'], errors='ignore')

        # Apply sleep durations
        if 'temp_sleep' in df.columns and 'sleep_end' in df.columns and 'sleep' in df.columns:
            for i in range(len(df)):
                if df['temp_sleep'].iloc[i] != -1:
                    sleep_end_time = df['sleep_end'].iloc[i]
                    mask = (df['index'] >= df['index'].iloc[i]) & (df['index'] <= sleep_end_time)
                    df.loc[mask, 'sleep'] = df['sleep'].iloc[i]
            #drop temporary columns
            df = df.drop(columns=['temp_sleep', 'sleep_end'], errors='ignore')

        # Apply work durations
        if 'temp_work' in df.columns and 'work_end' in df.columns and 'work' in df.columns:
            for i in range(len(df)):
                if df['temp_work'].iloc[i] != -1:
                    work_end_time = df['work_end'].iloc[i]
                    mask = (df['index'] >= df['index'].iloc[i]) & (df['index'] <= work_end_time)
                    df.loc[mask, 'work'] = df['work'].iloc[i]

            #drop temporary columns
            df = df.drop(columns=['temp_work', 'work_end'], errors='ignore')

        # Apply exercise durations
        if ('temp_exercise_intensity' in df.columns and 'exer_dur' in df.columns and 
            'exercise_intensity' in df.columns):
            for i in range(len(df)):
                if df['temp_exercise_intensity'].iloc[i] != -1:
                    exercise_duration = df['exer_dur'].iloc[i]
                    exercise_start = df['index'].iloc[i]
                    
                    for j in range(i, len(df)):
                        time_diff = (df['index'].iloc[j] - exercise_start).total_seconds() / 60.0
                        if float(exercise_duration) < time_diff:
                            break
                        df.iloc[j, df.columns.get_loc('exercise_intensity')] = df['exercise_intensity'].iloc[i]
            #drop temporary columns
            df = df.drop(columns=['temp_exercise_intensity', 'exer_dur'], errors='ignore')

        return df
    
    def _check_missing_timesteps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for missing timesteps and flag them."""
        for i in range(1, len(df)):
            gap = (df['index'].iloc[i] - df['index'].iloc[i-1]).total_seconds() / 60.0
            if gap != self.sampling_rate:
                df.iloc[i, df.columns.get_loc('missing')] = gap
        
        return df

    def handle_missing_data(self, 
                           df: pd.DataFrame, 
                           strategy: str = 'interpolate',
                           max_gap: int = None) -> pd.DataFrame:
        """
        Handle missing data in the time series.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing data
                     ('interpolate', 'forward_fill', 'drop', 'zero')
            max_gap: Maximum gap size to interpolate (in time steps)
            
        Returns:
            DataFrame with missing data handled
        """
        df = df.copy()
        
        if max_gap is None:
            max_gap = self.max_gap_minutes // self.sampling_rate
        
        if strategy == 'interpolate':
            # Interpolate with limit on gap size
            for col in df.select_dtypes(include=[np.number]).columns:
                if col != self.time_column:
                    df[col] = df[col].interpolate(method='linear', limit=max_gap)
        
        elif strategy == 'forward_fill':
            # Forward fill with limit
            for col in df.select_dtypes(include=[np.number]).columns:
                if col != self.time_column:
                    df[col] = df[col].fillna(method='ffill', limit=max_gap)
        
        elif strategy == 'drop':
            # Drop rows with missing glucose values
            df = df.dropna(subset=[self.target_column])
        
        elif strategy == 'zero':
            # Fill missing values with zero (for insulin, carbs, etc.)
            insulin_cols = [col for col in ['bolus', 'basal'] if col in df.columns]
            lifestyle_cols = [col for col in ['carbs', 'exercise_intensity'] if col in df.columns]
            
            for col in insulin_cols + lifestyle_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        
        return df
    
    def detect_outliers(self, 
                       df: pd.DataFrame, 
                       method: str = 'iqr',
                       columns: List[str] = None) -> pd.DataFrame:
        """
        Detect outliers in the data.
        
        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr', 'z_score', 'isolation_forest')
            columns: Columns to check for outliers
            
        Returns:
            DataFrame with outlier flags
        """
        df = df.copy()
        
        if columns is None:
            columns = [self.target_column, 'hr', 'gsr', 'st']
        
        outlier_flags = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce')
                
                if method == 'iqr':
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = (values < lower_bound) | (values > upper_bound)
                
                elif method == 'z_score':
                    z_scores = np.abs((values - values.mean()) / values.std())
                    outliers = z_scores > 3
                
                outlier_flags[f'{col}_outlier'] = outliers
        
        # Combine outlier flags with original data
        result = pd.concat([df, outlier_flags], axis=1)
        return result
    
    #TODO: verify
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for blood glucose forecasting.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure glucose column is numeric
        if self.target_column in df.columns:
            df[self.target_column] = pd.to_numeric(df[self.target_column], errors='coerce')
        
        # Time-based features
        if self.time_column in df.columns:
            df['hour'] = df[self.time_column].dt.hour
            df['day_of_week'] = df[self.time_column].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Time since midnight
            df['time_since_midnight'] = (
                df['hour'] * 60 + df[self.time_column].dt.minute
            )
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Glucose-derived features
        if self.target_column in df.columns:
            # Glucose rate of change
            df['glucose_diff'] = df[self.target_column].diff()
            df['glucose_diff_2'] = df['glucose_diff'].diff()  # Second derivative
            
            # Rolling statistics
            for window in [3, 6, 12]:  # 15min, 30min, 1hr windows
                df[f'glucose_mean_{window}'] = (
                    df[self.target_column].rolling(window=window, min_periods=1).mean()
                )
                df[f'glucose_std_{window}'] = (
                    df[self.target_column].rolling(window=window, min_periods=1).std()
                )
                df[f'glucose_max_{window}'] = (
                    df[self.target_column].rolling(window=window, min_periods=1).max()
                )
                df[f'glucose_min_{window}'] = (
                    df[self.target_column].rolling(window=window, min_periods=1).min()
                )
        
        # Insulin features
        insulin_cols = ['bolus', 'basal']
        for col in insulin_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Cumulative insulin over different windows
                for window in [6, 12, 24]:  # 30min, 1hr, 2hr windows
                    df[f'{col}_cumsum_{window}'] = (
                        df[col].rolling(window=window, min_periods=1).sum()
                    )
        
        # Meal/carb features
        if 'carbs' in df.columns:
            df['carbs'] = pd.to_numeric(df['carbs'], errors='coerce')
            df['carbs'] = df['carbs'].fillna(0)
            
            # Cumulative carbs over different windows
            for window in [6, 12, 24]:  # 30min, 1hr, 2hr windows
                df[f'carbs_cumsum_{window}'] = (
                    df['carbs'].rolling(window=window, min_periods=1).sum()
                )
        
        # Exercise features
        if 'exercise_intensity' in df.columns:
            df['exercise_intensity'] = pd.to_numeric(df['exercise_intensity'], errors='coerce')
            df['exercise_intensity'] = df['exercise_intensity'].fillna(0)
            
            # Exercise in the last hour
            df['exercise_last_hour'] = (
                df['exercise_intensity'].rolling(window=12, min_periods=1).max()
            )
        
        # Sleep and work stress binary indicators
        for col in ['sleep', 'work']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[f'is_{col}'] = (df[col] > 0).astype(int)
        
        return df
    
    #TODO: verify
    def normalize_features(self, 
                          df: pd.DataFrame, 
                          feature_groups: Dict[str, List[str]] = None,
                          scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Normalize features by groups.
        
        Args:
            df: Input DataFrame
            feature_groups: Dictionary mapping group names to column lists
            scaler_type: Type of scaler ('standard', 'minmax')
            
        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        
        if feature_groups is None:
            feature_groups = {
                'glucose': [self.target_column],
                'physiological': ['hr', 'gsr', 'st'],
                'insulin': ['bolus', 'basal'],
                'meal': ['carbs'],
                'lifestyle': ['exercise_intensity']
            }
        
        # Choose scaler
        if scaler_type == 'standard':
            ScalerClass = StandardScaler
        elif scaler_type == 'minmax':
            ScalerClass = MinMaxScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Apply scaling to each group
        for group_name, columns in feature_groups.items():
            available_cols = [col for col in columns if col in df.columns]
            
            if available_cols:
                scaler = ScalerClass()
                
                # Fit and transform
                df[available_cols] = scaler.fit_transform(
                    df[available_cols].fillna(0)
                )
                
                # Store scaler for later use
                setattr(self, f'{group_name}_scaler', scaler)
        
        return df
    
    #TODO: verify
    def create_sequences(self, 
                        df: pd.DataFrame,
                        sequence_length: int = 12,
                        prediction_horizon: int = 6,
                        step_size: int = 1,
                        target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series forecasting.
        
        Args:
            df: Input DataFrame
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
            step_size: Step size for sliding window
            target_column: Target column name
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if target_column is None:
            target_column = self.target_column
        
        # Ensure target column exists and is numeric
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        df = df.copy()
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        
        # Drop rows with NaN in target column
        df = df.dropna(subset=[target_column])
        
        if len(df) < sequence_length + prediction_horizon:
            raise ValueError("DataFrame too short for specified sequence parameters")
        
        # Select numeric columns for features
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from features to avoid data leakage
        if target_column in feature_columns:
            feature_columns.remove(target_column)
        
        X, y = [], []
        
        for i in range(0, len(df) - sequence_length - prediction_horizon + 1, step_size):
            # Input sequence
            x_seq = df.iloc[i:i + sequence_length][feature_columns].values
            
            # Target sequence (can be single value or multiple)
            if prediction_horizon == 1:
                y_seq = df.iloc[i + sequence_length][target_column]
            else:
                y_seq = df.iloc[i + sequence_length:i + sequence_length + prediction_horizon][target_column].values
            
            # Check for NaN values
            if not (np.isnan(x_seq).any() or np.isnan(y_seq).any()):
                X.append(x_seq)
                y.append(y_seq)
        
        return np.array(X), np.array(y)
    
    def preprocess_patient_data(self, 
                               df: pd.DataFrame,
                               include_feature_engineering: bool = False,
                               normalize: bool = True,
                               handle_missing: str = 'interpolate'
                               ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for a single patient.
        
        Args:
            df: Input DataFrame for one patient (raw or basic preprocessed)
            include_feature_engineering: Whether to engineer additional features
            normalize: Whether to normalize features
            handle_missing: Strategy for missing data
            
        Returns:
            Fully preprocessed DataFrame
        """
        print("Starting preprocessing pipeline...")
        
        # Step 0: Apply basic preprocessing if needed (moved from loader)
        
        print("0. Applying basic preprocessing...")
        df = self.basic_preprocessing(df)
        
        # Step 1: Handle missing data
        print("1. Handling missing data...")
        df = self.handle_missing_data(df, strategy=handle_missing)
        
        # Step 2: Feature engineering
        if include_feature_engineering:
            print("2. Engineering features...")
            df = self.engineer_features(df)

        # Step 3: Detect outliers (but don't remove them yet)
        print("3. Detecting outliers...")
        df = self.detect_outliers(df)

        # Step 4: Normalize features
        if normalize:
            print("4. Normalizing features...")
            df = self.normalize_features(df)
        
        print("Preprocessing completed!")
        return df

def preprocess_ohiot1dm_data(patient_data: Dict[int, pd.DataFrame],
                            target_column: str = 'glucose',
                            **preprocessing_kwargs) -> Dict[int, pd.DataFrame]:
    """
    Preprocess data for multiple patients from OhioT1DM dataset.
    
    Args:
        patient_data: Dictionary mapping patient IDs to raw DataFrames
        target_column: Name of the glucose column
        **preprocessing_kwargs: Additional arguments for preprocessing
        
    Returns:
        Dictionary of preprocessed DataFrames
    """
    preprocessor = OhioBGDataPreprocessor(target_column=target_column)
    preprocessed_data = {}
    
    for patient_id, df in patient_data.items():
        print(f"\n=== Preprocessing patient {patient_id} ===")
        try:
            processed_df = preprocessor.preprocess_patient_data(
                df, 
                **preprocessing_kwargs
            )
            preprocessed_data[patient_id] = processed_df
            print(f"Successfully preprocessed patient {patient_id}")
        except Exception as e:
            print(f"Error preprocessing patient {patient_id}: {e}")
            continue
    
    return preprocessed_data


def extract_and_save_ohio_data(data_dir: str,
                         output_dir: str,
                         patient_ids: Optional[List[int]] = None,
                         modes: List[str] = ['train', 'test'],
                         version: str = '2020',
                         sampling_rate: int = 5,
                         apply_preprocessing: bool = True) -> None:
    """
    Extract data from XML files and save as CSV files.
    
    This function loads raw data and optionally applies preprocessing
    before saving to CSV files.
    
    Args:
        data_dir: Path to the data directory containing XML files
        output_dir: Directory to save extracted CSV files
        patient_ids: List of specific patient IDs to process (None for all)
        modes: List of modes to process ('train', 'test')
        version: Dataset version ('2018' or '2020')
        sampling_rate: Target sampling rate in minutes
        apply_preprocessing: Whether to apply preprocessing before saving
    """
    from .loaders import OhioT1DMDataLoader  # Import here to avoid circular imports
    
    loader = OhioT1DMDataLoader(data_dir, sampling_rate, version)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if patient_ids is None:
        patient_ids = loader.patient_ids[version]
    
    if apply_preprocessing:
        preprocessor = OhioBGDataPreprocessor()
    
    for mode in modes:
        print(f"\n=== Processing {mode} data for version {version} ===")
        
        for patient_id in patient_ids:
            try:
                print(f"\nProcessing patient {patient_id}...")
                df = loader.load_patient_data(patient_id, mode)
                
                if not df.empty:
                    if apply_preprocessing:
                        # Apply basic preprocessing
                        df = preprocessor.basic_preprocessing(df)
                    
                    suffix = "_processed" if apply_preprocessing else "_raw"
                    output_file = os.path.join(output_dir, f"{patient_id}_{mode}{suffix}.csv")
                    df.to_csv(output_file)
                    print(f"Saved data for patient {patient_id} to {output_file}")
                else:
                    print(f"No data available for patient {patient_id}")
                    
            except FileNotFoundError as e:
                print(f"Warning: Could not process patient {patient_id}: {e}")
                continue