"""
Data preprocessing utilities for blood glucose forecasting.

This module provides standardized preprocessing functions including:
- Missing data handling
- Feature engineering
- Time series windowing
"""

import pandas as pd  # type: ignore
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os


class OhioBGDataPreprocessor:
    """
    Comprehensive preprocessor for blood glucose time series data.
    
    This class provides standardized preprocessing steps for blood glucose
    forecasting tasks, ensuring consistent data quality and format across
    different experiments.
    
    This class works specificaly for OhioT1DM database.
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
        
        # Data quality parameters
        self.glucose_range = (40, 400)  # Valid glucose range in mg/dL 
        self.max_gap_minutes = 15  # Maximum acceptable gap in minutes
    
    def basic_preprocessing(self, df: pd.DataFrame):
        """
        Apply basic preprocessing to raw loaded data.
        
        Args:
            df: Raw loaded DataFrame
            
        Returns:
            Basic preprocessed DataFrame
        """
        if df.empty:
            print(f"    [BASIC] Input DataFrame is empty, skipping preprocessing")
            return df

        print(f"    [BASIC] Starting basic preprocessing on {len(df)} rows")
        df = df.copy()
        
        # Fill missing values with appropriate defaults 
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
            df['basal'] = df['basal'].ffill()
        
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
        df = df.dropna()
        
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
    
    def _apply_temporal_events(self, df: pd.DataFrame):
        """Apply temporal event durations to the time series."""
        
        print(f"        [TEMPORAL] Starting temporal events processing...")
        
        # Apply basal insulin rates with proper temporal logic
        print(f"        [TEMPORAL] Processing basal insulin rates...")
        df = self._apply_basal_rates(df)
        
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
    
    def _apply_basal_rates(self, df: pd.DataFrame):
        """
        Apply basal insulin rates with proper temporal logic.
        
        This method handles both regular basal rates and temporary basal rates.
        Basal rates persist until a new rate is set, and temporary basal rates override regular rates for their specified duration.

        Since basal rates are given in hourly units, they are converted to the sampling rate (e.g., 5-minute intervals).

        Args:
            df: DataFrame with basal rate data
            
        Returns:
            DataFrame with properly applied basal rates
        """
        if 'basal' not in df.columns:
            print("         [BASAL] No basal column found, skipping basal processing")
            return df
            
        df = df.copy()
        print(f"            [BASAL] Processing basal rates for {len(df)} rows")
        
        # Convert hourly basal rates to sampling rate (e.g., 5-minute rates)
        # Basal rates in the data are typically in units/hour
        # Convert to units per sampling interval
        sampling_factor = self.sampling_rate / 60.0  # Convert minutes to fraction of hour
        print(f"            [BASAL] Sampling factor: {sampling_factor:.4f} (converting {self.sampling_rate}min intervals)")
        
        # Step 1: Handle regular basal rates - forward fill valid values
        # Replace -1 with NaN for proper forward filling        
        df['basal'] = df['basal'].replace(-1, np.nan)
        
        # Forward fill basal rates (a basal rate persists until changed)
        df['basal'] = df['basal'].ffill()
        
        # Convert hourly rates to sampling interval rates    
        df['basal'] = pd.to_numeric(df['basal'], errors='coerce')
        df['basal'] = df['basal'] * sampling_factor
        # Step 2: Handle temporary basal rates (temp_basal events)
        if ('temp_basal' in df.columns and 'basal_end' in df.columns):            
            # Process each temp_basal event
            for i in range(len(df)):
                if (df['temp_basal'].iloc[i] != -1 and 
                    not pd.isna(df['temp_basal'].iloc[i]) and
                    df['basal_end'].iloc[i] != -1 and
                    not pd.isna(df['basal_end'].iloc[i])):
                    
                    temp_basal_rate = pd.to_numeric(df['temp_basal'].iloc[i], errors='coerce')
                    basal_end_time = df['basal_end'].iloc[i]
                    temp_start_time = df['index'].iloc[i]
                    
                    # Convert temp basal rate to sampling interval rate
                    temp_basal_rate_interval = temp_basal_rate * sampling_factor
                    
                    # Apply temp basal rate from start time to end time
                    mask = ((df['index'] >= temp_start_time) & 
                           (df['index'] <= basal_end_time))
                    
                    df.loc[mask, 'basal'] = temp_basal_rate_interval
            
            # Clean up temporary columns
            df = df.drop(columns=['temp_basal', 'basal_end'], errors='ignore')
        else:
            print(f"            [BASAL] No temporary basal columns found or incomplete temp basal data")
        
        # Fill any remaining NaN values with 0 (no basal insulin)
        final_nan_count = df['basal'].isna().sum()
        if final_nan_count > 0:
            df['basal'] = df['basal'].fillna(0)
        
        # Show summary statistics
        basal_stats = df['basal'].describe()
        print(f"            [BASAL] Final basal stats - Min: {basal_stats['min']:.4f}, Max: {basal_stats['max']:.4f}, Mean: {basal_stats['mean']:.4f}")
        
        return df
    
    def _check_missing_timesteps(self, df: pd.DataFrame):
        """Check for missing timesteps and flag them."""
        for i in range(1, len(df)):
            gap = (df['index'].iloc[i] - df['index'].iloc[i-1]).total_seconds() / 60.0 # Convert to minutes
            if gap != self.sampling_rate:
                df.iloc[i, df.columns.get_loc('missing')] = gap
        
        return df

    def _elapsed_since_last_valid(self, series: pd.Series,
                                  timestamps: pd.Series):
        """Minutes from each row back to the most recent observed value."""
        observed_at = timestamps.where(series.notna()).ffill()
        return (timestamps - observed_at).dt.total_seconds() / 60.0

    def _gap_span_minutes(self, series: pd.Series,
                          timestamps: pd.Series):
        """Total minutes bridged by the gap each row belongs to.

        For a row inside a run of missing values this is the distance from the
        last observed value to the next one, so a gap is judged as a whole
        rather than one step at a time. Rows at the edges of the series, where
        one side has no observation, get infinity and are never filled.
        """
        observed_at = timestamps.where(series.notna())
        before = observed_at.ffill()
        after = observed_at.bfill()
        span = (after - before).dt.total_seconds() / 60.0
        return span.fillna(np.inf)

    def handle_missing_data(self,
                           df: pd.DataFrame,
                           strategy: str = 'none',
                           max_gap: int = None):
        """
        Handle missing data in the time series.

        Gaps are judged in elapsed time, not in row count. The data is in 5 min gaps only where the sensor captured new results, so this function is used to hadle the missing data in between reads.

        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing data
                     ('none','interpolate', 'forward_fill', 'drop')
            max_gap: Maximum gap to bridge, in time steps. Defaults to
                     ``max_gap_minutes // sampling_rate``. Converted to minutes
                     internally, so it is a real time limit under both a
                     complete and an interrupted sampling grid.

        Returns:
            DataFrame with missing data handled
        """
        df = df.copy()

        if max_gap is None:
            max_gap = self.max_gap_minutes // self.sampling_rate
        if max_gap < 0:
            raise ValueError(f"max_gap must be non-negative, got {max_gap}")
        max_gap_minutes = max_gap * self.sampling_rate

        valid_strategies = ('none', 'interpolate', 'forward_fill', 'drop')
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown missing-data strategy {strategy!r}; expected one of "
                "'none', 'interpolate', 'forward_fill', 'drop'"
            )

        if self.target_column not in df.columns:
            print(f"    [MISS] No {self.target_column!r} column present; "
                  f"strategy {strategy!r} has nothing to act on")
            return df

        # Convert -1 values to NaN for the target column
        df[self.target_column] = df[self.target_column].replace(-1, np.nan)
        before = df[self.target_column].notna().sum()
        df[self.target_column] = pd.to_numeric(df[self.target_column],
                                               errors='coerce')
        unparseable = before - df[self.target_column].notna().sum()
        if unparseable:
            print(f"    [MISS] WARNING: {unparseable} non-numeric "
                  f"{self.target_column} value(s) coerced to NaN")
        # A -1 written as text survives the replace above, which compares
        # against the integer sentinel; catch it once the column is numeric.
        df[self.target_column] = df[self.target_column].replace(-1, np.nan)

        timestamps = pd.to_datetime(
            df[self.time_column] if self.time_column in df.columns else df.index
        )
        timestamps = pd.Series(timestamps.values, index=df.index)

        target = df[self.target_column]
        gaps_before = int(target.isna().sum())
        filled = dropped = 0

        if strategy == 'none':
            pass

        elif strategy == 'interpolate':
            # method='time' weights by real elapsed time, so a step that is
            # wider than the nominal sampling rate is not treated as one tick.
            bridgeable = self._gap_span_minutes(target, timestamps) <= max_gap_minutes
            interpolated = target.interpolate(method='time',
                                              limit_direction='forward')
            df[self.target_column] = target.where(~(target.isna() & bridgeable),
                                                  interpolated)
            filled = gaps_before - int(df[self.target_column].isna().sum())

        elif strategy == 'forward_fill':
            # Carry a value forward only while it is still recent enough; the
            # elapsed time is measured from the observation itself, so a gap
            # made of absent rows counts the same as one made of NaN rows.
            within_limit = self._elapsed_since_last_valid(target, timestamps) <= max_gap_minutes
            carried = target.ffill()
            df[self.target_column] = target.where(~(target.isna() & within_limit),
                                                  carried)
            filled = gaps_before - int(df[self.target_column].isna().sum())

        elif strategy == 'drop':
            # Drop rows with missing glucose values
            df = df.dropna(subset=[self.target_column])
            dropped = gaps_before

        remaining = int(df[self.target_column].isna().sum())

        if strategy == 'none':
            print(f"    [MISS] Leaving {remaining} gap row(s) as NaN; windows touching them are skipped downstream")
        elif strategy == 'drop':
            print(f"    [MISS] Dropped {dropped} row(s) with a missing "
                  f"{self.target_column}")
            print("    [MISS] WARNING: dropping rows closes the gap in the row order without closing it in time, so a downstream window can span an outage without any NaN to stop it; do not report clinical metrics from this run")
        else:
            print(f"    [MISS] Using {strategy} strategy with max gap of {max_gap} steps ({max_gap_minutes} min)")
            print(f"    [MISS] Filled {filled} of {gaps_before} gap row(s); {remaining} left as NaN beyond the {max_gap_minutes}-minute limit")
            if filled:
                print("    [MISS] WARNING: filled values become prediction targets and were never measured; do not report clinical metrics from this run")
        return df
    
    def engineer_features(self, df: pd.DataFrame):
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
            
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            print(f"    [ENGINEERING] Engineering features for hour-based cyclicality")

            #drop non-cyclical columns
            df = df.drop(columns=['hour'], errors='ignore')
        
        return df
    
    def create_sequences(self, 
                        df: pd.DataFrame,
                        sequence_length: int = 12,
                        prediction_horizon: int = 6,
                        step_size: int = 1):
        """
        Create sequences for time series forecasting.
        
        Args:
            df: Input DataFrame
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
            step_size: Step size for sliding window
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        
        df = df.copy()
        df[self.target_column] = pd.to_numeric(df[self.target_column], errors='coerce')

        # Drop rows with NaN in target column
        df = df.dropna(subset=[self.target_column])

        if len(df) < sequence_length + prediction_horizon:
            raise ValueError("DataFrame too short for specified sequence parameters")
        
        # Select numeric columns for features
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from features to avoid data leakage
        if self.target_column in feature_columns:
            feature_columns.remove(self.target_column)

        X, y = [], []
        
        for i in range(0, len(df) - sequence_length - prediction_horizon + 1, step_size):
            # Input sequence
            x_seq = df.iloc[i:i + sequence_length][feature_columns].values
            
            # Target sequence (can be single value or multiple)
            if prediction_horizon == 1:
                y_seq = df.iloc[i + sequence_length][self.target_column]
            else:
                y_seq = df.iloc[i + sequence_length:i + sequence_length + prediction_horizon][self.target_column].values

            # Check for NaN values
            if not (np.isnan(x_seq).any() or np.isnan(y_seq).any()):
                X.append(x_seq)
                y.append(y_seq)
        
        return np.array(X), np.array(y)
    
    def preprocess_patient_data(self, 
                               df: pd.DataFrame,
                               include_feature_engineering: bool = False,
                               handle_missing: str = 'none'
                               ):
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
        
        print("Preprocessing completed!")
        return df

def preprocess_ohiot1dm_data(patient_data: Dict[int, pd.DataFrame],
                            target_column: str = 'glucose',
                            **preprocessing_kwargs):
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
        except Exception as exc:
            raise RuntimeError(f"Preprocessing failed for patient {patient_id}: {exc}") from exc
        preprocessed_data[patient_id] = processed_df
        print(f"Successfully preprocessed patient {patient_id}")
    
    return preprocessed_data


def extract_and_save_ohio_data(data_dir: str,
                         output_dir: str,
                         patient_ids: Optional[List[int]] = None,
                         modes: List[str] = ['train', 'test'],
                         version: str = '2020',
                         sampling_rate: int = 5,
                         apply_preprocessing: bool = True):
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
