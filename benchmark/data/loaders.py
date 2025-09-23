"""
Data loaders for various blood glucose datasets.

This module provides standardized data loading functions for different
blood glucose datasets commonly used in research. The loaders return
raw combined dataframes without any data alterations.

All data preprocessing and alterations should be done using the 
preprocessors module.

Supported dataset:
- OhioT1DM Dataset (2018 and 2020 versions)

For other datasets, please refer to the documentation for details on loading and preprocessing steps. 
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import datetime
import warnings
from typing import List, Dict, Optional, Tuple

warnings.simplefilter('ignore', Warning) # Ignore warnings for cleaner output


class OhioT1DMDataLoader:
    """
    Data loader for the OhioT1DM dataset.
    
    This loader handles XML files from the OhioT1DM dataset and converts them
    into raw structured pandas DataFrames with proper time series alignment.
    No data alterations or preprocessing is applied - the data is returned
    as-is from the XML files after basic parsing and merging.
    
    All data processing should be done using the OhioBGDataPreprocessor class.
    
    Attributes:
        data_dir (str): Path to the data directory
        sampling_rate (int): Target sampling rate in minutes (default: 5)
        version (str): Dataset version ('2018' or '2020')
    """
    
    def __init__(self, data_dir: str, sampling_rate: int = 5, version: str = '2020'):
        """
        Initialize the OhioT1DM data loader.
        
        Args:
            data_dir: Path to the data directory containing XML files
            sampling_rate: Target sampling rate in minutes
            version: Dataset version ('2018' or '2020')
        """
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.version = version
        
        # Patient IDs for different versions
        self.patient_ids = {
            '2018': [559, 563, 570, 575, 588, 591],
            '2020': [540, 544, 552, 567, 584, 596]
        }
    
    def round_minute(self, date_string: str, round2min: int = 5) -> datetime.datetime:
        """
        Round datetime to specified minute intervals.
        
        Args:
            date_string: Date string in format "%d-%m-%Y %H:%M:%S"
            round2min: Minutes to round to
            
        Returns:
            Rounded datetime object
        """
        date = datetime.datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
        new_min = ((date.minute // round2min) * round2min) # Round down to nearest interval E.g. 4 -> 0, 6->5 , 12 -> 10
        date = date.replace(minute=int(new_min), second=0)
        return date
    
    def get_cgm(self, root: ET.Element) -> pd.DataFrame:
        """Extract CGM glucose data from XML."""
        glucose = []
        glucose_ts = []
        for event in root.findall('glucose_level/event'):
            value = event.get('value')
            ts = event.get('ts')
            ts = self.round_minute(ts, self.sampling_rate)
            glucose.append(value)
            glucose_ts.append(ts)
        
        return pd.DataFrame({
            'ts': glucose_ts,
            'glucose': glucose
        }).set_index('ts')
    
    def get_fingerstick(self, root: ET.Element) -> pd.DataFrame:
        """Extract fingerstick glucose data from XML."""
        fingerstick = []
        fingerstick_ts = []
        for event in root.findall('finger_stick/event'):
            value = event.get('value')
            ts = event.get('ts')
            ts = self.round_minute(ts, self.sampling_rate)
            fingerstick.append(value)
            fingerstick_ts.append(ts)
        
        return pd.DataFrame({
            'ts': fingerstick_ts,
            'fingerstick': fingerstick
        }).set_index('ts')
    
    def get_gsr(self, root: ET.Element) -> pd.DataFrame:
        """Extract galvanic skin response data from XML."""
        gsr = []
        gsr_ts = []
        for event in root.findall('basis_gsr/event'):
            value = event.get('value')
            ts = event.get('ts')
            ts = self.round_minute(ts, self.sampling_rate)
            gsr.append(value)
            gsr_ts.append(ts)
        
        return pd.DataFrame({
            'ts': gsr_ts,
            'gsr': gsr
        }).set_index('ts')
    
    def get_heart_rate(self, root: ET.Element) -> pd.DataFrame:
        """Extract heart rate data from XML."""
        hr = []
        hr_ts = []
        for event in root.findall('basis_heart_rate/event'):
            value = event.get('value')
            ts = event.get('ts')
            ts = self.round_minute(ts, self.sampling_rate)
            hr.append(value)
            hr_ts.append(ts)
        
        return pd.DataFrame({
            'ts': hr_ts,
            'hr': hr
        }).set_index('ts')
    
    def get_skin_temperature(self, root: ET.Element) -> pd.DataFrame:
        """Extract skin temperature data from XML."""
        st = []
        st_ts = []
        for event in root.findall('basis_skin_temperature/event'):
            value = event.get('value')
            ts = event.get('ts')
            ts = self.round_minute(ts, self.sampling_rate)
            st.append(value)
            st_ts.append(ts)
        
        return pd.DataFrame({
            'ts': st_ts,
            'st': st
        }).set_index('ts')
    
    def get_basal(self, root: ET.Element) -> pd.DataFrame:
        """Extract basal insulin data from XML."""
        basal = []
        basal_ts = []
        for event in root.findall('basal/event'):
            value = event.get('value')
            ts = event.get('ts')
            ts = self.round_minute(ts, self.sampling_rate)
            basal.append(value)
            basal_ts.append(ts)
        
        return pd.DataFrame({
            'ts': basal_ts,
            'basal': basal
        }).set_index('ts')
    
    def get_temp_basal(self, root: ET.Element) -> pd.DataFrame:
        """Extract temporary basal insulin data from XML."""
        temp_basal = []
        temp_basal_ts = []
        basal_end = []
        
        for event in root.findall('temp_basal/event'):
            value = event.get('value')
            ts_begin = event.get('ts_begin')
            ts_end = event.get('ts_end')
            
            ts_begin = self.round_minute(ts_begin, self.sampling_rate)
            ts_end = self.round_minute(ts_end, self.sampling_rate)
            
            temp_basal.append(value)
            temp_basal_ts.append(ts_begin)
            basal_end.append(ts_end)
        
        return pd.DataFrame({
            'ts': temp_basal_ts,
            'temp_basal': temp_basal,
            'basal_end': basal_end
        }).set_index('ts')
    
    def get_bolus(self, root: ET.Element) -> pd.DataFrame:
        """Extract bolus insulin data from XML."""
        bolus = []
        bolus_ts = []
        bolus_end = []
        bolus_dur = []
        
        for event in root.findall('bolus/event'):
            dose = event.get('dose')
            ts_begin = event.get('ts_begin')
            ts_end = event.get('ts_end')
            
            ts_begin = self.round_minute(ts_begin, self.sampling_rate)
            ts_end = self.round_minute(ts_end, self.sampling_rate)
            
            duration = (ts_end - ts_begin).seconds // 60
            
            bolus.append(dose)
            bolus_ts.append(ts_begin)
            bolus_end.append(ts_end)
            bolus_dur.append(duration)
        
        return pd.DataFrame({
            'ts': bolus_ts,
            'bolus': bolus,
            'bolus_dur': bolus_dur,
            'bolus_end': bolus_end
        }).set_index('ts')
    
    def get_meal(self, root: ET.Element) -> pd.DataFrame:
        """Extract meal/carbohydrate data from XML."""
        carbs = []
        meal_ts = []
        meal_type = []
        
        for event in root.findall('meal/event'):
            carb_value = event.get('carbs')
            ts = event.get('ts')
            meal_type_value = event.get('type')
            
            ts = self.round_minute(ts, self.sampling_rate)
            
            carbs.append(carb_value)
            meal_ts.append(ts)
            meal_type.append(meal_type_value)
        
        return pd.DataFrame({
            'ts': meal_ts,
            'carbs': carbs,
            'meal_type': meal_type
        }).set_index('ts')
    
    def get_exercise(self, root: ET.Element) -> pd.DataFrame:
        """Extract exercise data from XML."""
        exercise_intensity = []
        exercise_ts = []
        exercise_dur = []
        
        for event in root.findall('exercise/event'):
            intensity = event.get('intensity')
            ts = event.get('ts')
            duration = event.get('duration')
            
            ts = self.round_minute(ts, self.sampling_rate)
            
            exercise_intensity.append(intensity)
            exercise_ts.append(ts)
            exercise_dur.append(duration)
        
        return pd.DataFrame({
            'ts': exercise_ts,
            'exercise_intensity': exercise_intensity,
            'exer_dur': exercise_dur
        }).set_index('ts')
    
    def get_sleep(self, root: ET.Element) -> pd.DataFrame:
        """Extract sleep data from XML."""
        sleep_quality = []
        sleep_ts = []
        sleep_dur = []
        sleep_end = []
        
        for event in root.findall('sleep/event'):
            quality = event.get('quality')
            # Note: ts_end and ts_begin are swapped in the dataset
            ts_begin = event.get('ts_end')
            ts_end = event.get('ts_begin')
            
            ts_begin = self.round_minute(ts_begin, self.sampling_rate)
            ts_end = self.round_minute(ts_end, self.sampling_rate)
            
            duration = (ts_end - ts_begin).seconds // 60
            
            sleep_quality.append(quality)
            sleep_ts.append(ts_begin)
            sleep_dur.append(duration)
            sleep_end.append(ts_end)
        
        return pd.DataFrame({
            'ts': sleep_ts,
            'sleep': sleep_quality,
            'sleep_dur': sleep_dur,
            'sleep_end': sleep_end
        }).set_index('ts')
    
    def get_work(self, root: ET.Element) -> pd.DataFrame:
        """Extract work stress data from XML."""
        work_intensity = []
        work_ts = []
        work_dur = []
        work_end = []
        
        for event in root.findall('work/event'):
            intensity = event.get('intensity')
            ts_begin = event.get('ts_begin')
            ts_end = event.get('ts_end')
            
            ts_begin = self.round_minute(ts_begin, self.sampling_rate)
            ts_end = self.round_minute(ts_end, self.sampling_rate)
            
            duration = (ts_end - ts_begin).seconds // 60
            
            work_intensity.append(intensity)
            work_ts.append(ts_begin)
            work_dur.append(duration)
            work_end.append(ts_end)
        
        return pd.DataFrame({
            'ts': work_ts,
            'work': work_intensity,
            'work_dur': work_dur,
            'work_end': work_end
        }).set_index('ts')

    
    def load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Load and parse a single XML file from the OhioT1DM dataset.
        
        Args:
            file_path: Path to the XML file
            
        Returns:
            Merged DataFrame with all data types
        """
        root = ET.parse(file_path).getroot()
        
        # Extract all data types
        dataframes = []
        
        # Core continuous glucose data
        cgm_df = self.get_cgm(root)
        if not cgm_df.empty:
            dataframes.append(cgm_df)
        
        # Wearable sensor data (GSR, HR, ST)
        gsr_df = self.get_gsr(root)
        if not gsr_df.empty:
            dataframes.append(gsr_df)
            
        hr_df = self.get_heart_rate(root)
        if not hr_df.empty:
            dataframes.append(hr_df)
            
        st_df = self.get_skin_temperature(root)
        if not st_df.empty:
            dataframes.append(st_df)
        
        # Insulin data
        basal_df = self.get_basal(root)
        if not basal_df.empty:
            dataframes.append(basal_df)
            
        temp_basal_df = self.get_temp_basal(root)
        if not temp_basal_df.empty:
            dataframes.append(temp_basal_df)
            
        bolus_df = self.get_bolus(root)
        if not bolus_df.empty:
            dataframes.append(bolus_df)
        
        # Lifestyle data
        meal_df = self.get_meal(root)
        if not meal_df.empty:
            dataframes.append(meal_df)
            
        exercise_df = self.get_exercise(root)
        if not exercise_df.empty:
            dataframes.append(exercise_df)
            
        sleep_df = self.get_sleep(root)
        if not sleep_df.empty:
            dataframes.append(sleep_df)
            
        work_df = self.get_work(root)
        if not work_df.empty:
            dataframes.append(work_df)
        
        # Merge all dataframes
        if not dataframes:
            return pd.DataFrame()
        
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = merged_df.join(df, how="outer")
        
        return merged_df
    
    # ---------------------------------
    
    def load_patient_data(self, patient_id: int, mode: str = 'train') -> pd.DataFrame:
        """
        Load raw data for a specific patient.
        
        Args:
            patient_id: Patient identifier
            mode: 'train' or 'test'
            
        Returns:
            Raw merged DataFrame for the patient (no processing applied)
        """
        file_pattern = f"{patient_id}-ws-{mode}ing.xml"
        file_path = os.path.join(
            self.data_dir, 
            'raw', 
            'ohiot1dm', 
            self.version, 
            mode, 
            file_pattern
        )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        print(f"Loading data for patient {patient_id} ({mode} set)")
        df = self.load_single_file(file_path)
        
        return df
    
    def load_all_patients(self, mode: str = 'train') -> Dict[int, pd.DataFrame]:
        """
        Load raw data for all patients in the specified version.
        
        Args:
            mode: 'train' or 'test'
            
        Returns:
            Dictionary mapping patient IDs to their raw DataFrames
        """
        patient_data = {}
        
        for patient_id in self.patient_ids[self.version]:
            try:
                df = self.load_patient_data(patient_id, mode)
                patient_data[patient_id] = df
                print(f"Successfully loaded patient {patient_id}")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        return patient_data
    
    def get_available_patients(self, mode: str = 'train') -> List[int]:
        """
        Get list of available patient IDs for the specified mode.
        
        Args:
            mode: 'train' or 'test'
            
        Returns:
            List of available patient IDs
        """
        available_patients = []
        
        for patient_id in self.patient_ids[self.version]:
            file_pattern = f"{patient_id}-ws-{mode}ing.xml"
            file_path = os.path.join(
                self.data_dir, 
                'raw', 
                'ohiot1dm', 
                self.version, 
                mode, 
                file_pattern
            )
            
            if os.path.exists(file_path):
                available_patients.append(patient_id)
        
        return available_patients


def load_ohiot1dm_data(data_dir: str, 
                      patient_ids: Optional[List[int]] = None,
                      mode: str = 'train',
                      version: str = '2020',
                      sampling_rate: int = 5) -> Dict[int, pd.DataFrame]:
    """
    Convenience function to load raw OhioT1DM dataset.
    
    Args:
        data_dir: Path to the data directory
        patient_ids: List of specific patient IDs to load (None for all)
        mode: 'train' or 'test'
        version: Dataset version ('2018' or '2020')
        sampling_rate: Target sampling rate in minutes
        
    Returns:
        Dictionary mapping patient IDs to their raw DataFrames
        
    Example:
        >>> data = load_ohiot1dm_data('/path/to/data', patient_ids=[540, 544], mode='train')
        >>> patient_540_data = data[540]
    """
    loader = OhioT1DMDataLoader(data_dir, sampling_rate, version)
    
    if patient_ids is None:
        return loader.load_all_patients(mode)
    else:
        patient_data = {}
        for patient_id in patient_ids:
            try:
                df = loader.load_patient_data(patient_id, mode)
                patient_data[patient_id] = df
            except FileNotFoundError as e:
                print(f"Warning: Could not load patient {patient_id}: {e}")
                continue
        return patient_data
