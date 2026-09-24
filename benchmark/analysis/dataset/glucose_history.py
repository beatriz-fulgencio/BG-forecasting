"""
Population glucose history: the cohort's train and test series, from raw files.

Per-patient glucose over the whole OhioT1DM record, split at the train/test
boundary, plus the descriptive statistics of each half. Both are properties of
the dataset: no prediction and no metric is read, so the figure and the CSV are
the same whichever model was trained.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from ..results_io import load_experiment_results, read_tracking
from ...evaluation.metrics import BGMetrics
from ...data.loaders import load_ohiot1dm_data
from ...data.preprocessors import OhioBGDataPreprocessor


#: Subdirectory of ``output_dir`` the figures and tables land in.
OUTPUT_SUBDIR = "patient_analysis"


class GlucoseHistoryAnalysis:
    """
    Per-patient train and test glucose history for the whole cohort.
    """

    def __init__(self,
                 experiment_dir: str,
                 output_dir: str,
                 data_root: str = "data",
                 mode: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Initialize the glucose history analysis.

        Args:
            experiment_dir: Reference run. Fixes the cohort, the dataset
                releases and the sampling rate; contributes no numbers of its
                own.
            output_dir: Directory to save analysis results
            data_root: Root directory for data (default: "data")
            mode: Training mode whose cohort to read (``regular``/``transfer``).
                Required only when the run holds more than one.
            seed: Read one seed instead of the cross-seed mean. The raw series
                is the same either way; this only changes which export the
                cohort is read from.
        """
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.data_root = Path(data_root)
        self.mode = mode
        self.seed = seed

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # "patient_analysis": these are per-patient descriptive views of the
        # cohort, grouped with the other per-patient dataset outputs rather
        # than under a name that only describes this one figure.
        (self.output_dir / OUTPUT_SUBDIR).mkdir(exist_ok=True)

        print("="*80)
        print("GLUCOSE HISTORY ANALYSIS - POPULATION TRAIN/TEST SERIES")
        print("="*80)
        print(f"Reference run: {self.experiment_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Data root: {self.data_root}")
        if self.mode:
            print(f"Training mode: {self.mode}")
        if self.seed is not None:
            print(f"Seed: {self.seed}")

    def load_and_preprocess_glucose_data(self, patient_ids: Optional[List[int]] = None):
        """
        Load and preprocess glucose data for all patients.
        
        Args:
            patient_ids: List of patient IDs to process (None for all)
            
        Returns:
            Dictionary with patient data: {patient_id: {'train': df, 'test': df}}
        """
        print(f"\n{'='*80}")
        print("LOADING GLUCOSE DATA")
        print(f"{'='*80}")
        
        try:
            # Resolve cohort and temporal settings from the completed experiment,
            # never from a hard-coded 12-patient/default-5-minute assumption.
            loaded = load_experiment_results(self.experiment_dir, mode=self.mode, seed=self.seed,
                                             with_series=False)
            if patient_ids is None:
                patient_ids = list(loaded.patients)
            sampling_rate = loaded.sampling_rate_minutes or 5
            tracking = read_tracking(loaded.experiment_dir)
            configured_versions = (
                tracking.get("config", {}).get("data", {}).get("version")
                if isinstance(tracking, dict) else None
            )
            versions = configured_versions if isinstance(configured_versions, list) else ['2018', '2020']
            # The loader expects the parent directory that contains raw/ohiot1dm/
            data_dir = self.data_root
            
            # Load train and test data
            print(f"\nLoading training data from: {data_dir}/raw/ohiot1dm")
            train_data = load_ohiot1dm_data(
                str(data_dir), 
                patient_ids=patient_ids,
                mode='train',
                version=versions, sampling_rate=sampling_rate
            )
            
            print("\nLoading test data...")
            test_data = load_ohiot1dm_data(
                str(data_dir),
                patient_ids=patient_ids, 
                mode='test',
                version=versions, sampling_rate=sampling_rate
            )
            
            # Preprocess data
            print(f"\n{'='*80}")
            print("PREPROCESSING GLUCOSE DATA")
            print(f"{'='*80}")
            
            preprocessor = OhioBGDataPreprocessor(sampling_rate=sampling_rate)
            patient_data = {}
            
            for patient_id in train_data.keys():
                print(f"\nPatient {patient_id}:")
                
                # Preprocess train data
                print("  Processing training data...")
                train_df = preprocessor.basic_preprocessing(train_data[patient_id])
                
                # Preprocess test data  
                print("  Processing test data...")
                test_df = preprocessor.basic_preprocessing(test_data[patient_id])
                
                patient_data[patient_id] = {
                    'train': train_df,
                    'test': test_df,
                    'sampling_rate_minutes': sampling_rate,
                    'releases': versions,
                }
                
                print(f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")
            
            return patient_data
            
        except Exception as e:
            print(f"✗ Failed to load glucose data: {e}")
            return {}
    
    def _calculate_glucose_statistics(self, glucose: np.ndarray):
        """Calculate glucose statistics."""
        # Ensure glucose is float type for calculations
        glucose = glucose.astype(float)
        return {
            'Mean': np.mean(glucose),
            'Median': np.median(glucose),
            'Std Dev': np.std(glucose),
            'Min': np.min(glucose),
            'Max': np.max(glucose),
            '25th percentile': np.percentile(glucose, 25),
            '75th percentile': np.percentile(glucose, 75),
            # Delegated rather than recomputed, so the population view and the
            # per-run metrics cannot drift apart on the thresholds.
            '% in range (70-180)': BGMetrics.time_in_range(glucose),
            '% hypoglycemia (<70)': BGMetrics.time_below_range(glucose),
            '% hyperglycemia (>180)': BGMetrics.time_above_range(glucose),
        }
    
    def plot_population_glucose_overview(self, patient_data: Dict, save_path: str):
        """
        Create overview plot showing all patients' glucose history.
        
        Args:
            patient_data: Dictionary with patient data
            save_path: Path to save figure
        """
        n_patients = len(patient_data)
        if n_patients == 0:
            print("No patient data available for glucose overview")
            return
            
        n_cols = 3
        n_rows = (n_patients + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
        if n_patients == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (patient_id, data) in enumerate(sorted(patient_data.items())):
            ax = axes[idx]
            
            train_df = data['train']
            test_df = data['test']
            
            train_glucose = train_df['glucose'].astype(float).values
            test_glucose = test_df['glucose'].astype(float).values
            
            # The raw loader preserves its timestamp index. Plot it directly,
            # leaving disconnected episodes as visible gaps instead of creating
            # a synthetic continuous clock from row number.
            train_time = pd.to_datetime(train_df.index, errors='coerce')
            test_time = pd.to_datetime(test_df.index, errors='coerce')
            
            # Plot
            ax.plot(train_time, train_glucose, 'b-', linewidth=0.5, alpha=0.6, label='Train')
            ax.plot(test_time, test_glucose, 'r-', linewidth=0.5, alpha=0.6, label='Test')
            
            # Clinical ranges
            ax.axhspan(70, 180, color='g', alpha=0.05)
            ax.axhspan(0, 70, color='r', alpha=0.05)
            ax.axhspan(180, 400, color='y', alpha=0.05)
            
            # Split line
            if len(train_time) and not pd.isna(train_time[-1]):
                ax.axvline(x=train_time[-1], color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_title(f'Patient {patient_id}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Recorded timestamp', fontsize=9)
            ax.set_ylabel('Glucose (mg/dL)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([40, 400])
            ax.legend(fontsize=8, loc='upper right')
        
        # Hide unused subplots
        for idx in range(n_patients, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Population Glucose History Overview',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved glucose overview: {Path(save_path).name}")
    
    def create_glucose_summary_report(self, patient_data: Dict, save_path: str):
        """
        Create glucose history summary report.
        
        Args:
            patient_data: Dictionary with patient data
            save_path: Path to save report
        """
        if not patient_data:
            print("No patient data available for glucose summary")
            return
            
        summary_data = []
        
        for patient_id in sorted(patient_data.keys()):
            train_df = patient_data[patient_id]['train']
            test_df = patient_data[patient_id]['test']
            
            train_glucose = train_df['glucose'].astype(float).values
            test_glucose = test_df['glucose'].astype(float).values
            
            train_stats = self._calculate_glucose_statistics(train_glucose)
            test_stats = self._calculate_glucose_statistics(test_glucose)
            
            summary_data.append({
                'Patient_ID': patient_id,
                'Train_Samples': len(train_glucose),
                'Test_Samples': len(test_glucose),
                'Train_Mean': train_stats['Mean'],
                'Test_Mean': test_stats['Mean'],
                'Train_Std': train_stats['Std Dev'],
                'Test_Std': test_stats['Std Dev'],
                'Train_TIR': train_stats['% in range (70-180)'],
                'Test_TIR': test_stats['% in range (70-180)'],
                'Train_Hypo': train_stats['% hypoglycemia (<70)'],
                'Test_Hypo': test_stats['% hypoglycemia (<70)'],
                'Train_Hyper': train_stats['% hyperglycemia (>180)'],
                'Test_Hyper': test_stats['% hyperglycemia (>180)']
            })
        
        # Save as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(save_path, index=False)
        
        print(f"✓ Saved glucose summary: {Path(save_path).name}")
    
    def run_glucose_history_analysis(self, patient_ids: Optional[List[int]] = None):
        """
        Run glucose history analysis.
        
        Args:
            patient_ids: List of patient IDs to analyze (None for all)
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*80}")
        print("GLUCOSE HISTORY ANALYSIS")
        print(f"{'='*80}")
        
        try:
            # Load glucose data
            patient_data = self.load_and_preprocess_glucose_data(patient_ids)
            
            if not patient_data:
                print("✗ No glucose data available - skipping glucose history analysis")
                return False
            
            # Create overview plot
            overview_path = self.output_dir / OUTPUT_SUBDIR / "population_glucose_overview.png"
            self.plot_population_glucose_overview(patient_data, str(overview_path))
            
            # Create summary report
            summary_path = self.output_dir / OUTPUT_SUBDIR / "population_glucose_summary.csv"
            self.create_glucose_summary_report(patient_data, str(summary_path))
            
            print(f"✓ Glucose history analysis completed")
            return True
            
        except Exception as e:
            print(f"✗ Glucose history analysis failed: {e}")
            return False

    
