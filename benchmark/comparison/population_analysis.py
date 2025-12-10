
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


from .patient_analysis import PatientAnalyzer
from ..data.loaders import load_ohiot1dm_data
from ..data.preprocessors import OhioBGDataPreprocessor


class PopulationAnalysis:
    """
    Population-level train vs test comparison analysis.
    """
    
    def __init__(self, experiment_dir: str, output_dir: str, data_root: str = "data"):
        """
        Initialize population analysis.
        
        Args:
            experiment_dir: Directory containing experiment results
            output_dir: Directory to save analysis results
            data_root: Root directory for data (default: "data")
        """
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.data_root = Path(data_root)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train_test_comparison").mkdir(exist_ok=True)
        (self.output_dir / "glucose_history").mkdir(exist_ok=True)
        
        print("="*80)
        print("POPULATION ANALYSIS - GLUCOSE HISTORY & TRAIN VS TEST COMPARISON")
        print("="*80)
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Data root: {self.data_root}")
    

    
    def run_train_test_comparison(self, model: str = 'RNN') -> str:
        """
        Run train vs test glucose comparison analysis.
        
        Args:
            model: Model name to analyze
            
        Returns:
            Path to saved comparison plot
        """
        print(f"\\n{'='*80}")
        print(f"TRAIN VS TEST GLUCOSE COMPARISON ({model})")
        print(f"{'='*80}")
        
        try:
            # Initialize analyzer
            analyzer = PatientAnalyzer(str(self.experiment_dir))
            analyzer.load_experiment_results()
            
            # Create comparison plot
            save_path = self.output_dir / "train_test_comparison" / f"train_test_glucose_comparison_{model}.png"
            
            plot_data = analyzer.plot_train_test_glucose_comparison(
                data_root=str(self.data_root),
                model_name=model,
                save_path=str(save_path),
                max_samples=3000,  # Balanced for performance
                use_pca_fallback=True
            )
            
            print(f"✓ Train/test comparison completed for {model}")
            return str(save_path)
            
        except Exception as e:
            print(f"✗ Train/test comparison failed for {model}: {e}")
            return None
    
    def load_and_preprocess_glucose_data(self, patient_ids: Optional[List[int]] = None) -> Dict:
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
            # The loader expects the parent directory that contains raw/ohiot1dm/
            data_dir = self.data_root
            
            # Load train and test data
            print(f"\nLoading training data from: {data_dir}/raw/ohiot1dm")
            train_data = load_ohiot1dm_data(
                str(data_dir), 
                patient_ids=patient_ids,
                mode='train',
                version=['2018', '2020']
            )
            
            print("\nLoading test data...")
            test_data = load_ohiot1dm_data(
                str(data_dir),
                patient_ids=patient_ids, 
                mode='test',
                version=['2018', '2020']
            )
            
            # Preprocess data
            print(f"\n{'='*80}")
            print("PREPROCESSING GLUCOSE DATA")
            print(f"{'='*80}")
            
            preprocessor = OhioBGDataPreprocessor()
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
                    'test': test_df
                }
                
                print(f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")
            
            return patient_data
            
        except Exception as e:
            print(f"✗ Failed to load glucose data: {e}")
            return {}
    
    def _calculate_glucose_statistics(self, glucose: np.ndarray) -> Dict:
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
            '% in range (70-180)': np.sum((glucose >= 70) & (glucose <= 180)) / len(glucose) * 100,
            '% hypoglycemia (<70)': np.sum(glucose < 70) / len(glucose) * 100,
            '% hyperglycemia (>180)': np.sum(glucose > 180) / len(glucose) * 100,
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
            
            train_time = np.arange(len(train_glucose))
            test_time = np.arange(len(train_glucose), len(train_glucose) + len(test_glucose))
            
            # Convert to hours
            train_time_hours = train_time * 5 / 60
            test_time_hours = test_time * 5 / 60
            
            # Plot
            ax.plot(train_time_hours, train_glucose, 'b-', linewidth=0.5, alpha=0.6, label='Train')
            ax.plot(test_time_hours, test_glucose, 'r-', linewidth=0.5, alpha=0.6, label='Test')
            
            # Clinical ranges
            ax.axhspan(70, 180, color='g', alpha=0.05)
            ax.axhspan(0, 70, color='r', alpha=0.05)
            ax.axhspan(180, 400, color='y', alpha=0.05)
            
            # Split line
            split_time = len(train_glucose) * 5 / 60
            ax.axvline(x=split_time, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_title(f'Patient {patient_id}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (hours)', fontsize=9)
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
    
    def run_glucose_history_analysis(self, patient_ids: Optional[List[int]] = None) -> bool:
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
            overview_path = self.output_dir / "glucose_history" / "population_glucose_overview.png"
            self.plot_population_glucose_overview(patient_data, str(overview_path))
            
            # Create summary report
            summary_path = self.output_dir / "glucose_history" / "population_glucose_summary.csv"
            self.create_glucose_summary_report(patient_data, str(summary_path))
            
            print(f"✓ Glucose history analysis completed")
            return True
            
        except Exception as e:
            print(f"✗ Glucose history analysis failed: {e}")
            return False

    
    def create_population_report(self, results_summary: Dict, save_path: str):
        """
        Create population analysis report.
        
        Args:
            results_summary: Dictionary containing all analysis results
            save_path: Path to save the report
        """
        report = []
        report.append("# Population Analysis Report")
        report.append("")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Experiment**: {self.experiment_dir.name}")
        report.append("")
        
        # Overview
        report.append("## Analysis Overview")
        report.append("")
        report.append("This report contains glucose history analysis and train vs test glucose pattern comparison.")
        report.append("")
        
        # Glucose history results
        if 'glucose_history' in results_summary:
            glucose_success = results_summary['glucose_history']
            report.append("## Glucose History Analysis")
            report.append("")
            if glucose_success:
                report.append("✓ **Status**: Successfully completed")
                report.append("")
                report.append("- **Population overview plot**: `glucose_history/population_glucose_overview.png`")
                report.append("- **Summary statistics**: `glucose_history/population_glucose_summary.csv`")
                report.append("- **Analysis**: Complete glucose time series for all patients showing train/test splits")
            else:
                report.append("✗ **Status**: Failed or no data available")
                report.append("")
                report.append("- Raw glucose data may not be available in the expected location")
                report.append("- Check data directory structure: `data/raw/ohiot1dm/`")
            report.append("")
        
        # Train/test comparison results
        if 'train_test_comparison' in results_summary:
            comparison_results = results_summary['train_test_comparison']
            report.append("## Train vs Test Comparison Results")
            report.append("")
            
            for model, plot_path in comparison_results.items():
                if plot_path:
                    report.append(f"### {model} Model")
                    report.append("")
                    report.append(f"- **Comparison plot**: `{Path(plot_path).name}`")
                    report.append(f"- **Analysis**: t-SNE visualization of glucose patterns comparing training and test sets")
                    report.append("")
        
        # File listings
        report.append("## Generated Files")
        report.append("")
        
        # List glucose history files
        glucose_dir = self.output_dir / "glucose_history"
        if glucose_dir.exists():
            files = list(glucose_dir.rglob('*.*'))
            if files:
                report.append("### Glucose History")
                report.append("")
                for file_path in sorted(files):
                    rel_path = file_path.relative_to(self.output_dir)
                    report.append(f"- `{rel_path}`")
                report.append("")
        
        # List train/test comparison files
        subdir_path = self.output_dir / "train_test_comparison"
        if subdir_path.exists():
            files = list(subdir_path.rglob('*.*'))
            if files:
                report.append("### Train Test Comparison")
                report.append("")
                for file_path in sorted(files):
                    rel_path = file_path.relative_to(self.output_dir)
                    report.append(f"- `{rel_path}`")
                report.append("")
        
        # Write report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✓ Population analysis report saved: {Path(save_path).name}")
    
    def run_complete_analysis(self, models: List[str] = ['RNN'], patient_ids: Optional[List[int]] = None) -> Dict:
        """
        Run complete population analysis.
        
        Args:
            models: List of model names to analyze
            patient_ids: Optional list of specific patient IDs to analyze
            
        Returns:
            Dictionary with all analysis results
        """
        results_summary = {}
        
        # Glucose History Analysis
        print(f"\n{'='*80}")
        print("GLUCOSE HISTORY ANALYSIS")
        print(f"{'='*80}")
        
        glucose_success = self.run_glucose_history_analysis(patient_ids)
        results_summary['glucose_history'] = glucose_success
        
        # Train vs Test Comparison
        print(f"\n{'='*80}")
        print("TRAIN VS TEST COMPARISON")
        print(f"{'='*80}")
        
        comparison_results = {}
        for model in models:
            try:
                plot_path = self.run_train_test_comparison(model)
                if plot_path:
                    comparison_results[model] = plot_path
            except Exception as e:
                print(f"✗ Train/test comparison failed for {model}: {e}")
        
        results_summary['train_test_comparison'] = comparison_results
        
        # Generate Final Report
        print(f"\n{'='*80}")
        print("GENERATING FINAL REPORT")
        print(f"{'='*80}")
        
        report_path = self.output_dir / "POPULATION_ANALYSIS_REPORT.md"
        self.create_population_report(results_summary, str(report_path))
        
        # Summary
        print(f"\n{'='*80}")
        print("POPULATION ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"\nAnalysis summary:")
        print(f"  - Glucose history: {'✓ Success' if glucose_success else '✗ Failed/No data'}")
        print(f"  - Train/test comparisons: {len(comparison_results)} models")
        print(f"  - Final report: POPULATION_ANALYSIS_REPORT.md")
        
        return results_summary

