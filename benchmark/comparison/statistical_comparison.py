"""
Statistical Comparison Framework for Blood Glucose Forecasting Experiments

This module provides comprehensive statistical analysis and comparison tools for evaluating
blood glucose forecasting models across different experimental conditions, including:

Statistical Methods:
==================
- Paired t-tests for within-patient comparisons
- Cohen's d for effect size estimation
- Comprehensive significance testing with multiple comparison corrections
- Confidence interval calculations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import json


class ExperimentComparator:
    """
    Compare results from multiple experiments using statistical tests and visualizations.
    """
    
    def __init__(self, experiment_dirs: List[str], experiment_names: List[str] = None):
        """
        Initialize the comparator with experiment directories.
        
        Args:
            experiment_dirs: List of paths to experiment directories
            experiment_names: Optional names for experiments (defaults to dir names)
        """
        self.experiment_dirs = [Path(d) for d in experiment_dirs]
        
        if experiment_names is None:
            self.experiment_names = [d.name for d in self.experiment_dirs]
        else:
            self.experiment_names = experiment_names
        
        self.results = {}
        self.patient_features = {}
        
    def load_experiment_results(self):
        """Load results from all experiments."""
        print("Loading experiment results...")
        
        for exp_dir, exp_name in zip(self.experiment_dirs, self.experiment_names):
            print(f"\n  Loading {exp_name}...")
            self.results[exp_name] = self._load_single_experiment(exp_dir)
            
        print(f"\n[OK] Loaded {len(self.results)} experiments")
        
    def _load_single_experiment(self, exp_dir: Path) -> Dict:
        """
        Load results from a single experiment directory.
        
        Args:
            exp_dir: Path to experiment directory
            
        Returns:
            Dictionary with patient_id -> model_name -> metrics
        """
        results = {}
        
        # Find comprehensive metrics JSON files first (new format)
        json_files = list(exp_dir.glob("comprehensive_metrics_*.json"))
        
        # Fallback to old format if comprehensive metrics not found
        if not json_files:
            json_files = list(exp_dir.glob("model_comparison_*.json"))
            use_old_format = True
        else:
            use_old_format = False
        
        if not json_files:
            print(f"    No JSON files found in {exp_dir}")
            return results
        
        # Use the most recent JSON file if multiple exist
        json_file = sorted(json_files)[-1]
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if use_old_format:
                # Old format: direct patient_id -> model_name -> metrics
                for patient_id_str, patient_data in data.items():
                    try:
                        patient_id = int(patient_id_str)
                    except:
                        continue
                    
                    if patient_id not in results:
                        results[patient_id] = {}
                    
                    for model_name, model_data in patient_data.items():
                        # Extract metrics directly from JSON
                        mae = model_data.get('mae')
                        rmse = model_data.get('rmse')
                        mape = model_data.get('mape')
                        mard = model_data.get('mard')
                        
                        # Load predictions from CSV if available and needed
                        y_true = None
                        y_pred = None
                        if 'predictions_file' in model_data:
                            pred_file = Path(model_data['predictions_file'])
                            if pred_file.exists():
                                try:
                                    df = pd.read_csv(pred_file)
                                    y_true = df['true_values'].values
                                    y_pred = df['predictions'].values
                                except Exception as e:
                                    print(f"    Warning: Could not load predictions from {pred_file}: {e}")
                        
                        results[patient_id][model_name] = {
                            'mae': mae,
                            'rmse': rmse,
                            'mape': mape,
                            'mard': mard,
                            'y_true': y_true,
                            'y_pred': y_pred
                        }
            else:
                # New format: comprehensive_metrics with patient_metrics section
                patient_metrics = data.get('patient_metrics', {})
                
                for patient_id_str, patient_data in patient_metrics.items():
                    try:
                        patient_id = int(patient_id_str)
                    except:
                        continue
                    
                    if patient_id not in results:
                        results[patient_id] = {}
                    
                    for model_name, model_data in patient_data.items():
                        # Extract metrics directly from comprehensive JSON
                        mae = model_data.get('mae')
                        rmse = model_data.get('rmse')
                        mape = model_data.get('mape')
                        mard = model_data.get('mard')
                        
                        # Extract additional metrics available in comprehensive format
                        tir = model_data.get('tir', {})
                        clarke_zones = model_data.get('clarke_zones', {})
                        parkes_zones = model_data.get('parkes_zones', {})
                        clarke_a_b = model_data.get('clarke_a_b')
                        parkes_a_b = model_data.get('parkes_a_b')
                        
                        # Look for predictions CSV in patient subdirectory
                        y_true = None
                        y_pred = None
                        patient_dir = exp_dir / f"patient_{patient_id}"
                        if patient_dir.exists():
                            pred_files = list(patient_dir.glob(f"{model_name}_patient_{patient_id}_predictions.csv"))
                            if pred_files:
                                pred_file = pred_files[0]
                                try:
                                    df = pd.read_csv(pred_file)
                                    if 'true_values' in df.columns and 'predictions' in df.columns:
                                        y_true = df['true_values'].values
                                        y_pred = df['predictions'].values
                                except Exception as e:
                                    print(f"    Warning: Could not load predictions from {pred_file}: {e}")
                        
                        results[patient_id][model_name] = {
                            'mae': mae,
                            'rmse': rmse,
                            'mape': mape,
                            'mard': mard,
                            'tir': tir,
                            'clarke_zones': clarke_zones,
                            'parkes_zones': parkes_zones,
                            'clarke_a_b': clarke_a_b,
                            'parkes_a_b': parkes_a_b,
                            'y_true': y_true,
                            'y_pred': y_pred
                        }
            
            print(f"    Found {len(results)} patients from {json_file.name} ({'new' if not use_old_format else 'old'} format)")
            
        except Exception as e:
            print(f"    Error loading JSON file {json_file}: {e}")
        
        return results
    
    def paired_ttest_comparison(self, model_name: str = None) -> pd.DataFrame:
        """
        Perform paired t-test comparison between experiments.
        
        Args:
            model_name: Specific model to compare (e.g., 'GRU', 'LSTM', 'RNN')
                       If None, compares all models
        
        Returns:
            DataFrame with t-test results
        """
        print(f"\n{'='*60}")
        print("PAIRED T-TEST COMPARISON")
        print(f"{'='*60}")
        
        if len(self.results) != 2:
            raise ValueError("Paired t-test requires exactly 2 experiments")
        
        exp1_name, exp2_name = self.experiment_names
        exp1_results = self.results[exp1_name]
        exp2_results = self.results[exp2_name]
        
        # Find common patients
        common_patients = set(exp1_results.keys()) & set(exp2_results.keys())
        
        if not common_patients:
            raise ValueError("No common patients found between experiments")
        
        print(f"Comparing {len(common_patients)} common patients")
        print(f"Experiment 1: {exp1_name}")
        print(f"Experiment 2: {exp2_name}\n")
        
        # Determine which models to compare
        if model_name:
            models_to_compare = [model_name]
        else:
            # Get all models from first patient
            first_patient = list(common_patients)[0]
            models_to_compare = list(exp1_results[first_patient].keys())
        
        ttest_results = []
        
        for model in models_to_compare:
            print(f"\n--- Model: {model} ---")
            
            # Collect MAE values for each patient
            exp1_mae = []
            exp2_mae = []
            patient_ids = []
            
            for patient_id in sorted(common_patients):
                if model in exp1_results[patient_id] and model in exp2_results[patient_id]:
                    exp1_mae.append(exp1_results[patient_id][model]['mae'])
                    exp2_mae.append(exp2_results[patient_id][model]['mae'])
                    patient_ids.append(patient_id)
            
            if len(exp1_mae) < 2:
                print(f"  Skipping {model} - insufficient data")
                continue
            
            exp1_mae = np.array(exp1_mae)
            exp2_mae = np.array(exp2_mae)
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(exp1_mae, exp2_mae)
            
            # Calculate effect size (Cohen's d for paired samples)
            diff = exp1_mae - exp2_mae
            cohen_d = np.mean(diff) / np.std(diff)
            
            # Calculate improvement percentage
            improvement_pct = ((exp1_mae - exp2_mae) / exp1_mae * 100).mean()
            
            print(f"  {exp1_name} MAE: {exp1_mae.mean():.3f} ± {exp1_mae.std():.3f} mg/dL")
            print(f"  {exp2_name} MAE: {exp2_mae.mean():.3f} ± {exp2_mae.std():.3f} mg/dL")
            print(f"  Difference: {(exp2_mae - exp1_mae).mean():.3f} mg/dL")
            print(f"  Improvement: {improvement_pct:.1f}%")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Cohen's d: {cohen_d:.3f}")
            
            if p_value < 0.05:
                if exp2_mae.mean() < exp1_mae.mean():
                    print(f"  *** {exp2_name} is SIGNIFICANTLY BETTER (p < 0.05) ***")
                else:
                    print(f"  *** {exp1_name} is SIGNIFICANTLY BETTER (p < 0.05) ***")
            else:
                print(f"  No significant difference (p >= 0.05)")
            
            ttest_results.append({
                'model': model,
                'exp1_name': exp1_name,
                'exp2_name': exp2_name,
                'exp1_mae_mean': exp1_mae.mean(),
                'exp1_mae_std': exp1_mae.std(),
                'exp2_mae_mean': exp2_mae.mean(),
                'exp2_mae_std': exp2_mae.std(),
                'difference': (exp2_mae - exp1_mae).mean(),
                'improvement_pct': improvement_pct,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohen_d': cohen_d,
                'n_patients': len(exp1_mae),
                'significant': p_value < 0.05
            })
        
        results_df = pd.DataFrame(ttest_results)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(results_df[['model', 'exp1_mae_mean', 'exp2_mae_mean', 'improvement_pct', 'p_value', 'significant']])
        
        return results_df
    
    def get_comprehensive_metrics_summary(self) -> pd.DataFrame:
        """
        Get a summary of all available metrics from comprehensive metrics format.
        
        Returns:
            DataFrame with summary statistics for all metrics across experiments
        """
        print(f"\n{'='*60}")
        print("COMPREHENSIVE METRICS SUMMARY")
        print(f"{'='*60}")
        
        data = []
        
        for exp_name, exp_results in self.results.items():
            for patient_id, patient_models in exp_results.items():
                for model_name, metrics in patient_models.items():
                    # Basic metrics
                    row = {
                        'experiment': exp_name,
                        'patient_id': patient_id,
                        'model': model_name,
                        'mae': metrics.get('mae'),
                        'rmse': metrics.get('rmse'),
                        'mape': metrics.get('mape'),
                        'mard': metrics.get('mard'),
                        'clarke_a_b': metrics.get('clarke_a_b'),
                        'parkes_a_b': metrics.get('parkes_a_b')
                    }
                    
                    # TIR metrics
                    tir = metrics.get('tir', {})
                    if isinstance(tir, dict):
                        row.update({
                            'time_in_range': tir.get('time_in_range'),
                            'time_below_range': tir.get('time_below_range'),
                            'time_above_range': tir.get('time_above_range')
                        })
                    
                    # Clarke zones
                    clarke = metrics.get('clarke_zones', {})
                    if isinstance(clarke, dict):
                        row.update({
                            'clarke_a': clarke.get('A'),
                            'clarke_b': clarke.get('B'),
                            'clarke_c': clarke.get('C'),
                            'clarke_d': clarke.get('D'),
                            'clarke_e': clarke.get('E')
                        })
                    
                    # Parkes zones
                    parkes = metrics.get('parkes_zones', {})
                    if isinstance(parkes, dict):
                        row.update({
                            'parkes_a': parkes.get('A'),
                            'parkes_b': parkes.get('B'),
                            'parkes_c': parkes.get('C'),
                            'parkes_d': parkes.get('D'),
                            'parkes_e': parkes.get('E')
                        })
                    
                    data.append(row)
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            print(f"Total records: {len(df)}")
            print(f"Experiments: {df['experiment'].unique()}")
            print(f"Models: {df['model'].unique()}")
            print(f"Patients: {sorted(df['patient_id'].unique())}")
            
            # Show summary statistics for key metrics
            key_metrics = ['mae', 'rmse', 'mape', 'clarke_a_b', 'parkes_a_b', 'time_in_range']
            available_metrics = [m for m in key_metrics if m in df.columns and df[m].notna().any()]
            
            if available_metrics:
                print(f"\nSummary statistics for key metrics:")
                summary = df.groupby(['experiment', 'model'])[available_metrics].agg(['mean', 'std', 'min', 'max'])
                print(summary.round(3))
        
        return df

    def compare_clinical_metrics(self, metric: str = 'clarke_a_b') -> pd.DataFrame:
        """
        Compare clinical metrics (Clarke zones, Parkes zones, TIR) across experiments.
        
        Args:
            metric: Clinical metric to compare ('clarke_a_b', 'parkes_a_b', 'time_in_range', etc.)
        
        Returns:
            DataFrame with comparison results
        """
        print(f"\n{'='*60}")
        print(f"CLINICAL METRIC COMPARISON: {metric.upper()}")
        print(f"{'='*60}")
        
        if len(self.results) != 2:
            raise ValueError("Clinical metric comparison requires exactly 2 experiments")
        
        exp1_name, exp2_name = self.experiment_names
        exp1_results = self.results[exp1_name]
        exp2_results = self.results[exp2_name]
        
        # Find common patients
        common_patients = set(exp1_results.keys()) & set(exp2_results.keys())
        
        if not common_patients:
            raise ValueError("No common patients found between experiments")
        
        # Get all models
        first_patient = list(common_patients)[0]
        models_to_compare = list(exp1_results[first_patient].keys())
        
        comparison_results = []
        
        for model in models_to_compare:
            print(f"\n--- Model: {model} ---")
            
            # Collect metric values for each patient
            exp1_values = []
            exp2_values = []
            patient_ids = []
            
            for patient_id in sorted(common_patients):
                if (model in exp1_results[patient_id] and model in exp2_results[patient_id]):
                    
                    # Extract metric value based on type
                    exp1_val = self._extract_metric_value(exp1_results[patient_id][model], metric)
                    exp2_val = self._extract_metric_value(exp2_results[patient_id][model], metric)
                    
                    if exp1_val is not None and exp2_val is not None:
                        exp1_values.append(exp1_val)
                        exp2_values.append(exp2_val)
                        patient_ids.append(patient_id)
            
            if len(exp1_values) < 2:
                print(f"  Skipping {model} - insufficient data for {metric}")
                continue
            
            exp1_values = np.array(exp1_values)
            exp2_values = np.array(exp2_values)
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(exp1_values, exp2_values)
            
            # Calculate improvement
            improvement = exp2_values.mean() - exp1_values.mean()
            improvement_pct = (improvement / exp1_values.mean() * 100) if exp1_values.mean() != 0 else 0
            
            print(f"  {exp1_name} {metric}: {exp1_values.mean():.3f} ± {exp1_values.std():.3f}")
            print(f"  {exp2_name} {metric}: {exp2_values.mean():.3f} ± {exp2_values.std():.3f}")
            print(f"  Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")
            print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                if improvement > 0:
                    print(f"  *** {exp2_name} is SIGNIFICANTLY BETTER (p < 0.05) ***")
                else:
                    print(f"  *** {exp1_name} is SIGNIFICANTLY BETTER (p < 0.05) ***")
            else:
                print(f"  No significant difference (p >= 0.05)")
            
            comparison_results.append({
                'model': model,
                'metric': metric,
                'exp1_name': exp1_name,
                'exp2_name': exp2_name,
                'exp1_mean': exp1_values.mean(),
                'exp1_std': exp1_values.std(),
                'exp2_mean': exp2_values.mean(),
                'exp2_std': exp2_values.std(),
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                't_statistic': t_stat,
                'p_value': p_value,
                'n_patients': len(exp1_values),
                'significant': p_value < 0.05
            })
        
        return pd.DataFrame(comparison_results)
    
    def _extract_metric_value(self, metrics: Dict, metric_name: str):
        """Extract metric value from metrics dictionary, handling nested structures."""
        if metric_name in metrics:
            return metrics[metric_name]
        
        # Handle nested metrics (TIR, Clarke zones, Parkes zones)
        if 'time_in_range' in metric_name and 'tir' in metrics:
            tir = metrics['tir']
            if isinstance(tir, dict):
                return tir.get('time_in_range')
        
        if 'time_below_range' in metric_name and 'tir' in metrics:
            tir = metrics['tir']
            if isinstance(tir, dict):
                return tir.get('time_below_range')
        
        if 'time_above_range' in metric_name and 'tir' in metrics:
            tir = metrics['tir']
            if isinstance(tir, dict):
                return tir.get('time_above_range')
        
        if 'clarke_' in metric_name and 'clarke_zones' in metrics:
            zone = metric_name.replace('clarke_', '').upper()
            clarke = metrics['clarke_zones']
            if isinstance(clarke, dict):
                return clarke.get(zone)
        
        if 'parkes_' in metric_name and 'parkes_zones' in metrics:
            zone = metric_name.replace('parkes_', '').upper()
            parkes = metrics['parkes_zones']
            if isinstance(parkes, dict):
                return parkes.get(zone)
        
        return None

    def plot_comparison_boxplots(self, save_path: str = None):
        """
        Create boxplots comparing MAE across experiments and models.
        
        Args:
            save_path: Path to save the plot
        """
        print(f"\n{'='*60}")
        print("BOXPLOT COMPARISON")
        print(f"{'='*60}")
        
        # Collect all data
        data = []
        
        for exp_name, exp_results in self.results.items():
            for patient_id, patient_models in exp_results.items():
                for model_name, metrics in patient_models.items():
                    data.append({
                        'Experiment': exp_name,
                        'Model': model_name,
                        'Patient': patient_id,
                        'MAE': metrics['mae']
                    })
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create grouped boxplot
        models = sorted(df['Model'].unique())
        experiments = sorted(df['Experiment'].unique())
        
        positions = []
        labels = []
        box_data = []
        
        for i, model in enumerate(models):
            for j, exp in enumerate(experiments):
                subset = df[(df['Model'] == model) & (df['Experiment'] == exp)]
                if not subset.empty:
                    pos = i * (len(experiments) + 1) + j
                    positions.append(pos)
                    labels.append(f"{model}\n{exp}")
                    box_data.append(subset['MAE'].values)
        
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
        
        # Color boxes by experiment
        colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))
        for i, patch in enumerate(bp['boxes']):
            exp_idx = i % len(experiments)
            patch.set_facecolor(colors[exp_idx])
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0, fontsize=9)
        ax.set_ylabel('MAE (mg/dL)', fontsize=12)
        ax.set_title('Model Performance Comparison Across Experiments', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], label=exp) 
                          for i, exp in enumerate(experiments)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Boxplot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def generate_comparison_report(self, output_dir: str = None):
        """
        Generate a comprehensive comparison report.
        
        Args:
            output_dir: Directory to save the report
        """
        if output_dir is None:
            output_dir = Path.cwd() / "comparison_results"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        # 1. Paired t-test
        ttest_results = self.paired_ttest_comparison()
        ttest_results.to_csv(output_dir / "paired_ttest_results.csv", index=False)
        print(f"\n[OK] Paired t-test results saved to {output_dir / 'paired_ttest_results.csv'}")
        
        # 2. Boxplots
        self.plot_comparison_boxplots(save_path=output_dir / "boxplot_comparison.png")
        
        # 3. Save patient features
        if self.patient_features:
            features_df = pd.DataFrame.from_dict(self.patient_features, orient='index')
            features_df.index.name = 'patient_id'
            features_df.to_csv(output_dir / "patient_features.csv")
        
        print(f"\n[OK] Comparison report saved to {output_dir}")
        print(f"  - paired_ttest_results.csv")
        print(f"  - boxplot_comparison.png")
        print(f"  - patient_features.csv")


def compare_experiments(experiment_dirs: List[str], 
                       experiment_names: List[str] = None,
                       output_dir: str = None
                       ) -> ExperimentComparator:
    """
    Convenience function to compare experiments.
    
    Args:
        experiment_dirs: List of paths to experiment directories
        experiment_names: Optional names for experiments
        output_dir: Directory to save results
    """
    comparator = ExperimentComparator(experiment_dirs, experiment_names)
    comparator.load_experiment_results()
    comparator.generate_comparison_report(output_dir)

    return comparator
