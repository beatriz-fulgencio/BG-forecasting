"""
Statistical Significance Testing for Model Comparisons.

This module provides functions to test whether differences in metrics
between experiments (e.g., different prediction horizons) are statistically significant.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from scipy import stats
from scipy.stats import (
    ttest_rel,  # Paired t-test
    wilcoxon,   # Wilcoxon signed-rank test (non-parametric)
    ttest_ind,  # Independent t-test
    mannwhitneyu,  # Mann-Whitney U test (non-parametric)
    shapiro,    # Shapiro-Wilk test for normality
    levene,     # Levene's test for equal variances
    friedmanchisquare  # Friedman test (non-parametric repeated measures)
)
import warnings


class StatisticalSignificanceTester:
    """
    Test statistical significance of metric differences between experiments.
    
    Supports:
    - Paired tests (same patients across experiments)
    - Independent tests (different patient sets)
    - Parametric tests (t-tests)
    - Non-parametric tests (Wilcoxon, Mann-Whitney)
    - Multiple comparison corrections (Bonferroni, Holm)
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical tester.
        
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
        self.results = {}
    
    def load_experiment_data(self, experiment_dir: Union[str, Path]) -> Dict:
        """
        Load experiment data from directory.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            Dictionary with experiment data
        """
        experiment_dir = Path(experiment_dir)
        
        # Find comprehensive metrics file
        metrics_files = list(experiment_dir.glob("comprehensive_metrics_*.json"))
        if not metrics_files:
            raise FileNotFoundError(f"No comprehensive_metrics_*.json found in {experiment_dir}")
        
        metrics_file = metrics_files[0]
        
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # Extract experiment name and prediction horizon
        exp_name = experiment_dir.name
        
        # Try to extract prediction horizon from name
        import re
        horizon_match = re.search(r'(\d+)min', exp_name)
        horizon_minutes = int(horizon_match.group(1)) if horizon_match else None
        
        return {
            'name': exp_name,
            'path': str(experiment_dir),
            'horizon_minutes': horizon_minutes,
            'horizon_steps': horizon_minutes // 5 if horizon_minutes else None,
            'data': data
        }
    
    def extract_patient_metrics(self, experiment_data: Dict, 
                                metric_names: List[str]) -> pd.DataFrame:
        """
        Extract specified metrics for all patients from experiment data.
        
        Args:
            experiment_data: Experiment data dictionary
            metric_names: List of metric names to extract (e.g., ['rmse', 'mae'])
            
        Returns:
            DataFrame with patient_id as index and metrics as columns
        """
        data = experiment_data['data']
        
        # Handle different data structures
        if 'patient_metrics' in data:
            patient_data_dict = data['patient_metrics']
        else:
            patient_data_dict = data
        
        rows = []
        
        for patient_id, patient_data in patient_data_dict.items():
            if not isinstance(patient_data, dict):
                continue
            
            row = {'patient_id': patient_id}
            
            # Try to extract metrics from different models
            for model_name in ['GRU', 'LSTM', 'RNN']:
                if model_name in patient_data:
                    model_data = patient_data[model_name]
                    if isinstance(model_data, dict):
                        for metric in metric_names:
                            if metric in model_data:
                                row[f'{model_name}_{metric}'] = model_data[metric]
            
            if len(row) > 1:  # Has at least one metric
                rows.append(row)
        
        df = pd.DataFrame(rows)
        if 'patient_id' in df.columns:
            df = df.set_index('patient_id')
        
        return df
    
    def test_normality(self, data: np.ndarray) -> Tuple[float, float, bool]:
        """
        Test if data follows normal distribution using Shapiro-Wilk test.
        
        Args:
            data: Data array
            
        Returns:
            Tuple of (statistic, p-value, is_normal)
        """
        if len(data) < 3:
            return np.nan, np.nan, False
        
        stat, p_value = shapiro(data)
        is_normal = p_value > self.alpha
        
        return stat, p_value, is_normal
    
    def paired_comparison(self, 
                         data1: np.ndarray, 
                         data2: np.ndarray,
                         test_type: str = 'auto',
                         alternative: str = 'two-sided') -> Dict:
        """
        Perform paired comparison between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            test_type: 'auto', 'parametric', or 'non-parametric'
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN pairs
        mask = ~(np.isnan(data1) | np.isnan(data2))
        data1_clean = data1[mask]
        data2_clean = data2[mask]
        
        n = len(data1_clean)
        
        if n < 3:
            return {
                'test': 'insufficient_data',
                'n': n,
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'effect_size': np.nan,
                'mean_diff': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan
            }
        
        # Calculate descriptive statistics
        mean_diff = np.mean(data2_clean - data1_clean)
        std_diff = np.std(data2_clean - data1_clean, ddof=1)
        
        # Determine test type
        if test_type == 'auto':
            # Test normality of differences
            _, p_norm, is_normal = self.test_normality(data2_clean - data1_clean)
            test_type = 'parametric' if is_normal else 'non-parametric'
        
        # Perform test
        if test_type == 'parametric':
            # Paired t-test
            stat, p_value = ttest_rel(data1_clean, data2_clean, alternative=alternative)
            test_name = 'paired_t_test'
            
            # Calculate effect size (Cohen's d for paired samples)
            effect_size = mean_diff / std_diff if std_diff > 0 else np.nan
            
        else:
            # Wilcoxon signed-rank test
            stat, p_value = wilcoxon(data1_clean, data2_clean, alternative=alternative)
            test_name = 'wilcoxon_signed_rank'
            
            # Calculate effect size (r = Z / sqrt(N))
            z_score = stat / np.sqrt(n)
            effect_size = z_score / np.sqrt(n)
        
        # Calculate confidence interval for mean difference
        se = std_diff / np.sqrt(n)
        ci_margin = stats.t.ppf(1 - self.alpha/2, n - 1) * se
        ci_lower = mean_diff - ci_margin
        ci_upper = mean_diff + ci_margin
        
        return {
            'test': test_name,
            'n': n,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'alternative': alternative
        }
    
    def independent_comparison(self,
                              data1: np.ndarray,
                              data2: np.ndarray,
                              test_type: str = 'auto',
                              alternative: str = 'two-sided') -> Dict:
        """
        Perform independent comparison between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            test_type: 'auto', 'parametric', or 'non-parametric'
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        data1_clean = data1[~np.isnan(data1)]
        data2_clean = data2[~np.isnan(data2)]
        
        n1, n2 = len(data1_clean), len(data2_clean)
        
        if n1 < 3 or n2 < 3:
            return {
                'test': 'insufficient_data',
                'n1': n1,
                'n2': n2,
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'effect_size': np.nan
            }
        
        # Determine test type
        if test_type == 'auto':
            # Test normality
            _, p_norm1, is_normal1 = self.test_normality(data1_clean)
            _, p_norm2, is_normal2 = self.test_normality(data2_clean)
            
            # Test equal variances
            _, p_var = levene(data1_clean, data2_clean)
            equal_var = p_var > self.alpha
            
            test_type = 'parametric' if (is_normal1 and is_normal2) else 'non-parametric'
        else:
            equal_var = True
        
        # Perform test
        if test_type == 'parametric':
            # Independent t-test
            stat, p_value = ttest_ind(data1_clean, data2_clean, 
                                     equal_var=equal_var,
                                     alternative=alternative)
            test_name = f'independent_t_test (equal_var={equal_var})'
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((n1-1)*np.var(data1_clean, ddof=1) + 
                                 (n2-1)*np.var(data2_clean, ddof=1)) / (n1 + n2 - 2))
            effect_size = (np.mean(data2_clean) - np.mean(data1_clean)) / pooled_std if pooled_std > 0 else np.nan
            
        else:
            # Mann-Whitney U test
            stat, p_value = mannwhitneyu(data1_clean, data2_clean, alternative=alternative)
            test_name = 'mann_whitney_u'
            
            # Calculate effect size (r = Z / sqrt(N))
            n_total = n1 + n2
            z_score = (stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n_total + 1) / 12)
            effect_size = z_score / np.sqrt(n_total)
        
        return {
            'test': test_name,
            'n1': n1,
            'n2': n2,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'mean1': np.mean(data1_clean),
            'mean2': np.mean(data2_clean),
            'mean_diff': np.mean(data2_clean) - np.mean(data1_clean),
            'alternative': alternative
        }
    
    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of p-values
            
        Returns:
            List of corrected p-values
        """
        n = len(p_values)
        return [min(p * n, 1.0) for p in p_values]
    
    def holm_correction(self, p_values: List[float]) -> List[float]:
        """
        Apply Holm-Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of p-values
            
        Returns:
            List of corrected p-values
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        corrected = np.zeros(n)
        for i, p in enumerate(sorted_p_values):
            corrected[sorted_indices[i]] = min(p * (n - i), 1.0)
        
        # Ensure monotonicity
        for i in range(1, n):
            corrected[sorted_indices[i]] = max(corrected[sorted_indices[i]], 
                                              corrected[sorted_indices[i-1]])
        
        return corrected.tolist()
    
    def compare_experiments(self,
                           experiments: List[Dict],
                           metrics: List[str],
                           models: List[str] = ['GRU', 'LSTM', 'RNN'],
                           comparison_type: str = 'paired',
                           correction_method: Optional[str] = 'holm') -> pd.DataFrame:
        """
        Compare multiple experiments across specified metrics.
        
        Args:
            experiments: List of experiment data dictionaries
            metrics: List of metric names to compare
            models: List of model names to include
            comparison_type: 'paired' or 'independent'
            correction_method: 'bonferroni', 'holm', or None
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        # Sort experiments by horizon
        experiments = sorted(experiments, key=lambda x: x.get('horizon_minutes', 0))
        
        # Extract metric data for all experiments
        exp_metrics = []
        for exp in experiments:
            df = self.extract_patient_metrics(exp, metrics)
            exp_metrics.append(df)
        
        # Pairwise comparisons
        p_values_for_correction = []
        comparison_records = []
        
        for i in range(len(experiments)):
            for j in range(i + 1, len(experiments)):
                exp1, exp2 = experiments[i], experiments[j]
                df1, df2 = exp_metrics[i], exp_metrics[j]
                
                for model in models:
                    for metric in metrics:
                        col1 = f'{model}_{metric}'
                        col2 = f'{model}_{metric}'
                        
                        if col1 not in df1.columns or col2 not in df2.columns:
                            continue
                        
                        # Align by patient_id for paired comparison
                        if comparison_type == 'paired':
                            common_patients = df1.index.intersection(df2.index)
                            if len(common_patients) < 3:
                                continue
                            
                            data1 = df1.loc[common_patients, col1].values
                            data2 = df2.loc[common_patients, col2].values
                            
                            result = self.paired_comparison(data1, data2)
                        else:
                            data1 = df1[col1].values
                            data2 = df2[col2].values
                            
                            result = self.independent_comparison(data1, data2)
                        
                        # Store result
                        record = {
                            'experiment_1': exp1['name'],
                            'horizon_1_min': exp1['horizon_minutes'],
                            'experiment_2': exp2['name'],
                            'horizon_2_min': exp2['horizon_minutes'],
                            'model': model,
                            'metric': metric,
                            **result
                        }
                        
                        comparison_records.append(record)
                        p_values_for_correction.append(result['p_value'])
        
        # Apply multiple comparison correction
        if correction_method and p_values_for_correction:
            if correction_method == 'bonferroni':
                corrected_p = self.bonferroni_correction(p_values_for_correction)
            elif correction_method == 'holm':
                corrected_p = self.holm_correction(p_values_for_correction)
            else:
                corrected_p = p_values_for_correction
            
            for i, record in enumerate(comparison_records):
                record['p_value_corrected'] = corrected_p[i]
                record['significant_corrected'] = corrected_p[i] < self.alpha
        
        results_df = pd.DataFrame(comparison_records)
        
        return results_df
    
    def interpret_effect_size(self, effect_size: float) -> str:
        """
        Interpret Cohen's d or correlation coefficient r.
        
        Args:
            effect_size: Effect size value
            
        Returns:
            String interpretation
        """
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return 'negligible'
        elif abs_effect < 0.5:
            return 'small'
        elif abs_effect < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def create_summary_report(self, 
                             results_df: pd.DataFrame,
                             output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Create a human-readable summary report of statistical tests.
        
        Args:
            results_df: DataFrame with comparison results
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 80,
            "STATISTICAL SIGNIFICANCE ANALYSIS",
            "=" * 80,
            "",
            f"Significance level (alpha): {self.alpha}",
            f"Total comparisons: {len(results_df)}",
            ""
        ]
        
        # Summary statistics
        if 'p_value_corrected' in results_df.columns:
            sig_count = results_df['significant_corrected'].sum()
            report_lines.append(f"Significant differences (corrected): {sig_count} / {len(results_df)}")
        else:
            sig_count = results_df['significant'].sum()
            report_lines.append(f"Significant differences: {sig_count} / {len(results_df)}")
        
        report_lines.extend(["", "=" * 80, "DETAILED RESULTS BY MODEL AND METRIC", "=" * 80, ""])
        
        # Group by model and metric
        for model in results_df['model'].unique():
            report_lines.append(f"\n{model} Model:")
            report_lines.append("-" * 80)
            
            model_df = results_df[results_df['model'] == model]
            
            for metric in model_df['metric'].unique():
                metric_df = model_df[model_df['metric'] == metric]
                
                report_lines.append(f"\n  Metric: {metric.upper()}")
                report_lines.append("  " + "-" * 76)
                
                for _, row in metric_df.iterrows():
                    h1 = row['horizon_1_min']
                    h2 = row['horizon_2_min']
                    
                    report_lines.append(f"\n  Comparison: {h1} min vs {h2} min")
                    report_lines.append(f"    Test: {row['test']}")
                    report_lines.append(f"    Sample size: {row.get('n', row.get('n1', 'N/A'))}")
                    
                    if 'mean_diff' in row:
                        report_lines.append(f"    Mean difference: {row['mean_diff']:.4f}")
                    
                    if 'ci_lower' in row and not np.isnan(row['ci_lower']):
                        report_lines.append(f"    95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
                    
                    report_lines.append(f"    p-value: {row['p_value']:.6f}")
                    
                    if 'p_value_corrected' in row:
                        report_lines.append(f"    p-value (corrected): {row['p_value_corrected']:.6f}")
                        sig_marker = "***" if row['significant_corrected'] else ""
                        report_lines.append(f"    Significant (corrected): {row['significant_corrected']} {sig_marker}")
                    else:
                        sig_marker = "***" if row['significant'] else ""
                        report_lines.append(f"    Significant: {row['significant']} {sig_marker}")
                    
                    if 'effect_size' in row and not np.isnan(row['effect_size']):
                        interpretation = self.interpret_effect_size(row['effect_size'])
                        report_lines.append(f"    Effect size: {row['effect_size']:.4f} ({interpretation})")
        
        report_lines.extend([
            "",
            "=" * 80,
            "INTERPRETATION GUIDE",
            "=" * 80,
            "",
            "Significance levels:",
            "  *** : p < 0.05 (significant)",
            "      : p >= 0.05 (not significant)",
            "",
            "Effect sizes (Cohen's d or r):",
            "  < 0.2  : negligible",
            "  0.2-0.5: small",
            "  0.5-0.8: medium",
            "  > 0.8  : large",
            "",
            "Note: Multiple comparison corrections (Bonferroni/Holm) control the",
            "family-wise error rate when performing multiple statistical tests.",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        
        return report
