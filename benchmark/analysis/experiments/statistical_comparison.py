"""
Statistical Comparison Framework for Blood Glucose Forecasting Experiments

- Paired t-tests for within-patient comparisons
- Paired Cohen's dz for effect size estimation
- Confidence intervals for the mean per-patient difference

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from ..results_io import ExperimentResults, load_experiments
from .statistical_significance import StatisticalSignificanceTester


class ExperimentComparator:
    """
    Compare results from multiple experiments using statistical tests and visualizations.
    """

    _TTEST_COLUMNS = (
        "model", "exp1_name", "exp2_name", "exp1_mae_mean", "exp1_mae_std",
        "exp2_mae_mean", "exp2_mae_std", "difference", "improvement_pct",
        "t_statistic", "p_value", "cohens_dz", "rank_biserial",
        "effect_direction", "ci_lower", "ci_upper", "wins", "losses", "ties",
        "n_patients", "significant",
    )
    
    def __init__(self,
                 experiment_dirs: List[str],
                 experiment_names: List[str] = None,
                 mode: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Initialize the comparator with experiment directories.

        Args:
            experiment_dirs: Paths to experiment directories.
            experiment_names: Optional labels. Default to each run's configured
                name, qualified by horizon and mode so two horizons of the same
                model do not collide in the report.
            mode: Training mode to compare (regular/transfer). 
            seed: Compare this seed alone instead of the cross-seed mean.
        """
        self.experiment_dirs = [Path(d) for d in experiment_dirs]
        self.mode = mode
        self.seed = seed

        self.experiment_names = list(experiment_names) if experiment_names else None
        if self.experiment_names is not None and len(self.experiment_names) != len(self.experiment_dirs):
            raise ValueError(
                f"{len(self.experiment_names)} names given for {len(self.experiment_dirs)} experiments"
            )

        self.experiments: List[ExperimentResults] = []
        self.results = {}

    def load_experiment_results(self):
        """Load every experiment through the shared results reader."""
        print("Loading experiment results...")

        self.experiments = load_experiments(
            self.experiment_dirs, mode=self.mode, seed=self.seed,
            allow_mixed_horizons=True,
        )
        if self.experiment_names is None:
            self.experiment_names = [experiment.label for experiment in self.experiments]

        duplicates = {
            name for name in self.experiment_names
            if self.experiment_names.count(name) > 1
        }
        if duplicates:
            raise ValueError(
                f"Experiment labels are not unique: {', '.join(sorted(duplicates))}. "
                f"Results are keyed by label, so duplicates would overwrite each other; "
                f"pass experiment_names to distinguish them."
            )

        self.results = {}
        for name, experiment in zip(self.experiment_names, self.experiments):
            self.results[name] = {
                patient_id: {model: dict(metrics) for model, metrics in per_model.items()}
                for patient_id, per_model in experiment.patient_metrics.items()
            }
            print(
                f"  {name}: {len(experiment.patients)} patients, "
                f"{', '.join(experiment.models)}, {experiment.aggregation}"
                + (f" over seeds {', '.join(map(str, experiment.seeds))}" if experiment.seeds else "")
            )

        print(f"\n[OK] Loaded {len(self.results)} experiments")
        return self.results

    def common_models(self):
        """Models present in every loaded experiment."""
        if not self.experiments:
            return []
        shared = set(self.experiments[0].models)
        for experiment in self.experiments[1:]:
            shared &= set(experiment.models)
        return sorted(shared)

    def paired_ttest_comparison(self, model_name: str = None):
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
        
        # Determine which models to compare. Taking them from one experiment
        # would raise later on any model the other one does not have.
        common_models = self.common_models()
        if model_name:
            if model_name not in common_models:
                raise ValueError(
                    f"Model {model_name!r} is not present in both "
                    f"{exp1_name} and {exp2_name}"
                )
            models_to_compare = [model_name]
        else:
            models_to_compare = common_models
        if not models_to_compare:
            raise ValueError(
                f"No model is present in both {exp1_name} and {exp2_name}"
            )
        
        ttest_results = []
        
        for model in models_to_compare:
            print(f"\n--- Model: {model} ---")
            
            # Collect MAE values for each patient
            exp1_mae = []
            exp2_mae = []
            excluded_pairs = 0
            
            for patient_id in sorted(common_patients):
                if model in exp1_results[patient_id] and model in exp2_results[patient_id]:
                    first_value = exp1_results[patient_id][model].get('mae')
                    second_value = exp2_results[patient_id][model].get('mae')
                    try:
                        first_value, second_value = float(first_value), float(second_value)
                    except (TypeError, ValueError):
                        excluded_pairs += 1
                        continue
                    if not (np.isfinite(first_value) and np.isfinite(second_value)):
                        excluded_pairs += 1
                        continue
                    exp1_mae.append(first_value)
                    exp2_mae.append(second_value)

            if excluded_pairs:
                print(f"  Excluding {excluded_pairs} pair(s) with missing or non-finite MAE")
            
            if len(exp1_mae) < 2:
                print(f"  Skipping {model} - insufficient data")
                continue
            
            exp1_mae = np.array(exp1_mae)
            exp2_mae = np.array(exp2_mae)
            
            inference = StatisticalSignificanceTester().paired_comparison(
                exp1_mae, exp2_mae, test_type="parametric"
            )
            t_stat = inference["statistic"]
            p_value = inference["p_value"]
            cohen_d = inference["cohens_dz"]
            
            # Percentage change in the cohort mean. Per-patient ratios would
            # become infinite for a valid zero-MAE baseline.
            improvement_pct = (
                100 * (exp1_mae.mean() - exp2_mae.mean()) / exp1_mae.mean()
                if exp1_mae.mean() != 0 else np.nan
            )
            
            print(f"  {exp1_name} MAE: {exp1_mae.mean():.3f} ± {exp1_mae.std(ddof=1):.3f} mg/dL")
            print(f"  {exp2_name} MAE: {exp2_mae.mean():.3f} ± {exp2_mae.std(ddof=1):.3f} mg/dL")
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
                'exp1_mae_std': exp1_mae.std(ddof=1),
                'exp2_mae_mean': exp2_mae.mean(),
                'exp2_mae_std': exp2_mae.std(ddof=1),
                'difference': (exp2_mae - exp1_mae).mean(),
                'improvement_pct': improvement_pct,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_dz': cohen_d,
                'rank_biserial': inference['rank_biserial'],
                'effect_direction': inference['effect_direction'],
                'ci_lower': inference['ci_lower'],
                'ci_upper': inference['ci_upper'],
                'wins': inference['wins'],
                'losses': inference['losses'],
                'ties': inference['ties'],
                'n_patients': len(exp1_mae),
                'significant': p_value < 0.05
            })
        
        # Keep the return value usable when a shared model has no sufficiently
        # complete patient pairs, rather than failing while selecting summary
        # columns from a column-less empty frame.
        results_df = pd.DataFrame(ttest_results, columns=self._TTEST_COLUMNS)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(results_df[['model', 'exp1_mae_mean', 'exp2_mae_mean', 'improvement_pct', 'p_value', 'significant']])
        
        return results_df

    def get_comprehensive_metrics_summary(self):
        """Return one row per experiment, patient, and model."""
        rows = []
        for experiment_name, experiment_results in self.results.items():
            for patient_id, patient_models in experiment_results.items():
                for model_name, metrics in patient_models.items():
                    row = {
                        "experiment": experiment_name,
                        "patient_id": patient_id,
                        "model": model_name,
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "mape": metrics.get("mape"),
                        "mard": metrics.get("mard"),
                        "clarke_a_b": metrics.get("clarke_a_b"),
                        "parkes_a_b": metrics.get("parkes_a_b"),
                    }
                    for source, prefix in (
                        ("tir", ""), ("clarke_zones", "clarke_"),
                        ("parkes_zones", "parkes_"),
                    ):
                        values = metrics.get(source, {})
                        if isinstance(values, dict):
                            for name, value in values.items():
                                row[f"{prefix}{str(name).lower()}"] = value
                    rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _extract_metric_value(metrics: Dict, metric_name: str):
        """Extract a scalar flat or conventional nested clinical metric."""
        if metric_name in metrics:
            return metrics[metric_name]
        nested = {
            "time_in_range": ("tir", "time_in_range"),
            "time_below_range": ("tir", "time_below_range"),
            "time_above_range": ("tir", "time_above_range"),
        }
        if metric_name in nested:
            parent, child = nested[metric_name]
        elif metric_name.startswith("clarke_"):
            parent, child = "clarke_zones", metric_name.removeprefix("clarke_").upper()
        elif metric_name.startswith("parkes_"):
            parent, child = "parkes_zones", metric_name.removeprefix("parkes_").upper()
        else:
            return None
        values = metrics.get(parent, {})
        return values.get(child) if isinstance(values, dict) else None

    def compare_clinical_metrics(self, metric: str = "clarke_a_b"):
        """Run paired inference for a clinical metric in two experiments."""
        if len(self.results) != 2:
            raise ValueError("Clinical metric comparison requires exactly 2 experiments")
        exp1_name, exp2_name = self.experiment_names
        exp1_results, exp2_results = self.results[exp1_name], self.results[exp2_name]
        common_patients = sorted(set(exp1_results) & set(exp2_results))
        if not common_patients:
            raise ValueError("No common patients found between experiments")

        rows = []
        for model in self.common_models():
            pairs = []
            excluded_pairs = 0
            for patient in common_patients:
                if model not in exp1_results[patient] or model not in exp2_results[patient]:
                    continue
                first_value = self._extract_metric_value(exp1_results[patient][model], metric)
                second_value = self._extract_metric_value(exp2_results[patient][model], metric)
                try:
                    first_value, second_value = float(first_value), float(second_value)
                except (TypeError, ValueError):
                    excluded_pairs += 1
                    continue
                if not (np.isfinite(first_value) and np.isfinite(second_value)):
                    excluded_pairs += 1
                    continue
                pairs.append((first_value, second_value))
            if excluded_pairs:
                print(
                    f"  Excluding {excluded_pairs} pair(s) with missing or non-finite {metric}"
                )
            if len(pairs) < 2:
                continue
            first_values, second_values = zip(*pairs)
            first = np.asarray(first_values, dtype=float)
            second = np.asarray(second_values, dtype=float)
            inference = StatisticalSignificanceTester().paired_comparison(
                first, second, test_type="parametric"
            )
            rows.append({
                "model": model,
                "metric": metric,
                "exp1_name": exp1_name,
                "exp2_name": exp2_name,
                "exp1_mean": float(first.mean()),
                "exp1_std": float(first.std(ddof=1)),
                "exp2_mean": float(second.mean()),
                "exp2_std": float(second.std(ddof=1)),
                "difference": inference["mean_diff"],
                "difference_pct": (
                    100 * inference["mean_diff"] / first.mean()
                    if first.mean() != 0 else np.nan
                ),
                "t_statistic": inference["statistic"],
                "p_value": inference["p_value"],
                "cohens_dz": inference["cohens_dz"],
                "rank_biserial": inference["rank_biserial"],
                "effect_direction": inference["effect_direction"],
                "ci_lower": inference["ci_lower"],
                "ci_upper": inference["ci_upper"],
                "wins": inference["wins"],
                "losses": inference["losses"],
                "ties": inference["ties"],
                "n_patients": len(first),
                "significant": inference["significant"],
                "status": inference["status"],
            })
        return pd.DataFrame(rows)
    
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
                    try:
                        mae = float(metrics.get('mae'))
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(mae):
                        continue
                    data.append({
                        'Experiment': exp_name,
                        'Model': model_name,
                        'Patient': patient_id,
                        'MAE': mae
                    })
        
        df = pd.DataFrame(data)
        if df.empty:
            print("[SKIP] No finite MAE values available for boxplot")
            return False
        
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
        return True
    
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

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")

        written = []

        # 1. The metrics the tests were run on, so the report is checkable.
        metrics_table = pd.concat(
            [experiment.to_frame() for experiment in self.experiments], ignore_index=True
        )
        metrics_table.to_csv(output_dir / "metrics_by_patient.csv", index=False)
        written.append("metrics_by_patient.csv")

        # 2. Paired t-test
        ttest_results = self.paired_ttest_comparison()
        ttest_results.to_csv(output_dir / "paired_ttest_results.csv", index=False)
        written.append("paired_ttest_results.csv")

        # 3. Boxplots
        if self.plot_comparison_boxplots(save_path=output_dir / "boxplot_comparison.png"):
            written.append("boxplot_comparison.png")

        print(f"\n[OK] Comparison report saved to {output_dir}")
        for name in written:
            print(f"  - {name}")


def compare_experiments(experiment_dirs: List[str],
                       experiment_names: List[str] = None,
                       output_dir: str = None,
                       mode: Optional[str] = None,
                       seed: Optional[int] = None
                       ):
    """
    Convenience function to compare experiments.

    Args:
        experiment_dirs: List of paths to experiment directories
        experiment_names: Optional names for experiments
        output_dir: Directory to save results
        mode: Training mode to compare; required only for multi-mode runs
        seed: Compare this seed alone instead of the cross-seed mean
    """
    comparator = ExperimentComparator(experiment_dirs, experiment_names, mode=mode, seed=seed)
    comparator.load_experiment_results()
    comparator.generate_comparison_report(output_dir)

    return comparator
