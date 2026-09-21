"""
Statistical Significance Testing for Model Comparisons.

This module provides functions to test whether differences in metrics
between experiments (e.g., different prediction horizons) are statistically significant.

- t-tests, Wilcoxon, Mann-Whitney, Shapiro, Levene, ranking, binomial sign test,
  and t quantiles
- Bonferroni, Holm, and Benjamini-Hochberg adjustments
- Array reductions and seeded resampling

"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Mapping, Tuple, Optional, Union
from pathlib import Path

from ..results_io import (
    ExperimentResults,
    load_experiment_results,
    validate_comparison_compatibility,
)
from scipy import stats
from scipy.stats import (
    binomtest,   # Exact paired sign test
    ttest_rel,  # Paired t-test
    wilcoxon,   # Wilcoxon signed-rank test (non-parametric)
    ttest_ind,  # Independent t-test
    mannwhitneyu,  # Mann-Whitney U test (non-parametric)
    shapiro,    # Shapiro-Wilk test for normality
    levene,     # Levene's test for equal variances
)
from statsmodels.stats.multitest import multipletests


def _metric_value(metrics: Dict[str, Any], name: str):
    """Look up a metric by flat name (or dotted path 

    Returns ``None`` when the metric is absent or not numeric.
    """
    head, _, tail = name.partition('.')
    value = metrics.get(head)
    if tail:
        value = value.get(tail) if isinstance(value, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def nested_patient_seed_bootstrap(
    seed_patient_values: Mapping[int, Mapping[Any, float]],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    replicates: int = 10_000,
    random_seed: int = 42,
    confidence_level: float = 0.95,
):
    """Percentile interval with one shared training seed per replicate.

    Each draw first selects exactly one seed, then resamples whole patients from
    that seed. This preserves the dependence among patients evaluated under the
    same fitted model instead of independently mixing seeds within a replicate.
    """
    if not seed_patient_values:
        raise ValueError("At least one seed is required")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    seeds = tuple(sorted(seed_patient_values))
    patient_sets = {seed: set(seed_patient_values[seed]) for seed in seeds}
    expected = patient_sets[seeds[0]]
    for seed in seeds[1:]:
        if patient_sets[seed] != expected:
            raise ValueError(
                f"Seed {seed} covers a different patient cohort from seed {seeds[0]}"
            )
    patients = tuple(sorted(expected, key=str))
    if not patients:
        raise ValueError("At least one patient is required")
    values = {
        seed: np.asarray([seed_patient_values[seed][patient] for patient in patients], dtype=float)
        for seed in seeds
    }
    if any(not np.all(np.isfinite(array)) for array in values.values()):
        raise ValueError("Bootstrap inputs must be finite for every patient and seed")

    per_seed = np.asarray([float(statistic(values[seed])) for seed in seeds], dtype=float)
    point = float(np.mean(per_seed)) if np.all(np.isfinite(per_seed)) else np.nan
    rng = np.random.default_rng(random_seed)
    draws = np.full(replicates, np.nan, dtype=float)
    for index in range(replicates):
        seed = seeds[int(rng.integers(0, len(seeds)))]
        patient_indices = rng.integers(0, len(patients), size=len(patients))
        draws[index] = float(statistic(values[seed][patient_indices]))
    valid = draws[np.isfinite(draws)]
    tail = (1 - confidence_level) / 2
    return {
        "estimate": point,
        "ci_low": float(np.quantile(valid, tail)) if valid.size else np.nan,
        "ci_high": float(np.quantile(valid, 1 - tail)) if valid.size else np.nan,
        "confidence_level": confidence_level,
        "n_patients": len(patients),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "replicates": replicates,
        "valid_replicates": int(valid.size),
        "random_seed": random_seed,
    }


class StatisticalSignificanceTester:
    """
    Test statistical significance of metric differences between experiments.
    
    Supports:
    - Paired tests (same patients across experiments)
    - Independent tests (different patient sets)
    - Parametric tests (t-tests)
    - Non-parametric tests (Wilcoxon, Mann-Whitney)
    - Exact paired sign tests, including explicit wins, losses, and ties
    - Precommitted parametric, Wilcoxon, or sign-test inference
    - Multiple comparison corrections (Bonferroni, Holm, Benjamini-Hochberg)
    - Standardized paired effects and confidence intervals
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical tester.
        
        Args:
            alpha: Significance level (default 0.05)
        """
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not np.isfinite(alpha):
            raise ValueError("alpha must be a finite number strictly between 0 and 1")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        self.alpha = float(alpha)
        self.results = {}
    
    def load_experiment_data(self,
                             experiment_dir: Union[str, Path],
                             mode: Optional[str] = None,
                             seed: Optional[int] = None):
        """
        Load one experiment through the shared results reader.

        Args:
            experiment_dir: Path to an experiment directory 
            mode: Training mode to read; required only for multi-mode runs.
            seed: Read this seed alone instead of the cross-seed mean.

        Returns:
            Dictionary with the loaded :class:`ExperimentResults` and the
            descriptive fields the comparison methods report.
        """
        results = load_experiment_results(
            experiment_dir, mode=mode, seed=seed, with_series=False
        )
        if results.horizon_minutes is None:
            raise ValueError(
                f"{experiment_dir}: prediction horizon is unknown, so this experiment "
                f"cannot be placed on the horizon axis. Runs record it in tracking.json."
            )
        return {
            'name': results.label,
            'path': str(results.experiment_dir),
            'mode': results.mode,
            'seeds': results.seeds,
            'aggregation': results.aggregation,
            'sampling_rate_minutes': results.sampling_rate_minutes,
            'horizon_minutes': results.horizon_minutes,
            'horizon_steps': results.horizon_steps,
            'results': results,
        }

    def extract_patient_metrics(self, experiment_data: Dict,
                                metric_names: List[str]):
        """
        Extract specified metrics for all patients from experiment data.

        Args:
            experiment_data: Experiment dictionary from :meth:`load_experiment_data`
            metric_names: Metric names to extract. Scalars (``mae``, ``rmse``,
                ``clarke_a_b``, ...) and dotted nested names
                (``tir.time_in_range``, ``clarke_zones.A``) are both accepted.

        Returns:
            DataFrame indexed by patient_id with one ``<MODEL>_<metric>`` column
            per model and metric.
        """
        results: ExperimentResults = experiment_data['results']

        rows = []
        for patient_id in results.patients:
            row: Dict[str, Any] = {'patient_id': patient_id}
            # Model names come from the run itself. A hard-coded list silently
            # produced an empty frame for any model outside it.
            for model, metrics in results.patient_metrics[patient_id].items():
                for metric in metric_names:
                    value = _metric_value(metrics, metric)
                    if value is not None:
                        row[f'{model}_{metric}'] = value
            if len(row) > 1:  # has at least one metric
                rows.append(row)

        df = pd.DataFrame(rows)
        if 'patient_id' in df.columns:
            df = df.set_index('patient_id')

        return df

    def test_normality(self, data: np.ndarray):
        """Run SciPy's Shapiro-Wilk implementation."""
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if len(data) < 3 or np.ptp(data) == 0:
            return np.nan, np.nan, False
        stat, p_value = shapiro(data)
        return float(stat), float(p_value), bool(p_value > self.alpha)

    @staticmethod
    def _effect_direction(mean_diff: float):
        """Direction convention used everywhere: experiment 2 minus experiment 1."""
        if not np.isfinite(mean_diff):
            return "undefined"
        if mean_diff > 0:
            return "experiment_2_higher"
        if mean_diff < 0:
            return "experiment_2_lower"
        return "no_difference"

    @staticmethod
    def _paired_effects(differences: np.ndarray) -> Dict[str, float]:
        """Compute paired Cohen's dz and matched-pairs rank-biserial r.

        References:
        - Cohen's dz follows the paired standardized mean difference used by
          Lakens (2013), implemented as mean(diff)/sample-SD(diff). Formula
          conformance is checked against SciPy's paired t statistic
          (``ttest_rel``, BSD-3-Clause), where dz = t/sqrt(n).
        
        """
        std_diff = float(np.std(differences, ddof=1)) if differences.size > 1 else np.nan
        mean_diff = float(np.mean(differences)) if differences.size else np.nan
        cohens_dz = mean_diff / std_diff if np.isfinite(std_diff) and std_diff > 0 else np.nan
        nonzero = differences[differences != 0]
        if nonzero.size:
            ranks = stats.rankdata(np.abs(nonzero))
            w_plus = float(ranks[nonzero > 0].sum())
            w_minus = float(ranks[nonzero < 0].sum())
            rank_biserial = (w_plus - w_minus) / (w_plus + w_minus)
        else:
            rank_biserial = 0.0
        return {
            "std_diff": std_diff,
            "cohens_dz": float(cohens_dz),
            "rank_biserial": float(rank_biserial),
        }

    def paired_sign_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        alternative: str = "two-sided",
    ):
        """Exact paired sign test, excluding ties from the binomial trial."""
        if alternative not in {"two-sided", "less", "greater"}:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
        first = np.asarray(data1, dtype=float)
        second = np.asarray(data2, dtype=float)
        if first.shape != second.shape:
            raise ValueError(f"Paired arrays must have the same shape: {first.shape} != {second.shape}")
        valid = np.isfinite(first) & np.isfinite(second)
        differences = second[valid] - first[valid]
        wins = int(np.count_nonzero(differences > 0))
        losses = int(np.count_nonzero(differences < 0))
        ties = int(np.count_nonzero(differences == 0))
        non_ties = wins + losses
        if non_ties:
            test = binomtest(wins, non_ties, p=0.5, alternative=alternative)
            p_value = float(test.pvalue)
            statistic = wins / non_ties
            sign_biserial = (wins - losses) / non_ties
        else:
            p_value, statistic, sign_biserial = 1.0, 0.5, 0.0
        return {
            "test": "exact_paired_sign_test",
            "n": int(valid.sum()),
            "n_total": int(first.size),
            "n_missing": int(first.size - valid.sum()),
            "n_non_ties": non_ties,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "statistic": float(statistic),
            "p_value": p_value,
            "significant": bool(np.isfinite(p_value) and p_value < self.alpha),
            "sign_biserial": float(sign_biserial),
            "alternative": alternative,
            "status": "all_ties" if non_ties == 0 else "ok",
        }
    
    def paired_comparison(self, 
                         data1: np.ndarray, 
                         data2: np.ndarray,
                         test_type: str = 'auto',
                         alternative: str = 'two-sided') -> Dict:
        """Compare paired observations with an explicit or automatic test."""
        
        aliases = {"wilcoxon": "non-parametric", "sign-test": "sign"}
        test_type = aliases.get(test_type, test_type)
        if test_type not in {"auto", "parametric", "non-parametric", "sign"}:
            raise ValueError("test_type must be auto, parametric, non-parametric, or sign")
        if alternative not in {"two-sided", "less", "greater"}:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
        first = np.asarray(data1, dtype=float)
        second = np.asarray(data2, dtype=float)
        if first.shape != second.shape:
            raise ValueError(f"Paired arrays must have the same shape: {first.shape} != {second.shape}")
        valid = np.isfinite(first) & np.isfinite(second)
        first, second = first[valid], second[valid]
        differences = second - first
        n = len(differences)
        sign = self.paired_sign_test(first, second, alternative=alternative)
        effects = self._paired_effects(differences)
        mean_diff = float(np.mean(differences)) if n else np.nan
        direction = self._effect_direction(mean_diff)

        if n > 1 and np.isfinite(effects["std_diff"]):
            se = effects["std_diff"] / np.sqrt(n)
           
            margin = float(stats.t.ppf(1 - self.alpha / 2, n - 1)) * se
            
            ci_lower, ci_upper = mean_diff - margin, mean_diff + margin
        else:
            ci_lower = ci_upper = np.nan

        base: Dict[str, Any] = {
            "n": n,
            "n_total": int(valid.size),
            "n_missing": int(valid.size - valid.sum()),
            "wins": sign["wins"],
            "losses": sign["losses"],
            "ties": sign["ties"],
            "mean_diff": mean_diff,
            "std_diff": effects["std_diff"],
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": 1 - self.alpha,
            "cohens_dz": effects["cohens_dz"],
            "rank_biserial": effects["rank_biserial"],
            "effect_direction": direction,
            "alternative": alternative,
            "sign_test_p_value": sign["p_value"],
            "sign_test_n_non_ties": sign["n_non_ties"],
            "status": "ok",
        }
        if n == 0:
            return {
                **base, "test": "insufficient_data", "statistic": np.nan,
                "p_value": np.nan, "significant": False,
                "effect_size": np.nan, "effect_size_type": None,
                "status": "insufficient_data",
            }
        if n < 3 and test_type != "sign":
            return {
                **base, "test": "insufficient_data", "statistic": np.nan,
                "p_value": np.nan, "significant": False,
                "effect_size": np.nan, "effect_size_type": None,
                "status": "insufficient_data",
            }
        if test_type == "auto":
            _, _, is_normal = self.test_normality(differences)
            test_type = "parametric" if is_normal else "non-parametric"

        if test_type == "sign":
            result = {
                "test": sign["test"],
                "statistic": sign["statistic"],
                "p_value": sign["p_value"],
            }
            effect_size, effect_name = sign["sign_biserial"], "sign_biserial"
            base["status"] = sign["status"]
        elif np.all(differences == 0):
            result = {
                "test": "paired_t_test" if test_type == "parametric" else "wilcoxon_signed_rank",
                "statistic": 0.0, "p_value": 1.0,
            }
            effect_size = 0.0 if test_type == "non-parametric" else np.nan
            effect_name = "rank_biserial_r" if test_type == "non-parametric" else "cohens_dz"
            base["status"] = "all_ties"
        elif test_type == "parametric":
            effect_size, effect_name = effects["cohens_dz"], "cohens_dz"
            if effects["std_diff"] == 0:
               
                result = {
                    "test": sign["test"],
                    "statistic": sign["statistic"],
                    "p_value": sign["p_value"],
                }
                effect_size, effect_name = sign["sign_biserial"], "sign_biserial"
               
                base["status"] = "constant_difference"
            else:
                scipy_result = ttest_rel(second, first, alternative=alternative)
                result = {
                    "test": "paired_t_test",
                    "statistic": float(scipy_result.statistic),
                    "p_value": float(scipy_result.pvalue),
                }
        else:
            scipy_result = wilcoxon(second, first, alternative=alternative)
            result = {"test": "wilcoxon_signed_rank", "statistic": float(scipy_result.statistic),
                      "p_value": float(scipy_result.pvalue)}
            effect_size, effect_name = effects["rank_biserial"], "rank_biserial_r"

        p_value = result["p_value"]
        return {
            **base, **result,
            "significant": bool(np.isfinite(p_value) and p_value < self.alpha),
            "effect_size": effect_size,
            "effect_size_type": effect_name,
        }
    
    def independent_comparison(self,
                              data1: np.ndarray,
                              data2: np.ndarray,
                              test_type: str = 'auto',
                              alternative: str = 'two-sided'):
        """Compare independent samples through SciPy (BSD-3-Clause)."""
        
        if test_type not in {"auto", "parametric", "non-parametric"}:
            raise ValueError("test_type must be auto, parametric, or non-parametric")
        if alternative not in {"two-sided", "less", "greater"}:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
        first = np.asarray(data1, dtype=float)
        second = np.asarray(data2, dtype=float)
        data1_clean = first[np.isfinite(first)]
        data2_clean = second[np.isfinite(second)]
        
        n1, n2 = len(data1_clean), len(data2_clean)
        
        if n1 < 3 or n2 < 3:
            return {
                'test': 'insufficient_data',
                'n1': n1,
                'n2': n2,
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'effect_size': np.nan,
                'effect_size_type': None,
                'mean_diff': np.nan, 'effect_direction': 'undefined',
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'status': 'insufficient_data',
            }
        
        # Determine test type
        if test_type == 'auto':
            # Test normality
            _, p_norm1, is_normal1 = self.test_normality(data1_clean)
            _, p_norm2, is_normal2 = self.test_normality(data2_clean)
            
            # Test equal variances
            _, p_var = levene(data1_clean, data2_clean)
            equal_var = bool(np.isfinite(p_var) and p_var > self.alpha)
            
            test_type = 'parametric' if (is_normal1 and is_normal2) else 'non-parametric'
        else:
            # A precommitted parametric comparison does not establish equal
            # variances. Welch's test retains the t-test assumption of
            # independent observations while remaining valid when variance or
            # sample size differs between experiments.
            equal_var = False
        
        # Perform test
        if test_type == 'parametric':
            # Independent t-test
            stat, p_value = ttest_ind(data2_clean, data1_clean,
                                     equal_var=equal_var,
                                     alternative=alternative)
            test_name = f'independent_t_test (equal_var={equal_var})'
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((n1-1)*np.var(data1_clean, ddof=1) + 
                                 (n2-1)*np.var(data2_clean, ddof=1)) / (n1 + n2 - 2))
            effect_size = (np.mean(data2_clean) - np.mean(data1_clean)) / pooled_std if pooled_std > 0 else np.nan
            effect_name = 'cohens_d'

        else:
            # Mann-Whitney U test
            stat, p_value = mannwhitneyu(data2_clean, data1_clean, alternative=alternative)
            test_name = 'mann_whitney_u'
            effect_name = 'rank_biserial_r'

            # Rank-biserial correlation
            effect_size = 2.0 * float(stat) / (n1 * n2) - 1.0

        mean_diff = float(np.mean(data2_clean) - np.mean(data1_clean))
        var1 = float(np.var(data1_clean, ddof=1))
        var2 = float(np.var(data2_clean, ddof=1))

        # The interval must rest on the same variance assumption as the test it is reported beside
        if equal_var:
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se2 = pooled_var * (1.0 / n1 + 1.0 / n2)
            df = n1 + n2 - 2
        else:
            se2 = var1 / n1 + var2 / n2
            df = (se2 ** 2 /
                  ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
                  if se2 > 0 else np.nan)

        if se2 > 0 and np.isfinite(df) and df > 0:
            margin = float(stats.t.ppf(1 - self.alpha / 2, df)) * np.sqrt(se2)
            ci_lower, ci_upper = mean_diff - margin, mean_diff + margin
        else:
            ci_lower = ci_upper = mean_diff
        
        return {
            'test': test_name,
            'n1': n1,
            'n2': n2,
            'statistic': stat,
            'p_value': p_value,
            'significant': bool(np.isfinite(p_value) and p_value < self.alpha),
            'effect_size': effect_size,
            'effect_size_type': effect_name,
            'mean1': np.mean(data1_clean),
            'mean2': np.mean(data2_clean),
            'mean_diff': mean_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': 1 - self.alpha,
            'effect_direction': self._effect_direction(mean_diff),
            'alternative': alternative,
            'status': 'ok' if np.isfinite(p_value) else 'undefined_test',
        }
    
    def bonferroni_correction(self, p_values: List[float]):
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of p-values
            
        Returns:
            List of corrected p-values
        """
        return self._multiple_testing_correction(p_values, "bonferroni")
    
    def holm_correction(self, p_values: List[float]):
        """
        Apply Holm-Bonferroni correction for multiple comparisons.
        """
        return self._multiple_testing_correction(p_values, "holm")

    def benjamini_hochberg_correction(self, p_values: List[float]) :
        """Benjamini-Hochberg FDR adjustment
        """
        return self._multiple_testing_correction(p_values, "fdr_bh")

    @staticmethod
    def _multiple_testing_correction(p_values: List[float], method: str) :
        """Adjust finite p-values using statsmodels."""
        
        values = np.asarray(p_values, dtype=float)
        adjusted = np.full(values.shape, np.nan, dtype=float)
        valid = np.isfinite(values)
        if np.any((values[valid] < 0) | (values[valid] > 1)):
            raise ValueError("p-values must lie between 0 and 1")
        if valid.any():
            adjusted[valid] = multipletests(values[valid], method=method)[1]
        return adjusted.tolist()
    
    def compare_experiments(self,
                           experiments: List[Dict],
                           metrics: List[str],
                           models: Optional[List[str]] = None,
                           comparison_type: str = 'paired',
                           correction_method: Optional[str] = 'holm',
                           test_type: str = 'auto',
                           alternative: str = 'two-sided'):
        """
        Compare multiple experiments across specified metrics.

        Args:
            experiments: List of experiment data dictionaries
            metrics: List of metric names to compare
            models: Model names to include. Defaults to the models present in
                every experiment, so the comparison is over a common set.
            comparison_type: 'paired' or 'independent'
            correction_method: 'bonferroni', 'holm', 'fdr_bh'/'bh', or None
            test_type: Precommitted test family: 'parametric',
                'non-parametric', or 'sign'. 'auto' retains the legacy
                normality-screen behaviour.
            alternative: Directional alternative under the consistent
                experiment-2-minus-experiment-1 convention.

        Returns:
            DataFrame with comparison results
        """
        if len(experiments) < 2:
            raise ValueError("Comparing horizons needs at least two experiments")
        if comparison_type not in {"paired", "independent"}:
            raise ValueError("comparison_type must be 'paired' or 'independent'")
        if comparison_type == "independent" and test_type in {"sign", "sign-test"}:
            raise ValueError("The exact sign test requires paired observations")

        validate_comparison_compatibility(
            [exp['results'] for exp in experiments],
            allow_mixed_horizons=True,
            allow_different_patient_cohorts=comparison_type == 'independent',
        )

        if models is None:
            models = sorted(
                set.intersection(*(set(exp['results'].models) for exp in experiments))
            )
            if not models:
                raise ValueError("No model is present in every experiment")

        # Sort experiments by horizon. load_experiment_data guarantees it is known.
        experiments = sorted(experiments, key=lambda x: x['horizon_minutes'])
        
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
                            data1 = df1.loc[common_patients, col1].values
                            data2 = df2.loc[common_patients, col2].values

                            result = self.paired_comparison(
                                data1, data2, test_type=test_type, alternative=alternative
                            )
                        else:
                            data1 = df1[col1].values
                            data2 = df2[col2].values
                            
                            result = self.independent_comparison(
                                data1, data2, test_type=test_type, alternative=alternative
                            )
                        
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
        
        aliases = {"bh": "fdr_bh", "benjamini-hochberg": "fdr_bh"}
        correction_method = aliases.get(correction_method, correction_method)
        supported_corrections = {None, "bonferroni", "holm", "fdr_bh"}
        if correction_method not in supported_corrections:
            raise ValueError(
                "correction_method must be bonferroni, holm, fdr_bh/bh, or None"
            )
        if p_values_for_correction:
            if correction_method == 'bonferroni':
                adjusted = self.bonferroni_correction(p_values_for_correction)
            elif correction_method == 'holm':
                adjusted = self.holm_correction(p_values_for_correction)
            elif correction_method == 'fdr_bh':
                adjusted = self.benjamini_hochberg_correction(p_values_for_correction)
            else:
                adjusted = [float(value) for value in p_values_for_correction]

            for record, adjusted_p in zip(comparison_records, adjusted):
                significant_adjusted = bool(
                    np.isfinite(adjusted_p) and adjusted_p < self.alpha
                )
                record['correction_method'] = correction_method or 'none'
                record['p_value_adjusted'] = adjusted_p
                record['significant_adjusted'] = significant_adjusted
                # Backward-compatible names for existing report consumers.
                record['p_value_corrected'] = adjusted_p
                record['significant_corrected'] = significant_adjusted
        
        results_df = pd.DataFrame(comparison_records)
        
        return results_df
    
    #: Conventional magnitude cut-points per effect-size measure
    EFFECT_SIZE_THRESHOLDS = {
        'cohens_d': (0.2, 0.5, 0.8),
        'cohens_dz': (0.2, 0.5, 0.8),
        'rank_biserial_r': (0.1, 0.3, 0.5),
        'sign_biserial': (0.1, 0.3, 0.5),
    }

    def interpret_effect_size(self, effect_size: float,
                              effect_size_type: str = 'cohens_d') :
        """
        Interpret an effect size against the cut-points for its own measure.

        Args:
            effect_size: Effect size value
            effect_size_type: 'cohens_d' or 'rank_biserial_r'. An unknown
                or missing name falls back to the Cohen's d cut-points

        Returns:
            String interpretation
        """
        if effect_size is None or np.isnan(effect_size):
            return 'undefined'
        small, medium, large = self.EFFECT_SIZE_THRESHOLDS.get(
            effect_size_type, self.EFFECT_SIZE_THRESHOLDS['cohens_d']
        )
        abs_effect = abs(effect_size)

        if abs_effect < small:
            return 'negligible'
        elif abs_effect < medium:
            return 'small'
        elif abs_effect < large:
            return 'medium'
        else:
            return 'large'
    
    def create_summary_report(self, 
                             results_df: pd.DataFrame,
                             output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Create a summary report of statistical tests.
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
        if 'p_value_adjusted' in results_df.columns:
            sig_count = results_df['significant_adjusted'].sum()
            methods = ", ".join(sorted(set(results_df['correction_method'].astype(str))))
            report_lines.append(
                f"Significant differences (adjustment: {methods}): {sig_count} / {len(results_df)}"
            )
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
                    if row.get('effect_direction'):
                        report_lines.append(f"    Direction: {row['effect_direction']}")
                    if all(name in row for name in ('wins', 'losses', 'ties')):
                        report_lines.append(
                            f"    Wins / losses / ties: {row['wins']} / {row['losses']} / {row['ties']}"
                        )
                    
                    if 'ci_lower' in row and not np.isnan(row['ci_lower']):
                        confidence_level = row.get('confidence_level', 1 - self.alpha)
                        report_lines.append(
                            f"    {100 * confidence_level:g}% CI: "
                            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                        )
                    
                    report_lines.append(f"    p-value: {row['p_value']:.6f}")
                    
                    if 'p_value_adjusted' in row:
                        report_lines.append(
                            f"    p-value ({row.get('correction_method', 'adjusted')}): "
                            f"{row['p_value_adjusted']:.6f}"
                        )
                        sig_marker = "***" if row['significant_adjusted'] else ""
                        report_lines.append(
                            f"    Significant (adjusted): {row['significant_adjusted']} {sig_marker}"
                        )
                    else:
                        sig_marker = "***" if row['significant'] else ""
                        report_lines.append(f"    Significant: {row['significant']} {sig_marker}")
                    
                    if 'effect_size' in row and not np.isnan(row['effect_size']):
                        measure = row.get('effect_size_type') or 'cohens_d'
                        interpretation = self.interpret_effect_size(row['effect_size'], measure)
                        report_lines.append(
                            f"    Effect size ({measure}): {row['effect_size']:.4f} ({interpretation})"
                        )
        
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
            "Effect sizes (Cohen's d/dz or rank/sign-biserial r):",
            "  < 0.2  : negligible",
            "  0.2-0.5: small",
            "  0.5-0.8: medium",
            "  > 0.8  : large",
            "",
            "Note: Bonferroni/Holm control family-wise error; Benjamini-Hochberg",
            "controls the false discovery rate within the declared test family.",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        
        return report
