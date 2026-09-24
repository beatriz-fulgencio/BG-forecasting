"""
Analyses of what a model did: read completed runs and compare them.

This is the reading half. :mod:`benchmark.experiments` is the writing half, 
it runs the experiments these analyses then read.

Every analysis here reaches its numbers through
:mod:`benchmark.analysis.results_io`, so none of them parses a directory name,
guesses a model, or hard-codes a metrics filename.

Comparing experiments
    :class:`~benchmark.analysis.experiments.statistical_comparison.ExperimentComparator`
    for two runs side by side;
    :class:`~benchmark.analysis.experiments.statistical_significance.StatisticalSignificanceTester`
    for significance across horizons.

Within one experiment
    :class:`~benchmark.analysis.experiments.patient_analysis.PatientAnalyzer` for
    per-patient performance, and
    :class:`~benchmark.analysis.experiments.population_analysis.PopulationAnalysis`
    for the population view. The descriptive glucose features these join
    against are a dataset property and come from
    :mod:`benchmark.analysis.dataset.patient_features` -> in the dataset module, not here

Localizing error
    :mod:`~benchmark.analysis.experiments.error_localization` splits one patient's
    error by ISO glycemic range and by rate of change, and draws the two per-patient figures, driven by
    ``RUN/figures/run_error_localization_figures.py``;
    :mod:`~benchmark.analysis.experiments.rapid_change` and
    :mod:`~benchmark.analysis.experiments.persistence` for the rate-of-change and
    baseline views;
    :mod:`~benchmark.analysis.experiments.error_by_range` for the cohort view of the same glycemic split, with nested patient x seed intervals, driven by
    ``RUN/experiments/run_error_by_range_analysis.py``. It uses the coarse three bands; ``error_localization`` draws the five ISO bands that nest inside them.
    :mod:`~benchmark.analysis.experiments.zone_d` applies the same nested
    bootstrap to the Clarke zone-D decomposition driven by ``RUN/experiments/run_zone_d_analysis.py``.

Across runs
    :mod:`~benchmark.analysis.experiments.transfer_inference` (transfer minus regular) and :mod:`~benchmark.analysis.experiments.patient_stability` (does the per-patient ranking hold up?).

Figures
    :mod:`~benchmark.analysis.experiments.shift_mae_plot` draws per-patient distributional shift against MAE. The shift itself is a dataset property (:mod:`benchmark.analysis.dataset.distribution_shift`); the error it is plotted against is not, which is why the figure lives here.

The analysis classes pull in matplotlib and scikit-learn, so they are
imported on first access rather than at package import: reading results should
not cost a plotting stack.
"""

from importlib import import_module
from typing import Any

# Attribute -> module providing it, resolved lazily by __getattr__ 
_LAZY = {
    "ExperimentComparator": "statistical_comparison",
    "compare_experiments": "statistical_comparison",
    "StatisticalSignificanceTester": "statistical_significance",
    "nested_patient_seed_bootstrap": "statistical_significance",
    "analyze_persistence_runs": "persistence",
    "persistence_metrics": "persistence",
    "plot_persistence": "persistence",
    "band_error_table": "error_localization",
    "condition_error_table": "error_localization",
    "iso_band_masks": "error_localization",
    "load_patient_points": "error_localization",
    "plot_clarke_localized": "error_localization",
    "plot_error_localization": "error_localization",
    "plot_error_localization_paper": "error_localization",
    "level_bin_table": "error_localization",
    "analyze_zone_d_runs": "zone_d",
    "analyze_zone_d_run": "zone_d",
    "patient_zone_d_counts": "zone_d",
    "analyze_error_by_range_runs": "error_by_range",
    "analyze_error_by_range_run": "error_by_range",
    "band_masks": "error_by_range",
    "patient_band_sums": "error_by_range",
    "patient_bootstrap": "error_by_range",
    "analyze_rapid_change_runs": "rapid_change",
    "rapid_change_masks": "rapid_change",
    "rapid_change_metrics": "rapid_change",
    "plot_rapid_change": "rapid_change",
    "analyze_transfer_inference": "transfer_inference",
    "paired_transfer_bootstrap": "transfer_inference",
    "plot_transfer_inference": "transfer_inference",
    "load_patient_stability": "patient_stability",
    "summarize_patient_stability": "patient_stability",
    "PatientAnalyzer": "patient_analysis",
    "leave_one_patient_out_explanatory_models": "patient_analysis",
    "analyze_patients": "patient_analysis",
    "PopulationAnalysis": "population_analysis",
    "plot_shift_vs_mae": "shift_mae_plot",
    "plot_shift_vs_mae_faceted": "shift_mae_plot",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    """Import an analysis module the first time one of its names is used."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module_name}", __name__), name)


def __dir__():
    return sorted(__all__)
