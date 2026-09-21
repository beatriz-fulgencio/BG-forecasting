"""
Analysis of the benchmark, split by: 

:mod:`benchmark.analysis.dataset`
    Analyses whose output is fixed by the OhioT1DM cohort and its train/test
    split: train->test distributional shift, test-series irregularity,
    per-patient glucose features, and the population's glucose history. 

:mod:`benchmark.analysis.experiments`
    Analyses of what a model did: testing across horizons and between experiments, 
    error localization by glycemic range and rate of change, persistence baselines, 
    transfer inference, patient stability, and the patient/population performance views. 

:mod:`benchmark.analysis.results_io`
    The one reader both halves share. It turns any supported results layout
    into an :class:`~benchmark.analysis.results_io.ExperimentResults`, so no
    analysis parses a directory name, guesses a model, or hard-codes a metrics
    filename. 
    
Reference implementations
    Every statistic computed under this package is cross-referenced against an
    external open-source implementation.
"""

from .results_io import (
    ExperimentResults,
    ResultsLayoutError,
    attach_prediction_context,
    get_modes,
    get_seeds,
    load_experiment_results,
    load_experiments,
    load_predictions,
    reconstruct_prediction_context,
    require_grouping_metadata,
    resolve_horizon_minutes,
    validate_comparison_compatibility,
    validate_prediction_context,
)

__all__ = [
    "ExperimentResults",
    "ResultsLayoutError",
    "attach_prediction_context",
    "get_modes",
    "get_seeds",
    "load_experiment_results",
    "load_experiments",
    "load_predictions",
    "reconstruct_prediction_context",
    "require_grouping_metadata",
    "resolve_horizon_minutes",
    "validate_comparison_compatibility",
    "validate_prediction_context",
]
