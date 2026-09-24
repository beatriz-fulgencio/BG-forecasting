"""
Analyses the patient dataset.

Everything here is fixed by the OhioT1DM cohort and its train/test split, so
the numbers are the same whichever architecture, mode or seed produced the run
they were read from: no predicted value and no metric ever feeds a feature.

Signal characterisation
    :mod:`~benchmark.analysis.dataset.distribution_shift` (train->test shift)
    and :mod:`~benchmark.analysis.dataset.signal_features` (irregularity),
    driven by ``RUN/dataset/run_dataset_analysis.py``. They use the reference
    run only to identify the cohort, releases, and sampling rate. Experiment
    analyses read the resulting table when relating these features to error.

Cohort description
    :mod:`~benchmark.analysis.dataset.patient_features` for per-patient level,
    variability, clinical-range and dynamics features over the held-out series,
    and :mod:`~benchmark.analysis.dataset.glucose_history` for the population's
    train and test series read straight from ``raw/ohiot1dm``. They are driven
    by ``RUN/dataset/run_dataset_analysis.py`` and take a loaded run:
    ``patient_features`` for its ``y_true`` and prediction-export timestamps,
    ``glucose_history`` for the cohort and sampling rate.
"""

from importlib import import_module
from typing import Any

# Attribute -> module providing it, resolved lazily by __getattr__ 
_LAZY = {
    "compute_distribution_shift": "distribution_shift",
    "compute_clinical_shift": "clinical_shift",
    "compute_patient_clinical_shifts": "clinical_shift",
    "summarize_clinical_glucose": "clinical_shift",
    "compute_signal_features": "signal_features",
    "compute_patient_features": "patient_features",
    "ground_truth_invariance": "patient_features",
    "GlucoseHistoryAnalysis": "glucose_history",
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
