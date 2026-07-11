"""
Distributional shift metrics between a patient's train and test glucose.

Formalizes the "how different is this patient's test glucose distribution from
their training distribution" question into a single per-patient number,
``shift_score`` (the Wasserstein / earth-mover's distance, in mg/dL), plus a
secondary KL divergence. These are used both as reported statistics and as
predictor variables in the per-patient explanatory modelling.

Data note: the preprocessing step marks missing glucose with ``-1``. All inputs
are therefore filtered to physiologically valid readings before any metric is
computed (see :func:`_clean_glucose`).
"""

from typing import Dict

import numpy as np
import pandas as pd  # type: ignore
from scipy.stats import wasserstein_distance, entropy  # type: ignore

# Physiologically plausible CGM range (mg/dL). Values outside this (including the
# ``-1`` missing-value sentinel) are dropped before computing any metric.
GLUCOSE_MIN = 20.0
GLUCOSE_MAX = 600.0

# Smoothing added to histogram densities so empty bins do not send KL to +inf.
_KL_EPSILON = 1e-10


def _clean_glucose(arr) -> np.ndarray:
    """
    Return glucose values as a float array, dropping NaNs and out-of-range
    readings (which removes the ``-1`` missing-value markers).
    """
    values = np.asarray(arr, dtype=float)
    values = values[~np.isnan(values)]
    return values[(values >= GLUCOSE_MIN) & (values <= GLUCOSE_MAX)]


def compute_distribution_shift(train_glucose, test_glucose, n_bins: int = 50) -> Dict:
    """
    Quantify the shift between train and test glucose distributions.

    Args:
        train_glucose: 1D array-like of training-set glucose values (mg/dL).
        test_glucose: 1D array-like of test-set glucose values (mg/dL).
        n_bins: Number of histogram bins used for the KL divergence.

    Returns:
        Dict with:
        - ``shift_score``: Wasserstein (earth-mover's) distance in mg/dL. This is
          the canonical, parameter-free shift metric. Larger = more shift.
        - ``kl_divergence``: KL(test || train), computed over a shared histogram
          of ``n_bins`` bins spanning the combined value range, with epsilon
          smoothing so disjoint support stays finite. Measures how surprising the
          test distribution is under the train distribution.
        - ``train_n`` / ``test_n``: valid-sample counts after cleaning.

        If either cleaned array is empty, ``shift_score`` and ``kl_divergence``
        are ``np.nan`` (rather than raising).
    """
    train = _clean_glucose(train_glucose)
    test = _clean_glucose(test_glucose)

    result = {
        "shift_score": np.nan,
        "kl_divergence": np.nan,
        "train_n": int(train.size),
        "test_n": int(test.size),
    }

    if train.size == 0 or test.size == 0:
        return result

    # Wasserstein distance: parameter-free, in mg/dL, handles disjoint support.
    result["shift_score"] = float(wasserstein_distance(train, test))

    # KL divergence over a shared binning of the combined range.
    lo = float(min(train.min(), test.min()))
    hi = float(max(train.max(), test.max()))
    if hi <= lo:
        # Degenerate (all identical) range -> no divergence.
        result["kl_divergence"] = 0.0
        return result

    bins = np.linspace(lo, hi, n_bins + 1)
    p_train, _ = np.histogram(train, bins=bins, density=True)
    p_test, _ = np.histogram(test, bins=bins, density=True)

    # Smooth and renormalize to proper probability vectors.
    p_train = p_train + _KL_EPSILON
    p_test = p_test + _KL_EPSILON
    p_train = p_train / p_train.sum()
    p_test = p_test / p_test.sum()

    result["kl_divergence"] = float(entropy(p_test, p_train))
    return result


def compute_patient_shift_scores(patient_data: Dict, n_bins: int = 50) -> pd.DataFrame:
    """
    Compute per-patient distributional shift scores.

    Args:
        patient_data: Mapping ``{patient_id: {'train': df, 'test': df}}`` where
            each DataFrame has a ``'glucose'`` column. This is exactly the
            structure returned by
            ``PopulationAnalysis.load_and_preprocess_glucose_data``.
        n_bins: Number of histogram bins for the KL divergence.

    Returns:
        DataFrame with one row per patient and columns
        ``patient_id, shift_score, kl_divergence, train_n, test_n``,
        sorted by ``patient_id``.
    """
    rows = []
    for patient_id in sorted(patient_data.keys()):
        train_df = patient_data[patient_id]["train"]
        test_df = patient_data[patient_id]["test"]

        shift = compute_distribution_shift(
            train_df["glucose"].values,
            test_df["glucose"].values,
            n_bins=n_bins,
        )
        rows.append({"patient_id": patient_id, **shift})

    return pd.DataFrame(rows)
