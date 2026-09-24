"""
Distributional shift metrics between a patient's train and test glucose : how different is this patient's test glucose distribution from their training distribution
"""

from typing import Dict

import numpy as np
import pandas as pd  # type: ignore
from scipy.stats import wasserstein_distance  # type: ignore
from scipy.spatial.distance import jensenshannon  # type: ignore

from ...glucose_ranges import GLUCOSE_PLAUSIBLE_RANGE_MG_DL

GLUCOSE_MIN, GLUCOSE_MAX = GLUCOSE_PLAUSIBLE_RANGE_MG_DL


def _clean_glucose(arr):
    """
    Return glucose values as a float array, dropping NaNs and out-of-range readings 
    """
    values = np.asarray(arr, dtype=float)
    values = values[~np.isnan(values)]
    return values[(values >= GLUCOSE_MIN) & (values <= GLUCOSE_MAX)]


def compute_distribution_shift(train_glucose, test_glucose, n_bins: int = 50):
    """
    Compute shift between train and test glucose distributions.

    Args:
        train_glucose: 1D array-like of training-set glucose values (mg/dL).
        test_glucose: 1D array-like of test-set glucose values (mg/dL).
        n_bins: Number of histogram bins used for the binned metrics.

    Returns:
        Dict with:
        - ``shift_score``: Wasserstein (earth-mover's) distance in mg/dL. This is
          the canonical, parameter-free shift metric. Larger = more shift.
        - ``js_divergence``: Jensen-Shannon divergence (nats) between the test
          and train histograms, over a shared binning of ``n_bins`` bins
          spanning the combined value range. Symmetric and bounded by ln 2.
        - ``out_of_support_mass``: fraction of test readings falling in bins the
          training set never reached. See the note below.
        - ``out_of_support_bins``: how many bins that mass is spread over.
        - ``train_n`` / ``test_n``: valid-sample counts after cleaning.

        If either cleaned array is empty, ``shift_score``, ``js_divergence`` and
        the two support fields are ``np.nan`` (rather than raising).

    Why Jensen-Shannon rather than KL
    ---------------------------------
    KL(test || train) is +inf whenever the test set visits a bin the training
    set never reached, which is ordinary for a real patient. Rescuing it needs
    a smoothing floor, and that floor sets the value it rescues: each such bin
    contributes about ``mass * log(1/eps)``, so KL grows linearly in
    ``log(1/eps)`` with no non-arbitrary place to stop.

    Measured across the 12-patient OhioT1DM cohort, only two patients are
    affected at all (540: 4 bins, 1.105% of test mass; 570: 1 bin, 0.219%),
    but for patient 540 roughly 73% of the KL reported at ``eps=1e-10`` was
    floor rather than measurement -- enough that it ranked as the cohort's most
    divergent patient under the floored KL and only 4th under JS. Worse, the choice of smoothing *scheme* --
    not merely its parameter -- reorders the cohort: under additive (Lidstone)
    smoothing with alpha >= 0.5, patient 570 overtakes 540 as the most-shifted
    patient. An eps-sensitivity check cannot reveal that, because within
    eps-smoothing the ranking is perfectly stable (Spearman 1.000 from 1e-6 to
    1e-14).

    Jensen-Shannon removes the problem rather than parameterising it: it scores
    each distribution against the mixture ``M = (P + Q) / 2``, which has support
    wherever either side does, so it is finite by construction with no floor and
    no free parameter. Across the cohort it agrees with the old floored KL at
    Spearman 0.944; the disagreement is concentrated in the two patients whose
    KL the floor was inflating.

    The support mismatch that used to be smuggled into the KL value is now
    reported directly as ``out_of_support_mass`` -- an interpretable quantity
    ("this fraction of the patient's test readings fall in glucose ranges never
    seen in training") that a reader can check, rather than one scaled by an
    arbitrary constant.

    Reference: Lin, J. (1991), "Divergence measures based on the Shannon
    entropy", IEEE Transactions on Information Theory 37(1), 145-151.
    ``scipy.spatial.distance.jensenshannon`` returns the *distance* (the square
    root of the divergence), so it is squared here to report the divergence.
    """
    train = _clean_glucose(train_glucose)
    test = _clean_glucose(test_glucose)

    result = {
        "shift_score": np.nan,
        "js_divergence": np.nan,
        "out_of_support_mass": np.nan,
        "out_of_support_bins": np.nan,
        "train_n": int(train.size),
        "test_n": int(test.size),
    }

    if train.size == 0 or test.size == 0:
        return result

    # Wasserstein distance: parameter-free, in mg/dL, handles disjoint support.
    result["shift_score"] = float(wasserstein_distance(train, test))

    # Binned metrics over a shared binning of the combined range.
    lo = float(min(train.min(), test.min()))
    hi = float(max(train.max(), test.max()))
    if hi <= lo:
        # Degenerate (all identical) range -> no divergence, nothing unsupported.
        result["js_divergence"] = 0.0
        result["out_of_support_mass"] = 0.0
        result["out_of_support_bins"] = 0
        return result

    bins = np.linspace(lo, hi, n_bins + 1)
    p_train, _ = np.histogram(train, bins=bins, density=False)
    p_test, _ = np.histogram(test, bins=bins, density=False)

    p_train = p_train / p_train.sum()
    p_test = p_test / p_test.sum()

    # Test mass the training set never reached. This is what made KL diverge;
    # reported on its own instead of being folded into the divergence.
    unsupported = (p_train == 0) & (p_test > 0)
    result["out_of_support_mass"] = float(p_test[unsupported].sum())
    result["out_of_support_bins"] = int(unsupported.sum())

    # Jensen-Shannon: finite without smoothing, because the mixture reference
    # has support wherever either distribution does. scipy returns the distance
    # (sqrt of the divergence), so square it.
    result["js_divergence"] = float(jensenshannon(p_test, p_train, base=np.e) ** 2)
    return result


def compute_patient_shift_scores(patient_data: Dict, n_bins: int = 50):
    """
    Compute per-patient distributional shift scores.

    Args:
        patient_data: Mapping ``{patient_id: {'train': df, 'test': df}}`` where
            each DataFrame has a ``'glucose'`` column. This is exactly the
            structure returned by
            ``PopulationAnalysis.load_and_preprocess_glucose_data``.
        n_bins: Number of histogram bins for the binned metrics.

    Returns:
        DataFrame with one row per patient and columns
        ``patient_id, shift_score, js_divergence, out_of_support_mass,
        out_of_support_bins, train_n, test_n``, sorted by ``patient_id``.
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
