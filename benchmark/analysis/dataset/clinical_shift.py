"""Clinical train-to-test glucose-profile change metrics.

This module complements :mod:`distribution_shift`.  It deliberately does not
collapse the clinical profile into one scalar: a change in time below range is
not interchangeable with an equal-sized change in time above range.

The reported ranges follow the International Consensus on Time in Range
(Battelino et al., *Diabetes Care*, 2019, doi:10.2337/dci19-0028): TIR
70--180 mg/dL, TBR <70 and <54 mg/dL, and TAR >180 and >250 mg/dL. Values are
percentages of valid CGM readings, so they represent time only when sampling
is regular; callers should report data coverage when timestamps or missingness
are available.
"""

from typing import Dict

import numpy as np
import pandas as pd

from ...glucose_ranges import GLUCOSE_PLAUSIBLE_RANGE_MG_DL


GLUCOSE_MIN, GLUCOSE_MAX = GLUCOSE_PLAUSIBLE_RANGE_MG_DL

_SUMMARY_FIELDS = (
    "mean_glucose_mg_dl",
    "glucose_cv_percent",
    "tir_70_180_percent",
    "tbr_lt_70_percent",
    "tbr_lt_54_percent",
    "tar_gt_180_percent",
    "tar_gt_250_percent",
)


def _clean_glucose(values):
    """Convert a glucose series to finite, physiologically plausible values."""
    glucose = np.asarray(values, dtype=float).reshape(-1)
    return glucose[
        np.isfinite(glucose)
        & (glucose >= GLUCOSE_MIN)
        & (glucose <= GLUCOSE_MAX)
    ]


def summarize_clinical_glucose(glucose):
    """Summarize one CGM series using consensus clinical glucose metrics.

    Returns NaN for every metric when no valid measurements remain.  CV is
    ``100 * population SD / mean glucose``; the population convention matches
    the descriptive analysis elsewhere in this package.
    """
    values = _clean_glucose(glucose)
    summary = {field: np.nan for field in _SUMMARY_FIELDS}
    summary["n"] = int(values.size)
    if not values.size:
        return summary

    mean = float(np.mean(values))
    summary.update({
        "mean_glucose_mg_dl": mean,
        "glucose_cv_percent": float(100.0 * np.std(values) / mean),
        "tir_70_180_percent": float(100.0 * np.mean((values >= 70) & (values <= 180))),
        "tbr_lt_70_percent": float(100.0 * np.mean(values < 70)),
        "tbr_lt_54_percent": float(100.0 * np.mean(values < 54)),
        "tar_gt_180_percent": float(100.0 * np.mean(values > 180)),
        "tar_gt_250_percent": float(100.0 * np.mean(values > 250)),
    })
    return summary


def compute_clinical_shift(train_glucose, test_glucose):
    """Compare clinical CGM summaries between a patient's train and test data.

    Each ``*_train`` and ``*_test`` field gives the corresponding profile
    metric.  Each ``delta_*`` field is **test minus train** (mg/dL for mean,
    percentage points for CV and range metrics).  No overall score is returned:
    worsening hypoglycemia must remain distinguishable from hyperglycemia.
    """
    train = summarize_clinical_glucose(train_glucose)
    test = summarize_clinical_glucose(test_glucose)
    result = {
        "train_n": train.pop("n"),
        "test_n": test.pop("n"),
    }
    for field in _SUMMARY_FIELDS:
        result[f"{field}_train"] = train[field]
        result[f"{field}_test"] = test[field]
        result[f"delta_{field}"] = test[field] - train[field]
    return result


def compute_patient_clinical_shifts(patient_data: Dict):
    """Compute one clinical train-to-test profile comparison per patient.

    ``patient_data`` maps patient IDs to ``{'train': df, 'test': df}`` entries,
    each with a ``glucose`` column, matching ``compute_patient_shift_scores``.
    """
    rows = []
    for patient_id in sorted(patient_data):
        series = patient_data[patient_id]
        shift = compute_clinical_shift(
            series["train"]["glucose"].values,
            series["test"]["glucose"].values,
        )
        rows.append({"patient_id": patient_id, **shift})
    return pd.DataFrame(rows)
