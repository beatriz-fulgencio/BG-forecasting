"""
Per-patient descriptive features of the forecastable held-out glucose targets.

Level, variability, clinical-range occupancy and dynamics are computed from a
patient's ``y_true`` values in a prediction export: the test-split targets for
which the configured model could construct a forecast. This is deliberately a
forecast-target description, not a full raw-CGM clinical profile; use
``clinical_shift`` for the latter's train-to-test comparison.

Dynamics are derived from actual target timestamps whenever a configured
prediction export is available, via :func:`benchmark.analysis.results_io.load_predictions`. 
"""

from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from ..results_io import ExperimentResults, ResultsLayoutError, load_predictions
from ...glucose_ranges import GLUCOSE_PLAUSIBLE_RANGE_MG_DL
from ..experiments.rapid_change import RAPID_RATE_MG_DL_PER_MIN, rapid_threshold_mg_dl


GLUCOSE_MIN, GLUCOSE_MAX = GLUCOSE_PLAUSIBLE_RANGE_MG_DL


def _valid_glucose_mask(values: np.ndarray):
    """Return finite, physiologically plausible target readings."""
    return (np.isfinite(values) & (values >= GLUCOSE_MIN) & (values <= GLUCOSE_MAX))


def _reference_truth(per_model: Dict[str, Dict[str, Any]], patient_id: int):
    """Choose a usable target series and reject disagreeing model exports."""
    available = [
        (model, np.asarray(metrics["y_true"], dtype=float).reshape(-1))
        for model, metrics in per_model.items()
        if metrics.get("y_true") is not None
    ]
    if not available:
        return None, None
    model, reference = available[0]
    for other_model, other in available[1:]:
        if other.shape != reference.shape or not np.array_equal(other, reference, equal_nan=True):
            raise ResultsLayoutError(
                f"patient {patient_id}: reference glucose differs between "
                f"{model} and {other_model}"
            )
    return model, reference


def _context_dir(results: ExperimentResults):
    """Directory holding the prediction export for the selected seed."""
    run_dir = Path(results.experiment_dir)
    if run_dir.name.startswith("seed_"):
        return run_dir
    if results.seeds:
        return run_dir / str(results.mode) / f"seed_{results.seeds[0]}"
    return run_dir


def compute_patient_features(results: ExperimentResults):
    """Describe every patient's held-out glucose series.

    Args:
        results: A loaded reference run. Only ``y_true`` is read from it; no
            prediction and no metric contributes to any feature.

    Returns:
        ``{patient_id: {feature_name: value}}``, skipping any patient whose
        ground-truth series is missing or empty.
    """
    print("Computing patient features...")

    patient_features: Dict[int, Dict[str, Any]] = {}

    for patient_id, per_model in results.patient_metrics.items():
        # The mapping order is not a valid data-selection rule. Select a model
        # with targets and fail before emitting a model-dependent "dataset"
        # feature if any available export disagrees.
        first_model, y_true = _reference_truth(per_model, patient_id)
        if y_true is None or y_true.size == 0:
            print(f"    Warning: No glucose data available for patient {patient_id}")
            continue

        valid = _valid_glucose_mask(y_true)
        glucose = y_true[valid]
        if not glucose.size:
            print(f"    Warning: No valid glucose data available for patient {patient_id}")
            continue

        # Basic statistical features
        mean_glucose = np.mean(glucose)
        std_glucose = np.std(glucose)

        # Dynamics are derived from actual target timestamps whenever a
        # configured prediction export is available.  Row adjacency alone
        # is unsafe: a CGM dropout would otherwise manufacture one large
        # change from readings either side of the gap.
        glucose_diff = np.diff(y_true)
        contiguous_pairs = valid[:-1] & valid[1:]
        try:
            context = load_predictions(
                _context_dir(results), patient_id, first_model, with_context=True
            )
            if context is not None:
                previous = pd.to_datetime(context["preceding_target_timestamp"], utc=True)
                target = pd.to_datetime(context["target_timestamp"], utc=True)
                expected = pd.Timedelta(minutes=results.sampling_rate_minutes or 5)
                current = context["true_values"].to_numpy(dtype=float)
                previous_glucose = context["preceding_target_glucose_mg_dl"].to_numpy(dtype=float)
                contiguous_pairs = (
                    (target - previous == expected).to_numpy(dtype=bool)
                    & _valid_glucose_mask(current)
                    & _valid_glucose_mask(previous_glucose)
                )
                glucose_diff = current - previous_glucose
        except (ResultsLayoutError, OSError, ValueError, KeyError):
            # Context could not be reconstructed for this run. Retain the
            # descriptive level/range features but do not claim that their
            # row-order dynamics are timestamp-aware.
            contiguous_pairs = np.zeros(len(glucose_diff), dtype=bool)

        observed_changes = glucose_diff[contiguous_pairs]
        mean_change = np.mean(observed_changes) if observed_changes.size else np.nan
        std_change = np.std(observed_changes) if observed_changes.size else np.nan

        # Range and quartiles
        glucose_range = np.max(glucose) - np.min(glucose)
        iqr = np.percentile(glucose, 75) - np.percentile(glucose, 25)

        # Time-based features
        n_target_rows = int(y_true.size)
        n_samples = int(glucose.size)

        # Clinical ranges
        hypo_percent = np.mean(glucose < 70) * 100
        hyper_percent = np.mean(glucose > 180) * 100
        in_range_percent = np.mean((glucose >= 70) & (glucose <= 180)) * 100
        severe_hypo_percent = np.mean(glucose < 54) * 100
        severe_hyper_percent = np.mean(glucose > 250) * 100

        # Glucose stability metrics
        threshold = rapid_threshold_mg_dl(results.sampling_rate_minutes or 5)
        n_rapid_changes = int(np.sum(np.abs(observed_changes) > threshold))
        stability_score = (100 - (n_rapid_changes / len(observed_changes) * 100)
                           if len(observed_changes) else np.nan)

        patient_features[patient_id] = {
            'mean_glucose': mean_glucose,
            'std_glucose': std_glucose,
            'min_glucose': np.min(glucose),
            'max_glucose': np.max(glucose),
            'median_glucose': np.median(glucose),
            'q25_glucose': np.percentile(glucose, 25),
            'q75_glucose': np.percentile(glucose, 75),
            'glucose_range': glucose_range,
            'iqr_glucose': iqr,
            'mean_change': mean_change,
            'std_change': std_change,
            'n_contiguous_changes': int(len(observed_changes)),
            'n_samples': n_samples,
            'n_target_rows': n_target_rows,
            'target_coverage_percent': float(100.0 * n_samples / n_target_rows),
            'source_series': 'forecastable_test_targets',
            'hypo_percent': hypo_percent,
            'hyper_percent': hyper_percent,
            'in_range_percent': in_range_percent,
            'severe_hypo_percent': severe_hypo_percent,
            'severe_hyper_percent': severe_hyper_percent,
            'n_rapid_changes': n_rapid_changes,
            'stability_score': stability_score,
            'rapid_change_threshold_mg_dl': threshold,
            'rapid_change_rate_mg_dl_per_min': RAPID_RATE_MG_DL_PER_MIN,
        }

    print(f"[DONE] Computed features for {len(patient_features)} patients")
    return patient_features


def ground_truth_invariance(results: ExperimentResults):
    """Check the ground truth really is the same under every model in the run.

    The features above are called dataset-only because ``y_true`` is the test
    split, not a prediction. That is a claim about the results on disk, so it is
    checked rather than assumed: a mismatch means the run's per-model exports
    disagree about the held-out series, and the feature table is only as
    trustworthy as whichever model happened to be read first.

    Returns:
        One row per patient, with ``identical`` set to None where the patient
        had no ground-truth series to compare.
    """
    rows = []
    for patient_id, models in sorted(results.patient_metrics.items()):
        series = {
            model: np.asarray(metrics.get("y_true"), dtype=float)
            for model, metrics in models.items()
            if metrics.get("y_true") is not None
        }
        reference_model = next(iter(series), None)
        missing_models = sorted(set(models) - set(series))
        if reference_model is None:
            rows.append({"patient_id": patient_id, "n_models": 0,
                         "identical": None, "note": "no ground-truth series"})
            continue
        reference = series[reference_model]
        identical = not missing_models and all(
            other.shape == reference.shape and np.array_equal(other, reference, equal_nan=True)
            for other in series.values()
        )
        rows.append({
            "patient_id": patient_id,
            "n_models": len(models),
            "n_models_with_truth": len(series),
            "n_samples": int(reference.size),
            "identical": bool(identical),
            "note": "" if identical else (
                f"missing y_true for {', '.join(missing_models)}" if missing_models
                else f"differs from {reference_model}"
            ),
        })

    return pd.DataFrame(rows)
