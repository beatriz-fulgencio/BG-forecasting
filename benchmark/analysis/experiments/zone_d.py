"""Patient-cluster uncertainty for the Clarke zone-D clinical decomposition.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...evaluation.clarke_ega import ClarkeEGA
from ..results_io import load_experiment_results, load_predictions

# The Clarke grid is defined on the measurement domain only, so both series are
# clipped to the CGM sensor range before they are classified. The same rule and
# the same constants as :mod:`benchmark.analysis.experiments.error_localization`.
GRID_LOW, GRID_HIGH = 40.0, 400.0
REFERENCE_ROUNDING_TOLERANCE = 0.001

# rate tolerance for noise
STORED_RATE_TOLERANCE_PP = 1e-6


class StoredMetricDriftWarning(UserWarning):
    """Raised when a run's stored Clarke zone-D rate disagrees with this one."""


def patient_zone_d_counts(predictions: pd.DataFrame, patient_id: int):
    """Classify exported predictions with the benchmark's Clarke grid policy."""
    required = {"true_values", "predictions"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"patient {patient_id}: predictions lack {sorted(required - set(predictions.columns))}")
    reference = predictions["true_values"].to_numpy(dtype=float)
    predicted = predictions["predictions"].to_numpy(dtype=float)
    if reference.size == 0 or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(predicted)):
        raise ValueError(f"patient {patient_id}: predictions must be nonempty and finite")
    if np.any((reference < GRID_LOW - REFERENCE_ROUNDING_TOLERANCE) |
              (reference > GRID_HIGH + REFERENCE_ROUNDING_TOLERANCE)):
        raise ValueError(f"patient {patient_id}: references must be within the 40–400 mg/dL CGM range")
    # Older exports may differ from an integer sensor reading by a fraction of a mg/dL after inverse normalization. The current runner recovers targets
    # exactly; this narrow tolerance only keeps those archived sensor-boundary
    # points classifiable.
    reference = np.clip(reference, GRID_LOW, GRID_HIGH)

    # Match the evaluator: raw output is retained for point errors, while
    # clinical-grid classification clips predictions to the sensor domain.
    zones = ClarkeEGA().analyze(reference, np.clip(predicted, GRID_LOW, GRID_HIGH))["zones"]
    zone_d = zones == 4
    missed_hypo = zone_d & (reference <= 70)
    missed_hyper = zone_d & (reference >= 240)
    if np.any(zone_d & ~(missed_hypo | missed_hyper)):
        raise ValueError(f"patient {patient_id}: zone-D point has no defined low/high decomposition")
    return {
        "patient_id": int(patient_id),
        "predictions": int(reference.size),
        "zone_d": int(np.count_nonzero(zone_d)),
        "missed_hypoglycemia": int(np.count_nonzero(missed_hypo)),
        "missed_hyperglycemia": int(np.count_nonzero(missed_hyper)),
    }


def _counts_matrix(counts_by_seed: Sequence[Sequence[Dict[str, int]]]):
    """Validate the per-seed count tables and stack them as (patients, seeds, 3).

    Every seed must cover the same cohort in the same order, and each patient's
    prediction count must agree across seeds -- a disagreement means the runs use
    different windowing or horizons and are not the same cell.
    """
    if isinstance(counts_by_seed, dict) or (counts_by_seed and isinstance(counts_by_seed[0], dict)):
        raise TypeError(
            "patient_bootstrap takes one count table per training seed; "
            "pass [counts] for a single run"
        )
    if not counts_by_seed:
        raise ValueError("Patient bootstrap requires at least one seed's counts")
    cohorts = [[row["patient_id"] for row in counts] for counts in counts_by_seed]
    if len(set(cohorts[0])) != len(cohorts[0]):
        raise ValueError("Patient counts contain a duplicate patient")
    if any(cohort != cohorts[0] for cohort in cohorts[1:]):
        raise ValueError("Every seed must cover the same patient cohort in the same order")
    if len(cohorts[0]) < 2:
        raise ValueError("Patient bootstrap requires at least two patients")

    matrix = np.array([
        [[row["predictions"], row["zone_d"], row["missed_hypoglycemia"]] for row in counts]
        for counts in counts_by_seed
    ], dtype=np.int64).transpose(1, 0, 2)  # (patients, seeds, 3)

    if np.any(matrix[..., 0] <= 0) or np.any(matrix[..., 1] > matrix[..., 0]) \
            or np.any(matrix[..., 2] > matrix[..., 1]):
        raise ValueError("Invalid patient zone-D counts")
    for counts in counts_by_seed:
        for row in counts:
            if "missed_hyperglycemia" in row and \
                    row["missed_hypoglycemia"] + row["missed_hyperglycemia"] != row["zone_d"]:
                raise ValueError(
                    f"patient {row['patient_id']}: missed hypo/hyper counts do not partition zone D"
                )
    # The denominator is fixed before seeding; if it moved, the runs differ in
    # something other than the seed.
    if matrix.shape[1] > 1 and not np.all(matrix[..., 0] == matrix[:, :1, 0]):
        raise ValueError("Prediction counts differ between seeds; the runs are not the same cell")
    return matrix


def patient_bootstrap(counts_by_seed: Sequence[Sequence[Dict[str, int]]], *,
                      replicates: int = 10000, bootstrap_seed: int = 42):
    """Nested bootstrap over patients and training seeds.
    """
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    matrix = _counts_matrix(counts_by_seed)
    n_patients, n_seeds, _ = matrix.shape

    # Point estimates average each patient over seeds before pooling
    seed_mean = matrix.mean(axis=1)
    totals = seed_mean.sum(axis=0)
    per_seed_totals = matrix.sum(axis=0)

    rng = np.random.default_rng(bootstrap_seed)
    patient_draw = rng.integers(0, n_patients, size=(replicates, n_patients))
    # The point estimate is the mean across observed fitted seeds. 
    seed_draw = rng.integers(0, n_seeds, size=(replicates, n_seeds))
    samples = matrix[
        patient_draw[:, :, None], seed_draw[:, None, :], :
    ].mean(axis=2).sum(axis=1)

    rates = samples[:, 1] / samples[:, 0]
    has_zone_d = samples[:, 1] > 0
    shares = samples[has_zone_d, 2] / samples[has_zone_d, 1]

    return {
        "analysis": "clarke_zone_d_decomposition",
        "unit_of_analysis": "prediction point; intervals resample patients and the observed training-seed set",
        "evidence_level": "descriptive",
        "patients": n_patients,
        "patients_with_zone_d": int(np.count_nonzero(seed_mean[:, 1])),
        "training_seeds": n_seeds,
        "predictions": int(totals[0]),
        "zone_d": float(totals[1]),
        "missed_hypoglycemia": float(totals[2]),
        "missed_hyperglycemia": float(totals[1] - totals[2]),
        "zone_d_rate": float(totals[1] / totals[0]),
        "zone_d_rate_ci95": [float(x) for x in np.quantile(rates, [0.025, 0.975])],
        "missed_hypoglycemia_share_of_zone_d": float(totals[2] / totals[1]) if totals[1] else None,
        "missed_hypoglycemia_share_ci95": (
            [float(x) for x in np.quantile(shares, [0.025, 0.975])] if shares.size else None
        ),
        "per_seed_zone_d_rate": [float(z / p) for p, z, _ in per_seed_totals],
        "per_seed_missed_hypoglycemia_share": [
            float(h / z) if z else None for _, z, h in per_seed_totals
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "share_replicates_with_no_zone_d": int(replicates - shares.size),
        "uncertainty_units": ["patient", "training_seed"] if n_seeds > 1 else ["patient"],
        "counts_are_seed_means": n_seeds > 1,
        "seed_resampling": (
            "complete observed seed set resampled with replacement and averaged"
            if n_seeds > 1 else "single observed seed"
        ),
    }


def _stored_zone_d_drift(
    counts: Dict[str, int], metrics: Dict[str, Any], run_dir: Path,
):
    """Compare this run's own stored zone-D rate with the one recomputed here for noise.     
    """
    zones = metrics.get("clarke_zones")
    if not isinstance(zones, dict) or "D" not in zones:
        return None
    try:
        stored_pp = float(zones["D"])
    except (TypeError, ValueError):
        return None
    total = counts["predictions"]
    if not total:
        return None
    recomputed_pp = 100.0 * counts["zone_d"] / total
    if abs(stored_pp - recomputed_pp) <= STORED_RATE_TOLERANCE_PP:
        return None
    stored_n = (metrics.get("prediction_diagnostics") or {}).get("n_points")
    # The implied denominator names the cause far better than the gap does: a
    # stored rate that is not a whole number of points over the exported row
    # count was computed against a different set of points.
    implied = round(counts["zone_d"] / (stored_pp / 100.0), 2) if stored_pp else None
    return {
        "run_dir": str(run_dir),
        "patient_id": counts["patient_id"],
        "stored_zone_d_pct": stored_pp,
        "recomputed_zone_d_pct": recomputed_pp,
        "difference_pp": recomputed_pp - stored_pp,
        "zone_d_points": counts["zone_d"],
        "exported_predictions": total,
        "stored_n_points": int(stored_n) if isinstance(stored_n, int) else None,
        "denominator_implied_by_stored_rate": implied,
    }


def _resolve_model(results, run_dir: Path, model: Optional[str]):
    """The model to read, from the caller or from the run's own metrics."""
    available = sorted({
        name for per_model in results.patient_metrics.values() for name in per_model
    })
    if model is not None:
        if model not in available:
            raise ValueError(
                f"{run_dir}: no model {model!r} in this run (available: {', '.join(available) or 'none'})"
            )
        return model
    if len(available) != 1:
        raise ValueError(
            f"{run_dir}: holds models {', '.join(available) or 'none'}; pass model=... to choose one"
        )
    return available[0]


def _read_run(
    run_dir: Path, model: Optional[str] = None,
):
    """Read one configured mode/seed subrun's cohort and prediction exports.
    """
    run_dir = Path(run_dir)
    # results_io reports a seed directory with no metrics.json as a malformed
    # parent ("expected aggregate_metrics.json"), which sends the reader lookingfor the wrong thing. A seed run that never completed is the common case here, so it is named before the generic layout error can fire.
    if run_dir.name.startswith("seed_") and not (run_dir / "metrics.json").is_file():
        raise ValueError(
            f"{run_dir}: no metrics.json, so this seed run did not complete; "
            "analyze the seeds that finished, or re-run this one"
        )
    results = load_experiment_results(run_dir, with_series=False)
    if results.aggregation != "single_seed" or len(results.seeds) != 1:
        raise ValueError(
            f"{run_dir}: must be a configured single-seed run. This looks like a parent "
            f"experiment directory; pass its <mode>/seed_<n> subdirectories instead."
        )
    if results.horizon_minutes is None or results.horizon_steps is None:
        raise ValueError(f"{run_dir}: prediction horizon is unknown")
    if results.sampling_rate_minutes is None:
        raise ValueError(f"{run_dir}: sampling rate is unknown")
    resolved_model = _resolve_model(results, run_dir, model)

    patient_ids = list(results.patients)
    if not patient_ids:
        raise ValueError("Run has no valid, unique configured patient cohort")
    rows = []
    references: Dict[int, np.ndarray] = {}
    stored_drift: List[Dict[str, Any]] = []
    for patient_id in patient_ids:
        frame = load_predictions(run_dir, patient_id, resolved_model)
        if frame is None:
            raise ValueError(
                f"patient {patient_id}: missing predictions in {run_dir}; set "
                "output.save_predictions: true for the publication run"
            )
        if "predictions" not in frame.columns:
            raise ValueError(
                f"patient {patient_id}: prediction export in {run_dir} holds no predicted "
                "column; set output.save_predictions: true for the publication run"
            )
        counts = patient_zone_d_counts(frame, patient_id)
        rows.append(counts)
        references[int(patient_id)] = frame["true_values"].to_numpy(dtype=float)
        drift = _stored_zone_d_drift(
            counts, results.patient_metrics[patient_id][resolved_model], run_dir
        )
        if drift is not None:
            stored_drift.append(drift)
    metadata = {
        "mode": results.mode,
        "training_seed": int(results.seeds[0]),
        "sampling_rate_minutes": results.sampling_rate_minutes,
        "prediction_horizon_steps": results.horizon_steps,
        "prediction_horizon_minutes": results.horizon_minutes,
        "model": resolved_model,
        "run_dir": str(run_dir),
        "stored_metric_drift": stored_drift,
    }
    return metadata, rows, references


def analyze_zone_d_runs(
    run_dirs: Sequence[Path], *, model: Optional[str] = None, replicates: int = 10000,
    bootstrap_seed: int = 42,
):
    """Pool every training seed of one mode/horizon cell into one nested bootstrap.
    """
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    metadata, counts_by_seed, references_by_seed = [], [], []
    for run_dir in run_dirs:
        run_metadata, rows, references = _read_run(Path(run_dir), model)
        metadata.append(run_metadata)
        counts_by_seed.append(rows)
        references_by_seed.append(references)

    for key in (
        "mode", "model", "prediction_horizon_minutes", "sampling_rate_minutes",
        "prediction_horizon_steps",
    ):
        values = {run[key] for run in metadata}
        if len(values) > 1:
            raise ValueError(
                f"All runs must share the same {key}; got {sorted(map(str, values))}"
            )
    seeds = [run["training_seed"] for run in metadata]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Training seeds must be distinct; got {seeds}")

    # Require every patient's complete, ordered reference series to match before attributing between-run variation to training alone.
    expected_references = references_by_seed[0]
    for run_metadata, references in zip(metadata[1:], references_by_seed[1:]):
        if set(references) != set(expected_references):
            raise ValueError(
                "Patient cohort differs between seeds; the runs are not the same cell"
            )
        for patient_id, expected in expected_references.items():
            if not np.array_equal(expected, references[patient_id]):
                raise ValueError(
                    "Reference glucose series differs between seeds for patient "
                    f"{patient_id} ({metadata[0]['run_dir']} vs {run_metadata['run_dir']})"
                )

    # Cross-check against each run's own stored Clarke metrics. 
    stored_drift = [entry for run in metadata for entry in run["stored_metric_drift"]]
    if stored_drift:
        worst = max(stored_drift, key=lambda entry: abs(entry["difference_pp"]))
        warnings.warn(
            f"zone-D rate disagrees with the stored clarke_zones in "
            f"{len(stored_drift)} of {sum(len(rows) for rows in counts_by_seed)} "
            f"patient-runs; worst is patient {worst['patient_id']} in "
            f"{worst['run_dir']} ({worst['recomputed_zone_d_pct']:.6f}% recomputed "
            f"vs {worst['stored_zone_d_pct']:.6f}% stored, "
            f"{worst['zone_d_points']} points over {worst['exported_predictions']} "
            f"exported rows against an implied denominator of "
            f"{worst['denominator_implied_by_stored_rate']}). The recomputed "
            f"values are used; see stored_metric_crosscheck in the summary.",
            StoredMetricDriftWarning,
            stacklevel=2,
        )

    summary = patient_bootstrap(counts_by_seed, replicates=replicates,
                                bootstrap_seed=bootstrap_seed)
    summary.update({
        "stored_metric_crosscheck": {
            "checked_patient_runs": sum(len(rows) for rows in counts_by_seed),
            "tolerance_pp": STORED_RATE_TOLERANCE_PP,
            "disagreeing_patient_runs": len(stored_drift),
            "agrees_with_stored_metrics": not stored_drift,
            "detail": stored_drift,
            "policy": (
                "the zone-D rate recomputed from the prediction exports is compared "
                "with the clarke_zones the run's own evaluator stored; the "
                "recomputed value is authoritative because the missed "
                "hypo/hyper decomposition cannot be recovered from the stored "
                "aggregate, and a disagreement is reported rather than fatal"
            ),
        },
        "mode": metadata[0]["mode"],
        "training_seed_values": seeds,
        "sampling_rate_minutes": metadata[0]["sampling_rate_minutes"],
        "prediction_horizon_steps": metadata[0]["prediction_horizon_steps"],
        "prediction_horizon_minutes": metadata[0]["prediction_horizon_minutes"],
        "model": metadata[0]["model"],
        "run_dirs": [run["run_dir"] for run in metadata],
    })

    per_patient = pd.DataFrame([
        {"training_seed": seed, **row}
        for seed, rows in zip(seeds, counts_by_seed) for row in rows
    ])
    per_patient["zone_d_rate"] = per_patient["zone_d"] / per_patient["predictions"]
    per_patient["missed_hypoglycemia_share_of_zone_d"] = (
        per_patient["missed_hypoglycemia"] / per_patient["zone_d"].replace(0, np.nan)
    )
    return summary, per_patient


def analyze_zone_d_run(
    run_dir: Path, *, model: Optional[str] = None, replicates: int = 10000,
    bootstrap_seed: int = 42,
):
    """Single-seed convenience wrapper around :func:`analyze_zone_d_runs`."""
    return analyze_zone_d_runs(
        [run_dir], model=model, replicates=replicates, bootstrap_seed=bootstrap_seed
    )
