"""Forecast error stratified by the glycemic range it was made in.

Every prediction is assigned to a band by its **reference** glucose -- never by
the prediction, which would let the model choose its own denominator:

    hypoglycemia   reference <  70 mg/dL
    in range       70 <= reference <= 180 mg/dL
    hyperglycemia  reference > 180 mg/dL

Band membership depends only on the reference series, which the seed never
touches, so a patient's per-band point counts are identical across seeds. That
invariant, and the complete ordered reference series, are enforced: if either
fails, the runs differ in something other than the seed.

Pooled statistics weight patients by their point count in the band, so a patient with four hypoglycemic points does not carry the same weight as one with four hundred. A patient with none contributes nothing rather than a NaN.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..results_io import load_experiment_results, load_predictions

# Band edges in mg/dL for hypoglycemia and hyperglycemia, with the in-range band between them. 
HYPO_BELOW = 70.0
HYPER_ABOVE = 180.0

BANDS = ("hypoglycemia", "in_range", "hyperglycemia")


def band_masks(reference: np.ndarray):
    """Partition predictions by the reference glucose they were made against."""
    return {
        "hypoglycemia": reference < HYPO_BELOW,
        "in_range": (reference >= HYPO_BELOW) & (reference <= HYPER_ABOVE),
        "hyperglycemia": reference > HYPER_ABOVE,
    }


def patient_band_sums(predictions: pd.DataFrame, patient_id: int):
    """Per-band point count, absolute-error sum, and signed-error sum.
    """
    required = {"true_values", "predictions"}
    if not required.issubset(predictions.columns):
        raise ValueError(
            f"patient {patient_id}: predictions lack {sorted(required - set(predictions.columns))}"
        )
    reference = predictions["true_values"].to_numpy(dtype=float)
    predicted = predictions["predictions"].to_numpy(dtype=float)
    if reference.size == 0 or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(predicted)):
        raise ValueError(f"patient {patient_id}: predictions must be nonempty and finite")

    error = predicted - reference
    row: Dict[str, Any] = {"patient_id": int(patient_id), "predictions": int(reference.size)}
    masks = band_masks(reference)
    for band in BANDS:
        mask = masks[band]
        row[f"{band}_n"] = int(np.count_nonzero(mask))
        row[f"{band}_abs_error_sum"] = float(np.abs(error[mask]).sum())
        row[f"{band}_signed_error_sum"] = float(error[mask].sum())
    if sum(row[f"{band}_n"] for band in BANDS) != row["predictions"]:
        raise ValueError(f"patient {patient_id}: bands do not partition the predictions")
    return row


def _sums_matrix(sums_by_seed: Sequence[Sequence[Dict[str, Any]]]):
    """Validate the per-seed tables and stack them as (patients, seeds, bands, 3).

    The trailing axis is (n, absolute-error sum, signed-error sum).
    """
    if isinstance(sums_by_seed, dict) or (sums_by_seed and isinstance(sums_by_seed[0], dict)):
        raise TypeError(
            "patient_bootstrap takes one table per training seed; pass [rows] for a single run"
        )
    if not sums_by_seed:
        raise ValueError("Patient bootstrap requires at least one seed's sums")
    cohorts = [[row["patient_id"] for row in rows] for rows in sums_by_seed]
    if len(set(cohorts[0])) != len(cohorts[0]):
        raise ValueError("Patient sums contain a duplicate patient")
    if any(cohort != cohorts[0] for cohort in cohorts[1:]):
        raise ValueError("Every seed must cover the same patient cohort in the same order")
    if len(cohorts[0]) < 2:
        raise ValueError("Patient bootstrap requires at least two patients")

    matrix = np.array([
        [[[row[f"{band}_n"], row[f"{band}_abs_error_sum"], row[f"{band}_signed_error_sum"]]
          for band in BANDS] for row in rows]
        for rows in sums_by_seed
    ], dtype=float).transpose(1, 0, 2, 3)  # (patients, seeds, bands, 3)

    if np.any(matrix[..., 0] < 0) or np.any(matrix[..., 1] < 0):
        raise ValueError("Invalid per-band counts or absolute-error sums")
    # Band membership is a function of the reference series alone, which the seed
    # never touches. If the counts moved, the runs are not the same cell.
    if matrix.shape[1] > 1 and not np.all(matrix[..., 0] == matrix[:, :1, :, 0]):
        raise ValueError(
            "Per-band prediction counts differ between seeds; the runs are not the same cell"
        )
    return matrix, cohorts[0]


def patient_bootstrap(sums_by_seed: Sequence[Sequence[Dict[str, Any]]], *,
                      replicates: int = 10000, bootstrap_seed: int = 42):
    """Nested bootstrap over patients and training seeds, per glycemic band.

    Returns pooled MAE and signed bias for each band with percentile intervals,
    plus the per-seed pooled values so the seed spread stays visible.
    """
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    matrix, patient_ids = _sums_matrix(sums_by_seed)
    n_patients, n_seeds, _, _ = matrix.shape

    seed_mean = matrix.mean(axis=1)          # (patients, bands, 3)
    totals = seed_mean.sum(axis=0)           # (bands, 3)
    per_seed_totals = matrix.sum(axis=0)     # (seeds, bands, 3)

    rng = np.random.default_rng(bootstrap_seed)
    patient_draw = rng.integers(0, n_patients, size=(replicates, n_patients))
    
    seed_draw = rng.integers(0, n_seeds, size=(replicates, n_seeds))
    samples = matrix[
        patient_draw[:, :, None], seed_draw[:, None, :], :, :
    ].mean(axis=2).sum(axis=1)  # (replicates, bands, 3)

    bands: Dict[str, Any] = {}
    for index, band in enumerate(BANDS):
        n_total = totals[index, 0]
        counts = samples[:, index, 0]
        # A replicate that drew no point in this band has no error to report; it is excluded from the interval rather than contributing a zero, and the surviving count is published next to the requested one.
        usable = counts > 0
        mae_draws = samples[usable, index, 1] / counts[usable]
        bias_draws = samples[usable, index, 2] / counts[usable]
        bands[band] = {
            "predictions": float(n_total),
            "share_of_all_predictions": float(n_total / totals[:, 0].sum()) if totals[:, 0].sum() else None,
            "patients_with_points": int(np.count_nonzero(seed_mean[:, index, 0])),
            "mae_mg_dl": float(totals[index, 1] / n_total) if n_total else None,
            "mae_ci95": ([float(x) for x in np.quantile(mae_draws, [0.025, 0.975])]
                         if mae_draws.size else None),
            "signed_bias_mg_dl": float(totals[index, 2] / n_total) if n_total else None,
            "signed_bias_ci95": ([float(x) for x in np.quantile(bias_draws, [0.025, 0.975])]
                                 if bias_draws.size else None),
            "per_seed_mae_mg_dl": [
                float(per_seed_totals[s, index, 1] / per_seed_totals[s, index, 0])
                if per_seed_totals[s, index, 0] else None for s in range(n_seeds)
            ],
            "per_seed_signed_bias_mg_dl": [
                float(per_seed_totals[s, index, 2] / per_seed_totals[s, index, 0])
                if per_seed_totals[s, index, 0] else None for s in range(n_seeds)
            ],
            "valid_replicates": int(mae_draws.size),
        }

    return {
        "analysis": "error_by_glycemic_range",
        "unit_of_analysis": "prediction point; intervals resample patients and the observed training-seed set",
        "evidence_level": "descriptive",
        "patients": n_patients,
        "patient_ids": patient_ids,
        "training_seeds": n_seeds,
        "predictions": float(totals[:, 0].sum()),
        "band_edges_mg_dl": {"hypoglycemia_below": HYPO_BELOW, "hyperglycemia_above": HYPER_ABOVE},
        "bands": bands,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "uncertainty_units": ["patient", "training_seed"] if n_seeds > 1 else ["patient"],
        "sums_are_seed_means": n_seeds > 1,
        "seed_resampling": (
            "complete observed seed set resampled with replacement and averaged"
            if n_seeds > 1 else "single observed seed"
        ),
        "signed_bias_convention": "predicted minus reference; negative means the model reads low"
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


def _read_run( run_dir: Path, model: Optional[str] = None,):
    """Read one configured mode/seed subrun's cohort and prediction exports."""
    run_dir = Path(run_dir)
    # results_io reports a seed directory with no metrics.json as a malformed parent ("expected aggregate_metrics.json"), which sends the reader looking for the wrong thing. A seed run that never completed is the common case here, so it is named before the generic layout error can fire.
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
        rows.append(patient_band_sums(frame, patient_id))
        references[int(patient_id)] = frame["true_values"].to_numpy(dtype=float)
    metadata = {
        "mode": results.mode,
        "training_seed": int(results.seeds[0]),
        "sampling_rate_minutes": results.sampling_rate_minutes,
        "prediction_horizon_steps": results.horizon_steps,
        "prediction_horizon_minutes": results.horizon_minutes,
        "model": resolved_model,
        "run_dir": str(run_dir),
    }
    return metadata, rows, references


def analyze_error_by_range_runs(
    run_dirs: Sequence[Path], *, model: Optional[str] = None, replicates: int = 10000,
    bootstrap_seed: int = 42,
):
    """Pool every training seed of one mode/horizon cell into one nested bootstrap."""
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    metadata, sums_by_seed, references_by_seed = [], [], []
    for run_dir in run_dirs:
        run_metadata, rows, references = _read_run(Path(run_dir), model)
        metadata.append(run_metadata)
        sums_by_seed.append(rows)
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
    
    expected_references = references_by_seed[0]
    for run_metadata, references in zip(metadata[1:], references_by_seed[1:]):
        if set(references) != set(expected_references):
            raise ValueError(
                "Patient cohort differs between seeds; the runs are not the same cell"
            )
        for patient_id, expected in expected_references.items():
            observed = references[patient_id]
            if not np.array_equal(expected, observed):
                raise ValueError(
                    "Reference glucose series differs between seeds for patient "
                    f"{patient_id} ({metadata[0]['run_dir']} vs {run_metadata['run_dir']})"
                )

    summary = patient_bootstrap(sums_by_seed, replicates=replicates,
                                bootstrap_seed=bootstrap_seed)
    summary.update({
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
        for seed, rows in zip(seeds, sums_by_seed) for row in rows
    ])
    for band in BANDS:
        n = per_patient[f"{band}_n"].replace(0, np.nan)
        per_patient[f"{band}_mae_mg_dl"] = per_patient[f"{band}_abs_error_sum"] / n
        per_patient[f"{band}_signed_bias_mg_dl"] = per_patient[f"{band}_signed_error_sum"] / n
    return summary, per_patient


def analyze_error_by_range_run(
    run_dir: Path, *, model: Optional[str] = None, replicates: int = 10000,
    bootstrap_seed: int = 42,
):
    """Single-seed convenience wrapper"""
    return analyze_error_by_range_runs(
        [run_dir], model=model, replicates=replicates, bootstrap_seed=bootstrap_seed
    )
