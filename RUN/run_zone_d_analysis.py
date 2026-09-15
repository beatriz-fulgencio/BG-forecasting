"""Patient-cluster uncertainty for the Clarke zone-D clinical decomposition.

Analyze one mode/horizon cell at a time, passing every training seed that ran in
it. Two things vary and both belong in the interval:

* which patients happen to be enrolled -- prediction rows are correlated within
  patients (and within episodes), so resampling draws whole patients, never rows;
* which training run you got -- initialization and batch order move the zone-D
  counts even with the cohort held fixed.

The bootstrap is therefore nested: each replicate resamples patients with
replacement and draws one training seed for the whole replicate. Drawing a single
seed per replicate (rather than an independent seed per patient) keeps the
common-mode component, since within one seed every patient's model shares the
same initialization stream. With one seed supplied this reduces to the ordinary
patient-cluster bootstrap, and the summary says so.

A patient's prediction count is fixed before any seeding -- the seed splits
train/validation, never the test window -- so the rate denominator is identical
across seeds and only the zone-D counts move. That invariant is enforced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from benchmark.evaluation.clarke_ega import ClarkeEGA


GRID_LOW, GRID_HIGH = 40.0, 400.0
REFERENCE_ROUNDING_TOLERANCE = 0.001


def patient_zone_d_counts(predictions: pd.DataFrame, patient_id: int) -> Dict[str, int]:
    """Classify exported predictions with the benchmark's Clarke grid policy."""
    required = {"true_glucose_mg_dl", "predicted_glucose_mg_dl"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"patient {patient_id}: prediction CSV lacks {sorted(required - set(predictions.columns))}")
    reference = predictions["true_glucose_mg_dl"].to_numpy(dtype=float)
    predicted = predictions["predicted_glucose_mg_dl"].to_numpy(dtype=float)
    if reference.size == 0 or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(predicted)):
        raise ValueError(f"patient {patient_id}: predictions must be nonempty and finite")
    if np.any((reference < GRID_LOW - REFERENCE_ROUNDING_TOLERANCE) |
              (reference > GRID_HIGH + REFERENCE_ROUNDING_TOLERANCE)):
        raise ValueError(f"patient {patient_id}: references must be within the 40–400 mg/dL CGM range")
    # Older exports may differ from an integer sensor reading by a fraction of a
    # mg/dL after inverse normalization. The current runner recovers targets
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


def _counts_matrix(counts_by_seed: Sequence[Sequence[Dict[str, int]]]) -> np.ndarray:
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
                      replicates: int = 10000, bootstrap_seed: int = 42) -> Dict[str, Any]:
    """Nested bootstrap over patients and training seeds.

    Args:
        counts_by_seed: One :func:`patient_zone_d_counts` table per training seed,
            each covering the same cohort in the same order. ``[counts]`` for a
            single run.
        replicates: Bootstrap replicates.
        bootstrap_seed: RNG seed for the resampling itself (not a training seed).

    Returns:
        Pooled point estimates over seed-mean counts, percentile CIs, and the
        per-seed pooled estimates so the seed spread stays visible.
    """
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    matrix = _counts_matrix(counts_by_seed)
    n_patients, n_seeds, _ = matrix.shape

    # Point estimates average each patient over seeds before pooling. Because the
    # denominator is seed-invariant, this equals the mean of the per-seed pooled
    # rates -- the two natural estimators coincide for the rate.
    seed_mean = matrix.mean(axis=1)
    totals = seed_mean.sum(axis=0)
    per_seed_totals = matrix.sum(axis=0)

    rng = np.random.default_rng(bootstrap_seed)
    patient_draw = rng.integers(0, n_patients, size=(replicates, n_patients))
    seed_draw = rng.integers(0, n_seeds, size=replicates)
    samples = matrix[patient_draw, seed_draw[:, None], :].sum(axis=1)

    rates = samples[:, 1] / samples[:, 0]
    has_zone_d = samples[:, 1] > 0
    shares = samples[has_zone_d, 2] / samples[has_zone_d, 1]

    note = (
        "Percentile CIs resample patients with replacement and draw one training "
        "seed per replicate, so they reflect combined between-patient and "
        "training-run variability within this mode/horizon cell."
        if n_seeds > 1 else
        "Percentile CIs resample patients within this one mode/seed/horizon run; "
        "they do not include training-seed uncertainty."
    )
    return {
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
        "note": note,
    }


def _read_run(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, int]]]:
    """Read one configured mode/seed subrun's cohort and prediction exports."""
    tracking_path = run_dir / "tracking.json"
    if not tracking_path.is_file():
        raise ValueError(f"Not a configured mode/seed run (missing {tracking_path})")
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    params = tracking["data_params"]
    missing_keys = [key for key in ("mode", "seed", "prediction_horizon_minutes")
                    if key not in params]
    if missing_keys:
        # A parent experiment logs `modes`/`seeds`; its subruns log the singular
        # forms. Pointing at the parent is the common mistake, so name it.
        hint = (" This looks like a parent experiment directory; pass its "
                "<mode>/seed_<n> subdirectories instead."
                if {"modes", "seeds"} & set(params) else "")
        raise ValueError(f"{run_dir}: tracking.json lacks {missing_keys}.{hint}")
    patient_ids = [int(pid) for pid in params["patient_ids"]]
    if not patient_ids or len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Run has no valid, unique configured patient cohort")
    model_name = tracking["config"]["model"]["type"].upper()
    rows = []
    for patient_id in patient_ids:
        csv_path = run_dir / f"patient_{patient_id}" / f"{model_name}_predictions.csv"
        if not csv_path.is_file():
            raise ValueError(
                f"patient {patient_id}: missing {csv_path}; set output.save_predictions: true for the publication run"
            )
        rows.append(patient_zone_d_counts(pd.read_csv(csv_path), patient_id))
    metadata = {
        "mode": params["mode"],
        "training_seed": params["seed"],
        "prediction_horizon_minutes": params["prediction_horizon_minutes"],
        "model": model_name,
        "run_dir": str(run_dir),
    }
    return metadata, rows


def analyze_runs(run_dirs: Sequence[Path], *, replicates: int = 10000,
                 bootstrap_seed: int = 42) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Pool every training seed of one mode/horizon cell into one nested bootstrap.

    Mixing modes, horizons or models would average over the very thing being
    compared, so those must match across the supplied runs; the training seeds
    must differ.
    """
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    metadata, counts_by_seed = [], []
    for run_dir in run_dirs:
        run_metadata, rows = _read_run(Path(run_dir))
        metadata.append(run_metadata)
        counts_by_seed.append(rows)

    for key in ("mode", "prediction_horizon_minutes", "model"):
        values = {run[key] for run in metadata}
        if len(values) > 1:
            raise ValueError(
                f"All runs must share the same {key}; got {sorted(map(str, values))}"
            )
    seeds = [run["training_seed"] for run in metadata]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Training seeds must be distinct; got {seeds}")

    summary = patient_bootstrap(counts_by_seed, replicates=replicates,
                                bootstrap_seed=bootstrap_seed)
    summary.update({
        "mode": metadata[0]["mode"],
        "training_seed_values": seeds,
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


def analyze_run(run_dir: Path, *, replicates: int = 10000,
                bootstrap_seed: int = 42) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Single-seed convenience wrapper around :func:`analyze_runs`."""
    return analyze_runs([run_dir], replicates=replicates, bootstrap_seed=bootstrap_seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, action="append", type=Path, dest="run_dirs",
                        help="Configured mode/seed directory, e.g. .../regular/seed_42. "
                             "Repeat once per training seed of the same mode/horizon cell.")
    parser.add_argument("--output-dir", type=Path,
                        help="Analysis output directory (default: zone_d_analysis beside the run(s))")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    try:
        summary, per_patient = analyze_runs(args.run_dirs, replicates=args.bootstrap_replicates,
                                            bootstrap_seed=args.bootstrap_seed)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif len(args.run_dirs) == 1:
        output_dir = args.run_dirs[0] / "zone_d_analysis"
    else:
        output_dir = args.run_dirs[0].parent / "zone_d_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_patient.to_csv(output_dir / "per_patient.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
