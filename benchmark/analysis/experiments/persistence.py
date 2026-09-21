"""Persistence-baseline analysis for glucose forecasts.

The persistence prediction at any horizon is the glucose observed at the
forecast origin. Prediction context is loaded through :mod:`results_io`, which
strictly reconstructs archived context and verifies target values and row order.

"""

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from ..results_io import load_experiment_results, load_predictions



def persistence_metrics(predictions: pd.DataFrame, *, source: str = "predictions"):
    """Compute persistence MAE/RMSE from forecast-origin glucose."""
    
    required = {"true_values", "forecast_origin_glucose_mg_dl"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{source}: missing persistence columns {sorted(missing)}")
    reference = predictions["true_values"].to_numpy(dtype=float)
    origin = predictions["forecast_origin_glucose_mg_dl"].to_numpy(dtype=float)
    if reference.size == 0:
        raise ValueError(f"{source}: persistence series is empty")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(origin)):
        raise ValueError(f"{source}: persistence inputs must be finite")
    return {
        "n_predictions": int(reference.size),
        "persistence_mae_mg_dl": float(mean_absolute_error(reference, origin)),
        "persistence_rmse_mg_dl": float(root_mean_squared_error(reference, origin)),
    }


def analyze_persistence_runs(
    run_dirs: Sequence[Path], *, tolerance: float = 1e-12
):
    """Analyze configured single-seed runs and verify baseline invariance."""
    if not run_dirs:
        raise ValueError("At least one configured mode/seed run is required")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    rows = []
    # Keep the actual persistence inputs until each duplicate patient/horizon
    # has been checked. Equal MAE/RMSE alone does not prove equal forecasts.
    baseline_inputs: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]] = {}
    for run_dir_value in run_dirs:
        run_dir = Path(run_dir_value)
        results = load_experiment_results(run_dir, with_series=False)
        if results.aggregation != "single_seed" or len(results.seeds) != 1:
            raise ValueError(
                f"{run_dir}: persistence inputs must be configured single-seed runs; "
                f"got {results.aggregation}"
            )
        if results.horizon_minutes is None or results.horizon_steps is None:
            raise ValueError(f"{run_dir}: prediction horizon is unknown")
        if results.sampling_rate_minutes is None:
            raise ValueError(f"{run_dir}: sampling rate is unknown")
        for patient_id in results.patients:
            for model in results.patient_metrics[patient_id]:
                predictions = load_predictions(
                    run_dir, patient_id, model, with_context=True
                )
                if predictions is None:
                    raise ValueError(
                        f"{run_dir}: patient {patient_id} model {model} has no prediction CSV"
                    )
                metrics = persistence_metrics(
                    predictions,
                    source=f"{run_dir}: patient {patient_id} model {model}",
                )
                baseline_key = (
                    patient_id, results.sampling_rate_minutes,
                    results.horizon_steps, results.horizon_minutes,
                )
                inputs = (
                    predictions["true_values"].to_numpy(dtype=float),
                    predictions["forecast_origin_glucose_mg_dl"].to_numpy(dtype=float),
                )
                expected = baseline_inputs.setdefault(baseline_key, inputs)
                if (
                    expected[0].shape != inputs[0].shape
                    or expected[1].shape != inputs[1].shape
                    or not np.allclose(expected[0], inputs[0], rtol=0.0, atol=tolerance)
                    or not np.allclose(expected[1], inputs[1], rtol=0.0, atol=tolerance)
                ):
                    raise ValueError(
                        f"Persistence inputs are not invariant for patient/horizon "
                        f"{baseline_key}: target or forecast-origin glucose differs"
                    )
                rows.append({
                    "experiment": results.name,
                    "run_dir": str(run_dir),
                    "model": model,
                    "mode": results.mode,
                    "training_seed": results.seeds[0],
                    "patient_id": patient_id,
                    "sampling_rate_minutes": results.sampling_rate_minutes,
                    "horizon_steps": results.horizon_steps,
                    "horizon_minutes": results.horizon_minutes,
                    "aggregation": results.aggregation,
                    **metrics,
                })

    per_run = pd.DataFrame(rows).sort_values(
        ["horizon_minutes", "patient_id", "model", "mode", "training_seed"]
    ).reset_index(drop=True)
    group_columns = [
        "patient_id", "sampling_rate_minutes", "horizon_steps", "horizon_minutes"
    ]
    metric_columns = [
        "n_predictions", "persistence_mae_mg_dl", "persistence_rmse_mg_dl"
    ]
    canonical_rows = []
    max_deviation = {name: 0.0 for name in metric_columns}
    compared_groups = 0
    for key, group in per_run.groupby(group_columns, sort=True, dropna=False):
        first = group.iloc[0]
        if len(group) > 1:
            compared_groups += 1
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            deviation = float(np.max(np.abs(values - values[0])))
            max_deviation[metric] = max(max_deviation[metric], deviation)
            if deviation > tolerance:
                labels = group[["model", "mode", "training_seed", metric]].to_dict("records")
                raise ValueError(
                    f"Persistence is not invariant for patient/horizon {key}: {labels}"
                )
        canonical_rows.append({
            **dict(zip(group_columns, key)),
            **{metric: first[metric] for metric in metric_columns},
            "source_runs": int(len(group)),
            "models_checked": ",".join(sorted(group["model"].astype(str).unique())),
            "modes_checked": ",".join(sorted(group["mode"].astype(str).unique())),
            "seeds_checked": ",".join(map(str, sorted(group["training_seed"].unique()))),
        })
    per_patient = pd.DataFrame(canonical_rows)
    summary = {
        "analysis": "persistence_baseline",
        "unit_of_analysis": "patient within prediction horizon",
        "evidence_level": "descriptive",
        "forecast_definition": "forecast-origin glucose carried forward to the final target",
        "run_count": len(run_dirs),
        "source_rows": int(len(per_run)),
        "patient_horizon_rows": int(len(per_patient)),
        "horizons_minutes": sorted(per_patient["horizon_minutes"].unique().tolist()),
        "patients": sorted(per_patient["patient_id"].unique().tolist()),
        "models": sorted(per_run["model"].unique().tolist()),
        "modes": sorted(per_run["mode"].unique().tolist()),
        "training_seeds": sorted(int(value) for value in per_run["training_seed"].unique()),
        "invariance_groups_compared": compared_groups,
        "invariance_tolerance": tolerance,
        "maximum_duplicate_deviation": max_deviation,
        "invariance_confirmed": True
    }
    return per_run, per_patient, summary


def plot_persistence(per_patient: pd.DataFrame, output_path: Path) -> None:
    """Plot patient-level persistence errors by horizon."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for patient_id, group in per_patient.groupby("patient_id", sort=True):
        ordered = group.sort_values("horizon_minutes")
        axes[0].plot(
            ordered["horizon_minutes"], ordered["persistence_mae_mg_dl"],
            marker="o", alpha=0.55, linewidth=1, label=str(patient_id),
        )
        axes[1].plot(
            ordered["horizon_minutes"], ordered["persistence_rmse_mg_dl"],
            marker="o", alpha=0.55, linewidth=1,
        )
    axes[0].set_ylabel("Persistence MAE (mg/dL)")
    axes[1].set_ylabel("Persistence RMSE (mg/dL)")
    for axis in axes:
        axis.set_xlabel("Prediction horizon (minutes)")
        axis.grid(alpha=0.25)
    if per_patient["patient_id"].nunique() <= 12:
        axes[0].legend(title="Patient", fontsize=7, ncol=2)
    figure.suptitle("Persistence baseline by patient and horizon")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
