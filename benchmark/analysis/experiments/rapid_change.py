"""Timestamp-aware, one-interval rapid-change error localization.

The 3 mg/dL/min threshold follows the Dexcom ``>3 mg/dL/min`` rapidly-rising/
falling category.  This analysis is deliberately a *one-interval proxy*, not a
reimplementation of a device trend arrow: it classifies the reference change
over exactly the run's sampling interval (15 mg/dL at 5 minutes), while Dexcom
also describes its rapid category as more than 45 mg/dL over 15 minutes.

The interval check is intentionally strict.  Rows with a gap, duplicate, or
malformed timestamp are excluded instead of deriving a rate from an arbitrary
prediction-row adjacency.

References: Klonoff & Kerr (2017), doi:10.1177/1932296817723260; Dexcom G5
Mobile User Guide, trend-arrow table; and OpenAPS oref0's timestamped-delta
concept (MIT, https://github.com/openaps/oref0).  The custom bootstrap follows
the percentile construction documented by SciPy but resamples patients and the
complete observed seed set together because a one-seed draw estimates a
different estimand.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from ..results_io import load_experiment_results, load_predictions


#: Clinical threshold used by this analysis's one-interval rapid-change proxy.
RAPID_RATE_MG_DL_PER_MIN = 3.0


def rapid_threshold_mg_dl(sampling_rate_minutes: int):
    """The step for the one-interval rapid-change proxy.

    15 mg/dL at the 5-minute sampling every config in this repository uses.
    Deriving it keeps the clinical meaning fixed if a dataset is ever sampled
    at another interval, where a hard-coded step would quietly become a
    different rate.
    """
    if sampling_rate_minutes <= 0:
        raise ValueError("sampling_rate_minutes must be positive")
    return RAPID_RATE_MG_DL_PER_MIN * sampling_rate_minutes

CONDITIONS = ("calm", "rapid")


def rapid_change_masks(
    predictions: pd.DataFrame,
    *,
    sampling_rate_minutes: int,
    threshold_mg_dl: Optional[float] = None,
    source: str = "predictions",
):
    """Return separated calm/rapid-proxy masks and the excluded-row mask.

    ``preceding_target_*`` comes from the prediction-context contract and is
    never inferred from prediction-row order.  

    ``threshold_mg_dl`` defaults to the step that the clinical threshold
    implies at this run's sampling interval -- 15 mg/dL at 5-minute sampling.
    Pass a number to override it; the module docstring explains why the
    default is expressed as a rate.
    """
    if threshold_mg_dl is None:
        threshold_mg_dl = rapid_threshold_mg_dl(sampling_rate_minutes)
    threshold_mg_dl = float(threshold_mg_dl)
    required = {
        "true_values", "predictions", "preceding_target_timestamp",
        "preceding_target_glucose_mg_dl", "target_timestamp",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{source}: missing rapid-change columns {missing}")
    if sampling_rate_minutes <= 0:
        raise ValueError("sampling_rate_minutes must be positive")
    if not np.isfinite(threshold_mg_dl) or threshold_mg_dl < 0:
        raise ValueError("threshold_mg_dl must be finite and non-negative")

    reference = pd.to_numeric(predictions["true_values"], errors="coerce").to_numpy(float)
    previous = pd.to_numeric(
        predictions["preceding_target_glucose_mg_dl"], errors="coerce"
    ).to_numpy(float)
   
   
    previous_time = pd.to_datetime(
        predictions["preceding_target_timestamp"], errors="coerce", utc=True
    )
    target_time = pd.to_datetime(predictions["target_timestamp"], errors="coerce", utc=True)
    expected = pd.Timedelta(minutes=sampling_rate_minutes)
    contiguous = (target_time - previous_time == expected).to_numpy(dtype=bool)
    finite = np.isfinite(reference) & np.isfinite(previous)
    eligible = contiguous & finite
    rapid = eligible & (np.abs(reference - previous) > threshold_mg_dl)
    calm = eligible & ~rapid
    return {"calm": calm, "rapid": rapid}, ~eligible


def rapid_change_metrics(
    predictions: pd.DataFrame,
    *,
    sampling_rate_minutes: int,
    threshold_mg_dl: Optional[float] = None,
    source: str = "predictions",
):
    """Compute calm/rapid errors and explicitly count excluded timestamp gaps."""
    
    masks, excluded = rapid_change_masks(
        predictions, sampling_rate_minutes=sampling_rate_minutes,
        threshold_mg_dl=threshold_mg_dl, source=source,
    )
    reference = predictions["true_values"].to_numpy(dtype=float)
    forecast = predictions["predictions"].to_numpy(dtype=float)
    if not np.all(np.isfinite(forecast)):
        raise ValueError(f"{source}: predictions must be finite")
    result: Dict[str, Dict[str, float | int]] = {}
    for condition, mask in masks.items():
        errors = forecast[mask] - reference[mask]
        count = int(mask.sum())
        result[condition] = {
            "n_predictions": count,
            "absolute_error_sum": float(np.abs(errors).sum()),
            "squared_error_sum": float(np.square(errors).sum()),
            "signed_error_sum": float(errors.sum()),
            "mae_mg_dl": float(mean_absolute_error(reference[mask], forecast[mask])) if count else np.nan,
            "rmse_mg_dl": float(root_mean_squared_error(reference[mask], forecast[mask])) if count else np.nan,
            "signed_bias_mg_dl": float(errors.mean()) if count else np.nan,
            "excluded_noncontiguous_or_invalid": int(excluded.sum()),
        }
    return result


def _bootstrap_summary(
    rows: pd.DataFrame, *, replicates: int, random_seed: int
):
    """Nested patient and seed-set percentile intervals for one result cell.

    Each replicate resamples the whole observed seed set with replacement and
    applies one shared patient resample to each drawn seed.  This estimates the
    same cross-seed mean as the point estimate, rather than the performance of
    a single randomly selected fitted model.
    """
    
    seeds = tuple(sorted(int(seed) for seed in rows.training_seed.unique()))
    patient_sets = [set(group.patient_id) for _, group in rows.groupby("training_seed")]
    if not patient_sets or any(cohort != patient_sets[0] for cohort in patient_sets[1:]):
        raise ValueError("rapid-change bootstrap requires the same patient cohort for every seed")
    patients = np.array(sorted(patient_sets[0]), dtype=int)
    if not len(patients):
        raise ValueError("rapid-change bootstrap has no patients")
    indexed = rows.set_index(["training_seed", "patient_id", "condition"]).sort_index()
    rng = np.random.default_rng(random_seed)
    output: Dict[str, Any] = {}
    for condition in CONDITIONS:
        samples = []
        for _ in range(replicates):
            draw = rng.choice(patients, size=len(patients), replace=True)
            drawn_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            picked = pd.concat([
                indexed.loc[(int(seed), draw, condition)]
                for seed in drawn_seeds
            ])
            # Duplicate seed and patient draws are intentional bootstrap draws.
            count = float(picked["n_predictions"].sum())
            if count == 0:
                continue
            samples.append((
                float(picked["absolute_error_sum"].sum() / count),
                float(np.sqrt(picked["squared_error_sum"].sum() / count)),
                float(picked["signed_error_sum"].sum() / count),
            ))
        values = np.asarray(samples, dtype=float)
        output[condition] = {
            "valid_replicates": int(len(values)),
            "mae_ci95": [float(value) for value in np.percentile(values[:, 0], [2.5, 97.5])] if len(values) else [np.nan, np.nan],
            "rmse_ci95": [float(value) for value in np.percentile(values[:, 1], [2.5, 97.5])] if len(values) else [np.nan, np.nan],
            "signed_bias_ci95": [float(value) for value in np.percentile(values[:, 2], [2.5, 97.5])] if len(values) else [np.nan, np.nan],
        }
    return output


def _rate_phrase(per_seed_patient: pd.DataFrame):
    """Describe the applied threshold as a rate, plus the step it came to."""
    parts = sorted({
        (round(float(r.rapid_threshold_mg_dl) / int(r.sampling_rate_minutes), 6),
         float(r.rapid_threshold_mg_dl), int(r.sampling_rate_minutes))
        for r in per_seed_patient.itertuples()
    })
    return "; ".join(
        f"{rate:g} mg/dL/min ({step:g} mg/dL over {interval:g} min)"
        for rate, step, interval in parts
    )


def analyze_rapid_change_runs(
    run_dirs: Sequence[Path], *, threshold_mg_dl: Optional[float] = None,
    bootstrap_replicates: int = 10_000, bootstrap_seed: int = 42,
):
    """Load configured seed runs, localize errors, and form nested intervals."""
    if not run_dirs:
        raise ValueError("At least one configured single-seed run directory is required")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    rows = []
    # A seed changes fitted weights, never the reference/context series that
    # assigns calm/rapid-proxy membership.
    seed_contexts: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    for run_dir_value in run_dirs:
        run_dir = Path(run_dir_value)
        results = load_experiment_results(run_dir, with_series=False)
        if results.aggregation != "single_seed" or len(results.seeds) != 1:
            raise ValueError(f"{run_dir}: must be a configured single-seed run")
        if results.sampling_rate_minutes is None:
            raise ValueError(f"{run_dir}: sampling rate is unknown")
        if results.horizon_minutes is None or results.horizon_steps is None:
            raise ValueError(f"{run_dir}: horizon is unknown")
        # Resolve per run: the recorded threshold must be the number actually
        # applied, and a run sampled at another interval would imply another
        # step for the same clinical rate.
        run_threshold = (
            rapid_threshold_mg_dl(results.sampling_rate_minutes)
            if threshold_mg_dl is None else threshold_mg_dl
        )
        for patient_id in results.patients:
            for model in results.patient_metrics[patient_id]:
                frame = load_predictions(run_dir, patient_id, model, with_context=True)
                if frame is None:
                    raise ValueError(f"{run_dir}: missing predictions for patient {patient_id}, {model}")
                context = pd.DataFrame({
                    "true_values": pd.to_numeric(frame["true_values"], errors="coerce"),
                    "preceding_target_glucose_mg_dl": pd.to_numeric(
                        frame["preceding_target_glucose_mg_dl"], errors="coerce"
                    ),
                    "preceding_target_timestamp": pd.to_datetime(
                        frame["preceding_target_timestamp"], errors="coerce", utc=True
                    ),
                    "target_timestamp": pd.to_datetime(
                        frame["target_timestamp"], errors="coerce", utc=True
                    ),
                })
                context_key = (
                    model, results.mode, results.sampling_rate_minutes,
                    results.horizon_steps, results.horizon_minutes, patient_id,
                )
                expected_context = seed_contexts.setdefault(context_key, context)
                if not expected_context.equals(context):
                    raise ValueError(
                        f"{run_dir}: reference/context differs across seeds for "
                        f"patient {patient_id}, {model}"
                    )
                metrics = rapid_change_metrics(
                    frame, sampling_rate_minutes=results.sampling_rate_minutes,
                    threshold_mg_dl=run_threshold,
                    source=f"{run_dir}: patient {patient_id}, {model}",
                )
                for condition in CONDITIONS:
                    rows.append({
                        "experiment": results.name, "run_dir": str(run_dir), "model": model,
                        "mode": results.mode, "training_seed": int(results.seeds[0]),
                        "patient_id": int(patient_id),
                        "sampling_rate_minutes": results.sampling_rate_minutes,
                        "horizon_steps": results.horizon_steps, "horizon_minutes": results.horizon_minutes,
                        "aggregation": results.aggregation, "condition": condition,
                        "rapid_threshold_mg_dl": run_threshold, **metrics[condition],
                    })
    per_seed_patient = pd.DataFrame(rows).sort_values(
        ["model", "mode", "horizon_minutes", "training_seed", "patient_id", "condition"]
    ).reset_index(drop=True)
    group_columns = ["model", "mode", "sampling_rate_minutes", "horizon_steps", "horizon_minutes"]
    summaries = []
    for group_key, group in per_seed_patient.groupby(group_columns, sort=True):
        # The exact target/context check above protects membership; retain these
        # aggregate checks as a guard against future row-assembly changes.
        count_check = group.pivot_table(
            index=["patient_id", "condition"], columns="training_seed", values="n_predictions", aggfunc="first"
        )
        excluded_check = group.pivot_table(
            index=["patient_id", "condition"], columns="training_seed", values="excluded_noncontiguous_or_invalid", aggfunc="first"
        )
        if count_check.isna().any().any() or (count_check.nunique(axis=1) != 1).any() or (excluded_check.nunique(axis=1) != 1).any():
            raise ValueError(f"rapid-change membership differs across seeds for {group_key}")
        intervals = _bootstrap_summary(group, replicates=bootstrap_replicates, random_seed=bootstrap_seed)
        for condition in CONDITIONS:
            selected = group[group.condition == condition]
            # Counts are fixed across seeds
            per_patient = selected.groupby("patient_id", as_index=False)[
                ["n_predictions", "absolute_error_sum", "squared_error_sum", "signed_error_sum"]
            ].mean()
            count = float(per_patient.n_predictions.sum())
            metric = {
                "mae_mg_dl": float(per_patient.absolute_error_sum.sum() / count) if count else np.nan,
                "rmse_mg_dl": float(np.sqrt(per_patient.squared_error_sum.sum() / count)) if count else np.nan,
                "signed_bias_mg_dl": float(per_patient.signed_error_sum.sum() / count) if count else np.nan,
            }
            summaries.append({
                **dict(zip(group_columns, group_key)), "condition": condition,
                "rapid_threshold_mg_dl": float(selected["rapid_threshold_mg_dl"].iloc[0]),
                "patients": int(per_patient.patient_id.nunique()),
                "patients_with_points": int((per_patient.n_predictions > 0).sum()),
                "predictions": count, "excluded_noncontiguous_or_invalid": float(
                    selected.groupby("patient_id").excluded_noncontiguous_or_invalid.first().sum()
                ),
                "training_seeds": int(selected.training_seed.nunique()), **metric, **intervals[condition],
            })
    summary_table = pd.DataFrame(summaries)
    summary = {
        "analysis": "rapid_change_localization",
        "unit_of_analysis": "prediction point; percentile intervals resample patients and the observed seed set",
        "evidence_level": "descriptive",
        "rapid_rate_mg_dl_per_min": sorted({
            round(float(row.rapid_threshold_mg_dl) / int(row.sampling_rate_minutes), 6)
            for row in per_seed_patient.itertuples()
        }),
        "rapid_threshold_mg_dl": sorted(
            {float(v) for v in per_seed_patient["rapid_threshold_mg_dl"]}
        ),
        "rapid_definition": (
            "one-interval rapid-change proxy: absolute reference change across exactly one sampling interval, "
            f"exceeding {_rate_phrase(per_seed_patient)}"
        ),
        "calm_definition": "the complement, over an admissible contiguous pair",
        "rapid_reference": (
            f"{RAPID_RATE_MG_DL_PER_MIN:g} mg/dL/min is the clinical threshold used by this "
            "one-interval proxy; it is not a device-arrow reimplementation"
        ),
        "gap_policy": "rows without exactly contiguous timestamps, including duplicates, are excluded",
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
    }
    return per_seed_patient, summary_table, summary


def plot_rapid_change(summary_table: pd.DataFrame, output_path: Path):
    """Plot MAE/RMSE by rapid-change condition using Matplotlib (PSF-based)."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for axis, metric, label in zip(axes, ("mae_mg_dl", "rmse_mg_dl"), ("MAE", "RMSE")):
        for index, (_, row) in enumerate(summary_table.iterrows()):
            position = index
            low, high = row[f"{metric.removesuffix('_mg_dl')}_ci95"]
            estimate = float(row[metric])
            yerr = np.clip([[estimate - low], [high - estimate]], 0, None)
            axis.errorbar(position, estimate, yerr=yerr, fmt="o", capsize=4)
        axis.set_xticks(range(len(summary_table)), [f"{r.model}\n{r.mode}\n{r.horizon_minutes:g}m\n{r.condition}" for r in summary_table.itertuples()])
        axis.set_ylabel(f"{label} (mg/dL)")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Forecast error at calm versus rapid glucose changes")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
