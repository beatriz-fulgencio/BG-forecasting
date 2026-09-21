"""Paired transfer-learning versus regular-learning inference."""

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..results_io import (
    get_modes,
    get_seeds,
    load_experiment_results,
    require_grouping_metadata,
)
from .statistical_significance import StatisticalSignificanceTester


CELL_COLUMNS = ("model", "horizon_minutes", "horizon_steps", "sampling_rate_minutes")


def _validate_seed_maps(values: Mapping[int, Mapping[int, float]], *, label: str):
    if not values:
        raise ValueError(f"{label}: no completed seeds")
    cohorts = {seed: set(table) for seed, table in values.items()}
    expected = cohorts[next(iter(cohorts))]
    if not expected:
        raise ValueError(f"{label}: empty patient cohort")
    for seed, cohort in cohorts.items():
        if cohort != expected:
            raise ValueError(f"{label}: seed {seed} has a different patient cohort")
        if not np.all(np.isfinite(list(values[seed].values()))):
            raise ValueError(f"{label}: seed {seed} contains non-finite MAE")
    return tuple(sorted(expected))


def paired_transfer_bootstrap(
    regular: Mapping[int, Mapping[int, float]],
    transfer: Mapping[int, Mapping[int, float]],
    *,
    replicates: int = 10_000, 
    random_seed: int = 42,
):
    """Two nested patient x seed intervals for the TL-RL effect.

    Both resample whole patients inside a replicate, so patients evaluated under one fitted model stay together. They differ in what they assume about
    training seeds, and therefore in what they are intervals *for*.

    ``ci95``
        Seeds are resampled with replacement and averaged within the replicate.
        This is a confidence interval for the cross-seed mean effect: the same
        quantity ``tl_minus_rl_mae_mg_dl`` estimates, and the same quantity the
        Wilcoxon and sign tests are testing, since those run on patient
        cross-seed means. It narrows as seeds are added.

    ``pi95``
        Exactly one seed is drawn for the whole replicate, matching
        and the shift screens. It spans where a *single* randomly drawn fitted
        model's effect falls, so it does not narrow as seeds are added and must
        not be read as uncertainty about the reported mean.
        
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if set(regular) != set(transfer):
        raise ValueError("regular and transfer must have identical completed seed sets")
    patients = _validate_seed_maps(regular, label="regular")
    transfer_patients = _validate_seed_maps(transfer, label="transfer")
    if patients != transfer_patients:
        raise ValueError("regular and transfer must have identical patient cohorts")
    seeds = tuple(sorted(regular))
    reg = np.asarray([[regular[seed][patient] for patient in patients] for seed in seeds], dtype=float)
    tl = np.asarray([[transfer[seed][patient] for patient in patients] for seed in seeds], dtype=float)
    differences = tl - reg
    n_seeds, n_patients = len(seeds), len(patients)

    def _draw(resample_seeds: bool, rng: np.random.Generator):
        """Replicate-level absolute and relative TL-RL effects."""
        absolute = np.empty(replicates, dtype=float)
        relative = np.empty(replicates, dtype=float)
        for index in range(replicates):
            seed_indices = (
                rng.integers(0, n_seeds, size=n_seeds) if resample_seeds
                else rng.integers(0, n_seeds, size=1)
            )
            patient_indices = rng.integers(0, n_patients, size=n_patients)
            block = np.ix_(seed_indices, patient_indices)
            absolute[index] = float(np.mean(differences[block]))
            baseline = float(np.mean(reg[block]))
            relative[index] = 100 * absolute[index] / baseline if baseline else np.nan
        return absolute, relative

    pi_absolute, pi_relative = _draw(False, np.random.default_rng(random_seed))
    ci_absolute, ci_relative = _draw(True, np.random.default_rng(np.random.SeedSequence([random_seed, 1])))

    def _interval(values: np.ndarray):
        finite = values[np.isfinite(values)]
        if not finite.size:
            return np.nan, np.nan, 0
        return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)), int(finite.size)

    ci_low, ci_high, _ = _interval(ci_absolute)
    pi_low, pi_high, _ = _interval(pi_absolute)
    ci_pct_low, ci_pct_high, ci_pct_valid = _interval(ci_relative)
    pi_pct_low, pi_pct_high, pi_pct_valid = _interval(pi_relative)
    return {
        "tl_minus_rl_mae_mg_dl": float(np.mean(differences)),
        "tl_minus_rl_mae_ci95_low": ci_low, "tl_minus_rl_mae_ci95_high": ci_high,
        "tl_minus_rl_mae_pi95_low": pi_low, "tl_minus_rl_mae_pi95_high": pi_high,
        "tl_minus_rl_pct": float(100 * np.mean(differences) / np.mean(reg)),
        "tl_minus_rl_pct_ci95_low": ci_pct_low, "tl_minus_rl_pct_ci95_high": ci_pct_high,
        "tl_minus_rl_pct_pi95_low": pi_pct_low, "tl_minus_rl_pct_pi95_high": pi_pct_high,
        "ci95_interpretation": "confidence interval for the cross-seed mean effect",
        "pi95_interpretation": "spread of one randomly drawn fitted model's effect",
        "n_patients": n_patients, "training_seeds": n_seeds,
        "matched_seeds": ",".join(map(str, seeds)), "replicates": replicates,
        "pct_ci95_valid_replicates": ci_pct_valid, "pct_pi95_valid_replicates": pi_pct_valid,
        "random_seed": random_seed,
    }


def _load_parent(parent: Path):
    """Load one parent run into seed-level and cross-seed patient tables."""
    modes = set(get_modes(parent))
    if not {"regular", "transfer"} <= modes:
        raise ValueError(f"{parent}: transfer inference requires regular and transfer modes")
    regular_seeds, transfer_seeds = set(get_seeds(parent, "regular")), set(get_seeds(parent, "transfer"))
    if regular_seeds != transfer_seeds:
        raise ValueError(
            f"{parent}: modes have unequal seed sets regular={sorted(regular_seeds)}, "
            f"transfer={sorted(transfer_seeds)}"
        )
    if not regular_seeds:
        raise ValueError(f"{parent}: no completed regular/transfer seed pairs")

    rows = []
    metadata = None
    for seed in sorted(regular_seeds):
        regular = load_experiment_results(parent, mode="regular", seed=seed, with_series=False)
        transfer = load_experiment_results(parent, mode="transfer", seed=seed, with_series=False)
        require_grouping_metadata(parent, regular)
        require_grouping_metadata(parent, transfer)
        if (
            tuple(getattr(regular, field) for field in CELL_COLUMNS[1:])
            != tuple(getattr(transfer, field) for field in CELL_COLUMNS[1:])
            or regular.models != transfer.models
        ):
            raise ValueError(f"{parent} seed {seed}: regular/transfer metadata disagree")
        if set(regular.patients) != set(transfer.patients):
            raise ValueError(f"{parent} seed {seed}: regular/transfer patient cohorts disagree")
        metadata = regular
        for patient_id in regular.patients:
            for model in regular.models:
                regular_metrics = regular.metrics_for(patient_id, model)
                transfer_metrics = transfer.metrics_for(patient_id, model)
                if transfer_metrics is None or regular_metrics is None:
                    raise ValueError(f"{parent} seed {seed}: missing paired {model} metrics for {patient_id}")
                regular_mae, transfer_mae = regular_metrics.get("mae"), transfer_metrics.get("mae")
                if not isinstance(regular_mae, (int, float)) or not isinstance(transfer_mae, (int, float)):
                    raise ValueError(f"{parent} seed {seed}: patient {patient_id} has no numeric MAE")
                rows.append({
                    "experiment": regular.name, "experiment_dir": str(parent), "model": model,
                    "horizon_minutes": regular.horizon_minutes, "horizon_steps": regular.horizon_steps,
                    "sampling_rate_minutes": regular.sampling_rate_minutes, "patient_id": patient_id,
                    "training_seed": seed, "regular_mae_mg_dl": float(regular_mae),
                    "transfer_mae_mg_dl": float(transfer_mae),
                    "tl_minus_rl_mae_mg_dl": float(transfer_mae - regular_mae),
                    "tl_minus_rl_pct": float(100 * (transfer_mae - regular_mae) / regular_mae)
                    if regular_mae else np.nan,
                })
    per_seed = pd.DataFrame(rows)
    if metadata is None:
        raise ValueError(f"{parent}: no readable paired runs")
    # Validate that all seeds retained the same patient cohort for every model.
    for model, group in per_seed.groupby("model"):
        _validate_seed_maps(
            {seed: dict(zip(frame.patient_id, frame.regular_mae_mg_dl))
             for seed, frame in group.groupby("training_seed")},
            label=f"{parent} {model} regular",
        )
    per_patient = per_seed.groupby(
        ["experiment", "experiment_dir", "model", "horizon_minutes", "horizon_steps", "sampling_rate_minutes", "patient_id"],
        as_index=False,
    ).agg(
        regular_mae_mg_dl=("regular_mae_mg_dl", "mean"),
        transfer_mae_mg_dl=("transfer_mae_mg_dl", "mean"),
        tl_minus_rl_mae_mg_dl=("tl_minus_rl_mae_mg_dl", "mean"),
        n_matched_seeds=("training_seed", "nunique"),
    )
    per_patient["tl_minus_rl_pct"] = 100 * per_patient["tl_minus_rl_mae_mg_dl"] / per_patient["regular_mae_mg_dl"]
    return per_seed, per_patient


def analyze_transfer_inference(
    parents: Sequence[Path], *, replicates: int = 10_000, random_seed: int = 42
):
    """Produce seed-level, patient-mean, and inferential TL−RL tables."""
    if not parents:
        raise ValueError("At least one configured parent experiment is required")
    loaded = [_load_parent(Path(parent)) for parent in parents]
    per_seed = pd.concat([item[0] for item in loaded], ignore_index=True)
    per_patient = pd.concat([item[1] for item in loaded], ignore_index=True)
    if per_patient.duplicated([*CELL_COLUMNS, "patient_id"]).any():
        raise ValueError(
            "More than one parent supplied the same model/horizon/patient cell "
            "(including horizon steps and sampling rate)"
        )

    tester = StatisticalSignificanceTester()
    rows = []
    for cell, patient_frame in per_patient.groupby(list(CELL_COLUMNS), sort=True):
        model, horizon, horizon_steps, sampling_rate = cell
        seed_frame = per_seed[
            (per_seed.model == model)
            & (per_seed.horizon_minutes == horizon)
            & (per_seed.horizon_steps == horizon_steps)
            & (per_seed.sampling_rate_minutes == sampling_rate)
        ]
        regular = {seed: dict(zip(frame.patient_id, frame.regular_mae_mg_dl))
                   for seed, frame in seed_frame.groupby("training_seed")}
        transfer = {seed: dict(zip(frame.patient_id, frame.transfer_mae_mg_dl))
                    for seed, frame in seed_frame.groupby("training_seed")}
        bootstrap = paired_transfer_bootstrap(regular, transfer, replicates=replicates, random_seed=random_seed)
        paired = tester.paired_comparison(
            patient_frame.regular_mae_mg_dl.to_numpy(), patient_frame.transfer_mae_mg_dl.to_numpy(),
            test_type="non-parametric",
        )
        rows.append({
            "model": model, "horizon_minutes": horizon,
            "horizon_steps": horizon_steps, "sampling_rate_minutes": sampling_rate,
            "regular_mae_patient_mean_mg_dl": float(patient_frame.regular_mae_mg_dl.mean()),
            "transfer_mae_patient_mean_mg_dl": float(patient_frame.transfer_mae_mg_dl.mean()),
            "wilcoxon_statistic": paired["statistic"], "wilcoxon_p_value": paired["p_value"],
            "exact_sign_p_value": paired["sign_test_p_value"], "cohens_dz": paired["cohens_dz"],
            "rank_biserial": paired["rank_biserial"], "effect_direction": paired["effect_direction"],
            # The shared tester counts a "win" as data2 > data1, and data2 is
            # transfer, so its wins are patients transfer made *worse*. Spelling
            # that out here rather than forwarding "wins" keeps a reader of the
            # published CSV from reading the column as "transfer won".
            "patients_transfer_better": paired["losses"],
            "patients_transfer_worse": paired["wins"],
            "patients_tied": paired["ties"],
            "unit_of_analysis": "patient cross-seed mean", "evidence_level": "confirmatory",
            **bootstrap,
        })
    if not rows:
        raise ValueError(
            "No model/horizon cell survived assembly; the parent runs share no comparable cell"
        )
    inference = pd.DataFrame(rows)
    inference["wilcoxon_p_value_bh"] = tester.benjamini_hochberg_correction(inference.wilcoxon_p_value.tolist())
    inference["exact_sign_p_value_bh"] = tester.benjamini_hochberg_correction(inference.exact_sign_p_value.tolist())
    inference["wilcoxon_significant_bh"] = inference.wilcoxon_p_value_bh < tester.alpha
    inference["exact_sign_significant_bh"] = inference.exact_sign_p_value_bh < tester.alpha
    summary = {
        "analysis": "transfer_inference", "unit_of_analysis": "patient cross-seed mean",
        "evidence_level": "confirmatory", "difference_convention": "transfer MAE minus regular MAE; negative favours transfer",
        "parents": [str(Path(parent)) for parent in parents], "models": sorted(inference.model.unique().tolist()),
        "horizons_minutes": sorted(inference.horizon_minutes.unique().tolist()), "bootstrap_replicates": replicates,
        "bootstrap_seed": random_seed, "multiple_testing": "Benjamini-Hochberg separately for Wilcoxon and sign-test horizon/model families",
        "interval_conventions": {
            "ci95": "confidence interval for the cross-seed mean effect; seeds resampled with replacement",
            "pi95": "spread of one randomly drawn fitted model's effect; one shared seed per replicate",
        }
    }
    return per_seed, per_patient, inference, summary


def plot_transfer_inference(inference: pd.DataFrame, output_path: Path):
    """Plot TL−RL bootstrap intervals by horizon using Matplotlib (PSF-based)."""
    import matplotlib.pyplot as plt

    if inference.empty:
        raise ValueError("Cannot plot an empty transfer-inference table")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for (model, steps, rate), frame in inference.groupby(
        ["model", "horizon_steps", "sampling_rate_minutes"], sort=True
    ):
        ordered = frame.sort_values("horizon_minutes")
        estimate = ordered["tl_minus_rl_mae_mg_dl"].to_numpy(dtype=float)
        lower = ordered["tl_minus_rl_mae_ci95_low"].to_numpy(dtype=float)
        upper = ordered["tl_minus_rl_mae_ci95_high"].to_numpy(dtype=float)
        # A percentile interval is not forced to straddle the point estimate, and Matplotlib rejects a negative yerr, so clip rather than crash on a plot.
        yerr = np.vstack([
            np.clip(estimate - lower, 0, None), np.clip(upper - estimate, 0, None),
        ])
        axis.errorbar(
            ordered["horizon_minutes"], estimate, yerr=yerr, marker="o",
            capsize=3, label=f"{model} ({steps:g} steps, {rate:g}-min)",
        )
    axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Prediction horizon (minutes)")
    axis.set_ylabel("Transfer − regular MAE (mg/dL)")
    axis.set_title("Transfer-learning effect with nested patient × seed 95% CIs")
    axis.grid(alpha=0.25)
    axis.legend(title="Model")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
