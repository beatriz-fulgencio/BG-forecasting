"""Descriptive stability of patient difficulty across model conditions.

Implementation references and licences
--------------------------------------
Ranks and Spearman correlations call SciPy (BSD-3-Clause); Benjamini-Hochberg
adjustment calls statsmodels (BSD-3-Clause) through
``StatisticalSignificanceTester``; tabulation and resampling-free aggregation use
pandas and NumPy (BSD-3-Clause). Kendall's W is the standard tie-corrected
rank-concordance formula (Kendall & Babington Smith, 1939), implemented from the
published equation; no third-party source code is copied. It is conformance
tested against ``scipy.stats.friedmanchisquare`` via the identity
``chi2_F = m (n - 1) W``, and against the Spearman identity
``W = ((m - 1) * mean pairwise rho + 1) / m``, in
``tests/test_reference_conformance.py``.

Everything here is descriptive. The Spearman p-values test ``rho = 0`` --- that
patient difficulty is *unrelated* across conditions --- which is not the question
this analysis asks; ``spearman_rho`` and its spread are the quantities to read.
The p-values are nevertheless Benjamini-Hochberg adjusted within each declared
family, because an unadjusted column of hundreds of p-values invites exactly the
thresholding this project's ``CORRELATION_MULTIPLE_COMPARISON.md`` warns about.
"""

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from ..results_io import (
    get_modes,
    get_seeds,
    load_experiment_results,
    require_grouping_metadata,
)
from .statistical_significance import StatisticalSignificanceTester

IMPLEMENTATION_REFERENCES = {
    "ranks_and_spearman": {"implementation": "SciPy", "license": "BSD-3-Clause"},
    "benjamini_hochberg": {"implementation": "statsmodels", "license": "BSD-3-Clause"},
    "aggregation": {"implementation": "pandas/NumPy", "license": "BSD-3-Clause"},
    "kendalls_w": {
        "implementation": "Kendall & Babington Smith (1939) formula",
        "license": "formula",
        "conformance": "scipy.stats.friedmanchisquare, chi2_F = m (n - 1) W",
    },
}

# Declared multiple-testing families. Each table is corrected on its own, and the
# family size is the number of rows in it.
SPEARMAN_FAMILIES = {
    "pairwise_spearman": "rank agreement between model/horizon/mode conditions",
    "seed_rank_sensitivity": "rank agreement between training seeds within a condition",
}


def kendalls_w(rank_matrix: np.ndarray):
    """Tie-corrected Kendall coefficient of concordance for patient ranks.
    """
    # Reference: Kendall, M. G. & Babington Smith, B. (1939), "The Problem of
    # m Rankings", Annals of Mathematical Statistics 10(3), 275-287, equations
    # for W and its tie correction. Transcribed; SciPy ships no W.
    #
    #              12 S                                    _
    #   W = --------------------- ,   S = sum_i (R_i - R)^2
    #       m^2 (n^3 - n) - m T
    #
    #   R_i = sum over raters of object i's rank
    #   T   = sum over raters j of sum_k (t_jk^3 - t_jk), t_jk the sizes of
    #         rater j's groups of tied ranks
    #
    
    ranks = np.asarray(rank_matrix, dtype=float)
    if ranks.ndim != 2 or min(ranks.shape) < 2 or not np.all(np.isfinite(ranks)):
        return np.nan
    n, m = ranks.shape  # patients, raters/conditions
    sums = ranks.sum(axis=1)
    s = float(np.square(sums - sums.mean()).sum())
    tie_term = 0.0
    for column in ranks.T:
        _, counts = np.unique(column, return_counts=True)
        tie_term += float(np.sum(counts ** 3 - counts))
    denominator = m * m * (n ** 3 - n) - m * tie_term
    return 12 * s / denominator if denominator > 0 else np.nan


def common_cohort_ranks(pivot: pd.DataFrame):
    """Ranks over the patients every condition covers, re-ranked within column.
    """
    common = pivot.dropna()
    if common.empty:
        return np.empty((0, pivot.shape[1]), dtype=float)
    return np.column_stack([
        rankdata(common[column].to_numpy(dtype=float), method="average")
        for column in common.columns
    ])


def _spearman(pivot: pd.DataFrame, first: Any, second: Any):
    """Spearman rho and p-value over the patients both conditions rank."""
    paired = pivot[[first, second]].dropna()
    if len(paired) < 3:
        return np.nan, np.nan, len(paired)
   
    result = spearmanr(paired[first].to_numpy(), paired[second].to_numpy())
    return float(result.statistic), float(result.pvalue), len(paired)


def _adjust_family(frame: pd.DataFrame, family: str):
    """Benjamini-Hochberg adjust one declared family of Spearman p-values."""
    if frame.empty:
        for column in ("spearman_p_value_bh", "spearman_significant_bh"):
            frame[column] = pd.Series(dtype="float64" if column.endswith("bh") else "bool")
        frame["family"] = pd.Series(dtype="object")
        frame["family_size"] = pd.Series(dtype="int64")
        return frame
    tester = StatisticalSignificanceTester()
    frame = frame.copy()
    frame["spearman_p_value_bh"] = tester.benjamini_hochberg_correction(
        frame["spearman_p_value"].tolist()
    )
    frame["spearman_significant_bh"] = frame["spearman_p_value_bh"] < tester.alpha
    frame["family"] = family
    frame["family_size"] = len(frame)
    return frame


def load_patient_stability(parents: Sequence[Path]):
    """Load one MAE row per patient x model x horizon x mode x seed."""
    if not parents:
        raise ValueError("At least one configured parent experiment is required")
    rows: List[Dict[str, Any]] = []
    for parent_value in parents:
        parent = Path(parent_value)
        modes = get_modes(parent)
        if not modes:
            raise ValueError(f"{parent}: no configured modes")
        for mode in modes:
            seeds = get_seeds(parent, mode)
            if not seeds:
                raise ValueError(f"{parent}: mode {mode} has no complete seeds")
            for seed in seeds:
                result = load_experiment_results(parent, mode=mode, seed=seed, with_series=False)
                # Horizon, steps, and sampling rate are all groupby keys below.
                require_grouping_metadata(parent, result)
                for patient_id in result.patients:
                    for model, metrics in result.patient_metrics[patient_id].items():
                        mae = metrics.get("mae")
                        if not isinstance(mae, (int, float)) or not np.isfinite(mae):
                            raise ValueError(f"{parent} {mode} seed {seed}: invalid MAE for {patient_id}/{model}")
                        rows.append({"experiment": result.name, "experiment_dir": str(parent), "patient_id": patient_id,
                                     "model": model, "horizon_minutes": result.horizon_minutes,
                                     "horizon_steps": result.horizon_steps, "sampling_rate_minutes": result.sampling_rate_minutes,
                                     "mode": mode, "training_seed": seed, "mae_mg_dl": float(mae)})
    frame = pd.DataFrame(rows)
    if frame.duplicated(["model", "horizon_minutes", "mode", "training_seed", "patient_id"]).any():
        raise ValueError("Duplicate patient/model/horizon/mode/seed MAE rows")
    # scipy.stats.rankdata, method="average" (SciPy, BSD-3-Clause): rank 1 is the
    # lowest MAE, so a low rank means an easy patient, and tied MAEs share a
    # midrank rather than being ordered arbitrarily.
    frame["patient_rank"] = frame.groupby(["model", "horizon_minutes", "mode", "training_seed"])["mae_mg_dl"].transform(
        lambda values: rankdata(values, method="average")
    )
    return frame.sort_values(["model", "horizon_minutes", "mode", "training_seed", "patient_id"]).reset_index(drop=True)


def summarize_patient_stability(seed_table: pd.DataFrame):
    """Return cross-seed patient means, Spearman pairs, W, and difficulty spreads."""
    if seed_table.empty:
        raise ValueError("Cannot summarise an empty patient-stability table")
    group = ["patient_id", "model", "horizon_minutes", "horizon_steps", "sampling_rate_minutes", "mode"]
    patient = seed_table.groupby(group, as_index=False).agg(mae_mg_dl=("mae_mg_dl", "mean"),
        mae_seed_sd_mg_dl=("mae_mg_dl", "std"), n_seeds=("training_seed", "nunique"))

    patient["patient_rank"] = patient.groupby(["model", "horizon_minutes", "mode"])["mae_mg_dl"].transform(
        lambda values: rankdata(values, method="average")
    )
    patient["condition"] = patient.apply(lambda row: f"{row.model}|{row.horizon_minutes}m|{row['mode']}", axis=1)
    pivot = patient.pivot(index="patient_id", columns="condition", values="patient_rank")

    spearman_rows = []
    for first, second in combinations(pivot.columns, 2):
        rho, p_value, n_patients = _spearman(pivot, first, second)
        spearman_rows.append({"condition_1": first, "condition_2": second, "n_patients": n_patients,
                              "spearman_rho": rho, "spearman_p_value": p_value,
                              "evidence_level": "descriptive"})

    concordance_rows = []
    for (horizon, mode), frame in patient.groupby(["horizon_minutes", "mode"], sort=True):
        matrix = common_cohort_ranks(frame.pivot(index="patient_id", columns="model", values="patient_rank"))
        concordance_rows.append({"scope": "models_within_horizon_mode", "horizon_minutes": horizon, "mode": mode,
                                 "n_patients": len(matrix), "n_conditions": matrix.shape[1], "kendalls_w": kendalls_w(matrix),
                                 "evidence_level": "descriptive"})
    all_matrix = common_cohort_ranks(pivot)
    concordance_rows.append({"scope": "all_model_horizon_mode_conditions", "horizon_minutes": np.nan, "mode": "all",
                             "n_patients": len(all_matrix), "n_conditions": all_matrix.shape[1], "kendalls_w": kendalls_w(all_matrix),
                             "evidence_level": "descriptive"})

    spread_rows = []
    for (model, horizon, mode), frame in patient.groupby(["model", "horizon_minutes", "mode"], sort=True):
        ordered = frame.sort_values("mae_mg_dl")
        # Disjoint tails: with fewer than two patients per quartile the same
        # patient would otherwise be counted as both the easy and the hard end.
        count = max(1, int(np.ceil(len(ordered) / 4)))
        if 2 * count > len(ordered):
            count = len(ordered) // 2
        easy = float(ordered.head(count).mae_mg_dl.mean()) if count else np.nan
        hard = float(ordered.tail(count).mae_mg_dl.mean()) if count else np.nan
        spread_rows.append({"model": model, "horizon_minutes": horizon, "mode": mode, "n_patients": len(ordered),
                            "n_per_quartile": count,
                            "easy_quartile_mae_mg_dl": easy, "hard_quartile_mae_mg_dl": hard,
                            "hard_easy_spread_mg_dl": hard - easy,
                            "patient_mae_range_mg_dl": float(ordered.mae_mg_dl.max() - ordered.mae_mg_dl.min()),
                            "evidence_level": "descriptive"})
    return (patient, _adjust_family(pd.DataFrame(spearman_rows), "pairwise_spearman"),
            pd.DataFrame(concordance_rows), pd.DataFrame(spread_rows))


def seed_rank_sensitivity(seed_table: pd.DataFrame):
    """Pairwise Spearman stability of patient ranks across training seeds."""
    rows = []
    for (model, horizon, mode), frame in seed_table.groupby(["model", "horizon_minutes", "mode"], sort=True):
        pivot = frame.pivot(index="patient_id", columns="training_seed", values="patient_rank")
        for first, second in combinations(pivot.columns, 2):
            rho, p_value, n_patients = _spearman(pivot, first, second)
            rows.append({"model": model, "horizon_minutes": horizon, "mode": mode,
                         "seed_1": first, "seed_2": second, "n_patients": n_patients,
                         "spearman_rho": rho, "spearman_p_value": p_value, "evidence_level": "descriptive"})
    return _adjust_family(pd.DataFrame(rows), "seed_rank_sensitivity")


def plot_hard_easy_spread(spread: pd.DataFrame, output_path: Path):
    """Plot descriptive hard/easy MAE spreads with Matplotlib (PSF-based)."""
    import matplotlib.pyplot as plt

    if spread.empty:
        raise ValueError("Cannot plot an empty stability spread table")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for (model, mode), frame in spread.groupby(["model", "mode"], sort=True):
        ordered = frame.sort_values("horizon_minutes")
        axis.plot(ordered.horizon_minutes, ordered.hard_easy_spread_mg_dl, marker="o", label=f"{model} {mode}")
    axis.set(xlabel="Prediction horizon (minutes)", ylabel="Hard − easy quartile MAE (mg/dL)", title="Patient difficulty spread")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
