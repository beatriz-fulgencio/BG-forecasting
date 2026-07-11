"""
Scatter of per-patient distributional shift vs prediction error.

Builds a compact, high-information figure: one point per patient per horizon,
``shift_score`` (Wasserstein train->test glucose shift, x) against ``MAE`` (y),
colored by prediction horizon, with a per-horizon trend line and Spearman
correlation annotated. This tests whether patients whose test glucose drifts
further from training are harder to predict.

Aggregates across several single-experiment result directories (one per
horizon), reusing ``PatientAnalyzer`` to obtain per-patient ``shift_score`` and
``mae`` for a chosen model.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from scipy.stats import spearmanr  # type: ignore

# NOTE: PatientAnalyzer is imported lazily inside build_shift_mae_dataframe (the
# only function that needs it). Importing it at module top pulls in sklearn +
# seaborn alongside torch, which can deadlock (OpenMP) for callers that only need
# the lightweight parse_horizon / HORIZON_COLORS helpers.

# Fixed color per horizon so figures are comparable across runs.
HORIZON_COLORS: Dict[int, str] = {
    15: "#1b9e77",
    30: "#d95f02",
    45: "#7570b3",
    60: "#e7298a",
}


def parse_horizon(name: str) -> Optional[int]:
    """Extract the prediction horizon in minutes from an experiment dir name.

    e.g. ``experiment_20251104_034100_30min_tl`` -> ``30``. Returns None if no
    ``<n>min`` token is present.
    """
    match = re.search(r"(\d+)\s*min", name)
    return int(match.group(1)) if match else None


def build_shift_mae_dataframe(experiment_dirs: List[str],
                              data_root: str = "data",
                              model_name: str = "GRU") -> pd.DataFrame:
    """
    Assemble a per-patient (shift_score, mae, horizon) table across experiments.

    Args:
        experiment_dirs: Single-experiment result directories (each containing a
            ``comprehensive_metrics_*.json``). The horizon is parsed from each
            directory name.
        data_root: Raw data location (used to compute per-patient shift_score).
        model_name: Model whose MAE to use (default ``"GRU"``).

    Returns:
        DataFrame with columns
        ``patient_id, horizon, shift_score, kl_divergence, mae, experiment``.
        Rows with missing shift_score or mae are dropped.
    """
    from .patient_analysis import PatientAnalyzer  # lazy: see note at module top

    frames = []
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        horizon = parse_horizon(exp_path.name)
        if horizon is None:
            print(f"    Skipping (no horizon in name): {exp_path.name}")
            continue

        analyzer = PatientAnalyzer(str(exp_path))
        analyzer.load_experiment_results()
        analyzer.data_root = data_root
        df = analyzer.create_performance_dataframe(model_name)

        if df is None or df.empty or "shift_score" not in df.columns or "mae" not in df.columns:
            print(f"    No usable shift_score/mae for model '{model_name}' in {exp_path.name}")
            continue

        keep = ["patient_id", "shift_score", "kl_divergence", "mae"]
        keep = [c for c in keep if c in df.columns]
        sub = df[keep].copy()
        sub["horizon"] = horizon
        sub["experiment"] = exp_path.name
        frames.append(sub)

    if not frames:
        return pd.DataFrame(
            columns=["patient_id", "horizon", "shift_score", "kl_divergence", "mae", "experiment"]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["shift_score", "mae"])
    return combined.sort_values(["horizon", "patient_id"]).reset_index(drop=True)


def plot_shift_vs_mae(df: pd.DataFrame,
                      output_path: str,
                      model_name: str = "GRU",
                      title: Optional[str] = None) -> str:
    """
    Render the shift_score-vs-MAE scatter (single panel, colored by horizon).

    Each point is one patient at one horizon; each horizon gets a least-squares
    trend line and a Spearman r (annotated in the legend). Also reports the
    pooled Spearman correlation across all points.

    Args:
        df: Output of :func:`build_shift_mae_dataframe`.
        output_path: Where to save the PNG.
        model_name: Used only for labelling.
        title: Optional custom title.

    Returns:
        The output path.
    """
    if df.empty:
        raise ValueError("No data to plot: shift/MAE DataFrame is empty.")

    fig, ax = plt.subplots(figsize=(8, 6))

    for horizon in sorted(df["horizon"].unique()):
        g = df[df["horizon"] == horizon]
        color = HORIZON_COLORS.get(horizon, None)

        # Per-horizon Spearman correlation (needs >= 3 points to be meaningful).
        if len(g) >= 3:
            rho, p = spearmanr(g["shift_score"], g["mae"])
            label = f"{horizon} min (ρ={rho:.2f}, p={p:.3f}, n={len(g)})"
        else:
            rho, label = np.nan, f"{horizon} min (n={len(g)})"

        ax.scatter(g["shift_score"], g["mae"], color=color, s=55,
                   alpha=0.85, edgecolor="white", linewidth=0.5, label=label, zorder=3)

        # Least-squares trend line across the horizon's x-range.
        if len(g) >= 2 and g["shift_score"].nunique() >= 2:
            slope, intercept = np.polyfit(g["shift_score"], g["mae"], 1)
            xs = np.linspace(g["shift_score"].min(), g["shift_score"].max(), 50)
            ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.8, alpha=0.7, zorder=2)

    # Pooled correlation across all points.
    if len(df) >= 3:
        rho_all, p_all = spearmanr(df["shift_score"], df["mae"])
        pooled = f"Pooled Spearman ρ = {rho_all:.2f} (p = {p_all:.3f}, n = {len(df)})"
    else:
        pooled = ""

    ax.set_xlabel("Distributional shift  (Wasserstein train→test glucose, mg/dL)")
    ax.set_ylabel(f"{model_name} MAE  (mg/dL)")
    ax.set_title(title or f"Per-patient distributional shift vs {model_name} error")
    ax.legend(title="Prediction horizon", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25, zorder=0)
    if pooled:
        ax.text(0.02, 0.98, pooled, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved shift-vs-MAE scatter: {out}")
    return str(out)
