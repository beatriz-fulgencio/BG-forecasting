"""
Figures for per-patient distributional shift vs prediction error.

Two renderings of the same table, one point per patient per horizon, plotting
``shift_score`` (Wasserstein train->test glucose shift, x) against ``MAE`` (y):

- :func:`plot_shift_vs_mae` -- a single panel, colored by horizon, for comparing
  horizons against each other.
- :func:`plot_shift_vs_mae_faceted` -- one panel per horizon, for reading a
  single horizon's relationship without the others overlapping it.

Both take the long patient x horizon table built by the Phase-1 analysis
(``RUN/run_phase1_shift_analysis.py``), which owns the statistics; this module
only draws. Correlations are annotated per horizon and never pooled across
horizons, because each patient contributes a row to every horizon and pooling
them would count one patient as several independent observations.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from scipy.stats import pearsonr, spearmanr  # type: ignore

# Fixed color per horizon so figures are comparable across runs.
HORIZON_COLORS: Dict[int, str] = {
    15: "#1b9e77",
    30: "#d95f02",
    45: "#7570b3",
    60: "#e7298a",
}

_REQUIRED_COLUMNS = ("horizon", "shift_score", "mae")


def parse_horizon(name: str) -> Optional[int]:
    """Extract the prediction horizon in minutes from an experiment dir name.

    e.g. ``experiment_20251104_034100_30min_tl`` -> ``30``. Returns None if no
    ``<n>min`` token is present.
    """
    match = re.search(r"(\d+)\s*min", name)
    return int(match.group(1)) if match else None


def _plottable(df: pd.DataFrame, horizons: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Drop rows that cannot be plotted, and optionally restrict to horizons.

    The analysis table carries a row for every patient at every horizon, with
    NaN where a patient has no MAE, so dropping here keeps every caller from
    having to remember to.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"shift/MAE table is missing column(s): {', '.join(missing)}")
    if horizons is not None:
        df = df[df["horizon"].isin(list(horizons))]
    return df.dropna(subset=["shift_score", "mae"])


def _trend_line(ax, g: pd.DataFrame, color, linewidth: float) -> None:
    if len(g) >= 2 and g["shift_score"].nunique() >= 2:
        slope, intercept = np.polyfit(g["shift_score"], g["mae"], 1)
        xs = np.linspace(g["shift_score"].min(), g["shift_score"].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=color, linewidth=linewidth,
                alpha=0.75, zorder=2)


def plot_shift_vs_mae(df: pd.DataFrame,
                      output_path: str,
                      model_name: str = "GRU",
                      horizons: Optional[Sequence[int]] = None,
                      title: Optional[str] = None) -> str:
    """
    Render the shift_score-vs-MAE scatter (single panel, colored by horizon).

    Each point is one patient at one horizon; each horizon gets a least-squares
    trend line and a Spearman rho (annotated in the legend). Correlations are
    reported separately by horizon because each patient appears once per horizon.

    Args:
        df: Long patient x horizon table with ``horizon``, ``shift_score`` and
            ``mae`` columns. Rows missing shift_score or mae are dropped.
        output_path: Where to save the PNG.
        model_name: Used only for labelling.
        horizons: Restrict to these horizons; default plots every horizon present.
        title: Optional custom title.

    Returns:
        The output path.
    """
    df = _plottable(df, horizons)
    if df.empty:
        raise ValueError("No data to plot: shift/MAE table has no complete rows.")

    fig, ax = plt.subplots(figsize=(8, 6))

    for horizon in sorted(df["horizon"].unique()):
        g = df[df["horizon"] == horizon]
        color = HORIZON_COLORS.get(horizon, None)

        # Per-horizon Spearman correlation (needs >= 3 points to be meaningful).
        if len(g) >= 3:
            rho, p = spearmanr(g["shift_score"], g["mae"])
            label = f"{horizon} min (ρ={rho:.2f}, p={p:.3f}, n={len(g)})"
        else:
            label = f"{horizon} min (n={len(g)})"

        ax.scatter(g["shift_score"], g["mae"], color=color, s=55,
                   alpha=0.85, edgecolor="white", linewidth=0.5, label=label, zorder=3)
        _trend_line(ax, g, color, linewidth=1.8)

    ax.set_xlabel("Distributional shift  (Wasserstein train→test glucose, mg/dL)")
    ax.set_ylabel(f"{model_name} MAE  (mg/dL)")
    ax.set_title(title or f"Per-patient distributional shift vs {model_name} error")
    ax.legend(title="Prediction horizon", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25, zorder=0)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved shift-vs-MAE scatter: {out}")
    return str(out)


def plot_shift_vs_mae_faceted(df: pd.DataFrame,
                              horizons: Sequence[int],
                              output_path: str,
                              model_name: str = "GRU") -> str:
    """
    Render one shift_score-vs-MAE panel per horizon on a shared grid.

    Each panel is a self-contained n-patient scatter, so its Pearson/Spearman
    annotation is a within-horizon correlation over independent patients.

    Args:
        df: Long patient x horizon table (see :func:`plot_shift_vs_mae`).
        horizons: Horizons to draw, one panel each, in the given order.
        output_path: Where to save the PNG.
        model_name: Used only for labelling.

    Returns:
        The output path.
    """
    horizons = list(horizons)
    if not horizons:
        raise ValueError("No horizons to plot.")
    df = _plottable(df)

    ncols = 2
    nrows = int(np.ceil(len(horizons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows), squeeze=False)
    for i, horizon in enumerate(horizons):
        ax = axes[i // ncols][i % ncols]
        g = df[df["horizon"] == horizon]
        color = HORIZON_COLORS.get(horizon, "#333333")
        ax.scatter(g["shift_score"], g["mae"], color=color, s=60, alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=3)
        _trend_line(ax, g, color, linewidth=2)
        if len(g) >= 3:
            pr, pp = pearsonr(g["shift_score"], g["mae"])
            sr, _ = spearmanr(g["shift_score"], g["mae"])
            ax.set_title(f"{horizon} min   Pearson r={pr:.2f} (p={pp:.2f}),  Spearman ρ={sr:.2f}")
        else:
            ax.set_title(f"{horizon} min   (n={len(g)})")
        ax.set_xlabel("Distributional shift (Wasserstein train→test, mg/dL)")
        ax.set_ylabel(f"{model_name} MAE (mg/dL)")
        ax.grid(True, alpha=0.25, zorder=0)
    for j in range(len(horizons), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Per-patient distributional shift vs {model_name} MAE, by horizon",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved faceted figure: {out}")
    return str(out)
