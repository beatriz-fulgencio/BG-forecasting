"""
Figures for per-patient distributional shift vs prediction error.
"""

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


def _plottable(df: pd.DataFrame, horizons: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Drop rows that cannot be plotted, and optionally restrict to horizons.

    The analysis table carries a row for every patient at every horizon, with
    NaN where a patient has no MAE, so dropping here keeps every caller from
    having to remember to.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"shift/MAE table is missing column(s): {', '.join(missing)}")
    # Plotting and SciPy correlations both require finite numeric input.  Coerce
    # rather than relying on Matplotlib's later, less actionable failures.
    df = df.copy()
    for column in _REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if horizons is not None:
        df = df[df["horizon"].isin(list(horizons))]
    finite = np.isfinite(df[list(_REQUIRED_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    return df.loc[finite].copy()


def _correlation_is_defined(g: pd.DataFrame) -> bool:
    """Pearson/Spearman need enough rows and variation in both variables."""
    return len(g) >= 3 and g["shift_score"].nunique() >= 2 and g["mae"].nunique() >= 2


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

        # Per-horizon Spearman correlation needs enough patients and variation
        # in both variables; SciPy otherwise emits a warning and returns NaN.
        if _correlation_is_defined(g):
            rho, p = spearmanr(g["shift_score"], g["mae"])
            label = f"{horizon} min (ρ={rho:.2f}, unadj. p={p:.3f}, n={len(g)})"
        else:
            label = f"{horizon} min (correlation undefined, n={len(g)})"

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
        if _correlation_is_defined(g):
            pr, pp = pearsonr(g["shift_score"], g["mae"])
            sr, _ = spearmanr(g["shift_score"], g["mae"])
            ax.set_title(
                f"{horizon} min   Pearson r={pr:.2f} (unadj. p={pp:.2f}),  Spearman ρ={sr:.2f}"
            )
        else:
            ax.set_title(f"{horizon} min   correlation undefined (n={len(g)})")
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
