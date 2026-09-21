"""Per-patient error localization: ISO glycemic bands and rapid glucose change.

This module answers *where* a patient's forecast error sits, as opposed to how
large it is on average.  
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...evaluation.clarke_ega import ClarkeEGA
from .rapid_change import (
    CONDITIONS,
    RAPID_RATE_MG_DL_PER_MIN,
    rapid_change_masks,
    rapid_threshold_mg_dl,
)
from ..results_io import load_experiment_results, load_predictions

# The Clarke grid is defined on the measurement domain only, so predictions are
# clipped to the CGM sensor range before they are classified 
GRID_LOW, GRID_HIGH = 40.0, 400.0

# Consensus CGM ranges (ATTD/ISO).  
SEVERE_HYPO_BELOW = 54.0
HYPO_BELOW = 70.0
HYPER_ABOVE = 180.0
SEVERE_HYPER_ABOVE = 250.0

ISO_BANDS: Tuple[str, ...] = (
    "severe_hypoglycemia",
    "hypoglycemia",
    "target_range",
    "hyperglycemia",
    "severe_hyperglycemia",
)

#: Short labels for figure axes.
ISO_BAND_LABELS: Dict[str, str] = {
    "severe_hypoglycemia": "Severe hypo\n<54",
    "hypoglycemia": "Hypo\n54-69",
    "target_range": "Target\n70-180",
    "hyperglycemia": "Hyper\n181-250",
    "severe_hyperglycemia": "Severe hyper\n>250",
}

#: Diverging palette: hypoglycemia cool, target neutral, hyperglycemia warm.
ISO_BAND_COLORS: Dict[str, str] = {
    "severe_hypoglycemia": "#08306b",
    "hypoglycemia": "#4292c6",
    "target_range": "#7f7f7f",
    "hyperglycemia": "#fd8d3c",
    "severe_hyperglycemia": "#cb181d",
}

ZONE_FILL_ALPHA = 0.10
RAPID_MARKER = "s"
# Scatter-only colours. 
CLARKE_CALM_COLOR = "#9aa4ad"
CLARKE_RAPID_COLOR = "#e8481c"

CONDITION_COLORS: Dict[str, str] = {"calm": "#4d4d4d", "rapid": "#d94801"}

#: A band with fewer points than this is drawn, but marked not interpretable.
MIN_INTERPRETABLE_N = 10


def iso_band_masks(reference: np.ndarray):
    """Partition predictions into the five ISO glycemic ranges.

    The band comes from the reference glucose, so the partition is identical for
    every model scored against the same test split.
    Args:
        reference: Reference glucose values (mg/dL).

    Returns:
        Mapping from band name to a boolean mask over ``reference``.  The masks
        are disjoint and cover every finite element.
    """
    reference = np.asarray(reference, dtype=float)
    return {
        "severe_hypoglycemia": reference < SEVERE_HYPO_BELOW,
        "hypoglycemia": (reference >= SEVERE_HYPO_BELOW) & (reference < HYPO_BELOW),
        "target_range": (reference >= HYPO_BELOW) & (reference <= HYPER_ABOVE),
        "hyperglycemia": (reference > HYPER_ABOVE) & (reference <= SEVERE_HYPER_ABOVE),
        "severe_hyperglycemia": reference > SEVERE_HYPER_ABOVE,
    }


def band_error_table(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    min_interpretable_n: int = MIN_INTERPRETABLE_N,
):
    """Per-band count, MAE and signed bias.

    Signed bias is predicted - reference; positive means the model reads
    high.  It is reported alongside MAE because the two carry different clinical
    information: MAE says how far off the forecast is, bias says in which
    direction, and only the direction determines whether an error crosses a
    treatment threshold.

    Args:
        reference: Reference glucose (mg/dL).
        predicted: Raw, unclipped model output (mg/dL).
        min_interpretable_n: Bands below this count are flagged rather than dropped, so a near-empty band stays visible in the figure.

    Returns:
        One row per band, with columns ``band``, ``n``, ``mae``, ``signed_bias`` and ``interpretable``.  An empty band reports ``NaN`` statistics rather than zero, which would read as a perfect forecast.
    """
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if reference.shape != predicted.shape:
        raise ValueError(
            f"reference/prediction shape mismatch: {reference.shape} vs {predicted.shape}"
        )
    if reference.size == 0:
        raise ValueError("At least one prediction is required")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Reference and prediction values must be finite")

    error = predicted - reference
    masks = iso_band_masks(reference)
    rows: List[Dict[str, Any]] = []
    for band in ISO_BANDS:
        mask = masks[band]
        count = int(np.count_nonzero(mask))
        rows.append({
            "band": band,
            "n": count,
            "mae": float(np.abs(error[mask]).mean()) if count else float("nan"),
            "signed_bias": float(error[mask].mean()) if count else float("nan"),
            "interpretable": count >= min_interpretable_n,
        })
    table = pd.DataFrame(rows)
    if int(table["n"].sum()) != reference.size:
        raise ValueError("ISO bands do not partition the predictions")
    return table


def condition_error_table(
    frame: pd.DataFrame,
    *,
    sampling_rate_minutes: int,
    threshold_mg_dl: Optional[float] = None,
    source: str = "predictions",
):
    """Per-condition count and MAE for calm versus rapid glucose change.

    The masks come from :func:`benchmark.analysis.experiments.rapid_change.rapid_change_masks`

    Args:
        frame: Prediction frame carrying the prediction-context columns
        sampling_rate_minutes: CGM sampling interval.
        threshold_mg_dl: Absolute reference change that counts as rapid.
        source: Label used in error messages.

    Returns:
        One row per condition with columns ``condition``, ``n``, ``mae``,
        ``signed_bias``, plus an ``excluded`` count repeated on each row.
    """
    # ``None`` means the clinical rapid rate at this run's sampling interval; resolve it here because the panel labels print the step.
    if threshold_mg_dl is None:
        threshold_mg_dl = rapid_threshold_mg_dl(sampling_rate_minutes)

    masks, excluded = rapid_change_masks(
        frame,
        sampling_rate_minutes=sampling_rate_minutes,
        threshold_mg_dl=threshold_mg_dl,
        source=source,
    )
    reference = pd.to_numeric(frame["true_values"], errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(frame["predictions"], errors="coerce").to_numpy(float)
    error = predicted - reference
    rows = []
    for condition in CONDITIONS:
        mask = masks[condition]
        count = int(np.count_nonzero(mask))
        rows.append({
            "condition": condition,
            "n": count,
            "mae": float(np.abs(error[mask]).mean()) if count else float("nan"),
            "signed_bias": float(error[mask].mean()) if count else float("nan"),
            "excluded": int(np.count_nonzero(excluded)),
        })
    return pd.DataFrame(rows)


def load_patient_points(
    run_dirs: Sequence[Path],
    patient_id: int,
    *,
    model: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load one patient's predictions across the training seeds of one cell.
    Args:
        run_dirs: One configured ``<mode>/seed_<n>`` directory per training seed.
        patient_id: The patient to localize.
        model: Model name; inferred from the run when omitted.

    Returns:
        ``(frame, metadata)``.  ``frame`` carries the prediction-context columns
        plus a ``training_seed`` column.  
    """
    if not run_dirs:
        raise ValueError("At least one configured single-seed run directory is required")

    frames: List[pd.DataFrame] = []
    metadata: Dict[str, Any] = {}
    seeds: List[int] = []
    for run_dir_value in run_dirs:
        run_dir = Path(run_dir_value)
        results = load_experiment_results(run_dir, with_series=False)
        if results.aggregation != "single_seed" or len(results.seeds) != 1:
            raise ValueError(
                f"{run_dir}: must be a configured single-seed run "
                "(pass each <mode>/seed_<n> directory, not the parent experiment)"
            )
        if results.sampling_rate_minutes is None:
            raise ValueError(f"{run_dir}: sampling rate is unknown")
        if results.horizon_minutes is None:
            raise ValueError(f"{run_dir}: horizon is unknown")
        if patient_id not in results.patients:
            raise ValueError(
                f"{run_dir}: patient {patient_id} is not in this run's cohort "
                f"({', '.join(str(p) for p in results.patients)})"
            )
        available = tuple(results.patient_metrics[patient_id])
        resolved_model = model or (available[0] if len(available) == 1 else None)
        if resolved_model is None:
            raise ValueError(
                f"{run_dir}: run holds several models ({', '.join(available)}); pass --model"
            )
        if resolved_model not in available:
            raise ValueError(
                f"{run_dir}: no predictions for model {resolved_model}; have {', '.join(available)}"
            )

        cell = {
            "model": resolved_model,
            "mode": results.mode,
            "horizon_minutes": results.horizon_minutes,
            "sampling_rate_minutes": results.sampling_rate_minutes,
        }
        if metadata and {k: metadata[k] for k in cell} != cell:
            raise ValueError(
                "All run directories must belong to one model/mode/horizon cell; "
                f"got {metadata} then {cell}"
            )
        metadata.update(cell)

        frame = load_predictions(run_dir, patient_id, resolved_model, with_context=True)
        if frame is None:
            raise ValueError(
                f"{run_dir}: missing predictions for patient {patient_id}, {resolved_model}; "
                "set output.save_predictions: true for the publication run"
            )
        seed = int(results.seeds[0])
        if seed in seeds:
            raise ValueError(f"Training seeds must be distinct; {seed} appears twice")
        seeds.append(seed)
        frame = frame.copy()
        frame["training_seed"] = seed
        frames.append(frame)

    pooled = pd.concat(frames, ignore_index=True)
    metadata.update({
        "patient_id": int(patient_id),
        "training_seeds": seeds,
        "n_predictions": int(len(pooled)),
        "run_dirs": [str(Path(d)) for d in run_dirs],
    })
    return pooled, metadata


def _cell_caption(metadata: Dict[str, Any]) -> str:
    seeds = metadata["training_seeds"]
    seed_text = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {', '.join(map(str, seeds))} pooled"
    return (
        f"patient {metadata['patient_id']} | {metadata['model']} | "
        f"{metadata['horizon_minutes']:g}-min | {metadata['mode']} learning | {seed_text}"
    )


# Figure: error localization
def plot_error_localization(
    frame: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: Path,
    *,
    threshold_mg_dl: Optional[float] = None,
    min_interpretable_n: int = MIN_INTERPRETABLE_N,
    bins: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the four-panel error-localization figure for one patient.

    Panels, clockwise from top left:

    1. MAE by ISO glycemic band, each bar labelled with its point count.  Bands
       below ``min_interpretable_n`` are hatched and greyed rather than dropped.
    2. Signed bias by band, with a zero line.  This is the panel that shows the
       asymmetry MAE cannot: hypoglycemic errors point one way and
       hyperglycemic errors the other.
    3. Calm versus rapid mean absolute error.
    4. Mean absolute error against reference glucose, binned, with the band
       edges marked.  This is the continuous form of panel 1 and shows that the
       band effect is not an artefact of where the edges were drawn.
    """
    if threshold_mg_dl is None:
        threshold_mg_dl = rapid_threshold_mg_dl(metadata["sampling_rate_minutes"])

    import matplotlib.pyplot as plt

    reference = pd.to_numeric(frame["true_values"], errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(frame["predictions"], errors="coerce").to_numpy(float)
    bands = band_error_table(reference, predicted, min_interpretable_n=min_interpretable_n)
    conditions = condition_error_table(
        frame,
        sampling_rate_minutes=metadata["sampling_rate_minutes"],
        threshold_mg_dl=threshold_mg_dl,
        source=f"patient {metadata['patient_id']}",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    positions = np.arange(len(ISO_BANDS))
    labels = [ISO_BAND_LABELS[band] for band in ISO_BANDS]
    colors = [ISO_BAND_COLORS[band] for band in ISO_BANDS]

    def _style_band_axis(axis, values: np.ndarray) -> None:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_xlabel("Reference glucose range (mg/dL)", fontsize=9)
        axis.grid(axis="y", alpha=0.25)
        # Headroom for the per-bar count annotations, which otherwise collide
        # with the axis frame on whichever band happens to be tallest.
        _add_headroom(axis, values)

    def _add_headroom(axis, values: np.ndarray, fraction: float = 0.18) -> None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        span = float(np.abs(finite).max()) or 1.0
        low, high = axis.get_ylim()
        axis.set_ylim(
            min(low, finite.min() - fraction * span) if finite.min() < 0 else low,
            max(high, finite.max() + fraction * span),
        )

    def _draw_bars(axis, values: np.ndarray) -> None:
        """Bars, with non-interpretable bands hatched and their count called out."""
        for position, value, color, (_, row) in zip(positions, values, colors, bands.iterrows()):
            if not np.isfinite(value):
                continue
            interpretable = bool(row["interpretable"])
            axis.bar(
                position, value,
                color=color if interpretable else "#d9d9d9",
                edgecolor="black", linewidth=0.6,
                hatch=None if interpretable else "///",
            )
            offset = np.sign(value) * 0.02 * max(1.0, float(np.nanmax(np.abs(values))))
            axis.annotate(
                f"n={int(row['n'])}" + ("" if interpretable else "\n(too few)"),
                (position, value + offset),
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.5,
            )

    # Panel 1: magnitude by band.
    ax = axes[0, 0]
    _draw_bars(ax, bands["mae"].to_numpy(float))
    ax.set_ylabel("MAE (mg/dL)", fontsize=9)
    ax.set_title("Error magnitude by glycemic range", fontsize=10, fontweight="bold")
    _style_band_axis(ax, bands["mae"].to_numpy(float))

    # Panel 2: direction by band -- the part MAE averages away.
    ax = axes[0, 1]
    _draw_bars(ax, bands["signed_bias"].to_numpy(float))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Signed bias, predicted - reference (mg/dL)", fontsize=9)
    ax.set_title("Error direction by glycemic range", fontsize=10, fontweight="bold")
    _style_band_axis(ax, bands["signed_bias"].to_numpy(float))

    # Panel 3: rate of change.
    ax = axes[1, 0]
    condition_positions = np.arange(len(CONDITIONS))
    values = conditions.set_index("condition").loc[list(CONDITIONS), "mae"].to_numpy(float)
    counts = conditions.set_index("condition").loc[list(CONDITIONS), "n"].to_numpy(int)
    ax.bar(
        condition_positions, values,
        color=[CONDITION_COLORS[c] for c in CONDITIONS],
        edgecolor="black", linewidth=0.6,
    )
    for position, value, count in zip(condition_positions, values, counts):
        if np.isfinite(value):
            ax.annotate(f"{value:.1f}\nn={count}", (position, value), ha="center",
                        va="bottom", fontsize=8)
    ax.set_xticks(condition_positions)
    ax.set_xticklabels([
        f"Calm\n|delta| <= {threshold_mg_dl:g} mg/dL",
        f"Rapid\n|delta| > {threshold_mg_dl:g} mg/dL",
    ], fontsize=8)
    ax.set_ylabel("MAE (mg/dL)", fontsize=9)
    ax.set_title("Error during rapid glucose change", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    _add_headroom(ax, values)
    excluded = int(conditions["excluded"].iloc[0])
    if excluded:
        ax.annotate(
            f"{excluded} rows excluded (timestamps not one interval apart)",
            (0.5, -0.28), xycoords="axes fraction", ha="center", fontsize=7, color="#555555",
        )

    # Panel 4: the continuous form of panel 1.
    ax = axes[1, 1]
    absolute_error = np.abs(predicted - reference)
    edges = np.linspace(reference.min(), reference.max(), bins + 1)
    index = np.clip(np.digitize(reference, edges) - 1, 0, bins - 1)
    centers, means = [], []
    for bin_index in range(bins):
        mask = index == bin_index
        if np.count_nonzero(mask) >= min_interpretable_n:
            centers.append(0.5 * (edges[bin_index] + edges[bin_index + 1]))
            means.append(float(absolute_error[mask].mean()))
    ax.scatter(reference, absolute_error, s=4, alpha=0.12, color="#4d4d4d", linewidths=0)
    if centers:
        ax.plot(centers, means, color="#cb181d", linewidth=2, marker="o", markersize=4,
                label=f"binned mean (bins with n>={min_interpretable_n})")
        ax.legend(fontsize=7.5, loc="upper left")
    for edge in (SEVERE_HYPO_BELOW, HYPO_BELOW, HYPER_ABOVE, SEVERE_HYPER_ABOVE):
        if reference.min() <= edge <= reference.max():
            ax.axvline(edge, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Reference glucose (mg/dL)", fontsize=9)
    ax.set_ylabel("|error| (mg/dL)", fontsize=9)
    ax.set_title("Error against reference glucose", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)

    figure.suptitle(
        f"Error localization -- {_cell_caption(metadata)}",
        fontsize=12, fontweight="bold",
    )
    figure.text(
        0.5, 0.005,
        "Single patient, descriptive: no confidence intervals. Cohort uncertainty is in the "
        "nested patient x seed bootstraps (error-by-range, rapid-change).",
        ha="center", fontsize=7.5, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return bands, conditions


#: Zone colours of the figure
PAPER_BAND_COLORS: Dict[str, str] = {
    "severe_hypoglycemia": "#7b0000",
    "hypoglycemia": "#d62728",
    "target_range": "#2ca02c",
    "hyperglycemia": "#ff7f0e",
    "severe_hyperglycemia": "#8c564b",
}

#: Short band names
PAPER_BAND_LABELS: Dict[str, str] = {
    "severe_hypoglycemia": "severe_hypo",
    "hypoglycemia": "hypo",
    "target_range": "in_range",
    "hyperglycemia": "hyper",
    "severe_hyperglycemia": "severe_hyper",
}

#: Percentile above which an error counts as a spike in the temporal panel.
SPIKE_PERCENTILE = 90.0

#: Width of the fixed glucose bins in the error-versus-level panel (mg/dL).
LEVEL_BIN_WIDTH = 20.0


def level_bin_table(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    width: float = LEVEL_BIN_WIDTH,
    min_interpretable_n: int = MIN_INTERPRETABLE_N,
):
    """Mean absolute error in fixed-width reference-glucose bins.

    Args:
        reference: Reference glucose (mg/dL).
        predicted: Raw, unclipped model output (mg/dL).
        width: Bin width in mg/dL.
        min_interpretable_n: Bins below this count are dropped.

    Returns:
        Columns ``bin_low``, ``bin_high``, ``center``, ``n``, ``mae``; empty if
        no bin clears the threshold.
    """
    reference = np.asarray(reference, dtype=float)
    absolute_error = np.abs(np.asarray(predicted, dtype=float) - reference)
    if width <= 0:
        raise ValueError("width must be positive")
    low = np.floor(reference.min() / width) * width
    high = np.ceil(reference.max() / width) * width
    edges = np.arange(low, high + width, width)

    if len(edges) < 2 or edges[-1] <= reference.max():
        edges = np.append(edges, edges[-1] + width)
    index = np.digitize(reference, edges)
    rows = []
    for position in range(1, len(edges)):
        mask = index == position
        count = int(np.count_nonzero(mask))
        if count >= min_interpretable_n:
            rows.append({
                "bin_low": float(edges[position - 1]),
                "bin_high": float(edges[position]),
                "center": float(0.5 * (edges[position - 1] + edges[position])),
                "n": count,
                "mae": float(absolute_error[mask].mean()),
            })
    return pd.DataFrame(rows, columns=["bin_low", "bin_high", "center", "n", "mae"])


def plot_error_localization_paper(
    frame: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: Path,
    *,
    threshold_mg_dl: Optional[float] = None,
    min_interpretable_n: int = MIN_INTERPRETABLE_N,
    spike_percentile: float = SPIKE_PERCENTILE,
    bin_width: float = LEVEL_BIN_WIDTH,
):
    """Draw figure for error localization.

    Four panels, reading clockwise from top left:

    1. Mean absolute error by glycemic zone, each bar labelled with its value
       and its point count.
    2. Absolute error against reference glucose, with the fixed-width binned
       mean over it and the target range shaded.
    3. Absolute error through the test period, as a rolling mean over the raw
       trace, with the errors at or above ``spike_percentile`` marked.
    4. Calm versus rapid mean absolute error, with their ratio in the title.

    Returns:
        ``(band_table, condition_table, level_bin_table)`` -- every number the
        figure draws, for export beside the image.
    """
    if threshold_mg_dl is None:
        threshold_mg_dl = rapid_threshold_mg_dl(metadata["sampling_rate_minutes"])

    import matplotlib.pyplot as plt

    reference = pd.to_numeric(frame["true_values"], errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(frame["predictions"], errors="coerce").to_numpy(float)
    absolute_error = np.abs(predicted - reference)

    bands = band_error_table(reference, predicted, min_interpretable_n=min_interpretable_n)
    conditions = condition_error_table(
        frame,
        sampling_rate_minutes=metadata["sampling_rate_minutes"],
        threshold_mg_dl=threshold_mg_dl,
        source=f"patient {metadata['patient_id']}",
    )
    levels = level_bin_table(
        reference, predicted, width=bin_width, min_interpretable_n=min_interpretable_n
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: magnitude by zone.
    axis = axes[0, 0]
    drawn = bands[np.isfinite(bands["mae"].to_numpy(float))]
    bars = axis.bar(
        [PAPER_BAND_LABELS[band] for band in drawn["band"]],
        drawn["mae"].to_numpy(float),
        color=[PAPER_BAND_COLORS[band] for band in drawn["band"]],
    )
    for bar, (_, row) in zip(bars, drawn.iterrows()):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{row['mae']:.1f}\n(n={int(row['n'])})",
                  ha="center", va="bottom", fontsize=8)
    axis.set_ylabel("Mean |error| (mg/dL)")
    axis.set_title("Error by glucose zone")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.3)

    # Panel 2: the continuous form of panel 1.
    axis = axes[0, 1]
    axis.scatter(reference, absolute_error, s=4, alpha=0.15, color="#1f77b4")
    if not levels.empty:
        axis.plot(levels["center"], levels["mae"], "o-", color="#d62728", lw=2,
                  label="binned mean |error|")
        axis.legend(fontsize=9)
    axis.axvspan(HYPO_BELOW, HYPER_ABOVE, color="green", alpha=0.05)
    axis.set_xlabel("True glucose (mg/dL)")
    axis.set_ylabel("|error| (mg/dL)")
    axis.set_title("Error vs glucose level")
    axis.grid(alpha=0.3)

    # Panel 3: where in the test period the error sits.
    axis = axes[1, 0]
    count = len(absolute_error)
    position = np.arange(count) / count
    window = max(1, count // 100)
    rolling = pd.Series(absolute_error).rolling(window, min_periods=1, center=True).mean()
    spike = absolute_error >= float(np.percentile(absolute_error, spike_percentile))
    axis.plot(position, absolute_error, color="#cccccc", lw=0.4, alpha=0.6, zorder=1)
    axis.plot(position, rolling.to_numpy(), color="#1f77b4", lw=1.5, zorder=2,
              label=f"rolling mean (win={window})")
    axis.scatter(position[spike], absolute_error[spike], s=8, color="#d62728", zorder=3,
                 label=f"spikes (>={spike_percentile:g}th pct)")
    axis.set_xlabel("Fraction through test period")
    axis.set_ylabel("|error| (mg/dL)")
    axis.set_title("Error over time")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    # Panel 4: rate of change.
    axis = axes[1, 1]
    by_condition = conditions.set_index("condition")
    values = [float(by_condition.loc[condition, "mae"]) for condition in CONDITIONS]
    axis.bar(
        [f"calm\n(|\u0394|<={threshold_mg_dl:g})", f"rapid\n(|\u0394|>{threshold_mg_dl:g})"],
        values, color=["#2ca02c", "#d62728"],
    )
    for offset, value in enumerate(values):
        if np.isfinite(value):
            axis.text(offset, value, f"{value:.1f}", ha="center", va="bottom", fontsize=10)
    calm, rapid = values
    ratio = rapid / calm if np.isfinite(calm) and calm > 0 and np.isfinite(rapid) else float("nan")
    axis.set_ylabel("Mean |error| (mg/dL)")
    axis.set_title(
        f"Error during rapid glucose change (ratio={ratio:.2f}\u00d7)"
        if np.isfinite(ratio) else "Error during rapid glucose change"
    )
    axis.grid(axis="y", alpha=0.3)

    figure.suptitle(
        f"Error localization -- {_cell_caption(metadata)}, "
        f"overall MAE={absolute_error.mean():.2f} mg/dL",
        fontsize=14, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return bands, conditions, levels


# Figure: Clarke grid, localized
def plot_clarke_localized(
    frame: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: Path,
    *,
    threshold_mg_dl: Optional[float] = None,
    point_size: float = 9.0,
) -> pd.DataFrame:
    """Draw the Clarke grid twice, coloured by glycemic band and by rate of change.
    
    Predictions are clipped to the sensor range for this figure only, matching
    the zone-D analysis; the point-error panels of  :func:`plot_error_localization` keep raw output.

    Returns:
        A zone x band contingency table (counts), which is the quantity behind
        the "large hyperglycemic errors stay in zone B" claim.
    """
    if threshold_mg_dl is None:
        threshold_mg_dl = rapid_threshold_mg_dl(metadata["sampling_rate_minutes"])

    import matplotlib.pyplot as plt

    reference = pd.to_numeric(frame["true_values"], errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(frame["predictions"], errors="coerce").to_numpy(float)
    grid_reference = np.clip(reference, GRID_LOW, GRID_HIGH)
    grid_predicted = np.clip(predicted, GRID_LOW, GRID_HIGH)

    analyzer = ClarkeEGA()
    clarke = analyzer.analyze(grid_reference, grid_predicted)
    zones = clarke["zones"]
    band_masks = iso_band_masks(reference)
    condition_masks, excluded = rapid_change_masks(
        frame,
        sampling_rate_minutes=metadata["sampling_rate_minutes"],
        threshold_mg_dl=threshold_mg_dl,
        source=f"patient {metadata['patient_id']}",
    )

    zone_map = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    contingency = pd.DataFrame(
        [
            {
                "band": band,
                **{
                    f"zone_{letter}": int(np.count_nonzero(band_masks[band] & (zones == number)))
                    for number, letter in zone_map.items()
                },
                "n": int(np.count_nonzero(band_masks[band])),
            }
            for band in ISO_BANDS
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 7.6))

    for axis in axes:
        analyzer.draw_localized_grid(axis, alpha=ZONE_FILL_ALPHA)

    ax = axes[0]
    for band in ISO_BANDS:
        mask = band_masks[band]
        if not np.any(mask):
            continue
        ax.scatter(
            grid_reference[mask], grid_predicted[mask],
            s=point_size, c=ISO_BAND_COLORS[band], alpha=0.65, linewidths=0,
            label=f"{ISO_BAND_LABELS[band].replace(chr(10), ' ')} (n={int(mask.sum())})",
        )
    ax.set_title("Coloured by glycemic range", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    zone_d_total = int(contingency["zone_D"].sum())
    if zone_d_total:
        hypo_zone_d = int(
            contingency.loc[
                contingency["band"].isin(("severe_hypoglycemia", "hypoglycemia")), "zone_D"
            ].sum()
        )
        severe_hyper = contingency.loc[
            contingency["band"] == "severe_hyperglycemia"
        ].iloc[0]
        same_side = int(severe_hyper["zone_A"] + severe_hyper["zone_B"])
        ax.annotate(
            f"zone D: {zone_d_total} points, {hypo_zone_d} ({100 * hypo_zone_d / zone_d_total:.1f}%) "
            f"hypoglycemia predicted normal\n"
            f"severe hyperglycemia: {same_side} of {int(severe_hyper['n'])} "
            f"({100 * same_side / max(int(severe_hyper['n']), 1):.1f}%) in zone A or B, "
            f"{int(severe_hyper['zone_D'])} in zone D",
            (0.5, -0.135), xycoords="axes fraction", ha="center", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92,
                      edgecolor="black", linewidth=0.8),
        )

    ax = axes[1]
    # Rapid points are a few dozen among several thousand calm ones, so they are
    # drawn large, black-edged and on a higher layer; at a shared size and style
    # they simply disappear, which is what made this panel unreadable.
    calm_style = dict(s=12, c=CLARKE_CALM_COLOR, alpha=0.5,
                      linewidths=0, zorder=3)
    rapid_style = dict(s=52, c=CLARKE_RAPID_COLOR, alpha=0.95,
                       marker=RAPID_MARKER, edgecolors="black", linewidth=0.6,
                       zorder=4)
    for condition in CONDITIONS:
        mask = condition_masks[condition]
        if not np.any(mask):
            continue
        ax.scatter(
            grid_reference[mask], grid_predicted[mask],
            **(calm_style if condition == "calm" else rapid_style),
            label=f"{condition} (n={int(mask.sum())})",
        )
    ax.set_title(
        f"Coloured by rate of change (rapid: |delta| > {threshold_mg_dl:g} mg/dL)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    # The standard Clarke score, stated rather than interpreted. Without it the
    # figure shows where the errors are but not whether the model "passes".
    percentage = clarke["percentage"]
    a_plus_b = percentage[0] + percentage[1]
    cde_count = int(clarke["total"][2:].sum())
    ax.text(
        0.98, 0.03,
        f"Clarke A+B = {a_plus_b:.1f}%  (A {percentage[0]:.1f}%, B {percentage[1]:.1f}%)\n"
        f"C/D/E = {percentage[2] + percentage[3] + percentage[4]:.2f}%  "
        f"(n={cde_count} of {len(grid_reference)})",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#555", alpha=0.92),
    )
    if np.any(excluded):
        ax.annotate(
            f"{int(excluded.sum())} points excluded from the calm/rapid split "
            "(timestamps not one interval apart)",
            (0.5, -0.135), xycoords="axes fraction", ha="center", va="top",
            fontsize=8.5, color="#555555",
        )

    figure.suptitle(
        f"Clarke error grid, localized -- {_cell_caption(metadata)}",
        fontsize=12, fontweight="bold",
    )
    figure.text(
        0.5, 0.015,
        "One patient, a worst-case view rather than a cohort estimate; "
        "predictions clipped to the 40-400 mg/dL sensor range for the grid only.",
        ha="center", fontsize=8, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.95))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return contingency
