#!/usr/bin/env python3
"""Clarke error grid for one patient, localized.
usage:
    python RUN/figures/run_clarke_localized_figure.py \
        --run-dir results/experiments/<parent>/transfer/seed_41 \
        --patient 540 --model GRU \
        --output-dir results/analysis/glycemic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.analysis.experiments.rapid_change import (
    rapid_change_masks,
    rapid_threshold_mg_dl,
)
from benchmark.analysis.experiments.error_localization import (
    GRID_LOW, GRID_HIGH, ISO_BANDS, ISO_BAND_COLORS, ISO_BAND_LABELS,
    iso_band_masks,
)
from benchmark.analysis.results_io import (
    ResultsLayoutError, load_experiment_results, load_predictions,
)
from benchmark.evaluation.clarke_ega.clarke_ega import ClarkeEGA

# Rapid-change points are a handful among thousands, so they are drawn large,
# black-edged and above the calm layer. Shared with the pooled-seed panel in
# ``benchmark.analysis.experiments.error_localization`` so the two agree.
RAPID_MARKER = "s"

EXPECTED_PATIENT_ERRORS = (ValueError, TypeError, OSError, KeyError, ResultsLayoutError)


def draw_grid(axis, grid):
    """Clarke boundary lines and zone letters over a faint zone fill.
    """
    grid.draw_localized_grid(axis, alpha=0.10)     # faint, so points dominate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="One seed directory, e.g. <parent>/transfer/seed_41")
    parser.add_argument("--patient", type=int, action="append", dest="patients",
                        help="Patient to draw; repeat for several.")
    parser.add_argument("--all-patients", action="store_true",
                        help="Draw every patient in the run's cohort.")
    parser.add_argument("--model", default="GRU")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rapid-threshold", type=float, default=None,
                        help="|delta reference| between adjacent samples, mg/dL. "
                             "Default: the clinical rapid rate (3 mg/dL/min) at the "
                             "run's sampling interval -- 15 mg/dL at 5 min.")
    args = parser.parse_args(argv)

    results = load_experiment_results(args.run_dir, with_series=False)
    if args.all_patients:
        if args.patients:
            print("pass --patient or --all-patients, not both", file=sys.stderr)
            return 2
        patients = list(results.patients)
    else:
        patients = list(args.patients or ())
    if not patients:
        print("no patients selected: pass --patient or --all-patients", file=sys.stderr)
        return 2

    # One patient's missing export must not stop the rest of the cohort.
    failed = []
    for patient_id in patients:
        try:
            status = _draw_one(args, results, patient_id)
        except EXPECTED_PATIENT_ERRORS as error:
            print(f"patient {patient_id}: {type(error).__name__}: {error}", file=sys.stderr)
            status = 1
        if status:
            failed.append(patient_id)
    if failed:
        print(f"failed for patient(s): {', '.join(map(str, failed))}", file=sys.stderr)
        return 1
    return 0


def _draw_one(args, results, patient_id: int):
    frame = load_predictions(args.run_dir, patient_id, args.model, with_context=True)
    if frame is None or "predictions" not in frame:
        print(f"No predictions for patient {patient_id} / {args.model} under {args.run_dir}",
              file=sys.stderr)
        return 2

    sampling_rate = results.sampling_rate_minutes or 5

    reference = frame["true_values"].to_numpy(dtype=float)
    predicted = frame["predictions"].to_numpy(dtype=float)
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(predicted).all():
        raise ValueError("prediction export must contain finite glucose pairs")

    # Match the pooled Clarke figure: the grid is defined over the 40-400 mg/dL
    # sensor domain, so both plotted points and zone scores use that domain.
    grid_reference = np.clip(reference, GRID_LOW, GRID_HIGH)
    grid_predicted = np.clip(predicted, GRID_LOW, GRID_HIGH)
    clipped = int(np.count_nonzero((grid_reference != reference) |
                                   (grid_predicted != predicted)))
    if clipped:
        print(f"note: {clipped} pair(s) clipped to {GRID_LOW:g}-{GRID_HIGH:g} mg/dL "
              "for the Clarke grid", file=sys.stderr)

    # Timestamp-aware, via the same masks the rest of the analysis uses. 
    masks, excluded = rapid_change_masks(
        frame, sampling_rate_minutes=sampling_rate,
        threshold_mg_dl=args.rapid_threshold,
        source=f"patient {patient_id}, {args.model}",
    )
    rapid = masks["rapid"]
    calm = masks["calm"]
    threshold = args.rapid_threshold
    if threshold is None:
        threshold = rapid_threshold_mg_dl(sampling_rate)
    if excluded.any():
        print(f"note: {int(excluded.sum())} point(s) excluded from the calm/rapid "
              f"split (timestamps not exactly {sampling_rate} min apart)", file=sys.stderr)

    band_masks = iso_band_masks(reference)

    grid = ClarkeEGA()
    result = grid.analyze(grid_reference, grid_predicted)
    percentage = result["percentage"]                  # [A, B, C, D, E]
    a_plus_b = percentage[0] + percentage[1]
    cde_count = int(result["total"][2:].sum())
    print(f"Clarke A={percentage[0]:.1f}% B={percentage[1]:.1f}% C={percentage[2]:.2f}% "
          f"D={percentage[3]:.2f}% E={percentage[4]:.2f}%  | A+B={a_plus_b:.1f}%  "
          f"C/D/E n={cde_count} of {len(reference)}")

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.4, 6.3))

    draw_grid(left, grid)
    for band in ISO_BANDS:
        mask = band_masks[band]
        if mask.any():
            name = ISO_BAND_LABELS[band].replace("\n", " ")
            left.scatter(grid_reference[mask], grid_predicted[mask], s=14,
                         c=ISO_BAND_COLORS[band], alpha=0.75, edgecolors="none",
                         zorder=3, label=f"{name} (n={int(mask.sum())})")
    left.set_title("Colored by glycemic zone", fontsize=11, fontweight="bold")
    left.legend(loc="upper left", fontsize=7.5, framealpha=0.92, title="reference range",
                title_fontsize=8, markerscale=1.4)

    draw_grid(right, grid)
    right.scatter(grid_reference[calm], grid_predicted[calm], s=12, c="#9aa4ad", alpha=0.5,
                  edgecolors="none", zorder=3, label=f"calm (n={int(calm.sum())})")
    right.scatter(grid_reference[rapid], grid_predicted[rapid], s=52, c="#e8481c", alpha=0.95,
                  edgecolors="black", linewidth=0.6, marker=RAPID_MARKER, zorder=4,
                  label=f"rapid change |Δ|>{threshold:.0f} (n={int(rapid.sum())})")
    right.set_title("Colored by rapid glucose change", fontsize=11, fontweight="bold")
    right.legend(loc="upper left", fontsize=8, framealpha=0.92, markerscale=1.2)

    # The standard instrument score, stated rather than interpreted.
    readout = (f"Clarke A+B = {a_plus_b:.1f}%  (A {percentage[0]:.1f}%, B {percentage[1]:.1f}%)\n"
               f"C/D/E = {percentage[2] + percentage[3] + percentage[4]:.2f}%  "
               f"(n={cde_count} of {len(reference)})")
    right.text(0.98, 0.03, readout, transform=right.transAxes, ha="right", va="bottom",
               fontsize=8.2, bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                       ec="#555", alpha=0.92))
    figure.text(0.5, 0.01,
                f"Clarke grid values clipped to {GRID_LOW:g}-{GRID_HIGH:g} mg/dL "
                f"for {clipped} of {len(reference)} pair(s).",
                ha="center", fontsize=8, color="#555555")

    try:
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"clarke_localized_patient{patient_id}_{args.model}.png"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    finally:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
