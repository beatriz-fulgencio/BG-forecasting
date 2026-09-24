"""Per-patient error-localization figures.

usage:
    python -m RUN.figures.run_error_localization_figures \\
      --run-dir results/experiments/experiment_ID/transfer/seed_41 \\
      --run-dir results/experiments/experiment_ID/transfer/seed_42 \\
      --run-dir results/experiments/experiment_ID/transfer/seed_43 \\
      --patient 540

The supporting tables are written next to the images, so every number in a
caption has a file behind it:

* ``error_by_iso_band_<stem>.csv``    -- n, MAE and signed bias per band
* ``error_by_condition_<stem>.csv``   -- n, MAE and signed bias for calm and rapid
* ``error_by_level_bin_<stem>.csv``   -- the binned mean the published panel draws
                                 (``--style paper``/``both`` only)
* ``clarke_zone_by_band_<stem>.csv``  -- the zone x band contingency table
* ``figure_metadata_<stem>.json``     -- cell, seeds, source run directories
"""

import argparse
import json
import sys
from pathlib import Path

# Support both ``python -m RUN.figures.run_error_localization_figures`` and
# ``python RUN/figures/run_error_localization_figures.py``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.analysis.experiments.error_localization import (  # noqa: E402
    MIN_INTERPRETABLE_N,
    load_patient_points,
    plot_clarke_localized,
    plot_error_localization,
    plot_error_localization_paper,
)
from benchmark.analysis.experiments.rapid_change import (  # noqa: E402
    rapid_threshold_mg_dl,
)
from benchmark.analysis.results_io import (  # noqa: E402
    ResultsLayoutError,
    load_experiment_results,
)

# ResultsLayoutError is a RuntimeError, so it is named explicitly rather than
# being caught by the ValueError arm -- pointing at a parent experiment instead
# of its <mode>/seed_<n> subruns is the common mistake and must stay actionable.
_EXPECTED_ERRORS = (ValueError, TypeError, OSError, KeyError, ResultsLayoutError)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir", required=True, action="append", type=Path, dest="run_dirs",
        help="Configured mode/seed directory, e.g. .../transfer/seed_42. "
             "Repeat once per training seed of the same model/mode/horizon cell.",
    )
    parser.add_argument("--patient", type=int, action="append", dest="patients",
                        help="Patient ID to localize, e.g. 540. Repeat for several; "
                             "omit and pass --all-patients for the whole cohort.")
    parser.add_argument("--all-patients", action="store_true",
                        help="Draw every patient in the run's cohort. The run is read "
                             "once and the patients are looped inside, which is why "
                             "this is cheaper than one invocation per patient.")
    parser.add_argument("--model", default=None,
                        help="Model name; inferred when the run holds only one.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: error_localization beside the run(s)).")
    parser.add_argument("--clarke-output-dir", type=Path, default=None,
                        help="Send the Clarke grid and its zone x band table here "
                             "instead of alongside the error-localization figure, so every glycemic-zone "
                             "view of the benchmark collects in one place "
                             "(default: --output-dir).")
    parser.add_argument("--rapid-threshold", type=float, default=None,
                        help="Absolute mg/dL reference change across one sampling "
                             "interval that counts as rapid. Default: the step the "
                             "clinical rapid rate (3 mg/dL/min, a CGM double arrow) "
                             "implies at the run's sampling interval -- 15 mg/dL at 5 min.")
    parser.add_argument("--min-interpretable-n", type=int, default=MIN_INTERPRETABLE_N,
                        help="Bands below this count are drawn hatched and flagged "
                             f"rather than dropped (default: {MIN_INTERPRETABLE_N}).")
    parser.add_argument("--style", choices=("benchmark", "paper", "both"), default="benchmark",
                        help="Which cut of the error-localization figure to draw: 'benchmark' adds the "
                             "signed-bias panel, 'paper' is the published four panels, "
                             "'both' writes each (default: benchmark).")
    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif len(args.run_dirs) == 1:
        output_dir = args.run_dirs[0] / "error_localization"
    else:
        output_dir = args.run_dirs[0].parent / "error_localization"
    output_dir.mkdir(parents=True, exist_ok=True)

    clarke_dir = args.clarke_output_dir or output_dir
    clarke_dir.mkdir(parents=True, exist_ok=True)

    try:
        patients = _resolve_patients(args)
    except _EXPECTED_ERRORS as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    if not patients:
        print("✗ no patients selected: pass --patient or --all-patients", file=sys.stderr)
        return 2

    # One patient failing must not sink the rest of the cohort
    failures = []
    drawn = 0
    for patient_id in patients:
        try:
            _draw_one_patient(patient_id, args, output_dir, clarke_dir,
                              report=len(patients) == 1)
            drawn += 1
        except _EXPECTED_ERRORS as exc:
            failures.append(f"patient {patient_id}: {exc}")
            print(f"✗ patient {patient_id}: {exc}", file=sys.stderr)

    if len(patients) > 1:
        print(f"\ndrew {drawn} of {len(patients)} patients into {output_dir}")
    for line in failures:
        print(f"FAILED {line}", file=sys.stderr)
    return 1 if failures else 0


def _resolve_patients(args):
    """The patient IDs to draw, from the flags or from the run's own cohort."""
    if args.all_patients:
        if args.patients:
            raise ValueError("pass --patient or --all-patients, not both")
        results = load_experiment_results(args.run_dirs[0], with_series=False)
        return list(results.patients)
    return list(args.patients or ())


def _draw_one_patient(patient_id: int, args, output_dir: Path, clarke_dir: Path,
                      *, report: bool):
    """Draw both figures and write the supporting tables for one patient.
    """
    frame, metadata = load_patient_points(args.run_dirs, patient_id, model=args.model)

    stem = f"patient{metadata['patient_id']}_{metadata['model']}"
    figure4 = clarke_dir / f"clarke_localized_pooled_{stem}.png"
    plain = output_dir / f"error_localization_{stem}.png"
    benchmark_figure = (
        output_dir / f"error_localization_{stem}_benchmark.png"
        if args.style == "both" else plain
    )

    figures = [str(figure4)]
    levels = None
    if args.style in ("benchmark", "both"):
        bands, conditions = plot_error_localization(
            frame, metadata, benchmark_figure,
            threshold_mg_dl=args.rapid_threshold,
            min_interpretable_n=args.min_interpretable_n,
        )
        figures.append(str(benchmark_figure))
    if args.style in ("paper", "both"):
        bands, conditions, levels = plot_error_localization_paper(
            frame, metadata, plain,
            threshold_mg_dl=args.rapid_threshold,
            min_interpretable_n=args.min_interpretable_n,
        )
        figures.append(str(plain))
    contingency = plot_clarke_localized(
        frame, metadata, figure4, threshold_mg_dl=args.rapid_threshold,
    )

    bands.to_csv(output_dir / f"error_by_iso_band_{stem}.csv", index=False)
    conditions.to_csv(output_dir / f"error_by_condition_{stem}.csv", index=False)
    contingency.to_csv(clarke_dir / f"clarke_zone_by_band_{stem}.csv", index=False)
    if levels is not None:
        levels.to_csv(output_dir / f"error_by_level_bin_{stem}.csv", index=False)

    # Record the step actually applied, never the bare CLI value: the default is
    # a rate, and a caption quoting "None" mg/dL helps nobody.
    resolved_threshold = (
        rapid_threshold_mg_dl(metadata["sampling_rate_minutes"])
        if args.rapid_threshold is None else float(args.rapid_threshold)
    )
    metadata = {
        **metadata,
        "rapid_threshold_mg_dl": resolved_threshold,
        "rapid_rate_mg_dl_per_min": resolved_threshold / metadata["sampling_rate_minutes"],
        "min_interpretable_n": args.min_interpretable_n,
        "evidence_level": "descriptive",
        "note": (
            "Single-patient figures. No confidence intervals: cohort uncertainty for "
            "the same quantities is reported by the nested patient x seed bootstraps "
            "in run_error_by_range_analysis.py and run_rapid_change_analysis.py."
        ),
        "figure2_style": args.style,
        "figures": sorted(figures),
    }
    (output_dir / f"figure_metadata_{stem}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if report:
        print(json.dumps(metadata, indent=2))
        print(f"\n{bands.to_string(index=False)}")
        print(f"\n{conditions.to_string(index=False)}")
        print(f"\n{contingency.to_string(index=False)}")
    else:
        print(f"wrote {stem}")


if __name__ == "__main__":
    sys.exit(main())
