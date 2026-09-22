"""The analyses as a property of the dataset, not of a model.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Loading SciPy, statsmodels, and the torch-backed data loader together can
# stall on multi-threaded BLAS builds. These cohort calculations need one thread.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.analysis.dataset.glucose_history import (
    OUTPUT_SUBDIR as GLUCOSE_HISTORY_SUBDIR,
    GlucoseHistoryAnalysis,
)
from benchmark.analysis.dataset.patient_features import (
    compute_patient_features,
    ground_truth_invariance,
)
from benchmark.analysis.dataset.clinical_shift import compute_patient_clinical_shifts
from benchmark.analysis.dataset.signal_table import (
    TABLE_NAME, META_NAME, compute_signal_table, reference_signature,
    write_signal_table,
)
from benchmark.analysis.results_io import load_experiment_results

# Column order for the feature table. compute_patient_features returns a dict
# per patient; pinning the order here keeps the CSV stable across runs instead
# of inheriting Python's insertion order from the analyzer's internals.
FEATURE_COLUMNS = (
    "mean_glucose", "median_glucose", "std_glucose", "min_glucose", "max_glucose",
    "q25_glucose", "q75_glucose", "glucose_range", "iqr_glucose",
    "mean_change", "std_change", "n_contiguous_changes",
    "hypo_percent", "in_range_percent", "hyper_percent",
    "severe_hypo_percent", "severe_hyper_percent",
    "n_rapid_changes", "stability_score", "n_samples",
    "n_target_rows", "target_coverage_percent", "source_series",
    "rapid_change_threshold_mg_dl", "rapid_change_rate_mg_dl_per_min",
)


def population_glucose_history(args, failures: list):
    """Train/test glucose history for the whole cohort, from the raw files."""
    try:
        population = GlucoseHistoryAnalysis(
            str(args.experiment_dir), str(args.output_dir),
            data_root=args.data_root, mode=args.mode, seed=args.seed,
        )
        if not population.run_glucose_history_analysis():
            failures.append("population glucose history: no glucose data available")
            return False
    except Exception as error:          # one analysis must not sink the other
        failures.append(f"population glucose history: {type(error).__name__}: {error}")
        return False

    history = args.output_dir / GLUCOSE_HISTORY_SUBDIR
    print(f"wrote {history / 'population_glucose_overview.png'}")
    print(f"wrote {history / 'population_glucose_summary.csv'}")
    return True


def patient_glucose_features(args, failures: list):
    """Per-patient descriptive features over the held-out ground truth."""
    try:
        results = load_experiment_results(
            args.experiment_dir, mode=args.mode, seed=args.seed,
        )
        features = compute_patient_features(results)
    except Exception as error:
        failures.append(f"patient features: {type(error).__name__}: {error}")
        return False

    if not features:
        failures.append("patient features: no patient had a usable ground-truth series")
        return False

    frame = pd.DataFrame.from_dict(features, orient="index")
    frame.index.name = "patient_id"
    frame = frame.sort_index().reset_index()
    # Any feature the analyzer adds later still reaches the CSV; it just lands
    # after the pinned ones rather than silently disappearing.
    ordered = [column for column in FEATURE_COLUMNS if column in frame.columns]
    extra = [c for c in frame.columns if c not in ordered and c != "patient_id"]
    frame = frame[["patient_id", *ordered, *extra]]

    path = args.output_dir / "patient_glucose_features.csv"
    frame.to_csv(path, index=False)
    print(f"wrote {path}")

    if args.check_invariance:
        report_invariance(results, args.output_dir)
    return True


def clinical_glucose_shift(args, failures: list):
    """Per-patient clinical CGM profile change between train and test splits."""
    try:
        population = GlucoseHistoryAnalysis(
            str(args.experiment_dir), str(args.output_dir),
            data_root=args.data_root, mode=args.mode, seed=args.seed,
        )
        patient_data = population.load_and_preprocess_glucose_data()
        if not patient_data:
            failures.append("clinical glucose shift: no glucose data available")
            return False
        frame = compute_patient_clinical_shifts(patient_data)
    except Exception as error:
        failures.append(f"clinical glucose shift: {type(error).__name__}: {error}")
        return False

    path = args.output_dir / "clinical_glucose_shift.csv"
    frame.to_csv(path, index=False)
    print(f"wrote {path}")
    return True


def dataset_signal_features(args, failures: list):
    """Model-independent train/test shift and irregularity table."""
    try:
        settings = reference_signature(args.experiment_dir, args.mode, args.seed)
        population = GlucoseHistoryAnalysis(
            str(args.experiment_dir), str(args.output_dir),
            data_root=args.data_root, mode=args.mode, seed=args.seed,
        )
        patient_data = population.load_and_preprocess_glucose_data()
        if not patient_data:
            raise ValueError("no glucose data available")
        frame = compute_signal_table(patient_data, args.data_root)
        write_signal_table(frame, args.output_dir, settings, args.data_root)
    except Exception as error:
        failures.append(f"dataset signal features: {type(error).__name__}: {error}")
        return False
    print(f"wrote {args.output_dir / TABLE_NAME}")
    print(f"wrote {args.output_dir / META_NAME}")
    return True


def report_invariance(results, output_dir: Path):
    """Write the ground-truth invariance check and warn if any patient fails it.

    The check itself lives beside the features it substantiates, in
    ``benchmark.analysis.dataset.patient_features``; this only records it.
    """
    frame = ground_truth_invariance(results)
    path = output_dir / "ground_truth_invariance.csv"
    frame.to_csv(path, index=False)
    checked = frame[frame["identical"].notna()]
    disagreeing = checked[~checked["identical"].astype(bool)]
    if len(disagreeing):
        print(f"WARNING: ground truth differs across models for "
              f"{len(disagreeing)} patient(s); see {path}", file=sys.stderr)
    else:
        print(f"ground truth identical across models for {len(checked)} patient(s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", required=True, type=Path,
                        help="Reference run: fixes cohort, releases, sampling rate "
                             "and test split. Contributes no numbers of its own.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", default=None, choices=("regular", "transfer"),
                        help="Required only when the run holds both modes")
    parser.add_argument("--seed", type=int, default=None,
                        help="Read one seed instead of the cross-seed mean. The "
                             "ground truth is the same either way; this only "
                             "changes which export it is read from.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--skip-population", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-clinical-shift", action="store_true")
    parser.add_argument("--skip-signal", action="store_true")
    parser.add_argument("--check-invariance", action="store_true",
                        help="Verify the ground truth agrees across the run's models")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list = []
    ran = []

    if not args.skip_population:
        ran.append(("population_glucose_history", population_glucose_history(args, failures)))
    if not args.skip_features:
        ran.append(("patient_glucose_features", patient_glucose_features(args, failures)))
    if not args.skip_clinical_shift:
        ran.append(("clinical_glucose_shift", clinical_glucose_shift(args, failures)))
    if not args.skip_signal:
        ran.append(("dataset_signal_features", dataset_signal_features(args, failures)))

    if not ran:
        print("Nothing to do: all analyses were skipped.", file=sys.stderr)
        return 2

    (args.output_dir / "summary.json").write_text(json.dumps({
        "analysis": "dataset_only",
        "reference_run": str(args.experiment_dir),
        "mode": args.mode,
        "seed": args.seed,
        "data_root": args.data_root,
        "completed": {name: bool(ok) for name, ok in ran},
        "failures": failures,
    }, indent=2) + "\n")

    for line in failures:
        print(f"FAILED {line}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
