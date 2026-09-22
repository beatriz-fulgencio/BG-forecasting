"""Cross-horizon statistical comparison of completed benchmark runs.

For a given model and training mode, does forecast quality differ significantly between prediction horizons, and by how much?

Usage:
Every recorded run, regular mode, one family per model::

    python RUN/experiments/run_comparison_analysis.py \\
        --parents-file results/full_run_logs/parents.txt \\
        --mode regular \\
        --output-dir results/analysis/comparison_regular

Specific directories, transfer mode::

    python RUN/experiments/run_comparison_analysis.py \\
        --experiments results/experiments/experiment_A results/experiments/experiment_B \\
        --mode transfer --output-dir results/analysis/comparison_transfer
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.analysis.results_io import (
    ExperimentResults,
    ResultsLayoutError,
    load_experiment_results,
)
from benchmark.analysis.experiments.statistical_significance import StatisticalSignificanceTester

# Metrics reported by default
DEFAULT_METRICS = ("mae", "rmse", "clarke_a_b", "tir.time_in_range")


def read_parents_file(path: Path):
    
    directories: List[Path] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{number}: expected '<model> <minutes> <directory>', got {line!r}")
        directories.append(Path(fields[2]))
    if not directories:
        raise ValueError(f"{path}: no runs recorded")
    return directories


def load_all(
    experiment_dirs: Sequence[Path], mode: Optional[str], seed: Optional[int]
):
    """Load every experiment, reporting the ones that cannot be read.
    """
    loaded: List[ExperimentResults] = []
    skipped: List[Dict[str, str]] = []
    for directory in experiment_dirs:
        try:
            results = load_experiment_results(directory, mode=mode, seed=seed, with_series=False)
        except (ResultsLayoutError, ValueError, OSError) as error:
            print(f"  SKIP {directory}: {error}", file=sys.stderr)
            skipped.append({"directory": str(directory), "reason": str(error)})
            continue
        if results.horizon_minutes is None:
            print(f"  SKIP {directory}: prediction horizon unknown", file=sys.stderr)
            skipped.append({"directory": str(directory), "reason": "prediction horizon unknown"})
            continue
        loaded.append(results)
        print(
            f"  {results.label}: {results.horizon_minutes} min, "
            f"{len(results.patients)} patients, {', '.join(results.models)}, "
            f"{results.aggregation}"
            + (f" over seeds {', '.join(map(str, results.seeds))}" if results.seeds else "")
        )
    return loaded, skipped


def group_by_model(experiments: Sequence[ExperimentResults]):
    """Split loaded runs into one horizon family per model."""
    families: Dict[str, List[ExperimentResults]] = defaultdict(list)
    for results in experiments:
        for model in results.models:
            families[model].append(results)
    return dict(families)


def check_one_run_per_horizon(model: str, experiments: Sequence[ExperimentResults]):
    """Reject a family holding two runs at the same horizon.
    """
    seen: Dict[int, List[str]] = defaultdict(list)
    for results in experiments:
        seen[results.horizon_minutes].append(str(results.experiment_dir))
    duplicated = {horizon: dirs for horizon, dirs in seen.items() if len(dirs) > 1}
    if duplicated:
        detail = "; ".join(
            f"{horizon} min: {', '.join(sorted(dirs))}" for horizon, dirs in sorted(duplicated.items())
        )
        raise ValueError(
            f"{model} has more than one run at the same horizon ({detail}). "
            f"Pass exactly one directory per horizon."
        )


def analyse_family(
    model: str,
    experiments: Sequence[ExperimentResults],
    metrics: Sequence[str],
    tester: StatisticalSignificanceTester,
    comparison_type: str,
    correction: Optional[str],
    test_type: str,
    output_dir: Path,
):
    """Run, save and summarise one model's cross-horizon comparison."""
    check_one_run_per_horizon(model, experiments)

    horizons = sorted(results.horizon_minutes for results in experiments)
    print(f"\n=== {model}: {len(experiments)} horizons ({', '.join(f'{h}' for h in horizons)} min) ===")

    payloads = [
        {
            "name": results.label,
            "path": str(results.experiment_dir),
            "mode": results.mode,
            "seeds": results.seeds,
            "aggregation": results.aggregation,
            "horizon_minutes": results.horizon_minutes,
            "horizon_steps": results.horizon_steps,
            "results": results,
        }
        for results in experiments
    ]

    frame = tester.compare_experiments(
        payloads,
        metrics=list(metrics),
        models=[model],
        comparison_type=comparison_type,
        correction_method=correction,
        test_type=test_type,
    )
    if frame.empty:
        raise ValueError(f"no comparable metrics for {model}: {', '.join(metrics)}")

    expected = {
        (first, second, metric)
        for first, second in combinations(horizons, 2)
        for metric in metrics
    }
    observed = {
        (min(row.horizon_1_min, row.horizon_2_min),
         max(row.horizon_1_min, row.horizon_2_min), row.metric)
        for row in frame.itertuples(index=False)
    }
    missing = sorted(expected - observed)
    if missing:
        detail = ", ".join(f"{first}/{second} min {metric}" for first, second, metric in missing)
        raise ValueError(f"missing requested comparisons for {model}: {detail}")

    model_dir = output_dir / model.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(model_dir / "horizon_comparison.csv", index=False)
    tester.create_summary_report(frame, output_path=model_dir / "horizon_comparison.txt")

    metrics_table = pd.concat(
        [table.loc[table["model"] == model]
         for results in experiments for table in [results.to_frame()]],
        ignore_index=True,
    )
    metrics_table.to_csv(model_dir / "metrics_by_patient.csv", index=False)

    significance_column = (
        "significant_corrected" if "significant_corrected" in frame.columns else "significant"
    )
    print(
        f"  {int(frame[significance_column].sum())}/{len(frame)} comparisons significant "
        f"at alpha={tester.alpha}"
    )
    print(f"  written to {model_dir}")
    return frame


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiments", nargs="+", type=Path,
                        help="experiment directories, one per horizon per model")
    source.add_argument("--parents-file", type=Path,
                        help="parents.txt written by RUN/run_training.sh")
    parser.add_argument("--mode",
                        help="training mode to analyse (regular/transfer); "
                             "required when a run holds more than one")
    parser.add_argument("--seed", type=int,
                        help="analyse this seed alone instead of the cross-seed mean")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS),
                        help="metric names; nested metrics use dots (tir.time_in_range)")
    parser.add_argument("--comparison-type", choices=("paired", "independent"), default="paired",
                        help="paired is correct when horizons share a patient cohort")
    parser.add_argument(
        "--correction", choices=("holm", "bonferroni", "fdr_bh", "bh", "none"),
        default="holm",
        help="multiplicity adjustment across all horizon pairs; fdr_bh/bh controls FDR",
    )
    parser.add_argument(
        "--test-family", choices=("auto", "parametric", "non-parametric", "sign"),
        default="auto",
        help="precommit to paired t, Wilcoxon, or exact sign inference; "
             "'auto' retains legacy normality-based selection",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not 0 < args.alpha < 1:
        parser.error("--alpha must lie strictly between 0 and 1")

    experiment_dirs = (
        read_parents_file(args.parents_file) if args.parents_file else list(args.experiments)
    )

    print(f"Loading {len(experiment_dirs)} experiment(s)"
          + (f" (mode={args.mode})" if args.mode else "")
          + (f" (seed={args.seed})" if args.seed is not None else ""))
    experiments, skipped_experiments = load_all(experiment_dirs, args.mode, args.seed)
    if not experiments:
        print("No readable experiments.", file=sys.stderr)
        return 1

    # ``test_family`` is passed to ``compare_experiments`` below, so one shared
    # tester covers parametric, Wilcoxon, and sign-test analysis plans.
    tester = StatisticalSignificanceTester(alpha=args.alpha)
    correction = None if args.correction == "none" else args.correction

    args.output_dir.mkdir(parents=True, exist_ok=True)
    families = group_by_model(experiments)

    summary: Dict[str, object] = {
        "mode": args.mode,
        "seed": args.seed,
        "metrics": list(args.metrics),
        "comparison_type": args.comparison_type,
        "correction": args.correction,
        "test_family": args.test_family,
        "alpha": args.alpha,
        "models": {},
        "skipped_experiments": skipped_experiments,
        "skipped_single_horizon_models": {},
        "failed_models": {},
    }
    failures: List[str] = []

    for model, family in sorted(families.items()):
        if len(family) < 2:
            horizon = family[0].horizon_minutes
            print(f"\n=== {model}: only one horizon ({horizon} min); cross-horizon family skipped ===")
            summary["skipped_single_horizon_models"][model] = {
                "horizons_minutes": [horizon],
                "reason": "cross-horizon inference requires at least two horizons",
            }
            continue
        try:
            frame = analyse_family(
                model, family, args.metrics, tester,
                args.comparison_type, correction, args.test_family, args.output_dir,
            )
        except (ValueError, OSError) as error:
            print(f"  FAILED {model}: {error}", file=sys.stderr)
            failures.append(model)
            summary["failed_models"][model] = str(error)
            continue
        significance_column = (
            "significant_corrected" if "significant_corrected" in frame.columns else "significant"
        )
        summary["models"][model] = {
            "horizons_minutes": sorted(r.horizon_minutes for r in family),
            "patients": sorted(set().union(*(set(r.patients) for r in family))),
            "seeds": sorted(set().union(*(set(r.seeds) for r in family))),
            "comparisons": int(len(frame)),
            "significant": int(frame[significance_column].sum()),
        }

    summary_path = args.output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")

    if not summary["models"] and not failures and not skipped_experiments:
        print("No model had two or more horizons; all families were explicitly skipped.")

    if failures or skipped_experiments:
        print(f"Failed models: {', '.join(failures) or 'none'}; "
              f"skipped experiments: {len(skipped_experiments)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
