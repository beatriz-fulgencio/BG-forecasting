"""Forecast error stratified by the glycemic range it was made in.

usage:
    python -m RUN.experiments.run_error_by_range_analysis \\
      --run-dir results/experiments/experiment_ID/regular/seed_41 \\
      --run-dir results/experiments/experiment_ID/regular/seed_42 \\
      --run-dir results/experiments/experiment_ID/regular/seed_43
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.analysis.experiments.error_by_range import analyze_error_by_range_runs
from benchmark.analysis.results_io import ResultsLayoutError


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, action="append", type=Path, dest="run_dirs",
                        help="Configured mode/seed directory, e.g. .../regular/seed_42. "
                             "Repeat once per training seed of the same mode/horizon cell.")
    parser.add_argument("--model", default=None,
                        help="Model name; inferred when the run holds only one.")
    parser.add_argument("--output-dir", type=Path,
                        help="Analysis output directory (default: <experiment>/analysis/"
                             "error_by_range_<mode>)")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    try:
        summary, per_patient = analyze_error_by_range_runs(
            args.run_dirs, model=args.model, replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    # ResultsLayoutError is a RuntimeError, so it is named explicitly rather than
    # caught by the ValueError
    except (ValueError, TypeError, OSError, KeyError, ResultsLayoutError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        mode_dir = args.run_dirs[0].resolve().parent
        output_dir = mode_dir.parent / "analysis" / f"error_by_range_{summary['mode']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_patient.to_csv(output_dir / "per_patient.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
