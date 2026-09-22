"""Patient-cluster uncertainty for the Clarke zone-D clinical decomposition.

usage:
    python -m RUN.experiments.run_zone_d_analysis \\
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

from benchmark.analysis.experiments.zone_d import analyze_zone_d_runs
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
                        help="Analysis output directory (default: zone_d_analysis beside the run(s))")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    try:
        summary, per_patient = analyze_zone_d_runs(
            args.run_dirs, model=args.model, replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (ValueError, TypeError, OSError, KeyError, ResultsLayoutError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif len(args.run_dirs) == 1:
        output_dir = args.run_dirs[0] / "zone_d_analysis"
    else:
        output_dir = args.run_dirs[0].parent / "zone_d_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_patient.to_csv(output_dir / "per_patient.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
