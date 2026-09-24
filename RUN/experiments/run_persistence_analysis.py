"""Compute the forecast-origin persistence baseline for configured runs.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.analysis.experiments.persistence import analyze_persistence_runs, plot_persistence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", action="append", required=True, dest="run_dirs", type=Path,
        help="Configured <mode>/seed_<n> directory; repeat for every run to check.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--invariance-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    try:
        per_run, per_patient, summary = analyze_persistence_runs(
            args.run_dirs, tolerance=args.invariance_tolerance
        )
    except (ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.output_dir / "per_patient_run.csv", index=False)
    per_patient.to_csv(args.output_dir / "per_patient.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_persistence(per_patient, args.output_dir / "persistence_by_horizon.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
