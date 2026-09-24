"""Localize forecast error to calm versus rapid, timestamp-contiguous changes.

Pass all seed directories for one or more model/mode/horizon cells.  The command
writes seed-level patient rows, cell summaries, and a generic plot; it makes no
article-specific selection.  The implementation reference and licence review
are recorded in ``benchmark.analysis.experiments.rapid_change``: pandas and SciPy are
BSD-3-Clause projects whose licences permit reuse with retained notices.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.analysis.experiments.rapid_change import analyze_rapid_change_runs, plot_rapid_change
from benchmark.analysis.results_io import ResultsLayoutError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path, dest="run_dirs")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold-mg-dl", type=float, default=None,
                        help="Absolute reference change counted as rapid. Default: the "
                             "step implied by the clinical rapid rate (3 mg/dL/min) at "
                             "the run's own sampling interval, i.e. 15 mg/dL at 5 min.")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    try:
        per_seed_patient, summary_table, summary = analyze_rapid_change_runs(
            args.run_dirs, threshold_mg_dl=args.threshold_mg_dl,
            bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed,
        )
    # ResultsLayoutError is a RuntimeError, so it escapes the three below: an
    # unreadable run directory is a data problem like the rest and belongs in the
    # same one-line message, not in a traceback.
    except (ValueError, OSError, KeyError, ResultsLayoutError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_patient.to_csv(args.output_dir / "per_patient_seed.csv", index=False)
    summary_table.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps({**summary, "cells": summary_table.to_dict("records")}, indent=2), encoding="utf-8"
    )
    plot_rapid_change(summary_table, args.output_dir / "rapid_change_localization.png")
    print(json.dumps({**summary, "cells": summary_table.to_dict("records")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
