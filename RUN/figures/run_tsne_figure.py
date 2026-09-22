#!/usr/bin/env python3
"""t-SNE of test-period glucose windows, coloured by patient and by MAE.

Writes ``tsne_<split>_<MODEL>.png``. MAE is never an input to the embedding --
it colours an embedding built from the glucose windows alone; see
``PatientAnalyzer.plot_glucose_tsne``.

usage:

    python RUN/figures/run_tsne_figure.py \
        --experiment-dir results/experiments/experiment_20260915_185310_889f1664_e91bac \
        --model GRU --mode transfer \
        --output-dir results/analysis/tsne
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.analysis.experiments.patient_analysis import PatientAnalyzer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", required=True, type=Path,
                        help="Parent run directory (the one recorded in parents.txt)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="GRU", help="Model name as the metrics record it")
    parser.add_argument("--mode", default="transfer", choices=("regular", "transfer"))
    parser.add_argument("--seed", type=int, default=None,
                        help="Analyse one seed instead of the cross-seed mean")
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    tsne_path = args.output_dir / f"tsne_test_{args.model}.png"
    try:
        analyzer = PatientAnalyzer(
            str(args.experiment_dir), mode=args.mode, seed=args.seed,
        )
        result = analyzer.plot_glucose_tsne(
            model_name=args.model, save_path=str(tsne_path), max_samples=args.max_samples,
        )
        if result is None:
            failures.append("t-SNE: no embeddable windows were recovered")
        else:
            print(f"wrote {tsne_path}")
    except Exception as error:                      # one figure must not sink the other
        failures.append(f"t-SNE: {type(error).__name__}: {error}")

    for line in failures:
        print(f"FAILED {line}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
