"""Independent paired transfer-versus-regular inference for configured parents.

Uses patient cross-seed mean MAE for Wilcoxon, exact sign tests, effects, and
Benjamini-Hochberg correction.
"""

import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from benchmark.analysis.experiments.transfer_inference import analyze_transfer_inference, plot_transfer_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", action="append", required=True, type=Path, dest="parents")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--resampling-seed", type=int, default=42)
    args = parser.parse_args()
    try:
        per_seed, per_patient, inference, summary = analyze_transfer_inference(
            args.parents, replicates=args.bootstrap_replicates, random_seed=args.resampling_seed
        )
    except (ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if inference.empty:
        print("ERROR: no comparable model/horizon cells were produced", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "per_patient_seed.csv", index=False)
    per_patient.to_csv(args.output_dir / "per_patient.csv", index=False)
    inference.to_csv(args.output_dir / "transfer_inference.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_transfer_inference(inference, args.output_dir / "transfer_inference.png")
    print(inference.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
