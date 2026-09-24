"""Describe patient difficulty stability across model, horizon, mode, and seed.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from benchmark.analysis.experiments.patient_stability import (
    IMPLEMENTATION_REFERENCES,
    SPEARMAN_FAMILIES,
    load_patient_stability,
    plot_hard_easy_spread,
    seed_rank_sensitivity,
    summarize_patient_stability,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", action="append", required=True, type=Path, dest="parents")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        seed = load_patient_stability(args.parents)
        patient, spearman, concordance, spread = summarize_patient_stability(seed)
        sensitivity = seed_rank_sensitivity(seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in [
            ("per_patient_seed.csv", seed),
            ("per_patient.csv", patient),
            ("pairwise_spearman.csv", spearman),
            ("kendalls_w.csv", concordance),
            ("hard_easy_spread.csv", spread),
            ("seed_rank_sensitivity.csv", sensitivity),
        ]:
            frame.to_csv(args.output_dir / name, index=False)
        plot_hard_easy_spread(spread, args.output_dir / "hard_easy_spread.png")
    except (ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    (args.output_dir / "summary.json").write_text(json.dumps({
        "analysis": "patient_stability",
        "unit_of_analysis": "patient",
        "evidence_level": "descriptive",
        "n_patients": int(patient.patient_id.nunique()),
        "n_conditions": int(patient.condition.nunique()),
        "multiple_testing": "Benjamini-Hochberg within each declared family, separately",
        "spearman_families": {
            name: {"description": description,
                   "size": int(frame.shape[0])}
            for (name, description), frame in zip(SPEARMAN_FAMILIES.items(), (spearman, sensitivity))
        },
        "implementation_references": IMPLEMENTATION_REFERENCES,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
