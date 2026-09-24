#!/usr/bin/env python
"""Recombine the single-seed runs produced by run_seed_major.sh.

run_seed_major.sh runs one seed at a time so the whole grid can be covered
seed by seed. Each of those runs lands in its own parent directory holding a
single seed, whose aggregate therefore reports null for every dispersion field
-- one seed has no spread. The downstream analysis expects what the original
configs produced: one parent directory per model x horizon cell containing all
three seeds, with cross-seed means and Student-t intervals.

This script assembles that. For each cell it gathers the single-seed parents,
places their mode/seed_N directories under one parent, and recomputes the
aggregate with the benchmark's own _aggregate_runs, so the merged numbers are
produced by exactly the code path a three-seed run would have used.

Seed directories are symlinked by default; each is about 30 MB and copying the
full grid would duplicate roughly 2 GB. Pass --copy for real copies if the
result needs to stand alone (archiving, uploading, moving off this machine).

Usage:
  python merge_seed_runs.py --dry-run
  python merge_seed_runs.py
  python merge_seed_runs.py --models gru --horizons 30 --copy
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from benchmark.configs.config_manager import load_config
from benchmark.experiments.configured import _aggregate_rows, _aggregate_runs

MODES = ("regular", "transfer")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/experiments",
                        help="where the single-seed parent runs live")
    parser.add_argument("--out", default="results/experiments_merged",
                        help="where the merged three-seed parents are written")
    parser.add_argument("--configs-dir", default="configs",
                        help="directory holding the base full_<model>_<horizon>min.yaml configs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--models", nargs="+", default=["gru", "lstm", "rnn"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[15, 30, 45, 60])
    parser.add_argument("--copy", action="store_true",
                        help="copy seed directories instead of symlinking them")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be merged and stop")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing merged directory")
    return parser.parse_args(argv)


def index_single_seed_runs(results_dir: Path) -> Dict[str, Path]:
    """Map experiment name -> parent directory, for completed runs only.

    Directory names are timestamped and hashed, so the experiment name recorded
    in resolved_config.yaml is the only stable handle. A run without
    aggregate_metrics.json did not finish and is deliberately not indexed.
    """
    found: Dict[str, Path] = {}
    if not results_dir.is_dir():
        return found
    for resolved in sorted(results_dir.glob("*/resolved_config.yaml")):
        parent = resolved.parent
        if not (parent / "aggregate_metrics.json").is_file():
            continue
        try:
            name = yaml.safe_load(resolved.read_text(encoding="utf-8"))["experiment"]["name"]
        except (yaml.YAMLError, KeyError, TypeError):
            continue
        # Later directories win, so a re-run supersedes an earlier attempt.
        found[name] = parent
    return found


def load_seed_run(parent: Path, mode: str, seed: int) -> Optional[Dict[str, Any]]:
    """Rebuild one entry of the `runs` list that _aggregate_runs consumes.

    A seed directory stores its per-patient results under final_results, keyed
    model -> patient -> result. _aggregate_runs wants them keyed by patient,
    reading the model name back out of each result's model_info.
    """
    tracking = parent / mode / f"seed_{seed}" / "tracking.json"
    if not tracking.is_file():
        return None
    payload = json.loads(tracking.read_text(encoding="utf-8"))
    final_results = payload.get("final_results") or {}
    results: Dict[str, Any] = {}
    for model_name, patients in final_results.items():
        if not isinstance(patients, dict):
            continue
        for patient_id, result in patients.items():
            if patient_id in results:
                raise RuntimeError(
                    f"{tracking}: patient {patient_id} appears under more than one "
                    f"model ({model_name}); cannot merge unambiguously"
                )
            results[patient_id] = result
    if not results:
        return None
    return {"mode": mode, "seed": seed, "results": results}


def place(source: Path, target: Path, copy: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    if copy:
        shutil.copytree(source, target)
    else:
        target.symlink_to(source.resolve(), target_is_directory=True)


def merge_cell(model: str, horizon: int, seeds: List[int], runs_index: Dict[str, Path],
               args: argparse.Namespace) -> Optional[str]:
    """Merge one model x horizon cell. Returns an error string, or None on success."""
    cell = f"full_{model}_{horizon}min"
    sources: Dict[int, Path] = {}
    missing: List[int] = []
    for seed in seeds:
        parent = runs_index.get(f"{cell}_seed{seed}")
        if parent is None:
            missing.append(seed)
        else:
            sources[seed] = parent
    if missing:
        return f"{cell}: missing completed run(s) for seed(s) {', '.join(map(str, missing))}"

    base_config_path = Path(args.configs_dir) / f"{cell}.yaml"
    if not base_config_path.is_file():
        return f"{cell}: base config not found at {base_config_path}"
    config = load_config(base_config_path)

    from grid_protocol import matches
    for seed, parent in sources.items():
        if not matches(parent / "resolved_config.yaml", base_config_path, seed):
            return f"{cell}: seed {seed} configuration differs from the current grid; rerun it before merging"

    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        for mode in MODES:
            run = load_seed_run(sources[seed], mode, seed)
            if run is not None:
                runs.append(run)
    if not runs:
        return f"{cell}: no seed run held usable results"

    found_modes = sorted({run["mode"] for run in runs})
    found_seeds = sorted({run["seed"] for run in runs})
    if args.dry_run:
        print(f"  {cell}: would merge seeds {found_seeds}, modes {found_modes} "
              f"({len(runs)} seed-run(s))")
        return None

    out_dir = Path(args.out) / cell
    if out_dir.exists():
        if not args.force:
            return f"{cell}: {out_dir} already exists (use --force to replace)"
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for seed in seeds:
        for mode in MODES:
            source = sources[seed] / mode / f"seed_{seed}"
            if source.is_dir():
                place(source, out_dir / mode / f"seed_{seed}", args.copy)

    aggregates = _aggregate_runs(runs, config)
    (out_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregates, indent=2), encoding="utf-8")
    pd.DataFrame(_aggregate_rows(aggregates)).to_csv(
        out_dir / "aggregate_metrics.csv", index=False)
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    (out_dir / "merge_manifest.json").write_text(json.dumps({
        "cell": cell,
        "seeds": found_seeds,
        "modes": found_modes,
        "seed_directories": "copied" if args.copy else "symlinked",
        "sources": {str(seed): str(path) for seed, path in sources.items()},
        "aggregated_seed_runs": len(runs),
    }, indent=2), encoding="utf-8")

    seeds_per_aggregate = {len(entry["seeds"]) for entry in aggregates}
    print(f"  {cell}: merged seeds {found_seeds} -> {out_dir} "
          f"({len(aggregates)} aggregate rows, {sorted(seeds_per_aggregate)} seed(s) each)")
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    runs_index = index_single_seed_runs(Path(args.results_dir))
    print(f"Found {len(runs_index)} completed run(s) in {args.results_dir}")
    print(f"Merging seeds {args.seeds} for {len(args.models) * len(args.horizons)} cell(s)"
          + (" [dry run]" if args.dry_run else ""))

    problems: List[str] = []
    for model in args.models:
        for horizon in args.horizons:
            problem = merge_cell(model, horizon, args.seeds, runs_index, args)
            if problem:
                problems.append(problem)

    if problems:
        print("\nNot merged:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\nAll requested cells merged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
