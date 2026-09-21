#!/usr/bin/env python3
"""Command-line interface for the blood glucose forecasting benchmark."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .configs.config_manager import (
    ConfigError,
    SUPPORTED_DATASETS,
    SUPPORTED_METRICS,
    SUPPORTED_MODELS,
)
from .experiments.runner import exit_code, run_experiment_from_config
from .experiments.tracking import load_experiment


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Blood Glucose Forecasting Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --config configs/example_experiment.yaml
  %(prog)s resume --run-dir results/experiments/experiment_ID/transfer/seed_42
  %(prog)s analyze --experiment-dir results/experiments/experiment_ID
  %(prog)s compare --experiments experiment_1 experiment_2
  %(prog)s list --models
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one validated experiment config")
    run_parser.add_argument("--config", required=True, help="path to a versioned YAML config")

    resume_parser = subparsers.add_parser(
        "resume", help="resume missing patients in one interrupted seed run"
    )
    resume_parser.add_argument(
        "--run-dir", required=True,
        help="existing mode/seed directory containing tracking.json and resolved_config.yaml",
    )

    analyze_parser = subparsers.add_parser("analyze", help="print metrics from one completed run")
    analyze_parser.add_argument("--experiment-dir", required=True, help="experiment directory")
    analyze_parser.add_argument("--metrics", nargs="+", help="metrics to include")

    compare_parser = subparsers.add_parser("compare", help="compare completed experiment runs")
    compare_parser.add_argument("--experiments", nargs="+", required=True, help="experiment directories")
    compare_parser.add_argument(
        "--output",
        default="results/comparisons",
        help="output directory, or a .json output path",
    )

    list_parser = subparsers.add_parser("list", help="list implemented components")
    list_parser.add_argument("--models", action="store_true")
    list_parser.add_argument("--datasets", action="store_true")
    list_parser.add_argument("--metrics", action="store_true")
    return parser


def _normalize_metrics(metrics: Optional[List[str]]):
    if metrics is None:
        return list(SUPPORTED_METRICS)
    aliases = {"time_in_range": "tir", "clarke": "clarke_ega", "parkes": "parkes_ega"}
    normalized = [aliases.get(metric.lower(), metric.lower()) for metric in metrics]
    unknown = sorted(set(normalized) - set(SUPPORTED_METRICS))
    if unknown:
        raise ConfigError(f"Unsupported metric(s): {', '.join(unknown)}")
    return normalized


def _result_key(metric: str):
    return {"clarke_ega": "clarke_zones", "parkes_ega": "parkes_zones"}.get(metric, metric)


# A run whose seeds partly failed is still readable -- refusing it would strand
# the seeds that did finish -- but it reports its failures alongside the results.
TERMINAL_STATUSES = ("completed", "completed_with_failures")


def _load_completed(path: str):
    experiment = load_experiment(path)
    status = experiment.get("status")
    if status not in TERMINAL_STATUSES:
        raise RuntimeError(f"Experiment {path} is not completed (status: {status!r})")
    return experiment


def _with_failures(experiment: Dict[str, Any], analysis: Dict[str, Any]):
    """Carry failed subruns into the analysis so partial results read as partial."""
    failed = experiment.get("final_results", {}).get("failed_runs")
    if failed:
        analysis["failed_runs"] = failed
        analysis["partial"] = True
    return analysis


def _analysis(experiment: Dict[str, Any], metrics: List[str]):
    config = experiment.get("config", {})
    preprocessing = config.get("preprocessing", {})
    horizon_steps = preprocessing.get("prediction_horizon")
    sampling_rate = preprocessing.get("sampling_rate")
    horizon_minutes = (
        horizon_steps * sampling_rate
        if isinstance(horizon_steps, int) and isinstance(sampling_rate, int) else None
    )
    aggregates = experiment.get("final_results", {}).get("aggregate", [])
    if aggregates:
        rows: List[Dict[str, Any]] = []
        for aggregate in aggregates:
            row: Dict[str, Any] = {
                "mode": aggregate["mode"],
                "model": aggregate["model"],
                "patient_id": aggregate["patient_id"],
                "seeds": aggregate["seeds"],
            }
            if horizon_minutes is not None:
                row["horizon_minutes"] = horizon_minutes
            aggregate_metrics = aggregate["metrics"]
            for metric in metrics:
                key = _result_key(metric)
                if key in aggregate_metrics:
                    row[metric] = aggregate_metrics[key]
                    continue
                prefix = f"{key}."
                nested = {
                    name[len(prefix):]: summary
                    for name, summary in aggregate_metrics.items()
                    if name.startswith(prefix)
                }
                if nested:
                    row[metric] = nested
            rows.append(row)
        return _with_failures(
            experiment,
            {"experiment_id": experiment.get("experiment_id"), "results": rows},
        )

    rows: List[Dict[str, Any]] = []
    for model_key, model in experiment.get("models", {}).items():
        evaluation = model.get("evaluation_results")
        if not evaluation:
            continue
        row: Dict[str, Any] = {
            "model": model_key,
            "patient_id": model.get("patient_id"),
        }
        if horizon_minutes is not None:
            row["horizon_minutes"] = horizon_minutes
        for metric in metrics:
            key = _result_key(metric)
            if key in evaluation:
                row[metric] = evaluation[key]
        rows.append(row)
    if not rows:
        raise RuntimeError("No evaluated models were found in tracking.json")
    return _with_failures(
        experiment,
        {"experiment_id": experiment.get("experiment_id"), "results": rows},
    )


def _output_path(value: str):
    path = Path(value)
    return path if path.suffix.lower() == ".json" else path / "comparison.json"


def _run(args: argparse.Namespace):
    return exit_code(run_experiment_from_config(args.config))


def _resume(args: argparse.Namespace):
    from .experiments.configured import resume_configured_seed_run

    result = resume_configured_seed_run(args.run_dir)
    print(f"Completed {result['mode']} seed {result['seed']}: {result['experiment_dir']}")
    return 0


def _analyze(args: argparse.Namespace):
    metrics = _normalize_metrics(args.metrics)
    result = _analysis(_load_completed(args.experiment_dir), metrics)
    if result.get("partial"):
        print(
            f"Warning: {len(result['failed_runs'])} subrun(s) failed; "
            "these results cover the successful runs only",
            file=sys.stderr,
        )
    print(json.dumps(result, indent=2))
    return 0


def _compare(args: argparse.Namespace):
    if len(args.experiments) < 2:
        raise ConfigError("compare requires at least two experiment directories")
    analyses = [
        _analysis(_load_completed(path), list(SUPPORTED_METRICS))
        for path in args.experiments
    ]
    model_results = []
    for analysis in analyses:
        for row in analysis["results"]:
            model_results.append({"experiment_id": analysis["experiment_id"], **row})

    def mae_mean(row):
        value = row.get("mae")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
            return float(value["mean"])
        return None

    comparable_mae = [row for row in model_results if mae_mean(row) is not None]
    mae_ranking = []
    for row in comparable_mae:
        ranking_row = {
            "experiment_id": row["experiment_id"],
            "model": row["model"],
            "patient_id": row["patient_id"],
            "mae_mean": mae_mean(row),
        }
        for optional_key in ("mode", "seeds"):
            if optional_key in row:
                ranking_row[optional_key] = row[optional_key]
        mae_ranking.append(ranking_row)
    mae_ranking.sort(key=lambda row: row["mae_mean"])
    partitioned = {}
    for row in comparable_mae:
        key = (row.get("horizon_minutes"), row.get("mode"))
        partitioned.setdefault(key, []).append(row)
    mae_ranking_by_horizon_mode = []
    for (horizon, mode), rows in sorted(
        partitioned.items(), key=lambda item: (item[0][0] is None, item[0][0] or 0, item[0][1] or "")
    ):
        ordered = sorted(rows, key=lambda row: mae_mean(row))
        for rank, row in enumerate(ordered, start=1):
            mae_ranking_by_horizon_mode.append({
                "horizon_minutes": horizon,
                "mode": mode,
                "rank": rank,
                "experiment_id": row["experiment_id"],
                "model": row["model"],
                "patient_id": row["patient_id"],
                "mae_mean": mae_mean(row),
            })
    comparison = {
        "experiment_count": len(analyses),
        "experiments": analyses,
        "model_results": model_results,
        "mae_ranking": mae_ranking,
        "mae_ranking_by_horizon_mode": mae_ranking_by_horizon_mode,
    }
    output = _output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Comparison saved to {output}")
    return 0


def _list(args: argparse.Namespace):
    show_all = not (args.models or args.datasets or args.metrics)
    if show_all or args.models:
        print("Models: " + ", ".join(SUPPORTED_MODELS))
    if show_all or args.datasets:
        print("Datasets: " + ", ".join(SUPPORTED_DATASETS))
    if show_all or args.metrics:
        print("Metrics: " + ", ".join(SUPPORTED_METRICS))
    return 0


def main(argv: Optional[List[str]] = None):
    """Run the CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": _run, "resume": _resume, "analyze": _analyze,
        "compare": _compare, "list": _list,
    }
    try:
        return handlers[args.command](args)
    except (ConfigError, FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, ConfigError) else 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
