"""Entry point for running one versioned experiment configuration.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from ..configs.config_manager import (
    ConfigError,
    ExperimentConfig,
    load_config,
    validate_data_files,
)


def run_experiment(config: ExperimentConfig, patient_ids: List[int]) -> Dict[str, Any]:
    """Run every configured mode/seed combination and report on each one.

    Args:
        config: A validated experiment configuration.
        patient_ids: Patient IDs resolved from the config by
            :func:`~benchmark.configs.config_manager.validate_data_files`.

    Returns:
        The experiment summary from
        :func:`~benchmark.experiments.configured.run_configured_experiment`,
        carrying ``failed_runs`` when some mode/seed combinations failed. Pass
        it to :func:`exit_code` to turn that into a process exit status.
    """
    _report_plan(config, patient_ids)
    from .configured import run_configured_experiment

    experiment = run_configured_experiment(config, patient_ids)
    _report_outcome(experiment)
    return experiment


def run_experiment_from_config(config_path: str = None, **kwargs) -> Dict[str, Any]:
    """Load and validate a YAML configuration, then run it.

    Args:
        config_path: Path to a versioned YAML configuration file.
        **kwargs: Rejected. Overrides belong in the YAML file, where they are
            validated and recorded with the run.

    Returns:
        The experiment summary, as described on :func:`run_experiment`.
    """
    if not config_path:
        raise ValueError("config_path is required; direct parameter execution is not supported")
    if kwargs:
        raise ValueError("Configuration overrides are not supported; update the YAML file instead")

    config = load_config(config_path)
    return run_experiment(config, validate_data_files(config))


def exit_code(experiment: Dict[str, Any]) -> int:
    """0 when every mode/seed finished, 1 when any of them failed.

    A partly failed run still writes results for the seeds that finished, so it
    returns rather than raising; the exit status is what tells a caller the
    aggregate covers less than it was asked for.
    """
    return 1 if experiment.get("failed_runs") else 0


def _report_plan(config: ExperimentConfig, patient_ids: List[int]) -> None:
    """Say what is about to run, before anything slow starts."""
    print(f"Config: {config.source_path}")
    print(f"Experiment: {config.experiment.name}")
    print(f"Dataset: OhioT1DM {config.data.version}; patients: {patient_ids}")
    print(f"Model: {config.model.type}; training mode: {config.training.mode}")
    print(f"Seeds: {config.training.seeds}; device: {config.training.device}")
    print(f"Output root: {config.output.directory}")


def _report_outcome(experiment: Dict[str, Any]) -> None:
    """Name every run directory, so the next command has something to read."""
    for run in experiment["runs"]:
        print(f"Completed {run['mode']} seed {run['seed']}: {run['experiment_dir']}")
    for failure in experiment.get("failed_runs", []):
        print(
            f"FAILED {failure['mode']} seed {failure['seed']}: {failure['error']}",
            file=sys.stderr,
        )
    print(f"Parent experiment: {experiment['experiment_dir']}")


def main(argv: Optional[List[str]] = None) -> int:
    """Run one experiment and return a process exit code."""
    parser = argparse.ArgumentParser(description="Run one benchmark experiment")
    parser.add_argument("--config", required=True, help="path to a versioned YAML config")
    args = parser.parse_args(argv)
    try:
        return exit_code(run_experiment_from_config(args.config))
    except (ConfigError, FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, ConfigError) else 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130

if __name__ == "__main__":
    sys.exit(main())
