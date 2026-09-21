"""
Blood Glucose Forecasting Benchmark

A reproducible benchmark framework for comparing blood glucose prediction models.
This package provides standardized data processing, model interfaces, evaluation
metrics, and experiment management for fair comparison of different approaches.

"""

from typing import Any, List

__version__ = "0.1.0"
__author__ = "Blood Glucose Forecasting Research Group"

__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "load_config",
    "validate_data_files",
    "run_experiment",
    "run_experiment_from_config",
    "run_configured_experiment",
]

_CONFIG_EXPORTS = frozenset(
    {"ConfigError", "ExperimentConfig", "load_config", "validate_data_files"}
)
_RUNNER_EXPORTS = frozenset({"run_experiment", "run_experiment_from_config"})


def __getattr__(name: str):
    """Resolve the public API lazily (PEP 562)."""
    if name in _CONFIG_EXPORTS:
        from . import configs

        return getattr(configs, name)
    if name in _RUNNER_EXPORTS:
        from .experiments import runner

        return getattr(runner, name)
    if name == "run_configured_experiment":
        # The engine behind ``run_experiment``, for callers that have already
        # resolved a config and want no reporting around the run.
        from .experiments.configured import run_configured_experiment

        return run_configured_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__ + ["__version__", "__author__"])
