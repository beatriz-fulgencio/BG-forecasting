import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark.configs import load_config
from benchmark.experiments import configured


def test_all_mode_seed_combinations_use_one_parent(monkeypatch, tmp_path):
    config = load_config("benchmark/configs/example_experiment.yaml")
    config = replace(
        config,
        training=replace(config.training, mode="both", seeds=[11, 22]),
        output=replace(config.output, directory=str(tmp_path), export_format=["json"]),
    )
    calls = []

    def fake_run_mode(config, mode, seed, patient_ids, frames, experiment_dir):
        calls.append((mode, seed, patient_ids, experiment_dir))
        return {
            "mode": mode,
            "seed": seed,
            "experiment_id": f"{mode}-{seed}",
            "experiment_dir": str(experiment_dir),
            "results": {
                "540": {
                    "mae": float(seed),
                    "model_info": {"model_name": "GRU"},
                }
            },
        }

    monkeypatch.setattr(configured, "_load_patient_frames", lambda config, patients: {})
    monkeypatch.setattr(configured, "_run_mode", fake_run_mode)
    result = configured.run_configured_experiment(config, [540, 544])
    assert [(mode, seed) for mode, seed, _, _ in calls] == [
        ("regular", 11), ("regular", 22), ("transfer", 11), ("transfer", 22)
    ]
    parent = tmp_path / f"experiment_{result['experiment_id']}"
    assert result["experiment_dir"] == str(parent)
    assert (parent / "aggregate_metrics.json").is_file()
    assert len(result["runs"]) == 4


def test_aggregate_runs_computes_sample_statistics():
    config = load_config("benchmark/configs/example_experiment.yaml")
    runs = [
        {
            "mode": "regular",
            "seed": seed,
            "results": {
                "540": {
                    "mae": mae,
                    "clarke_zones": {"A": zone_a},
                    "model_info": {"model_name": "GRU"},
                }
            },
        }
        for seed, mae, zone_a in [(1, 10.0, 90.0), (2, 14.0, 94.0)]
    ]
    aggregate = configured._aggregate_runs(runs, config)[0]
    assert aggregate["seeds"] == [1, 2]
    mae = aggregate["metrics"]["mae"]
    assert mae["mean"] == pytest.approx(12.0)
    # Sample standard deviation (ddof=1), not the population one: seeds are a
    # sample of training runs. Population sd of [10, 14] would be 2.0.
    assert mae["std"] == pytest.approx(math.sqrt(8.0))
    assert mae["sem"] == pytest.approx(math.sqrt(8.0) / math.sqrt(2))
    assert mae["min"] == 10.0 and mae["max"] == 14.0 and mae["n"] == 2
    assert aggregate["metrics"]["clarke_zones.A"]["mean"] == pytest.approx(92.0)


def _stub_run_mode(results_for):
    """Build a ``_run_mode`` stub that records calls and returns canned results."""
    calls = []

    def fake_run_mode(config, mode, seed, patient_ids, frames, experiment_dir):
        calls.append((mode, seed, experiment_dir))
        return {
            "mode": mode,
            "seed": seed,
            "experiment_id": f"{mode}-{seed}",
            "experiment_dir": str(experiment_dir),
            "results": results_for(mode, seed),
        }

    return calls, fake_run_mode


def test_three_seeds_produce_one_subrun_directory_each(monkeypatch, tmp_path):
    config = load_config("benchmark/configs/example_experiment.yaml")
    config = replace(
        config,
        training=replace(config.training, mode="regular", seeds=[43, 7, 101]),
        output=replace(config.output, directory=str(tmp_path), export_format=["json"]),
    )
    calls, fake_run_mode = _stub_run_mode(
        lambda mode, seed: {"540": {"mae": float(seed), "model_info": {"model_name": "GRU"}}}
    )
    monkeypatch.setattr(configured, "_load_patient_frames", lambda config, patients: {})
    monkeypatch.setattr(configured, "_run_mode", fake_run_mode)

    result = configured.run_configured_experiment(config, [540, 544])

    parent = tmp_path / f"experiment_{result['experiment_id']}"
    assert [directory for _, _, directory in calls] == [
        parent / "regular" / f"seed_{seed}" for seed in (43, 7, 101)
    ]
    aggregate = result["aggregate"][0]
    assert aggregate["seeds"] == [43, 7, 101]
    assert aggregate["metrics"]["mae"]["n"] == 3


def test_aggregate_rows_flatten_one_row_per_metric():
    aggregates = [{
        "mode": "regular",
        "patient_id": 540,
        "model": "GRU",
        "seeds": [41, 42],
        "metrics": {
            "mae": {"mean": 12.0, "std": 2.0, "min": 10.0, "max": 14.0, "n": 2},
            "rmse": {"mean": 15.0, "std": 1.0, "min": 14.0, "max": 16.0, "n": 2},
        },
    }]
    rows = configured._aggregate_rows(aggregates)
    assert [row["metric"] for row in rows] == ["mae", "rmse"]
    assert rows[0] == {
        "mode": "regular", "patient_id": 540, "model": "GRU", "seeds": "41,42",
        "metric": "mae", "mean": 12.0, "std": 2.0, "min": 10.0, "max": 14.0, "n": 2,
    }


def test_modes_and_patients_aggregate_into_separate_groups():
    config = load_config("benchmark/configs/example_experiment.yaml")
    runs = [
        {
            "mode": mode,
            "seed": seed,
            "results": {
                "540": {"mae": mae, "model_info": {"model_name": "GRU"}},
                "544": {"mae": mae + 100.0, "model_info": {"model_name": "GRU"}},
            },
        }
        for mode in ("regular", "transfer")
        for seed, mae in [(1, 10.0), (2, 14.0)]
    ]
    aggregates = configured._aggregate_runs(runs, config)
    assert [(item["mode"], item["patient_id"]) for item in aggregates] == [
        ("regular", 540), ("regular", 544), ("transfer", 540), ("transfer", 544)
    ]
    assert all(item["metrics"]["mae"]["n"] == 2 for item in aggregates)
    assert aggregates[1]["metrics"]["mae"]["mean"] == pytest.approx(112.0)


def test_single_seed_reports_no_spread_rather_than_zero():
    """One seed has no spread to report; zero would read as perfect agreement."""
    config = load_config("benchmark/configs/example_experiment.yaml")
    runs = [{
        "mode": "regular",
        "seed": 42,
        "results": {"540": {"mae": 12.5, "model_info": {"model_name": "GRU"}}},
    }]
    summary = configured._aggregate_runs(runs, config)[0]["metrics"]["mae"]
    assert summary == {
        "mean": 12.5, "std": None, "sem": None, "ci95_low": None,
        "ci95_high": None, "min": 12.5, "max": 12.5, "n": 1,
    }


def test_confidence_interval_uses_student_t():
    """With few seeds the t interval is much wider than a normal one."""
    summary = configured._summarize_across_seeds([10.0, 14.0, 12.0])
    assert summary["mean"] == pytest.approx(12.0)
    assert summary["std"] == pytest.approx(2.0)
    assert summary["sem"] == pytest.approx(2.0 / math.sqrt(3))
    # t(0.975, df=2) = 4.302..., well above the normal 1.96.
    half_width = summary["ci95_high"] - summary["mean"]
    assert half_width == pytest.approx(4.302653 * summary["sem"], rel=1e-5)
    assert summary["ci95_low"] == pytest.approx(summary["mean"] - half_width)


def test_confidence_interval_brackets_the_mean():
    summary = configured._summarize_across_seeds([18.0, 21.0, 19.5, 20.2, 17.9])
    assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]
    assert summary["n"] == 5


def test_aggregate_rows_expose_the_interval_columns():
    """The CSV must carry the interval, not just the mean and spread."""
    config = load_config("benchmark/configs/example_experiment.yaml")
    runs = [
        {"mode": "regular", "seed": seed,
         "results": {"540": {"mae": mae, "model_info": {"model_name": "GRU"}}}}
        for seed, mae in [(1, 10.0), (2, 14.0), (3, 12.0)]
    ]
    row = configured._aggregate_rows(configured._aggregate_runs(runs, config))[0]
    assert {"mean", "std", "sem", "ci95_low", "ci95_high", "min", "max", "n"} <= set(row)
    assert row["seeds"] == "1,2,3"


def test_metric_missing_from_one_seed_is_counted_only_where_present():
    config = load_config("benchmark/configs/example_experiment.yaml")
    runs = [
        {
            "mode": "regular",
            "seed": 1,
            "results": {"540": {"mae": 10.0, "rmse": 11.0,
                                "model_info": {"model_name": "GRU"}}},
        },
        {
            "mode": "regular",
            "seed": 2,
            "results": {"540": {"mae": 14.0, "model_info": {"model_name": "GRU"}}},
        },
    ]
    metrics = configured._aggregate_runs(runs, config)[0]["metrics"]
    assert metrics["mae"]["n"] == 2
    assert metrics["rmse"]["n"] == 1


def test_prediction_diagnostics_aggregate_across_seeds():
    """Implausible-prediction counts must reach the cross-seed summary."""
    config = load_config("benchmark/configs/example_experiment.yaml")
    runs = [
        {
            "mode": "regular",
            "seed": seed,
            "results": {
                "540": {
                    "mae": 10.0,
                    "prediction_diagnostics": {
                        "n_points": 100,
                        "n_implausible_predictions": implausible,
                        "implausible_prediction_rate": implausible / 100,
                    },
                    "model_info": {"model_name": "GRU"},
                }
            },
        }
        for seed, implausible in [(1, 0), (2, 4)]
    ]
    metrics = configured._aggregate_runs(runs, config)[0]["metrics"]
    summary = metrics["prediction_diagnostics.n_implausible_predictions"]
    assert summary["mean"] == 2.0
    assert summary["max"] == 4.0
    assert summary["n"] == 2


def _failing_run_mode(fail_on):
    """Stub whose given (mode, seed) pairs raise instead of returning results."""
    def fake_run_mode(config, mode, seed, patient_ids, frames, experiment_dir):
        if (mode, seed) in fail_on:
            raise RuntimeError(f"CUDA out of memory on seed {seed}")
        return {
            "mode": mode, "seed": seed, "experiment_id": f"{mode}-{seed}",
            "experiment_dir": str(experiment_dir),
            "results": {"540": {"mae": float(seed), "model_info": {"model_name": "GRU"}}},
        }
    return fake_run_mode


def _three_seed_config(tmp_path):
    config = load_config("benchmark/configs/example_experiment.yaml")
    return replace(
        config,
        training=replace(config.training, mode="regular", seeds=[11, 22, 33]),
        output=replace(config.output, directory=str(tmp_path), export_format=["json"]),
    )


def test_one_failed_seed_does_not_discard_the_others(monkeypatch, tmp_path):
    config = _three_seed_config(tmp_path)
    monkeypatch.setattr(configured, "_load_patient_frames", lambda config, patients: {})
    monkeypatch.setattr(configured, "_run_mode", _failing_run_mode({("regular", 22)}))

    result = configured.run_configured_experiment(config, [540, 544])

    assert [run["seed"] for run in result["runs"]] == [11, 33]
    assert [failure["seed"] for failure in result["failed_runs"]] == [22]
    assert "CUDA out of memory" in result["failed_runs"][0]["error"]
    # The surviving seeds still produce an aggregate, and it says n=2.
    assert result["aggregate"][0]["metrics"]["mae"]["n"] == 2
    parent = tmp_path / f"experiment_{result['experiment_id']}"
    assert (parent / "aggregate_metrics.json").is_file()


def test_partial_runs_are_marked_so_they_cannot_read_as_whole(monkeypatch, tmp_path):
    config = _three_seed_config(tmp_path)
    monkeypatch.setattr(configured, "_load_patient_frames", lambda config, patients: {})
    monkeypatch.setattr(configured, "_run_mode", _failing_run_mode({("regular", 22)}))

    result = configured.run_configured_experiment(config, [540, 544])

    tracking = json.loads(
        (Path(result["experiment_dir"]) / "tracking.json").read_text(encoding="utf-8")
    )
    assert tracking["status"] == "completed_with_failures"
    assert [f["seed"] for f in tracking["final_results"]["failed_runs"]] == [22]


def test_every_seed_failing_is_still_fatal(monkeypatch, tmp_path):
    config = _three_seed_config(tmp_path)
    monkeypatch.setattr(configured, "_load_patient_frames", lambda config, patients: {})
    monkeypatch.setattr(
        configured, "_run_mode",
        _failing_run_mode({("regular", 11), ("regular", 22), ("regular", 33)}),
    )
    with pytest.raises(RuntimeError, match="all 3 failed"):
        configured.run_configured_experiment(config, [540, 544])


def test_a_clean_run_is_not_marked_partial(monkeypatch, tmp_path):
    config = _three_seed_config(tmp_path)
    monkeypatch.setattr(configured, "_load_patient_frames", lambda config, patients: {})
    monkeypatch.setattr(configured, "_run_mode", _failing_run_mode(set()))

    result = configured.run_configured_experiment(config, [540, 544])

    assert result["failed_runs"] == []
    tracking = json.loads(
        (Path(result["experiment_dir"]) / "tracking.json").read_text(encoding="utf-8")
    )
    assert tracking["status"] == "completed"
    assert "failed_runs" not in tracking["final_results"]
