from dataclasses import replace

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


def test_aggregate_runs_computes_population_statistics():
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
    assert aggregate["metrics"]["mae"] == {
        "mean": 12.0, "std": 2.0, "min": 10.0, "max": 14.0, "n": 2
    }
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


def test_single_seed_reports_zero_spread_and_n_of_one():
    config = load_config("benchmark/configs/example_experiment.yaml")
    runs = [{
        "mode": "regular",
        "seed": 42,
        "results": {"540": {"mae": 12.5, "model_info": {"model_name": "GRU"}}},
    }]
    summary = configured._aggregate_runs(runs, config)[0]["metrics"]["mae"]
    assert summary == {"mean": 12.5, "std": 0.0, "min": 12.5, "max": 12.5, "n": 1}


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
