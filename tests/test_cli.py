import json
from pathlib import Path

from benchmark import cli


def _tracking(path: Path, experiment_id: str, mae: float, status: str = "completed"):
    path.mkdir(parents=True)
    (path / "tracking.json").write_text(json.dumps({
        "experiment_id": experiment_id,
        "status": status,
        "models": {
            "GRU_patient_540": {
                "patient_id": 540,
                "evaluation_results": {"mae": mae, "rmse": mae + 1},
            }
        },
    }), encoding="utf-8")


def _parent_tracking(path: Path, experiment_id: str, mean_mae: float):
    path.mkdir(parents=True)
    summary = {"mean": mean_mae, "std": 1.0, "min": mean_mae - 1,
               "max": mean_mae + 1, "n": 2}
    (path / "tracking.json").write_text(json.dumps({
        "experiment_id": experiment_id,
        "status": "completed",
        "models": {},
        "final_results": {
            "aggregate": [{
                "mode": "regular",
                "patient_id": 540,
                "model": "GRU",
                "seeds": [41, 42],
                "metrics": {"mae": summary, "clarke_zones.A": summary},
            }]
        },
    }), encoding="utf-8")


def test_list_reports_only_implemented_components(capsys):
    assert cli.main(["list"]) == 0
    output = capsys.readouterr().out
    assert "rnn, lstm, gru" in output
    assert "ohiot1dm" in output
    assert "transformer" not in output


def test_missing_config_has_nonzero_exit(capsys, tmp_path):
    assert cli.main(["run", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert "Configuration file not found" in capsys.readouterr().err


def test_analyze_selects_requested_metrics(capsys, tmp_path):
    experiment = tmp_path / "experiment"
    _tracking(experiment, "one", 12.5)
    assert cli.main(["analyze", "--experiment-dir", str(experiment), "--metrics", "mae"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["results"] == [{"model": "GRU_patient_540", "patient_id": 540, "mae": 12.5}]


def test_analyze_rejects_incomplete_experiment(capsys, tmp_path):
    experiment = tmp_path / "experiment"
    _tracking(experiment, "one", 12.5, status="error")
    assert cli.main(["analyze", "--experiment-dir", str(experiment)]) == 1
    assert "not completed" in capsys.readouterr().err


def test_analyze_reads_parent_multi_seed_aggregates(capsys, tmp_path):
    experiment = tmp_path / "experiment"
    _parent_tracking(experiment, "parent", 11.0)
    assert cli.main([
        "analyze", "--experiment-dir", str(experiment),
        "--metrics", "mae", "clarke_ega",
    ]) == 0
    row = json.loads(capsys.readouterr().out)["results"][0]
    assert row["seeds"] == [41, 42]
    assert row["mae"]["mean"] == 11.0
    assert row["clarke_ega"]["A"]["n"] == 2


def test_compare_requires_two_experiments(capsys, tmp_path):
    experiment = tmp_path / "experiment"
    _tracking(experiment, "one", 12.5)
    assert cli.main(["compare", "--experiments", str(experiment)]) == 2
    assert "at least two" in capsys.readouterr().err


def test_compare_writes_strict_comparison(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _tracking(first, "one", 12.5)
    _tracking(second, "two", 10.0)
    output = tmp_path / "comparison.json"
    assert cli.main([
        "compare", "--experiments", str(first), str(second), "--output", str(output)
    ]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert [item["experiment_id"] for item in result["experiments"]] == ["one", "two"]


def test_compare_ranks_parent_experiments_by_mean_mae(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _parent_tracking(first, "one", 12.5)
    _parent_tracking(second, "two", 10.0)
    output = tmp_path / "comparison.json"
    assert cli.main([
        "compare", "--experiments", str(first), str(second), "--output", str(output)
    ]) == 0
    ranking = json.loads(output.read_text(encoding="utf-8"))["mae_ranking"]
    assert [row["experiment_id"] for row in ranking] == ["two", "one"]
    assert ranking[0]["mae_mean"] == 10.0
