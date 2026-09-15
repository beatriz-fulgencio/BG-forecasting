import json

from benchmark.experiments.tracking import ExperimentTracker


def test_tracking_preserves_integer_and_boolean_types(tmp_path):
    tracker = ExperimentTracker(tmp_path, config={"schema_version": 1, "enabled": True})
    tracker.log_data_params({"patient_ids": [540, 544]})
    data = json.loads((tracker.get_experiment_dir() / "tracking.json").read_text())
    assert data["config"]["schema_version"] == 1
    assert isinstance(data["config"]["schema_version"], int)
    assert data["config"]["enabled"] is True
    assert data["data_params"]["patient_ids"] == [540, 544]


def test_identical_configs_get_distinct_experiment_ids(tmp_path):
    """A parent and its only subrun share a config, so the ID must still differ."""
    config = {"training": {"mode": "regular", "seeds": [42]}}
    parent = ExperimentTracker(tmp_path, config=config)
    child = ExperimentTracker(
        tmp_path / "regular", config=config, experiment_dir=tmp_path / "regular/seed_42"
    )
    assert parent.experiment_id != child.experiment_id


def test_repeated_runs_of_one_config_do_not_share_a_directory(tmp_path):
    config = {"training": {"mode": "regular", "seeds": [42]}}
    first = ExperimentTracker(tmp_path, config=config)
    second = ExperimentTracker(tmp_path, config=config)
    assert first.experiment_dir != second.experiment_dir
    assert len(list(tmp_path.glob("experiment_*"))) == 2
