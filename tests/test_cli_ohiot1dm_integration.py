"""Real-data public-command integration test.

OhioT1DM cannot be redistributed, so contributors without an approved local
copy get a skip rather than a synthetic substitute.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PATIENT_FILE = REPO_ROOT / "data/raw/ohiot1dm/2020/train/540-ws-training.xml"


@pytest.mark.skipif(not PATIENT_FILE.is_file(), reason="OhioT1DM is not installed locally")
def test_public_command_creates_manifest_predictions_and_metrics(tmp_path):
    output_root = tmp_path / "results"
    config = {
        "schema_version": 1,
        "experiment": {"name": "cli_integration"},
        "data": {
            "dataset": "ohiot1dm",
            "root": str(REPO_ROOT / "data"),
            "version": "2020",
            "patients": [540],
            "train_ratio": 0.9,
            "validation_ratio": 0.1,
        },
        "preprocessing": {
            "window_size": 12,
            "prediction_horizon": 6,
            "sampling_rate": 5,
            "unimodal": True,
            "include_feature_engineering": True,
            "normalization": "standardize",
        },
        "model": {
            "type": "gru",
            "architecture": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0},
        },
        "training": {
            "mode": "regular",
            "epochs": 1,
            "pretrain_epochs": 1,
            "finetune_epochs": 1,
            "batch_size": 64,
            "learning_rate": 0.001,
            "early_stopping_patience": 1,
            "seeds": [41, 42],
            "device": "cpu",
        },
        "evaluation": {"metrics": ["mae", "rmse"]},
        "output": {
            "directory": str(output_root),
            "save_model": False,
            "save_predictions": True,
            "generate_plots": False,
            "export_format": ["json", "csv"],
        },
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    completed = subprocess.run(
        [sys.executable, "-m", "benchmark.cli", "run", "--config", str(config_path)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr

    experiment_dirs = list(output_root.glob("experiment_*"))
    assert len(experiment_dirs) == 1
    experiment_dir = experiment_dirs[0]
    tracking = json.loads((experiment_dir / "tracking.json").read_text(encoding="utf-8"))
    assert tracking["status"] == "completed"
    assert tracking["config"]["model"]["type"] == "gru"
    assert tracking["config"]["training"]["mode"] == "regular"
    assert (experiment_dir / "resolved_config.yaml").is_file()
    assert (experiment_dir / "aggregate_metrics.json").is_file()
    assert (experiment_dir / "aggregate_metrics.csv").is_file()
    aggregate = json.loads(
        (experiment_dir / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    assert aggregate[0]["seeds"] == [41, 42]
    assert aggregate[0]["metrics"]["mae"]["n"] == 2
    for seed in (41, 42):
        seed_dir = experiment_dir / f"regular/seed_{seed}"
        assert (seed_dir / "tracking.json").is_file()
        assert (seed_dir / "metrics.json").is_file()
        assert (seed_dir / "metrics.csv").is_file()
        assert (seed_dir / "patient_540/GRU_predictions.csv").is_file()
        # Each subrun must record the single seed it actually ran, so that the directory can be replayed on its own.
        resolved = yaml.safe_load(
            (seed_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        assert resolved["training"]["seeds"] == [seed]
        assert resolved["training"]["mode"] == "regular"
