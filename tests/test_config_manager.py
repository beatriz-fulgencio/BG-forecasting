from pathlib import Path

import pytest
import yaml

from benchmark.configs.config_manager import ConfigError, load_config, validate_data_files


def _config(root: Path, **training):
    return {
        "schema_version": 1,
        "experiment": {"name": "test"},
        "data": {
            "dataset": "ohiot1dm",
            "root": str(root),
            "version": "2020",
            "patients": [540, 544],
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
            "batch_size": 4,
            "learning_rate": 0.001,
            "early_stopping_patience": 1,
            "seeds": [7],
            "device": "cpu",
            **training,
        },
        "evaluation": {"metrics": ["mae", "clarke_ega"]},
        "output": {
            "directory": str(root / "results"),
            "save_model": False,
            "save_predictions": False,
            "generate_plots": False,
            "export_format": ["json"],
        },
    }


def _write_config(tmp_path: Path, config) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_config_returns_typed_normalized_config(tmp_path):
    config = load_config(_write_config(tmp_path, _config(tmp_path)))
    assert config.schema_version == 1
    assert config.model.type == "gru"
    assert config.training.mode == "regular"
    assert config.training.seeds == [7]
    assert config.training.learning_rate == pytest.approx(0.001)
    assert config.data.patient_ids() == [540, 544]


def test_combined_release_selection_resolves_all_twelve_patients(tmp_path):
    raw = _config(tmp_path)
    raw["data"].update(version="both", patients="all")
    config = load_config(_write_config(tmp_path, raw))
    assert len(config.data.patient_ids()) == 12
    assert config.data.version_for_patient(559) == "2018"
    assert config.data.version_for_patient(540) == "2020"


def test_combined_release_preflight_checks_each_patient_source(tmp_path):
    raw = _config(tmp_path)
    raw["data"].update(version="both", patients=[559, 540])
    for patient_id, version in [(559, "2018"), (540, "2020")]:
        for mode, suffix in (("train", "training"), ("test", "testing")):
            path = tmp_path / "raw" / "ohiot1dm" / version / mode / f"{patient_id}-ws-{suffix}.xml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    config = load_config(_write_config(tmp_path, raw))
    assert validate_data_files(config) == [559, 540]


def test_training_mode_is_required(tmp_path):
    raw = _config(tmp_path)
    del raw["training"]["mode"]
    with pytest.raises(ConfigError, match="training.mode is required"):
        load_config(_write_config(tmp_path, raw))


def test_seeds_are_required(tmp_path):
    raw = _config(tmp_path)
    del raw["training"]["seeds"]
    with pytest.raises(ConfigError, match="training.seeds is required"):
        load_config(_write_config(tmp_path, raw))


def test_seeds_must_be_unique_non_negative_integers(tmp_path):
    with pytest.raises(ConfigError, match="must not contain duplicates"):
        load_config(_write_config(tmp_path, _config(tmp_path, seeds=[7, 7])))
    with pytest.raises(ConfigError, match=r"training.seeds\[1\]"):
        load_config(_write_config(tmp_path, _config(tmp_path, seeds=[7, -1])))


def test_legacy_seed_key_is_rejected(tmp_path):
    raw = _config(tmp_path)
    raw["training"]["seed"] = raw["training"].pop("seeds")[0]
    with pytest.raises(ConfigError, match="seed"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("mode", ["regular", "transfer", "both"])
def test_all_training_modes_are_supported(tmp_path, mode):
    config = load_config(_write_config(tmp_path, _config(tmp_path, mode=mode)))
    assert config.training.mode == mode


def test_unknown_keys_fail_instead_of_being_ignored(tmp_path):
    raw = _config(tmp_path)
    raw["training"]["epocs"] = 3
    with pytest.raises(ConfigError, match="epocs"):
        load_config(_write_config(tmp_path, raw))


def test_transfer_requires_two_patients(tmp_path):
    raw = _config(tmp_path, mode="transfer")
    raw["data"]["patients"] = [540]
    with pytest.raises(ConfigError, match="at least two"):
        load_config(_write_config(tmp_path, raw))


def test_data_preflight_reports_expected_missing_xml(tmp_path):
    config = load_config(_write_config(tmp_path, _config(tmp_path)))
    with pytest.raises(ConfigError) as error:
        validate_data_files(config)
    message = str(error.value)
    assert "OhioT1DM data is required" in message
    assert "540-ws-training.xml" in message
    assert "540-ws-testing.xml" in message


def test_data_preflight_resolves_selected_patients(tmp_path):
    for patient_id in (540, 544):
        for mode, suffix in (("train", "training"), ("test", "testing")):
            path = tmp_path / "raw" / "ohiot1dm" / "2020" / mode / f"{patient_id}-ws-{suffix}.xml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    config = load_config(_write_config(tmp_path, _config(tmp_path)))
    assert validate_data_files(config) == [540, 544]


def test_multiple_seeds_load_in_declared_order(tmp_path):
    config = load_config(_write_config(tmp_path, _config(tmp_path, seeds=[43, 7, 101])))
    assert config.training.seeds == [43, 7, 101]


@pytest.mark.parametrize("seeds", [42, "42", {}, [], [7, "8"], [7, 8.0], [7, True], [7, None]])
def test_seeds_must_be_a_non_empty_list_of_integers(tmp_path, seeds):
    with pytest.raises(ConfigError, match="seeds"):
        load_config(_write_config(tmp_path, _config(tmp_path, seeds=seeds)))


def test_zero_is_an_acceptable_seed(tmp_path):
    config = load_config(_write_config(tmp_path, _config(tmp_path, seeds=[0, 1])))
    assert config.training.seeds == [0, 1]


def test_seeds_above_the_rng_limit_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match=r"training.seeds\[1\] must be at most 4294967295"):
        load_config(_write_config(tmp_path, _config(tmp_path, seeds=[7, 2 ** 32])))


def test_largest_usable_seed_is_accepted(tmp_path):
    config = load_config(_write_config(tmp_path, _config(tmp_path, seeds=[2 ** 32 - 1])))
    assert config.training.seeds == [2 ** 32 - 1]
