"""Reusable train/test glucose covariates for the shift versus error analysis."""

import glob
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .distribution_shift import _clean_glucose, compute_distribution_shift
from .signal_features import compute_signal_features
from ..results_io import load_experiment_results, read_tracking
from ...configs.config_manager import OHIO_PATIENTS

TABLE_NAME = "dataset_signal_features.csv"
META_NAME = "dataset_signal_features.meta.json"
REQUIRED_COLUMNS = (
    "patient_id", "sampling_rate_minutes", "insulin_type", "train_std", "test_std",
    "train_tir", "test_tir", "train_n", "test_n", "shift_score",
    "js_divergence", "out_of_support_mass", "out_of_support_bins",
    "sample_entropy", "sample_entropy_valid_n", "sample_entropy_used_n",
    "sample_entropy_template_n", "autocorr_lag1", "autocorr_valid_n",
    "autocorr_adjacent_pair_n", "train_sample_entropy",
    "train_sample_entropy_valid_n", "train_sample_entropy_used_n",
    "train_sample_entropy_template_n", "train_autocorr_lag1",
    "train_autocorr_valid_n", "train_autocorr_adjacent_pair_n",
)


def resolved_cohort(config):
    data = config["data"]
    version = data["version"]
    patients = data["patients"]
    if patients == "all":
        patients = OHIO_PATIENTS[version]
    return sorted(int(pid) for pid in patients)


def signature(config):
    """Settings that determine the raw train/test series used by this table."""
    patients = resolved_cohort(config)
    releases = sorted(release for release in ("2018", "2020")
                      if set(patients) & set(OHIO_PATIENTS[release]))
    return {
        "dataset": config["data"]["dataset"],
        "releases": releases,
        "patient_ids": patients,
        "sampling_rate_minutes": config["preprocessing"]["sampling_rate"],
    }


def glucose_stats(arr):
    g = _clean_glucose(arr)
    if g.size == 0:
        return {"std": np.nan, "tir": np.nan, "n": 0}
    return {"std": float(np.std(g)),
            "tir": float(np.mean((g >= 70) & (g <= 180)) * 100.0),
            "n": int(g.size)}


def find_insulin_type(data_root, pid):
    for path in glob.glob(str(Path(data_root) / "raw/ohiot1dm" / "*" /
                              "train" / f"{pid}-ws-training.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as error:
            print(f"WARNING: could not parse insulin_type XML for patient {pid} ({path}): {error}")
            continue
        return root.attrib.get("insulin_type")
    return None


def compute_signal_table(patient_data, data_root):
    """Compute the same covariates previously built inside the experiment runner."""
    rows = []
    for pid, pair in sorted(patient_data.items()):
        train = pair["train"]["glucose"].to_numpy()
        test = pair["test"]["glucose"].to_numpy()
        shift = compute_distribution_shift(train, test)
        tr, te = glucose_stats(train), glucose_stats(test)
        test_signal = compute_signal_features(test)
        train_signal = compute_signal_features(train)
        rows.append({
            "patient_id": int(pid),
            "sampling_rate_minutes": pair["sampling_rate_minutes"],
            "insulin_type": find_insulin_type(data_root, pid),
            "train_std": tr["std"], "test_std": te["std"],
            "train_tir": tr["tir"], "test_tir": te["tir"],
            "train_n": shift["train_n"], "test_n": shift["test_n"],
            "shift_score": shift["shift_score"],
            "js_divergence": shift["js_divergence"],
            "out_of_support_mass": shift["out_of_support_mass"],
            "out_of_support_bins": shift["out_of_support_bins"],
            **test_signal,
            **{f"train_{name}": value for name, value in train_signal.items()},
        })
    return pd.DataFrame(rows)


def reference_signature(experiment_dir, mode=None, seed=None):
    loaded = load_experiment_results(experiment_dir, mode=mode, seed=seed,
                                     with_series=False)
    tracking = read_tracking(loaded.experiment_dir)
    if not tracking or "config" not in tracking:
        raise ValueError(f"{experiment_dir}: tracking.json has no config")
    value = signature(tracking["config"])
    if sorted(loaded.patients) != value["patient_ids"]:
        raise ValueError("reference run patients disagree with tracking config")
    return value


def write_signal_table(frame, output_dir, settings, data_root):
    output_dir = Path(output_dir)
    if frame.empty or frame.patient_id.duplicated().any():
        raise ValueError("signal table requires one row per patient")
    if sorted(frame.patient_id.astype(int).tolist()) != settings["patient_ids"]:
        raise ValueError("signal table does not cover the configured cohort")
    if not frame.sampling_rate_minutes.eq(settings["sampling_rate_minutes"]).all():
        raise ValueError("signal table sampling rate disagrees with configuration")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / TABLE_NAME, index=False)
    (output_dir / META_NAME).write_text(json.dumps({
        **settings, "data_root": str(Path(data_root).resolve()),
    }, indent=2) + "\n")


def read_signal_table(path, expected_settings, data_root=None):
    """Validate provenance and patient coverage before joining any model errors."""
    path = Path(path)
    meta_path = path.with_name(META_NAME)
    instruction = "Run RUN/run_dataset_analysis.sh signal first."
    if not path.is_file() or not meta_path.is_file():
        raise ValueError(f"Missing dataset signal table or metadata at {path}. {instruction}")
    try:
        metadata = json.loads(meta_path.read_text())
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ValueError(f"Cannot read dataset signal table: {error}. {instruction}") from error
    for key, expected in expected_settings.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Dataset signal table {key} differs from selected runs. {instruction}")
    if data_root is not None and metadata.get("data_root") != str(Path(data_root).resolve()):
        raise ValueError(f"Dataset signal table data root differs from selected runs. {instruction}")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset signal table missing columns {sorted(missing)}. {instruction}")
    if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        raise ValueError(f"Dataset signal table has missing or duplicate patients. {instruction}")
    try:
        numeric_ids = pd.to_numeric(frame.patient_id, errors="raise")
        if not np.isfinite(numeric_ids).all() or not np.equal(numeric_ids, numeric_ids.astype(int)).all():
            raise ValueError("non-integral patient ID")
        patient_ids = numeric_ids.astype(int).tolist()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Dataset signal table has invalid patient IDs. {instruction}") from error
    if sorted(patient_ids) != expected_settings["patient_ids"]:
        raise ValueError(f"Dataset signal table patient rows differ from selected runs. {instruction}")
    if not frame.sampling_rate_minutes.eq(expected_settings["sampling_rate_minutes"]).all():
        raise ValueError(f"Dataset signal table row sampling rates differ from selected runs. {instruction}")
    return {int(row["patient_id"]): row for row in frame.to_dict(orient="records")}
