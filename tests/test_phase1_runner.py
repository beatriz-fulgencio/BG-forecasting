"""Unit tests for the RUN/run_phase1_shift_analysis.py helper logic.

These cover the parts most likely to break silently: experiment discovery
(flat vs nested layout), the ambiguity guard, MAE parsing, table assembly, and
the correlation/regression drivers. They avoid the heavy torch-backed data
loaders (which the module imports lazily), so the suite stays fast.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_phase1_shift_analysis as r


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _write_metrics(path: Path, mae_by_patient_model: dict):
    """Write a minimal comprehensive_metrics JSON at ``path``.

    ``mae_by_patient_model`` maps ``{patient_id: {model: mae}}``.
    """
    patient_metrics = {
        str(pid): {model: {"mae": mae} for model, mae in models.items()}
        for pid, models in mae_by_patient_model.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"patient_metrics": patient_metrics}))


def _make_experiment(root: Path, name: str, maes: dict, nested: bool = False):
    """Create an experiment dir with a metrics JSON (flat or nested layout)."""
    exp = root / name
    if nested:
        _write_metrics(exp / "run_abc123" / "comprehensive_metrics_20260101_000000.json", maes)
    else:
        _write_metrics(exp / "comprehensive_metrics_20260101_000000.json", maes)
    return exp


# --------------------------------------------------------------------------- #
# parse_horizon
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected", [
    ("experiment_20251104_034100_30min_tl", 30),
    ("experiment_x_15min_rl", 15),
    ("foo_60 min_tl", 60),
    ("no_horizon_here", None),
])
def test_parse_horizon(name, expected):
    assert r.parse_horizon(name) == expected


# --------------------------------------------------------------------------- #
# _find_metrics_json: flat and nested layouts
# --------------------------------------------------------------------------- #
def test_find_metrics_json_flat(tmp_path):
    exp = _make_experiment(tmp_path, "exp_30min_tl", {540: {"GRU": 1.0}}, nested=False)
    found = r._find_metrics_json(exp)
    assert found is not None and found.parent == exp


def test_find_metrics_json_nested(tmp_path):
    exp = _make_experiment(tmp_path, "exp_30min_tl", {540: {"GRU": 1.0}}, nested=True)
    found = r._find_metrics_json(exp)
    assert found is not None and found.parent.parent == exp


def test_find_metrics_json_missing(tmp_path):
    (tmp_path / "empty").mkdir()
    assert r._find_metrics_json(tmp_path / "empty") is None


# --------------------------------------------------------------------------- #
# _read_patient_mae
# --------------------------------------------------------------------------- #
def test_read_patient_mae_filters_model_and_missing(tmp_path):
    metrics = tmp_path / "m.json"
    _write_metrics(metrics, {
        540: {"GRU": 10.0, "LSTM": 9.0},
        544: {"GRU": None},          # missing mae -> dropped
        552: {"LSTM": 8.0},          # no GRU -> dropped for GRU
    })
    got = r._read_patient_mae(metrics, "GRU")
    assert got == {540: 10.0}


def test_read_patient_mae_ignores_non_integer_keys(tmp_path):
    metrics = tmp_path / "m.json"
    metrics.write_text(json.dumps({"patient_metrics": {
        "540": {"GRU": {"mae": 10.0}},
        "population": {"GRU": {"mae": 5.0}},  # non-int id -> skipped
    }}))
    assert r._read_patient_mae(metrics, "GRU") == {540: 10.0}


# --------------------------------------------------------------------------- #
# load_mae_by_horizon: happy path + ambiguity guard
# --------------------------------------------------------------------------- #
def test_load_mae_by_horizon_one_per_horizon(tmp_path):
    _make_experiment(tmp_path, "exp_15min_tl", {540: {"GRU": 5.0}})
    _make_experiment(tmp_path, "exp_30min_tl", {540: {"GRU": 10.0}}, nested=True)
    # An rl experiment must be ignored when mode='tl'.
    _make_experiment(tmp_path, "exp_30min_rl", {540: {"GRU": 99.0}})

    out = r.load_mae_by_horizon(str(tmp_path), mode="tl", model="GRU")
    assert out == {15: {540: 5.0}, 30: {540: 10.0}}


def test_load_mae_by_horizon_raises_on_duplicate_horizon(tmp_path):
    _make_experiment(tmp_path, "expA_30min_tl", {540: {"GRU": 10.0}})
    _make_experiment(tmp_path, "expB_30min_tl", {540: {"GRU": 11.0}})
    with pytest.raises(r.AmbiguousExperimentsError) as exc:
        r.load_mae_by_horizon(str(tmp_path), mode="tl", model="GRU")
    msg = str(exc.value)
    assert "30min" in msg and "expA_30min_tl" in msg and "expB_30min_tl" in msg


def test_load_mae_by_horizon_empty_when_no_match(tmp_path):
    _make_experiment(tmp_path, "exp_30min_rl", {540: {"GRU": 10.0}})
    assert r.load_mae_by_horizon(str(tmp_path), mode="tl", model="GRU") == {}


# --------------------------------------------------------------------------- #
# glucose_stats
# --------------------------------------------------------------------------- #
def test_glucose_stats_basic():
    # _clean_glucose keeps [GLUCOSE_MIN, GLUCOSE_MAX] = [20, 600]; 700 and the -1
    # sentinel are dropped, 250 is kept. TIR counts readings in [70, 180].
    s = r.glucose_stats(np.array([70, 120, 180, 250, 700, -1], dtype=float))
    assert s["n"] == 4                       # 70, 120, 180, 250
    assert s["tir"] == pytest.approx(75.0)   # 3 of 4 in [70, 180]
    assert s["std"] >= 0


def test_glucose_stats_empty():
    s = r.glucose_stats(np.array([-1, -1], dtype=float))
    assert s["n"] == 0 and np.isnan(s["std"]) and np.isnan(s["tir"])


# --------------------------------------------------------------------------- #
# build_tables
# --------------------------------------------------------------------------- #
def _feats(pid, shift, entropy=0.3, ac=0.99):
    return {
        "patient_id": pid, "insulin_type": "Humalog",
        "train_std": 50.0, "test_std": 55.0,
        "train_tir": 60.0, "test_tir": 65.0,
        "train_n": 1000, "test_n": 500,
        "shift_score": shift, "kl_divergence": 0.1,
        "sample_entropy": entropy, "autocorr_lag1": ac,
    }


def test_build_tables_shapes_and_columns():
    feats = {540: _feats(540, 10.0), 544: _feats(544, 20.0)}
    mae_by_horizon = {15: {540: 5.0, 544: 6.0}, 30: {540: 9.0, 544: 11.0}}
    wide, long, horizons = r.build_tables(feats, mae_by_horizon)

    assert horizons == [15, 30]
    assert len(wide) == 2
    assert {"mae_15", "mae_30", "sample_entropy", "autocorr_lag1"} <= set(wide.columns)
    # long: 2 patients x 2 horizons = 4 rows, with signal features carried through.
    assert len(long) == 4
    assert {"sample_entropy", "autocorr_lag1", "horizon", "mae"} <= set(long.columns)


def test_build_tables_drops_rows_missing_mae():
    feats = {540: _feats(540, 10.0), 544: _feats(544, 20.0)}
    mae_by_horizon = {30: {540: 9.0}}  # 544 absent -> NaN mae -> dropped from long
    _, long, _ = r.build_tables(feats, mae_by_horizon)
    assert long["patient_id"].tolist() == [540]


# --------------------------------------------------------------------------- #
# run_correlations
# --------------------------------------------------------------------------- #
def _long_two_horizons(seed=0, n=10):
    rng = np.random.default_rng(seed)
    rows = []
    for h, scale in [(15, 1.0), (60, 4.0)]:
        for pid in range(n):
            shift = 5 + pid * 2.0
            mae = scale * (2 + 0.1 * shift + rng.normal(0, 0.3))
            rows.append({
                "patient_id": pid, "horizon": h, "shift_score": shift,
                "kl_divergence": 0.1, "test_std": 50.0 + pid, "test_tir": 60.0,
                "train_std": 50.0, "train_tir": 60.0,
                "sample_entropy": 0.3 + 0.01 * pid, "autocorr_lag1": 0.99,
                "mae": mae,
            })
    return pd.DataFrame(rows)


def test_run_correlations_has_within_horizon_row():
    long = _long_two_horizons()
    corr = r.run_correlations(long, horizons=[15, 60])
    assert "pooled" in corr["horizon"].values
    assert "pooled_within_horizon_z" in corr["horizon"].values
    # Standardizing within horizon should not be weaker than the raw pooled r here
    # (raw pooling mixes the 4x scale difference between horizons).
    raw = corr.loc[corr.horizon == "pooled", "pearson_r"].iloc[0]
    zed = corr.loc[corr.horizon == "pooled_within_horizon_z", "pearson_r"].iloc[0]
    assert zed >= raw


def test_run_correlations_single_horizon_pooled_equals_horizon():
    long = _long_two_horizons()
    long = long[long.horizon == 15]
    corr = r.run_correlations(long, horizons=[15])
    r15 = corr.loc[corr.horizon == 15, "pearson_r"].iloc[0]
    rz = corr.loc[corr.horizon == "pooled_within_horizon_z", "pearson_r"].iloc[0]
    assert r15 == pytest.approx(rz)


# --------------------------------------------------------------------------- #
# run_regressions (exercises the lazy statsmodels import)
# --------------------------------------------------------------------------- #
def test_run_regressions_multi_horizon_reports_interaction():
    long = _long_two_horizons()
    txt = r.run_regressions(long, horizons=[15, 60])
    assert "[15 min]" in txt and "[60 min]" in txt
    assert "pooled interaction" in txt
    assert "shift_score:horizon" in txt
    assert "pooled + signal" in txt


def test_run_regressions_single_horizon_skips_interaction():
    long = _long_two_horizons()
    long = long[long.horizon == 15]
    txt = r.run_regressions(long, horizons=[15])
    assert "skipped (needs >= 2 horizons" in txt
