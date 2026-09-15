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


def test_configured_metrics_use_saved_horizon_and_reject_duplicate_runs(tmp_path):
    def configured(path, horizon, mae):
        path.write_text(json.dumps({"540": {
            "mae": mae,
            "model_info": {"model_name": "GRU", "mode": "regular",
                           "target_units": "mg/dL",
                           "prediction_horizon_minutes": horizon},
        }}))
    p15 = tmp_path / "run_a.json"
    p30 = tmp_path / "run_b.json"
    configured(p15, 15, 5.0)
    configured(p30, 30, 9.0)
    assert r.load_configured_mae_by_horizon([p15, p30], "GRU", "regular") == {
        15: {540: 5.0}, 30: {540: 9.0},
    }
    with pytest.raises(r.AmbiguousExperimentsError):
        r.load_configured_mae_by_horizon([p15, p15], "GRU", "regular")


def test_aggregate_loader_and_paired_transfer_benefit(tmp_path):
    parent = tmp_path / "run_30min"
    parent.mkdir()
    config = {
        "model": {"type": "gru"},
        "data": {"dataset": "ohiot1dm", "version": "2020", "patients": [540, 544]},
        "preprocessing": {"prediction_horizon": 6, "sampling_rate": 5},
    }
    (parent / "tracking.json").write_text(json.dumps({"config": config}))
    rows = []
    for mode, maes in [("regular", {540: 10.0, 544: 12.0}),
                       ("transfer", {540: 8.0, 544: 11.0})]:
        for pid, mae in maes.items():
            rows.append({"mode": mode, "patient_id": pid, "model": "GRU",
                         "seeds": [42, 7], "metrics": {"mae": {"mean": mae}}})
    aggregate = parent / "aggregate_metrics.json"
    aggregate.write_text(json.dumps(rows))
    by_mode, seeds = r.load_aggregate_mae_by_horizon([aggregate], "GRU")
    feats = {540: _feats(540, 10), 544: _feats(544, 20)}
    benefit, long, horizons = r.build_transfer_benefit_long(feats, by_mode, seeds)
    assert horizons == [30]
    row = benefit.set_index("patient_id").loc[540]
    assert row["transfer_benefit_mg_dl"] == 2.0
    # The screening outcome is the same number named for its own direction, so
    # the two CSVs never disagree by an unstated sign.
    assert row["transfer_minus_regular_mae"] == -2.0
    assert long.set_index("patient_id").loc[540, "mae"] == -2.0
    assert benefit.matched_seeds.eq("7,42").all()
    with pytest.raises(ValueError, match="different completed seeds"):
        r.build_transfer_benefit_long(feats, by_mode,
                                      {**seeds, ("transfer", 30, 540): (42,)})


@pytest.mark.parametrize("missing_mode", ["regular", "transfer"])
def test_transfer_benefit_rejects_missing_patient_mode_pair(missing_mode):
    feats = {540: _feats(540, 10), 544: _feats(544, 20)}
    by_mode = {
        "regular": {30: {540: 10.0, 544: 12.0}},
        "transfer": {30: {540: 8.0, 544: 11.0}},
    }
    seeds = {(mode, 30, pid): (7, 42) for mode in by_mode for pid in feats}
    del by_mode[missing_mode][30][544]
    with pytest.raises(ValueError, match=rf"30 min patient 544: missing {missing_mode} MAE"):
        r.build_transfer_benefit_long(feats, by_mode, seeds)


def test_transfer_benefit_rejects_missing_glucose_features():
    by_mode = {
        "regular": {30: {540: 10.0, 544: 12.0}},
        "transfer": {30: {540: 8.0, 544: 11.0}},
    }
    seeds = {(mode, 30, pid): (42,) for mode in by_mode for pid in (540, 544)}
    with pytest.raises(ValueError, match="30 min patient 544: missing glucose features"):
        r.build_transfer_benefit_long({540: _feats(540, 10)}, by_mode, seeds)


@pytest.mark.parametrize("missing_modes", [["transfer"], ["regular", "transfer"]])
def test_configured_transfer_comparison_requires_declared_cohort(tmp_path, missing_modes):
    parent = tmp_path / "run_30min"
    parent.mkdir()
    (parent / "tracking.json").write_text(json.dumps({"config": {
        "model": {"type": "gru"},
        "data": {"dataset": "ohiot1dm", "version": "2020", "patients": [540, 544]},
        "preprocessing": {"prediction_horizon": 6, "sampling_rate": 5},
    }}))
    rows = []
    for mode in ("regular", "transfer"):
        for pid in (540, 544):
            if pid == 544 and mode in missing_modes:
                continue
            rows.append({"mode": mode, "patient_id": pid, "model": "GRU",
                         "seeds": [42], "metrics": {"mae": {"mean": 10.0}}})
    aggregate = parent / "aggregate_metrics.json"
    aggregate.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="30 min patient 544: missing"):
        r.load_aggregate_mae_by_horizon([aggregate], "GRU", require_transfer_pairs=True)
    # Ordinary single-mode analysis is unchanged.
    r.load_aggregate_mae_by_horizon([aggregate], "GRU")


def test_aggregate_loader_merges_disjoint_ohio_versions_per_horizon(tmp_path):
    paths = []
    for version, pid in [("2020", 540), ("2018", 559)]:
        for horizon_steps in [6, 12]:
            parent = tmp_path / f"run_{version}_{horizon_steps}"
            parent.mkdir()
            config = {
                "model": {"type": "gru"},
                "data": {"dataset": "ohiot1dm", "version": version, "patients": [pid]},
                "preprocessing": {"prediction_horizon": horizon_steps, "sampling_rate": 5},
            }
            (parent / "tracking.json").write_text(json.dumps({"config": config}))
            aggregate = parent / "aggregate_metrics.json"
            aggregate.write_text(json.dumps([{
                "mode": "regular", "patient_id": pid, "model": "GRU",
                "seeds": [42], "metrics": {"mae": {"mean": 10.0}},
            }]))
            paths.append(aggregate)
    by_mode, _ = r.load_aggregate_mae_by_horizon(paths, "GRU")
    assert set(by_mode["regular"][30]) == {540, 559}
    assert set(by_mode["regular"][60]) == {540, 559}
    with pytest.raises(ValueError, match="cohort differs between horizons"):
        r.load_aggregate_mae_by_horizon(paths[:-1], "GRU")


def test_configured_sampling_rate_is_shared_across_selected_runs(tmp_path):
    paths = []
    for name, rate in [("h15", 10), ("h30", 10)]:
        parent = tmp_path / name
        parent.mkdir()
        (parent / "tracking.json").write_text(json.dumps({
            "config": {"preprocessing": {"sampling_rate": rate}}
        }))
        paths.append(parent / "aggregate_metrics.json")
    assert r.configured_sampling_rate(paths) == 10
    (tmp_path / "h30" / "tracking.json").write_text(json.dumps({
        "config": {"preprocessing": {"sampling_rate": 5}}
    }))
    with pytest.raises(ValueError, match="different sampling rates"):
        r.configured_sampling_rate(paths)


def test_glucose_feature_loader_uses_selected_sampling_rate(monkeypatch, tmp_path):
    from benchmark.data import loaders, preprocessors

    seen = []
    glucose = pd.DataFrame({"glucose": 120 + np.sin(np.linspace(0, 8, 100))})

    def fake_load(root, *, patient_ids, mode, version, sampling_rate):
        seen.append((mode, sampling_rate))
        return {540: glucose}

    class FakePreprocessor:
        def __init__(self, sampling_rate):
            seen.append(("preprocessor", sampling_rate))

        def basic_preprocessing(self, frame):
            return frame

    monkeypatch.setattr(loaders, "load_ohiot1dm_data", fake_load)
    monkeypatch.setattr(preprocessors, "OhioBGDataPreprocessor", FakePreprocessor)
    features = r.load_glucose_features(tmp_path, [540], sampling_rate=10)
    assert seen == [("train", 10), ("test", 10), ("preprocessor", 10)]
    assert features[540]["sampling_rate_minutes"] == 10


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
        "train_sample_entropy": entropy, "train_autocorr_lag1": ac,
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
                "train_sample_entropy": 0.2 + 0.02 * pid,
                "train_autocorr_lag1": 0.98,
                "mae": mae,
            })
    return pd.DataFrame(rows)


def test_run_correlations_has_within_horizon_row():
    long = _long_two_horizons()
    corr = r.run_correlations(long, horizons=[15, 60])
    assert "patient_mean_within_horizon_z" in corr["horizon"].values
    assert "pooled" not in corr["horizon"].values
    combined = corr.loc[corr.horizon == "patient_mean_within_horizon_z"].iloc[0]
    patient = r.patient_difficulty_table(long, [15, 60])
    assert combined.n_patients == 10  # 20 repeated rows are only 10 patients
    assert combined.pearson_r == pytest.approx(
        r.pearsonr(patient.shift_score, patient.difficulty_z).statistic
    )


def test_run_correlations_single_horizon_patient_score_equals_horizon():
    long = _long_two_horizons()
    long = long[long.horizon == 15]
    corr = r.run_correlations(long, horizons=[15])
    r15 = corr.loc[corr.horizon == 15, "pearson_r"].iloc[0]
    rz = corr.loc[corr.horizon == "patient_mean_within_horizon_z", "pearson_r"].iloc[0]
    assert r15 == pytest.approx(rz)


def test_patient_difficulty_excludes_missing_horizon():
    long = _long_two_horizons()
    long = long[~((long.patient_id == 0) & (long.horizon == 60))]
    patient = r.patient_difficulty_table(long, [15, 60])
    assert len(patient) == 9
    assert 0 not in patient.patient_id.values


def test_repeating_horizon_does_not_create_extra_independent_patients():
    one = _long_two_horizons().query("horizon == 15").copy()
    repeated = pd.concat([one, one.assign(horizon=30),
                          one.assign(horizon=45), one.assign(horizon=60)])
    first = r.run_correlations(one, [15]).iloc[-1]
    four = r.run_correlations(repeated, [15, 30, 45, 60]).iloc[-1]
    assert four.n_patients == first.n_patients == 10
    assert four.pearson_p == pytest.approx(first.pearson_p)


def test_no_complete_patient_reports_missing_combined_result():
    long = _long_two_horizons().query("horizon == 15")
    combined = r.run_correlations(long, [15, 60]).iloc[-1]
    assert combined.n_patients == 0
    assert np.isnan(combined.pearson_p)


def test_screen_comparison_uses_matched_patients_and_reports_availability():
    long = _long_two_horizons(n=12)
    long.loc[long.patient_id == 0, "train_sample_entropy"] = np.nan
    screens = r.compare_difficulty_screens(long, [15, 60])
    combined = screens[screens.horizon == "patient_mean_within_horizon_z"]
    assert set(combined.screen) == {"shift_score", "sample_entropy", "autocorr_lag1",
                                    "train_sample_entropy", "train_autocorr_lag1"}
    assert combined.n_patients.eq(11).all()
    assert combined.top_quartile_n.eq(3).all()
    # The hit rate is only readable against what a random screen would score.
    assert combined.top_quartile_chance_rate.eq(3 / 11).all()
    assert combined.loc[combined.screen == "train_sample_entropy",
                        "data_available"].iloc[0] == "train"
    ranks = r.rank_difficulty_screens(long, [15, 60])
    combined_ranks = ranks[ranks.horizon == "patient_mean_within_horizon_z"]
    assert combined_ranks.patient_id.nunique() == 11
    assert len(combined_ranks) == 11 * 5
    assert combined_ranks.outcome_units.eq("within_horizon_z").all()
    assert combined_ranks.outcome_name.eq("mae_within_horizon_z").all()


def test_both_irregularity_features_have_a_pre_training_twin():
    """Review-Points 2.2 asks whether irregularity computed before training
    matches shift; that needs the train-only twin of *both* features."""
    train_only = {name for name, (available, _) in r._SCREEN_CANDIDATES.items()
                  if available == "train"}
    assert train_only == {"train_sample_entropy", "train_autocorr_lag1"}
    for test_name, train_name in [("sample_entropy", "train_sample_entropy"),
                                  ("autocorr_lag1", "train_autocorr_lag1")]:
        assert r._SCREEN_CANDIDATES[test_name][1] == r._SCREEN_CANDIDATES[train_name][1]


def test_transfer_benefit_screens_name_their_own_outcome():
    long = _long_two_horizons(n=12).rename(columns={"mae": "transfer_minus_regular_mae"})
    long["mae"] = long["transfer_minus_regular_mae"]
    screens = r.compare_difficulty_screens(long, [15, 60], target="least_transfer_benefit")
    per_horizon = screens[screens.horizon == "15"]
    assert per_horizon.outcome_name.eq("transfer_minus_regular_mae").all()
    assert per_horizon.outcome_units.eq("mg/dL").all()
    combined = screens[screens.horizon == "patient_mean_within_horizon_z"]
    assert combined.outcome_name.eq("transfer_minus_regular_mae_within_horizon_z").all()


def test_unknown_screening_target_is_rejected():
    with pytest.raises(ValueError, match="Unknown screening target"):
        r.compare_difficulty_screens(_long_two_horizons(), [15, 60], target="nonsense")


# --------------------------------------------------------------------------- #
# run_regressions (exercises the lazy statsmodels import)
# --------------------------------------------------------------------------- #
def test_run_regressions_multi_horizon_uses_patients():
    long = _long_two_horizons()
    txt = r.run_regressions(long, horizons=[15, 60])
    assert "[15 min]" in txt and "[60 min]" in txt
    assert "[patient horizon trend]" in txt
    assert "n_patients=10" in txt
    assert "Five predictors on ≈12 patients" in txt
    assert "[patient + signal]" in txt
    assert "[pooled" not in txt


def test_run_regressions_single_horizon_skips_interaction():
    long = _long_two_horizons()
    long = long[long.horizon == 15]
    txt = r.run_regressions(long, horizons=[15])
    assert "[patient horizon trend] skipped (needs >= 2 horizons" in txt


# --------------------------------------------------------------------------- #
# --mode / result-source pairing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode,configured,hint", [
    ("tl", True, "--mode transfer"),
    ("rl", True, "--mode regular"),
    ("transfer", False, "--mode tl"),
    ("regular", False, "--mode rl"),
])
def test_mode_spelling_must_match_the_selected_result_source(mode, configured, hint):
    with pytest.raises(ValueError, match="--mode") as excinfo:
        r._check_mode_matches_source(mode, configured)
    assert hint in str(excinfo.value)


@pytest.mark.parametrize("mode,configured", [
    ("tl", False), ("rl", False), ("regular", True), ("transfer", True),
])
def test_matching_mode_and_source_is_accepted(mode, configured):
    assert r._check_mode_matches_source(mode, configured) is None


# --------------------------------------------------------------------------- #
# per-patient release resolution
# --------------------------------------------------------------------------- #
def _aggregate_parent(tmp_path, name, version, patients, steps, pids):
    parent = tmp_path / name
    parent.mkdir()
    (parent / "tracking.json").write_text(json.dumps({"config": {
        "model": {"type": "gru"},
        "data": {"dataset": "ohiot1dm", "version": version, "patients": patients},
        "preprocessing": {"prediction_horizon": steps, "sampling_rate": 5},
    }}))
    aggregate = parent / "aggregate_metrics.json"
    aggregate.write_text(json.dumps([
        {"mode": "regular", "patient_id": pid, "model": "GRU",
         "seeds": [42], "metrics": {"mae": {"mean": 10.0}}}
        for pid in pids
    ]))
    return aggregate


ALL_OHIO = [540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596]
OHIO_2018 = [559, 563, 570, 575, 588, 591]
OHIO_2020 = [540, 544, 552, 567, 584, 596]


def test_combined_release_parent_mixes_with_split_parents(tmp_path):
    # Same 12-patient cohort at both horizons, reached two different ways. The
    # release comes from each patient ID, so the version labels need not match.
    paths = [
        _aggregate_parent(tmp_path, "h30_both", "both", "all", 6, ALL_OHIO),
        _aggregate_parent(tmp_path, "h60_2018", "2018", OHIO_2018, 12, OHIO_2018),
        _aggregate_parent(tmp_path, "h60_2020", "2020", OHIO_2020, 12, OHIO_2020),
    ]
    by_mode, _ = r.load_aggregate_mae_by_horizon(paths, "GRU")
    assert sorted(by_mode["regular"][30]) == ALL_OHIO
    assert sorted(by_mode["regular"][60]) == ALL_OHIO


def test_patient_from_the_wrong_release_is_rejected(tmp_path):
    path = _aggregate_parent(tmp_path, "h30_bad", "2018", [559, 540], 6, [559, 540])
    with pytest.raises(ValueError, match="patient 540 belongs to OhioT1DM 2020"):
        r.load_aggregate_mae_by_horizon([path], "GRU")


def test_unknown_patient_id_is_rejected(tmp_path):
    path = _aggregate_parent(tmp_path, "h30_unknown", "both", [999], 6, [999])
    with pytest.raises(ValueError, match="999 is not an OhioT1DM patient ID"):
        r.load_aggregate_mae_by_horizon([path], "GRU")
