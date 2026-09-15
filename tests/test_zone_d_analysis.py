"""Zone-D decomposition must treat patients, not prediction rows, as units."""

import json

import pandas as pd
import pytest

from RUN.run_zone_d_analysis import (
    analyze_run,
    analyze_runs,
    patient_bootstrap,
    patient_zone_d_counts,
)


def test_zone_d_decomposition_uses_clarke_classes():
    frame = pd.DataFrame({
        "true_glucose_mg_dl": [60, 250, 100, 60],
        "predicted_glucose_mg_dl": [100, 150, 100, 60],
    })
    counts = patient_zone_d_counts(frame, 540)
    assert counts == {
        "patient_id": 540,
        "predictions": 4,
        "zone_d": 2,
        "missed_hypoglycemia": 1,
        "missed_hyperglycemia": 1,
    }


def test_archived_reference_rounding_at_sensor_boundary_is_tolerated():
    frame = pd.DataFrame({
        "true_glucose_mg_dl": [400.000009],
        "predicted_glucose_mg_dl": [300.0],
    })
    assert patient_zone_d_counts(frame, 540)["predictions"] == 1
    frame.loc[0, "true_glucose_mg_dl"] = 401.0
    with pytest.raises(ValueError, match="CGM range"):
        patient_zone_d_counts(frame, 540)


def test_bootstrap_draws_entire_patients_and_is_reproducible():
    counts = [
        {"patient_id": 540, "predictions": 4, "zone_d": 2, "missed_hypoglycemia": 2},
        {"patient_id": 559, "predictions": 6, "zone_d": 1, "missed_hypoglycemia": 0},
    ]
    summary = patient_bootstrap([counts], replicates=500, bootstrap_seed=7)
    assert summary == patient_bootstrap([counts], replicates=500, bootstrap_seed=7)
    assert summary["zone_d_rate"] == pytest.approx(0.3)
    assert summary["missed_hypoglycemia_share_of_zone_d"] == pytest.approx(2 / 3)
    # With two whole-patient draws, the extremes are the all-540 and all-559
    # samples. A row-level bootstrap would produce a different support.
    assert summary["zone_d_rate_ci95"] == [pytest.approx(1 / 6), pytest.approx(0.5)]
    assert summary["missed_hypoglycemia_share_ci95"] == [0.0, 1.0]


def test_bootstrap_handles_zero_zone_d_patient_without_dropping_it():
    counts = [
        {"patient_id": 540, "predictions": 4, "zone_d": 1, "missed_hypoglycemia": 1},
        {"patient_id": 559, "predictions": 6, "zone_d": 0, "missed_hypoglycemia": 0},
    ]
    summary = patient_bootstrap([counts], replicates=1000, bootstrap_seed=3)
    assert summary["patients"] == 2
    assert summary["patients_with_zone_d"] == 1
    assert summary["share_replicates_with_no_zone_d"] > 0
    assert summary["missed_hypoglycemia_share_ci95"] == [1.0, 1.0]


def test_analyze_run_requires_all_configured_prediction_files(tmp_path):
    run_dir = tmp_path / "regular" / "seed_42"
    run_dir.mkdir(parents=True)
    tracking = {
        "config": {"model": {"type": "gru"}},
        "data_params": {
            "patient_ids": [540, 559],
            "mode": "regular",
            "seed": 42,
            "prediction_horizon_minutes": 30,
        },
    }
    (run_dir / "tracking.json").write_text(json.dumps(tracking), encoding="utf-8")
    patient_dir = run_dir / "patient_540"
    patient_dir.mkdir()
    pd.DataFrame({
        "true_glucose_mg_dl": [60, 100],
        "predicted_glucose_mg_dl": [100, 100],
    }).to_csv(patient_dir / "GRU_predictions.csv", index=False)
    with pytest.raises(ValueError, match="patient 559: missing.*save_predictions"):
        analyze_run(run_dir)

    other_dir = run_dir / "patient_559"
    other_dir.mkdir()
    pd.DataFrame({
        "true_glucose_mg_dl": [250, 100],
        "predicted_glucose_mg_dl": [150, 100],
    }).to_csv(other_dir / "GRU_predictions.csv", index=False)
    summary, per_patient = analyze_run(run_dir, replicates=100, bootstrap_seed=2)
    assert summary["patients"] == 2
    assert summary["patients_with_zone_d"] == 2
    assert summary["prediction_horizon_minutes"] == 30
    assert per_patient["patient_id"].tolist() == [540, 559]
    assert per_patient["missed_hypoglycemia_share_of_zone_d"].tolist() == [1.0, 0.0]


def test_bootstrap_reports_undefined_share_without_losing_zone_d_rate():
    summary = patient_bootstrap([[
        {"patient_id": 540, "predictions": 2, "zone_d": 0, "missed_hypoglycemia": 0},
        {"patient_id": 559, "predictions": 2, "zone_d": 0, "missed_hypoglycemia": 0},
    ]])
    assert summary["zone_d_rate"] == 0.0
    assert summary["zone_d_rate_ci95"] == [0.0, 0.0]
    assert summary["missed_hypoglycemia_share_of_zone_d"] is None
    assert summary["missed_hypoglycemia_share_ci95"] is None


# --------------------------------------------------------------------------- #
# nested patients x training-seeds bootstrap
# --------------------------------------------------------------------------- #
def _seed_table(zone_d, *, predictions=100, patients=12):
    return [{"patient_id": pid, "predictions": predictions, "zone_d": zone_d,
             "missed_hypoglycemia": zone_d} for pid in range(patients)]


def test_one_seed_is_drawn_per_replicate_so_common_mode_survives():
    """Seeds move every patient together, so a replicate must be pure.

    With one seed per replicate the resampling distribution has only the two
    per-seed rates in its support. Drawing a seed independently per patient would
    average them to ~0.06 and collapse the interval.
    """
    summary = patient_bootstrap([_seed_table(10), _seed_table(2)],
                                replicates=4000, bootstrap_seed=5)
    assert summary["per_seed_zone_d_rate"] == [pytest.approx(0.10), pytest.approx(0.02)]
    assert summary["zone_d_rate_ci95"] == [pytest.approx(0.02), pytest.approx(0.10)]
    assert summary["zone_d_rate"] == pytest.approx(0.06)


def test_including_seed_variance_widens_the_interval():
    disagreeing = [_seed_table(4), _seed_table(12), _seed_table(8)]
    nested = patient_bootstrap(disagreeing, replicates=4000, bootstrap_seed=5)
    single = patient_bootstrap([disagreeing[0]], replicates=4000, bootstrap_seed=5)
    width = lambda s: s["zone_d_rate_ci95"][1] - s["zone_d_rate_ci95"][0]
    assert width(nested) > width(single)
    assert nested["uncertainty_units"] == ["patient", "training_seed"]
    assert "combined between-patient and training-run" in nested["note"]
    # A lone seed keeps the old estimand, and says so.
    assert single["uncertainty_units"] == ["patient"]
    assert "do not include training-seed uncertainty" in single["note"]
    assert single["counts_are_seed_means"] is False


def test_point_estimate_is_the_mean_of_the_per_seed_pooled_rates():
    summary = patient_bootstrap([_seed_table(3), _seed_table(10), _seed_table(5)],
                                replicates=200, bootstrap_seed=1)
    assert summary["zone_d_rate"] == pytest.approx(
        sum(summary["per_seed_zone_d_rate"]) / 3)
    assert summary["training_seeds"] == 3
    assert summary["counts_are_seed_means"] is True
    # Seed-mean counts are reported as means, not silently truncated to ints.
    assert summary["zone_d"] == pytest.approx(12 * 6.0)


def test_a_flat_count_table_is_rejected_with_the_new_shape_named():
    with pytest.raises(TypeError, match=r"one count table per training seed"):
        patient_bootstrap(_seed_table(4))


@pytest.mark.parametrize("mutate,message", [
    (lambda t: t[0].__setitem__(0, {**t[0][0], "patient_id": 99}), "same patient cohort"),
    (lambda t: t[0].__setitem__(0, {**t[0][0], "predictions": 50}), "not the same cell"),
    (lambda t: t[0].__setitem__(0, {**t[0][0], "zone_d": 5, "missed_hypoglycemia": 1,
                                    "missed_hyperglycemia": 1}), "partition zone D"),
])
def test_inconsistent_seed_tables_are_rejected(mutate, message):
    tables = [_seed_table(4), _seed_table(4)]
    mutate(tables)
    with pytest.raises(ValueError, match=message):
        patient_bootstrap(tables, replicates=50)


def _write_run(root, mode, seed, zone_d_patients, *, horizon=30, model="gru"):
    run_dir = root / mode / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "tracking.json").write_text(json.dumps({
        "config": {"model": {"type": model}},
        "data_params": {"patient_ids": [540, 559], "mode": mode, "seed": seed,
                        "prediction_horizon_minutes": horizon},
    }), encoding="utf-8")
    for pid in (540, 559):
        patient_dir = run_dir / f"patient_{pid}"
        patient_dir.mkdir()
        # A true 60 predicted as 100 is zone D (missed hypoglycemia); 100->100 is zone A.
        true = [60, 100] if pid in zone_d_patients else [100, 100]
        pd.DataFrame({"true_glucose_mg_dl": true,
                      "predicted_glucose_mg_dl": [100, 100]}).to_csv(
            patient_dir / f"{model.upper()}_predictions.csv", index=False)
    return run_dir


def test_analyze_runs_pools_every_seed_of_one_cell(tmp_path):
    runs = [_write_run(tmp_path, "regular", 41, {540}),
            _write_run(tmp_path, "regular", 42, {540, 559})]
    summary, per_patient = analyze_runs(runs, replicates=500, bootstrap_seed=3)
    assert summary["training_seeds"] == 2
    assert summary["training_seed_values"] == [41, 42]
    assert summary["per_seed_zone_d_rate"] == [pytest.approx(0.25), pytest.approx(0.5)]
    assert summary["zone_d_rate"] == pytest.approx(0.375)
    assert summary["mode"] == "regular" and summary["prediction_horizon_minutes"] == 30
    # One row per patient per seed, so the raw counts stay auditable.
    assert len(per_patient) == 4
    assert sorted(per_patient["training_seed"].unique()) == [41, 42]


def test_analyze_runs_refuses_to_average_across_modes_or_horizons(tmp_path):
    mixed_mode = [_write_run(tmp_path, "regular", 41, {540}),
                  _write_run(tmp_path, "transfer", 42, {540})]
    with pytest.raises(ValueError, match="same mode"):
        analyze_runs(mixed_mode, replicates=50)

    other = tmp_path / "other"
    mixed_horizon = [_write_run(other, "regular", 41, {540}, horizon=30),
                     _write_run(other, "regular", 42, {540}, horizon=60)]
    with pytest.raises(ValueError, match="same prediction_horizon_minutes"):
        analyze_runs(mixed_horizon, replicates=50)


def test_repeating_one_seed_is_rejected(tmp_path):
    run = _write_run(tmp_path, "regular", 41, {540})
    with pytest.raises(ValueError, match="seeds must be distinct"):
        analyze_runs([run, run], replicates=50)


def test_pointing_at_the_parent_experiment_says_so(tmp_path):
    parent = tmp_path / "experiment"
    parent.mkdir()
    (parent / "tracking.json").write_text(json.dumps({
        "config": {"model": {"type": "gru"}},
        "data_params": {"patient_ids": [540, 559], "seeds": [41, 42],
                        "modes": ["regular", "transfer"]},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="parent experiment directory"):
        analyze_run(parent)
