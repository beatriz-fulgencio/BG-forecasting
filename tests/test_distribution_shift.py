"""Unit tests for benchmark.comparison.distribution_shift."""

import numpy as np
import pandas as pd
import pytest

from benchmark.comparison.distribution_shift import (
    _clean_glucose,
    compute_distribution_shift,
    compute_patient_shift_scores,
    GLUCOSE_MIN,
    GLUCOSE_MAX,
)


def test_clean_glucose_drops_sentinel_and_out_of_range():
    raw = np.array([-1.0, 0.0, 120.0, 5.0, 700.0, np.nan, 180.0])
    cleaned = _clean_glucose(raw)
    # Only 120 and 180 are inside [GLUCOSE_MIN, GLUCOSE_MAX] and finite.
    assert sorted(cleaned.tolist()) == [120.0, 180.0]
    assert cleaned.min() >= GLUCOSE_MIN and cleaned.max() <= GLUCOSE_MAX


def test_identical_distributions_have_zero_shift():
    rng = np.random.default_rng(0)
    g = 120 + 30 * rng.standard_normal(2000)
    g = np.clip(g, GLUCOSE_MIN + 1, GLUCOSE_MAX - 1)
    res = compute_distribution_shift(g, g.copy())
    assert res["shift_score"] == pytest.approx(0.0, abs=1e-9)
    assert res["kl_divergence"] == pytest.approx(0.0, abs=1e-6)


def test_constant_offset_equals_wasserstein_distance():
    # Wasserstein distance between X and X + c is exactly |c|.
    base = np.linspace(80, 200, 1000)
    shifted = base + 15.0
    res = compute_distribution_shift(base, shifted)
    assert res["shift_score"] == pytest.approx(15.0, rel=1e-6)


def test_shift_score_is_monotonic_in_offset():
    base = np.linspace(80, 200, 1000)
    small = compute_distribution_shift(base, base + 5.0)["shift_score"]
    large = compute_distribution_shift(base, base + 40.0)["shift_score"]
    assert large > small


def test_empty_input_returns_nan_not_error():
    res = compute_distribution_shift([-1, -1, -1], [120, 130])
    assert np.isnan(res["shift_score"])
    assert np.isnan(res["kl_divergence"])
    assert res["train_n"] == 0
    assert res["test_n"] == 2


def test_counts_reflect_cleaned_sizes():
    res = compute_distribution_shift([120, 130, -1, 700], [110, -1])
    assert res["train_n"] == 2  # 120, 130
    assert res["test_n"] == 1   # 110


def test_compute_patient_shift_scores_shape_and_sorting():
    def frame(vals):
        return pd.DataFrame({"glucose": vals})

    patient_data = {
        552: {"train": frame([120, 130, 140]), "test": frame([121, 131, 141])},
        540: {"train": frame([100, 110, 120]), "test": frame([150, 160, 170])},
    }
    df = compute_patient_shift_scores(patient_data)
    assert list(df.columns) == ["patient_id", "shift_score", "kl_divergence", "train_n", "test_n"]
    assert df["patient_id"].tolist() == [540, 552]  # sorted
    # Patient 540 has a much larger train->test offset than 552.
    s540 = df.loc[df.patient_id == 540, "shift_score"].iloc[0]
    s552 = df.loc[df.patient_id == 552, "shift_score"].iloc[0]
    assert s540 > s552
