"""Unit tests for benchmark.comparison.signal_features."""

import numpy as np
import pandas as pd
import pytest

from benchmark.comparison.signal_features import (
    lag1_autocorrelation,
    sample_entropy,
    compute_signal_features,
    compute_patient_signal_features,
)


def test_sample_entropy_sine_less_than_noise():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40 * np.pi, 1500)
    sine = 120 + 40 * np.sin(t)
    noise = 120 + 40 * rng.standard_normal(1500)
    assert sample_entropy(sine) < sample_entropy(noise)


def test_sample_entropy_constant_is_zero():
    assert sample_entropy(np.full(500, 120.0)) == 0.0


def test_sample_entropy_too_short_is_nan():
    assert np.isnan(sample_entropy([120.0, 121.0]))


def test_lag1_autocorr_smooth_near_one():
    ramp = np.linspace(80, 200, 1500)
    assert lag1_autocorrelation(ramp) == pytest.approx(1.0, abs=1e-6)


def test_lag1_autocorr_shuffled_is_lower_than_smooth():
    rng = np.random.default_rng(1)
    ramp = np.linspace(80, 200, 1500)
    smooth = lag1_autocorrelation(ramp)
    shuffled = lag1_autocorrelation(rng.permutation(ramp))
    assert shuffled < smooth
    assert abs(shuffled) < 0.2  # essentially uncorrelated


def test_lag1_autocorr_ignores_sentinel_gaps():
    # A perfectly smooth series broken by -1 sentinels. Only consecutive valid
    # pairs should be used, so the correlation must stay high (not corrupted by
    # false adjacencies spanning the gaps).
    series = np.array([100, 101, 102, -1, 200, 201, 202, -1, 90, 91, 92], dtype=float)
    r = lag1_autocorrelation(series)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_lag1_autocorr_insufficient_pairs_is_nan():
    assert np.isnan(lag1_autocorrelation([120.0]))
    # Valid points never adjacent -> no usable pairs.
    assert np.isnan(lag1_autocorrelation([120.0, -1, 130.0, -1, 140.0]))


def test_compute_signal_features_keys():
    feats = compute_signal_features(np.linspace(80, 200, 500))
    assert set(feats) == {"sample_entropy", "autocorr_lag1"}


def test_compute_patient_signal_features_shape_and_sorting():
    def frame(vals):
        return pd.DataFrame({"glucose": vals})

    patient_data = {
        596: {"test": frame(np.linspace(80, 200, 400))},
        540: {"test": frame(120 + 20 * np.sin(np.linspace(0, 20 * np.pi, 400)))},
    }
    df = compute_patient_signal_features(patient_data)
    assert list(df.columns) == ["patient_id", "sample_entropy", "autocorr_lag1"]
    assert df["patient_id"].tolist() == [540, 596]  # sorted
