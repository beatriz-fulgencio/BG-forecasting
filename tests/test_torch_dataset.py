"""Regression tests for the PyTorch OhioT1DM dataset."""

import numpy as np
import pandas as pd

from benchmark.data.torch_dataset import OhioDataset


def test_standardization_ignores_missing_values_outside_valid_sequences():
    glucose = np.array(
        [100.0, 105.0, 110.0, np.nan, 120.0, 125.0, 130.0, 135.0],
        dtype=np.float32,
    )
    dataset = OhioDataset(
        pd.DataFrame({"glucose": glucose}),
        sequence_length=2,
        prediction_horizon=1,
        unimodal=True,
        feature_columns=["glucose"],
    )

    expected_mean = np.nanmean(glucose)
    expected_std = np.nanstd(glucose)

    assert dataset.mean[0] == expected_mean
    assert dataset.std[0] == expected_std
    np.testing.assert_allclose(
        dataset[0].numpy().ravel(),
        (glucose[:2] - expected_mean) / expected_std,
    )
    assert np.isfinite(dataset.get_target(0).item())


def test_standardization_centers_constant_features():
    dataset = OhioDataset(
        pd.DataFrame({"glucose": np.full(5, 120.0, dtype=np.float32)}),
        sequence_length=2,
        prediction_horizon=1,
        unimodal=True,
        feature_columns=["glucose"],
    )

    assert dataset.std[0] == 0.0
    np.testing.assert_array_equal(dataset[0].numpy(), np.zeros((2, 1)))
    assert dataset.get_target(0).item() == 0.0
