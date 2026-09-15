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


def test_inverse_transform_target_restores_mg_dl_values():
    dataset = OhioDataset(
        pd.DataFrame({"glucose": [100.0, 110.0, 120.0, 130.0]}),
        sequence_length=2,
        prediction_horizon=1,
        unimodal=True,
    )
    standardized = np.array([-1.0, 0.0, 1.0])
    expected = standardized * dataset.std[0] + dataset.mean[0]
    np.testing.assert_allclose(dataset.inverse_transform_target(standardized), expected)
    assert dataset.target_units == "mg/dL"


def test_external_normalization_vectors_are_validated():
    frame = pd.DataFrame({"glucose": [100.0, 110.0, 120.0, 130.0]})
    with np.testing.assert_raises(ValueError):
        OhioDataset(frame, 2, 1, external_mean=[100.0], unimodal=True)
    with np.testing.assert_raises(ValueError):
        OhioDataset(frame, 2, 1, external_mean=[100.0], external_std=[1.0, 2.0], unimodal=True)


def test_inverse_transform_reference_recovers_exact_sensor_readings():
    """float32 storage makes the round trip lossy; readings are whole mg/dL."""
    readings = [100.0, 110.0, 180.0, 70.0, 400.0, 40.0, 250.0, 130.0]
    dataset = OhioDataset(
        pd.DataFrame({"glucose": readings}),
        sequence_length=2,
        prediction_horizon=1,
        unimodal=True,
    )
    standardized = np.array(
        [dataset.get_target(i).item() for i in range(len(dataset))], dtype=np.float64
    )
    lossy = dataset.inverse_transform_target(standardized)
    recovered = dataset.inverse_transform_reference(standardized)

    # The snapped values are exactly the integers the sensor reported.
    np.testing.assert_array_equal(recovered, np.round(lossy))
    assert np.all(recovered == np.round(recovered))
    # And they are no further from the lossy values than the round-trip error.
    assert np.max(np.abs(recovered - lossy)) < 1e-3


def test_inverse_transform_reference_passes_through_non_integer_values():
    """Guards the snap if preprocessing ever yields interpolated glucose."""
    dataset = OhioDataset(
        pd.DataFrame({"glucose": [100.0, 110.0, 120.0, 130.0]}),
        sequence_length=2,
        prediction_horizon=1,
        unimodal=True,
    )
    mean, std = dataset.mean[0], dataset.std[0]
    # A value sitting squarely between two integers must survive untouched.
    standardized = np.array([(123.5 - mean) / std], dtype=np.float64)
    recovered = dataset.inverse_transform_reference(standardized)
    np.testing.assert_allclose(recovered, [123.5], atol=1e-4)
