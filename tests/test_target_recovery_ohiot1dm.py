"""Real-data check that standardized targets recover the exact CGM readings.

OhioT1DM reports whole mg/dL, but features are stored as float32, so the
standardize/inverse round trip drifts by ~2e-5 mg/dL. That is far below any
reported metric except the threshold-counting ones: a reading sitting exactly on
the 180 mg/dL time-in-range boundary lands just above it and is counted on the
wrong side. These tests run against the real dataset because the effect depends
on the actual distribution of readings.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

from benchmark.data.loaders import load_ohiot1dm_data
from benchmark.data.preprocessors import preprocess_ohiot1dm_data
from benchmark.data.torch_dataset import prepare_personal_data

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
CASES = [("2018", 588), ("2020", 540)]


def _available(version, patient_id):
    base = DATA_ROOT / "raw" / "ohiot1dm" / version
    return (base / "train" / f"{patient_id}-ws-training.xml").is_file() and (
        base / "test" / f"{patient_id}-ws-testing.xml"
    ).is_file()


@pytest.fixture(scope="module")
def targets():
    """Standardized test targets per case, with the dataset that scaled them."""
    prepared = {}
    for version, patient_id in CASES:
        if not _available(version, patient_id):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            raw_train = load_ohiot1dm_data(
                str(DATA_ROOT), patient_ids=[patient_id], mode="train",
                version=version, sampling_rate=5,
            )
            raw_test = load_ohiot1dm_data(
                str(DATA_ROOT), patient_ids=[patient_id], mode="test",
                version=version, sampling_rate=5,
            )
            train = preprocess_ohiot1dm_data(raw_train, include_feature_engineering=True)
            test = preprocess_ohiot1dm_data(raw_test, include_feature_engineering=True)
            _, dataset = prepare_personal_data(
                {"train": train[patient_id], "test": test[patient_id]}, 12, 6, True
            )
        # get_target returns the whole prediction horizon; every element of it
        # is a sensor reading, so flatten rather than taking one step.
        standardized = np.concatenate(
            [np.atleast_1d(dataset.get_target(i).numpy()) for i in range(len(dataset))]
        ).astype(np.float64)
        prepared[(version, patient_id)] = (dataset, standardized)
    if not prepared:
        pytest.skip("OhioT1DM is not installed locally")
    return prepared


def test_recovered_targets_are_exact_integer_readings(targets):
    for (version, patient_id), (dataset, standardized) in targets.items():
        recovered = dataset.inverse_transform_reference(standardized)
        assert np.all(recovered == np.round(recovered)), (
            f"{version}/{patient_id}: recovered targets are not whole mg/dL"
        )
        # Every reading lies within the sensor's reporting range.
        assert recovered.min() >= 40.0
        assert recovered.max() <= 400.0


def test_snapping_only_corrects_round_trip_error(targets):
    """The correction must never move a value by half a reading or more."""
    for (version, patient_id), (dataset, standardized) in targets.items():
        lossy = dataset.inverse_transform_target(standardized)
        recovered = dataset.inverse_transform_reference(standardized)
        drift = np.max(np.abs(recovered - lossy))
        assert drift < 1e-3, f"{version}/{patient_id}: moved targets by {drift}"


def test_time_in_range_is_computed_on_the_true_readings(targets):
    """Boundary readings must be counted on the side the sensor reported."""
    for (version, patient_id), (dataset, standardized) in targets.items():
        lossy = dataset.inverse_transform_target(standardized)
        recovered = dataset.inverse_transform_reference(standardized)
        truth = np.round(lossy)

        def tir(values):
            return float(((values >= 70) & (values <= 180)).mean() * 100)

        assert tir(recovered) == pytest.approx(tir(truth)), (
            f"{version}/{patient_id}: TIR still disagrees with the true readings"
        )
        # The uncorrected round trip is what this guards against; for patients
        # with readings pinned at a boundary it lands on the wrong side.
        boundary_readings = int(np.count_nonzero((truth == 180) | (truth == 70)))
        if boundary_readings:
            assert tir(lossy) <= tir(truth) + 1e-9
