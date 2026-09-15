"""Test that population pretraining never sees a held-out test split."""

import pytest

from benchmark.data import torch_dataset


class _StubDataset:
    mean = 100.0
    std = 20.0

    def __len__(self):
        return 3


def test_leave_one_out_pretraining_uses_source_train_only(monkeypatch):
    seen = []

    def stub_ohio_dataset(raw_df, **kwargs):
        seen.append(raw_df)
        return _StubDataset()

    monkeypatch.setattr(torch_dataset, "OhioDataset", stub_ohio_dataset)
    monkeypatch.setattr(torch_dataset, "prepare_patient_datasets",
                        lambda *args, **kwargs: (_StubDataset(), _StubDataset()))
    patient_data = {
        pid: {"train": object(), "test": object()}
        for pid in (540, 559, 570)
    }

    source, _, _ = torch_dataset.prepare_multi_patient_dataset(
        patient_data, target_patient_id=540
    )
    assert len(source.datasets) == 2
    assert seen == [patient_data[559]["train"], patient_data[570]["train"]]
    assert patient_data[540]["train"] not in seen
    assert all(data["test"] not in seen for data in patient_data.values())


def test_combined_population_dataset_uses_train_only(monkeypatch):
    seen = []

    def stub_ohio_dataset(raw_df, **kwargs):
        seen.append(raw_df)
        return _StubDataset()

    monkeypatch.setattr(torch_dataset, "OhioDataset", stub_ohio_dataset)
    patient_data = {
        pid: {"train": object(), "test": object()}
        for pid in (540, 559)
    }
    combined = torch_dataset.prepare_multi_patient_dataset(patient_data)
    assert len(combined.datasets) == 2
    assert seen == [patient_data[540]["train"], patient_data[559]["train"]]


def test_missing_source_training_split_fails_clearly(monkeypatch):
    monkeypatch.setattr(torch_dataset, "prepare_patient_datasets",
                        lambda *args, **kwargs: (_StubDataset(), _StubDataset()))
    patient_data = {540: {"train": object(), "test": object()},
                    559: {"test": object()}}
    with pytest.raises(KeyError, match="Patient 559 has no training split"):
        torch_dataset.prepare_multi_patient_dataset(patient_data, target_patient_id=540)
