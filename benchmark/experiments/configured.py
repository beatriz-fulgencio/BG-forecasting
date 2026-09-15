"""Reliable execution path for versioned experiment configurations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ..configs.config_manager import ExperimentConfig
from ..data.loaders import load_ohiot1dm_data
from ..data.preprocessors import preprocess_ohiot1dm_data
from ..data.torch_dataset import prepare_multi_patient_dataset, prepare_personal_data
from ..evaluation.evaluator import BGEvaluator
from ..models.rnn import GRUBGModel, LSTMBGModel, RNNBGModel
from .tracking import ExperimentTracker


MODEL_REGISTRY = {
    "rnn": RNNBGModel,
    "lstm": LSTMBGModel,
    "gru": GRUBGModel,
}


class _ForecastingDataset:
    """Adapt an OhioDataset to PyTorch's ``(features, target)`` convention."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.dataset.get_target(index)


class _ConcatForecastingDataset:
    """Expose targets from the OhioDataset members of a ConcatDataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        local_index = index
        for dataset in self.dataset.datasets:
            if local_index < len(dataset):
                return dataset[local_index], dataset.get_target(local_index)
            local_index -= len(dataset)
        raise IndexError(index)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device is cuda, but CUDA is not available")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("training.device is mps, but Apple MPS is not available")
    return requested


def _load_patient_frames(config: ExperimentConfig, patient_ids: Iterable[int]) -> Dict[int, Dict[str, pd.DataFrame]]:
    frames: Dict[int, Dict[str, pd.DataFrame]] = {}
    for patient_id in patient_ids:
        raw_train = load_ohiot1dm_data(
            config.data.root,
            patient_ids=[patient_id],
            mode="train",
            version=config.data.version,
            sampling_rate=config.preprocessing.sampling_rate,
        )
        raw_test = load_ohiot1dm_data(
            config.data.root,
            patient_ids=[patient_id],
            mode="test",
            version=config.data.version,
            sampling_rate=config.preprocessing.sampling_rate,
        )
        if patient_id not in raw_train or patient_id not in raw_test:
            raise RuntimeError(f"OhioT1DM train/test data could not be loaded for patient {patient_id}")
        train = preprocess_ohiot1dm_data(
            raw_train,
            include_feature_engineering=config.preprocessing.include_feature_engineering,
        )
        test = preprocess_ohiot1dm_data(
            raw_test,
            include_feature_engineering=config.preprocessing.include_feature_engineering,
        )
        if patient_id not in train or patient_id not in test:
            raise RuntimeError(f"Preprocessing failed for patient {patient_id}")
        frames[patient_id] = {"train": train[patient_id], "test": test[patient_id]}
    return frames


def _split_target_dataset(
    dataset, config: ExperimentConfig, seed: int, seed_offset: int = 0
):
    validation_size = max(1, round(len(dataset) * config.data.validation_ratio))
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise RuntimeError(
            f"Only {len(dataset)} training sequences are available; at least two are required"
        )
    return torch.utils.data.random_split(
        _ForecastingDataset(dataset),
        [train_size, validation_size],
        generator=_torch_generator(seed + seed_offset),
    )


def _make_loaders(
    config: ExperimentConfig,
    mode: str,
    seed: int,
    patient_id: int,
    frames: Dict[int, Dict[str, pd.DataFrame]],
):
    prep = config.preprocessing
    if mode == "regular":
        train_dataset, test_dataset = prepare_personal_data(
            frames[patient_id], prep.window_size, prep.prediction_horizon, prep.unimodal
        )
        train_subset, validation_subset = _split_target_dataset(
            train_dataset, config, seed, patient_id
        )
        global_loader = None
    else:
        global_dataset, train_dataset, test_dataset = prepare_multi_patient_dataset(
            frames,
            sequence_length=prep.window_size,
            prediction_horizon=prep.prediction_horizon,
            target_patient_id=patient_id,
            unimodal=prep.unimodal,
        )
        if len(global_dataset) == 0:
            raise RuntimeError(f"No source-patient sequences are available for patient {patient_id}")
        train_subset, validation_subset = _split_target_dataset(
            train_dataset, config, seed, patient_id
        )
        global_loader = DataLoader(
            _ConcatForecastingDataset(global_dataset),
            batch_size=config.training.batch_size,
            shuffle=True,
            generator=_torch_generator(seed + patient_id + 1),
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=_torch_generator(seed + patient_id + 2),
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        _ForecastingDataset(test_dataset),
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    return global_loader, train_loader, validation_loader, test_loader, train_dataset, test_dataset


def _last_horizon(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values
    return values[:, -1]


def _targets(loader: DataLoader) -> np.ndarray:
    batches = [target.numpy() for _, target in loader]
    if not batches:
        raise RuntimeError("The test dataset contains no valid sequences")
    return np.concatenate(batches, axis=0)


def _glucose_scale(dataset) -> Tuple[float, float]:
    glucose_index = dataset.feature_columns.index("glucose")
    mean = float(dataset.mean[glucose_index])
    std = float(dataset.std[glucose_index])
    return mean, std if std > 0 else 1.0


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metric_rows(results: Dict[int, Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    excluded = {"model_info", "training_history", "artifacts"}
    for patient_id, patient_result in results.items():
        row: Dict[str, Any] = {"patient_id": patient_id, "model": model_name}
        for name, value in patient_result.items():
            if name in excluded:
                continue
            if isinstance(value, dict):
                for nested_name, nested_value in value.items():
                    row[f"{name}.{nested_name}"] = nested_value
            elif isinstance(value, (int, float, np.generic)):
                row[name] = value
        rows.append(row)
    return rows


def _save_plots(
    patient_dir: Path,
    model_name: str,
    patient_id: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_loader: DataLoader,
    test_dataset,
    config: ExperimentConfig,
) -> Dict[str, str]:
    if not config.output.generate_plots:
        return {}
    # Plotting is optional and relatively expensive to import, so keep it off
    # the training path when output.generate_plots is false.
    from ..evaluation.visualisation import (
        create_prediction_dashboard,
        plot_clarke_analysis,
        plot_parkes_analysis,
    )

    sequences = np.concatenate([sequence.numpy() for sequence, _ in test_loader], axis=0)
    mean, std = _glucose_scale(test_dataset)
    glucose_index = test_dataset.feature_columns.index("glucose")
    sequences[:, :, glucose_index] = sequences[:, :, glucose_index] * std + mean
    artifacts: Dict[str, str] = {}
    dashboard = patient_dir / f"{model_name}_dashboard.png"
    create_prediction_dashboard(
        y_true=y_true,
        y_pred=y_pred,
        title=f"{model_name.upper()} model for patient {patient_id}",
        prediction_horizon=config.preprocessing.prediction_horizon,
        sequence_length=config.preprocessing.window_size,
        input_sequences=sequences,
        sample_interval_minutes=config.preprocessing.sampling_rate,
        save_path=dashboard,
    )
    artifacts["dashboard"] = str(dashboard)
    if "clarke_ega" in config.evaluation.metrics:
        path = patient_dir / f"{model_name}_clarke_ega.png"
        plot_clarke_analysis(y_true, y_pred, title=f"Clarke EGA - {model_name.upper()}", save_path=path)
        artifacts["clarke_plot"] = str(path)
    if "parkes_ega" in config.evaluation.metrics:
        path = patient_dir / f"{model_name}_parkes_ega.png"
        plot_parkes_analysis(y_true, y_pred, diabetes_type=1, title=f"Parkes EGA - {model_name.upper()}", save_path=path)
        artifacts["parkes_plot"] = str(path)
    return artifacts


def _flatten_numeric(prefix: str, value: Any, output: Dict[str, float]) -> None:
    if isinstance(value, dict):
        for name, nested in value.items():
            nested_prefix = f"{prefix}.{name}" if prefix else name
            _flatten_numeric(nested_prefix, nested, output)
    elif isinstance(value, (int, float, np.generic)) and not isinstance(value, bool):
        output[prefix] = float(value)


def _aggregate_runs(
    runs: List[Dict[str, Any]], config: ExperimentConfig
) -> List[Dict[str, Any]]:
    result_keys = {
        {"clarke_ega": "clarke_zones", "parkes_ega": "parkes_zones"}.get(metric, metric)
        for metric in config.evaluation.metrics
    }
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for run in runs:
        for patient_id_text, result in run["results"].items():
            patient_id = int(patient_id_text)
            model_name = result["model_info"]["model_name"]
            key = (run["mode"], patient_id, model_name)
            group = grouped.setdefault(key, {"seeds": [], "values": {}})
            group["seeds"].append(run["seed"])
            for result_key in result_keys:
                if result_key not in result:
                    continue
                flattened: Dict[str, float] = {}
                _flatten_numeric(result_key, result[result_key], flattened)
                for metric_name, metric_value in flattened.items():
                    group["values"].setdefault(metric_name, []).append(metric_value)

    aggregates: List[Dict[str, Any]] = []
    for (mode, patient_id, model_name), group in sorted(grouped.items()):
        metrics: Dict[str, Dict[str, Any]] = {}
        for metric_name, values in sorted(group["values"].items()):
            array = np.asarray(values, dtype=float)
            metrics[metric_name] = {
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=0)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "n": int(array.size),
            }
        aggregates.append({
            "mode": mode,
            "patient_id": patient_id,
            "model": model_name,
            "seeds": list(group["seeds"]),
            "metrics": metrics,
        })
    return aggregates


def _aggregate_rows(aggregates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for aggregate in aggregates:
        for metric, summary in aggregate["metrics"].items():
            rows.append({
                "mode": aggregate["mode"],
                "patient_id": aggregate["patient_id"],
                "model": aggregate["model"],
                "seeds": ",".join(str(seed) for seed in aggregate["seeds"]),
                "metric": metric,
                **summary,
            })
    return rows


def _run_mode(
    config: ExperimentConfig,
    mode: str,
    seed: int,
    patient_ids: List[int],
    frames: Dict[int, Dict[str, pd.DataFrame]],
    experiment_dir: Path,
) -> Dict[str, Any]:
    _seed_everything(seed)
    resolved = config.to_dict()
    resolved["training"]["mode"] = mode
    resolved["training"]["seeds"] = [seed]
    tracker = ExperimentTracker(
        experiment_dir.parent, config=resolved, experiment_dir=experiment_dir
    )
    tracker.start_experiment()
    tracker.log_data_params({
        "dataset": config.data.dataset,
        "patient_ids": patient_ids,
        "version": config.data.version,
        "sequence_length": config.preprocessing.window_size,
        "prediction_horizon": config.preprocessing.prediction_horizon,
        "batch_size": config.training.batch_size,
        "seed": seed,
        "mode": mode,
    })
    (experiment_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    try:
        evaluator = BGEvaluator()
        model_class = MODEL_REGISTRY[config.model.type]
        model_name = config.model.type.upper()
        results: Dict[int, Dict[str, Any]] = {}

        for patient_id in patient_ids:
            loaders = _make_loaders(config, mode, seed, patient_id, frames)
            global_loader, train_loader, validation_loader, test_loader, train_dataset, test_dataset = loaders
            model = model_class(
                model_name=f"{model_name}_patient_{patient_id}",
                sequence_length=config.preprocessing.window_size,
                prediction_horizon=config.preprocessing.prediction_horizon,
                feature_dim=train_dataset.data.shape[1],
                hyperparameters=dict(config.model.architecture),
                device=_resolve_device(config.training.device),
            )
            tracker.log_training_start(model_name, patient_id, dict(config.model.architecture))
            fit_args = {
                "validation_loader": validation_loader,
                "learning_rate": config.training.learning_rate,
                "early_stopping_patience": config.training.early_stopping_patience,
            }
            if mode == "transfer":
                pretrain_history = model.fit(
                    train_loader=global_loader,
                    epochs=config.training.pretrain_epochs,
                    **fit_args,
                )
                finetune_history = model.fit(
                    train_loader=train_loader,
                    epochs=config.training.finetune_epochs,
                    **fit_args,
                )
                history = {
                    "epochs_completed": pretrain_history["epochs_completed"] + finetune_history["epochs_completed"],
                    "best_val_loss": finetune_history["best_val_loss"],
                    "pretrain_history": pretrain_history,
                    "finetune_history": finetune_history,
                }
            else:
                history = model.fit(
                    train_loader=train_loader,
                    epochs=config.training.epochs,
                    **fit_args,
                )
            tracker.log_training_completion(model_name, patient_id, history)

            predictions = _last_horizon(model.predict(test_loader))
            true_values = _last_horizon(_targets(test_loader))
            mean, std = _glucose_scale(test_dataset)
            predictions = predictions * std + mean
            true_values = true_values * std + mean
            metrics = evaluator.compute_metrics(
                y_true=true_values,
                y_pred=predictions,
                metrics=config.evaluation.metrics,
            )
            patient_dir = experiment_dir / f"patient_{patient_id}"
            patient_dir.mkdir(parents=True, exist_ok=True)
            artifacts: Dict[str, str] = {}
            if config.output.save_predictions:
                path = patient_dir / f"{model_name}_predictions.csv"
                pd.DataFrame({
                    "true_glucose_mg_dl": true_values,
                    "predicted_glucose_mg_dl": predictions,
                    "absolute_error_mg_dl": np.abs(true_values - predictions),
                }).to_csv(path, index=False)
                artifacts["predictions"] = str(path)
            if config.output.save_model:
                path = patient_dir / f"{model_name}_patient_{patient_id}.pth"
                model.save_model(path)
                artifacts["model"] = str(path)
            artifacts.update(
                _save_plots(patient_dir, model_name, patient_id, true_values, predictions,
                            test_loader, test_dataset, config)
            )
            patient_result = dict(metrics)
            patient_result["model_info"] = {
                "model_name": model_name,
                "patient_id": patient_id,
                "mode": mode,
                "seed": seed,
                "prediction_horizon_steps": config.preprocessing.prediction_horizon,
                "prediction_horizon_minutes": config.preprocessing.prediction_horizon * config.preprocessing.sampling_rate,
                "feature_dim": train_dataset.data.shape[1],
            }
            patient_result["training_history"] = history
            patient_result["artifacts"] = artifacts
            results[patient_id] = patient_result
            tracker.log_evaluation_results(model_name, patient_id, patient_result)

        serializable_results = _serializable(results)
        if "json" in config.output.export_format:
            (experiment_dir / "metrics.json").write_text(
                json.dumps(serializable_results, indent=2), encoding="utf-8"
            )
        if "csv" in config.output.export_format:
            pd.DataFrame(_metric_rows(results, model_name)).to_csv(
                experiment_dir / "metrics.csv", index=False
            )
        tracker.end_experiment({model_name: serializable_results})
        return {
            "mode": mode,
            "seed": seed,
            "experiment_id": tracker.experiment_id,
            "experiment_dir": str(experiment_dir),
            "results": serializable_results,
        }
    except Exception as exc:
        tracker.log_error(str(exc), context=f"training.mode={mode}, seed={seed}")
        raise


def run_configured_experiment(config: ExperimentConfig, patient_ids: List[int]) -> Dict[str, Any]:
    """Run all configured mode/seed combinations and aggregate their metrics."""
    print(f"Resolved device: {_resolve_device(config.training.device)}")
    modes = ["regular", "transfer"] if config.training.mode == "both" else [config.training.mode]
    parent_tracker = ExperimentTracker(
        Path(config.output.directory), config=config.to_dict()
    )
    parent_tracker.start_experiment()
    parent_dir = parent_tracker.get_experiment_dir()
    (parent_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    parent_tracker.log_data_params({
        "dataset": config.data.dataset,
        "patient_ids": patient_ids,
        "version": config.data.version,
        "sequence_length": config.preprocessing.window_size,
        "prediction_horizon": config.preprocessing.prediction_horizon,
        "batch_size": config.training.batch_size,
        "seeds": config.training.seeds,
        "modes": modes,
    })

    try:
        frames = _load_patient_frames(config, patient_ids)
        runs: List[Dict[str, Any]] = []
        for mode in modes:
            for seed in config.training.seeds:
                child_dir = parent_dir / mode / f"seed_{seed}"
                runs.append(
                    _run_mode(config, mode, seed, patient_ids, frames, child_dir)
                )
        aggregates = _aggregate_runs(runs, config)
        if not aggregates:
            raise RuntimeError("No successful seed runs were available to aggregate")
        if "json" in config.output.export_format:
            (parent_dir / "aggregate_metrics.json").write_text(
                json.dumps(aggregates, indent=2), encoding="utf-8"
            )
        if "csv" in config.output.export_format:
            pd.DataFrame(_aggregate_rows(aggregates)).to_csv(
                parent_dir / "aggregate_metrics.csv", index=False
            )
        run_summaries = [
            {key: run[key] for key in ("mode", "seed", "experiment_id", "experiment_dir")}
            for run in runs
        ]
        final_results = {"runs": run_summaries, "aggregate": aggregates}
        parent_tracker.end_experiment(final_results)
        return {
            "experiment_id": parent_tracker.experiment_id,
            "experiment_dir": str(parent_dir),
            "runs": runs,
            "aggregate": aggregates,
        }
    except Exception as exc:
        parent_tracker.log_error(str(exc), context="multi-seed experiment")
        raise
