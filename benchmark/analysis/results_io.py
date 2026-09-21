"""
Main reader for the experiment results for the comparison modules.
"""

import json
import contextlib
import io
import re
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import yaml  # type: ignore

TRUE_COLUMN = "true_glucose_mg_dl"
PRED_COLUMN = "predicted_glucose_mg_dl"
CONTEXT_COLUMNS = (
    "sequence_id",
    "forecast_origin_timestamp",
    "forecast_origin_glucose_mg_dl",
    "preceding_target_timestamp",
    "preceding_target_glucose_mg_dl",
    "target_timestamp",
)
_HISTORY_COLUMN = re.compile(r"^history_glucose_(\d+)_mg_dl$")

# Metrics every seed records, flat and nested respectively.
SCALAR_METRICS = ("mae", "rmse", "mape", "mard")
NESTED_METRICS = ("tir", "clarke_zones", "parkes_zones", "prediction_diagnostics")

_HORIZON_IN_NAME = re.compile(r"(\d+)\s*min")


class ResultsLayoutError(RuntimeError):
    """Raised when a directory does not hold results this module can read."""
    
# helpers 
def _read_json(path: Path):
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as error:
        raise ResultsLayoutError(f"{path}: cannot read valid JSON ({error})") from error


def _as_float(value: Any):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _zone_a_b(zones: Dict[str, Any], explicit: Any):
    """Combined clinically acceptable zones. """
    direct = _as_float(explicit)
    if direct is not None:
        return direct
    a, b = _as_float(zones.get("A")), _as_float(zones.get("B"))
    return None if a is None or b is None else a + b


def _canonical_metrics(raw: Dict[str, Any]):
    """Normalise one patient/model metric block into the shape analyses expect."""
    metrics: Dict[str, Any] = {name: _as_float(raw.get(name)) for name in SCALAR_METRICS}
    for name in NESTED_METRICS:
        value = raw.get(name)
        metrics[name] = dict(value) if isinstance(value, dict) else {}
    # Derived A/B zones
    metrics["clarke_a_b"] = _zone_a_b(metrics["clarke_zones"], raw.get("clarke_a_b"))
    metrics["parkes_a_b"] = _zone_a_b(metrics["parkes_zones"], raw.get("parkes_a_b"))
    # Populated by the caller that knows which run directory to read from.
    metrics["y_true"] = None
    metrics["predictions"] = None
    metrics["n_predictions"] = 0
    return metrics

# Layout navigation ----

def read_tracking(experiment_dir: Path):
    """reads``tracking.json``, or ``None`` when the directory has none."""
    tracking_path = Path(experiment_dir) / "tracking.json"
    return _read_json(tracking_path) if tracking_path.is_file() else None


def _configured_parent(path: Path):
    """Return the configured parent for a parent, mode, or seed directory."""
    path = Path(path)
    for candidate in (path, path.parent, path.parent.parent):
        if (candidate / "tracking.json").is_file():
            return candidate
    return None


def _configured_metadata(path: Path):
    parent = _configured_parent(path)
    tracking = read_tracking(parent) if parent is not None else None
    config = tracking.get("config", {}) if isinstance(tracking, dict) else {}
    preprocessing = config.get("preprocessing", {}) if isinstance(config, dict) else {}
    data = config.get("data", {}) if isinstance(config, dict) else {}
    evaluation = config.get("evaluation", {}) if isinstance(config, dict) else {}
    data_params = tracking.get("data_params", {}) if isinstance(tracking, dict) else {}

    steps = preprocessing.get("prediction_horizon")
    rate = preprocessing.get("sampling_rate")
    patients = data_params.get("patient_ids", data.get("patients"))
    requested_metrics = evaluation.get("metrics", ()) if isinstance(evaluation, dict) else ()
    return {
        "parent": parent,
        "horizon_steps": steps if isinstance(steps, int) and steps > 0 else None,
        "sampling_rate_minutes": rate if isinstance(rate, int) and rate > 0 else None,
        "patient_cohort": (
            tuple(sorted(patients))
            if isinstance(patients, list)
            and patients
            and all(isinstance(patient, int) and not isinstance(patient, bool) for patient in patients)
            else None
        ),
        "requested_metrics": tuple(requested_metrics) if isinstance(requested_metrics, list) else (),
    }


def _seed_payload(seed_dir: Path):
    """Read and structurally validate a configured seed's completed metrics."""
    metrics_path = Path(seed_dir) / "metrics.json"
    if not metrics_path.is_file():
        raise ResultsLayoutError(f"{seed_dir}: no metrics.json")
    raw = _read_json(metrics_path)
    if not isinstance(raw, dict) or not raw:
        raise ResultsLayoutError(f"{metrics_path}: expected a non-empty patient mapping")

    metadata = _configured_metadata(seed_dir)
    expected = metadata["patient_cohort"]
    expected_mode = seed_dir.parent.name if seed_dir.name.startswith("seed_") else None
    try:
        expected_seed = int(seed_dir.name.split("_", 1)[1]) if seed_dir.name.startswith("seed_") else None
    except (IndexError, ValueError):
        expected_seed = None
    required_metrics = {
        "clarke_ega": "clarke_zones",
        "parkes_ega": "parkes_zones",
    }
    found: set[int] = set()
    for patient_key, entry in raw.items():
        try:
            patient_id = int(patient_key)
        except (TypeError, ValueError) as error:
            raise ResultsLayoutError(
                f"{metrics_path}: invalid patient key {patient_key!r}"
            ) from error
        if patient_id in found:
            raise ResultsLayoutError(f"{metrics_path}: duplicate patient {patient_id}")
        if not isinstance(entry, dict):
            raise ResultsLayoutError(f"{metrics_path}: patient {patient_id} has no metric mapping")
        model_info = entry.get("model_info")
        if not isinstance(model_info, dict) or not isinstance(model_info.get("model_name"), str):
            raise ResultsLayoutError(
                f"{metrics_path}: patient {patient_id} records no model_info.model_name"
            )
        if model_info.get("patient_id", patient_id) != patient_id:
            raise ResultsLayoutError(
                f"{metrics_path}: patient {patient_id} has mismatched model_info.patient_id"
            )
        if expected_mode is not None and model_info.get("mode", expected_mode) != expected_mode:
            raise ResultsLayoutError(
                f"{metrics_path}: patient {patient_id} records mode {model_info.get('mode')!r}, "
                f"expected {expected_mode!r}"
            )
        if expected_seed is not None and model_info.get("seed", expected_seed) != expected_seed:
            raise ResultsLayoutError(
                f"{metrics_path}: patient {patient_id} records seed {model_info.get('seed')!r}, "
                f"expected {expected_seed}"
            )
        recorded_steps = model_info.get("prediction_horizon_steps")
        if (metadata["horizon_steps"] is not None and recorded_steps is not None
                and recorded_steps != metadata["horizon_steps"]):
            raise ResultsLayoutError(
                f"{metrics_path}: patient {patient_id} records horizon {recorded_steps} steps, "
                f"expected {metadata['horizon_steps']}"
            )
        if not any(_as_float(entry.get(metric)) is not None for metric in SCALAR_METRICS):
            raise ResultsLayoutError(
                f"{metrics_path}: patient {patient_id} has no numeric scalar metric"
            )
        for requested in metadata["requested_metrics"]:
            stored = required_metrics.get(requested, requested)
            value = entry.get(stored)
            valid = isinstance(value, dict) if stored in NESTED_METRICS else _as_float(value) is not None
            if not valid:
                raise ResultsLayoutError(
                    f"{metrics_path}: patient {patient_id} is missing requested metric {stored!r}"
                )
        found.add(patient_id)

    if expected is not None and found != set(expected):
        missing = sorted(set(expected) - found)
        extra = sorted(found - set(expected))
        raise ResultsLayoutError(
            f"{metrics_path}: incomplete patient cohort"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
        )
    return raw


def get_modes(experiment_dir: Path):
    """Training modes (``regular``, ``transfer``) in a parent run."""
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        return ()
    modes = []
    for child in experiment_dir.iterdir():
        if child.is_dir() and get_seeds(experiment_dir, child.name):
            modes.append(child.name)
    return tuple(sorted(modes))


def get_seeds(experiment_dir: Path, mode: str):
    """Seed IDs with a complete ``metrics.json``.

    OBS:. Interrupted, malformed, and partial seed directories are  not
    shown as available. Loading one directly still raises the detailed
    validation error.
    """
    seeds = []
    for child in sorted((Path(experiment_dir) / mode).glob("seed_*")):
        if not child.is_dir():
            continue
        try:
            seed = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        try:
            _seed_payload(child)
        except ResultsLayoutError:
            continue
        seeds.append(seed)
    return tuple(sorted(seeds))


def _iter_seed_metrics(experiment_dir: Path) :
    """GO over every ``metrics.json`` in a parent run, in order."""
    for mode in get_modes(experiment_dir):
        for seed in get_seeds(experiment_dir, mode):
            candidate = experiment_dir / mode / f"seed_{seed}" / "metrics.json"
            if candidate.is_file():
                yield candidate


def resolve_horizon_minutes(experiment_dir: Path) :
    """Prediction horizon in minutes"""
    experiment_dir = Path(experiment_dir)

    configured = _configured_metadata(experiment_dir)
    if configured["horizon_steps"] is not None and configured["sampling_rate_minutes"] is not None:
        return configured["horizon_steps"] * configured["sampling_rate_minutes"]

    tracking = read_tracking(experiment_dir)
    if tracking:
        preprocessing = tracking.get("config", {}).get("preprocessing", {})
        steps = preprocessing.get("prediction_horizon")
        rate = preprocessing.get("sampling_rate")
        if isinstance(steps, int) and isinstance(rate, int) and steps > 0 and rate > 0:
            return steps * rate

    own_metrics = experiment_dir / "metrics.json"
    candidates = [own_metrics] if own_metrics.is_file() else list(_iter_seed_metrics(experiment_dir))
    for metrics_path in candidates:
        for entry in _read_json(metrics_path).values():
            if not isinstance(entry, dict):
                continue
            minutes = entry.get("model_info", {}).get("prediction_horizon_minutes")
            if isinstance(minutes, int) and minutes > 0:
                return minutes
        break

    return None


def experiment_name(experiment_dir: Path) :
    """Run name: the configured name when recorded, else the directory.
    
    Run dirs are named experiment_<date>_<time>_<config hash>. 
    The tracking files carry the real name: every run under results/ records. E.g. config.experiment.name = 'full_gru_15min'.

    """
    parent = _configured_parent(Path(experiment_dir))
    tracking = read_tracking(parent) if parent is not None else read_tracking(Path(experiment_dir))
    if tracking:
        name = tracking.get("config", {}).get("experiment", {}).get("name")
        if isinstance(name, str) and name:
            return name
    return Path(experiment_dir).name


# Predictions ---

def _history_columns(frame: pd.DataFrame):
    columns = [column for column in frame.columns if _HISTORY_COLUMN.match(str(column))]
    return sorted(columns, key=lambda column: int(_HISTORY_COLUMN.match(str(column)).group(1)))


def _has_prediction_context(frame: pd.DataFrame): 
    return set(CONTEXT_COLUMNS) <= set(frame.columns) and bool(_history_columns(frame))


def validate_prediction_context(
    frame: pd.DataFrame, *, window_size: Optional[int] = None, source: str = "prediction frame"
): 
    """Validate the context schema and its row order."""
    missing = sorted(set(CONTEXT_COLUMNS) - set(frame.columns))
    if missing:
        raise ResultsLayoutError(f"{source}: missing prediction-context columns {missing}")
    history_columns = _history_columns(frame)
    if window_size is not None and len(history_columns) != window_size:
        raise ResultsLayoutError(
            f"{source}: expected {window_size} glucose-history columns, found {len(history_columns)}"
        )
    expected_history_columns = [
        f"history_glucose_{index:02d}_mg_dl" for index in range(len(history_columns))
    ]
    if history_columns != expected_history_columns:
        raise ResultsLayoutError(
            f"{source}: glucose-history columns are not a contiguous zero-based sequence"
        )

    sequence_ids = pd.to_numeric(frame["sequence_id"], errors="coerce").to_numpy()
    if not np.array_equal(sequence_ids, np.arange(len(frame))):
        raise ResultsLayoutError(
            f"{source}: sequence_id must be exactly 0..{max(len(frame) - 1, 0)} in row order"
        )
    for column in (
        "forecast_origin_timestamp", "preceding_target_timestamp", "target_timestamp"
    ):
        timestamps = pd.to_datetime(frame[column], errors="coerce")
        if timestamps.isna().any():
            raise ResultsLayoutError(f"{source}: {column} contains an invalid timestamp")

    numeric_columns = [
        *history_columns,
        "forecast_origin_glucose_mg_dl",
        "preceding_target_glucose_mg_dl",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ResultsLayoutError(f"{source}: {column} contains a non-finite glucose value")
    if history_columns:
        origin = pd.to_numeric(frame["forecast_origin_glucose_mg_dl"]).to_numpy(dtype=float)
        history_origin = pd.to_numeric(frame[history_columns[-1]]).to_numpy(dtype=float)
        if not np.array_equal(origin, history_origin):
            raise ResultsLayoutError(
                f"{source}: forecast-origin glucose does not equal the final history value"
            )


def attach_prediction_context(
    predictions: pd.DataFrame,
    context: pd.DataFrame,
    *,
    source: str = "reconstructed prediction context",
) :
    """Attach context only when targets and row order agree exactly."""
    if len(predictions) != len(context):
        raise ResultsLayoutError(
            f"{source}: {len(context)} context rows for {len(predictions)} prediction rows"
        )
    if "true_values" not in predictions or "true_values" not in context:
        raise ResultsLayoutError(f"{source}: target column is required for exact alignment")
    observed = predictions["true_values"].to_numpy(dtype=float)
    reconstructed = context["true_values"].to_numpy(dtype=float)
    if not np.array_equal(observed, reconstructed, equal_nan=True):
        mismatch = int(np.flatnonzero(~np.isclose(observed, reconstructed, rtol=0, atol=0, equal_nan=True))[0])
        raise ResultsLayoutError(
            f"{source}: target/order mismatch at row {mismatch}: "
            f"prediction CSV has {observed[mismatch]}, reconstructed data has {reconstructed[mismatch]}"
        )

    validate_prediction_context(context, source=source)
    merged = predictions.copy()
    for column in context.columns:
        if column == "true_values":
            continue
        expected = context[column].reset_index(drop=True)
        if column in merged:
            if column.endswith("_timestamp"):
                same = np.array_equal(
                    pd.to_datetime(merged[column], errors="coerce").to_numpy(),
                    pd.to_datetime(expected, errors="coerce").to_numpy(),
                )
            else:
                same = np.array_equal(merged[column].to_numpy(), expected.to_numpy(), equal_nan=True)
            if not same:
                raise ResultsLayoutError(
                    f"{source}: existing {column} does not match reconstructed row order"
                )
        else:
            merged[column] = expected.to_numpy()
    validate_prediction_context(merged, source=source)
    return merged


def _read_resolved_config(run_dir: Path) :
    parent = _configured_parent(run_dir)
    candidates = [run_dir / "resolved_config.yaml"]
    if parent is not None:
        candidates.append(parent / "resolved_config.yaml")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ResultsLayoutError(f"{path}: cannot read resolved configuration ({error})") from error
        if isinstance(raw, dict):
            return raw
    tracking = read_tracking(parent) if parent is not None else None
    config = tracking.get("config") if isinstance(tracking, dict) else None
    return config if isinstance(config, dict) else None


def _resolve_data_root(run_dir: Path, configured_root: Any):
    if not isinstance(configured_root, str) or not configured_root:
        return None
    root = Path(configured_root)
    candidates = [root] if root.is_absolute() else [Path.cwd() / root]
    parent = _configured_parent(run_dir)
    tracking = read_tracking(parent) if parent is not None else None
    working_dir = (
        tracking.get("environment", {}).get("working_directory")
        if isinstance(tracking, dict) else None
    )
    if not root.is_absolute() and isinstance(working_dir, str):
        candidates.append(Path(working_dir) / root)
    for candidate in candidates:
        if (candidate / "raw" / "ohiot1dm").is_dir():
            return candidate.resolve()
    return None


def _patient_version(run_dir: Path, config: Dict[str, Any], patient_id: int) :
    parent = _configured_parent(run_dir)
    tracking = read_tracking(parent) if parent is not None else None
    versions = tracking.get("data_params", {}).get("version_by_patient", {}) if tracking else {}
    version = versions.get(str(patient_id)) if isinstance(versions, dict) else None
    if version in ("2018", "2020"):
        return version
    configured_version = config.get("data", {}).get("version")
    if configured_version in ("2018", "2020"):
        return configured_version
    from ..configs.config_manager import OHIO_PATIENTS
    for candidate in ("2018", "2020"):
        if patient_id in OHIO_PATIENTS[candidate]:
            return candidate
    raise ResultsLayoutError(f"{run_dir}: cannot resolve OhioT1DM release for patient {patient_id}")


#helpers
@lru_cache(maxsize=64)
def _processed_test_frame_cached(
    data_root: str,
    version: str,
    patient_id: int,
    sampling_rate: int,
    include_feature_engineering: bool,
) :
    from ..data.loaders import load_ohiot1dm_data
    from ..data.preprocessors import preprocess_ohiot1dm_data

    with contextlib.redirect_stdout(io.StringIO()):
        raw = load_ohiot1dm_data(
            data_root, patient_ids=[patient_id], mode="test", version=version,
            sampling_rate=sampling_rate,
        )
        processed = preprocess_ohiot1dm_data(
            raw, include_feature_engineering=include_feature_engineering,
        )
    return processed[patient_id]


@lru_cache(maxsize=256)
def _reconstruct_context_cached(
    data_root: str,
    version: str,
    patient_id: int,
    sampling_rate: int,
    window_size: int,
    horizon_steps: int,
    unimodal: bool,
    include_feature_engineering: bool,
) :
    from ..data.torch_dataset import OhioDataset

    processed = _processed_test_frame_cached(
        data_root, version, patient_id, sampling_rate, include_feature_engineering
    )
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = OhioDataset(
            processed, sequence_length=window_size,
            prediction_horizon=horizon_steps, unimodal=unimodal,
        )
    return dataset.prediction_context_frame()


# used for rapid-change, error-localization and  persistence analysis for time inffered analysis
def reconstruct_prediction_context(
    run_dir: Path,
    patient_id: int,
    predictions: Optional[pd.DataFrame] = None,
) :
    """Rebuild prediction context from raw OhioT1DM test data and run config."""
    run_dir = Path(run_dir)
    config = _read_resolved_config(run_dir)
    if config is None:
        raise ResultsLayoutError(f"{run_dir}: no resolved run configuration for context reconstruction")
    preprocessing = config.get("preprocessing", {})
    data = config.get("data", {})
    required = {
        "window_size": preprocessing.get("window_size"),
        "prediction_horizon": preprocessing.get("prediction_horizon"),
        "sampling_rate": preprocessing.get("sampling_rate"),
    }
    if any(not isinstance(value, int) or value <= 0 for value in required.values()):
        raise ResultsLayoutError(f"{run_dir}: incomplete preprocessing config {required}")
    data_root = _resolve_data_root(run_dir, data.get("root"))
    if data_root is None:
        raise ResultsLayoutError(
            f"{run_dir}: configured OhioT1DM data root {data.get('root')!r} is unavailable"
        )
    version = _patient_version(run_dir, config, patient_id)
    context = _reconstruct_context_cached(
        str(data_root), version, patient_id, required["sampling_rate"],
        required["window_size"], required["prediction_horizon"],
        bool(preprocessing.get("unimodal", False)),
        bool(preprocessing.get("include_feature_engineering", True)),
    ).copy()
    if predictions is not None:
        return attach_prediction_context(
            predictions, context,
            source=f"{run_dir}: reconstructed context for patient {patient_id}",
        )
    validate_prediction_context(
        context, window_size=required["window_size"],
        source=f"{run_dir}: reconstructed context for patient {patient_id}",
    )
    return context

def load_predictions(
    run_dir: Path,
    patient_id: int,
    model: str,
    *,
    with_context: bool = False,
):
    """Read one patient/model prediction CSV.
    """
    patient_dir = Path(run_dir) / f"patient_{patient_id}"
    if not patient_dir.is_dir():
        return None

    for pattern in (f"{model}_predictions.csv", f"{model}_patient_{patient_id}_predictions.csv"):
        matches = sorted(patient_dir.glob(pattern))
        if not matches:
            continue
        frame = pd.read_csv(matches[0])
        if TRUE_COLUMN not in frame.columns:
            raise ResultsLayoutError(
                f"{matches[0]}: no ground-truth column; expected {TRUE_COLUMN}"
            )
        canonical = pd.DataFrame({"true_values": frame[TRUE_COLUMN].to_numpy(dtype=float)})
        has_predictions = PRED_COLUMN in frame.columns
        if has_predictions:
            canonical["predictions"] = frame[PRED_COLUMN].to_numpy(dtype=float)
        # Keep timestamps and prediction context losslessly. Only the two
        # value columns are renamed to their canonical spellings.
        selected = {TRUE_COLUMN, PRED_COLUMN} if has_predictions else {TRUE_COLUMN}
        for column in frame.columns:
            if column not in selected:
                canonical[column] = frame[column].to_numpy()
        if with_context:
            if _has_prediction_context(canonical):
                validate_prediction_context(canonical, source=str(matches[0]))
            else:
                canonical = reconstruct_prediction_context(run_dir, patient_id, canonical)
        return canonical
    return None


def _attach_series(
    patient_metrics: Dict[int, Dict[str, Dict[str, Any]]],
    run_dir: Path,
    include_predictions: bool,
) :
    """Fill ``y_true``, ``n_predictions`` and ``predictions`` from the run's CSVs.

    Mutates ``patient_metrics`` in place, replacing the placeholders reserved by
    ``_canonical_metrics``. ``predictions`` is attached only when
    ``include_predictions`` is set. A patient/model with no prediction file keeps
    its placeholders, so a metrics-only run still loads.
    """
    for patient_id, per_model in patient_metrics.items():
        for model, metrics in per_model.items():
            frame = load_predictions(run_dir, patient_id, model)
            if frame is None:
                continue
            metrics["y_true"] = frame["true_values"].to_numpy(dtype=float)
            metrics["n_predictions"] = int(len(frame))
            if include_predictions and "predictions" in frame.columns:
                metrics["predictions"] = frame["predictions"].to_numpy(dtype=float)


def _patient_model_keys(
    patient_metrics: Mapping[int, Mapping[str, Mapping[str, Any]]]
) :
    return {
        (patient_id, model)
        for patient_id, per_model in patient_metrics.items()
        for model in per_model
    }


def _validate_seed_cohorts(
    experiment_dir: Path,
    mode: str,
    seeds: Sequence[int],
):
    """Load completed seeds and validate them before loading the seed mean."""
    by_seed: Dict[int, Dict[int, Dict[str, Dict[str, Any]]]] = {}
    reference_keys: Optional[set[Tuple[int, str]]] = None
    reference_seed: Optional[int] = None
    for seed in seeds:
        metrics = _load_seed_metrics(experiment_dir / mode / f"seed_{seed}")
        keys = _patient_model_keys(metrics)
        if reference_keys is None:
            reference_keys, reference_seed = keys, seed
        elif keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise ResultsLayoutError(
                f"{experiment_dir}/{mode}: seed {seed} has a different patient/model cohort "
                f"from seed {reference_seed}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
            )
        by_seed[seed] = metrics

    for patient_id, model in reference_keys or set():
        counts = {
            seed: by_seed[seed][patient_id][model]
            .get("prediction_diagnostics", {})
            .get("n_points")
            for seed in seeds
        }
        recorded = {count for count in counts.values() if count is not None}
        if recorded and (len(recorded) != 1 or any(count is None for count in counts.values())):
            raise ResultsLayoutError(
                f"{experiment_dir}/{mode}: recorded prediction counts differ across seeds for "
                f"patient {patient_id} / {model}: {counts}"
            )

    membership: Dict[int, Tuple[int, ...]] = {}
    for patient_id, _model in reference_keys or set():
        membership[patient_id] = tuple(
            seed for seed, metrics in by_seed.items() if patient_id in metrics
        )
    return by_seed, dict(sorted(membership.items()))


def _attach_verified_seed_series(
    patient_metrics: Dict[int, Dict[str, Dict[str, Any]]],
    experiment_dir: Path,
    mode: str,
    seeds: Sequence[int],
) :
    """Attach reference series."""
    for patient_id, per_model in patient_metrics.items():
        for model, metrics in per_model.items():
            frames = [
                load_predictions(experiment_dir / mode / f"seed_{seed}", patient_id, model)
                for seed in seeds
            ]
            if all(frame is None for frame in frames):
                continue
            missing = [seed for seed, frame in zip(seeds, frames) if frame is None]
            if missing:
                raise ResultsLayoutError(
                    f"{experiment_dir}/{mode}: prediction CSV missing for patient {patient_id} / "
                    f"{model} in seeds {missing}"
                )
            present = [frame for frame in frames if frame is not None]
            first = present[0]
            first_truth = first["true_values"].to_numpy(dtype=float)
            for seed, frame in zip(seeds[1:], present[1:]):
                truth = frame["true_values"].to_numpy(dtype=float)
                if len(frame) != len(first):
                    raise ResultsLayoutError(
                        f"{experiment_dir}/{mode}: prediction count differs across seeds for "
                        f"patient {patient_id} / {model} ({len(first)} vs {len(frame)} at seed {seed})"
                    )
                if not np.array_equal(first_truth, truth, equal_nan=True):
                    raise ResultsLayoutError(
                        f"{experiment_dir}/{mode}: reference glucose differs across seeds for "
                        f"patient {patient_id} / {model} (seed {seeds[0]} vs {seed})"
                    )
            metrics["y_true"] = first_truth
            metrics["n_predictions"] = int(len(first))


# Per-layout readers---

def _load_seed_metrics(seed_dir: Path):
    """Patient x model metrics from one configured seed's ``metrics.json``."""
    metrics_path = Path(seed_dir) / "metrics.json"

    patient_metrics: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for patient_key, entry in _seed_payload(seed_dir).items():
        try:
            patient_id = int(patient_key)
        except (TypeError, ValueError):
            continue
        model = entry.get("model_info", {}).get("model_name")
        if not isinstance(model, str) or not model:
            raise ResultsLayoutError(
                f"{metrics_path}: patient {patient_key} records no model_info.model_name"
            )
        patient_metrics.setdefault(patient_id, {})[model] = _canonical_metrics(entry)
    if not patient_metrics:
        raise ResultsLayoutError(f"{metrics_path}: no patient metrics")
    return patient_metrics


def _unflatten(summaries: Dict[str, Any]) :
    """Turn ``{"clarke_zones.A": {...}}`` into nested means plus their dispersion.
    """
    nested: Dict[str, Any] = {}
    dispersion: Dict[str, Dict[str, Any]] = {}
    for name, summary in summaries.items():
        if isinstance(summary, dict):
            mean = summary.get("mean")
            dispersion[name] = {
                key: summary.get(key)
                for key in ("std", "sem", "ci95_low", "ci95_high", "min", "max", "n")
            }
        else:
            mean = summary
        head, _, tail = name.partition(".")
        if tail:
            nested.setdefault(head, {})[tail] = mean
        else:
            nested[head] = mean
    return nested, dispersion


def _load_aggregate_metrics(
    experiment_dir: Path, mode: str, available_seeds: Sequence[int]
) :
    """Seed-mean patient x model metrics for one mode of a configured parent run."""
    aggregate_path = Path(experiment_dir) / "aggregate_metrics.json"
    rows = _read_json(aggregate_path)
    if not isinstance(rows, list) or not rows:
        raise ResultsLayoutError(f"{aggregate_path}: no aggregate metric rows")

    patient_metrics: Dict[int, Dict[str, Dict[str, Any]]] = {}
    seeds: set[int] = set()
    patient_seed_membership: Dict[int, Tuple[int, ...]] = {}
    expected_seeds = tuple(sorted(available_seeds))
    for row in rows:
        if not isinstance(row, dict):
            raise ResultsLayoutError(f"{aggregate_path}: aggregate row is not a mapping")
        if row.get("mode") != mode:
            continue
        try:
            patient_id, model = int(row["patient_id"]), row["model"]
        except (KeyError, TypeError, ValueError) as error:
            raise ResultsLayoutError(f"{aggregate_path}: malformed aggregate row {row!r}") from error
        if not isinstance(model, str) or not model:
            raise ResultsLayoutError(f"{aggregate_path}: patient {patient_id} has no model")
        row_seeds = tuple(sorted(row.get("seeds", ())))
        if row_seeds != expected_seeds:
            raise ResultsLayoutError(
                f"{aggregate_path}: patient {patient_id} / {model} aggregates seeds "
                f"{row_seeds}, but completed seeds are {expected_seeds}"
            )
        summaries = row.get("metrics", {})
        if not isinstance(summaries, dict) or not summaries:
            raise ResultsLayoutError(
                f"{aggregate_path}: patient {patient_id} / {model} has no aggregate metrics"
            )
        nested, dispersion = _unflatten(summaries)
        if not any(_as_float(nested.get(metric)) is not None for metric in SCALAR_METRICS):
            raise ResultsLayoutError(
                f"{aggregate_path}: patient {patient_id} / {model} has no numeric scalar metric mean"
            )
        wrong_n = sorted(
            name for name, summary in summaries.items()
            if isinstance(summary, dict)
            and summary.get("n") is not None
            and summary.get("n") != len(row_seeds)
        )
        if wrong_n:
            raise ResultsLayoutError(
                f"{aggregate_path}: patient {patient_id} / {model} summaries {wrong_n} do not "
                f"contain exactly {len(row_seeds)} seed values"
            )
        metrics = _canonical_metrics(nested)
        metrics["dispersion"] = dispersion
        metrics["seeds"] = row_seeds
        if patient_id in patient_metrics and model in patient_metrics[patient_id]:
            raise ResultsLayoutError(
                f"{aggregate_path}: duplicate {mode} rows for patient {patient_id} / {model}"
            )
        patient_metrics.setdefault(patient_id, {})[model] = metrics
        seeds.update(row_seeds)
        prior_membership = patient_seed_membership.get(patient_id)
        if prior_membership is not None and prior_membership != row_seeds:
            raise ResultsLayoutError(
                f"{aggregate_path}: patient {patient_id} has inconsistent seed membership"
            )
        patient_seed_membership[patient_id] = row_seeds

    if not patient_metrics:
        available = sorted({row.get("mode") for row in rows if isinstance(row, dict)})
        raise ResultsLayoutError(
            f"{aggregate_path}: no rows for mode {mode!r} (available: {', '.join(map(str, available))})"
        )
    return patient_metrics, tuple(sorted(seeds)), dict(sorted(patient_seed_membership.items()))


# Public result object

@dataclass(frozen=True)
class ExperimentResults:
    """One experiment's metrics, normalised for the analyses."""

    experiment_dir: Path
    name: str
    aggregation: str     # "seed_mean" | "single_seed"
    mode: Optional[str]
    seeds: Tuple[int, ...]
    sampling_rate_minutes: Optional[int]
    horizon_steps: Optional[int]
    horizon_minutes: Optional[int]
    patient_cohort: Tuple[int, ...]
    patient_seed_membership: Dict[int, Tuple[int, ...]]
    patient_metrics: Dict[int, Dict[str, Dict[str, Any]]]

    @property
    def models(self) :
        return tuple(sorted({model for per_model in self.patient_metrics.values() for model in per_model}))

    @property
    def patients(self):
        return tuple(sorted(self.patient_metrics))

    @property
    def label(self) :
        """Short identifier that survives being put on an axis or in a table."""
        parts: List[str] = [self.name]
        if self.horizon_minutes is not None and not _HORIZON_IN_NAME.search(self.name):
            parts.append(f"{self.horizon_minutes}min")
        if self.mode:
            parts.append(self.mode)
        return "_".join(parts)

    def metrics_for(self, patient_id: int, model: str) :
        return self.patient_metrics.get(patient_id, {}).get(model)

    def to_frame(self) :
        """Tidy one-row-per-patient-and-model table of the scalar metrics."""
        rows: List[Dict[str, Any]] = []
        for patient_id in self.patients:
            for model, metrics in sorted(self.patient_metrics[patient_id].items()):
                row: Dict[str, Any] = {
                    "experiment": self.label,
                    "patient_id": patient_id,
                    "model": model,
                    "mode": self.mode,
                    "sampling_rate_minutes": self.sampling_rate_minutes,
                    "horizon_steps": self.horizon_steps,
                    "horizon_minutes": self.horizon_minutes,
                    "aggregation": self.aggregation,
                    "n_seeds": len(self.seeds),
                    "seed_membership": ",".join(
                        map(str, self.patient_seed_membership.get(patient_id, ()))
                    ),
                }
                for name in SCALAR_METRICS:
                    row[name] = metrics.get(name)
                row["clarke_a_b"] = metrics.get("clarke_a_b")
                row["parkes_a_b"] = metrics.get("parkes_a_b")
                for group in ("tir", "clarke_zones", "parkes_zones"):
                    for key, value in (metrics.get(group) or {}).items():
                        row[f"{group}.{key}"] = value
                rows.append(row)
        return pd.DataFrame(rows)

# Entry points

def load_experiment_results(
    experiment_dir: Any,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
    with_series: bool = True,
) :
    """Load one experiment.

    Args:
        experiment_dir: A configured parent run or a single ``seed_<n>`` dir
        mode: Training mode to read from a configured parent run. 
        seed: Read this seed alone instead of the cross-seed mean. Opt-in, so a
            multi-seed run is never reduced to one arbitrary seed by accident.
        with_series: Attach ``y_true`` (and ``predictions`` for a single seed)
            from the prediction CSVs. Set ``False`` to skip that file reading when only the metrics are needed.

    Raises:
        ResultsLayoutError: the directory holds no readable results
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise ResultsLayoutError(f"{experiment_dir} is not a directory")

    name = experiment_name(experiment_dir)
    horizon = resolve_horizon_minutes(experiment_dir)
    configured = _configured_metadata(experiment_dir)
    sampling_rate = configured["sampling_rate_minutes"]
    horizon_steps = configured["horizon_steps"]
    if horizon_steps is None and horizon is not None and sampling_rate:
        if horizon % sampling_rate == 0:
            horizon_steps = horizon // sampling_rate

    # --- a single configured seed directory, passed directly ---------------- #
    if (experiment_dir / "metrics.json").is_file() and not (experiment_dir / "aggregate_metrics.json").is_file():
        patient_metrics = _load_seed_metrics(experiment_dir)
        own_seed = None
        if experiment_dir.name.startswith("seed_"):
            try:
                own_seed = int(experiment_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                own_seed = None
        if seed is not None and own_seed is not None and seed != own_seed:
            raise ResultsLayoutError(
                f"{experiment_dir} is seed {own_seed}, not requested seed {seed}"
            )
        own_seed = seed if own_seed is None else own_seed
        recorded_modes = {
            entry.get("model_info", {}).get("mode")
            for entry in _seed_payload(experiment_dir).values()
            if isinstance(entry, dict)
        }
        recorded_modes.discard(None)
        inferred_mode = next(iter(recorded_modes)) if len(recorded_modes) == 1 else experiment_dir.parent.name
        if mode is not None and mode != inferred_mode:
            raise ResultsLayoutError(
                f"{experiment_dir} records mode {inferred_mode!r}, not requested mode {mode!r}"
            )
        if with_series:
            _attach_series(patient_metrics, experiment_dir, include_predictions=True)
        cohort = tuple(sorted(patient_metrics))
        membership = {
            patient_id: (() if own_seed is None else (own_seed,)) for patient_id in cohort
        }
        return ExperimentResults(
            experiment_dir=experiment_dir, name=name,
            aggregation="single_seed", mode=inferred_mode,
            seeds=() if own_seed is None else (own_seed,),
            sampling_rate_minutes=sampling_rate, horizon_steps=horizon_steps,
            horizon_minutes=horizon, patient_cohort=cohort,
            patient_seed_membership=membership, patient_metrics=patient_metrics,
        )

    # --- configured parent run ---------------------------------------------- #
    modes = get_modes(experiment_dir)
    if not modes:
        raise ResultsLayoutError(
            f"{experiment_dir} holds no readable results "
            f"(expected aggregate_metrics.json with <mode>/seed_<n>/ subdirectories)"
        )
    if mode is None:
        if len(modes) > 1:
            raise ResultsLayoutError(
                f"{experiment_dir} holds modes {', '.join(modes)}; pass mode=... to choose one "
                f"(they are different training regimes and must not be pooled)"
            )
        mode = modes[0]
    elif mode not in modes:
        raise ResultsLayoutError(
            f"{experiment_dir} has no mode {mode!r} (available: {', '.join(modes)})"
        )

    available_seeds = get_seeds(experiment_dir, mode)
    if not available_seeds:
        raise ResultsLayoutError(
            f"{experiment_dir}/{mode} has no seed with a readable, complete metrics.json"
        )
    if seed is not None:
        if seed not in available_seeds:
            raise ResultsLayoutError(
                f"{experiment_dir}/{mode} has no seed {seed} "
                f"(available: {', '.join(map(str, available_seeds)) or 'none'})"
            )
        seed_dir = experiment_dir / mode / f"seed_{seed}"
        patient_metrics = _load_seed_metrics(seed_dir)
        if with_series:
            _attach_series(patient_metrics, seed_dir, include_predictions=True)
        cohort = tuple(sorted(patient_metrics))
        return ExperimentResults(
            experiment_dir=experiment_dir, name=name,
            aggregation="single_seed", mode=mode, seeds=(seed,),
            sampling_rate_minutes=sampling_rate, horizon_steps=horizon_steps,
            horizon_minutes=horizon, patient_cohort=cohort,
            patient_seed_membership={patient_id: (seed,) for patient_id in cohort},
            patient_metrics=patient_metrics,
        )

    if not (experiment_dir / "aggregate_metrics.json").is_file():
        raise ResultsLayoutError(
            f"{experiment_dir} has no aggregate_metrics.json; pass seed=<n> to read a single seed "
            f"(available: {', '.join(map(str, available_seeds)) or 'none'})"
        )
    by_seed, actual_membership = _validate_seed_cohorts(
        experiment_dir, mode, available_seeds
    )
    patient_metrics, seeds, aggregate_membership = _load_aggregate_metrics(
        experiment_dir, mode, available_seeds
    )
    aggregate_keys = _patient_model_keys(patient_metrics)
    seed_keys = _patient_model_keys(by_seed[available_seeds[0]])
    if aggregate_keys != seed_keys:
        raise ResultsLayoutError(
            f"{experiment_dir}/aggregate_metrics.json: aggregate patient/model cohort does not "
            f"match completed seed metrics"
        )
    if aggregate_membership != actual_membership:
        raise ResultsLayoutError(
            f"{experiment_dir}/aggregate_metrics.json: per-patient seed membership does not "
            f"match completed seed metrics"
        )
    if with_series:
        _attach_verified_seed_series(patient_metrics, experiment_dir, mode, available_seeds)
    cohort = tuple(sorted(patient_metrics))
    return ExperimentResults(
        experiment_dir=experiment_dir, name=name,
        aggregation="seed_mean", mode=mode, seeds=seeds,
        sampling_rate_minutes=sampling_rate, horizon_steps=horizon_steps,
        horizon_minutes=horizon, patient_cohort=cohort,
        patient_seed_membership=actual_membership, patient_metrics=patient_metrics,
    )


GROUPING_METADATA_FIELDS = ("horizon_minutes", "horizon_steps", "sampling_rate_minutes")


def require_grouping_metadata(
    experiment_dir: Any,
    results: "ExperimentResults",
    fields: Sequence[str] = GROUPING_METADATA_FIELDS,
):
    """Refuse a run with unresolved grouping fields: NaN keys silently empty every grouped table."""
    for field in fields:
        if getattr(results, field) is None:
            raise ValueError(
                f"{experiment_dir}: {field} could not be resolved from the run's "
                "tracking.json, and this analysis groups on it, so the run cannot "
                "be analysed."
            )


def validate_comparison_compatibility(
    experiments: Sequence[ExperimentResults],
    *,
    allow_mixed_horizons: bool = False,
    allow_mixed_modes: bool = False,
    allow_mixed_aggregations: bool = False,
    allow_different_seed_cohorts: bool = False,
    allow_different_patient_cohorts: bool = False,
) :
    """Refuse to compare runs that differ in anything the caller did not opt to vary."""
    if len(experiments) < 2:
        return

    def require_one(label: str, values: Sequence[Any], allowed: bool):
        distinct = {repr(value) for value in values}
        if len(distinct) > 1 and not allowed:
            explanation = (
                " These aggregation kinds represent different quantities."
                if label == "aggregation kinds" else ""
            )
            raise ValueError(
                f"Comparison mixes {label}: {', '.join(sorted(distinct))}.{explanation} "
                f"Use a specialised analysis that explicitly supports this difference."
            )

    require_one("aggregation kinds", [result.aggregation for result in experiments],
                allow_mixed_aggregations)
    require_one("training modes", [result.mode for result in experiments], allow_mixed_modes)
    require_one("prediction horizons", [result.horizon_minutes for result in experiments],
                allow_mixed_horizons)
    require_one("sampling rates", [result.sampling_rate_minutes for result in experiments], False)
    require_one("patient cohorts", [result.patient_cohort for result in experiments],
                allow_different_patient_cohorts)
    require_one("seed cohorts", [result.seeds for result in experiments],
                allow_different_seed_cohorts)
    require_one(
        "per-patient seed membership",
        [tuple(sorted(result.patient_seed_membership.items())) for result in experiments],
        allow_different_seed_cohorts or allow_different_patient_cohorts,
    )


def load_experiments(
    experiment_dirs: Sequence[Any],
    mode: Optional[str] = None,
    seed: Optional[int] = None,
    with_series: bool = True,
    *,
    allow_mixed_horizons: bool = False,
    allow_mixed_modes: bool = False,
    allow_mixed_aggregations: bool = False,
    allow_different_seed_cohorts: bool = False,
    allow_different_patient_cohorts: bool = False,
):
    """Load experiments and enforce safe comparison invariants."""
    results = [
        load_experiment_results(path, mode=mode, seed=seed, with_series=with_series)
        for path in experiment_dirs
    ]
    validate_comparison_compatibility(
        results,
        allow_mixed_horizons=allow_mixed_horizons,
        allow_mixed_modes=allow_mixed_modes,
        allow_mixed_aggregations=allow_mixed_aggregations,
        allow_different_seed_cohorts=allow_different_seed_cohorts,
        allow_different_patient_cohorts=allow_different_patient_cohorts,
    )
    return results
