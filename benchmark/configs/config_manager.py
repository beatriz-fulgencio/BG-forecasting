"""Experiment configuration."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml


SCHEMA_VERSION = 1
SUPPORTED_MODELS = ("rnn", "lstm", "gru", "transformer")
SUPPORTED_DATASETS = ("ohiot1dm",)
SUPPORTED_METRICS = (
    "mae", "rmse", "mape", "mard", "tir", "clarke_ega", "parkes_ega"
)
SUPPORTED_TRAINING_MODES = ("regular", "transfer", "both")
MAX_SEED = 2 ** 32 - 1
OHIO_PATIENTS = {
    "2018": (559, 563, 570, 575, 588, 591),
    "2020": (540, 544, 552, 567, 584, 596),
}
OHIO_PATIENTS["both"] = tuple(sorted((*OHIO_PATIENTS["2018"], *OHIO_PATIENTS["2020"])))


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _mapping(value: Any, path: str):
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: Sequence[str], path: str):
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(f"Unknown key(s) in {path}: {', '.join(unknown)}")


def _positive_int(value: Any, path: str):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def _non_negative_int(value: Any, path: str):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{path} must be a non-negative integer")
    return value


def _probability(value: Any, path: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number between 0 and 1")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ConfigError(f"{path} must be between 0 and 1")
    return result


def _boolean(value: Any, path: str):
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be true or false")
    return value


def _string(value: Any, path: str):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ExperimentMetadata:
    name: str
    description: str = ""
    author: str = ""

    @classmethod
    def from_dict(cls, raw: Any):
        data = _mapping(raw, "experiment")
        _reject_unknown(data, ("name", "description", "author"), "experiment")
        if "name" not in data:
            raise ConfigError("experiment.name is required")
        description = data.get("description", "")
        author = data.get("author", "")
        if not isinstance(description, str):
            raise ConfigError("experiment.description must be a string")
        if not isinstance(author, str):
            raise ConfigError("experiment.author must be a string")
        return cls(_string(data["name"], "experiment.name"), description, author)


PatientSelection = Union[str, List[int]]


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "ohiot1dm"
    root: str = "data"
    version: str = "2020"
    patients: PatientSelection = "all"
    train_ratio: float = 0.9
    validation_ratio: float = 0.1

    @classmethod
    def from_dict(cls, raw: Any):
        data = _mapping(raw, "data")
        _reject_unknown(
            data,
            ("dataset", "root", "version", "patients", "train_ratio", "validation_ratio"),
            "data",
        )
        dataset = str(data.get("dataset", "ohiot1dm")).lower()
        if dataset not in SUPPORTED_DATASETS:
            raise ConfigError(f"data.dataset must be one of: {', '.join(SUPPORTED_DATASETS)}")
        root = _string(data.get("root", "data"), "data.root")
        version = str(data.get("version", "2020"))
        if version not in OHIO_PATIENTS:
            raise ConfigError("data.version must be '2018', '2020', or 'both'")

        patients_raw = data.get("patients", "all")
        if patients_raw == "all":
            patients: PatientSelection = "all"
        elif isinstance(patients_raw, list) and patients_raw:
            if any(isinstance(item, bool) or not isinstance(item, int) for item in patients_raw):
                raise ConfigError("data.patients must be 'all' or a non-empty list of integers")
            if len(set(patients_raw)) != len(patients_raw):
                raise ConfigError("data.patients must not contain duplicates")
            invalid = sorted(set(patients_raw) - set(OHIO_PATIENTS[version]))
            if invalid:
                raise ConfigError(f"Patient(s) {invalid} do not belong to OhioT1DM {version}")
            patients = list(patients_raw)
        else:
            raise ConfigError("data.patients must be 'all' or a non-empty list of integers")

        train_ratio = _probability(data.get("train_ratio", 0.9), "data.train_ratio")
        validation_ratio = _probability(
            data.get("validation_ratio", 0.1), "data.validation_ratio"
        )
        if abs(train_ratio + validation_ratio - 1.0) > 1e-9:
            raise ConfigError("data.train_ratio and data.validation_ratio must sum to 1")
        if train_ratio == 0 or validation_ratio == 0:
            raise ConfigError("data train and validation ratios must both be greater than 0")
        return cls(dataset, root, version, patients, train_ratio, validation_ratio)

    def patient_ids(self):
        return list(OHIO_PATIENTS[self.version]) if self.patients == "all" else list(self.patients)

    def version_for_patient(self, patient_id: int):
        """Resolve a selected patient to its original OhioT1DM release."""
        if patient_id not in self.patient_ids():
            raise ConfigError(f"Patient {patient_id} is not selected in data.patients")
        for release in ("2018", "2020"):
            if patient_id in OHIO_PATIENTS[release]:
                return release
        raise ConfigError(f"Unknown OhioT1DM patient {patient_id}")


@dataclass(frozen=True)
class PreprocessingConfig:
    window_size: int = 12
    prediction_horizon: int = 6
    sampling_rate: int = 5
    unimodal: bool = False
    include_feature_engineering: bool = True
    normalization: str = "standardize"

    @classmethod
    def from_dict(cls, raw: Any):
        data = _mapping(raw, "preprocessing")
        _reject_unknown(
            data,
            ("window_size", "prediction_horizon", "sampling_rate", "unimodal",
             "include_feature_engineering", "normalization"),
            "preprocessing",
        )
        normalization = str(data.get("normalization", "standardize")).lower()
        if normalization != "standardize":
            raise ConfigError("preprocessing.normalization currently supports only 'standardize'")
        sampling_rate = _positive_int(data.get("sampling_rate", 5), "preprocessing.sampling_rate")
        if sampling_rate != 5:
            raise ConfigError("preprocessing.sampling_rate currently supports only 5 minutes")
        return cls(
            _positive_int(data.get("window_size", 12), "preprocessing.window_size"),
            _positive_int(data.get("prediction_horizon", 6), "preprocessing.prediction_horizon"),
            sampling_rate,
            _boolean(data.get("unimodal", False), "preprocessing.unimodal"),
            _boolean(data.get("include_feature_engineering", True), "preprocessing.include_feature_engineering"),
            normalization,
        )


@dataclass(frozen=True)
class ModelConfig:
    type: str
    architecture: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any):
        data = _mapping(raw, "model")
        _reject_unknown(data, ("type", "architecture"), "model")
        if "type" not in data:
            raise ConfigError("model.type is required")
        model_type = _string(data["type"], "model.type").lower()
        if model_type not in SUPPORTED_MODELS:
            raise ConfigError(f"model.type must be one of: {', '.join(SUPPORTED_MODELS)}")
        architecture = dict(_mapping(data.get("architecture", {}), "model.architecture"))
        if model_type == "transformer":
            allowed = {
                "d_model", "nhead", "num_layers", "dim_feedforward", "dropout",
                "attention_dropout", "batch_first",
            }
            _reject_unknown(architecture, tuple(allowed), "model.architecture")
            d_model = _positive_int(architecture.get("d_model", 128), "model.architecture.d_model")
            nhead = _positive_int(architecture.get("nhead", 4), "model.architecture.nhead")
            if d_model % nhead:
                raise ConfigError("model.architecture.d_model must be divisible by nhead")
            num_layers = _positive_int(architecture.get("num_layers", 3), "model.architecture.num_layers")
            dim_feedforward = _positive_int(
                architecture.get("dim_feedforward", d_model * 4),
                "model.architecture.dim_feedforward",
            )
            dropout = _probability(architecture.get("dropout", 0.1), "model.architecture.dropout")
            attention_dropout = _probability(
                architecture.get("attention_dropout", 0.1), "model.architecture.attention_dropout"
            )
            batch_first = _boolean(
                architecture.get("batch_first", True), "model.architecture.batch_first"
            )
            if not batch_first:
                raise ConfigError("model.architecture.batch_first currently supports only true")
            return cls(model_type, {
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
                "attention_dropout": attention_dropout,
                "batch_first": batch_first,
            })
        allowed = {"hidden_size", "num_layers", "dropout", "batch_first"}
        if model_type == "rnn":
            allowed.add("nonlinearity")
        _reject_unknown(architecture, tuple(allowed), "model.architecture")
        hidden_size = _positive_int(architecture.get("hidden_size", 64), "model.architecture.hidden_size")
        num_layers = _positive_int(architecture.get("num_layers", 2), "model.architecture.num_layers")
        dropout_default = 0.0 if num_layers == 1 else 0.2
        dropout = _probability(architecture.get("dropout", dropout_default), "model.architecture.dropout")
        if num_layers == 1 and dropout != 0:
            raise ConfigError("model.architecture.dropout must be 0 when num_layers is 1")
        batch_first = _boolean(
            architecture.get("batch_first", True), "model.architecture.batch_first"
        )
        if not batch_first:
            raise ConfigError("model.architecture.batch_first currently supports only true")
        normalized: Dict[str, Any] = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_first": batch_first,
        }
        if model_type == "rnn":
            nonlinearity = str(architecture.get("nonlinearity", "tanh")).lower()
            if nonlinearity not in {"tanh", "relu"}:
                raise ConfigError("model.architecture.nonlinearity must be 'tanh' or 'relu'")
            normalized["nonlinearity"] = nonlinearity
        return cls(model_type, normalized)


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    seeds: List[int]
    epochs: int = 100
    pretrain_epochs: int = 50
    finetune_epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 0.001
    #: Learning rate for the fine-tuning stage of a transfer run. ``None`` reuses
    #: ``learning_rate``. Cui et al. (https://github.com/r-cui/GluPred, MIT) drop
    #: it for fine-tuning -- 3e-4 pre-train, 5e-5 fine-tune -- which is the
    #: schedule this project's transfer mode follows.
    finetune_learning_rate: Optional[float] = None
    early_stopping_patience: int = 10
    device: str = "auto"

    @classmethod
    def from_dict(cls, raw: Any):
        data = _mapping(raw, "training")
        _reject_unknown(
            data,
            ("mode", "seeds", "epochs", "pretrain_epochs", "finetune_epochs", "batch_size",
             "learning_rate", "finetune_learning_rate", "early_stopping_patience",
             "device"),
            "training",
        )
        if "mode" not in data:
            raise ConfigError("training.mode is required and must be regular, transfer, or both")
        mode = _string(data["mode"], "training.mode").lower()
        if mode not in SUPPORTED_TRAINING_MODES:
            raise ConfigError(f"training.mode must be one of: {', '.join(SUPPORTED_TRAINING_MODES)}")
        if "seeds" not in data:
            raise ConfigError("training.seeds is required and must be a non-empty list of integers")
        seeds = data["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ConfigError("training.seeds must be a non-empty list of integers")
        normalized_seeds = []
        for index, seed in enumerate(seeds):
            seed_path = f"training.seeds[{index}]"
            normalized_seed = _non_negative_int(seed, seed_path)
            if normalized_seed > MAX_SEED:
                raise ConfigError(f"{seed_path} must be at most {MAX_SEED}")
            normalized_seeds.append(normalized_seed)
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ConfigError("training.seeds must not contain duplicates")
        learning_rate = data.get("learning_rate", 0.001)
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ConfigError("training.learning_rate must be a positive number")

        finetune_learning_rate = data.get("finetune_learning_rate")
        if finetune_learning_rate is not None:
            if (isinstance(finetune_learning_rate, bool)
                    or not isinstance(finetune_learning_rate, (int, float))
                    or finetune_learning_rate <= 0):
                raise ConfigError(
                    "training.finetune_learning_rate must be a positive number"
                )
            finetune_learning_rate = float(finetune_learning_rate)
        device = _string(data.get("device", "auto"), "training.device").lower()
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigError("training.device must be auto, cpu, cuda, or mps")
        return cls(
            mode,
            normalized_seeds,
            _positive_int(data.get("epochs", 100), "training.epochs"),
            _positive_int(data.get("pretrain_epochs", 50), "training.pretrain_epochs"),
            _positive_int(data.get("finetune_epochs", 25), "training.finetune_epochs"),
            _positive_int(data.get("batch_size", 32), "training.batch_size"),
            float(learning_rate),
            finetune_learning_rate,
            _positive_int(data.get("early_stopping_patience", 10), "training.early_stopping_patience"),
            device,
        )


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["mae", "rmse", "mard"])

    @classmethod
    def from_dict(cls, raw: Any):
        data = _mapping(raw, "evaluation")
        _reject_unknown(data, ("metrics",), "evaluation")
        metrics = data.get("metrics", ["mae", "rmse", "mard"])
        if not isinstance(metrics, list) or not metrics or any(not isinstance(metric, str) for metric in metrics):
            raise ConfigError("evaluation.metrics must be a non-empty list of strings")
        aliases = {"time_in_range": "tir", "clarke": "clarke_ega", "parkes": "parkes_ega"}
        normalized = [aliases.get(metric.lower(), metric.lower()) for metric in metrics]
        invalid = sorted(set(normalized) - set(SUPPORTED_METRICS))
        if invalid:
            raise ConfigError(
                f"Unsupported evaluation metric(s): {', '.join(invalid)}. "
                f"Supported metrics: {', '.join(SUPPORTED_METRICS)}"
            )
        if len(set(normalized)) != len(normalized):
            raise ConfigError("evaluation.metrics must not contain duplicates")
        return cls(normalized)


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "results/experiments"
    save_model: bool = True
    save_predictions: bool = True
    generate_plots: bool = True
    export_format: List[str] = field(default_factory=lambda: ["json", "csv"])

    @classmethod
    def from_dict(cls, raw: Any):
        data = _mapping(raw, "output")
        _reject_unknown(data, ("directory", "save_model", "save_predictions", "generate_plots", "export_format"), "output")
        formats = data.get("export_format", ["json", "csv"])
        if not isinstance(formats, list) or not formats or any(item not in {"json", "csv"} for item in formats):
            raise ConfigError("output.export_format must be a non-empty list containing only 'json' and 'csv'")
        if len(set(formats)) != len(formats):
            raise ConfigError("output.export_format must not contain duplicates")
        return cls(
            _string(data.get("directory", "results/experiments"), "output.directory"),
            _boolean(data.get("save_model", True), "output.save_model"),
            _boolean(data.get("save_predictions", True), "output.save_predictions"),
            _boolean(data.get("generate_plots", True), "output.generate_plots"),
            list(formats),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment: ExperimentMetadata
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    source_path: Path = field(compare=False, repr=False)

    def to_dict(self):
        result = asdict(self)
        result.pop("source_path", None)
        return result


def load_config(config_path: Union[str, Path]):
    """Load and validate a version-1 YAML experiment configuration."""
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read configuration {path}: {exc}") from exc

    data = _mapping(raw, "configuration")
    required = {"schema_version", "experiment", "data", "preprocessing", "model", "training", "evaluation", "output"}
    _reject_unknown(data, tuple(required), "configuration")
    missing = sorted(required - set(data))
    if missing:
        raise ConfigError(f"Missing required section(s): {', '.join(missing)}")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION} (got {data['schema_version']!r})")

    config = ExperimentConfig(
        SCHEMA_VERSION,
        ExperimentMetadata.from_dict(data["experiment"]),
        DataConfig.from_dict(data["data"]),
        PreprocessingConfig.from_dict(data["preprocessing"]),
        ModelConfig.from_dict(data["model"]),
        TrainingConfig.from_dict(data["training"]),
        EvaluationConfig.from_dict(data["evaluation"]),
        OutputConfig.from_dict(data["output"]),
        path.resolve(),
    )
    if config.training.mode in {"transfer", "both"} and len(config.data.patient_ids()) < 2:
        raise ConfigError("transfer training requires at least two patients")
    return config


def validate_data_files(config: ExperimentConfig):
    """Validate the OhioT1DM layout and return the resolved patient IDs."""
    root = Path(config.data.root).expanduser()
    patient_ids = config.data.patient_ids()
    missing: List[Path] = []
    for patient_id in patient_ids:
        base = root / "raw" / "ohiot1dm" / config.data.version_for_patient(patient_id)
        expected = (
            base / "train" / f"{patient_id}-ws-training.xml",
            base / "test" / f"{patient_id}-ws-testing.xml",
        )
        missing.extend(path for path in expected if not path.is_file())
    if missing:
        preview = "\n  ".join(str(path) for path in missing[:6])
        remainder = len(missing) - 6
        suffix = f"\n  ... and {remainder} more" if remainder > 0 else ""
        raise ConfigError(
            "OhioT1DM data is required but the following file(s) are missing:\n"
            f"  {preview}{suffix}\n"
            "Download OhioT1DM separately and place it under "
            f"{root / 'raw' / 'ohiot1dm'}/{{2018,2020}}/{{train,test}}/ "
            "as required by the selected patients."
        )
    return patient_ids
