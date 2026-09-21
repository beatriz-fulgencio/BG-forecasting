# Blood Glucose Forecasting Benchmark

A reproducible benchmark framework for comparing blood glucose prediction models on OhioT1DM across evaluation metrics and experimental protocols.

## Overview

This benchmark provides a standardized framework for:
- **Data Management**: Consistent preprocessing and validation across datasets
- **Model Implementation**: Standardized interfaces for different model types
- **Evaluation**: Domain-specific metrics and protocols for fair comparison
- **Reproducibility**: Configuration-driven experiments with version control
- **Results Tracking**: Comprehensive logging and comparison tools

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Add the required data

OhioT1DM is not bundled with the repository. Request it from the
[dataset page](https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html)
and place its XML files under `data/raw/ohiot1dm/<version>/{train,test}` as
shown in the repository-level README. There is no no-data quick start.

### 3. Run Example Experiment
```bash
python -m benchmark.cli run --config configs/example_experiment.yaml
```

### 4. View Results
Results are saved in `results/experiments/` as parent experiments containing:
- The resolved configuration and parent tracking manifest
- Aggregate mean, sample standard deviation, standard error, 95% t-interval,
  minimum, maximum, and count across seeds. Seeds are a sample of training runs,
  so the spread uses `ddof=1`; a single seed reports `null` for the dispersion
  fields rather than a zero that would read as perfect agreement
- A failed seed is recorded in `failed_runs` and the remaining seeds still
  aggregate; the parent run is then marked `completed_with_failures`
- A `<mode>/seed_<n>/` subrun with its own manifest, predictions, metrics, and enabled artifacts

## Directory Structure

```
benchmark/
├── data/               # Data loading and preprocessing
├── models/            # Model implementations
├── evaluation/        # Metrics and evaluation protocols
├── configs/           # Experiment configurations
├── experiments/       # Experiment management
├── results/           # Output storage
└── utils/             # Common utilities
```

## Supported Models

### Deep Learning
- RNN
- LSTM
- GRU
- Transformer encoder (separately tuned 30-minute extension)

## Supported Datasets

- **OhioT1DM**: Type 1 diabetes dataset with CGM and lifestyle data

## Evaluation Metrics

### Standard Regression Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

### Domain-Specific Metrics
- Clarke Error Grid Analysis (EGA)
- Parkes Error Grid Analysis (PEGA)
- Time in Range (TIR) metrics, reported as prediction minus reference in
  percentage points: a positive `time_in_range` means the model places more
  readings inside 70-180 mg/dL than the reference does

## Configuration

Experiments use a strict, versioned YAML schema. Unknown keys and unsupported
values are errors rather than ignored options. See
[`configs/default.yaml`](configs/default.yaml) for every field and
[`configs/example_experiment.yaml`](configs/example_experiment.yaml) for the
two-patient smoke run.

### Key Configuration Sections
- **Data**: OhioT1DM version, root, patients, and train/validation ratio
- **Preprocessing**: Input window, prediction horizon, and feature selection
- **Model**: Model type and architecture
- **Training**: Required mode (`regular`, `transfer`, or `both`), required `seeds` list, optimizer settings, and device
- **Evaluation**: Metrics to compute in mg/dL
- **Output**: Result storage and artifact options

Before clinical metrics run, predictions and targets are inverse-transformed to
mg/dL using the *training* statistics, and checked for finite values. Non-finite
values are fatal for both arrays, because a NaN makes every downstream metric
undefined.

The 20–600 mg/dL plausibility range is then applied asymmetrically. A target
outside it is fatal. A prediction outside it is counted, warned about, and
recorded in `prediction_diagnostics`, which is aggregated across seeds alongside
the configured metrics. Prediction artifacts include explicit units and horizon
metadata.

### Error-grid domain

The Clarke and Parkes grids are published on the CGM measurement domain and
their zone rules are defined only there, so the grid range is not a tunable. The
OhioT1DM CGM reports within 40–400 mg/dL and saturates at both ends: across both
dataset versions all 166,533 readings fall in that interval, with 335 pinned at
400 and 206 at 40. Every reference value is therefore already censored.

Predictions are clipped to the same 40–400 mg/dL interval before the two grid
analyses, and **only** before those — MAE, RMSE, MAPE, MARD, TIR and the exported
prediction CSVs all use raw model output. This keeps the comparison symmetric
with the censored reference, keeps both grids on their published domains, and
keeps every pair classified, so no zone percentage is computed over a shifting
denominator. The cost is that clipping slightly flatters the grid metrics, so
`prediction_diagnostics` reports `n_predictions_clipped_for_grids` and
`clipped_prediction_rate` next to the zone percentages, and both are aggregated
across seeds. Report them alongside any published EGA result.

## Running Experiments

### Single Experiment
```bash
python -m benchmark.cli run --config configs/my_experiment.yaml
```

## Result Analysis

### View Experiment Results
```bash
bg-forecast analyze --experiment-dir results/experiments/experiment_ID
```

### Compare Multiple Models
```bash
bg-forecast compare --experiments \
  results/experiments/experiment_A \
  results/experiments/experiment_B
```

## Contributing

### Adding New Models
1. Inherit from `BaseBGModel` in `models/base_model.py`
2. Implement required methods: `fit()`, `predict()`, `validate()`
3. Register the implementation in `experiments/configured.py`
4. Add its architecture fields to `configs/config_manager.py`
5. Include unit tests

### Adding New Datasets
1. Implement loader in `data/loaders.py`
2. Add validation in `configs/config_manager.py`
3. Update configuration schema
4. Add documentation

### Adding New Metrics
1. Implement metric in `evaluation/metrics.py`
2. Add to evaluation protocols
3. Update reporting functions
4. Include visualization support

## Reproducibility Guidelines

1. **Version Control**: All configurations and code versions are tracked
2. **Random Seeds**: Every value in `training.seeds` produces an independently seeded subrun
3. **Environment**: Use provided requirements.txt for consistent dependencies
4. **Documentation**: All experiments must include detailed descriptions

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{bg_forecasting_benchmark,
  title={Blood Glucose Forecasting Benchmark},
  author={Blood Glucose Forecasting Research Group},
  year={2024},
  url={https://github.com/beatriz-fulgencio/BG-forecasting}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Support

For questions and support, please open an issue on the GitHub repository.
