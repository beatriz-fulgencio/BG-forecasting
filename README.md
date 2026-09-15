# Blood Glucose Forecasting Benchmark

A reproducible benchmark framework for comparing blood glucose prediction models on OhioT1DM across evaluation metrics and experimental protocols.

## Overview

This repository provides a configuration-driven framework for blood glucose forecasting research with:

- **Reproducible Experiments**: Configuration-driven experimental setup
- **Standardized Evaluation**: Domain-specific metrics and protocols
- **Implemented Recurrent Models**: RNN, LSTM, and GRU
- **OhioT1DM Integration**: 2018 and 2020 cohorts
- **Results Tracking**: Comprehensive experiment logging and comparison

## Quick Start

### 1. Install

```bash
git clone https://github.com/beatriz-fulgencio/BG-forecasting.git
cd BG-forecasting
pip install -r requirements.txt
pip install -e .
```

### 2. Obtain OhioT1DM (required)

This repository does not include the OhioT1DM dataset and does not provide a
synthetic-data quick start. Request and download the XML data from the
[OhioT1DM dataset page](https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html),
and comply with its access and data-use terms.

Place the downloaded files in this exact layout:

```text
data/raw/ohiot1dm/
├── 2018/
│   ├── train/559-ws-training.xml ...
│   └── test/559-ws-testing.xml ...
└── 2020/
    ├── train/540-ws-training.xml ...
    └── test/540-ws-testing.xml ...
```

The CLI validates the train and test XML files for every patient selected in
the configuration before creating an experiment run. A missing file produces
an actionable error and a nonzero exit status.

### 3. Run an experiment

```bash
bg-forecast run --config benchmark/configs/example_experiment.yaml
# Equivalent without the installed console script:
python -m benchmark.cli run --config benchmark/configs/example_experiment.yaml
```

The example is a real-data, two-patient GRU smoke run. Results are written to
`results/experiments/` by default. One parent experiment contains its fully
resolved configuration, aggregate metrics, and one subdirectory per training
mode and seed:

```text
experiment_ID/
├── tracking.json
├── resolved_config.yaml
├── aggregate_metrics.{json,csv}
├── regular/seed_42/...
└── transfer/seed_42/...
```

Each seed subrun contains its own tracking manifest, selected metric exports,
and any model, prediction, or plot artifacts enabled under `output`.

### Training modes

`training.mode` is required and has three values:

- `regular`: train a separate model on each target patient's data.
- `transfer`: pretrain using the other selected patients, then fine-tune for each target patient. At least two patients are required.
- `both`: execute and track both protocols as separate runs using the same data and model configuration.

Optimization settings (`epochs`, `batch_size`, `learning_rate`, early stopping,
`seeds`, and device) belong under `training`. `training.seeds` is a required,
non-empty list; use `[42]` for one run or `[42, 43, 44]` for multiple runs.
Aggregate exports report the mean, population standard deviation, minimum,
maximum, and count across seeds. Network shape belongs under
`model.architecture`. See
[`benchmark/configs/default.yaml`](benchmark/configs/default.yaml) for the
complete version-1 schema.

### Inspect results

```bash
bg-forecast analyze --experiment-dir results/experiments/experiment_ID
bg-forecast compare --experiments results/experiments/experiment_A results/experiments/experiment_B
bg-forecast list
```

Batch execution is intentionally not part of the CLI.

## Documentation

See the [benchmark README](benchmark/README.md) for comprehensive documentation including:
- Detailed setup instructions
- Configuration options
- Supported models and datasets
- Evaluation metrics
- API reference

## Repository Structure

```
BG-forecasting/
├── benchmark/              # Main benchmark framework
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model implementations
│   ├── evaluation/        # Metrics and evaluation protocols
│   ├── configs/           # Example configurations
│   ├── experiments/       # Experiment management
│   ├── results/           # Result format documentation
│   ├── utils/             # Common utilities
│   └── README.md          # Detailed documentation
├── results/               # Generated experiment outputs (default)
├── requirements.txt       # Dependencies
├── setup.py              # Package installation
└── README.md             # This file
```

## Contributing

We welcome contributions! Please see the [benchmark documentation](benchmark/README.md) for guidelines on:
- Adding new models
- Supporting new datasets
- Implementing new metrics
- Contributing to documentation

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
