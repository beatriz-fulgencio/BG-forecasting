# Blood Glucose Forecasting Benchmark

A reproducible benchmark framework for comparing blood glucose prediction models across different datasets, evaluation metrics, and experimental protocols.

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

### 2. Run Example Experiment
```bash
python -m benchmark.experiments.runner --config configs/example_experiment.yaml
```

### 3. View Results
Results are saved in `results/experiments/` with timestamped directories containing:
- Configuration files
- Model predictions
- Evaluation metrics
- Visualizations

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

### Traditional ML
- Linear Regression
- Support Vector Regression
- Random Forest
- Gradient Boosting
- ARIMA/SARIMA

### Deep Learning
- LSTM/GRU Networks
- Transformer Models
- CNN-based Approaches
- Attention Mechanisms

### Physiological
- Compartmental Models
- PK/PD Models
- Physics-Informed Neural Networks
- Hybrid Approaches

## Supported Datasets

- **OhioT1DM**: Type 1 diabetes dataset with CGM and lifestyle data
- **REPLACE-BG**: Multi-center clinical trial data
- **Tidepool**: Real-world diabetes management data
- **Custom**: Support for custom CSV formats

## Evaluation Metrics

### Standard Regression Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

### Domain-Specific Metrics
- Clarke Error Grid Analysis (EGA)
- Parkes Error Grid Analysis (PEGA)
- Time in Range (TIR) metrics
- Continuous Glucose Error Grid Analysis (CG-EGA)

## Configuration

Experiments are configured using YAML files. See `configs/example_experiment.yaml` for a complete example.

### Key Configuration Sections
- **Data**: Dataset selection and preprocessing parameters
- **Model**: Model type and hyperparameters
- **Evaluation**: Metrics and validation protocols
- **Output**: Result storage and reporting options

## Running Experiments

### Single Experiment
```bash
python -m benchmark.experiments.runner --config configs/my_experiment.yaml
```

### Batch Experiments
```bash
python -m benchmark.experiments.runner --config-dir configs/batch/
```

### Cross-Validation
Enable in configuration:
```yaml
evaluation:
  cross_validation:
    enabled: true
    folds: 5
    strategy: "time_series"
```

## Result Analysis

### View Experiment Results
```python
from benchmark.evaluation.reporting import load_results
results = load_results("results/experiments/2024-01-01_12-00-00_my_experiment")
```

### Compare Multiple Models
```python
from benchmark.evaluation.reporting import compare_experiments
comparison = compare_experiments([
    "experiment1_results/",
    "experiment2_results/"
])
```

## Contributing

### Adding New Models
1. Inherit from `BaseModel` in `models/base_model.py`
2. Implement required methods: `fit()`, `predict()`, `validate()`
3. Add configuration support
4. Include unit tests

### Adding New Datasets
1. Implement loader in `data/loaders.py`
2. Add validation in `data/validators.py`
3. Update configuration schema
4. Add documentation

### Adding New Metrics
1. Implement metric in `evaluation/metrics.py`
2. Add to evaluation protocols
3. Update reporting functions
4. Include visualization support

## Reproducibility Guidelines

1. **Version Control**: All configurations and code versions are tracked
2. **Random Seeds**: Fixed seeds for reproducible results
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