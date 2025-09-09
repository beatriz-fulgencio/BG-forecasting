# Blood Glucose Forecasting Benchmark

A reproducible benchmark framework for comparing blood glucose prediction models across different datasets, evaluation metrics, and experimental protocols.

## Overview

This repository provides a standardized framework for blood glucose forecasting research with:

- **Reproducible Experiments**: Configuration-driven experimental setup
- **Standardized Evaluation**: Domain-specific metrics and protocols
- **Multiple Model Types**: Traditional ML, deep learning, and physiological models
- **Dataset Integration**: Support for common BG datasets
- **Results Tracking**: Comprehensive experiment logging and comparison

## Quick Start

### Installation
```bash
git clone https://github.com/beatriz-fulgencio/BG-forecasting.git
cd BG-forecasting
pip install -r requirements.txt
pip install -e .
```

### Run Example Experiment
```bash
python -m benchmark.cli run --config benchmark/configs/example_experiment.yaml
```

### View Results
Results are automatically saved in `benchmark/results/experiments/` with detailed metrics and visualizations.

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
│   ├── results/           # Output storage
│   ├── utils/             # Common utilities
│   └── README.md          # Detailed documentation
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