# Results Directory Structure

This directory contains the outputs from benchmark experiments:

## Directory Structure:
```
results/
├── experiments/          # Individual experiment results
│   ├── YYYY-MM-DD_HH-MM-SS_experiment_name/
│   │   ├── config.yaml          # Experiment configuration
│   │   ├── model.pkl           # Trained model (if saved)
│   │   ├── predictions.csv     # Model predictions
│   │   ├── metrics.json        # Evaluation metrics
│   │   ├── plots/              # Generated visualizations
│   │   └── logs/               # Training logs
├── comparisons/          # Cross-experiment comparisons
│   ├── comparison_YYYY-MM-DD/
│   │   ├── summary.csv         # Aggregate metrics
│   │   ├── statistical_tests.json
│   │   └── comparison_plots/
└── leaderboard/          # Best results tracking
    ├── overall_leaderboard.csv
    └── dataset_specific/
```

## File Formats:
- **config.yaml**: Experiment configuration in YAML format
- **predictions.csv**: Columns: [timestamp, subject_id, actual, predicted]
- **metrics.json**: Evaluation metrics in JSON format
- **logs/**: Training logs and debugging information