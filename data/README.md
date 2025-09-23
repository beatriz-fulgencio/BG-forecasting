# Data Directory

This directory is intended for storing blood glucose datasets used in benchmark experiments.

## Directory Structure

```
data/
├── raw/                    # Original, unprocessed datasets
│   ├── ohiot1dm/          # Ohio T1DM Dataset
│   └── custom/            # Custom datasets
├── processed/             # Preprocessed and cleaned datasets
└── metadata/              # Dataset descriptions and schemas
```

## Supported Datasets

### OhioT1DM Dataset
- **Description**: Type 1 diabetes dataset with continuous glucose monitoring
- **Source**: Ohio University
- **Format**: CSV files with CGM, insulin, meal, and activity data
- **Subjects**: Multiple patients with different data availability

### Custom Datasets
- **Description**: User-provided datasets
- **Format**: CSV with standardized column names
- **Required columns**: timestamp, subject_id, glucose
- **Optional columns**: insulin, carbs, activity, etc.

## Data Format Requirements

All datasets should be converted to a standardized format with the following columns:

### Required Columns
- `timestamp`: ISO format datetime (YYYY-MM-DD HH:MM:SS)
- `subject_id`: Unique identifier for each subject
- `glucose`: Blood glucose value in mg/dL

### Optional Columns
- `insulin`: Insulin dose in units
- `carbs`: Carbohydrate intake in grams
- `activity`: Activity level or type
- `sleep`: Sleep status or quality
- `stress`: Stress level
- `heart_rate`: Heart rate in BPM

## Usage

Data loading is handled automatically by the benchmark framework:

```python
from benchmark.data.loaders import load_dataset

# Load dataset
data = load_dataset('ohiot1dm', subjects=['559', '563'])

# Data is returned in standardized format
print(data.columns)  # ['timestamp', 'subject_id', 'glucose', ...]
```

## Adding New Datasets

1. Place raw data in appropriate subdirectory
2. Implement loader in `benchmark/data/loaders.py`
3. Add validation rules in `benchmark/data/validators.py`
4. Update configuration schema
5. Add documentation here

## Notes

- This directory is excluded from version control via `.gitignore`
- Large datasets should be stored externally and downloaded as needed
- Always include proper licensing and attribution for datasets