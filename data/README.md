# Data Directory

This directory is intended for storing blood glucose datasets used in benchmark experiments.

## Directory Structure

```
data/
├── raw/                    # Original, unprocessed datasets
│   └── ohiot1dm/          # OhioT1DM XML data (download separately)
│       ├── 2018/{train,test}/
│       └── 2020/{train,test}/
```

## Supported Datasets

### OhioT1DM Dataset
- **Description**: Type 1 diabetes dataset with continuous glucose monitoring
- **Source**: Ohio University
- **Format**: Original XML files with CGM, insulin, meal, and activity data
- **Subjects**: Multiple patients with different data availability

The data is not redistributed in this repository. Request it from the
[OhioT1DM dataset page](https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html)
and follow its data-use terms.

## Usage

Data loading is handled automatically by the benchmark framework:

```python
from benchmark.data.loaders import load_ohiot1dm_data

data = load_ohiot1dm_data('data', patient_ids=[540], mode='train', version='2020')

print(data[540].columns)
```

## Adding New Datasets

1. Implement a loader in `benchmark/data/loaders.py`.
2. Register and validate it in `benchmark/configs/config_manager.py`.
3. Add unit and integration tests.
4. Document its on-disk format and access terms here.

## Notes

- This directory is excluded from version control via `.gitignore`
- Large datasets should be stored externally and downloaded as needed
- Always include proper licensing and attribution for datasets
