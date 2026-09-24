# Parkes (Consensus) Error Grid Analysis

Professional implementation of the Parkes Consensus Error Grid Analysis for evaluating blood glucose predictions in Type 1 diabetes.

## Overview

The Parkes Error Grid Analysis, also known as the Consensus Error Grid, was developed in 2000 by Joan L. Parkes and colleagues as a refinement of the Clarke Error Grid. It provides a consensus-based approach to assessing the clinical significance of glucose measurement errors, specifically designed for Type 1 diabetes management.

## Clinical Zones

The Parkes EGA uses five zones (A-E) with boundaries determined by clinical consensus:

### Zone A: Clinically Accurate
- **Definition**: Values leading to clinically correct treatment decisions
- **Clinical Impact**: No effect on clinical action
- **Target**: >95% of predictions
- **Treatment Decision**: Correct therapy

### Zone B: Benign Errors
- **Definition**: Values leading to benign or no treatment changes
- **Clinical Impact**: Minimal or no clinical impact
- **Acceptable**: <5% of predictions
- **Treatment Decision**: Acceptable deviation

### Zone C: Overcorrection Errors
- **Definition**: Values leading to unnecessary corrective action
- **Clinical Impact**: Overcorrection of acceptable glucose
- **Target**: 0% of predictions
- **Treatment Decision**: Potentially harmful

### Zone D: Dangerous Failure to Detect
- **Definition**: Values failing to detect potentially dangerous glucose levels
- **Clinical Impact**: Failure to treat when necessary
- **Target**: 0% of predictions
- **Treatment Decision**: Dangerous omission

### Zone E: Erroneous Treatment
- **Definition**: Values leading to opposite treatment
- **Clinical Impact**: Treatment confuses hypoglycemia with hyperglycemia (or vice versa)
- **Target**: 0% of predictions
- **Treatment Decision**: Critically dangerous

## Key Differences from Clarke EGA

The Parkes EGA differs from Clarke EGA in several important ways:

1. **Consensus-Based**: Boundaries determined by diabetes specialists' consensus
2. **Type 1 Specific**: Optimized for Type 1 diabetes management decisions
3. **Refined Boundaries**: More nuanced zone boundaries, especially in hypoglycemic range
4. **Modern Clinical Practice**: Reflects contemporary treatment approaches

## Usage

### Basic Analysis

```python
from benchmark.evaluation.parkes_ega import ParkesEGA
import numpy as np

# Your data
y_true = np.array([120, 85, 150, 200, 75])  # Reference glucose (mg/dL)
y_pred = np.array([118, 88, 145, 195, 78])  # Predicted glucose (mg/dL)

# Create analyzer
parkes = ParkesEGA()

# Perform analysis
results = parkes.analyze(y_true, y_pred)

# View results
print("Parkes Error Grid Analysis Results:")
print(f"Total points: {results['total_points']}")
print("\nZone Distribution:")
for zone in ['A', 'B', 'C', 'D', 'E']:
    count = results['zone_counts'][zone]
    percentage = results['percentages'][zone]
    print(f"  Zone {zone}: {count} points ({percentage:.2f}%)")
```

### Visualization

```python
import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Plot on existing axes
parkes.plot_on_axes(
    ax=ax,
    y_true=y_true,
    y_pred=y_pred,
    point_size=40,
    color_zones=True,
    color_points=True
)

plt.tight_layout()
plt.savefig('parkes_ega.png', dpi=300)
plt.show()
```

### Generate Complete Figure

```python
# Generate standalone figure
fig = parkes.plot(
    y_true=y_true,
    y_pred=y_pred,
    title="Parkes (Consensus) Error Grid Analysis",
    save_figure=True,
    filename='parkes_analysis.png'
)
```

## API Reference

### ParkesEGA Class

```python
class ParkesEGA:
    """Parkes (Consensus) Error Grid Analysis implementation."""
```

#### Methods

##### `analyze(y_true, y_pred)`

Performs Parkes EGA on prediction data.

**Parameters:**
- `y_true` (array-like): Reference glucose values in mg/dL
- `y_pred` (array-like): Predicted glucose values in mg/dL

**Returns:**
- `dict`: Results dictionary containing:
  - `zones` (ndarray): Zone classification (1-5) for each point
  - `zone_counts` (dict): Count of points in each zone (A-E)
  - `percentages` (dict): Percentage of points in each zone (A-E)
  - `total_points` (int): Total number of points analyzed

**Example:**
```python
results = parkes.analyze(y_true, y_pred)
print(f"Zone A: {results['percentages']['A']:.1f}%")
```

##### `plot(y_true, y_pred, **kwargs)`

Creates a complete Parkes EGA plot.

**Parameters:**
- `y_true` (array-like): Reference glucose values
- `y_pred` (array-like): Predicted glucose values
- `figsize` (tuple): Figure size in inches (default: (8, 8))
- `point_size` (float): Size of data points (default: 30)
- `title` (str): Plot title (default: "Parkes Error Grid Analysis")
- `save_figure` (bool): Whether to save figure (default: False)
- `filename` (str): Filename for saved figure (default: 'Parkes_EGA')
- `alpha` (float): Zone fill transparency (default: 0.5)
- `color_zones` (bool): Whether to color zones (default: True)
- `color_points` (bool): Whether to color points by zone (default: True)

**Returns:**
- `matplotlib.figure.Figure`: The generated figure

**Example:**
```python
fig = parkes.plot(
    y_true=y_true,
    y_pred=y_pred,
    title="My Model - Parkes EGA",
    save_figure=True,
    filename='my_parkes_analysis.png',
    point_size=50
)
```

##### `plot_on_axes(ax, y_true, y_pred, **kwargs)`

Plots Parkes EGA on an existing matplotlib axes.

**Parameters:**
- `ax` (matplotlib.axes.Axes): The axes to plot on
- `y_true` (array-like): Reference glucose values
- `y_pred` (array-like): Predicted glucose values
- `point_size` (float): Size of data points (default: 30)
- `alpha` (float): Zone fill transparency (default: 0.5)
- `color_zones` (bool): Whether to color zones (default: True)
- `color_points` (bool): Whether to color points by zone (default: True)

**Returns:**
- `matplotlib.axes.Axes`: The modified axes

**Example:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot Parkes EGA on first axes
parkes.plot_on_axes(ax1, y_true, y_pred)

# Plot Clarke EGA on second axes
from benchmark.evaluation.clarke_ega import ClarkeEGA
clarke = ClarkeEGA()
clarke.plot_on_axes(ax2, y_true, y_pred)

plt.tight_layout()
plt.show()
```

## Zone Classification Algorithm

The Parkes EGA uses a sophisticated piecewise linear boundary system developed through expert consensus:

### Mathematical Boundaries

The zone boundaries are defined by consensus-derived piecewise linear functions. Key boundary points include:

**Lower Hypoglycemic Region (Reference < 70 mg/dL):**
- More sensitive to detection failures
- Stricter boundaries than Clarke EGA
- Emphasis on avoiding dangerous hypoglycemia

**Euglycemic Region (70-180 mg/dL):**
- Tolerance for small deviations
- Focus on avoiding unnecessary treatment changes

**Hyperglycemic Region (Reference > 180 mg/dL):**
- Graduated response to prediction errors
- Balance between correction and overcorrection

The implementation uses optimized NumPy operations with vectorized boundary checks for efficient processing.

## Clinical Interpretation

### Regulatory and Clinical Standards

**Consensus Guidelines:**
- Zone A: >95% for clinical acceptance
- Zone A+B: ≥99% combined
- Zone C+D+E: <1% combined
- Zone D+E: 0% for regulatory approval

**Clinical Research Standards:**
- Zone A ≥97%: Excellent performance
- Zone A ≥95%: Good performance
- Zone A ≥90%: Acceptable performance
- Zone A <90%: Needs improvement

### Model Quality Assessment

| Zone A Percentage | Clinical Assessment | Recommendation                         |
|-------------------|---------------------|----------------------------------------|
| >97%              | Excellent           | Optimal for Type 1 diabetes management |
| 95-97%            | Very Good           | Suitable for clinical deployment       |
| 90-95%            | Good                | Acceptable with caution                |
| 85-90%            | Moderate.           | Requires improvement                   |
| <85%              | Poor                | Not suitable for clinical use          |

## Visualization Features

### Color Schemes

Consistent with Clarke EGA for easy comparison:

- **Zone A**: Green (clinically accurate)
- **Zone B**: Light green (benign errors)
- **Zone C**: Yellow (overcorrection)
- **Zone D**: Orange (dangerous failure)
- **Zone E**: Red (erroneous treatment)

### Plot Elements

1. **Zone Boundaries**: Black lines showing consensus-based boundaries
2. **Perfect Agreement Line**: Dashed diagonal (45°) line
3. **Data Points**: Colored by zone classification
4. **Legend**: Zone descriptions with counts and percentages
5. **Grid**: Light gridlines for reference
6. **Axes**: Labeled with glucose concentration (mg/dL)
7. **Title**: Indicates "Parkes (Consensus)" to distinguish from Clarke

### Integration with Statistical Metrics

```python
from benchmark.evaluation.metrics import BGMetrics

# Comprehensive evaluation
metrics = BGMetrics.calculate_comprehensive_metrics(y_true, y_pred)

print("Comprehensive Evaluation Report:")
print("\nStatistical Metrics:")
print(f"  RMSE: {metrics['rmse']:.2f} mg/dL")
print(f"  MAE: {metrics['mae']:.2f} mg/dL")
print(f"  MAPE: {metrics['mape']:.2f}%")

print("\nClarke Error Grid:")
print(f"  Zone A: {metrics['clarke_zones']['A']:.2f}%")
print(f"  Zone A+B: {metrics['clarke_zones']['A'] + metrics['clarke_zones']['B']:.2f}%")

print("\nParkes Error Grid:")
print(f"  Zone A: {metrics['parkes_zones']['A']:.2f}%")
print(f"  Zone A+B: {metrics['parkes_zones']['A'] + metrics['parkes_zones']['B']:.2f}%")

print("\nClinical Assessment:")
if (metrics['rmse'] < 15 and 
    metrics['parkes_zones']['A'] > 95 and
    metrics['parkes_zones']['D'] + metrics['parkes_zones']['E'] == 0):
    print("✓ Excellent: Meets all clinical standards for Type 1 diabetes")
elif (metrics['rmse'] < 20 and metrics['parkes_zones']['A'] > 90):
    print("✓ Good: Acceptable for Type 1 diabetes management")
else:
    print("⚠ Needs improvement for Type 1 diabetes applications")
```

## Troubleshooting

### Common Issues

**Issue: Different results from Clarke EGA**
```python
# This is expected! Parkes uses different boundaries
# Both analyses are valid but for different purposes

clarke_results = ClarkeEGA().analyze(y_true, y_pred)
parkes_results = ParkesEGA().analyze(y_true, y_pred)

print("This is normal:")
print(f"  Clarke Zone A: {clarke_results['percentages']['A']:.2f}%")
print(f"  Parkes Zone A: {parkes_results['percentages']['A']:.2f}%")
print("\nParkes is more stringent for Type 1 diabetes")
```

**Issue: Unit conversion needed**
```python
# If your data is in mmol/L, convert to mg/dL
def mmol_to_mgdl(glucose_mmol):
    return glucose_mmol * 18.018

y_true_mgdl = mmol_to_mgdl(y_true_mmol)
y_pred_mgdl = mmol_to_mgdl(y_pred_mmol)

results = parkes.analyze(y_true_mgdl, y_pred_mgdl)
```

**Issue: Unexpected zone distribution**
```python
# Validate your data range
print(f"Reference range: {y_true.min():.1f}-{y_true.max():.1f} mg/dL")
print(f"Prediction range: {y_pred.min():.1f}-{y_pred.max():.1f} mg/dL")

# Check for outliers
outliers = (y_pred < 20) | (y_pred > 500)
if np.any(outliers):
    print(f"Warning: {np.sum(outliers)} outlier predictions detected")
```

## References

1. **Original Paper**:
   Parkes, J. L., Slatin, S. L., Pardo, S., & Ginsberg, B. H. (2000). A new consensus error grid to evaluate the clinical significance of inaccuracies in the measurement of blood glucose. *Diabetes Care*, 23(8), 1143-1148.

2. **Clinical Applications**:
   - Kovatchev, B. P. (2017). "Metrics for glycaemic control—from HbA1c to continuous glucose monitoring." *Nature Reviews Endocrinology*, 13(7), 425-436.

3. **Consensus Guidelines**:
   - Klonoff, D. C., et al. (2013). "Clinical accuracy of a continuous glucose monitoring system." *Diabetes Technology & Therapeutics*, 15(10), 881-888.

4. **Type 1 Diabetes Standards**:
   - American Diabetes Association. (2021). "Standards of Medical Care in Diabetes."

## Acknowledgements

This project adapts code from [Parkes_EGA_MATLAB](https://github.com/4OH4/Parkes_EGA_MATLAB) by [4OH4](https://github.com/4OH4) to python. Retrieved October 5, 2025. 
License: MIT (included as [license](LICENSE.4OH4))

## See Also

- [Clarke Error Grid Analysis](../clarke_ega/README.md) - Original error grid analysis
- [Evaluation Module README](../README.md) - Complete evaluation module documentation
- [Metrics Documentation](../metrics.py) - Statistical metrics implementation

## License

This implementation is part of the BG-Forecasting project. See the main LICENSE file for details.
