# Results Format

This file documents the layout `benchmark.cli run` writes and the analysis code
reads. Generated results are **not** stored here -- they go to the directory named
by `output.directory` in the experiment config, `results/experiments/` by default,
which is gitignored.

## Layout

One *parent experiment* per config. It holds the resolved configuration and the
cross-seed aggregate; each training mode and seed gets its own subdirectory.

```text
results/experiments/experiment_<timestamp>_<config-hash>_<run-id>/
├── tracking.json                # resolved config, environment, run status
├── resolved_config.yaml         # the config as actually applied
├── aggregate_metrics.json       # cross-seed summaries, one row per mode/patient/model
├── aggregate_metrics.csv        # the same rows, flattened
├── summary.md
├── regular/                     # training mode
│   └── seed_41/
│       ├── metrics.json         # this seed's per-patient metrics
│       ├── metrics.csv
│       ├── tracking.json
│       ├── resolved_config.yaml
│       ├── summary.md
│       └── patient_540/
│           ├── GRU_predictions.csv
│           ├── GRU_dashboard.png
│           ├── GRU_clarke_ega.png
│           └── GRU_parkes_ega.png
└── transfer/
    └── seed_41/ ...
```

A seed that fails leaves no directory: the runner keeps every seed that finished
rather than discarding the run, so a mode may hold fewer seeds than were
configured. `aggregate_metrics.json` records the seeds each row actually covers.

## File formats

**`metrics.json`** -- one entry per patient, keyed by patient ID as a string:

```json
{
  "540": {
    "mae": 30.93, "rmse": 40.85, "mape": 21.87, "mard": 21.87,
    "tir":           {"time_in_range": 9.04, "time_below_range": -4.65, "time_above_range": -4.39},
    "clarke_zones":  {"A": 57.64, "B": 35.55, "C": 0.30, "D": 6.51, "E": 0.0},
    "parkes_zones":  {"A": 61.10, "B": 35.96, "C": 2.71, "D": 0.22, "E": 0.0},
    "prediction_diagnostics": {"n_points": 2689, "n_implausible_predictions": 0, "...": "..."},
    "model_info":    {"model_name": "RNN", "patient_id": 540, "mode": "regular",
                      "seed": 41, "prediction_horizon_minutes": 60, "...": "..."},
    "training_history": {"train_losses": ["..."], "val_losses": ["..."]}
  }
}
```

`model_info` is what identifies a result. The model name, training mode, seed and
horizon live there rather than in the directory name, so nothing downstream has to
parse a path to know what it is reading.

TIR values are *differences* between predicted and true time-in-range, in
percentage points, so they are signed and legitimately negative.

**`aggregate_metrics.json`** -- a list of rows, one per mode/patient/model. Nested
metrics are flattened with dots and each leaf becomes a summary over seeds:

```json
[
  {
    "mode": "regular", "patient_id": 540, "model": "RNN", "seeds": [41, 42, 43],
    "metrics": {
      "mae":            {"mean": 31.24, "std": 0.30, "sem": 0.17,
                         "ci95_low": 30.51, "ci95_high": 31.98,
                         "min": 30.93, "max": 31.51, "n": 3},
      "clarke_zones.A": {"mean": 56.84, "...": "..."}
    }
  }
]
```

**`<MODEL>_predictions.csv`** -- written when `output.save_predictions` is enabled.
Columns: `true_glucose_mg_dl`, `predicted_glucose_mg_dl`, `absolute_error_mg_dl`.
Ground truth is fixed by the test split, so it is identical across seeds; only the
predictions move.

## Reading results in code

Do not parse these files directly. `benchmark.comparison.results_io` returns the
same object for every supported layout, including the older
`comprehensive_metrics_*.json` runs:

```python
from benchmark.comparison import load_experiment_results

results = load_experiment_results(
    "results/experiments/experiment_ID", mode="regular"
)
results.horizon_minutes       # 60, from the resolved config
results.models                # ('RNN',) -- read from the run, not assumed
results.seeds                 # (41, 42, 43)
results.metrics_for(540, "RNN")["mae"]
results.to_frame()            # tidy patient x model table
```

By default this returns the **cross-seed mean**; pass `seed=41` for a single seed.
A run holding more than one mode requires `mode=` rather than silently choosing
one. See the module docstring for why a seed mean exposes no prediction series.
