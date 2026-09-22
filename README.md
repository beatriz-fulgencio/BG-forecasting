# Blood Glucose Forecasting Benchmark

A reproducible benchmark framework for comparing blood glucose prediction models on
OhioT1DM across evaluation metrics and experimental protocols.

The repository trains a matrix of recurrent models (RNN, LSTM, GRU) at four prediction
horizons under two learning protocols — 24 model x horizon x mode cells — plus a
30-minute Transformer reproducing a published configuration, evaluates each with point
and clinical metrics, and runs thirteen analysis stages over the results, from Clarke
zone-D bootstraps to distributional-shift permutation tests. Every stage exports the
table behind each figure it draws.

- **Reproducible experiments** — one YAML config per run, a resolved copy written next
  to the results, multi-seed aggregation with explicit dispersion
- **Clinical evaluation** — Clarke and Parkes error grids, time-in-range, error by
  glycemic band, error during rapid glucose change
- **Two learning protocols** — per-patient regular learning and population-pretrained
  transfer learning, with paired inference between them
- **OhioT1DM integration** — 2018 and 2020 cohorts, 12 patients, mixable in one run
- **Publication figures** — per-patient error grids and dashboards, error-localization
  panels, t-SNE embeddings, shift and transfer plots ([gallery](docs/FIGURES.md))

---

## What's in this repository

### Models — `benchmark/models/`

| `model.type` | Class | Architecture |
| --- | --- | --- |
| `rnn` | `RNNBGModel` | stacked `nn.RNN` -> last timestep -> `Linear(hidden, horizon)`, direct multi-step |
| `lstm` | `LSTMBGModel` | stacked `nn.LSTM`, same head |
| `gru` | `GRUBGModel` | stacked `nn.GRU`, same head |
| `transformer` | `TransformerBGModel` | pre-norm causal self-attention encoder, **autoregressive**: one step at a time, each prediction appended to the window |

Recurrent hyperparameters under `model.architecture`: `hidden_size` (64),
`num_layers` (2), `dropout` (0.2, forced to 0 with a single layer), and
`nonlinearity` (`tanh`/`relu`, RNN only). Transformer: `d_model` (128), `nhead` (4,
must divide `d_model`), `num_layers` (3), `dim_feedforward` (512), `dropout` and
`attention_dropout` (0.1). Training is Adam + MSE with gradient clipping,
`ReduceLROnPlateau`, early stopping and best-validation restore
(`benchmark/models/base_model.py`).

CNN, encoder-decoder and cross-attention variants appear only as TODO comments. They
are not implemented.

### Metrics — `benchmark/evaluation/`

`BGMetrics.calculate_comprehensive_metrics` returns, in mg/dL:

| Key | Meaning |
| --- | --- |
| `rmse`, `mae`, `mape`, `mard` | point-error metrics, NaN-skipping |
| `clarke_zones` | Clarke A-E, percent |
| `parkes_zones` | Parkes A-E, percent (Type 1 boundaries) |
| `time_in_range`, `time_below_range`, `time_above_range` | share of *predicted* readings in 70-180 / below 70 / above 180 |
| `n_samples`, `n_missing`, `n_outside_grid_domain` | what was evaluated and what was excluded |

`BGEvaluator.compute_metrics` selects from these by name
(`mae`, `rmse`, `mape`, `mard`, `tir`, `clarke`/`clarke_ega`, `parkes`/`parkes_ega`)
and validates units, shape and finiteness first. `tir` is stored as *prediction minus
reference* percentage points, so it reads as compression or expansion of time in range
rather than as an absolute figure. `benchmark/glucose_ranges.py` holds the two ranges
the whole codebase shares: the 20-600 mg/dL plausibility filter and the 40-400 mg/dL
CGM sensor range.

### Data — `benchmark/data/`

OhioT1DM only. 2018 release: patients 559, 563, 570, 575, 588, 591. 2020 release:
540, 544, 552, 567, 584, 596. `data.version: both` resolves all twelve, and an
explicit patient list may mix the two.

`loaders.py` parses every signal in the XML (CGM, fingerstick, basal and temp basal,
bolus, meal, exercise, sleep, work, heart rate, GSR, skin temperature) onto a
five-minute grid. `preprocessors.py` expands temporal events over their durations,
adds cyclical hour-of-day features, and implements `handle_missing_data` with the
strategies `none`, `interpolate`, `forward_fill` and `drop`. `torch_dataset.py` builds
windows that contain no missing value, standardizes with train statistics reused at
test time, and exposes the inverse transforms and prediction context the analyses need.

Features are either unimodal (`glucose`) or multimodal
(`glucose`, `basal`, `bolus`, `carbs` plus the cyclical hour columns).

### Analysis — `benchmark/analysis/`

Split by what the answer is a property of:

- `analysis/dataset/` — fixed by the cohort and its train/test split, the same numbers
  whichever model was trained: distributional shift, signal irregularity, per-patient
  glucose features, clinical profile shift, population glucose history.
- `analysis/experiments/` — properties of what a model did: zone-D rates, error by
  glycemic band, calm versus rapid change, persistence baselines, transfer inference,
  patient stability, cross-horizon and cross-model significance, error localization.

Both halves read results through one shared reader, `benchmark.analysis.results_io`.

### Configs — `configs/` (29 files)

| Group | Files | Purpose |
| --- | --- | --- |
| Template | `default.yaml` | the documented version-1 schema |
| Examples | 6 x `example_*.yaml` | small real-data runs, one per training mode, plus a cross-release one |
| Production | 13 x `full_*.yaml` | the published matrix: 12 patients, both releases, `mode: both`, seeds 41-43 |
| RL vs TL | 5 x `rl_vs_tl_*.yaml` | the regular-versus-transfer comparison arm |
| Smoke | 2 x `smoke_full_pipeline_*.yaml` | cheapest run that still exercises every downstream stage |
| Sensitivity | `converged_gru_30min.yaml`, `verify_gru_30min_p540.yaml` | convergence arm and a single-patient reproduction check |

Per-patient figures are written only when a config sets `output.generate_plots: true`.
That is the case for `default.yaml`, `example_experiment.yaml` and every `full_*`
config; the remaining example, smoke and `rl_vs_tl_*` configs deliberately turn it off,
so those runs produce metrics and predictions but no images.

### Tests — `tests/`

648 tests in 32 modules, run with `pytest tests` from the repository root
(`pip install -e ".[dev]"` first). `tests/conftest.py` puts `RUN/` on the path and
caps BLAS threads before numpy loads.

Several exist to keep a specific claim honest:

- `test_reference_conformance.py` — every hand-written metric and statistical test is
  pinned against an independent reference implementation
- `test_clarke_boundaries.py` — reads the drawn boundaries and zone letters back out of
  the rendered axes and checks them against `ClarkeEGA.analyze`
- `test_training_schedule.py` — pins the two-stage transfer schedule to Cui et al.,
  including the two deliberate departures, so neither is silently "corrected"
- `test_transfer_pretraining_splits.py` — population pretraining must never see a
  held-out test split
- `test_target_recovery_ohiot1dm.py` — the standardize/inverse round trip recovers
  exact CGM readings
- `test_handle_missing_data.py` — every missing-data strategy measures gaps in elapsed
  time, not row count
- `test_out_of_domain_predictions.py` — regression test for the seeds killed by
  negative forecasts reaching the Clarke domain check

---

## Quick Start

### 1. Install

Python 3.9 or newer is required.

```bash
git clone https://github.com/beatriz-fulgencio/BG-forecasting.git
cd BG-forecasting
pip install -r requirements.txt
pip install -e .
# Development tools (pytest, black, flake8, mypy):
pip install -e ".[dev]"
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
bg-forecast run --config configs/example_experiment.yaml
# Equivalent without the installed console script:
python -m benchmark.cli run --config configs/example_experiment.yaml
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

To select patients from both OhioT1DM releases in one experiment, set
`data.version: both`. With `data.patients: all`, this resolves all 12 patients;
an explicit list can mix IDs from 2018 and 2020. The loader reads each patient
from the correct release and the tracking manifest records that mapping.
[`example_combined_ohiot1dm.yaml`](configs/example_combined_ohiot1dm.yaml)
is a two-patient cross-release GRU smoke run using both training modes.

### Training modes

`training.mode` is required and has three values:

The compact examples are [`example_regular.yaml`](configs/example_regular.yaml), [`example_transfer.yaml`](configs/example_transfer.yaml), and [`example_both.yaml`](configs/example_both.yaml). For a named regular-learning versus transfer-learning comparison, use [`example_rl_vs_tl.yaml`](configs/example_rl_vs_tl.yaml).

- `regular`: train a separate model on each target patient's data.
- `transfer`: pretrain on the training splits of the other selected patients,
  then fine-tune for each target patient. The target patient and every test
  split are excluded from pretraining. At least two patients are required.
- `both`: execute and track both protocols as separate runs using the same data and model configuration.

Optimization settings (`epochs`, `batch_size`, `learning_rate`, early stopping,
`seeds`, and device) belong under `training`. `training.seeds` is a required,
non-empty list; use `[42]` for one run or `[42, 43, 44]` for multiple runs.
Aggregate exports report the mean, sample standard deviation (`ddof=1`),
standard error, a 95% Student-t interval, minimum, maximum, and count across
seeds. A run with one seed reports `null` for the dispersion fields. If a seed
fails, the others still aggregate and the failure is listed under `failed_runs`,
with the run marked `completed_with_failures` so a partial result is never read
as a whole one. Network shape belongs under
`model.architecture`. See
[`configs/default.yaml`](configs/default.yaml) for the
complete version-1 schema.

---

## Clarke and Parkes error grids

Both grids are implemented in this repository as ports of published MATLAB
implementations, each kept beside its own upstream licence.

| | Clarke | Parkes |
| --- | --- | --- |
| Module | `benchmark/evaluation/clarke_ega/` | `benchmark/evaluation/parkes_ega/` |
| Zones | A-E | A-E, plus an out-of-range bucket excluded from the reported percentages |
| Domain | 0-400 mg/dL | 0-550 mg/dL |
| Ported from | Edgar Guevara Codina's MATLAB, BSD ([`LICENSE.Guevara`](benchmark/evaluation/clarke_ega/LICENSE.Guevara)) | Rupert Thomas / 4OH4's MATLAB, MIT ([`LICENSE.40H4`](benchmark/evaluation/parkes_ega/LICENSE.40H4)) |
| Reference | Maran et al. 2002; Kovatchev et al. 2004 | Pfützner et al. 2013, *Technical Aspects of the Parkes Error Grid* |

<p align="center">
  <img src="docs/figures/clarke_ega.png" alt="Clarke error grid" width="49%">
  <img src="docs/figures/parkes_ega.png" alt="Parkes error grid" width="49%">
</p>

**Parkes is Type 1 only.** `parkes_ega/boundaries.py` defines `get_boundaries_type1`
and nothing else. The `diabetes_type` argument accepted by `plot_parkes_analysis` and
`create_prediction_dashboard` changes the figure title and nothing about the geometry,
so passing `2` draws Type 1 boundaries under a Type 2 label. Region overlap is resolved
E -> D -> C -> B -> A, matching the MATLAB original.

The Clarke grid's zone letters are placed by `ZONE_LABEL_ANCHORS`, and every anchor is
a point that `analyze` actually classifies into that zone. `tests/test_clarke_boundaries.py`
reads the letters and boundary segments back out of the rendered axes and checks them,
because a letter drawn in the wrong region mislabels a figure while leaving every
reported number correct — exactly the failure that went unnoticed in a published
figure. Zones B, C, D and E each occupy two disjoint regions and are therefore
labelled twice.

### Nothing is clipped before a reported metric

Clinical metrics and prediction exports use glucose in mg/dL. Each subrun records the
prediction horizon in steps and minutes, the target units, and the 20-600 mg/dL
plausibility range used before evaluation. Targets outside that range abort the run,
because they can only come from broken data or a broken inverse transform.

Predictions are evaluated exactly as the model produced them. An earlier version
clipped both arrays onto the CGM range for the error grids alone, which meant the
point-error and grid columns of one results row described two different predictors.
Instead:

- a pair falling outside a grid's own domain is **excluded** from that grid's
  percentages, counted in `n_outside_grid_domain`, and reported through an
  `OutOfDomainWarning` naming how many were dropped. Clipping is rejected because it
  can only move a point to a better zone — a prediction of 500 against a reference of
  380 would become Zone A — and so silently flatters the result;
- `prediction_diagnostics` counts, per subrun and aggregated across seeds,
  `n_implausible_predictions` (outside 20-600 mg/dL) and
  `n_predictions_outside_sensor_range` (outside the 40-400 mg/dL sensor range), along
  with the observed prediction and target extremes. These are counts, not corrections:
  an implausible prediction is a property of the model that the benchmark should
  measure rather than a reason to stop.

Clipping to 40-400 mg/dL survives in exactly two downstream analyses, where the
question is about the clinical grid rather than about point error: the Clarke zone-D
bootstrap (`benchmark/analysis/experiments/zone_d.py`) and the localized Clarke figure
(`error_localization.py`), for the grid only. Both say so in their exported metadata.

---

## Figures

Training runs and analysis stages together produce a family of publication figures.
The full gallery, with a catalogue of every figure the code can emit and the command
that produces each, is in **[`docs/FIGURES.md`](docs/FIGURES.md)**.

The per-patient dashboard collects the whole evaluation of one patient onto one page —
the prediction trace, the metrics, both error grids, and worked single-window
forecasts:

![Prediction dashboard](docs/figures/prediction_dashboard.png)

The analysis stages localize the error rather than summarising it. This is the Clarke
grid for one patient drawn twice, coloured by glycemic range and by rapid glucose
change:

![Localized Clarke grid](docs/figures/clarke_localized.png)

Where figures land:

| Figure | Location |
| --- | --- |
| Error grids, dashboard | `<experiment>/<mode>/seed_<n>/patient_<id>/` |
| Error localization, pooled Clarke | `<experiment>/analysis/figures_<mode>/` |
| Single-seed Clarke grids | `results/analysis/glycemic_analysis/<model>_<min>min_<mode>/` |
| Cohort-level plots (shift, transfer, persistence, stability, t-SNE) | `results/analysis/<stage>/` |
| Dataset-only plots | `results/analysis/dataset/patient_analysis/` |

Everything is PNG: 300 dpi for the per-patient and paper figures, 200 dpi for the
cohort line and error-bar plots.

### Per-patient error-localization figures

`figures` draws the two per-patient publication figures for each recorded cell:
error magnitude and direction across the five ISO glycemic ranges, calm versus
rapid glucose change, and the Clarke grid coloured by each of those two
splits. The five bands nest exactly inside the three that the `error-range`
bootstrap reports, which `tests/test_error_localization.py` enforces, so the
figure and the intervals beside it always describe the same subsets.

These are single-patient illustrations, not cohort estimates, and they carry no
confidence intervals — the uncertainty for the same quantities is what
`error-range` and `rapid-change` already report. Both stages draw every patient
of every recorded cell, in one training mode:

```bash
bash RUN/run_experiments_analysis.sh figures     # error localization, per patient, into the run dirs
bash RUN/run_experiments_analysis.sh glycemic    # Clarke grids -> analysis/glycemic_analysis/
FIGURE_PATIENTS="540 584" FIGURE_MODE=regular \
  bash RUN/run_experiments_analysis.sh figures   # narrow it while iterating
```

`glycemic_analysis/` holds one folder per model × horizon × mode, each with both
forms of the Clarke grid for all 12 patients — the single-seed form and the
pooled-seed one — as PNG, plus the zone × band table behind each.

Generated filenames carry no figure number: the numbering belongs to one
submission rather than to the code. `ARTICLE_REPRODUCIBILITY_AUDIT.md` maps each
article figure to the file that produces it.

For one cell on its own, pass each training seed the way `zone-d` takes them:

```bash
python -m RUN.figures.run_error_localization_figures \
  --run-dir results/experiments/experiment_ID/transfer/seed_41 \
  --run-dir results/experiments/experiment_ID/transfer/seed_42 \
  --run-dir results/experiments/experiment_ID/transfer/seed_43 \
  --all-patients
```

Every number a caption could quote is exported next to the images —
`error_by_iso_band_<patient>_<model>.csv`, `error_by_condition_…csv`,
`clarke_zone_by_band_…csv` and `figure_metadata_…json` — so no figure claim is
left without a file behind
it. A band holding too few points to support a mean is drawn hatched and
labelled with its count rather than dropped, and flagged `interpretable=False`
in the exported table.

---

## Methods and protocol decisions

### 30-minute Transformer replication

The original publication matrix remains RNN, LSTM, and GRU at 15, 30, 45, and
60 minutes. The Transformer is an explicit 30-minute extension, shared by
regular and transfer learning.

It is **not tuned here**. Its architecture and optimiser settings are Cui et
al.'s published configuration
([GluPred](https://github.com/r-cui/GluPred), MIT, `utils/opt.py`): num_layers
3, d_model 128, heads 4, d_ff 512, dropout 0.1, attention_dropout 0.1, pre_lr
3e-4, ft_lr 5e-5, 10 + 10 epochs, batch 16. A second hyperparameter search
would make this row a different kind of result from the others, so it is a
reproduction rather than a search. Every non-architecture field in
[`configs/full_transformer_30min.yaml`](configs/full_transformer_30min.yaml)
is copied verbatim from `full_gru_30min.yaml` -- same cohort, split, seeds and
preprocessing -- so a rank disagreement between the two can only come from the
architecture.

That configuration is committed, so the run needs no preparatory step:

```bash
MODELS=transformer HORIZONS=30 bash RUN/run_training.sh
```

Analysis stages discover recorded models from `parents.txt`. The Transformer
therefore receives the patient, seed-aware, TL−RL, glucose-range, and Clarke
Zone-D analyses for its 30-minute cell. Cross-horizon inference records an
explicit single-horizon skip, while cross-model rankings are partitioned by
horizon and training mode.

### Missing CGM data

Gaps in the glucose series are **not** filled. A missing reading stays as NaN,
holding its place on the uniform sampling grid, and the dataset skips any window
that touches one — so the model never sees a gap and never learns across one.

This is a deliberate methods choice rather than a default left unexamined. Any
gap-filling strategy puts a value the sensor never reported into the prediction
target, which would mean MAE, Clarke EGA and the zone-D analysis were scored
partly against invented readings. `handle_missing_data` still offers
`interpolate`, `forward_fill` and `drop` for input-side experiments; each warns
that its output must not be used for reported clinical metrics, and `drop` in
particular breaks the uniform grid so windows can silently span a gap.

All four strategies measure a gap in elapsed time rather than in row count. The
series is on a five-minute grid only where the sensor reported, so an outage can
appear either as NaN rows or as rows that are simply absent, and consecutive
rows are sometimes hours apart; a row-based limit would let `interpolate` or
`forward_fill` bridge an arbitrarily long outage in `max_gap` steps. `max_gap`
is therefore converted to `max_gap * sampling_rate` minutes and enforced against
the timestamps: `forward_fill` carries a value only while it is still that
recent, and `interpolate` judges a gap whole, bridging it only if the entire
span is within the limit rather than filling its first few rows from an anchor
far outside it. The count of values actually filled, and of those left as NaN
beyond the limit, is reported per patient.

The cost is about 1.4% of otherwise usable training windows (0.8–3.2% depending
on the patient). `tests/test_target_recovery_ohiot1dm.py` enforces the
invariant: every recovered prediction target is a whole mg/dL reading inside the
sensor's 40–400 range.

### Clinical inference

For publication clinical analysis, set `output.save_predictions: true`. After
each configured mode/horizon cell, pass every completed training seed to the
nested bootstrap, for example:

```bash
python -m RUN.experiments.run_zone_d_analysis \
  --run-dir results/experiments/experiment_ID/regular/seed_42 \
  --run-dir results/experiments/experiment_ID/regular/seed_43 \
  --run-dir results/experiments/experiment_ID/regular/seed_44
```

This exports `zone_d_analysis/summary.json` and `per_patient.csv`, including
patient-cluster and training-seed bootstrap 95% intervals for the Clarke
zone-D rate and the missed-hypoglycemia share of zone D. Keep mode and horizon
constant across the repeated `--run-dir` arguments; analyze other cells
separately.

The shift–difficulty analysis uses the same patient-level independence rule:
`RUN/experiments/run_shift_analysis.py` writes whole-patient permutation tests,
per-seed sensitivity results, and nested patient × seed uncertainty when parent
configured aggregate runs are supplied. See
[`SHIFT_DIFFICULTY_INFERENCE.md`](SHIFT_DIFFICULTY_INFERENCE.md) for the
method, reporting rules, and reproducible invocation.
Transfer-benefit runs additionally write a paired TL−RL MAE effect-size table
with patient-level bootstrap intervals for each horizon cell.
See [`MULTIPLE_COMPARISON_GUIDANCE.md`](MULTIPLE_COMPARISON_GUIDANCE.md) for
the recommended testing-family and p-value reporting policy.
For the specific Pearson/Spearman rule, see
[`CORRELATION_MULTIPLE_COMPARISON.md`](CORRELATION_MULTIPLE_COMPARISON.md).

---

## Analyse the dataset and compare experiments

`benchmark/analysis/` is split by what the answer is a property of.
`benchmark/analysis/dataset/` holds the analyses fixed by the OhioT1DM cohort and
its train/test split — distributional shift, signal irregularity, per-patient
glucose features, population glucose history — which give the same numbers
whichever model was trained. `benchmark/analysis/experiments/` holds the analyses of
what a model did: significance testing, error localization, persistence
baselines, transfer inference, patient stability.

Both halves read results through the one shared reader,
`benchmark.analysis.results_io`, which normalises the multi-seed layout
the runner writes and the older single-run layout into one object, so no analysis
parses a directory name or assumes which models a run contains:

```python
from benchmark.analysis import load_experiment_results

results = load_experiment_results("results/experiments/experiment_ID", mode="regular")
results.horizon_minutes, results.models, results.seeds   # 60, ('RNN',), (41, 42, 43)
results.to_frame()                                       # tidy patient x model table
```

Reads default to the **cross-seed mean**; `seed=41` selects one seed instead. A
run holding both `regular` and `transfer` requires `mode=`, rather than one being
chosen silently.

To test whether forecast quality differs across horizons, with patients paired
across horizons and p-values corrected over the whole family of horizon pairs:

```bash
python RUN/experiments/run_comparison_analysis.py \
  --experiments results/experiments/experiment_* \
  --mode regular \
  --output-dir results/analysis/horizons_regular
```

Runs are grouped by model, so each architecture gets its own horizon family and no
test compares an LSTM against a GRU as if they were one estimator at two horizons.
Each family writes `horizon_comparison.csv`, a readable `horizon_comparison.txt`,
and the `metrics_by_patient.csv` the tests were computed from. This is also stage
`horizons` of `RUN/run_experiments_analysis.sh`.

---

## Running the full pipeline

`RUN/` has three drivers and one shared configuration file.
`RUN/run_training.sh` trains every model, horizon, mode and seed, recording each
finished cell in `results/full_run_logs/parents.txt`.
`RUN/run_experiments_analysis.sh` reads that list and analyses model results;
`RUN/run_dataset_analysis.sh` reads one recorded run only for cohort settings
and runs all dataset-only analyses, including a reusable per-patient signal
table (`results/analysis/dataset/dataset_signal_features.csv`). The experiment
`shift` stage reads that table and checks its cohort, releases, sampling rate,
and raw data root against the selected runs. Run the dataset driver first;
the shift stage fails if its table is missing or incompatible. Shared settings live in
`RUN/_common.sh`.

```bash
bash RUN/run_training.sh           # train everything; hours
bash RUN/run_experiments_analysis.sh --check   # verify inputs, run nothing
bash RUN/run_dataset_analysis.sh               # every dataset-only analysis
bash RUN/run_experiments_analysis.sh           # every analysis stage
```

### Experiment stages, in order

| Stage | What it answers | Cost |
| --- | --- | --- |
| `zone-d` | Clarke zone-D nested patient x seed bootstrap | ~1 min, all cells |
| `error-range` | error by glycemic band, nested bootstrap | ~1 min, all cells |
| `rapid-change` | calm versus rapid glucose change, nested bootstrap | ~2 min per cell |
| `figures` | per-patient error-localization figures | ~1 min per cell |
| `glycemic` | Clarke grids per patient, paper and pooled forms | ~4 min, all cells |
| `tsne` | t-SNE of test-period glucose windows | ~5 min, one cell |
| `persistence` | forecast-origin persistence baseline | ~4 min |
| `shift` | shift-difficulty permutation and bootstrap tests | ~5 min per model |
| `transfer` | confirmatory paired TL-RL inference, per model | seconds |
| `stability` | patient-difficulty stability across conditions | seconds |
| `horizons` | cross-horizon significance per model and mode | seconds |
| `compare` | cross-model comparison | seconds |
| `summary` | where everything landed | instant |

Cross-seed aggregates are not among them — training writes `aggregate_metrics.json`
per run and the stages read it.

### Dataset stages

`population` (per-patient train/test glucose history and descriptive statistics),
`features` (per-patient level, variability, clinical-range and dynamics features),
`clinical` (train-to-test change in clinical CGM profile), and `signal`
(distribution shift and signal irregularity, writing the
`dataset_signal_features.csv` that the experiment `shift` stage requires).

### Shared settings — `RUN/_common.sh`

`MODELS`, `HORIZONS`, `SEEDS`, `CONFIG_PREFIX`, `LOG_DIR`, `ANALYSIS_DIR`,
`DATA_ROOT`, `REPLICATES` (20000), `PERMUTATIONS` (10000), `RAPID_THRESHOLD`
(15 mg/dL per 5-minute interval, the CGM double-arrow boundary),
`FIGURE_PATIENTS` / `FIGURE_MODE` / `FIGURE_MODEL` / `FIGURE_HORIZON` /
`FIGURE_SEED`, `TEST_FAMILY` (precommitted non-parametric) and `CORRECTION` (Holm).

### Preflight

The preflight checks the things that would otherwise fail only after the
compute: importable dependencies, a run list whose directories all exist, raw
CGM data for the shift analysis, per-patient prediction CSVs for the zone-D
bootstrap, seed directories left half-written by an interrupted training run, and
regular/transfer seed sets that do not match — which is what the paired transfer
analyses refuse to run on. The shift stage dominates the runtime at roughly 5 minutes per
model; `--quick` cuts the resample counts for a smoke check and is not suitable
for publication.

For a single run, `PatientAnalyzer` covers per-patient performance and
`PopulationAnalysis` the population view; both take the same `mode` and `seed`
arguments.

---

## Command line

```bash
bg-forecast run     --config configs/example_experiment.yaml
bg-forecast resume  --run-dir results/experiments/experiment_ID/regular/seed_42
bg-forecast analyze --experiment-dir results/experiments/experiment_ID
bg-forecast compare --experiments results/experiments/experiment_A results/experiments/experiment_B
bg-forecast list
```

`resume` continues one interrupted mode/seed directory, using the `tracking.json` and
`resolved_config.yaml` written beside it. `analyze` prints metrics as JSON and accepts
`--metrics`. `compare` takes two or more experiments and writes `comparison.json` with
per-model results and an MAE ranking partitioned by horizon and mode. `list` prints the
supported models, datasets and metrics. Exit codes: `0` success, `2` configuration
error, `1` other failure, `130` interrupt.

Batch execution is intentionally not part of the CLI.

---

## Repository Structure

```text
BG-forecasting/
├── benchmark/                  # Main benchmark package
│   ├── cli.py                  # bg-forecast entry point
│   ├── glucose_ranges.py       # shared plausibility and sensor ranges
│   ├── data/                   # OhioT1DM loading, preprocessing, torch datasets
│   ├── models/                 # rnn.py (RNN/LSTM/GRU), transformer.py, base_model.py
│   ├── evaluation/             # metrics, evaluator, visualisation, reporting
│   │   ├── clarke_ega/         # Clarke error grid (ported, BSD)
│   │   └── parkes_ega/         # Parkes error grid (ported, MIT)
│   ├── analysis/               # results_io + dataset/ and experiments/ analyses
│   ├── configs/                # config loading and schema validation
│   ├── experiments/            # runner, configured execution engine, tracking
│   ├── utils/                  # logging and I/O helpers
│   └── README.md               # package documentation
├── RUN/                        # Drivers: dataset/, experiments/, figures/ + shell entry points
├── configs/                    # Experiment configurations
├── docs/                       # FIGURES.md and the example images it shows
├── tests/                      # 648 tests
├── data/                       # OhioT1DM goes here (gitignored)
├── results/                    # Generated experiment outputs (default, gitignored)
├── requirements.txt
├── setup.py
└── README.md                   # This file
```

## Documentation

- [`docs/FIGURES.md`](docs/FIGURES.md) — every figure, with examples and the command that draws it
- [`benchmark/README.md`](benchmark/README.md) — package-level setup, configuration and API reference
- [`data/README.md`](data/README.md) — expected data layout and dataset terms
- [`RESULTS_CATALOGUE.md`](RESULTS_CATALOGUE.md) — inventory of every result the pipeline currently produces
- [`COMPARISON_ANALYSES.md`](COMPARISON_ANALYSES.md) — every comparison in the codebase, its independent unit, its test, its correction
- [`SHIFT_DIFFICULTY_INFERENCE.md`](SHIFT_DIFFICULTY_INFERENCE.md) — the inference protocol for the shift and irregularity analyses
- [`MULTIPLE_COMPARISON_GUIDANCE.md`](MULTIPLE_COMPARISON_GUIDANCE.md) — when and how to correct, and how to define a family
- [`CORRELATION_MULTIPLE_COMPARISON.md`](CORRELATION_MULTIPLE_COMPARISON.md) — Pearson and Spearman on one relationship are one family
- [`ERROR_BY_RANGE_RESULTS.md`](ERROR_BY_RANGE_RESULTS.md) — results of the glycemic-band error analysis
- [`TRAINING_SCHEDULE_SENSITIVITY.md`](TRAINING_SCHEDULE_SENSITIVITY.md) — protocol for the early-stopping sensitivity arm
- [`ARTICLE_REPRODUCIBILITY_AUDIT.md`](ARTICLE_REPRODUCIBILITY_AUDIT.md) — what can and cannot be regenerated, figure by figure
- [`Review-Points.md`](Review-Points.md) — reviewer points and the actions taken

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

Two components are ports of third-party implementations and keep their own licences
beside the code: the Clarke error grid
([`benchmark/evaluation/clarke_ega/LICENSE.Guevara`](benchmark/evaluation/clarke_ega/LICENSE.Guevara), BSD)
and the Parkes error grid
([`benchmark/evaluation/parkes_ega/LICENSE.40H4`](benchmark/evaluation/parkes_ega/LICENSE.40H4), MIT).
The Transformer follows Cui et al.'s [GluPred](https://github.com/r-cui/GluPred) (MIT).
The OhioT1DM dataset is not redistributed here and is governed by its own data-use
agreement.
