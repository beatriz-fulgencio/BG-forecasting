# Open-source and research-publication readiness report

Audit date: 2026-09-14  
Branch and commit: `bench2` at `946dd8f`  
Scope: contributor onboarding, packaging, extension points, experiment reproducibility,
scientific correctness, tests, documentation, licensing, and repository governance.

## Executive verdict

The repository is a useful research prototype, but it is not ready for public release
as a reusable benchmark package. Its strongest pieces are the domain-specific data
pipeline, recurrent model implementations, clinical error grids, experiment tracking,
and the recently added analysis tests. Its main weakness is that these pieces are not
connected by one working, validated public workflow.

The highest-risk problem is not cosmetic developer friction. A documented command can
exit successfully without running anything, while the implemented runner evaluates
standardized predictions as if they were mg/dL. A contributor can therefore believe an
experiment succeeded when it did not, or publish metrics with the wrong units.

Recommendation: do not make a numbered archival release until the P0 items below are
closed. The code can become easy to extend without a rewrite, but the work should start
with one correct end-to-end path and then extract stable interfaces from that path.

## What is already good

1. The top-level separation into data, models, evaluation, experiments, and comparison
   modules is directionally correct.
2. `BaseBGModel` and `BasePyTorchBGModel` provide the beginning of a common model
   contract.
3. Test and train datasets share training-set normalization statistics. The missing-value
   normalization defect found during this audit has a regression test in
   `tests/test_torch_dataset.py`.
4. Clarke and Parkes error-grid implementations are separated and retain upstream
   license files in the source tree.
5. `ExperimentTracker` records the Git commit, branch, Python version, device, timing,
   configuration, and results.
6. The Phase-1 distribution-shift and signal-feature analysis has focused unit tests,
   explicit small-sample caveats, and clear CLI failure codes.
7. Raw data and generated model artifacts are excluded from Git by default.

## Measured developer-experience scorecard

| Dimension | Score | Evidence | Method |
|---|---:|---|---|
| Getting started | 1/10 | Both documented run commands complete without an experiment | Tested |
| API and CLI | 2/10 | Useful internal classes, but public CLI and documented package API are placeholders | Tested + code review |
| Error messages | 2/10 | `argparse` handles syntax errors, but missing data/configuration often becomes a warning, empty result, or false success | Tested |
| Documentation | 3/10 | Extensive prose exists, but many files, APIs, models, datasets, and features described there do not exist | Tested + code review |
| Upgrade path | 1/10 | Version `0.1.0` exists; no changelog, tags, releases, migration policy, or deprecation policy | Repository review |
| Developer environment | 2/10 | Focused tests pass; no CI, modern build config, lint/type config, coverage setup, or clean default test command | Tested |
| Community | 1/10 | MIT license and GitHub Issues exist; no contributing guide, conduct policy, security policy, templates, discussions, or releases | GitHub + repository review |
| DX measurement | 0/10 | No contributor feedback mechanism or onboarding measurement | Repository review |

Overall developer experience: **1.5/10**.  
Measured time to hello world: **unreachable**. The CLI returns in 0.09 seconds with
`CLI implementation pending...`; the alternate documented runner imports for about
22.5 seconds and exits silently.

## Audit evidence

| Check | Result |
|---|---|
| `python -m benchmark.cli run --config benchmark/configs/example_experiment.yaml` | Exit 0, no experiment; prints `CLI implementation pending...` |
| Same command with a nonexistent config path | Exit 0, same placeholder output |
| `python -m benchmark.experiments.runner --config ...` | Exit 0 after heavy imports, no output or experiment |
| Documented `from benchmark import Experiment` | `ImportError` |
| Documented `load_dataset(...)` | `ImportError` |
| Documented reporting imports | `load_results` and the documented `compare_experiments` location do not exist |
| `pytest -q tests` | 38 passed |
| Bare `pytest -q` | Collection error in `benchmark/data/test_data.py` due to top-level imports |
| Two-patient real-data GRU verification | Passed; finite training and evaluation for patients 540 and 544 |
| Wheel build | Succeeds, but omits `benchmark/experiments`, `benchmark/comparison`, `config_manager.py`, and nested third-party licenses |
| Repository scale | 65 tracked files; about 12,577 Python lines in `benchmark/`; 453 test lines; 44 TODO/FIXME markers |
| GitHub metadata | Repository currently private; blank description/homepage; Issues enabled; Discussions disabled; one open setup issue; no releases |

## P0: publication blockers

### 1. Make one documented command actually run one experiment

Evidence:

- `benchmark/cli.py:70-73` only echoes parsed arguments.
- `benchmark/experiments/runner.py` has no module entry point.
- `benchmark/configs/config_manager.py` contains only TODO comments.
- `run_experiment_from_config()` discards the supplied YAML path and uses `{}`.
- The example configuration schema does not match the keys consumed by
  `ExperimentRunner` (`paths`, `training`, and `data.patient_ids`).

Change:

1. Define one versioned configuration schema using typed dataclasses, Pydantic, or a
   JSON Schema-backed loader.
2. Implement `load_config()`, validation, defaults, and useful messages for unknown or
   incompatible fields.
3. Add a console entry point such as `bg-forecast run --config ...` and keep
   `python -m benchmark.cli` as an equivalent path.
4. Make the CLI select exactly the configured dataset, model, metrics, patients, and
   output options.
5. Exit nonzero when configuration, data loading, training, evaluation, or artifact
   generation fails. A run with zero successful patient/model cells must fail.
6. Ship a tiny synthetic fixture so the quick start works without access-controlled
   clinical data.

Acceptance criteria:

- A fresh environment can run one CPU experiment in under five minutes.
- The command prints the resolved config, seed, device, progress, output directory, and
  a final success/failure summary.
- A missing config, missing data, unsupported model, or unknown metric exits nonzero and
  explains the fix.
- An integration test invokes the public command and checks the resulting manifest,
  predictions, and metrics.

### 2. Fix the scientific evaluation path before regenerating article results

Evidence:

- `OhioDataset` standardizes inputs and glucose targets.
- `ExperimentRunner.evaluate_model()` passes those standardized predictions and targets
  directly to MAE, MARD, time-in-range, Clarke, and Parkes calculations, then labels the
  results as mg/dL (`runner.py:557-615`, `822-832`).
- Clinical thresholds such as 70 and 180 are therefore applied in standardized units.
- Model `learning_rate` is stored in the hyperparameter dictionary but `train_model()`
  does not pass it to `fit()`; changing it in configuration has no effect.
- Early stopping tracks the best validation loss but does not save or restore the best
  weights (`base_model.py:431-458`). Evaluation uses the final weights.
- `random_split()` is called without a seeded generator, and no experiment seed is
  applied even though documentation claims fixed seeds.
- Broad warning suppression and exception handling can hide numerical and cell-level
  failures.

Change:

1. Give datasets/scalers an explicit `inverse_transform_target()` API. Convert targets
   and predictions to mg/dL exactly once before all clinical evaluation and export.
2. Store units and horizon semantics in prediction artifacts; validate plausible glucose
   ranges before clinical metrics run.
3. Pass every resolved optimizer/training option into `fit()` and record the resolved
   value, not merely the requested value.
4. Snapshot and restore the best model state during early stopping.
5. Seed Python, NumPy, PyTorch, data-loader generators, and splits from one config field;
   record deterministic-mode and device caveats.
6. Replace global warning suppression with targeted warnings and structured failure
   reporting.
7. Add reference-value tests for MAE/RMSE/MARD/TIR and Clarke/Parkes zones, including
   shape mismatch, NaN, zero, out-of-range, and units cases.

Acceptance criteria:

- A test proves that a standardized tensor round-trips to known mg/dL values.
- A runner integration test produces the same metrics as direct evaluation in mg/dL.
- Changing learning rate and seed changes the resolved run metadata and behavior.
- Early-stopped evaluation uses the minimum-validation-loss checkpoint.
- All article tables and figures are regenerated after this change, not reused.

### 3. Publish a complete, modern package

Evidence:

- `setup.py` uses `find_packages()`, but `benchmark/experiments`,
  `benchmark/comparison`, and `benchmark/configs` lack `__init__.py` files.
- The built wheel omits the experiment runner, comparison/analysis code, and
  `config_manager.py`.
- Nested Clarke and Parkes license files are not present in the wheel.
- There is no `pyproject.toml`; the build emits a legacy `setup.py bdist_wheel`
  deprecation warning.
- All 24 requirements, including TensorFlow, PyTorch, MLflow, W&B, test, formatting, and
  typing tools, are installed as mandatory runtime dependencies.
- There is no console-script entry point.

Change:

1. Move metadata and build configuration to `pyproject.toml` using setuptools package
   discovery that includes every intended package.
2. Add missing `__init__.py` files or deliberately adopt namespace-package discovery.
3. Inspect the built wheel in CI and test it from a clean virtual environment outside the
   repository.
4. Split dependencies into a small core plus extras, for example `[torch]`, `[analysis]`,
   `[tracking]`, `[plots]`, and `[dev]`. Remove unused dependencies.
5. Include configuration templates, notices, and all third-party license files in source
   and wheel distributions.
6. Consider a less collision-prone import package name than the generic `benchmark`
   before the first public release.

Acceptance criteria:

- `pip install .` followed by the quick-start command works outside the checkout.
- Wheel-content tests prove that runner, comparison, configs, and license files ship.
- Importing core metrics does not require TensorFlow, MLflow, W&B, plotting libraries, or
  a font-cache build.

### 4. Provide an executable reproduction of the article

Evidence:

- Generated experiment results are ignored and no small canonical result set or manifest
  is tracked.
- `new_analysis_30min_tl_vs_rl.md` refers to local `test_results/...` experiments and a
  local `.venv` but does not provide the command/config that created those experiments.
- The only runnable article-analysis script consumes already-generated metrics; the model
  training stage is not reproducible from a public command.
- Hardware/library versions, seeds, dataset checksums, and exact cohort exclusions are
  not captured together in one immutable manifest.

Change:

1. Add `reproduce/` with numbered commands or one orchestrator that goes from validated
   raw inputs to every reported table and figure.
2. Commit immutable article configs for every architecture, regime, horizon, and seed.
3. Generate a manifest containing code version, config hash, dataset version/checksums,
   patient IDs, feature definitions, split policy, normalization source, seeds, package
   versions, hardware, and output checksums.
4. Publish a small expected-results fixture for CI and archive full results separately.
5. Make every article table/figure name traceable to a script and manifest entry.
6. Add a code/data availability statement matching the repository and the OhioT1DM
   access conditions.

Acceptance criteria:

- A reviewer can start from the documented data layout and regenerate one representative
  table/figure with one command.
- Every reported number maps to an experiment ID, seed set, and source artifact.
- Repeated CPU smoke runs are deterministic within documented tolerance.

### 5. Make documentation describe only implemented behavior

Evidence:

- The main and benchmark READMEs claim configuration-driven experiments, batch runs,
  cross-validation, custom datasets, multiple traditional/modern/physiological models,
  and result-loading APIs that are not implemented.
- `data/README.md` describes CSV input and `load_dataset()`, while the implemented loader
  is OhioT1DM XML-specific.
- `benchmark/data/README.md` documents unsupported preprocessing arguments and features.
- `STRUCTURE_OVERVIEW.md` lists nonexistent files such as `validators.py`,
  `deep_learning.py`, and `protocols.py`, and labels planned components as implemented.
- The Parkes README links to `LICENSE.4OH4`, but the file is named `LICENSE.40H4`.

Change:

1. Replace the top-level README with a tested golden path, prerequisites, expected output,
   data acquisition/layout, current support matrix, and troubleshooting.
2. Clearly separate “implemented now” from “roadmap.” Remove unsupported claims rather
   than documenting aspirational APIs as present.
3. Execute every code block in documentation in CI using the synthetic fixture.
4. Add API docs generated from the supported public interfaces and one tutorial for each
   extension type.
5. Remove or regenerate duplicate PDF documentation so Markdown and PDF cannot drift.

Acceptance criteria:

- Every import and shell command in public docs is executable.
- Supported models/datasets/metrics are generated from the same registries used by the
  CLI.
- No documentation refers to nonexistent symbols or files.

### 6. Close licensing, citation, and data-distribution gaps

Evidence:

- The root MIT license is present and upstream Clarke/Parkes licenses exist in source.
- Those upstream licenses are omitted from the built wheel.
- Adaptation from GluPred is described, but there is no consolidated third-party notice
  identifying copied/adapted files and licenses.
- There is no `CITATION.cff`, article citation, DOI, `NOTICE`, or machine-readable project
  metadata beyond basic `setup.py` fields.
- Data acquisition, original dataset citation, access terms, and redistribution limits
  are not documented precisely. Raw data is correctly ignored, but contributors have no
  downloader/preflight or checksum guidance.

Change:

1. Audit provenance file by file and add `THIRD_PARTY_NOTICES.md` with upstream project,
   source URL, copyright, license, retrieved version/date, and modified files.
2. Ensure all license/notice files ship in source and wheel distributions.
3. Add `CITATION.cff` and a complete BibTeX entry for the accompanying article; archive
   the release with a DOI and pin that DOI in the article.
4. Add a data card with OhioT1DM citation, official acquisition steps, permitted use,
   expected layout, checksums where lawful, preprocessing assumptions, and a clear
   statement that clinical data is not bundled.
5. Add a research-only/non-clinical-use disclaimer without weakening the actual license.

Acceptance criteria:

- Automated package inspection finds all required third-party notices.
- GitHub renders the intended software citation.
- A contributor can verify a lawful dataset copy without committing it.

### 7. Establish a release-blocking test and CI baseline

Evidence:

- `pytest -q tests` passes 38 tests, but bare `pytest` fails while collecting
  `benchmark/data/test_data.py` because it uses top-level imports.
- Most tests cover the new comparison analysis. There are no committed runner, CLI,
  configuration, tracker, RNN training, reporting, or clinical-metric integration tests.
- There is no GitHub Actions workflow, coverage configuration, lint configuration, type
  configuration, pre-commit setup, or supported Python matrix.
- The package advertises Python 3.8-3.10 classifiers while `python_requires` is open-ended.

Change:

1. Move example scripts out of test discovery or convert them into proper tests with
   package-relative imports.
2. Add fast unit tests for each public contract and a synthetic end-to-end test.
3. Add optional, explicitly selected real-data tests that skip with an actionable message
   when OhioT1DM is unavailable.
4. Add CI for the supported Python/OS matrix: formatting/lint, types, unit tests,
   documentation examples, wheel/sdist build, clean-wheel install, license inventory,
   and reproducibility smoke test.
5. Measure coverage by risk, not one global percentage. Require high coverage for config,
   scaling, splitting, clinical metrics, artifact schemas, and failure paths.

Acceptance criteria:

- `pytest` from the repository root passes.
- The quick-start integration test runs on every pull request.
- CI rejects incomplete wheels, stale docs examples, missing licenses, and invalid configs.

## P1: make extension safe and predictable

### 8. Define public contracts and registries

Today, extension requires editing central conditional logic:

- Models are hard-coded in `ExperimentRunner.run_experiment()`.
- Metrics are selected by an `if/elif` chain in `BGEvaluator`.
- The loader and dataset classes are OhioT1DM-specific.
- Configuration options and documentation are not derived from implementations.

Create small, typed contracts:

```text
CLI -> validated ExperimentConfig -> registries -> staged ExperimentRunner
                                      | model factory
                                      | dataset adapter
                                      | metric callable
                                      | reporter/exporter
                                      v
                              versioned artifact bundle
```

Recommended interfaces:

- `DatasetAdapter`: `discover()`, `load_split()`, `validate()`, and metadata/provenance.
- `WindowTransform` or `ForecastDataset`: explicit features, targets, timestamps, units,
  scaler state, and inverse transforms.
- `ForecastModel`: `fit()`, `predict()`, `save()`, `load()`, and capabilities metadata.
- `Metric`: stable name, required units/shapes, output schema, and callable implementation.
- `Reporter`: consumes a versioned run result rather than runner internals.

Use explicit registration decorators or dictionaries first. Add Python entry-point plugins
only after the internal contracts are stable. A third-party model should be addable in a
separate module without modifying the runner.

### 9. Split the runner into composable stages

`runner.py` is about 1,400 lines and mixes loading, preprocessing, splitting, transfer
learning, training, evaluation, plotting, export, and console output. Split it into stages
with explicit inputs/outputs:

1. Resolve and validate configuration.
2. Resolve dataset and create a split manifest.
3. Fit preprocessing state on training data only.
4. Build model from registry.
5. Train with callbacks/checkpointing.
6. Predict into a typed prediction table.
7. Evaluate using declared units and horizons.
8. Export a versioned artifact bundle.

Each stage should be callable independently and testable with in-memory fixtures. Keep a
thin orchestrator for the common path. This gives contributors escape hatches without
copying the entire runner.

### 10. Validate boundaries instead of silently coercing or skipping

Add explicit validation for:

- Configuration fields, incompatible options, and unknown names.
- Dataset columns, dtypes, timestamps, duplicates, sampling interval, physiological
  ranges, missingness, and minimum contiguous-window counts.
- External normalization vectors: both mean and std must be supplied, lengths must match,
  and values must be finite.
- Prediction/target shapes, horizon ordering, units, finite values, and patient alignment.
- Empty datasets, empty data loaders, and zero-success experiment matrices.

Use typed exceptions with context and remediation. Library layers should not print and
continue by default; the CLI should decide whether a declared `--keep-going` mode is
appropriate and summarize every failure.

### 11. Separate minimal core dependencies from integrations

The metric/data core should import quickly and work with NumPy/Pandas. PyTorch models,
TensorFlow models, experiment trackers, plotting, statistical analysis, and spreadsheet
export should be optional extras loaded only when selected. Avoid package-level
`try/except ImportError` blocks that turn broken internal imports into vague dependency
warnings.

This lowers install time and conflict risk while making extension requirements obvious.

### 12. Stabilize and version artifacts

Define a run bundle with a schema version and at least:

- resolved config and config hash;
- run/experiment ID and parent comparison ID;
- code/data/environment provenance;
- split manifest and preprocessing/scaler state;
- model checkpoint and training history;
- long-form predictions keyed by patient, timestamp, horizon, and units;
- metrics with metric version/parameters;
- warnings, partial failures, and completion status;
- checksums for every artifact.

Avoid timestamp-only discovery and “latest matching JSON” as the primary interface. Add a
supported loader that validates schema versions and migrations.

## P2: contributor and maintenance quality

### 13. Add community health files before making the repository public

Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue forms, a pull-request
template, support boundaries, and labels for model/dataset/metric contributions. Include
the local setup, test commands, style rules, public API policy, required scientific tests,
and a review checklist for data leakage and units.

Populate the GitHub description, topics, homepage/docs link, and Discussions policy when
the repository changes from private to public.

### 14. Define versioning and release policy

Use semantic versioning from the first public tag. Add `CHANGELOG.md`, one source of truth
for the version, a compatibility table, deprecation warnings, migration notes, and a
release checklist. Build and test artifacts in CI, publish a GitHub release, then archive
the exact article release with a DOI.

### 15. Replace print debugging with structured logging and progress

Library code should use `logging` and keep default output quiet. The CLI can provide human
progress plus `--quiet`, `--verbose`, and machine-readable modes. Remove full DataFrame
column dumps and global warning suppression. Every failure summary should name the
patient/model/stage and point to the detailed log.

### 16. Profile preprocessing and import cost

The alternate documented runner spent roughly 22.5 seconds importing and building a
Matplotlib font cache before doing no work. Preprocessing also performs repeated row-wise
DataFrame loops. After correctness and packaging are fixed, add timing benchmarks for
import, preprocessing, sequence construction, and one training epoch. Lazy-load plotting
and heavyweight integrations, and vectorize the highest-cost preprocessing paths.

## Recommended implementation order

### Milestone A: trustworthy experiment core

1. Fix inverse target transforms and clinical-unit evaluation.
2. Restore best early-stopping weights; wire seeds and all training options.
3. Add metric, scaling, split, and end-to-end synthetic tests.
4. Regenerate representative article results and compare them with prior outputs.

### Milestone B: one public golden path

1. Implement and validate the configuration schema.
2. Implement the CLI and one CPU synthetic example.
3. Make errors nonzero and actionable.
4. Rewrite the README around the tested command.

### Milestone C: extension architecture and packaging

1. Introduce registries/contracts and split the runner into stages.
2. Move to `pyproject.toml`, optional extras, and complete package discovery.
3. Add clean-wheel, documentation, and license tests.
4. Document one model, dataset, and metric extension tutorial.

### Milestone D: article release

1. Add the reproduction workflow, frozen configs, manifests, and result provenance.
2. Finish third-party notices, data card, citation, and availability statement.
3. Add CI and community health files.
4. Make the repository public, create the versioned release, archive it, and insert the
   DOI into the article and repository metadata.

## Release checklist

- [ ] One fresh-install quick start completes and produces checked artifacts.
- [ ] Invalid inputs fail nonzero with remediation.
- [ ] Metrics are evaluated and exported in declared clinical units.
- [ ] Best checkpoint, seed, split, scaler, and resolved hyperparameters are recorded.
- [ ] All article numbers trace to immutable configs and manifests.
- [ ] `pytest` passes from the repository root.
- [ ] CI tests source and wheel distributions across supported environments.
- [ ] Wheel contains all intended code, configs, and license/notice files.
- [ ] Documentation examples execute in CI and describe only shipped behavior.
- [ ] Data acquisition, licensing, citation, and non-redistribution guidance are clear.
- [ ] `CITATION.cff`, article citation, DOI, changelog, and version tag agree.
- [ ] Contribution, conduct, security, issue, and PR guidance is present.

## Final assessment

The project should be published as a **research software package in active development**,
not yet as a general benchmark framework supporting many datasets and model families.
Narrowing the public promise to the code that works today will improve credibility more
than adding another architecture first. Once the correct OhioT1DM-to-GRU/LSTM/RNN path is
executable, tested, and packaged, the proposed registries and staged runner will make new
models, datasets, and metrics straightforward to add without weakening the research
protocol.
