# Full-grid Cui-inspired TL and patience analysis

Run `notebooks/run_cui_full_grid_on_colab.ipynb`. It covers all three recurrent
models, four horizons, three seeds and twelve patients. TL trains for 10 epochs
at initial LR 0.0003 plus 10 at 0.00005, with fresh Adam in stage two. The three
single-stage RL arms start at 0.0003: cap 200/patience 5, cap 200/patience 15,
and fixed 20 epochs. TL is fitted once per model/horizon/seed and shared.

The model setup is 128 hidden units, two layers, dropout 0.2, batch 16, 12 input
steps and four features. This adapts Cui's schedule, not their architecture,
120-minute input or test-based checkpoint selection. Best-validation checkpoints
and the current benchmark preprocessing are retained. The 20-epoch RL arm uses
best-validation weights, not necessarily the final weights; epoch counts do not
make compute equal across RL and pooled-source TL.

## Commit before running

- `notebooks/run_cui_full_grid_on_colab.ipynb`
- `RUN/experiments/run_cui_full_grid.py`
- `RUN/experiments/check_cui_full_grid.py`
- `benchmark/configs/config_manager.py`
- `benchmark/experiments/configured.py`
- this document

The twelve existing `configs/full_*.yaml` and the normal analysis scripts must
already exist on the branch. The helper pins all settings above in memory,
so no additional 36 manual arm configs are needed. Saved resolved configs and
`protocol.json` record what was used.

## Workflow and outputs

A separate checkout and fingerprinted `MyDrive/bg-results/cui_full_grid/` directory
prevent mixing old outputs. Training is seed-major and resumable at the completed
mode/seed level (12 patients per job). Default: 144 jobs, including 36 shared TL
jobs and 108 RL jobs. An interrupted job reruns in full. SHA checks cover saved
predictions, histories and metrics; completed TL copies remain identical.

The helper builds standard paired analysis parents for each arm and combines all
three seeds using the repository aggregator. The notebook runs the dataset driver
and every stage of `RUN/run_experiments_analysis.sh` for each arm. Stages save to
Drive and verify previous output hashes before skipping. Three self-contained
analysis copies require more Drive space. Training logs remain under each job;
analysis logs and run lists are under `logs/<arm>/`.

`sensitivity/full_grid_sensitivity.csv` adds paired change-versus-patience-5
intervals and BH over 72 benefit tests (3 models x 4 horizons x 3 arms x 2 metrics).
The same patient/seed bootstrap draws are used across conditions. Per-arm analysis
q-values use their original, narrower families. Shared TL outputs are not independent
replications. Change intervals are pointwise, exploratory, not equivalence tests.

`SMOKE_TEST=True` exercises one cell/seed, two patients and a tiny model. It skips
the population inference stages and labels all outputs SMOKE. It is not evidence.
Local validation uses synthetic glucose records; publication-scale GPU training
and complete raw-data analyses must be run in Colab.
