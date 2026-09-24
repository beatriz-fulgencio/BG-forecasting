# Cui-inspired TL with RL schedule sensitivity

Run `notebooks/run_schedule_sensitivity_on_colab.ipynb` for GRU at 30 minutes,
12 OhioT1DM patients and seeds 41–43. This replaces the notebook's former
50-epoch sensitivity protocol; it does not overwrite that experiment's results.

| Arm | RL cap | RL patience | Shared TL |
|---|---|---|---|
| Baseline | 200 | 5 | fixed 10 + 10 |
| Longer patience | 200 | 15 | same fitted reference |
| Fixed budget | 20 | 20 (disabled) | same fitted reference |

All RL arms use a single training stage starting at LR 0.0003, with the same
plateau scheduler, batch size 16, initialization, data and validation split.
TL starts at 0.0003 on non-target patients and resets Adam for 10 target epochs
at 0.00005. Both stages always run 10 epochs. The new optional
`training.transfer_early_stopping_patience` overrides the shared patience for
TL only; the notebook sets it to 20. Other configs retain their previous
behavior when this field is absent. `regular_schedule` is explicitly
`single_stage` here, even if the base grid uses two-stage RL.

This addresses R3 #3 within the new protocol. The architecture remains GRU,
128 units, two layers, dropout 0.2, and a 60-minute input window. This is not an
exact reproduction of Cui's self-attention model: validation-based scheduling
and best-checkpoint restoration, preprocessing and batching remain our own.
It also differs from the article's 64-unit/batch-64/0.001 run, so do not attribute
old-to-new changes solely to patience. Twenty RL target epochs are not equal
compute to TL's pooled-source plus target stages.

## Before running

Commit/push:
- `benchmark/configs/config_manager.py`
- `benchmark/experiments/configured.py`
- `notebooks/run_schedule_sensitivity_on_colab.ipynb`
- `RUN/experiments/check_schedule_sensitivity.py`
- `RUN/experiments/check_cui_schedule.py`
- this document

The notebook derives its arm configs from `configs/full_gru_30min.yaml`, explicitly
pinning all schedule and architecture choices above. It validates the new schema
before training. Select GPU in Colab and run the notebook in order.

## Reliability and interpretation

Outputs: `MyDrive/bg-results/cui_patience_sensitivity/<fingerprint>/`, including
protocol provenance, arm runs, training logs, summary and stopping diagnostics.
All 12 patients and all three seeds must complete. Cached results require matching
resolved configs and code/data/environment fingerprints. Completed arms restore
from Drive; an interrupted arm reruns in full.

The notebook resets RNGs for each patient (`seed * 100000 + patient_id`) and
checks the common validation-loss trajectory across RL arms. TL histories must
show exactly 10+10 epochs. Results use joint patient/seed bootstrap intervals,
paired changes versus baseline, and BH correction across three benefit tests.
The fixed-20 arm retains the best validation checkpoint, not the final weights.

An interval containing zero does not demonstrate equivalence. Report the size
and uncertainty of all changes. Inference is limited to this model/horizon and
cohort. The experiment has not been run at publication scale locally.

Offline checks:
`python RUN/experiments/check_schedule_sensitivity.py`
`python RUN/experiments/check_cui_schedule.py`
