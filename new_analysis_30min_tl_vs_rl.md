# 30-min shift→MAE analysis: transfer-learning vs from-scratch (GRU)

Fresh 2026 runs, single horizon (30 min), 12 patients, GRU. Same
`shift_score` (Wasserstein train→test glucose) as the multi-horizon Phase-1
analysis. Two experiments:

- **tl** — `test_results/experiment_20260710_151113_30min_tl` → `new_analysis_30min_tl/`
- **rl** — `test_results/experiment_20260710_151113_30min_rl` → `new_analysis_30min_rl/`

Reproduce (the runner now caps BLAS/OpenMP threads internally, so no env vars are
needed; the results dir holds symlinks to just these two experiments so each
horizon maps to exactly one run):
```bash
PYTHONPATH=. .venv/bin/python RUN/run_phase1_shift_analysis.py \
    --results-dir <dir-with-the-two-experiments> --output-dir new_analysis_30min_tl --model GRU --mode tl
# ...and again with --mode rl --output-dir new_analysis_30min_rl
```

## Headline: distributional shift tracks error more strongly for from-scratch models

| Metric (shift_score vs MAE, 30 min, n=12) | tl (transfer-learned) | rl (from scratch) |
|---|---|---|
| Pearson r (p) | 0.34 (0.27) | **0.62 (0.032)** |
| Spearman ρ (p) | 0.28 (0.38) | 0.52 (0.080) |
| OLS `MAE ~ shift+std+TIR`, R² | 0.53 | 0.86 |
| shift_score coef (p) in that OLS | +0.164 (0.052) | +0.105 (0.107) |
| dominant predictor in that OLS | test_TIR (p=0.043) | **test_std (p=0.001)** |
| + signal features: shift_score coef (p) | +0.027 (0.758) | **+0.191 (0.046)** |

**Interpretation.** In the from-scratch (rl) model the bivariate correlation between
train→test shift and MAE is strong and significant (r = 0.62, p = 0.03); in the
transfer-learned (tl) model it is weaker and not significant (r = 0.34, p = 0.27).
This is consistent with the paper's transfer-learning story: **population
pretraining buffers a patient against their own distributional drift, so shift
predicts error more for models trained from scratch.** For rl, once glucose spread
(`test_std`) is in the model it dominates (p = 0.001) — shift and spread are
correlated — but shift_score becomes significant again (p = 0.046) in the fuller
model that also includes the signal features.

## Caveats
- **Single horizon (30 min):** no shift×horizon interaction test here (that lives in
  the 4-horizon `new_analysis/` report). Pooled rows equal the 30-min row by
  construction.
- **n = 12**, one horizon: treat p-values as exploratory. The signal-augmented model
  (7 predictors) is especially collinearity-prone at this sample size.
- These are GRU only. The 2026 runs also include RNN / LSTM / Transformer — rerun
  with `--model` to check robustness across architectures.
