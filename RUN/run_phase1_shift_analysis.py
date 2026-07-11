#!/usr/bin/env python3
"""
Phase 1 analysis: per-patient distributional shift vs prediction error.

Builds the Phase-1 deliverables in one pass (glucose loaded once):

  1. Per-patient distributional shift metric (Wasserstein train->test glucose)
     via benchmark.comparison.distribution_shift.
  2. A 12-row patient feature table: shift_score, KL, train/test std, train/test
     TIR, insulin_type (real metadata from the XML), temporal-irregularity
     features (sample_entropy, autocorr_lag1), and GRU MAE per horizon.
  3. Pearson + Spearman correlation between shift_score and MAE, per horizon,
     pooled, and pooled with MAE standardized within horizon (the honest pooled
     number). Plus OLS regressions (MAE ~ shift_score + std + TIR) per horizon, a
     formal shift_score x horizon interaction test, and a pooled model augmented
     with the signal features — R² reported honestly for the small sample.
  4. All four horizons (15/30/45/60 min), GRU / transfer-learning.
  5. Figure: shift_score vs MAE scatter, faceted by horizon (+ colored single
     panel for reference).

Notes on patient metadata: the OhioT1DM XML exposes `weight` (a constant
placeholder = 99, so uninformative and excluded) and `insulin_type`. Patient
age and pump model are NOT in the data files (only in the dataset paper), so
they are not included here.

Usage (from repo root):
    PYTHONPATH=. python RUN/run_phase1_shift_analysis.py \
        --results-dir RESULT-test_results --output-dir new_analysis --model GRU --mode tl
"""

import os
import sys
import json
import argparse
import glob as _glob
import xml.etree.ElementTree as ET
from pathlib import Path

# Cap BLAS/OpenMP thread pools BEFORE numpy/scipy/torch are imported. Loading
# scipy + statsmodels + torch (via the data loaders) together can deadlock at
# import time on multi-threaded BLAS builds; this analysis is tiny so a single
# thread is plenty and avoids the hang. Must run before the first numeric import
# to take effect, and we do not clobber values the caller has already set.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

# NOTE: statsmodels and the torch-backed data loaders are imported lazily inside
# the functions that use them (run_regressions / load_glucose_features). They are
# slow, heavy imports; keeping them out of module scope lets the lightweight
# helpers (parse/select/table logic) be imported and unit-tested cheaply.
from benchmark.comparison.distribution_shift import compute_distribution_shift, _clean_glucose
from benchmark.comparison.signal_features import compute_signal_features
from benchmark.comparison.shift_mae_plot import parse_horizon, HORIZON_COLORS

VERSION = ['2018', '2020']


def glucose_stats(arr):
    """Std and % time-in-range (70-180) on cleaned glucose."""
    g = _clean_glucose(arr)
    if g.size == 0:
        return {"std": np.nan, "tir": np.nan, "n": 0}
    return {
        "std": float(np.std(g)),
        "tir": float(np.mean((g >= 70) & (g <= 180)) * 100.0),
        "n": int(g.size),
    }


def find_insulin_type(data_root, pid):
    """Read insulin_type attribute from a patient's training XML (metadata).

    Returns ``None`` if no XML is found for the patient. A file that exists but
    fails to parse is a real problem (corrupt/renamed data), so it is warned
    about rather than silently skipped.
    """
    for path in _glob.glob(f"{data_root}/raw/ohiot1dm/*/train/{pid}-ws-training.xml"):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as e:
            print(f"    WARNING: could not parse insulin_type XML for patient {pid} "
                  f"({path}): {e}")
            continue
        return root.attrib.get("insulin_type")
    return None


def load_glucose_features(data_root, patient_ids):
    """Load preprocessed train/test glucose once; return per-patient feature dict."""
    # Lazy import: pulls in torch; kept out of module scope (see note at top).
    from benchmark.data.loaders import load_ohiot1dm_data
    from benchmark.data.preprocessors import OhioBGDataPreprocessor

    print(f"Loading + preprocessing glucose for {len(patient_ids)} patients...")
    train_raw = load_ohiot1dm_data(str(data_root), patient_ids=patient_ids, mode='train', version=VERSION)
    test_raw = load_ohiot1dm_data(str(data_root), patient_ids=patient_ids, mode='test', version=VERSION)
    pre = OhioBGDataPreprocessor()

    feats = {}
    for pid in patient_ids:
        if pid not in train_raw or pid not in test_raw:
            print(f"  skip {pid}: missing raw data")
            continue
        tr = pre.basic_preprocessing(train_raw[pid])['glucose'].values
        te = pre.basic_preprocessing(test_raw[pid])['glucose'].values
        shift = compute_distribution_shift(tr, te)
        ts, es = glucose_stats(tr), glucose_stats(te)
        # Temporal-irregularity features from the (ordered) test glucose series,
        # same source as shift_score. Tests whether erraticness adds explanatory
        # power beyond spread (std) and drift (shift_score).
        sig = compute_signal_features(te)
        feats[pid] = {
            "patient_id": pid,
            "insulin_type": find_insulin_type(data_root, pid),
            "train_std": ts["std"], "test_std": es["std"],
            "train_tir": ts["tir"], "test_tir": es["tir"],
            "train_n": shift["train_n"], "test_n": shift["test_n"],
            "shift_score": shift["shift_score"],
            "kl_divergence": shift["kl_divergence"],
            "sample_entropy": sig["sample_entropy"],
            "autocorr_lag1": sig["autocorr_lag1"],
        }
    return feats


class AmbiguousExperimentsError(RuntimeError):
    """Raised when more than one experiment matches the same (mode, horizon)."""


def _find_metrics_json(exp: Path):
    """Latest comprehensive_metrics JSON for an experiment, or None.

    Supports both the flat layout (``exp/comprehensive_metrics_*.json``) and the
    newer nested layout (``exp/<run-hash>/comprehensive_metrics_*.json``).
    """
    jsons = sorted(exp.glob("comprehensive_metrics_*.json"))
    if not jsons:
        jsons = sorted(exp.rglob("comprehensive_metrics_*.json"))
    return jsons[-1] if jsons else None


def _read_patient_mae(metrics_path: Path, model: str) -> dict:
    """{patient_id: mae} for ``model`` from a comprehensive_metrics JSON."""
    with open(metrics_path) as fh:
        data = json.load(fh)
    per_patient = {}
    for pid_str, models in data.get("patient_metrics", {}).items():
        try:
            pid = int(pid_str)
        except (ValueError, TypeError):
            continue
        if model in models and models[model].get("mae") is not None:
            per_patient[pid] = float(models[model]["mae"])
    return per_patient


def load_mae_by_horizon(results_dir, mode, model):
    """``{horizon: {patient_id: mae}}`` for ``model`` from ``*_<mode>`` experiments.

    Raises:
        AmbiguousExperimentsError: if two or more experiment directories resolve
            to the same horizon for this mode. Silently keeping only the last one
            (the previous behaviour) could mix results from different runs without
            any warning, so the caller must disambiguate (point ``--results-dir``
            at a directory holding exactly one experiment per horizon).
    """
    # horizon -> list of (exp_name, per_patient_mae) so duplicates are detected
    # rather than silently overwritten.
    candidates: dict = {}
    for exp in sorted(Path(results_dir).glob(f"*_{mode}")):
        horizon = parse_horizon(exp.name)
        if horizon is None:
            continue
        metrics_path = _find_metrics_json(exp)
        if metrics_path is None:
            continue
        per_patient = _read_patient_mae(metrics_path, model)
        if per_patient:
            candidates.setdefault(horizon, []).append((exp.name, per_patient))

    duplicates = {h: [n for n, _ in v] for h, v in candidates.items() if len(v) > 1}
    if duplicates:
        detail = "; ".join(
            f"horizon {h}min matched {len(names)} experiments: {', '.join(sorted(names))}"
            for h, names in sorted(duplicates.items())
        )
        raise AmbiguousExperimentsError(
            f"Ambiguous experiments for mode '{mode}', model '{model}': {detail}. "
            f"Point --results-dir at a directory containing exactly one experiment "
            f"per horizon (e.g. a folder of symlinks to the specific runs you want)."
        )

    out = {}
    for horizon in sorted(candidates):
        (exp_name, per_patient), = candidates[horizon]  # exactly one after the check
        out[horizon] = per_patient
        print(f"  {exp_name}: horizon={horizon}, {len(per_patient)} patients")
    return out


def build_tables(feats, mae_by_horizon):
    """Return (wide 12-row feature table, long patient x horizon table)."""
    horizons = sorted(mae_by_horizon)
    wide_rows, long_rows = [], []
    for pid, f in sorted(feats.items()):
        row = dict(f)
        for h in horizons:
            row[f"mae_{h}"] = mae_by_horizon[h].get(pid, np.nan)
            long_rows.append({
                "patient_id": pid, "horizon": h,
                "shift_score": f["shift_score"], "kl_divergence": f["kl_divergence"],
                "test_std": f["test_std"], "test_tir": f["test_tir"],
                "train_std": f["train_std"], "train_tir": f["train_tir"],
                "sample_entropy": f.get("sample_entropy", np.nan),
                "autocorr_lag1": f.get("autocorr_lag1", np.nan),
                "mae": mae_by_horizon[h].get(pid, np.nan),
            })
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows).dropna(subset=["shift_score", "mae"])
    return wide, long, horizons


def run_correlations(long, horizons):
    """Pearson + Spearman of shift_score vs MAE, per horizon and pooled.

    Note: the raw ``pooled`` row correlates shift_score against raw MAE across all
    horizons, but MAE scale grows ~4x from 15->60 min, which washes out the
    within-horizon signal. ``pooled_within_horizon_z`` first z-scores MAE within
    each horizon, so the pooled correlation reflects the actual per-horizon
    relationship rather than the horizon-driven scale differences.
    """
    long = long.copy()
    long["mae_z"] = long.groupby("horizon")["mae"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )

    rows = []
    for h in horizons + ["pooled", "pooled_within_horizon_z"]:
        y_col = "mae_z" if h == "pooled_within_horizon_z" else "mae"
        if h in ("pooled", "pooled_within_horizon_z"):
            g = long
        else:
            g = long[long["horizon"] == h]
        g = g.dropna(subset=["shift_score", y_col])
        if len(g) >= 3:
            pr, pp = pearsonr(g["shift_score"], g[y_col])
            sr, sp = spearmanr(g["shift_score"], g[y_col])
        else:
            pr = pp = sr = sp = np.nan
        rows.append({"horizon": h, "n": len(g),
                     "pearson_r": pr, "pearson_p": pp,
                     "spearman_rho": sr, "spearman_p": sp})
    return pd.DataFrame(rows)


def run_regressions(long, horizons):
    """OLS MAE ~ shift_score + test_std + test_tir per horizon, plus pooled."""
    import statsmodels.formula.api as smf  # lazy: heavy import (see note at top)

    lines = []
    lines.append("OLS regression: mae ~ shift_score + test_std + test_tir")
    lines.append("(n=12 per horizon; small-sample — interpret R^2 cautiously)\n")

    def fit(df, label):
        df = df.dropna(subset=["mae", "shift_score", "test_std", "test_tir"])
        if len(df) < 5:
            lines.append(f"[{label}] insufficient data (n={len(df)})\n")
            return
        m = smf.ols("mae ~ shift_score + test_std + test_tir", data=df).fit()
        lines.append(f"[{label}]  n={int(m.nobs)}  R^2={m.rsquared:.3f}  adj_R^2={m.rsquared_adj:.3f}  F_p={m.f_pvalue:.3f}")
        for name in m.params.index:
            lines.append(f"    {name:>12}: coef={m.params[name]:+.4f}  p={m.pvalues[name]:.3f}")
        lines.append("")

    for h in horizons:
        fit(long[long["horizon"] == h], f"{h} min")

    pooled = long.copy()
    pooled = pooled.dropna(subset=["mae", "shift_score", "test_std", "test_tir"])
    if len(pooled) >= 5:
        m = smf.ols("mae ~ shift_score + test_std + test_tir + C(horizon)", data=pooled).fit()
        lines.append(f"[pooled + C(horizon)]  n={int(m.nobs)}  R^2={m.rsquared:.3f}  adj_R^2={m.rsquared_adj:.3f}  F_p={m.f_pvalue:.3f}")
        for name in ["shift_score", "test_std", "test_tir"]:
            if name in m.params.index:
                lines.append(f"    {name:>12}: coef={m.params[name]:+.4f}  p={m.pvalues[name]:.3f}")
        lines.append("")

    # Formal test of "shift matters more at longer horizons": horizon as a
    # CONTINUOUS covariate, so the shift_score:horizon interaction is a single
    # coefficient. A significant positive interaction is the one-number version of
    # the per-horizon coefficient trend (0.087 -> 0.389), instead of eyeballing it.
    lines.append("Interaction test (horizon continuous): mae ~ shift_score * horizon + test_std + test_tir")
    if pooled["horizon"].nunique() < 2:
        lines.append("[pooled interaction] skipped (needs >= 2 horizons; single-horizon run)\n")
    elif len(pooled) >= 6:
        m = smf.ols("mae ~ shift_score * horizon + test_std + test_tir", data=pooled).fit()
        lines.append(f"[pooled interaction]  n={int(m.nobs)}  R^2={m.rsquared:.3f}  adj_R^2={m.rsquared_adj:.3f}  F_p={m.f_pvalue:.3f}")
        for name in ["shift_score", "horizon", "shift_score:horizon", "test_std", "test_tir"]:
            if name in m.params.index:
                lines.append(f"    {name:>20}: coef={m.params[name]:+.5f}  p={m.pvalues[name]:.3f}")
        lines.append("")

    # Do the temporal-irregularity features add explanatory power beyond
    # shift/std/TIR? Pooled (n~48) has enough power for this; per-horizon (n=12,
    # 5 predictors) would be too thin, so it is only fitted pooled.
    lines.append("Augmented with signal features (pooled): "
                 "mae ~ shift_score + test_std + test_tir + sample_entropy + autocorr_lag1 + C(horizon)")
    aug = pooled.dropna(subset=["sample_entropy", "autocorr_lag1"])
    if len(aug) >= 8:
        m = smf.ols(
            "mae ~ shift_score + test_std + test_tir + sample_entropy + autocorr_lag1 + C(horizon)",
            data=aug,
        ).fit()
        lines.append(f"[pooled + signal + C(horizon)]  n={int(m.nobs)}  R^2={m.rsquared:.3f}  adj_R^2={m.rsquared_adj:.3f}  F_p={m.f_pvalue:.3f}")
        for name in ["shift_score", "test_std", "test_tir", "sample_entropy", "autocorr_lag1"]:
            if name in m.params.index:
                lines.append(f"    {name:>14}: coef={m.params[name]:+.4f}  p={m.pvalues[name]:.3f}")
        lines.append("    (check: does shift_score stay significant once irregularity is controlled?)")
        lines.append("")
    else:
        lines.append(f"[pooled + signal] insufficient data (n={len(aug)})\n")
    return "\n".join(lines)


def plot_faceted(long, horizons, out_path, model):
    n = len(horizons)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows), squeeze=False)
    for i, h in enumerate(horizons):
        ax = axes[i // ncols][i % ncols]
        g = long[long["horizon"] == h].dropna(subset=["shift_score", "mae"])
        color = HORIZON_COLORS.get(h, "#333333")
        ax.scatter(g["shift_score"], g["mae"], color=color, s=60, alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=3)
        if len(g) >= 2 and g["shift_score"].nunique() >= 2:
            slope, intercept = np.polyfit(g["shift_score"], g["mae"], 1)
            xs = np.linspace(g["shift_score"].min(), g["shift_score"].max(), 50)
            ax.plot(xs, slope * xs + intercept, color=color, lw=2, alpha=0.8, zorder=2)
        if len(g) >= 3:
            pr, pp = pearsonr(g["shift_score"], g["mae"])
            sr, sp = spearmanr(g["shift_score"], g["mae"])
            ax.set_title(f"{h} min   Pearson r={pr:.2f} (p={pp:.2f}),  Spearman ρ={sr:.2f}")
        else:
            ax.set_title(f"{h} min")
        ax.set_xlabel("Distributional shift (Wasserstein train→test, mg/dL)")
        ax.set_ylabel(f"{model} MAE (mg/dL)")
        ax.grid(True, alpha=0.25, zorder=0)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Per-patient distributional shift vs {model} MAE, by horizon", y=1.0, fontsize=13)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved faceted figure: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Phase 1 shift-vs-MAE analysis.")
    ap.add_argument("--results-dir", default="test_results")
    ap.add_argument("--output-dir", default="new_analysis")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model", default="GRU",
                    choices=["RNN", "LSTM", "GRU", "Transformer"])
    ap.add_argument("--mode", default="tl", choices=["tl", "rl"])
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading MAE by horizon...")
    try:
        mae_by_horizon = load_mae_by_horizon(args.results_dir, args.mode, args.model)
    except AmbiguousExperimentsError as e:
        print(f"✗ {e}")
        return 2
    if not mae_by_horizon:
        print("✗ No MAE data found.")
        return 1
    patient_ids = sorted({pid for h in mae_by_horizon.values() for pid in h})

    feats = load_glucose_features(args.data_root, patient_ids)
    if not feats:
        print("✗ No glucose features computed.")
        return 1

    wide, long, horizons = build_tables(feats, mae_by_horizon)

    # Persist tables
    wide.to_csv(out / "patient_feature_table.csv", index=False)
    long.to_csv(out / "shift_mae_long.csv", index=False)
    corr = run_correlations(long, horizons)
    corr.to_csv(out / "shift_mae_correlations.csv", index=False)
    reg_txt = run_regressions(long, horizons)
    (out / "regression_summary.txt").write_text(reg_txt)

    plot_faceted(long, horizons, str(out / f"shift_vs_mae_{args.model}_{args.mode}_faceted.png"), args.model)

    # Console summary
    print("\n=== patient_feature_table.csv ===")
    print(wide.to_string(index=False))
    print("\n=== correlations (shift_score vs MAE) ===")
    print(corr.to_string(index=False))
    print("\n=== regression summary ===")
    print(reg_txt)
    print(f"\n✓ All Phase-1 outputs written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
