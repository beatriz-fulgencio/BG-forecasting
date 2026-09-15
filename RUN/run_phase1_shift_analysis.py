#!/usr/bin/env python3
"""
Phase 1 analysis: per-patient distributional shift vs prediction error.

Builds the Phase-1 deliverables in one pass (glucose loaded once):

  1. Per-patient distributional shift metric (Wasserstein train->test glucose)
     via benchmark.comparison.distribution_shift.
  2. A 12-row patient feature table: shift_score, KL, train/test std, train/test
     TIR, insulin_type (real metadata from the XML), temporal-irregularity
     features (sample_entropy, autocorr_lag1), and GRU MAE per horizon.
  3. Pearson + Spearman correlation between shift_score and MAE per horizon,
     plus a combined-horizon score with one standardized MAE value per patient.
     Regressions and the horizon-trend test also use patients as independent
     observations. The five-predictor signal model is exploratory at n=12.
  4. Matched-patient comparison of shift, test-series irregularity, and
     train-series entropy as screens for difficulty; with both training modes,
     also screens for smallest regular-minus-transfer MAE benefit.
  5. Figures: shift_score vs MAE scatter, faceted by horizon and as a
     single panel colored by horizon. Both annotate within-horizon
     correlations only.

Notes on patient metadata: the OhioT1DM XML exposes `weight` (a constant
placeholder = 99, so uninformative and excluded) and `insulin_type`. Patient
age and pump model are NOT in the data files (only in the dataset paper), so
they are not included here.

Usage (from repo root):
    PYTHONPATH=. python RUN/run_phase1_shift_analysis.py \
        --results-dir RESULT-test_results --output-dir new_analysis --model GRU --mode tl

For configured multi-seed runs, repeat --configured-aggregate with one parent
aggregate_metrics.json for each version (2018 and 2020) at each horizon. Use
--mode transfer --compare-transfer-benefit when both modes were run in those
parents. The analysis uses the mean MAE across completed seeds and requires
matched regular/transfer seed sets for the benefit comparison.
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
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

# NOTE: statsmodels and the torch-backed data loaders are imported lazily inside
# the functions that use them (run_regressions / load_glucose_features). They are
# slow, heavy imports; keeping them out of module scope lets the lightweight
# helpers (parse/select/table logic) be imported and unit-tested cheaply.
from benchmark.comparison.distribution_shift import compute_distribution_shift, _clean_glucose
from benchmark.comparison.signal_features import compute_signal_features
from benchmark.comparison.shift_mae_plot import (
    parse_horizon,
    plot_shift_vs_mae,
    plot_shift_vs_mae_faceted,
)
from benchmark.configs.config_manager import OHIO_PATIENTS

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


def load_glucose_features(data_root, patient_ids, sampling_rate=5):
    """Load train/test glucose at the experiment's sampling rate once."""
    # Lazy import: pulls in torch; kept out of module scope (see note at top).
    from benchmark.data.loaders import load_ohiot1dm_data
    from benchmark.data.preprocessors import OhioBGDataPreprocessor

    print(f"Loading + preprocessing glucose for {len(patient_ids)} patients at {sampling_rate}-minute intervals...")
    train_raw = load_ohiot1dm_data(
        str(data_root), patient_ids=patient_ids, mode='train', version=VERSION,
        sampling_rate=sampling_rate,
    )
    test_raw = load_ohiot1dm_data(
        str(data_root), patient_ids=patient_ids, mode='test', version=VERSION,
        sampling_rate=sampling_rate,
    )
    pre = OhioBGDataPreprocessor(sampling_rate=sampling_rate)

    feats = {}
    for pid in patient_ids:
        if pid not in train_raw or pid not in test_raw:
            print(f"  skip {pid}: missing raw data")
            continue
        tr = pre.basic_preprocessing(train_raw[pid])['glucose'].values
        te = pre.basic_preprocessing(test_raw[pid])['glucose'].values
        shift = compute_distribution_shift(tr, te)
        ts, es = glucose_stats(tr), glucose_stats(te)
        # Keep test-derived irregularity for the explanatory model, and add a
        # training-derived version for a genuinely pre-training screen.
        sig = compute_signal_features(te)
        train_sig = compute_signal_features(tr)
        feats[pid] = {
            "patient_id": pid,
            "sampling_rate_minutes": sampling_rate,
            "insulin_type": find_insulin_type(data_root, pid),
            "train_std": ts["std"], "test_std": es["std"],
            "train_tir": ts["tir"], "test_tir": es["tir"],
            "train_n": shift["train_n"], "test_n": shift["test_n"],
            "shift_score": shift["shift_score"],
            "kl_divergence": shift["kl_divergence"],
            "sample_entropy": sig["sample_entropy"],
            "autocorr_lag1": sig["autocorr_lag1"],
            "train_sample_entropy": train_sig["sample_entropy"],
            "train_autocorr_lag1": train_sig["autocorr_lag1"],
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


def load_configured_mae_by_horizon(metrics_paths, model, mode):
    """Read one configured seed's ``metrics.json`` per horizon.

    Horizon, model, mode, and units come from the saved patient metadata. This
    avoids inferring horizons from directory names and prevents mixing seeds or
    duplicate runs at the same horizon without a deliberate selection.
    """
    out = {}
    for path in metrics_paths:
        path = Path(path)
        with path.open() as fh:
            metrics = json.load(fh)
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"No patient metrics in {path}")
        horizons = set()
        patient_mae = {}
        for pid, entry in metrics.items():
            info = entry.get("model_info", {})
            if info.get("model_name", "").upper() != model.upper():
                continue
            if info.get("mode") != mode:
                raise ValueError(f"{path}: patient {pid} has mode {info.get('mode')}, expected {mode}")
            if info.get("target_units") != "mg/dL":
                raise ValueError(f"{path}: patient {pid} does not report mg/dL targets")
            horizon = info.get("prediction_horizon_minutes")
            if not isinstance(horizon, int) or horizon <= 0:
                raise ValueError(f"{path}: patient {pid} has no valid horizon in model_info")
            horizons.add(horizon)
            if entry.get("mae") is not None:
                patient_mae[int(pid)] = float(entry["mae"])
        if not patient_mae or len(horizons) != 1:
            raise ValueError(f"{path}: expected one {model} horizon with patient MAEs")
        horizon, = horizons
        if horizon in out:
            raise AmbiguousExperimentsError(
                f"Two configured metrics files selected for {horizon} min; choose one seed/run per horizon"
            )
        out[horizon] = patient_mae
        print(f"  {path}: horizon={horizon}, {len(patient_mae)} patients")
    return dict(sorted(out.items()))


def configured_sampling_rate(metrics_paths):
    """Read and validate the sampling rate shared by selected configured runs."""
    rates = {}
    for metrics_path in metrics_paths:
        tracking_path = Path(metrics_path).with_name("tracking.json")
        with tracking_path.open() as fh:
            tracking = json.load(fh)
        rate = tracking["config"]["preprocessing"]["sampling_rate"]
        if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
            raise ValueError(f"{tracking_path}: invalid configured sampling rate {rate!r}")
        rates[str(metrics_path)] = rate
    distinct = set(rates.values())
    if len(distinct) != 1:
        detail = ", ".join(f"{path}={rate}" for path, rate in rates.items())
        raise ValueError(f"Configured experiments use different sampling rates: {detail}")
    return distinct.pop()


def _resolve_patient_release(patient_id, declared_version, path):
    """Return the OhioT1DM release a patient ID belongs to, checking the config.

    ``declared_version`` is the parent config's ``data.version``. ``both`` admits
    either release; a single-release label must match the ID, which is how a
    hand-edited or mislabelled tracking config gets caught.
    """
    for release in ("2018", "2020"):
        if patient_id in OHIO_PATIENTS[release]:
            if declared_version not in ("both", release):
                raise ValueError(
                    f"{path}: patient {patient_id} belongs to OhioT1DM {release}, "
                    f"but the run is configured as version {declared_version}"
                )
            return release
    raise ValueError(f"{path}: {patient_id} is not an OhioT1DM patient ID")


def load_aggregate_mae_by_horizon(aggregate_paths, model, require_transfer_pairs=False):
    """Read seed-mean MAE by mode/horizon/patient from configured parent runs.

    Each parent aggregate is one horizon. The saved tracking config supplies
    the horizon and dataset identity; its rows supply the mode and seed set.
    Disjoint 2018/2020 patient cohorts can be combined at the same horizon, and
    so can a ``version: both`` parent, because each patient's release is resolved
    from its own ID rather than from the parent's version label. Duplicate
    patient/mode/horizon inputs and changing cohorts are rejected.
    """
    by_mode = {}
    seed_sets = {}
    dataset_name = None
    source_patients = {}
    for path in aggregate_paths:
        path = Path(path)
        tracking_path = path.with_name("tracking.json")
        with tracking_path.open() as fh:
            tracking = json.load(fh)
        config = tracking["config"]
        if config["model"]["type"].upper() != model.upper():
            raise ValueError(f"{path}: configured model is not {model}")
        dataset = config["data"]["dataset"]
        if dataset_name is None:
            dataset_name = dataset
        elif dataset != dataset_name:
            raise ValueError(f"{path}: dataset differs from another horizon")
        version = config["data"]["version"]
        if version not in OHIO_PATIENTS:
            raise ValueError(f"{path}: unknown configured OhioT1DM version {version!r}")
        configured_patients = config["data"]["patients"]
        if configured_patients == "all":
            configured_patients = OHIO_PATIENTS[version]
        horizon = config["preprocessing"]["prediction_horizon"] * config["preprocessing"]["sampling_rate"]
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"{path}: invalid configured horizon")
        source_patients.setdefault(horizon, set()).update(configured_patients)
        # OhioT1DM patient IDs are disjoint between releases, so a patient's
        # release follows from its ID. Checking the ID against the parent's
        # version label -- rather than comparing labels between parents -- is
        # what catches a wrong config, and it lets a `version: both` parent sit
        # alongside separate 2018/2020 parents for the same cohort.
        for pid in configured_patients:
            _resolve_patient_release(pid, version, path)
        with path.open() as fh:
            aggregates = json.load(fh)
        if not isinstance(aggregates, list) or not aggregates:
            raise ValueError(f"{path}: no aggregate metric rows")
        selected = 0
        for row in aggregates:
            if row.get("model", "").upper() != model.upper():
                continue
            mode, pid = row["mode"], int(row["patient_id"])
            mae = row.get("metrics", {}).get("mae", {}).get("mean")
            if mae is None:
                continue
            key = (mode, horizon, pid)
            if key in seed_sets:
                raise ValueError(f"{path}: duplicate aggregate MAE for {key}")
            by_mode.setdefault(mode, {}).setdefault(horizon, {})[pid] = float(mae)
            seed_sets[key] = tuple(sorted(row.get("seeds", [])))
            selected += 1
        if not selected:
            raise ValueError(f"{path}: no {model} MAE rows")
        print(f"  {path}: horizon={horizon}, {selected} mode/patient rows")
    cohorts = list(source_patients.values())
    if any(cohort != cohorts[0] for cohort in cohorts[1:]):
        raise ValueError("Configured patient cohort differs between horizons; supply both dataset versions at each horizon")
    if require_transfer_pairs:
        validate_transfer_pairs(by_mode, seed_sets, source_patients)
    return {mode: dict(sorted(horizons.items())) for mode, horizons in by_mode.items()}, seed_sets


def validate_transfer_pairs(by_mode, seed_sets, expected_patients=None):
    """Reject incomplete mode/seed pairs before transfer-benefit analysis.

    When run tracking supplies a cohort, require both modes for every declared
    patient at every horizon; neither mode may silently omit a patient.
    """
    regular = by_mode.get("regular", {})
    transfer = by_mode.get("transfer", {})
    horizons = set(regular) | set(transfer)
    if expected_patients is not None:
        horizons |= set(expected_patients)
    if not horizons:
        raise ValueError("Transfer-benefit comparison needs regular and transfer MAE")
    for horizon in sorted(horizons):
        reg_patients = set(regular.get(horizon, {}))
        tl_patients = set(transfer.get(horizon, {}))
        patients = reg_patients | tl_patients
        if expected_patients is not None:
            patients |= set(expected_patients.get(horizon, ()))
        for pid in sorted(patients):
            missing = [mode for mode, present in (("regular", pid in reg_patients),
                                                   ("transfer", pid in tl_patients))
                       if not present]
            if missing:
                raise ValueError(
                    f"{horizon} min patient {pid}: missing {' and '.join(missing)} MAE "
                    "for transfer-benefit comparison"
                )
            if expected_patients is not None and pid not in expected_patients.get(horizon, ()):
                raise ValueError(f"{horizon} min patient {pid}: not in the configured cohort")
            reg_seeds = seed_sets.get(("regular", horizon, pid), ())
            tl_seeds = seed_sets.get(("transfer", horizon, pid), ())
            if not reg_seeds or reg_seeds != tl_seeds:
                raise ValueError(
                    f"{horizon} min patient {pid}: regular and transfer have different completed seeds"
                )


def build_transfer_benefit_long(feats, by_mode, seed_sets):
    """Paired regular-vs-transfer MAE per matched patient and horizon.

    Screening ranks patients by ``transfer_mae - regular_mae``: how much *worse*
    transfer is, so that higher means less benefit from transfer and the screens
    keep their "higher = higher outcome" direction without a hidden negation.
    That column is the exact negation of ``transfer_benefit_mg_dl``, and both are
    written out so the two CSVs cross-reference by name rather than by sign.

    Both modes must have the same completed seed set for each paired
    patient/horizon; otherwise the aggregate means are not paired.
    """
    validate_transfer_pairs(by_mode, seed_sets)
    regular = by_mode["regular"]
    transfer = by_mode["transfer"]
    transfer_minus_regular = {}
    benefit_rows = []
    for horizon in sorted(regular):
        transfer_minus_regular[horizon] = {}
        missing_features = sorted(set(regular[horizon]) - set(feats))
        if missing_features:
            raise ValueError(
                f"{horizon} min patient {missing_features[0]}: missing glucose features "
                "for transfer-benefit comparison"
            )
        for pid in sorted(regular[horizon]):
            reg_seeds = seed_sets.get(("regular", horizon, pid), ())
            regular_mae, transfer_mae = regular[horizon][pid], transfer[horizon][pid]
            transfer_minus_regular[horizon][pid] = transfer_mae - regular_mae
            benefit_rows.append({"patient_id": pid, "horizon": horizon,
                                 "regular_mae": regular_mae,
                                 "transfer_mae": transfer_mae,
                                 "transfer_benefit_mg_dl": regular_mae - transfer_mae,
                                 "transfer_minus_regular_mae": transfer_mae - regular_mae,
                                 "matched_seeds": ",".join(map(str, reg_seeds))})
    _, long, horizons = build_tables(feats, transfer_minus_regular)
    return pd.DataFrame(benefit_rows), long, horizons


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
                "train_sample_entropy": f.get("train_sample_entropy", np.nan),
                "train_autocorr_lag1": f.get("train_autocorr_lag1", np.nan),
                "mae": mae_by_horizon[h].get(pid, np.nan),
            })
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows).dropna(subset=["shift_score", "mae"])
    return wide, long, horizons


def patient_difficulty_table(long, horizons):
    """One combined-horizon difficulty value per complete-case patient.

    Each horizon is standardized across the *same* complete-case patients before
    averaging. This avoids both the different MAE scales across horizons and the
    false sample size obtained by treating repeated horizons as independent.
    Patients missing any requested horizon are excluded from this combined score.
    """
    empty = pd.DataFrame(columns=[
        "patient_id", "difficulty_z", "shift_score", "test_std", "test_tir",
        "sample_entropy", "autocorr_lag1", "train_sample_entropy",
        "train_autocorr_lag1", "mae_slope_per_min",
    ])
    if not horizons:
        return empty
    g = long[long["horizon"].isin(horizons)].dropna(
        subset=["patient_id", "shift_score", "mae"]
    ).copy()
    if g.duplicated(["patient_id", "horizon"]).any():
        raise ValueError("Duplicate patient/horizon MAE rows; select one run per horizon")
    complete = g.groupby("patient_id")["horizon"].nunique()
    g = g[g["patient_id"].isin(complete[complete == len(horizons)].index)]
    if g.empty:
        return empty
    g["mae_z"] = g.groupby("horizon")["mae"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else np.nan
    )
    # A constant horizon cannot be standardized. Exclude the combined result
    # rather than silently averaging only the remaining horizons.
    if g["mae_z"].isna().any():
        return empty
    patient = g.groupby("patient_id", as_index=False).agg(
        difficulty_z=("mae_z", "mean"),
        shift_score=("shift_score", "first"),
        test_std=("test_std", "first"),
        test_tir=("test_tir", "first"),
        sample_entropy=("sample_entropy", "first"),
        autocorr_lag1=("autocorr_lag1", "first"),
        train_sample_entropy=("train_sample_entropy", "first"),
        train_autocorr_lag1=("train_autocorr_lag1", "first"),
    )
    # A patient-level slope tests whether difficulty rises faster with horizon
    # for patients with larger shift, without pretending the horizons are new patients.
    if len(horizons) >= 2:
        slopes = g.groupby("patient_id").apply(
            lambda s: np.polyfit(s["horizon"], s["mae"], 1)[0],
            include_groups=False,
        ).rename("mae_slope_per_min")
        patient = patient.merge(slopes, on="patient_id")
    return patient


def run_correlations(long, horizons):
    """Correlations with one independent observation per patient in every row."""
    rows = []
    groups = [(h, long[long["horizon"] == h], "mae") for h in horizons]
    groups.append(("patient_mean_within_horizon_z",
                   patient_difficulty_table(long, horizons), "difficulty_z"))
    for h, group, y_col in groups:
        g = group.dropna(subset=["shift_score", y_col])
        n = g["patient_id"].nunique()
        if len(g) != n:
            raise ValueError("Correlation input contains repeated patient rows")
        if n >= 3 and g["shift_score"].nunique() > 1 and g[y_col].nunique() > 1:
            pr, pp = pearsonr(g["shift_score"], g[y_col])
            sr, sp = spearmanr(g["shift_score"], g[y_col])
        else:
            pr = pp = sr = sp = np.nan
        rows.append({"horizon": h, "n_patients": n,
                     "pearson_r": pr, "pearson_p": pp,
                     "spearman_rho": sr, "spearman_p": sp})
    return pd.DataFrame(rows)


# name -> (data a screen needs, sign that makes "higher = harder").
# The `data_available` column is the point of the comparison: shift_score cannot
# be computed until the test split exists, whereas the irregularity features have
# a training-only twin. Both irregularity features appear on both sides so the
# pre-training set is complete -- with only one of them you cannot say whether
# irregularity computed before training matches shift, which is the reviewers'
# question (Review-Points 2.2).
_SCREEN_CANDIDATES = {
    "shift_score": ("test + train", 1),
    "sample_entropy": ("test", 1),
    "autocorr_lag1": ("test", -1),
    "train_sample_entropy": ("train", 1),
    "train_autocorr_lag1": ("train", -1),
}

# Per-horizon outcome each screening target ranks patients by. Naming it in the
# CSV keeps a reader from having to infer a sign convention from the target name.
_SCREEN_TARGETS = {
    "forecast_difficulty": ("mae", "mg/dL"),
    "least_transfer_benefit": ("transfer_minus_regular_mae", "mg/dL"),
}


def _matched_screen_groups(long, horizons, target="forecast_difficulty"):
    """Yield (horizon label, matched-patient frame, outcome column, name, units).

    ``long``'s outcome column is always called ``mae`` because :func:`build_tables`
    builds it; ``target`` says what that column actually holds, so the CSVs can
    name it instead of leaving the reader to guess.
    """
    if target not in _SCREEN_TARGETS:
        raise ValueError(f"Unknown screening target {target!r}")
    outcome_name, outcome_units = _SCREEN_TARGETS[target]
    groups = [(str(h), long[long["horizon"] == h], "mae", outcome_name, outcome_units)
              for h in horizons]
    groups.append(("patient_mean_within_horizon_z",
                   patient_difficulty_table(long, horizons), "difficulty_z",
                   f"{outcome_name}_within_horizon_z", "within_horizon_z"))
    for label, group, outcome, name, units in groups:
        g = group.dropna(subset=["patient_id", outcome, *_SCREEN_CANDIDATES]).copy()
        if g["patient_id"].nunique() != len(g):
            raise ValueError("Screen comparison contains repeated patient rows")
        yield label, g, outcome, name, units


def rank_difficulty_screens(long, horizons, target="forecast_difficulty"):
    """Patient-level ranks for every screen and outcome on the matched cohort."""
    rows = []
    for label, g, outcome, outcome_name, outcome_units in _matched_screen_groups(
            long, horizons, target):
        outcome_ranks = g[outcome].rank(method="min", ascending=False)
        for name, (availability, direction) in _SCREEN_CANDIDATES.items():
            scores = direction * g[name]
            ranks = scores.rank(method="min", ascending=False)
            for pid, score, rank, observed, observed_rank in zip(
                    g["patient_id"], scores, ranks, g[outcome], outcome_ranks):
                rows.append({
                    "target": target, "horizon": label, "patient_id": pid,
                    "screen": name, "data_available": availability,
                    "screen_score": score, "screen_rank": rank,
                    "outcome_name": outcome_name,
                    "outcome_score": observed, "outcome_rank": observed_rank,
                    "outcome_units": outcome_units,
                })
    return pd.DataFrame(rows)


def compare_difficulty_screens(long, horizons, target="forecast_difficulty"):
    """Compare screens using matched patients, rank association, and top quartile.

    Higher shift and entropy, and lower autocorrelation, indicate a higher
    expected outcome. Test-derived screens require held-out CGM; the train_*
    screens can be computed before any model is trained.

    ``top_quartile_hit_rate`` is the share of the k highest-outcome patients that
    a screen's own k highest-scoring patients recover. Read it against
    ``top_quartile_chance_rate`` (= k/n), which is what a screen ranking patients
    at random scores: at n=12, k=3, chance is 0.25, so a hit rate of 0.33 is one
    patient better than chance, not a result.
    """
    rows = []
    for label, g, outcome, outcome_name, outcome_units in _matched_screen_groups(
            long, horizons, target):
        n = len(g)
        k = max(1, int(np.ceil(n / 4))) if n >= 8 else 0
        highest = set(g.nlargest(k, outcome)["patient_id"]) if k else set()
        for name, (availability, direction) in _SCREEN_CANDIDATES.items():
            score = direction * g[name]
            if n >= 3 and score.nunique() > 1 and g[outcome].nunique() > 1:
                rho, p = spearmanr(score, g[outcome])
            else:
                rho = p = np.nan
            identified = set(g.assign(screen_score=score).nlargest(k, "screen_score")["patient_id"]) if k else set()
            hits = len(highest & identified)
            rows.append({
                "target": target, "horizon": label, "screen": name,
                "data_available": availability,
                "outcome_name": outcome_name, "outcome_units": outcome_units,
                "n_patients": n, "spearman_rho": rho, "spearman_p": p,
                "top_quartile_n": k,
                "top_quartile_hits": hits if k else np.nan,
                "top_quartile_hit_rate": (hits / k) if k else np.nan,
                "top_quartile_chance_rate": (k / n) if k else np.nan,
            })
    return pd.DataFrame(rows)


def run_regressions(long, horizons):
    """Patient-level exploratory OLS, per horizon and combined horizon."""
    import statsmodels.formula.api as smf  # lazy: heavy import (see note at top)

    lines = []
    lines.append("Independent unit: patient. Horizons are repeated measurements, not extra patients.")
    lines.append("All regressions are exploratory at n≈12; coefficient p-values and R² are unstable.\n")

    def fit(df, label, outcome, outcome_desc):
        """Fit the shared 3-predictor model, naming the formula and its units.

        ``outcome`` differs between blocks -- mg/dL MAE per horizon, a unitless
        z-score for the combined block -- so coefficients are not comparable
        across blocks. Each block prints its own formula and outcome description
        rather than leaving that to be inferred.
        """
        formula = f"{outcome} ~ shift_score + test_std + test_tir"
        df = df.dropna(subset=[outcome, "shift_score", "test_std", "test_tir"])
        lines.append(f"Model: {formula}")
        lines.append(f"  outcome: {outcome_desc}")
        if len(df) < 5:
            lines.append(f"[{label}] insufficient data (n_patients={len(df)})\n")
            return
        m = smf.ols(formula, data=df).fit()
        lines.append(f"[{label}]  n_patients={int(m.nobs)}  R^2={m.rsquared:.3f}  adj_R^2={m.rsquared_adj:.3f}  F_p={m.f_pvalue:.3f}")
        for name in m.params.index:
            lines.append(f"    {name:>12}: coef={m.params[name]:+.4f}  p={m.pvalues[name]:.3f}")
        lines.append("")

    for h in horizons:
        fit(long[long["horizon"] == h], f"{h} min", "mae",
            f"MAE at {h} min, mg/dL")

    patient = patient_difficulty_table(long, horizons)
    if not patient.empty:
        fit(patient, "patient mean within-horizon z", "difficulty_z",
            "mean within-horizon z-score of MAE across "
            f"{len(horizons)} horizon(s), unitless (1.0 = one between-patient SD)")

    lines.append("Horizon trend: correlation of shift_score with each patient's MAE slope (mg/dL per min)")
    if len(horizons) < 2:
        lines.append("[patient horizon trend] skipped (needs >= 2 horizons; single-horizon run)\n")
    else:
        trend = patient.dropna(subset=["shift_score", "mae_slope_per_min"])
        if len(trend) >= 3 and trend["shift_score"].nunique() > 1 and trend["mae_slope_per_min"].nunique() > 1:
            tr, tp = pearsonr(trend["shift_score"], trend["mae_slope_per_min"])
            lines.append(f"[patient horizon trend]  n_patients={len(trend)}  Pearson r={tr:+.3f}  p={tp:.3f}\n")
        else:
            lines.append(f"[patient horizon trend] insufficient variation (n_patients={len(trend)})\n")

    lines.append("Model: difficulty_z ~ shift_score + test_std + test_tir + sample_entropy + autocorr_lag1")
    lines.append("  outcome: mean within-horizon z-score of MAE, unitless (same as the block above)")
    lines.append("Five predictors on ≈12 patients: exploratory only; do not use its p-values as confirmatory evidence.")
    aug = patient.dropna(subset=["difficulty_z", "shift_score", "test_std", "test_tir",
                                 "sample_entropy", "autocorr_lag1"])
    if len(aug) >= 8:
        m = smf.ols(
            "difficulty_z ~ shift_score + test_std + test_tir + sample_entropy + autocorr_lag1",
            data=aug,
        ).fit()
        lines.append(f"[patient + signal]  n_patients={int(m.nobs)}  R^2={m.rsquared:.3f}  adj_R^2={m.rsquared_adj:.3f}")
        for name in ["shift_score", "test_std", "test_tir", "sample_entropy", "autocorr_lag1"]:
            if name in m.params.index:
                lines.append(f"    {name:>14}: coef={m.params[name]:+.4f}")
        lines.append("")
    else:
        lines.append(f"[patient + signal] insufficient data (n_patients={len(aug)})\n")
    return "\n".join(lines)


# The two result sources spell the training modes differently: legacy runs encode
# them in the experiment directory name (``..._30min_tl``), configured runs record
# them in model_info/aggregate rows as ``regular``/``transfer``. Accepting all four
# spellings and silently finding nothing is the trap this guard closes.
LEGACY_MODES = {"tl", "rl"}
CONFIGURED_MODES = {"regular", "transfer"}
_MODE_EQUIVALENT = {"tl": "transfer", "rl": "regular",
                    "transfer": "tl", "regular": "rl"}


def _check_mode_matches_source(mode, configured):
    """Reject a --mode spelling that the selected result source never uses."""
    if configured and mode in LEGACY_MODES:
        raise ValueError(
            f"--mode {mode} names a legacy --results-dir experiment suffix; configured "
            f"runs record modes as regular/transfer. Use --mode {_MODE_EQUIVALENT[mode]}."
        )
    if not configured and mode in CONFIGURED_MODES:
        raise ValueError(
            f"--mode {mode} names a configured-run mode; --results-dir experiments are "
            f"matched by directory suffix. Use --mode {_MODE_EQUIVALENT[mode]}, or pass "
            f"--configured-aggregate/--configured-metrics."
        )


def main():
    ap = argparse.ArgumentParser(description="Phase 1 shift-vs-MAE analysis.")
    ap.add_argument("--results-dir", default="test_results")
    ap.add_argument("--configured-metrics", action="append", default=[],
                    help="Path to a configured seed's metrics.json; repeat once per horizon")
    ap.add_argument("--configured-aggregate", action="append", default=[],
                    help="Path to a parent aggregate_metrics.json; repeat once per horizon")
    ap.add_argument("--compare-transfer-benefit", action="store_true",
                    help="With configured aggregates, compare screens for smallest regular-minus-transfer MAE benefit")
    ap.add_argument("--output-dir", default="new_analysis")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model", default="GRU",
                    choices=["RNN", "LSTM", "GRU", "Transformer"])
    ap.add_argument("--mode", default="tl", choices=sorted(LEGACY_MODES | CONFIGURED_MODES),
                    help="tl/rl name legacy --results-dir experiment directories; "
                         "regular/transfer name the mode recorded by configured runs")
    args = ap.parse_args()

    out = Path(args.output_dir)

    print("Loading MAE by horizon...")
    try:
        by_mode = seed_sets = None
        sampling_rate = 5  # Legacy experiments use OhioT1DM's 5-minute grid.
        if args.configured_aggregate and args.configured_metrics:
            raise ValueError("Select configured aggregates or seed metrics, not both")
        if args.compare_transfer_benefit and not args.configured_aggregate:
            raise ValueError("--compare-transfer-benefit requires --configured-aggregate")
        _check_mode_matches_source(
            args.mode, bool(args.configured_aggregate or args.configured_metrics)
        )
        if args.configured_aggregate:
            by_mode, seed_sets = load_aggregate_mae_by_horizon(
                args.configured_aggregate, args.model,
                require_transfer_pairs=args.compare_transfer_benefit,
            )
            sampling_rate = configured_sampling_rate(args.configured_aggregate)
            mae_by_horizon = by_mode.get(args.mode, {})
            if not mae_by_horizon:
                raise ValueError(f"No aggregate MAE for mode {args.mode}")
        elif args.configured_metrics:
            mae_by_horizon = load_configured_mae_by_horizon(args.configured_metrics, args.model, args.mode)
            sampling_rate = configured_sampling_rate(args.configured_metrics)
        else:
            mae_by_horizon = load_mae_by_horizon(args.results_dir, args.mode, args.model)
    except (AmbiguousExperimentsError, ValueError) as e:
        print(f"✗ {e}")
        return 2
    if not mae_by_horizon:
        print("✗ No MAE data found.")
        return 1
    patient_ids = sorted({pid for h in mae_by_horizon.values() for pid in h})

    feats = load_glucose_features(args.data_root, patient_ids, sampling_rate=sampling_rate)
    if not feats:
        print("✗ No glucose features computed.")
        return 1

    wide, long, horizons = build_tables(feats, mae_by_horizon)
    if args.compare_transfer_benefit:
        try:
            benefit, benefit_long, benefit_horizons = build_transfer_benefit_long(
                feats, by_mode, seed_sets
            )
        except ValueError as e:
            print(f"✗ {e}")
            return 2
        benefit_screens = compare_difficulty_screens(
            benefit_long, benefit_horizons, target="least_transfer_benefit"
        )

    # Persist tables
    out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out / "patient_feature_table.csv", index=False)
    long.to_csv(out / "shift_mae_long.csv", index=False)
    patient_difficulty_table(long, horizons).to_csv(out / "patient_difficulty.csv", index=False)
    corr = run_correlations(long, horizons)
    corr.to_csv(out / "shift_mae_correlations.csv", index=False)
    screens = compare_difficulty_screens(long, horizons)
    screens.to_csv(out / "difficulty_screen_comparison.csv", index=False)
    rank_difficulty_screens(long, horizons).to_csv(out / "difficulty_screen_rankings.csv", index=False)
    if args.compare_transfer_benefit:
        benefit.to_csv(out / "patient_transfer_benefit.csv", index=False)
        benefit_screens.to_csv(out / "transfer_benefit_screen_comparison.csv", index=False)
        rank_difficulty_screens(
            benefit_long, benefit_horizons, target="least_transfer_benefit"
        ).to_csv(out / "transfer_benefit_screen_rankings.csv", index=False)
    reg_txt = run_regressions(long, horizons)
    (out / "regression_summary.txt").write_text(reg_txt)

    stem = f"shift_vs_mae_{args.model}_{args.mode}"
    plot_shift_vs_mae_faceted(long, horizons, str(out / f"{stem}_faceted.png"), args.model)
    plot_shift_vs_mae(long, str(out / f"{stem}.png"), args.model, horizons=horizons)

    # Console summary
    print("\n=== patient_feature_table.csv ===")
    print(wide.to_string(index=False))
    print("\n=== correlations (shift_score vs MAE) ===")
    print(corr.to_string(index=False))
    print("\n=== matched-patient difficulty screens (exploratory) ===")
    print(screens.to_string(index=False))
    if args.compare_transfer_benefit:
        print("\n=== matched-patient screens for least benefit from transfer ===")
        print(benefit_screens.to_string(index=False))
    print("\n=== regression summary ===")
    print(reg_txt)
    print(f"\n✓ All Phase-1 outputs written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
