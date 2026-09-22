#!/usr/bin/env python3
"""
Per-patient distributional shift vs prediction error.
"""

import os
import sys
import json
import argparse
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.analysis.dataset.signal_table import read_signal_table, signature
from benchmark.analysis.experiments.shift_mae_plot import (
    plot_shift_vs_mae,
    plot_shift_vs_mae_faceted,
)
from benchmark.configs.config_manager import OHIO_PATIENTS

def configured_signal_signature(paths):
    """Combine compatible release-specific parents into one dataset cohort."""
    settings = None
    for path in paths:
        tracking_path = tracking_for_metrics(path)
        with tracking_path.open() as fh:
            current = signature(json.load(fh)["config"])
        if settings is None:
            settings = current
        else:
            if any(current[key] != settings[key] for key in ("dataset", "sampling_rate_minutes")):
                raise ValueError(f"{tracking_path}: dataset settings differ across selected runs")
            settings["patient_ids"] = sorted(set(settings["patient_ids"]) | set(current["patient_ids"]))
            settings["releases"] = sorted(set(settings["releases"]) | set(current["releases"]))
    return settings


def tracking_for_metrics(path):
    """Find the configured parent's tracking file for aggregate or seed metrics."""
    for parent in (Path(path).parent, *Path(path).parents):
        candidate = parent / "tracking.json"
        if candidate.is_file():
            return candidate
    raise ValueError(f"{path}: no configured parent tracking.json")


class AmbiguousExperimentsError(RuntimeError):
    """Raised when more than one experiment matches the same (mode, horizon)."""


def load_configured_mae_by_horizon(metrics_paths, model, mode, *,
                                   require_configured_cohort=False):
    """Read one configured seed's ``metrics.json`` per horizon.

    Horizon, model, mode, and units come from the saved patient metadata. This
    avoids inferring horizons from directory names and prevents mixing seeds or
    duplicate runs at the same horizon without a deliberate selection.
    """
    out = {}
    for path in metrics_paths:
        path = Path(path)
        expected_patients = None
        if require_configured_cohort:
            tracking_path = tracking_for_metrics(path)
            with tracking_path.open() as fh:
                expected_patients = configured_patients(json.load(fh)["config"], tracking_path)
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
        if expected_patients is not None:
            _require_expected_patients(
                patient_mae, expected_patients, path=path, context="per-seed MAE rows"
            )
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
        tracking_path = tracking_for_metrics(metrics_path)
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


def configured_patients(config, path):
    """Return and validate the patient cohort declared by a run manifest."""
    version = config["data"]["version"]
    if version not in OHIO_PATIENTS:
        raise ValueError(f"{path}: unknown configured OhioT1DM version {version!r}")
    patients = config["data"]["patients"]
    if patients == "all":
        patients = OHIO_PATIENTS[version]
    patients = {int(pid) for pid in patients}
    for pid in patients:
        _resolve_patient_release(pid, version, path)
    return patients


def _require_expected_patients(observed, expected, *, path, context):
    """Reject missing or unexpected MAE rows before they change the cohort."""
    observed, expected = set(observed), set(expected)
    if observed == expected:
        return
    details = []
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing:
        details.append(f"missing {missing}")
    if unexpected:
        details.append(f"unexpected {unexpected}")
    raise ValueError(f"{path}: {context} does not match the configured patient cohort ({'; '.join(details)})")


def load_aggregate_mae_by_horizon(aggregate_paths, model, require_transfer_pairs=False,
                                  require_equal_transfer_seeds=True):
    """Read seed-mean MAE by mode/horizon/patient from configured parent runs.
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
        patients = configured_patients(config, path)
        horizon = config["preprocessing"]["prediction_horizon"] * config["preprocessing"]["sampling_rate"]
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"{path}: invalid configured horizon")
        source_patients.setdefault(horizon, set()).update(patients)
        # OhioT1DM patient IDs are disjoint between releases, so a patient's
        # release follows from its ID. Checking the ID against the parent's
        # version label -- rather than comparing labels between parents -- is
        # what catches a wrong config, and it lets a `version: both` parent sit
        # alongside separate 2018/2020 parents for the same cohort.
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
        validate_transfer_pairs(
            by_mode, seed_sets, source_patients,
            require_equal_seeds=require_equal_transfer_seeds,
        )
    for mode, by_horizon in by_mode.items():
        missing_horizons = sorted(set(source_patients) - set(by_horizon))
        if missing_horizons:
            raise ValueError(
                f"{mode}: aggregate MAE is missing configured horizon(s) {missing_horizons}"
            )
        for horizon, expected in source_patients.items():
            _require_expected_patients(
                by_horizon[horizon], expected, path=f"{mode} {horizon} min aggregate",
                context="MAE rows",
            )
    return {mode: dict(sorted(horizons.items())) for mode, horizons in by_mode.items()}, seed_sets


def load_configured_seed_mae_by_horizon(aggregate_paths, model, mode):
    """Load per-seed patient MAE tables for one configured mode.
    """
    out = {}
    for aggregate_path in aggregate_paths:
        aggregate_path = Path(aggregate_path)
        tracking_path = aggregate_path.with_name("tracking.json")
        with tracking_path.open() as fh:
            tracking = json.load(fh)
        config = tracking["config"]
        if config["model"]["type"].upper() != model.upper():
            raise ValueError(f"{aggregate_path}: configured model is not {model}")
        horizon = config["preprocessing"]["prediction_horizon"] * config["preprocessing"]["sampling_rate"]
        expected_patients = configured_patients(config, aggregate_path)
        with aggregate_path.open() as fh:
            aggregate = json.load(fh)
        seeds = sorted({int(seed) for row in aggregate
                        if row.get("model", "").upper() == model.upper()
                        and row.get("mode") == mode
                        for seed in row.get("seeds", [])})
        if not seeds:
            raise ValueError(f"{aggregate_path}: no completed seeds for mode {mode}")
        for seed in seeds:
            metrics_path = aggregate_path.parent / mode / f"seed_{seed}" / "metrics.json"
            if not metrics_path.is_file():
                raise ValueError(f"{metrics_path}: missing per-seed metrics for seed sensitivity")
            with metrics_path.open() as fh:
                metrics = json.load(fh)
            patient_mae = {}
            for pid, entry in metrics.items():
                info = entry.get("model_info", {})
                if info.get("model_name", "").upper() != model.upper():
                    continue
                if info.get("mode") != mode:
                    raise ValueError(f"{metrics_path}: patient {pid} has mode {info.get('mode')}, expected {mode}")
                # The horizon above comes from the parent config; the child run
                # records what it actually predicted. Disagreement means the
                # tracking manifests do not describe the same run.
                recorded = info.get("prediction_horizon_minutes")
                if recorded is not None and recorded != horizon:
                    raise ValueError(
                        f"{metrics_path}: patient {pid} reports a {recorded} min horizon, "
                        f"but the parent config declares {horizon} min"
                    )
                if info.get("target_units") not in (None, "mg/dL"):
                    raise ValueError(f"{metrics_path}: patient {pid} does not report mg/dL targets")
                if entry.get("mae") is not None:
                    patient_mae[int(pid)] = float(entry["mae"])
            if not patient_mae:
                raise ValueError(f"{metrics_path}: no patient MAE values")
            _require_expected_patients(
                patient_mae, expected_patients, path=metrics_path, context="per-seed MAE rows"
            )
            key = (seed, horizon)
            existing = out.setdefault(key, {})
            overlap = set(existing) & set(patient_mae)
            if overlap:
                raise AmbiguousExperimentsError(
                    f"Duplicate seed/horizon patients for {key}: {sorted(overlap)}"
                )
            existing.update(patient_mae)
    return out


def validate_seed_cohorts(seed_mae, *, strict=True):
    """Reject seed tables that do not cover the same patients at every horizon
    """
    if not seed_mae:
        raise ValueError("No per-seed patient MAE tables were loaded")

    def report(message):
        if strict:
            raise ValueError(message)
        print(f"    WARNING: {message}")

    seeds_by_horizon = {}
    for seed, horizon in seed_mae:
        seeds_by_horizon.setdefault(horizon, set()).add(seed)
    if len({tuple(sorted(seeds)) for seeds in seeds_by_horizon.values()}) > 1:
        detail = "; ".join(f"{horizon} min={sorted(seeds)}"
                           for horizon, seeds in sorted(seeds_by_horizon.items()))
        report(
            "Configured horizons have different completed seed sets; "
            f"cannot compute matched seed sensitivity ({detail})"
        )
    for horizon, seeds in sorted(seeds_by_horizon.items()):
        cohorts = {seed: frozenset(seed_mae[(seed, horizon)]) for seed in sorted(seeds)}
        expected = max(cohorts.values(), key=len)
        short = {seed: sorted(expected - cohort)
                 for seed, cohort in cohorts.items() if cohort != expected}
        if short:
            detail = "; ".join(f"seed {seed} missing {missing}"
                               for seed, missing in sorted(short.items()))
            report(
                f"{horizon} min: completed seeds cover different patients, so the "
                f"nested bootstrap would silently drop the difference ({detail}). "
                "Re-run the missing seed/patient cells, or restrict "
                "--configured-aggregate to the seeds that completed everywhere"
            )


def validate_transfer_pairs(by_mode, seed_sets, expected_patients=None,
                            *, require_equal_seeds=True):
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
            if not set(reg_seeds) & set(tl_seeds):
                raise ValueError(
                    f"{horizon} min patient {pid}: regular and transfer have no completed seed in common"
                )
            if require_equal_seeds and reg_seeds != tl_seeds:
                raise ValueError(
                    f"{horizon} min patient {pid}: regular and transfer have different completed seeds"
                )


def _mean_over_seeds(seed_mae, horizon, pid, seeds):
    """Mean MAE for one patient over exactly ``seeds``, or None if any is absent."""
    values = []
    for seed in seeds:
        table = seed_mae.get((seed, horizon), {})
        if pid not in table:
            return None
        values.append(float(table[pid]))
    return float(np.mean(values)) if values else None


def build_transfer_benefit_long(feats, by_mode, seed_sets, *, strict_seeds=True,
                                seed_mae_by_mode=None):
    """Paired regular-vs-transfer MAE per matched patient and horizon.
    """
    validate_transfer_pairs(by_mode, seed_sets, require_equal_seeds=strict_seeds)
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
            transfer_seeds = seed_sets.get(("transfer", horizon, pid), ())
            matched_seeds = tuple(sorted(set(reg_seeds) & set(transfer_seeds)))
            if not matched_seeds:
                raise ValueError(
                    f"{horizon} min patient {pid}: no completed seed is shared "
                    "by regular and transfer"
                )
            regular_mae, transfer_mae = regular[horizon][pid], transfer[horizon][pid]
            seed_match = "equal"
            if set(reg_seeds) != set(transfer_seeds):
                # The aggregate means average different seed sets; rebuild both
                # from the shared seeds so the difference is genuinely paired.
                if seed_mae_by_mode is None:
                    raise ValueError(
                        f"{horizon} min patient {pid}: regular and transfer completed "
                        f"different seeds ({list(reg_seeds)} vs {list(transfer_seeds)}); "
                        "per-seed tables are required to pair them"
                    )
                rebuilt = {
                    mode: _mean_over_seeds(seed_mae_by_mode.get(mode, {}),
                                           horizon, pid, matched_seeds)
                    for mode in ("regular", "transfer")
                }
                if rebuilt["regular"] is None or rebuilt["transfer"] is None:
                    raise ValueError(
                        f"{horizon} min patient {pid}: shared seeds "
                        f"{list(matched_seeds)} are missing from the per-seed MAE "
                        "tables, so the pair cannot be rebuilt"
                    )
                regular_mae, transfer_mae = rebuilt["regular"], rebuilt["transfer"]
                seed_match = "intersection"
            transfer_minus_regular[horizon][pid] = transfer_mae - regular_mae
            benefit_rows.append({"patient_id": pid, "horizon": horizon,
                                 "regular_mae": regular_mae,
                                 "transfer_mae": transfer_mae,
                                 "transfer_benefit_mg_dl": regular_mae - transfer_mae,
                                 "transfer_minus_regular_mae": transfer_mae - regular_mae,
                                 "matched_seeds": ",".join(map(str, matched_seeds)),
                                 "seed_match": seed_match})
    _, long, horizons = build_tables(feats, transfer_minus_regular)
    return pd.DataFrame(benefit_rows), long, horizons


def paired_transfer_effect_bootstrap(regular_seed_mae, transfer_seed_mae, horizons,
                                     *, 
                                     replicates=10000, random_seed=42):
    """Estimate the paired TL-RL MAE effect independently for each horizon.

    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(random_seed)
    rows = []
    for horizon in sorted(horizons):
        shared_seeds = sorted(
            {seed for seed, h in regular_seed_mae if h == horizon}
            & {seed for seed, h in transfer_seed_mae if h == horizon}
        )
        paired, baseline = {}, {}
        for seed in shared_seeds:
            regular = regular_seed_mae.get((seed, horizon), {})
            transfer = transfer_seed_mae.get((seed, horizon), {})
            patients = sorted(set(regular) & set(transfer))
            if patients:
                paired[seed] = {
                    pid: float(transfer[pid] - regular[pid]) for pid in patients
                }
                baseline[seed] = {pid: float(regular[pid]) for pid in patients}
        # A patient must be present for every seed used in the nested draw so
        # that the seed comparison remains paired in every replicate.
        common = sorted(set.intersection(*(set(values) for values in paired.values()))) \
            if paired else []
        usable_seeds = [seed for seed in shared_seeds if seed in paired]
        if len(common) < 1 or not usable_seeds:
            rows.append({
                "horizon": horizon, "tl_minus_rl_mae_mg_dl": np.nan,
                "ci95_low": np.nan, "ci95_high": np.nan,
                "tl_minus_rl_pct": np.nan,
                "pct_ci95_low": np.nan, "pct_ci95_high": np.nan,
                "n_patients": len(common), "training_seeds": len(usable_seeds),
                "matched_seeds": ",".join(map(str, usable_seeds)),
                "valid_replicates": 0, "replicates": replicates,
            })
            continue
        observed = np.asarray([
            [paired[seed][pid] for pid in common] for seed in usable_seeds
        ], dtype=float)
        observed_regular = np.asarray([
            [baseline[seed][pid] for pid in common] for seed in usable_seeds
        ], dtype=float)
        draws = np.empty(replicates, dtype=float)
        pct_draws = np.empty(replicates, dtype=float)
        for i in range(replicates):
            seed_index = int(rng.integers(0, len(usable_seeds)))
            patient_indices = rng.integers(0, len(common), size=len(common))
            difference = float(np.mean(observed[seed_index, patient_indices]))
            reference = float(np.mean(observed_regular[seed_index, patient_indices]))
            draws[i] = difference
            # A zero-MAE reference cannot happen with real CGM error, but a
            # NaN here is the honest value and is excluded from the interval
            # rather than propagating a division warning into the CSV.
            pct_draws[i] = 100.0 * difference / reference if reference else np.nan
        valid = draws[np.isfinite(draws)]
        pct_valid = pct_draws[np.isfinite(pct_draws)]
        mean_regular = float(np.mean(observed_regular))
        rows.append({
            "horizon": horizon,
            "tl_minus_rl_mae_mg_dl": float(np.mean(observed)),
            "ci95_low": float(np.quantile(valid, 0.025)) if valid.size else np.nan,
            "ci95_high": float(np.quantile(valid, 0.975)) if valid.size else np.nan,
            "tl_minus_rl_pct": (100.0 * float(np.mean(observed)) / mean_regular
                                if mean_regular else np.nan),
            "pct_ci95_low": (float(np.quantile(pct_valid, 0.025))
                             if pct_valid.size else np.nan),
            "pct_ci95_high": (float(np.quantile(pct_valid, 0.975))
                              if pct_valid.size else np.nan),
            "n_patients": len(common), "training_seeds": len(usable_seeds),
            "matched_seeds": ",".join(map(str, usable_seeds)),
            "valid_replicates": int(valid.size), "replicates": replicates,
        })
    return pd.DataFrame(rows)


def build_tables(feats, mae_by_horizon):
    """Return (wide 12-row feature table, long patient x horizon table)."""
    horizons = sorted(mae_by_horizon)
    signal_quality_columns = (
        "sample_entropy_valid_n", "sample_entropy_used_n",
        "sample_entropy_template_n", "autocorr_valid_n",
        "autocorr_adjacent_pair_n",
    )
    wide_rows, long_rows = [], []
    for pid, f in sorted(feats.items()):
        row = dict(f)
        for h in horizons:
            row[f"mae_{h}"] = mae_by_horizon[h].get(pid, np.nan)
            long_rows.append({
                "patient_id": pid, "horizon": h,
                "shift_score": f["shift_score"], "js_divergence": f["js_divergence"],
                "out_of_support_mass": f["out_of_support_mass"],
                "test_std": f["test_std"], "test_tir": f["test_tir"],
                "train_std": f["train_std"], "train_tir": f["train_tir"],
                "sample_entropy": f.get("sample_entropy", np.nan),
                "autocorr_lag1": f.get("autocorr_lag1", np.nan),
                "train_sample_entropy": f.get("train_sample_entropy", np.nan),
                "train_autocorr_lag1": f.get("train_autocorr_lag1", np.nan),
                **{name: f.get(name, np.nan) for name in signal_quality_columns},
                **{f"train_{name}": f.get(f"train_{name}", np.nan)
                   for name in signal_quality_columns},
                "mae": mae_by_horizon[h].get(pid, np.nan),
            })
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows).dropna(subset=["shift_score", "mae"])
    return wide, long, horizons


def patient_difficulty_table(long, horizons):
    """One combined-horizon difficulty value per complete-case patient."""
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
    """Correlations with one independent observation per patient in every row.
    """
    rows = []
    groups = [(str(h), long[long["horizon"] == h], "mae") for h in horizons]
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


def permutation_correlation(frame, x_col, y_col, *, permutations=10000,
                            random_seed=42, method="spearman"):
    """Test an association by shuffling complete patient assignments.
    """
    if permutations < 1:
        raise ValueError("permutations must be positive")
    data = frame.dropna(subset=[x_col, y_col, "patient_id"]).copy()
    if data["patient_id"].duplicated().any():
        raise ValueError("Permutation input contains repeated patient rows")
    if len(data) < 3 or data[x_col].nunique() < 2 or data[y_col].nunique() < 2:
        return {"statistic": np.nan, "p_value": np.nan, "n_patients": int(len(data)),
                "permutations": int(permutations), "method": method}
    statistic_fn = spearmanr if method == "spearman" else pearsonr
    observed = float(statistic_fn(data[x_col], data[y_col]).statistic)
    rng = np.random.default_rng(random_seed)
    shuffled = data[x_col].to_numpy(copy=True)
    null = np.empty(permutations, dtype=float)
    for i in range(permutations):
        null[i] = float(statistic_fn(rng.permutation(shuffled), data[y_col]).statistic)
    p_value = (1.0 + np.count_nonzero(np.abs(null) >= abs(observed))) / (permutations + 1.0)
    return {"statistic": observed, "p_value": float(p_value),
            "n_patients": int(len(data)), "permutations": int(permutations),
            "method": method}


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values, NaN preserved in place.
    """
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(values[finite], kind="stable")]
    n = order.size
    scaled = values[order] * n / np.arange(1, n + 1)
    adjusted[order] = np.minimum.accumulate(scaled[::-1])[::-1].clip(max=1.0)
    return adjusted


def run_permutation_tests(long, horizons, *, permutations=10000, random_seed=42):
    """Return patient-level permutation tests for shift and irregularity screens.
    """
    groups = [(str(h), long[long["horizon"] == h], "mae") for h in horizons]
    groups.append(("patient_mean_within_horizon_z",
                   patient_difficulty_table(long, horizons), "difficulty_z"))
    rows = []
    for label, group, outcome in groups:
        for screen in _SCREEN_CANDIDATES:
            if screen not in group:
                continue
            for method in ("pearson", "spearman"):
                result = permutation_correlation(
                    group, screen, outcome, permutations=permutations,
                    random_seed=random_seed, method=method,
                )
                rows.append({"horizon": label, "screen": screen,
                             "outcome": outcome, **result,
                             "family": f"shift_and_irregularity_screens/{method}"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_value_bh"] = frame.groupby("family")["p_value"].transform(benjamini_hochberg)
    return frame


def seed_sensitivity(seed_tables, horizons):
    """Summarize per-seed patient-level associations without treating seeds as patients."""
    rows = []
    for seed, long in sorted(seed_tables.items()):
        corr = run_correlations(long, horizons)
        corr.insert(0, "training_seed", seed)
        rows.append(corr)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def nested_patient_seed_bootstrap(seed_tables, horizons, *, replicates=10000,
                                  random_seed=42):
    """Bootstrap correlations by drawing one seed and whole patients per replicate.

    """
    if not seed_tables:
        raise ValueError("At least one seed table is required")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(random_seed)
    seeds = sorted(seed_tables)
    rows = []
    groups = [(str(h), lambda table, h=h: table[table["horizon"] == h], "mae")
              for h in horizons]
    groups.append(("patient_mean_within_horizon_z",
                   lambda table: patient_difficulty_table(table, horizons),
                   "difficulty_z"))
    for label, select, outcome in groups:
        observed = []
        for seed in seeds:
            frame = select(seed_tables[seed]).dropna(subset=["patient_id", "shift_score", outcome])
            if frame["patient_id"].duplicated().any():
                raise ValueError("Seed bootstrap input contains repeated patient rows")
            observed.append(frame)
        common = set(observed[0]["patient_id"])
        for frame in observed[1:]:
            common &= set(frame["patient_id"])
        common = sorted(common)
        if len(common) < 3:
            rows.append({"horizon": label, "outcome": outcome, "n_patients": len(common),
                         "training_seeds": len(seeds), "correlation": np.nan,
                         "ci95_low": np.nan, "ci95_high": np.nan,
                         "valid_replicates": 0, "replicates": replicates})
            continue
        by_seed = {
            seed: frame.set_index("patient_id").loc[common]
            for seed, frame in zip(seeds, observed)
        }
        draws = np.empty(replicates, dtype=float)
        for i in range(replicates):
            seed = seeds[int(rng.integers(0, len(seeds)))]
            patient_ids = rng.choice(common, size=len(common), replace=True)
            sample = by_seed[seed].loc[patient_ids]
            draws[i] = float(spearmanr(sample["shift_score"], sample[outcome]).statistic)
        point = float(np.nanmean([spearmanr(frame["shift_score"], frame[outcome]).statistic
                                  for frame in by_seed.values()]))
        valid = draws[np.isfinite(draws)]
        rows.append({"horizon": label, "outcome": outcome, "n_patients": len(common),
                     "training_seeds": len(seeds), "correlation": point,
                     "ci95_low": float(np.quantile(valid, 0.025)) if valid.size else np.nan,
                     "ci95_high": float(np.quantile(valid, 0.975)) if valid.size else np.nan,
                     "valid_replicates": int(valid.size), "replicates": replicates})
    return pd.DataFrame(rows)


def screen_sensitivity(seed_tables, horizons, target="forecast_difficulty"):
    """Run matched screen comparisons independently for each training seed."""
    rows = []
    for seed, long in sorted(seed_tables.items()):
        result = compare_difficulty_screens(long, horizons, target=target)
        if not result.empty:
            result.insert(0, "training_seed", int(seed))
            rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# name -> (data a screen needs, sign that makes "higher = harder").

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


def regression_sensitivity(seed_tables, horizons):
    """Fit the existing exploratory models independently for every seed."""
    import statsmodels.formula.api as smf
    rows = []
    for seed, long in sorted(seed_tables.items()):
        patient = patient_difficulty_table(long, horizons)
        models = [(str(h), long[long["horizon"] == h], "mae",
                   "shift_score + test_std + test_tir") for h in horizons]
        if not patient.empty:
            models.append(("patient_mean_within_horizon_z", patient, "difficulty_z",
                           "shift_score + test_std + test_tir"))
            models.append(("patient_mean_within_horizon_z_signal", patient, "difficulty_z",
                           "shift_score + test_std + test_tir + sample_entropy + autocorr_lag1"))
        for label, frame, outcome, predictors in models:
            formula = f"{outcome} ~ {predictors}"
            needed = [outcome] + [term.strip() for term in predictors.split("+")]
            frame = frame.dropna(subset=needed)
            if len(frame) <= len(needed):
                continue
            fit = smf.ols(formula, data=frame).fit()
            for term in fit.params.index:
                if term == "Intercept":
                    continue
                rows.append({"training_seed": int(seed), "analysis": label,
                             "outcome": outcome, "term": term,
                             "n_patients": int(fit.nobs),
                             "coefficient": float(fit.params[term]),
                             "std_error": float(fit.bse[term]),
                             "p_value": float(fit.pvalues[term])})
    return pd.DataFrame(rows)


def _ols_coefficients(design, outcome):
    """Least-squares coefficients for ``outcome ~ design``, NaN when rank-deficient.

    A bootstrap resample can repeat patients until a predictor column becomes
    collinear. That replicate carries no information about the coefficients, so it
    is marked invalid instead of contributing whatever the pseudo-inverse returns.
    """
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return np.full(design.shape[1], np.nan)
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    return coefficients


def nested_regression_uncertainty(seed_tables, horizons, *, replicates=10000,
                                  random_seed=42):
    """Percentile intervals for the exploratory coefficients, over patients x seeds.

    Each replicate draws one training seed for the whole replicate and then
    resamples complete patients with replacement, matching
    :func:`nested_patient_seed_bootstrap`. Every coefficient of a replicate comes
    out of the *same* fit, so the terms describe one resampled cohort rather than
    several unrelated ones.

    The fit is a least-squares solve on the design matrix rather than a
    statsmodels formula call. These models are plain additive numeric terms, so
    the two are the same estimator, and refitting through the formula API once per
    term costs minutes per analysis while buying nothing. :func:`run_regressions`
    remains the statsmodels path for the inference table.
    """
    if not seed_tables or replicates < 1:
        raise ValueError("seed tables and positive replicates are required")
    rng = np.random.default_rng(random_seed)
    seeds = sorted(seed_tables)
    rows = []
    definitions = [(str(h), "mae", ["shift_score", "test_std", "test_tir"],
                    lambda frame, h=h: frame[frame["horizon"] == h]) for h in horizons]
    definitions.append(("patient_mean_within_horizon_z", "difficulty_z",
                        ["shift_score", "test_std", "test_tir"],
                        lambda frame: patient_difficulty_table(frame, horizons)))
    definitions.append(("patient_mean_within_horizon_z_signal", "difficulty_z",
                        ["shift_score", "test_std", "test_tir", "sample_entropy", "autocorr_lag1"],
                        lambda frame: patient_difficulty_table(frame, horizons)))
    for label, outcome, predictors, select in definitions:
        fitted = []
        for seed in seeds:
            frame = select(seed_tables[seed]).dropna(subset=[outcome, *predictors]).copy()
            if frame["patient_id"].duplicated().any():
                raise ValueError("Regression bootstrap input contains repeated patient rows")
            fitted.append(frame.set_index("patient_id"))
        common = sorted(set(fitted[0].index).intersection(*(set(x.index) for x in fitted[1:])))
        if len(common) <= len(predictors) + 1:
            continue
        # One design matrix per seed, patients in a shared order, so a replicate
        # is an index draw rather than a reindex-and-rebuild.
        designs = []
        for frame in fitted:
            aligned = frame.loc[common]
            designs.append((
                np.column_stack([np.ones(len(common)),
                                 aligned[list(predictors)].to_numpy(dtype=float)]),
                aligned[outcome].to_numpy(dtype=float),
            ))
        observed = np.array([_ols_coefficients(design, values)
                             for design, values in designs])
        draws = np.empty((replicates, 1 + len(predictors)), dtype=float)
        for i in range(replicates):
            design, values = designs[int(rng.integers(0, len(designs)))]
            take = rng.integers(0, len(common), size=len(common))
            draws[i] = _ols_coefficients(design[take], values[take])
        for position, term in enumerate(predictors, start=1):
            valid = draws[np.isfinite(draws[:, position]), position]
            # A row is emitted even when nothing could be fitted -- a predictor
            # that is constant across patients makes the design rank-deficient,
            # and ``valid_replicates = 0`` says so out loud. Dropping the row
            # instead would leave the term missing from the CSV with no reason.
            rows.append({"analysis": label, "outcome": outcome, "term": term,
                         "n_patients": len(common), "training_seeds": len(seeds),
                         "coefficient": float(np.nanmean(observed[:, position]))
                         if np.isfinite(observed[:, position]).any() else np.nan,
                         "coefficient_ci95_low": float(np.quantile(valid, .025))
                         if valid.size else np.nan,
                         "coefficient_ci95_high": float(np.quantile(valid, .975))
                         if valid.size else np.nan,
                         "valid_replicates": int(valid.size), "replicates": replicates})
    return pd.DataFrame(rows)


CONFIGURED_MODES = {"regular", "transfer"}


def main():
    ap = argparse.ArgumentParser(description="Shift-vs-MAE analysis.")
    ap.add_argument("--configured-metrics", action="append", default=[],
                    help="Path to a configured seed's metrics.json; repeat once per horizon")
    ap.add_argument("--configured-aggregate", action="append", default=[],
                    help="Path to a parent aggregate_metrics.json; repeat once per horizon")
    ap.add_argument("--permutation-count", type=int, default=10000,
                    help="Whole-patient permutations for shift/irregularity tests")
    ap.add_argument("--bootstrap-count", type=int, default=10000,
                    help="Nested patient x seed bootstrap replicates")
    ap.add_argument("--resampling-seed", type=int, default=42,
                    help="Deterministic seed for permutation/bootstrap resampling")
    ap.add_argument("--allow-incomplete-seeds", action="store_true",
                    help="Proceed when regular/transfer completed different seeds: pair each "
                         "patient on the shared seeds only and warn about mismatched cohorts, "
                         "instead of refusing the run")
    ap.add_argument("--compare-transfer-benefit", action="store_true",
                    help="With configured aggregates, compare screens for smallest regular-minus-transfer MAE benefit")
    ap.add_argument("--output-dir", default="new_analysis")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--dataset-features", default="results/analysis/dataset/dataset_signal_features.csv",
                    help="Dataset-only signal table produced by RUN/run_dataset_analysis.sh signal")
    ap.add_argument("--model", default="GRU", type=str.upper,
                    choices=["RNN", "LSTM", "GRU", "TRANSFORMER"])
    ap.add_argument("--mode", default="transfer", choices=sorted(CONFIGURED_MODES),
                    help="the mode recorded by configured runs")
    args = ap.parse_args()

    out = Path(args.output_dir)

    print("Loading MAE by horizon...")
    try:
        by_mode = seed_sets = None
        seed_mae = None
        if args.configured_aggregate and args.configured_metrics:
            raise ValueError("Select configured aggregates or seed metrics, not both")
        if not (args.configured_aggregate or args.configured_metrics):
            raise ValueError(
                "Pass --configured-aggregate (one parent aggregate_metrics.json per "
                "horizon) or --configured-metrics (one seed metrics.json per horizon)"
            )
        if args.compare_transfer_benefit and not args.configured_aggregate:
            raise ValueError("--compare-transfer-benefit requires --configured-aggregate")
        if args.configured_aggregate:
            by_mode, seed_sets = load_aggregate_mae_by_horizon(
                args.configured_aggregate, args.model,
                require_transfer_pairs=args.compare_transfer_benefit,
                require_equal_transfer_seeds=not args.allow_incomplete_seeds,
            )
            sampling_rate = configured_sampling_rate(args.configured_aggregate)
            mae_by_horizon = by_mode.get(args.mode, {})
            if not mae_by_horizon:
                raise ValueError(f"No aggregate MAE for mode {args.mode}")
            seed_mae = load_configured_seed_mae_by_horizon(
                args.configured_aggregate, args.model, args.mode
            )
            validate_seed_cohorts(seed_mae, strict=not args.allow_incomplete_seeds)
        else:
            mae_by_horizon = load_configured_mae_by_horizon(
                args.configured_metrics, args.model, args.mode,
                require_configured_cohort=True,
            )
            sampling_rate = configured_sampling_rate(args.configured_metrics)
    except (AmbiguousExperimentsError, ValueError) as e:
        print(f"✗ {e}")
        return 2
    if not mae_by_horizon:
        print("✗ No MAE data found.")
        return 1
    try:
        paths = args.configured_aggregate or args.configured_metrics
        settings = configured_signal_signature(paths)
        if settings["sampling_rate_minutes"] != sampling_rate:
            raise ValueError("configured sampling rate differs from dataset settings")
        feats = read_signal_table(args.dataset_features, settings, args.data_root)
        observed_ids = {pid for horizon in mae_by_horizon.values() for pid in horizon}
        missing_ids = observed_ids - set(feats)
        if missing_ids:
            raise ValueError(f"Dataset signal table lacks MAE patients {sorted(missing_ids)}")
        feats = {pid: feats[pid] for pid in sorted(observed_ids)}
    except (OSError, KeyError, ValueError) as error:
        print(f"✗ {error}")
        return 2

    wide, long, horizons = build_tables(feats, mae_by_horizon)
    if args.compare_transfer_benefit:
        # Loaded before the pairing, not after: where the two modes completed
        # different seeds, the per-seed tables are what makes a paired
        # difference reconstructible at all.
        regular_seed_mae = load_configured_seed_mae_by_horizon(
            args.configured_aggregate, args.model, "regular"
        )
        transfer_seed_mae = load_configured_seed_mae_by_horizon(
            args.configured_aggregate, args.model, "transfer"
        )
        try:
            benefit, benefit_long, benefit_horizons = build_transfer_benefit_long(
                feats, by_mode, seed_sets,
                strict_seeds=not args.allow_incomplete_seeds,
                seed_mae_by_mode={"regular": regular_seed_mae,
                                  "transfer": transfer_seed_mae},
            )
        except ValueError as e:
            print(f"✗ {e}")
            return 2
        benefit_screens = compare_difficulty_screens(
            benefit_long, benefit_horizons, target="least_transfer_benefit"
        )
        transfer_effect = paired_transfer_effect_bootstrap(
            regular_seed_mae, transfer_seed_mae, benefit_horizons,
            replicates=args.bootstrap_count, random_seed=args.resampling_seed,
        )

    # Persist tables
    out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out / "patient_feature_table.csv", index=False)
    long.to_csv(out / "shift_mae_long.csv", index=False)
    patient_difficulty_table(long, horizons).to_csv(out / "patient_difficulty.csv", index=False)
    corr = run_correlations(long, horizons)
    corr.to_csv(out / "shift_mae_correlations.csv", index=False)
    permutation = run_permutation_tests(
        long, horizons, permutations=args.permutation_count,
        random_seed=args.resampling_seed,
    )
    permutation.to_csv(out / "shift_mae_permutation_tests.csv", index=False)
    if seed_mae is not None:
        seed_tables = {}
        for (seed, horizon), values in sorted(seed_mae.items()):
            seed_tables.setdefault(seed, {})[horizon] = values
        seed_long = {}
        for seed, by_horizon in seed_tables.items():
            _, seed_frame, _ = build_tables(feats, by_horizon)
            seed_long[seed] = seed_frame
        seed_horizons = sorted({horizon for _, horizon in seed_mae})
        seed_sensitivity(seed_long, seed_horizons).to_csv(
            out / "shift_mae_seed_sensitivity.csv", index=False
        )
        regression_sensitivity(seed_long, seed_horizons).to_csv(
            out / "shift_mae_regression_seed_sensitivity.csv", index=False
        )
        screen_sensitivity(seed_long, seed_horizons).to_csv(
            out / "difficulty_screen_seed_sensitivity.csv", index=False
        )
        if args.compare_transfer_benefit:
            transfer_seed_long = {}
            for seed in sorted({key[0] for key in regular_seed_mae} &
                               {key[0] for key in transfer_seed_mae}):
                by_horizon = {}
                for horizon in seed_horizons:
                    regular_values = regular_seed_mae.get((seed, horizon), {})
                    transfer_values = transfer_seed_mae.get((seed, horizon), {})
                    patients = sorted(set(regular_values) & set(transfer_values))
                    if not patients:
                        continue
                    by_horizon[horizon] = {
                        pid: transfer_values[pid] - regular_values[pid]
                        for pid in patients
                    }
                if by_horizon:
                    _, transfer_seed_long[seed], _ = build_tables(feats, by_horizon)
            screen_sensitivity(
                transfer_seed_long, seed_horizons, target="least_transfer_benefit"
            ).to_csv(out / "transfer_benefit_screen_seed_sensitivity.csv", index=False)
        nested_patient_seed_bootstrap(
            seed_long, seed_horizons, replicates=args.bootstrap_count,
            random_seed=args.resampling_seed,
        ).to_csv(out / "shift_mae_nested_uncertainty.csv", index=False)
        nested_regression_uncertainty(
            seed_long, seed_horizons, replicates=args.bootstrap_count,
            random_seed=args.resampling_seed,
        ).to_csv(out / "shift_mae_regression_nested_uncertainty.csv", index=False)
    else:
        pd.DataFrame([{
            "note": "Seed sensitivity requires --configured-aggregate parent runs.",
            "training_seeds": 1,
        }]).to_csv(out / "shift_mae_seed_sensitivity.csv", index=False)
        pd.DataFrame([{
            "note": "Nested patient x seed uncertainty requires --configured-aggregate parent runs.",
            "training_seeds": 1,
        }]).to_csv(out / "shift_mae_nested_uncertainty.csv", index=False)
        pd.DataFrame([{
            "note": "Seed-aware regression sensitivity requires --configured-aggregate parent runs.",
            "training_seeds": 1,
        }]).to_csv(out / "shift_mae_regression_seed_sensitivity.csv", index=False)
        pd.DataFrame([{
            "note": "Nested patient x seed coefficient intervals require --configured-aggregate parent runs.",
            "training_seeds": 1,
        }]).to_csv(out / "shift_mae_regression_nested_uncertainty.csv", index=False)
        pd.DataFrame([{
            "note": "Seed-aware screen sensitivity requires --configured-aggregate parent runs.",
            "training_seeds": 1,
        }]).to_csv(out / "difficulty_screen_seed_sensitivity.csv", index=False)
    screens = compare_difficulty_screens(long, horizons)
    screens.to_csv(out / "difficulty_screen_comparison.csv", index=False)
    rank_difficulty_screens(long, horizons).to_csv(out / "difficulty_screen_rankings.csv", index=False)
    if args.compare_transfer_benefit:
        benefit.to_csv(out / "patient_transfer_benefit.csv", index=False)
        transfer_effect.to_csv(out / "transfer_benefit_effect_bootstrap.csv", index=False)
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
    print(f"\n✓ All outputs written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
