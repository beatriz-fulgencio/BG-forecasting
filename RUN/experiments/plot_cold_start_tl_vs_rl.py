#!/usr/bin/env python3
"""Transfer versus regular learning across target-history budgets.

Reads the summary written by RUN/experiments/run_cold_start.py and draws the two
regimes against each other at every history budget (1/3/7 days, full), with the
same joint patient-and-seed resampling used everywhere else in the paper: whole
patients and whole seeds are drawn with replacement, and the same draws are
applied to both regimes so the benefit is a paired quantity.

    python RUN/experiments/plot_cold_start_tl_vs_rl.py --run-dir <cold-start run>

The run directory is the one holding summary/patient_seed_metrics.csv, under
either notebook's output root (cold_start/ or cold_start_followup/). Outputs land
next to that summary. Only numpy and matplotlib are needed.
"""
import argparse
import csv
import json
import pathlib

import numpy as np

REGIMES = ("regular", "transfer")
LABELS = {"regular": "Regular learning (RL)", "transfer": "Transfer learning (TL)"}
COLORS = {"regular": "#c0392b", "transfer": "#1f6fb4"}


def budget_key(budget):
    """Numeric budgets ascending, 'full' last."""
    return (1, 0.) if budget == "full" else (0, float(budget))


def read_summary(path, metric):
    """-> budgets, patients, seeds, {budget: {regime: (n_patients, n_seeds)}}, persistence."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} is empty")
    for regime in REGIMES:
        if f"{regime}_{metric}" not in rows[0]:
            raise SystemExit(f"{path} has no {regime}_{metric} column; "
                             f"available: {', '.join(sorted(rows[0]))}")

    budgets = sorted({r["budget_days"] for r in rows}, key=budget_key)
    patients = sorted({int(r["patient_id"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})

    index = {(r["budget_days"], int(r["patient_id"]), int(r["seed"])): r for r in rows}
    expected = len(budgets) * len(patients) * len(seeds)
    if len(index) != expected or len(rows) != expected:
        raise SystemExit(
            f"Incomplete or duplicated grid in {path}: {len(rows)} rows for "
            f"{len(budgets)} budgets x {len(patients)} patients x {len(seeds)} seeds. "
            "A figure over an unbalanced grid would compare different cohorts per budget.")

    data, persistence = {}, {}
    for budget in budgets:
        data[budget] = {
            regime: np.array([[float(index[(budget, p, s)][f"{regime}_{metric}"])
                               for s in seeds] for p in patients])
            for regime in REGIMES
        }
        persistence[budget] = np.array(
            [[float(index[(budget, p, s)][f"regular_persistence_{metric}"])
              for s in seeds] for p in patients])

    # The test window set is fixed across budgets by construction; if persistence
    # moves, the runs being plotted are not the comparison this figure claims.
    reference = persistence[budgets[0]]
    for budget in budgets[1:]:
        if not np.allclose(persistence[budget], reference):
            raise SystemExit("Persistence differs across budgets: the test set is not fixed.")
    return budgets, patients, seeds, data, reference


def draws(n_patients, n_seeds, replicates, seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, n_patients, size=(replicates, n_patients)),
            rng.integers(0, n_seeds, size=(replicates, n_seeds)))


def resample(matrix, patient_idx, seed_idx):
    return matrix[patient_idx[:, :, None], seed_idx[:, None, :]].mean(axis=(1, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                        help="cold-start run directory (contains summary/patient_seed_metrics.csv)")
    parser.add_argument("--metric", default="mae", choices=("mae", "rmse"))
    parser.add_argument("--bootstrap-count", type=int, default=20000)
    parser.add_argument("--resampling-seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None,
                        help="default: <run-dir>/summary")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--pdf", action="store_true", help="also write a vector copy")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    summary = run_dir / "summary" / "patient_seed_metrics.csv"
    if not summary.is_file():
        raise SystemExit(f"{summary} not found. Point --run-dir at a finished cold-start run.")
    out = pathlib.Path(args.output_dir) if args.output_dir else run_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)

    budgets, patients, seeds, data, persistence = read_summary(summary, args.metric)
    unit = "mg/dL"

    smoke = run_dir.name.startswith("SMOKE_")
    protocol = run_dir / "protocol.json"
    if protocol.is_file():
        smoke = smoke or bool(json.loads(protocol.read_text())["config"].get("smoke"))

    patient_idx, seed_idx = draws(len(patients), len(seeds),
                                  args.bootstrap_count, args.resampling_seed)

    # The full-history control, resampled with the same draws, so "does transfer
    # help more when there is less data" is a paired difference rather than two
    # independent estimates subtracted.
    control = ("full" if "full" in budgets else budgets[-1])
    control_benefit = resample(data[control]["regular"] - data[control]["transfer"],
                               patient_idx, seed_idx)

    rows = []
    for budget in budgets:
        rl, tl = data[budget]["regular"], data[budget]["transfer"]
        drawn = {regime: resample(data[budget][regime], patient_idx, seed_idx) for regime in REGIMES}
        benefit = resample(rl - tl, patient_idx, seed_idx)
        relative = 100 * benefit / drawn["regular"]
        rows.append(dict(
            budget_days=budget, n_patients=len(patients), n_seeds=len(seeds),
            regular=rl.mean(),
            regular_ci_low=np.quantile(drawn["regular"], .025),
            regular_ci_high=np.quantile(drawn["regular"], .975),
            transfer=tl.mean(),
            transfer_ci_low=np.quantile(drawn["transfer"], .025),
            transfer_ci_high=np.quantile(drawn["transfer"], .975),
            persistence=persistence.mean(),
            benefit=(rl - tl).mean(),
            benefit_ci_low=np.quantile(benefit, .025),
            benefit_ci_high=np.quantile(benefit, .975),
            benefit_pct=100 * (rl - tl).mean() / rl.mean(),
            benefit_pct_ci_low=np.quantile(relative, .025),
            benefit_pct_ci_high=np.quantile(relative, .975),
            control_budget=control,
            additional_vs_control=(benefit - control_benefit).mean(),
            additional_ci_low=np.quantile(benefit - control_benefit, .025),
            additional_ci_high=np.quantile(benefit - control_benefit, .975)))

    table_path = out / f"cold_start_tl_vs_rl_{args.metric}.csv"
    with open(table_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(budgets))
    ticks = ["full history" if b == "full" else f"{b} d" for b in budgets]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    # (a) both regimes on the same axis: the absolute errors the benefit is a
    # difference of, so a large relative gain on a bad model is visible as such.
    ax = axes[0]
    for regime in REGIMES:
        mean = np.array([r[regime] for r in rows])
        low = np.array([r[f"{regime}_ci_low"] for r in rows])
        high = np.array([r[f"{regime}_ci_high"] for r in rows])
        ax.plot(x, mean, marker="o", color=COLORS[regime], label=LABELS[regime], zorder=3)
        ax.fill_between(x, low, high, color=COLORS[regime], alpha=.18, lw=0, zorder=2)
    ax.axhline(persistence.mean(), color="gray", ls="--", lw=1,
               label=f"Persistence ({persistence.mean():.1f})")
    ax.set(xticks=x, xlabel="Target training history", ylabel=f"{args.metric.upper()} ({unit})",
           title="(a) Error by history budget")
    ax.set_xticklabels(ticks)
    ax.legend(fontsize=7, frameon=False)

    # (b) the paired difference, with the sign convention stated on the axis.
    ax = axes[1]
    benefit = np.array([r["benefit"] for r in rows])
    low = np.array([r["benefit_ci_low"] for r in rows])
    high = np.array([r["benefit_ci_high"] for r in rows])
    ax.plot(x, benefit, marker="o", color="#2c7a4b", zorder=3)
    ax.vlines(x, low, high, color="#2c7a4b", zorder=2)
    ax.axhline(0, color="gray", lw=.8)
    ax.set(xticks=x, xlabel="Target training history",
           ylabel=f"RL $-$ TL ({unit})", title="(b) Transfer benefit")
    ax.set_xticklabels(ticks)

    # (c) per-patient spread: the cohort mean in (b) can hide a benefit carried
    # by two patients, which is exactly what a screening claim would rest on.
    ax = axes[2]
    per_patient = np.array([
        100 * (data[b]["regular"].mean(axis=1) - data[b]["transfer"].mean(axis=1))
        / data[b]["regular"].mean(axis=1) for b in budgets]).T
    for series in per_patient:
        ax.plot(x, series, color="gray", lw=.7, alpha=.55, zorder=2)
    ax.plot(x, per_patient.mean(axis=0), marker="o", color="#2c7a4b", lw=2,
            label="cohort mean", zorder=3)
    ax.axhline(0, color="gray", lw=.8)
    ax.set(xticks=x, xlabel="Target training history", ylabel="Benefit (% of RL error)",
           title=f"(c) Per-patient benefit (n={len(patients)})")
    ax.set_xticklabels(ticks)
    ax.legend(fontsize=7, frameon=False)

    if smoke:
        fig.text(.5, .5, "SMOKE RUN - NOT EVIDENCE", fontsize=28, color="red",
                 alpha=.28, ha="center", va="center", rotation=18, zorder=10)

    figure_path = out / f"cold_start_tl_vs_rl_{args.metric}.png"
    fig.savefig(figure_path, dpi=args.dpi)
    if args.pdf:
        fig.savefig(figure_path.with_suffix(".pdf"))
    plt.close(fig)

    print(f"{len(patients)} patients x {len(seeds)} seeds x {len(budgets)} budgets"
          + ("   [SMOKE RUN]" if smoke else ""))
    header = f"{'budget':>12} {'RL':>8} {'TL':>8} {'benefit':>9} {'95% CI':>18} {'%':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['budget_days']:>12} {row['regular']:8.3f} {row['transfer']:8.3f} "
              f"{row['benefit']:9.3f} [{row['benefit_ci_low']:7.3f},{row['benefit_ci_high']:7.3f}] "
              f"{row['benefit_pct']:6.2f}%")
    print(f"\nFigure: {figure_path}\nTable : {table_path}")


if __name__ == "__main__":
    main()
