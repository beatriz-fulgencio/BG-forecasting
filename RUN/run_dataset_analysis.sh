#!/usr/bin/env bash
#
# Run only the analyses whose output is a property of the dataset.
#
#   bash RUN/run_dataset_analysis.sh            # preflight, then all analyses
#   bash RUN/run_dataset_analysis.sh --check    # preflight only, run nothing
#   bash RUN/run_dataset_analysis.sh population # one analysis only
#   bash RUN/run_dataset_analysis.sh features
#   bash RUN/run_dataset_analysis.sh clinical
#   bash RUN/run_dataset_analysis.sh signal
#
# RUN/run_experiments_analysis.sh analyses how the models performed, and every one of
# its stages has to be repeated for each of the 24 model x horizon x mode cells.
# The analyses here describe the cohort instead:
#
#   population  per-patient train/test glucose history and descriptive stats,
#               read from raw/ohiot1dm                          ~1 min
#   features    per-patient level, variability, clinical-range and dynamics
#               features over forecastable held-out targets      seconds
#   clinical    per-patient train-to-test changes in clinical CGM profile
#               (mean, CV, TIR, TBR and TAR)                    ~1 min
#   signal      train/test distribution shift and signal irregularity
#               for each patient                                ~1 min
#
# These dataset-only analyses are run once rather than per cell,
# and the numbers do not change if the models are retrained.
#
# What this deliberately leaves out. These look dataset-shaped but are not:
#
#   the t-SNE window embedding           the embedding is data-only, but the
#       figure's left panel colours it by patient MAE, so it belongs
#       to a cell. RUN/run_experiments_analysis.sh tsne draws it.
#   the train/test t-SNE comparison      same reason.
#   the persistence baseline             invariant across models, but it is
#       computed by aligning every prediction CSV, so it needs the runs.
#       RUN/run_experiments_analysis.sh persistence.
#   patient subgroups, comparison report  group patients by model error.
#
# The reference run
# -----------------
# The raw-series analyses take no number from a run, but need one to point at.
# The raw data directory does not record which 12 patients the experiments used,
# which OhioT1DM releases, what sampling rate, or where the train/test boundary
# falls -- all four come from a completed run's config, and all four change what
# the cohort statistics describe. So one recorded run is read for its settings.
#
# It defaults to the figure cell and falls back to the first recorded run.
# Override if you want a different one; the output should not depend on the
# choice, beyond the settings above:
#
#   REF_MODEL=lstm REF_HORIZON=60 bash RUN/run_dataset_analysis.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=RUN/_common.sh
. RUN/_common.sh

# The cell read for cohort and temporal settings. Same defaults as the figure
# stages, so the two describe the same cohort unless deliberately pointed apart.
REF_MODEL="${REF_MODEL:-$FIGURE_MODEL}"
REF_HORIZON="${REF_HORIZON:-$FIGURE_HORIZON}"
REF_MODE="${REF_MODE:-$FIGURE_MODE}"
DATASET_DIR="${DATASET_DIR:-$ANALYSIS_DIR/dataset}"

STAGE="all"
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        all|population|features|clinical|signal) STAGE="$arg" ;;
        *) echo "usage: $0 [all|population|features|clinical|signal] [--check]" >&2; exit 2 ;;
    esac
done

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
fail=0
note() { printf '  %-6s %s\n' "$1" "$2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

echo "=== preflight ==="

if python -c "import benchmark.analysis, scipy, sklearn, matplotlib, pandas" 2>/dev/null; then
    note "ok" "python imports (benchmark, scipy, sklearn, matplotlib, pandas)"
else
    note "FAIL" "cannot import benchmark or its analysis dependencies"
    note "" "  try: pip install -r requirements.txt   (and run from the repo root)"
fi

# The raw-series analyses read the CGM files directly; without them they write
# nothing at all, so this is a hard failure rather than a warning.
if [ -d "$DATA_ROOT/raw/ohiot1dm" ]; then
    note "ok" "raw data at $DATA_ROOT/raw/ohiot1dm"
else
    note "FAIL" "no raw data at $DATA_ROOT/raw/ohiot1dm -- the population analysis cannot run"
fi

# Resolve the reference run: the requested cell, else the first recorded run.
REF_PARENT=""
if [ ! -s "$PARENTS" ]; then
    note "FAIL" "no run list at $PARENTS -- run RUN/run_training.sh first"
else
    REF_PARENT=$(parent_for "$REF_MODEL" "$REF_HORIZON")
    if [ -n "$REF_PARENT" ]; then
        ref_label="$REF_MODEL ${REF_HORIZON}min (requested)"
    else
        read -r fallback_model fallback_minutes REF_PARENT < "$PARENTS" || true
        ref_label="$fallback_model ${fallback_minutes}min (fallback; no $REF_MODEL ${REF_HORIZON}min run recorded)"
    fi

    if [ -z "${REF_PARENT:-}" ]; then
        note "FAIL" "no usable run recorded in $PARENTS"
    elif [ ! -d "$REF_PARENT" ]; then
        note "FAIL" "reference run is missing: $REF_PARENT"
    else
        note "ok" "reference run: $ref_label"
    fi
fi

# The feature table reads each patient's held-out ground truth out of the
# prediction exports, so a run saved without predictions yields an empty table.
# The population analysis does not care, which is why this is a warning.
if [ -n "${REF_PARENT:-}" ] && [ -d "${REF_PARENT:-}" ]; then
    ref_seed=""
    for seed in $SEEDS; do
        [ -f "$REF_PARENT/$REF_MODE/seed_$seed/metrics.json" ] || continue
        ref_seed="$seed"
        break
    done
    if [ -z "$ref_seed" ]; then
        note "FAIL" "no completed $REF_MODE seed under $REF_PARENT (no metrics.json)"
    else
        ref_run="$REF_PARENT/$REF_MODE/seed_$ref_seed"
        n_preds=$(find "$ref_run" -name '*_predictions.csv' | wc -l | tr -d ' ')
        if [ "$n_preds" -eq 0 ]; then
            note "warn" "no prediction CSVs under $ref_run; the feature table will be empty"
            note "" "  the run needs output.save_predictions: true"
        else
            note "ok" "$n_preds prediction file(s) in $ref_run"
        fi
    fi
fi

echo
echo "=== plan ==="
printf '  analysis        : %s\n' "$STAGE"
printf '  reference run   : %s\n' "${REF_PARENT:-<unresolved>}"
printf '  mode            : %s\n' "$REF_MODE"
printf '  data root       : %s\n' "$DATA_ROOT"
printf '  output          : %s\n' "$DATASET_DIR"
printf '  NOTE            : the reference run fixes cohort, releases, sampling\n'
printf '                    rate and test split; it contributes no numbers.\n'

if [ "$fail" -ne 0 ]; then
    echo
    echo "Preflight failed; not starting. Fix the FAIL lines above." >&2
    exit 1
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo
    echo "Preflight only (--check). Nothing was run."
    exit 0
fi

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
mkdir -p "$DATASET_DIR"
LOG="$LOG_DIR/dataset_analysis_$(date '+%Y%m%d_%H%M%S').log"

skip_flags=""
case "$STAGE" in
    population) skip_flags="--skip-features --skip-clinical-shift --skip-signal" ;;
    features)   skip_flags="--skip-population --skip-clinical-shift --skip-signal" ;;
    clinical)   skip_flags="--skip-population --skip-features --skip-signal" ;;
    signal)     skip_flags="--skip-population --skip-features --skip-clinical-shift" ;;
esac

say "dataset-only analyses ($STAGE)"
echo "Logging to $LOG"

start=$(date +%s)
set +e
# shellcheck disable=SC2086
python -u RUN/dataset/run_dataset_analysis.py \
    --experiment-dir "$REF_PARENT" \
    --mode "$REF_MODE" \
    --data-root "$DATA_ROOT" \
    --check-invariance \
    $skip_flags \
    --output-dir "$DATASET_DIR" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
elapsed=$(( $(date +%s) - start ))

say "outputs"
printf '  glucose history : %s\n' "$DATASET_DIR/patient_analysis"
printf '  patient features: %s\n' "$DATASET_DIR/patient_glucose_features.csv"
printf '  clinical shift  : %s\n' "$DATASET_DIR/clinical_glucose_shift.csv"
printf '  signal features : %s\n' "$DATASET_DIR/dataset_signal_features.csv"
printf '  invariance check: %s\n' "$DATASET_DIR/ground_truth_invariance.csv"
printf '  summary         : %s\n' "$DATASET_DIR/summary.json"

echo
printf '=== finished in %dm %ds (exit %d) ===\n' \
    $((elapsed / 60)) $((elapsed % 60)) "$status"
echo "Log: $LOG"
exit "$status"
