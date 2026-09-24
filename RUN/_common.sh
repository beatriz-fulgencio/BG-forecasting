# Shared configuration and helpers for the RUN/ drivers.
#
# Sourced, never executed. RUN/run_training.sh trains models and records
# what it produced; RUN/run_experiments_analysis.sh reads that record and analyses it.
# Everything both need lives here so the two cannot drift apart.
#
# Override anything from the environment:
#   MODELS="gru" HORIZONS="15 30" bash RUN/run_training.sh

# Both drivers iterate unquoted $MODELS/$SEEDS/$HORIZONS, which relies on
# word-splitting. Pin IFS so an exported non-default value in the caller's
# environment cannot silently collapse those lists into one item.
IFS=$' \t\n'

MODELS="${MODELS:-gru lstm rnn}"
HORIZONS="${HORIZONS:-15 30 45 60}"
SEEDS="${SEEDS:-41 42 43}"
CONFIG_PREFIX="${CONFIG_PREFIX:-configs/full}"
LOG_DIR="${LOG_DIR:-results/full_run_logs}"
ANALYSIS_DIR="${ANALYSIS_DIR:-results/analysis}"
DATA_ROOT="${DATA_ROOT:-data}"
REPLICATES="${REPLICATES:-20000}"
PERMUTATIONS="${PERMUTATIONS:-10000}"
# Rapid-change stratification: the absolute mg/dL step between the two readings
# that bracket the forecast target, one sampling interval apart.
#
# 15 mg/dL is 3 mg/dL/min at the 5-minute sampling every config here uses, which
# is what a CGM reports as "rapidly rising"/"rapidly falling" -- the double trend
# arrow on the Dexcom scheme the Endocrine Society's dosing guidance is built
# around (Klonoff & Kerr, J Diabetes Sci Technol 2017;11(6):1063-1069; Aleppo et
# al., J Endocr Soc 2018;2(12):1320-1337). It replaces an earlier 20 mg/dL, which
# is 4 mg/dL/min -- not the clinical boundary but the "extreme and rare" rate of
# Rate Error-Grid Analysis, selecting a third as many points for the same effect.
# See benchmark/analysis/experiments/rapid_change.py for the full rationale.
RAPID_THRESHOLD="${RAPID_THRESHOLD:-15}"

# Per-patient error-localization and pooled-seed Clarke grid figures, drawn
# for every patient of every recorded model x horizon cell.
#
# "all" means the cell's own cohort, read from the run rather than listed here,
# so a different cohort needs no edit. The runner loops the patients internally
# and reads each cell once, which is why the whole cohort costs about what four
# separate invocations used to. Narrow it to specific patients when iterating:
#   FIGURE_PATIENTS="540 584" bash RUN/run_experiments_analysis.sh figures
#
# These stay single-patient illustrations of where error sits, not cohort
# estimates; the cohort-level uncertainty for the same quantities is what the
# error-range and rapid-change bootstraps report.
FIGURE_PATIENTS="${FIGURE_PATIENTS:-all}"
FIGURE_MODE="${FIGURE_MODE:-transfer}"

# The cell the cohort-level figures are drawn from (currently the t-SNE).
FIGURE_MODEL="${FIGURE_MODEL:-gru}"
FIGURE_HORIZON="${FIGURE_HORIZON:-30}"
# The Clarke grid figure is a single run, not a seed pool.
FIGURE_SEED="${FIGURE_SEED:-41}"

# Cross-horizon inference. 'non-parametric' precommits to Wilcoxon rather than
# letting a normality screen pick the test after seeing the data; the shared
# tester calls the screening path 'auto' and documents it as legacy behaviour.
TEST_FAMILY="${TEST_FAMILY:-non-parametric}"
CORRECTION="${CORRECTION:-holm}"

PARENTS="$LOG_DIR/parents.txt"

export PYTHONPATH="${PYTHONPATH:-.}"
# Unbuffered: without this, piping into tee makes Python switch to 8KB block
# buffering and the log lags minutes behind the run, which looks like a hang.
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"
touch "$PARENTS"

say() { printf '\n=== %s ===\n' "$1"; }

FAILURES=""
note_failure() {   # record and keep going; one bad cell must not kill the rest
    FAILURES="$FAILURES
  $1"
    echo "  !! $1" >&2
}

report_failures() {
    if [ -n "$FAILURES" ]; then
        say "completed with failures"
        printf '%s\n' "$FAILURES"
        return 1
    fi
    return 0
}

parent_for() {   # model horizon -> recorded directory, or empty
    awk -v m="$1" -v h="$2" '$1 == m && $2 == h { print $3; exit }' "$PARENTS"
}

recorded_models() {
    awk 'NF >= 3 && !seen[$1]++ { print $1 }' "$PARENTS"
}

require_parents() {
    if [ ! -s "$PARENTS" ]; then
        echo "No runs recorded in $PARENTS. Run RUN/run_training.sh first." >&2
        exit 1
    fi
}

# Every recorded parent directory, space separated, for the analyses that take
# a whole cohort of runs at once rather than one cell at a time.
all_parents() {
    awk 'NF >= 3 { printf " %s", $3 }' "$PARENTS"
}

# One model's recorded parent directories, space separated.
parents_for_model() {   # model
    awk -v m="$1" 'NF >= 3 && $1 == m { printf " %s", $3 }' "$PARENTS"
}
