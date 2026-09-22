#!/usr/bin/env bash
#
# Run every analysis stage over runs that have already been trained.
#
#   bash RUN/run_experiments_analysis.sh              # preflight, then every stage
#   bash RUN/run_experiments_analysis.sh --check      # preflight only, run nothing
#   bash RUN/run_experiments_analysis.sh --quick      # every stage, far fewer resamples
#   bash RUN/run_experiments_analysis.sh shift        # one stage only
#   bash RUN/run_experiments_analysis.sh transfer --quick
#
# Training is RUN/run_training.sh. This script never trains; it reads the
# run list that one writes ($LOG_DIR/parents.txt) and analyses those runs.
#
# Stages, in order, all reading results/experiments/. Cross-seed aggregates are
# not a stage: RUN/run_training.sh already writes aggregate_metrics.json and
# .csv per run, and `bg-forecast analyze` only reshapes those, so recomputing them
# here would restate training's own output. Every stage below reads them directly.
#
#   zone-d       Clarke zone-D nested patient x seed bootstrap    ~1 min, all cells
#   error-range  error by glycemic band, nested bootstrap         ~1 min, all cells
#   rapid-change calm vs rapid change, nested bootstrap           ~2 min per cell
#   figures      per-patient error-localization figures            ~1 min per cell
#   glycemic     Clarke grids per patient, paper + pooled forms     ~4 min, all cells
#   tsne         t-SNE of test-period glucose windows              ~5 min, one cell
#   persistence  forecast-origin persistence baseline             ~4 min, reads every prediction CSV
#   shift        shift-difficulty permutation + bootstrap tests   ~5 min per model
#   transfer     confirmatory paired TL-RL inference, per model   seconds
#   stability    patient-difficulty stability across conditions   seconds
#   horizons     cross-horizon significance per model and mode    seconds
#   compare      cross-model comparison                           seconds
#   summary      where everything landed                          instant
#
# Timings measured on 12 runs x 12 patients x 3 seeds at the default resample
# counts. Scale roughly with patients x seeds x replicates.
#
# The preflight exists because the expensive stages take minutes to hours, and
# every way they can fail (missing predictions, missing raw data, a stale run
# list, a broken regular/transfer seed pairing) is knowable in seconds first.
#
# Override anything from the environment, e.g.
#   REPLICATES=2000 PERMUTATIONS=1000 bash RUN/run_experiments_analysis.sh
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=RUN/_common.sh
. RUN/_common.sh

STAGE="all"
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --quick) REPLICATES="${QUICK_REPLICATES:-2000}"; PERMUTATIONS="${QUICK_PERMUTATIONS:-1000}"; QUICK=1 ;;
        all|zone-d|error-range|rapid-change|figures|glycemic|tsne|persistence|shift|transfer|stability|horizons|compare|summary)
            STAGE="$arg" ;;
        *) echo "usage: $0 [stage] [--check] [--quick]" >&2
           echo "stages: all zone-d error-range rapid-change figures glycemic tsne persistence shift transfer stability horizons compare summary" >&2
           exit 2 ;;
    esac
done
QUICK="${QUICK:-0}"

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
fail=0
note() { printf '  %-6s %s\n' "$1" "$2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

echo "=== preflight ==="

# 1. Python can import the package, and the analysis deps are present.
if python -c "import benchmark.analysis, scipy, sklearn, matplotlib" 2>/dev/null; then
    note "ok" "python imports (benchmark, scipy, sklearn, matplotlib)"
else
    note "FAIL" "cannot import benchmark or its analysis dependencies"
    note "" "  try: pip install -r requirements.txt   (and run from the repo root)"
fi

# 2. The run list the stages iterate over.
if [ ! -s "$PARENTS" ]; then
    note "FAIL" "no run list at $PARENTS -- run RUN/run_training.sh first"
else
    recorded=$(grep -cve '^[[:space:]]*$' "$PARENTS" || true)
    missing=0
    while read -r _model _minutes parent; do
        [ -n "${parent:-}" ] || continue
        [ -d "$parent" ] || { note "FAIL" "recorded run is missing: $parent"; missing=1; }
    done < "$PARENTS"
    [ "$missing" -eq 0 ] && note "ok" "$recorded recorded runs, all present"
fi

# 3. Raw data: the shift analysis reads the CGM series, and the stages that need prediction
#    context (persistence, rapid-change) reconstruct it from the same raw files.
if [ -d "$DATA_ROOT/raw/ohiot1dm" ]; then
    note "ok" "raw data at $DATA_ROOT/raw/ohiot1dm (needed by shift, persistence, rapid-change)"
else
    note "FAIL" "no raw data at $DATA_ROOT/raw/ohiot1dm -- shift, persistence and rapid-change will fail"
fi

# 4. Seed completeness, prediction files, and regular/transfer seed parity.
#
#    A seed killed mid-run leaves a directory behind holding only the patients it
#    reached, and no metrics.json (which the runner writes after evaluating all of
#    them). Such a directory looks finished to `[ -d ]` but fails the per-cell
#    bootstraps on the first patient it never reached -- after the bootstrap has
#    already spent its time.
#
#    Parity matters separately: the paired analyses (transfer, and the shift stage's
#    transfer-benefit comparison) need regular and transfer to have completed the
#    *same* seeds. A single failed seed breaks the pairing, and they refuse the
#    run rather than compare a 3-seed mean against a 2-seed one.
if [ -s "$PARENTS" ]; then
    complete=0; partial=0; short_preds=0; unpaired=0
    while read -r model minutes parent; do
        [ -d "${parent:-}" ] || continue
        regular_seeds=""; transfer_seeds=""
        for mode in regular transfer; do
            for seed in $SEEDS; do
                run_dir="$parent/$mode/seed_$seed"
                [ -d "$run_dir" ] || continue
                if [ ! -f "$run_dir/metrics.json" ]; then
                    note "warn" "$model ${minutes}min/$mode seed $seed is incomplete (no metrics.json); excluded"
                    partial=$((partial + 1))
                    continue
                fi
                complete=$((complete + 1))
                [ "$mode" = "regular" ] && regular_seeds="$regular_seeds $seed" || transfer_seeds="$transfer_seeds $seed"
                n_patients=$(find "$run_dir" -maxdepth 1 -type d -name 'patient_*' | wc -l | tr -d ' ')
                n_preds=$(find "$run_dir" -name '*_predictions.csv' | wc -l | tr -d ' ')
                if [ "$n_patients" -gt 0 ] && [ "$n_preds" -lt "$n_patients" ]; then
                    note "warn" "$model ${minutes}min/$mode seed $seed has $n_preds predictions for $n_patients patients"
                    short_preds=$((short_preds + 1))
                fi
            done
        done
        if [ "$regular_seeds" != "$transfer_seeds" ]; then
            note "warn" "$model ${minutes}min regular seeds[$regular_seeds ] != transfer[$transfer_seeds ]; paired analyses will skip this cell"
            unpaired=$((unpaired + 1))
        fi
    done < "$PARENTS"

    if [ "$complete" -eq 0 ]; then
        note "FAIL" "no completed seed runs found -- nothing to analyse"
    else
        note "ok" "$complete completed seed runs"
    fi
    [ "$unpaired" -eq 0 ] && note "ok" "regular/transfer seed sets match in every recorded run"
    if [ "$short_preds" -gt 0 ]; then
        note "warn" "zone-D, error-by-range, rapid-change and persistence need a prediction CSV per patient; set output.save_predictions: true"
    fi
    [ "$partial" -gt 0 ] && note "warn" "$partial partial seed director(ies) will be skipped, not analysed"
fi

echo
echo "=== plan ==="
printf '  stage           : %s\n' "$STAGE"
printf '  analysis output : %s\n' "$ANALYSIS_DIR"
printf '  logs            : %s\n' "$LOG_DIR"
printf '  bootstrap       : %s replicates\n' "$REPLICATES"
printf '  rapid threshold : %s mg/dL across one sampling interval\n' "$RAPID_THRESHOLD"
printf '  permutations    : %s\n' "$PERMUTATIONS"
printf '  horizon tests   : %s, %s correction\n' "$TEST_FAMILY" "$CORRECTION"
[ "$QUICK" = "1" ] && printf '  NOTE            : --quick resamples are for a smoke check, not for publication\n'

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
# Stages
# --------------------------------------------------------------------------
# Both per-cell bootstraps take the same input -- every completed seed of one
# mode/horizon cell, as repeated --run-dir flags -- so they discover it through
# one function rather than two copies that can drift apart.
#
# Completion is judged by metrics.json, which the runner writes only after a seed
# has evaluated every patient. The directory alone is not evidence.
#
# Sets CELL_RUN_DIRS, CELL_FOUND, CELL_MISSING, CELL_PARTIAL.
discover_cell_seeds() {   # parent mode
    parent_dir="$1"; cell_mode="$2"
    CELL_RUN_DIRS=""; CELL_FOUND=0; CELL_MISSING=""; CELL_PARTIAL=""
    for seed in $SEEDS; do
        seed_dir="$parent_dir/$cell_mode/seed_$seed"
        if [ -f "$seed_dir/metrics.json" ]; then
            CELL_RUN_DIRS="$CELL_RUN_DIRS --run-dir $seed_dir"
            CELL_FOUND=$((CELL_FOUND + 1))
        elif [ -d "$seed_dir" ]; then
            CELL_PARTIAL="$CELL_PARTIAL $seed"
        else
            CELL_MISSING="$CELL_MISSING $seed"
        fi
    done
}

report_cell_seeds() {   # label found
    [ -n "$CELL_MISSING" ] && \
        echo "  NOTE: seed(s)$CELL_MISSING absent; interval covers $2 seed(s)" >&2
    [ -n "$CELL_PARTIAL" ] && \
        echo "  NOTE: seed(s)$CELL_PARTIAL incomplete (no metrics.json); excluded" >&2
    return 0
}

# One per-cell bootstrap over every completed seed. $1 names the stage for the
# log, $2 the script, $3 the output subdirectory under <parent>/analysis/; any
# further arguments are passed through to the script unchanged.
run_cell_bootstrap() {   # label script output_prefix [extra flags...]
    label="$1"; script="$2"; output_prefix="$3"; shift 3
    extra="$*"
    require_parents
    while read -r model minutes parent; do
        for mode in regular transfer; do
            discover_cell_seeds "$parent" "$mode"
            if [ "$CELL_FOUND" -eq 0 ]; then
                note_failure "$label $model $minutes min/$mode: no seed directories"
                continue
            fi
            say "$model $minutes min / $mode: $label ($CELL_FOUND seed(s))"
            report_cell_seeds "$label" "$CELL_FOUND"
            # shellcheck disable=SC2086
            python -u "$script" $CELL_RUN_DIRS $extra \
                --bootstrap-replicates "$REPLICATES" \
                --output-dir "$parent/analysis/${output_prefix}_$mode" \
                || note_failure "$label $model $minutes min/$mode"
        done
    done < "$PARENTS"
}

stage_zone_d() {
    run_cell_bootstrap "zone-D nested bootstrap" RUN/experiments/run_zone_d_analysis.py zone_d
}

stage_error_range() {
    # Where the error lives and which way it points, by glycemic band. Same
    # inputs and same nested patient x seed bootstrap as zone-D; the difference
    # is that zone-D covers only the clinically dangerous predictions (0.3-3.7%
    # of them) while this covers all of them, signed.
    run_cell_bootstrap "error-by-range nested bootstrap" \
        RUN/experiments/run_error_by_range_analysis.py error_by_range
}

stage_rapid_change() {
    # Where the error lives when glucose is moving. Same per-cell inputs and
    # nested bootstrap as zone-D, but it reads each prediction's timestamped
    # context, so it costs roughly what persistence costs per cell rather than
    # what zone-D costs. RAPID_THRESHOLD is an absolute mg/dL step across one
    # sampling interval: at the 5-minute sampling every config uses, the 20
    # mg/dL default is >4 mg/dL/min and selects well under 1% of predictions.
    run_cell_bootstrap "rapid-change nested bootstrap" \
        RUN/experiments/run_rapid_change_analysis.py rapid_change \
        --threshold-mg-dl "$RAPID_THRESHOLD"
}

# Every glycemic-zone view of a cell collects in one folder, named for the cell
# rather than buried under the experiment directory, so the Clarke grids for all
# 12 models x horizons can be read side by side.
glycemic_cell_dir() {   # model minutes mode
    printf '%s/glycemic_analysis/%s_%smin_%s' "$ANALYSIS_DIR" "$1" "$2" "$3"
}

# FIGURE_PATIENTS -> the runner's patient selection flags. "all" defers to the
# cell's own recorded cohort; a list becomes one --patient per id.
figure_patient_flags() {
    if [ "$FIGURE_PATIENTS" = "all" ]; then
        printf '%s' "--all-patients"
    else
        for patient in $FIGURE_PATIENTS; do printf ' --patient %s' "$patient"; done
    fi
}

stage_figures() {
    require_parents
    # Error localization and the pooled-seed Clarke grid, for every patient of every recorded
    # model x horizon cell. Deliberately not a bootstrap stage: these are
    # single-patient illustrations of where error sits, and their cohort-level
    # uncertainty is what error-range and rapid-change above already report.
    # One mode only (FIGURE_MODE), because a figure comparing nothing does not
    # need its counterpart drawn beside it.
    #
    # One invocation per cell, not one per patient: the runner loops the cohort
    # internally, so each cell's seed directories are read once instead of once
    # per patient.
    patient_flags=$(figure_patient_flags)
    while read -r model minutes parent; do
        discover_cell_seeds "$parent" "$FIGURE_MODE"
        if [ "$CELL_FOUND" -eq 0 ]; then
            note_failure "figures $model $minutes min/$FIGURE_MODE: no seed directories"
            continue
        fi
        say "$model $minutes min / $FIGURE_MODE: error-localization figures ($FIGURE_PATIENTS patients)"
        log="$LOG_DIR/figures_${model}_${minutes}min.log"
        # shellcheck disable=SC2086
        python -u RUN/figures/run_error_localization_figures.py $CELL_RUN_DIRS \
            $patient_flags \
            --rapid-threshold "$RAPID_THRESHOLD" \
            --output-dir "$parent/analysis/figures_$FIGURE_MODE" \
            --clarke-output-dir "$(glycemic_cell_dir "$model" "$minutes" "$FIGURE_MODE")" \
            > "$log" 2>&1 \
            || note_failure "figures $model $minutes min/$FIGURE_MODE (see $log)"
    done < "$PARENTS"
}

stage_glycemic() {
    require_parents
    # The paper's form of the Clarke grid -- one seed, the worst-case
    # illustration -- drawn for every patient of every recorded cell, beside the
    # pooled-seed grid the `figures` stage writes to the same folder. The two
    # answer different questions (one run versus every completed seed) and are
    # read together, which is why they share a directory.
    #
    # PNG only: at 12 cells x 12 patients a paired PDF doubles the output for a
    # raster figure nobody typesets from.
    patient_flags=$(figure_patient_flags)
    while read -r model minutes parent; do
        run_dir="$parent/$FIGURE_MODE/seed_$FIGURE_SEED"
        if [ ! -f "$run_dir/metrics.json" ]; then
            note_failure "glycemic $model $minutes min/$FIGURE_MODE: $run_dir is missing or incomplete"
            continue
        fi
        out=$(glycemic_cell_dir "$model" "$minutes" "$FIGURE_MODE")
        say "$model $minutes min / $FIGURE_MODE seed $FIGURE_SEED: Clarke grids ($FIGURE_PATIENTS patients)"
        log="$LOG_DIR/glycemic_${model}_${minutes}min.log"
        # shellcheck disable=SC2086
        python -u RUN/figures/run_clarke_localized_figure.py \
            --run-dir "$run_dir" \
            $patient_flags \
            --model "$(printf '%s' "$model" | tr '[:lower:]' '[:upper:]')" \
            --rapid-threshold "$RAPID_THRESHOLD" \
            --output-dir "$out" \
            > "$log" 2>&1 \
            || note_failure "glycemic $model $minutes min/$FIGURE_MODE (see $log)"
    done < "$PARENTS"
}

stage_tsne() {
    require_parents
    # The t-SNE is cohort-level and model-independent in everything but the MAE
    # colouring, so it is drawn once for the FIGURE_MODEL/FIGURE_HORIZON cell
    # rather than per cell. The Clarke grid is drawn for every cell by the
    # `glycemic` stage; the cohort glucose overview is a dataset property and
    # belongs to RUN/run_dataset_analysis.sh.
    parent=$(parent_for "$FIGURE_MODEL" "$FIGURE_HORIZON")
    if [ -z "$parent" ]; then
        note_failure "t-SNE figure: no recorded $FIGURE_MODEL ${FIGURE_HORIZON}min run"
        return 0
    fi
    say "$FIGURE_MODEL $FIGURE_HORIZON min / $FIGURE_MODE: t-SNE"
    python -u RUN/figures/run_tsne_figure.py \
        --experiment-dir "$parent" \
        --model "$(printf '%s' "$FIGURE_MODEL" | tr '[:lower:]' '[:upper:]')" \
        --mode "$FIGURE_MODE" \
        --output-dir "$ANALYSIS_DIR/tsne" \
        > "$LOG_DIR/tsne.log" 2>&1 \
        || note_failure "t-SNE figure (see $LOG_DIR/tsne.log)"
}

stage_persistence() {
    require_parents
    # The forecast-origin baseline is a property of the data, not of a model, so
    # every completed seed of every run goes in one invocation: the analysis
    # checks the baseline really is invariant across all of them, which it can
    # only do by seeing all of them.
    say "forecast-origin persistence baseline"
    run_dirs=""
    while read -r _model _minutes parent; do
        for mode in regular transfer; do
            discover_cell_seeds "$parent" "$mode"
            run_dirs="$run_dirs $CELL_RUN_DIRS"
        done
    done < "$PARENTS"
    if [ -z "$run_dirs" ]; then
        note_failure "persistence: no completed seed directories"
        return 0
    fi
    # shellcheck disable=SC2086
    python -u RUN/experiments/run_persistence_analysis.py $run_dirs \
        --output-dir "$ANALYSIS_DIR/persistence" \
        > "$LOG_DIR/persistence.log" 2>&1 \
        || note_failure "persistence (see $LOG_DIR/persistence.log)"
}

stage_shift() {
    require_parents
    dataset_features="${DATASET_DIR:-$ANALYSIS_DIR/dataset}/dataset_signal_features.csv"
    if [ ! -f "$dataset_features" ] || [ ! -f "${dataset_features%.csv}.meta.json" ]; then
        note_failure "shift: missing dataset signal features; run RUN/run_dataset_analysis.sh signal first"
        return
    fi
    # One shift-difficulty analysis per model, over that model's horizons.
    for model in $(recorded_models); do
        aggregates=""
        while read -r m minutes parent; do
            [ "$m" = "$model" ] || continue
            aggregates="$aggregates --configured-aggregate $parent/aggregate_metrics.json"
        done < "$PARENTS"
        if [ -z "$aggregates" ]; then
            say "$model: no recorded runs, skipping shift-difficulty analysis"
            continue
        fi
        say "$model: shift-difficulty analysis"
        # The analysis script names models in upper case; MODELS is lower case
        # because it also builds config filenames. Translate rather than changing
        # either, since both spellings are load-bearing elsewhere.
        model_arg=$(printf '%s' "$model" | tr '[:lower:]' '[:upper:]')
        attempt_log="$LOG_DIR/phase1_${model}_with_transfer_benefit.log"
        set +e
        # shellcheck disable=SC2086
        python -u RUN/experiments/run_shift_analysis.py $aggregates \
            --mode transfer --compare-transfer-benefit \
            --model "$model_arg" --data-root "$DATA_ROOT" \
            --dataset-features "$dataset_features" \
            --permutation-count "$PERMUTATIONS" \
            --bootstrap-count "$REPLICATES" \
            --resampling-seed 42 \
            --output-dir "$ANALYSIS_DIR/phase1_$model" 2>&1 | tee "$attempt_log"
        status=${PIPESTATUS[0]}
        set -e
        # Exit 2 is ambiguous: argparse uses it for a usage error, and the script
        # itself returns it for a data problem such as the regular/transfer seed
        # sets differing -- which is precisely the retryable case. Distinguish on
        # the argparse signature instead, since a usage error fails before the
        # script prints anything of its own.
        if [ "$status" -ne 0 ] && grep -q "error: argument" "$attempt_log"; then
            note_failure "shift $model: bad arguments; see $attempt_log"
        elif [ "$status" -ne 0 ] && grep -q "Dataset signal table\|Missing dataset signal table\|dataset settings differ" "$attempt_log"; then
            note_failure "shift $model: incompatible dataset signal table; see $attempt_log"
        elif [ "$status" -ne 0 ]; then
            # The transfer-benefit comparison needs regular and transfer to share
            # a seed set; a failed training seed breaks that pairing on purpose.
            # The rest of the analysis does not depend on it, so keep that.
            echo "  NOTE: $model transfer-benefit comparison unavailable" \
                 "(regular/transfer seed sets may differ); retrying without it" >&2
            # shellcheck disable=SC2086
            python -u RUN/experiments/run_shift_analysis.py $aggregates \
                --mode transfer \
                --model "$model_arg" --data-root "$DATA_ROOT" \
                --dataset-features "$dataset_features" \
                --permutation-count "$PERMUTATIONS" \
                --bootstrap-count "$REPLICATES" \
                --resampling-seed 42 \
                --output-dir "$ANALYSIS_DIR/phase1_$model" \
                || note_failure "shift $model"
        fi
    done
}

stage_transfer() {
    require_parents
    # One invocation per model, so the Benjamini-Hochberg family is that model's
    # horizons rather than every model pooled. The analysis refuses a run whose
    # regular and transfer seed sets differ, which the preflight warns about
    # above; a refusal is recorded and the other models still run.
    for model in $(recorded_models); do
        dirs=""
        for parent in $(parents_for_model "$model"); do
            dirs="$dirs --experiment-dir $parent"
        done
        [ -n "$dirs" ] || continue
        say "$model: confirmatory transfer-vs-regular inference"
        # shellcheck disable=SC2086
        python -u RUN/experiments/run_transfer_inference.py $dirs \
            --bootstrap-replicates "$REPLICATES" \
            --resampling-seed 42 \
            --output-dir "$ANALYSIS_DIR/transfer_inference_$model" \
            || note_failure "transfer inference $model (regular/transfer seed sets may differ)"
    done
}

stage_stability() {
    require_parents
    # Deliberately one invocation over every run: Kendall's W is concordance
    # *between* conditions, so splitting by model would remove the comparison
    # the analysis exists to make.
    say "patient-difficulty stability across models, horizons and modes"
    dirs=""
    for parent in $(all_parents); do
        dirs="$dirs --experiment-dir $parent"
    done
    # shellcheck disable=SC2086
    python -u RUN/experiments/run_patient_stability_analysis.py $dirs \
        --output-dir "$ANALYSIS_DIR/patient_stability" \
        > "$LOG_DIR/patient_stability.log" 2>&1 \
        || note_failure "patient stability (see $LOG_DIR/patient_stability.log)"
}

stage_horizons() {
    require_parents
    # Cross-horizon significance per model. Modes are analysed separately: a
    # regular and a transfer run are different training regimes, and pooling
    # them would compare estimators rather than horizons.
    #
    # The test family is precommitted rather than chosen by a normality screen
    # after the data are in hand; see TEST_FAMILY in RUN/_common.sh.
    dirs="$(all_parents)"
    for mode in regular transfer; do
        say "cross-horizon significance ($mode, $TEST_FAMILY, $CORRECTION)"
        # shellcheck disable=SC2086
        python -u RUN/experiments/run_comparison_analysis.py --experiments $dirs \
            --mode "$mode" \
            --test-family "$TEST_FAMILY" \
            --correction "$CORRECTION" \
            --output-dir "$ANALYSIS_DIR/horizons_$mode" \
            || note_failure "horizon comparison ($mode)"
    done
}

stage_compare() {
    require_parents
    say "cross-model comparison"
    dirs="$(all_parents)"
    # shellcheck disable=SC2086
    python -u -m benchmark.cli compare --experiments $dirs \
        --output "$ANALYSIS_DIR/model_comparison.json" || \
        echo "NOTE: cross-model compare failed; per-model outputs are unaffected" >&2
}

stage_summary() {
    require_parents
    say "outputs"
    for model in $(recorded_models); do
        printf '  %-5s shift    : %s\n' "$model" "$ANALYSIS_DIR/phase1_$model"
        printf '  %-5s transfer : %s\n' "$model" "$ANALYSIS_DIR/transfer_inference_$model"
    done
    printf '  stability     : %s\n' "$ANALYSIS_DIR/patient_stability"
    printf '  t-SNE figure  : %s\n' "$ANALYSIS_DIR/tsne"
    printf '  persistence   : %s\n' "$ANALYSIS_DIR/persistence"
    printf '  horizons      : %s\n' "$ANALYSIS_DIR/horizons_{regular,transfer}"
    printf '  models        : %s\n' "$ANALYSIS_DIR/model_comparison.json"
    while read -r model minutes parent; do
        printf '  %-5s %2s min  : %s\n' "$model" "$minutes" "$parent"
        printf '                  zone-D    : %s\n' "$parent/analysis/zone_d_{regular,transfer}"
        printf '                  err/range : %s\n' "$parent/analysis/error_by_range_{regular,transfer}"
        printf '                  rapid-chg : %s\n' "$parent/analysis/rapid_change_{regular,transfer}"
        printf '                  figures   : %s\n' "$parent/analysis/figures_$FIGURE_MODE"
        printf '                  dashboards: %s\n' "$parent/<mode>/seed_<n>/patient_<id>/*.png"
    done < "$PARENTS"
}

echo
echo "Starting in 5s -- Ctrl-C to abort."
sleep 5

LOG="$LOG_DIR/full_analysis_$(date '+%Y%m%d_%H%M%S').log"
echo "Logging to $LOG"
echo

run_stages() {
    case "$STAGE" in
        all)         stage_zone_d; stage_error_range; stage_rapid_change; stage_figures
                     stage_glycemic; stage_tsne
                     stage_persistence; stage_shift; stage_transfer
                     stage_stability; stage_horizons; stage_compare
                     stage_summary ;;
        zone-d)       stage_zone_d ;;
        error-range)  stage_error_range ;;
        rapid-change) stage_rapid_change ;;
        figures)     stage_figures ;;
        glycemic)     stage_glycemic ;;
        tsne)         stage_tsne ;;
        persistence)  stage_persistence ;;
        shift)        stage_shift ;;
        transfer)     stage_transfer ;;
        stability)    stage_stability ;;
        horizons)     stage_horizons ;;
        compare)      stage_compare ;;
        summary)      stage_summary ;;
    esac
    report_failures
}

start=$(date +%s)
set +e
run_stages 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
elapsed=$(( $(date +%s) - start ))

echo
printf '=== finished in %dh %dm %ds (exit %d) ===\n' \
    $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60)) "$status"
echo "Log: $LOG"
echo "Results: $ANALYSIS_DIR"
exit "$status"
