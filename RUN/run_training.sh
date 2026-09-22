#!/usr/bin/env bash
#
# Train every model at every horizon, in both modes, over every seed.
#
#   bash RUN/run_training.sh            # every cell not already recorded
#   MODELS="gru" bash RUN/run_training.sh
#   MODELS="lstm" HORIZONS="60" bash RUN/run_training.sh
#
# This script only trains. Analysing what it produced is RUN/run_experiments_analysis.sh,
# which reads the run list this one writes.
#
# Each finished cell already carries its own cross-seed metrics: the runner writes
# <parent>/aggregate_metrics.json and .csv after the last seed. The analysis stages
# read those directly, so nothing downstream recomputes them.
#
# Resumable. Every finished run appends "<model> <minutes> <dir>" to
# $LOG_DIR/parents.txt and is skipped on a re-run, so an interrupted job
# continues where it stopped. Delete a line to force that cell to redo.
#
# Timing: hours. One cell is <horizons> x 2 modes x <seeds> fits.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=RUN/_common.sh
. RUN/_common.sh

if [ "$#" -gt 0 ]; then
    echo "usage: $0        (configure with MODELS/HORIZONS/SEEDS in the environment)" >&2
    exit 2
fi

total=0
for model in $MODELS; do for minutes in $HORIZONS; do total=$((total + 1)); done; done

echo "=== plan ==="
printf '  models     : %s\n' "$MODELS"
printf '  horizons   : %s min\n' "$HORIZONS"
printf '  seeds      : %s\n' "$SEEDS"
printf '  cells      : %s (each: 2 modes x %s seeds)\n' "$total" "$(printf '%s' "$SEEDS" | wc -w | tr -d ' ')"
printf '  run list   : %s\n' "$PARENTS"
printf '  logs       : %s\n' "$LOG_DIR"

index=0
for model in $MODELS; do
    for minutes in $HORIZONS; do
        index=$((index + 1))
        if [ -n "$(parent_for "$model" "$minutes")" ]; then
            say "[$index/$total] $model $minutes min: already recorded, skipping"
            continue
        fi
        config="${CONFIG_PREFIX}_${model}_${minutes}min.yaml"
        log="$LOG_DIR/run_${model}_${minutes}min.log"
        if [ ! -f "$config" ]; then
            echo "FAILED: $model $minutes min has no config at $config" >&2
            exit 1
        fi
        say "[$index/$total] $model $minutes min: training ($config)"
        date '+  started %Y-%m-%d %H:%M:%S'
        set +e
        python -u -m benchmark.cli run --config "$config" 2>&1 | tee "$log"
        status=${PIPESTATUS[0]}
        set -e
        date '+  finished %Y-%m-%d %H:%M:%S'
        parent=$(sed -n 's/^Parent experiment: //p' "$log" | head -1)
        if [ -z "$parent" ]; then
            echo "FAILED: $model $minutes min produced no parent directory (exit $status)." >&2
            echo "        See $log" >&2
            exit 1
        fi
        printf '%s %s %s\n' "$model" "$minutes" "$parent" >> "$PARENTS"
        [ "$status" -eq 0 ] || echo "WARNING: $model $minutes min exited $status; some seeds may have failed" >&2
    done
done

say "runs recorded"
cat "$PARENTS"
echo
echo "Cross-seed metrics: <parent>/aggregate_metrics.{json,csv}"
echo "Next: bash RUN/run_experiments_analysis.sh --check"
