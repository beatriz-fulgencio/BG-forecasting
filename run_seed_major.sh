#!/usr/bin/env bash
#
# Seed-major run of the publication grid.
#
# Order: one seed at a time across the whole grid.
#   seed 41 -> gru/lstm/rnn x 15/30/45/60 min (all 12 patients, both modes)
#   seed 42 -> the same twelve cells
#   seed 43 -> the same twelve cells
#
# The configs in configs/full_*.yaml carry seeds [41, 42, 43] and run all three
# inside a single parent run, which is seed-INNER. To get seed-major order this
# script derives one single-seed config per cell into configs/seedwise/ and runs
# those. Each derived run therefore lands in its own parent directory with
# single-seed aggregates (the cross-seed dispersion fields are null). Recombine
# the three seeds per cell afterwards with merge_seed_runs.py before running the
# downstream analysis, which expects one parent per cell with three seeds.
#
# Usage:
#   ./run_seed_major.sh                  # the full 3 x 12 grid, skipping finished cells
#   SEEDS="41" ./run_seed_major.sh       # one seed only
#   MODELS="gru" HORIZONS="30" ./run_seed_major.sh
#   DRY_RUN=1 ./run_seed_major.sh        # print the plan, run nothing
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-41 42 43}"
MODELS="${MODELS:-gru lstm rnn}"
HORIZONS="${HORIZONS:-15 30 45 60}"
DRY_RUN="${DRY_RUN:-0}"
GEN_DIR="configs/seedwise"
LOG_DIR="${LOG_DIR:-results/seed_major_logs}"
RESULTS_DIR="${RESULTS_DIR:-results/experiments}"

mkdir -p "$GEN_DIR" "$LOG_DIR"

# --- preflight -------------------------------------------------------------
# The point of this rerun is F=4 (glucose, basal, bolus, carbs). unimodal must
# be false, and feature engineering must be off or the cyclical hour columns
# push it to F=6. Fail here rather than after hours of training.
preflight() {
  local cfg="$1"
  [[ -f "$cfg" ]] || { echo "MISSING config: $cfg" >&2; return 1; }
  grep -qE '^  unimodal: *false' "$cfg" \
    || { echo "$cfg: unimodal is not false -> would train on glucose only (F=1)" >&2; return 1; }
  grep -qE '^  include_feature_engineering: *false' "$cfg" \
    || { echo "$cfg: include_feature_engineering is not false -> F=6, not 4" >&2; return 1; }
}

# --- has this cell already finished? ---------------------------------------
# A parent run is complete when its directory holds aggregate_metrics.json.
# Match on the experiment name recorded in resolved_config.yaml, because the
# directory names are timestamped and hashed.
already_done() {
  local name="$1" dir
  [[ -d "$RESULTS_DIR" ]] || return 1
  while IFS= read -r dir; do
    [[ -f "$(dirname "$dir")/aggregate_metrics.json" ]] && return 0
  done < <(grep -rlx "  name: ${name}" "$RESULTS_DIR"/*/resolved_config.yaml 2>/dev/null || true)
  return 1
}

# --- derive the single-seed config -----------------------------------------
derive_config() {
  local model="$1" horizon="$2" seed="$3"
  local src="configs/full_${model}_${horizon}min.yaml"
  local out="${GEN_DIR}/full_${model}_${horizon}min_seed${seed}.yaml"
  preflight "$src"
  sed -e "s|^  name: .*|  name: full_${model}_${horizon}min_seed${seed}|" \
      -e "s|^  description: .*|  description: ${model} regular vs transfer, 12 OhioT1DM patients, ${horizon} min horizon, seed ${seed}|" \
      -e "s|^  seeds: \[.*\]|  seeds: [${seed}]|" \
      "$src" > "$out"
  grep -qE "^  seeds: \[${seed}\]$" "$out" \
    || { echo "$out: seed substitution failed; check the seeds line in $src" >&2; return 1; }
  echo "$out"
}

# --- plan ------------------------------------------------------------------
# Check every source config up front. A bad flag must stop the script here,
# before a single model is trained, not hours later.
for model in $MODELS; do
  for horizon in $HORIZONS; do
    preflight "configs/full_${model}_${horizon}min.yaml" || {
      echo "Preflight failed. Nothing was run." >&2
      exit 1
    }
  done
done

total=0; planned=(); skipped=0
for seed in $SEEDS; do
  for model in $MODELS; do
    for horizon in $HORIZONS; do
      name="full_${model}_${horizon}min_seed${seed}"
      if already_done "$name"; then
        skipped=$((skipped + 1))
      else
        planned+=("${seed}:${model}:${horizon}")
        total=$((total + 1))
      fi
    done
  done
done

echo "=============================================================="
echo " Seed-major run"
echo " seeds:     $SEEDS"
echo " models:    $MODELS"
echo " horizons:  $HORIZONS"
echo " to run:    $total cell(s)    already complete: $skipped"
echo " logs:      $LOG_DIR"
echo "=============================================================="

if [[ "$total" -eq 0 ]]; then
  echo "Nothing left to run."
  exit 0
fi

if [[ "$DRY_RUN" != "0" ]]; then
  for item in "${planned[@]}"; do
    IFS=: read -r seed model horizon <<< "$item"
    echo "  would run  seed ${seed}  ${model}  ${horizon}min"
  done
  exit 0
fi

# --- run -------------------------------------------------------------------
started_at=$(date +%s)
index=0
failed=()

for item in "${planned[@]}"; do
  IFS=: read -r seed model horizon <<< "$item"
  index=$((index + 1))
  name="full_${model}_${horizon}min_seed${seed}"
  if ! cfg="$(derive_config "$model" "$horizon" "$seed")"; then
    echo "  SKIPPED ${name}: could not derive a single-seed config" >&2
    failed+=("$name")
    continue
  fi
  log="${LOG_DIR}/${name}.log"

  echo
  echo "--------------------------------------------------------------"
  echo "[${index}/${total}] seed ${seed}  ${model}  ${horizon} min"
  echo "  config: $cfg"
  echo "  log:    $log"
  echo "  start:  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "--------------------------------------------------------------"

  cell_start=$(date +%s)
  if "$PYTHON" -m benchmark.cli run --config "$cfg" 2>&1 | tee "$log"; then
    status="ok"
  else
    status="FAILED"
    failed+=("$name")
  fi
  cell_end=$(date +%s)
  printf "  %s  %s in %dm\n" "$status" "$name" $(( (cell_end - cell_start) / 60 ))
done

finished_at=$(date +%s)
echo
echo "=============================================================="
printf " Done: %d cell(s) in %.1f h\n" "$total" "$(echo "scale=2; ($finished_at - $started_at)/3600" | bc)"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo " FAILED (${#failed[@]}): ${failed[*]}"
  echo " Re-run this script; finished cells are skipped automatically."
else
  echo " All cells completed."
  echo " Next: python merge_seed_runs.py   (recombine the three seeds per cell)"
fi
echo "=============================================================="
[[ ${#failed[@]} -eq 0 ]]
