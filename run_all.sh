#!/usr/bin/env bash
# =============================================================================
#  run_all.sh — the one command per node.
#
#      TEMPERATURE=0.75 bash run_all.sh      # node 1
#      TEMPERATURE=1.0  bash run_all.sh      # node 2
#      TEMPERATURE=1.25 bash run_all.sh      # node 3
#
#  Pipeline (each stage skips itself if its marker exists, so re-running
#  after a crash resumes at the first unfinished stage):
#
#      [train env]  stage 1  generate synthetic corpus
#      [train env]  stage 2  cut one 5M-token subset from the ~10M-token corpus
#      [train env]  stage 3  train 40 epochs, save weights every 2 epochs (20 ckpts)
#      [eval  env]  stage 4  evaluate base + 20 checkpoints, build results bundle
#
#  (INTERLEAVE_EVAL=1 only matters if NUM_SUBSETS > 1.)
#
#  Environments are switched with `conda activate` inside a subshell per
#  stage; nothing in your current shell is touched.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

# ---- conda ------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not on PATH (needed to activate $TRAIN_ENV / $EVAL_ENV)" >&2; exit 1
fi
CONDA_BASE="$(conda info --base)"
run_in_env() {
    # run_in_env <env> <command...>   — runs the command with <env> activated,
    # in a subshell, inheriting all exported config variables.
    local env="$1"; shift
    (
        # shellcheck source=/dev/null
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$env" || { echo "ERROR: conda env '$env' not found. Run env/setup_*_env.sh" >&2; exit 1; }
        "$@"
    )
}

# ---- go ---------------------------------------------------------------------
exec > >(tee -a "$LOG_DIR/run_all.log") 2>&1
echo "==== run_all.sh started $(date -u) on $(hostname) ===="
print_config

T0=$(date +%s)
stage_time() { echo "---- $1 finished at $(date -u) (+$(( ($(date +%s) - T0) / 60 )) min total) ----"; }

run_in_env "$TRAIN_ENV" bash "$HERE/stages/1_generate.sh";  stage_time "stage 1 generate"
run_in_env "$TRAIN_ENV" bash "$HERE/stages/2_subset.sh";    stage_time "stage 2 subset"

if [ "$INTERLEAVE_EVAL" = "1" ]; then
    for s in $(seq 1 "$NUM_SUBSETS"); do
        export ONLY_SUBSET="$s"
        run_in_env "$TRAIN_ENV" bash "$HERE/stages/3_train.sh";    stage_time "stage 3 train subset_v$s"
        run_in_env "$EVAL_ENV"  bash "$HERE/stages/4_evaluate.sh"; stage_time "stage 4 eval  subset_v$s"
        unset ONLY_SUBSET
    done
    run_in_env "$EVAL_ENV" bash "$HERE/stages/4_evaluate.sh";  stage_time "stage 4 final bundle"
else
    run_in_env "$TRAIN_ENV" bash "$HERE/stages/3_train.sh";     stage_time "stage 3 train"
    run_in_env "$EVAL_ENV"  bash "$HERE/stages/4_evaluate.sh";  stage_time "stage 4 evaluate"
fi

echo "==== ALL DONE $(date -u) ===="
echo "Send back:  $RUN_ROOT/results_bundle_${TEMP_TAG}.tar.gz"
