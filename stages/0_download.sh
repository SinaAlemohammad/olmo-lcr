#!/usr/bin/env bash
# =============================================================================
#  Stage 0 (optional) — pre-download everything into HF_HOME so the compute
#  nodes can run with HF_OFFLINE=1 (no internet).
#
#  Run this ONCE on a machine that has internet AND both conda envs, with
#  HF_HOME pointing at the same location the compute nodes will use
#  (config.sh: HF_HOME defaults to $WORK_DIR/hf_cache).
#
#      bash stages/0_download.sh
#
#  Downloads
#    1. the base model weights + tokenizer                    (train env, ~26 GB)
#    2. every benchmark dataset the eval tasks use, plus the
#       HF "code_eval" metric script for humaneval. Done by
#       running lm_eval on CPU with a tiny model and --limit 2  (eval env, a few minutes)
#
#  Skip this if the nodes have internet: everything is fetched on first use.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

CONDA_BASE="$(conda info --base)"
run_in_env() { local env="$1"; shift; ( source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$env"; "$@" ); }

unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE   # we need the network here
log "HF_HOME = $HF_HOME"

log "1/2 model weights: $MODEL"
run_in_env "$TRAIN_ENV" python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL")
print("   ->", p)
PY

log "2/2 benchmark datasets + code_eval metric (CPU, tiny model, 2 items per task)"
TASKS="mmlu,arc_challenge,hellaswag,truthfulqa_mc2,gsm8k,minerva_math,humaneval"
TMP="$(mktemp -d)"
export HF_ALLOW_CODE_EVAL=1
run_in_env "$EVAL_ENV" lm_eval \
    --model hf --model_args "pretrained=EleutherAI/pythia-70m,dtype=float32" \
    --device cpu --tasks "$TASKS" --num_fewshot 0 --limit 2 --batch_size 4 \
    --gen_kwargs "max_gen_toks=8" --confirm_run_unsafe_code \
    --output_path "$TMP" > "$LOG_DIR/download_datasets.log" 2>&1 \
    || { log "lm_eval prefetch failed, see $LOG_DIR/download_datasets.log"; exit 1; }
rm -rf "$TMP"
log "done. Set HF_OFFLINE=1 on the compute nodes (or export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1)."
