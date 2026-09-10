#!/usr/bin/env bash
# =============================================================================
#  config.sh  —  the ONLY file you should normally need to edit.
#
#  Every stage script and run_all.sh does `source config.sh`.
#  Any variable written as "${NAME:-default}" can be overridden from the shell
#  without editing this file, e.g.
#
#      TEMPERATURE=0.75 bash run_all.sh
#      WORK_DIR=/local/scratch/lcr TEMPERATURE=1.0 bash run_all.sh
#      SMOKE=1 TEMPERATURE=1.25 bash run_all.sh          # 30-min end-to-end test
#
#  Everything below the "derived" line is computed from the values above.
# =============================================================================

# `set -a` exports every variable defined until `set +a`, so the stage scripts
# and the python programs they launch all see the same values.
set -a

# ---- repo root (do not edit) ------------------------------------------------
LCR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
#  EXPERIMENT
# =============================================================================
# Base model to self-train. Must be a *base* (pretrained-only) checkpoint.
MODEL="${MODEL:-allenai/OLMo-2-1124-13B}"

# Sampling temperature for the synthetic corpus. One node = one temperature.
# The paper sweeps {0.75, 1.0, 1.25}.
TEMPERATURE="${TEMPERATURE:-1.25}"

# Token the generator is conditioned on ("unconditional" generation).
#   eos     -> tokenizer.eos_token  (<|endoftext|> for OLMo-2; this is the document
#              boundary OLMo saw in pretraining, so it is the principled choice)
#   bos     -> tokenizer.bos_token  (OLMo-2 has none -> error)
#   newline -> the single token for "\n"  (what the Qwen runs in the paper used,
#              because Qwen has no BOS either)
START_TOKEN="${START_TOKEN:-eos}"

# Master seed. Generation uses SEED + round*100000 + gpu*1000 per worker.
SEED="${SEED:-42}"

# =============================================================================
#  STAGE 1 — GENERATION  (vLLM, one worker per GPU, data-parallel)
# =============================================================================
# Keep generating rounds until the corpus has at least this many tokens.
# We need NUM_SUBSETS * TOKENS_PER_SUBSET = 5M; generate double (10M) so the
# subset is a random half of a larger pool rather than the whole corpus.
TARGET_TOKENS="${TARGET_TOKENS:-10000000}"
# Samples each GPU produces per round. 8 GPUs x 3000 samples x ~400-500 tokens
# is ~10M tokens, so one round is usually enough.
GEN_SAMPLES_PER_GPU_ROUND="${GEN_SAMPLES_PER_GPU_ROUND:-3000}"
MAX_GEN_ROUNDS=6
MIN_TOKENS=50            # vLLM min_tokens: EOS is blocked before this length
MAX_TOKENS=1024          # hard cap per sample
GEN_GPU_MEM_UTIL=0.90    # vLLM gpu_memory_utilization

# =============================================================================
#  STAGE 2 — SUBSETS
# =============================================================================
NUM_SUBSETS="${NUM_SUBSETS:-1}"           # one 5M-token subset per temperature
TOKENS_PER_SUBSET="${TOKENS_PER_SUBSET:-5000000}"

# =============================================================================
#  STAGE 3 — TRAINING  (HF Trainer + DeepSpeed ZeRO-3, full-parameter)
# =============================================================================
LR="${LR:-1e-6}"
EPOCHS="${EPOCHS:-40}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-2}"     # 40 / 2 = 20 checkpoints per run
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"             # micro-batch per GPU (sequences of CUTOFF_LEN). OOM? -> 1
EFFECTIVE_BS="${EFFECTIVE_BS:-64}"              # paper: 64. Do not change. GRAD_ACCUM is derived below
                                                #   as EFFECTIVE_BS / (PER_DEVICE_BS * NUM_GPUS).
# DeepSpeed config. ds_z3.json = ZeRO-3 (default). ds_z3_offload.json = ZeRO-3 +
# optimizer states on CPU RAM: last resort if PER_DEVICE_BS=1 still OOMs; ~2x slower.
DS_CONFIG="${DS_CONFIG:-ds_z3.json}"
CUTOFF_LEN=2048                                 # packed sequence length
WARMUP_RATIO=0.1                                # cosine schedule, 10% warmup, decay to 0
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"   # 1 = on (needed for 13B on 80GB)
LOGGING_STEPS=10

# =============================================================================
#  STAGE 4 — EVALUATION  (lm-evaluation-harness, vLLM backend, 1 job per GPU)
# =============================================================================
# A  = paper Suite A, zero-shot: mmlu, arc_challenge, gsm8k, hellaswag,
#      truthfulqa_mc2, minerva_math, humaneval
# B  = paper Suite B, few-shot: arc_challenge 25, gsm8k 8, hellaswag 5, minerva_math 4
# AB = both
EVAL_SUITE="${EVAL_SUITE:-A}"
# greedy = paper protocol (single greedy sample).  sample = one sample at T=0.2, top_p=0.95
HUMANEVAL_MODE="${HUMANEVAL_MODE:-greedy}"
EVAL_BASE_MODEL="${EVAL_BASE_MODEL:-1}"          # also evaluate the untouched base model
EVAL_GPU_MEM_UTIL=0.80
EVAL_MAX_MODEL_LEN=4096
EVAL_LIMIT="${EVAL_LIMIT:-}"                      # "" = full benchmarks; N = first N items (smoke)
# Delete a run's checkpoints once every eval job for it has succeeded.
# Turn on if local disk cannot hold 20 x 26 GB.
DELETE_CKPT_AFTER_EVAL="${DELETE_CKPT_AFTER_EVAL:-0}"

# =============================================================================
#  ENVIRONMENT / PATHS
# =============================================================================
TRAIN_ENV="${TRAIN_ENV:-lcr-train}"      # conda env created by env/setup_train_env.sh
EVAL_ENV="${EVAL_ENV:-lcr-eval}"         # conda env created by env/setup_eval_env.sh

# All outputs go under WORK_DIR/T<temperature>/ . Put this on fast LOCAL disk.
WORK_DIR="${WORK_DIR:-$LCR_ROOT/work}"

# HuggingFace cache (model weights + benchmark datasets). Shared across temperatures.
HF_HOME="${HF_HOME:-$WORK_DIR/hf_cache}"
# Set to 1 if the compute nodes have no internet. Run stages/0_download.sh first.
HF_OFFLINE="${HF_OFFLINE:-0}"

# Interleave: train subset 1 -> eval subset 1 -> (delete) -> train subset 2 ...
# Only matters if NUM_SUBSETS > 1.
INTERLEAVE_EVAL="${INTERLEAVE_EVAL:-0}"

# =============================================================================
#  SMOKE TEST — tiny everything, same code path. ~30 min on 8xA100 for 13B.
# =============================================================================
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
    TARGET_TOKENS=120000
    GEN_SAMPLES_PER_GPU_ROUND=40
    NUM_SUBSETS=1
    TOKENS_PER_SUBSET=60000
    EPOCHS=2
    SAVE_EVERY_EPOCHS=1
    EVAL_LIMIT=8
fi

# =============================================================================
#  DERIVED (do not edit)
# =============================================================================
# GPU list. Respects an inherited CUDA_VISIBLE_DEVICES; otherwise all GPUs.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_LIST="$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LIST="$(nvidia-smi -L | awk '{print NR-1}' | paste -sd, -)"
else
    GPU_LIST="0"
fi
NUM_GPUS="$(echo "$GPU_LIST" | tr ',' '\n' | grep -c .)"

GRAD_ACCUM="${GRAD_ACCUM:-$(( EFFECTIVE_BS / (PER_DEVICE_BS * NUM_GPUS) ))}"
[ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1
ACTUAL_BS=$(( PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS ))
if [ "$ACTUAL_BS" -ne "$EFFECTIVE_BS" ]; then
    echo "WARNING: effective batch = $PER_DEVICE_BS x $GRAD_ACCUM x $NUM_GPUS = $ACTUAL_BS, not $EFFECTIVE_BS." >&2
    echo "         Pick PER_DEVICE_BS / NUM_GPUS so that PER_DEVICE_BS*NUM_GPUS divides $EFFECTIVE_BS." >&2
fi

TEMP_TAG="T${TEMPERATURE//./_}"                  # 1.25 -> T1_25
if [ "$SMOKE" = "1" ]; then
    RUN_ROOT="$WORK_DIR/${TEMP_TAG}_smoke"
else
    RUN_ROOT="$WORK_DIR/$TEMP_TAG"
fi
SYNTH_DIR="$RUN_ROOT/1_synthetic"
SUBSET_DIR="$RUN_ROOT/2_subsets"
TRAIN_DIR="$RUN_ROOT/3_train"
EVAL_DIR="$RUN_ROOT/4_evals"
RESULTS_DIR="$RUN_ROOT/results"
LOG_DIR="$RUN_ROOT/logs"
MARKER_DIR="$RUN_ROOT/.markers"

PYTHONUNBUFFERED=1
TOKENIZERS_PARALLELISM=false
if [ "$HF_OFFLINE" = "1" ]; then
    HF_HUB_OFFLINE=1
    HF_DATASETS_OFFLINE=1
fi

set +a

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$MARKER_DIR" "$HF_HOME"

# ---- helpers used by every stage --------------------------------------------
# A stage writes MARKER_DIR/<name>.done when it finishes; re-running skips it.
# To force a re-run, delete the marker (and the stage's output directory).
mark_done() { date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MARKER_DIR/$1.done"; }
is_done()   { [ -f "$MARKER_DIR/$1.done" ]; }
log()       { echo "[$(date -u +'%Y-%m-%d %H:%M:%S') $(basename "${BASH_SOURCE[1]}")] $*"; }

# Print the effective configuration (called by run_all.sh).
print_config() {
    cat <<EOF
================ effective configuration ================
  MODEL                 $MODEL
  TEMPERATURE           $TEMPERATURE   (tag $TEMP_TAG)
  START_TOKEN           $START_TOKEN
  SEED                  $SEED
  GPUs                  $NUM_GPUS  [$GPU_LIST]
  generation            target >= $TARGET_TOKENS tokens, $GEN_SAMPLES_PER_GPU_ROUND samples/GPU/round
  subsets               $NUM_SUBSETS x $TOKENS_PER_SUBSET tokens
  training              lr=$LR epochs=$EPOCHS save_every=$SAVE_EVERY_EPOCHS ep
                        batch: $PER_DEVICE_BS/gpu x $GRAD_ACCUM accum x $NUM_GPUS gpus = $(( PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS ))
                        cutoff=$CUTOFF_LEN grad_ckpt=$GRADIENT_CHECKPOINTING
  evaluation            suite=$EVAL_SUITE humaneval=$HUMANEVAL_MODE base=$EVAL_BASE_MODEL limit='${EVAL_LIMIT}'
  interleave/delete     interleave=$INTERLEAVE_EVAL delete_ckpt=$DELETE_CKPT_AFTER_EVAL
  envs                  train=$TRAIN_ENV eval=$EVAL_ENV
  WORK_DIR              $WORK_DIR
  RUN_ROOT              $RUN_ROOT
  HF_HOME               $HF_HOME  (offline=$HF_OFFLINE)
  SMOKE                 $SMOKE
=========================================================
EOF
}
