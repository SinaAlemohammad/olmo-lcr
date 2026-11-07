#!/usr/bin/env bash
# =============================================================================
#  run_pilecc.sh — OLMo-2-13B continued pretraining on Pile-CC instead of its
#  own samples: the generic-text control for the self-training runs (the OLMo
#  analogue of the Qwen Common Corpus baseline). One command = one seed:
#
#     [1] sample a random ~5M-token English Pile-CC set          CPU, ~10-30 min
#         (50-2048 tokens/doc, 8-gram decontaminated)
#     [2] train 40 epochs, save weights every 2 epochs           8 GPUs, ~6-8 h
#     [3] evaluate base + 20 checkpoints (Figure 4 suite)        8 GPUs, ~4-7 h
#
#  Only the training text differs from the synthetic runs. Every other setting
#  comes from config.sh unchanged: lr 1e-6, effective batch 64, 2048-token
#  packing with EOS after each document, cosine + 10% warmup, 40 epochs,
#  full-parameter ZeRO-3 bf16, lm-eval zero-shot suite A = Figure 4
#  (MMLU, ARC-C, GSM8K, HellaSwag, TruthfulQA-MC2, Minerva-MATH, HumanEval).
#
#  SEED drives every random choice: which Pile-CC documents are drawn, the
#  training seed (HF set_seed + data_seed, i.e. data order), and the vLLM /
#  lm-eval seeds.
#
#  Usage, from the repo root:
#      SEED=42 bash run_pilecc.sh
#      SEED=43 bash run_pilecc.sh
#      SEED=44 bash run_pilecc.sh
#      SMOKE=1 SEED=42 bash run_pilecc.sh       # ~30-min end-to-end test first
#
#  Optional overrides:
#      PILE_FILES='/data/pile/train/*.jsonl.zst'  local Pile copy (default: stream
#                                                 monology/pile-uncopyrighted from HF)
#      MIN_DOC_TOKENS=50 MAX_DOC_TOKENS=2048      document length filter (OLMo tokens)
#      LANG_MIN_PROB=0.9                          English filter threshold (langid)
#      HUMANEVAL_MODE=greedy|sample               must equal the synthetic runs'
#                                                 (see their config_snapshot.txt)
#      WORK_DIR=/local/scratch                    output root (default ./work, shared
#                                                 with the synthetic runs and their hf_cache)
#      EVAL_BASE_MODEL=0                          skip re-evaluating the base model
#      PER_DEVICE_BS=1 / DS_CONFIG=...            OOM fixes, as in README
#      DELETE_CKPT_AFTER_EVAL=1                   keep 0 if extraction runs later
#
#  Output:  work/Tpilecc_seed<SEED>/results_bundle_Tpilecc_seed<SEED>.tar.gz
#  Re-running resumes at the first unfinished step (markers in .markers/).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-42}"
# Same WORK_DIR as the synthetic runs: outputs land in work/Tpilecc_seed<SEED>/
# next to work/T1_25/ etc., and work/hf_cache (OLMo weights + benchmarks) is reused.
export WORK_DIR="${WORK_DIR:-$HERE/work}"
export EVAL_SUITE="${EVAL_SUITE:-A}"            # Figure 4 = zero-shot suite A
# TEMPERATURE is never used for sampling here (no generation). config.sh builds
# the run directory from it, so it doubles as the run tag: Tpilecc_seed<SEED>.
export TEMPERATURE="pilecc_seed${SEED}"
# shellcheck source=config.sh
source "$HERE/config.sh"

# ---- Pile-CC sampling knobs -------------------------------------------------
PILE_FILES="${PILE_FILES:-}"
MIN_DOC_TOKENS="${MIN_DOC_TOKENS:-50}"
MAX_DOC_TOKENS="${MAX_DOC_TOKENS:-2048}"
LANG_MIN_PROB="${LANG_MIN_PROB:-0.9}"
NGRAM_N=8
if [ "$SMOKE" = "1" ]; then SHUFFLE_BUFFER=2000; MAX_SKIP=10000
else                        SHUFFLE_BUFFER=20000; MAX_SKIP=500000; fi

# ---- conda ------------------------------------------------------------------
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH" >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
run_in_env() {
    local env="$1"; shift
    (
        # shellcheck source=/dev/null
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$env" || { echo "ERROR: conda env '$env' not found. Run env/setup_*_env.sh" >&2; exit 1; }
        "$@"
    )
}

exec > >(tee -a "$LOG_DIR/run_pilecc.log") 2>&1
echo "==== run_pilecc.sh started $(date -u) on $(hostname) ===="
print_config
cat <<EOF
================ Pile-CC sampling ================
  source                ${PILE_FILES:-hf://datasets/monology/pile-uncopyrighted (streamed)}
  doc length            $MIN_DOC_TOKENS..$MAX_DOC_TOKENS OLMo tokens
  language              en, langid prob >= $LANG_MIN_PROB
  decontamination       ${NGRAM_N}-gram, 7 Figure-4 benchmarks (audit.py rule)
  budget                $TOKENS_PER_SUBSET tokens   shuffle buffer $SHUFFLE_BUFFER   max skip $MAX_SKIP
==================================================
EOF
T0=$(date +%s)
step_time() { echo "---- $1 finished at $(date -u) (+$(( ($(date +%s) - T0) / 60 )) min total) ----"; }

RUN_DIR="$TRAIN_DIR/subset_v1"
TRAIN_FILE="$SUBSET_DIR/subset_v1.jsonl"

# ============================================================================
#  [1] sample Pile-CC
# ============================================================================
if is_done "1_sample_pilecc"; then
    log "[1] already done, skipping"
else
    run_in_env "$TRAIN_ENV" bash -c \
        'python -c "import zstandard, langid" 2>/dev/null || pip install --quiet "zstandard>=0.22" "langid==1.1.6"'
    rm -rf "$SUBSET_DIR"; mkdir -p "$SUBSET_DIR"
    # shellcheck disable=SC2086
    run_in_env "$TRAIN_ENV" python "$LCR_ROOT/src/sample_pilecc.py" \
        --out_dir "$SUBSET_DIR" \
        --seed "$SEED" \
        --model "$MODEL" \
        --tokens "$TOKENS_PER_SUBSET" \
        --min_doc_tokens "$MIN_DOC_TOKENS" \
        --max_doc_tokens "$MAX_DOC_TOKENS" \
        --lang_min_prob "$LANG_MIN_PROB" \
        --ngram "$NGRAM_N" \
        --shuffle_buffer "$SHUFFLE_BUFFER" \
        --max_skip "$MAX_SKIP" \
        ${PILE_FILES:+--files "$PILE_FILES"} \
        2>&1 | tee "$LOG_DIR/sample_pilecc.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] && [ -f "$TRAIN_FILE" ] || { log "ERROR: sampling failed"; exit 1; }
    mark_done "1_sample_pilecc"
fi
step_time "[1] sample"

# ============================================================================
#  [2] train — identical call to stages/3_train.sh
# ============================================================================
if is_done "2_train"; then
    log "[2] already done, skipping"
else
    # weights-only checkpoints cannot be resumed: start a half-finished run clean
    [ -d "$RUN_DIR" ] && { log "removing incomplete run $RUN_DIR"; rm -rf "$RUN_DIR"; }
    mkdir -p "$RUN_DIR"
    set +e
    run_in_env "$TRAIN_ENV" env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
        torchrun --standalone --nproc_per_node="$NUM_GPUS" "$LCR_ROOT/src/train.py" \
        --model "$MODEL" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$RUN_DIR" \
        --deepspeed "$LCR_ROOT/configs/$DS_CONFIG" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --save_every_epochs "$SAVE_EVERY_EPOCHS" \
        --per_device_bs "$PER_DEVICE_BS" \
        --grad_accum "$GRAD_ACCUM" \
        --cutoff_len "$CUTOFF_LEN" \
        --warmup_ratio "$WARMUP_RATIO" \
        --seed "$SEED" \
        --logging_steps "$LOGGING_STEPS" \
        --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
        2>&1 | tee "$LOG_DIR/train_subset_v1.log"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ] || [ ! -f "$RUN_DIR/TRAIN_DONE" ]; then
        log "ERROR: training failed (rc=$rc). See $LOG_DIR/train_subset_v1.log"; exit 1
    fi
    log "trained: $(ls -d "$RUN_DIR"/epoch-* | wc -l) checkpoints"
    mark_done "2_train"
fi
step_time "[2] train"

# ============================================================================
#  [3] evaluate base + checkpoints, build bundle — same as stages/4_evaluate.sh
# ============================================================================
if is_done "3_evaluate"; then
    log "[3] already done, skipping"
else
    mkdir -p "$EVAL_DIR" "$RESULTS_DIR"
    run_in_env "$EVAL_ENV" python - "$EVAL_DIR/models.json" <<'PY'
import glob, json, os, sys
models = []
if os.environ.get("EVAL_BASE_MODEL", "1") == "1":
    models.append({"label": "base", "path": os.environ["MODEL"], "subset": 0, "epoch": 0})
for d in sorted(glob.glob(f"{os.environ['TRAIN_DIR']}/subset_v1/epoch-*")):
    ep = int(d.rsplit("epoch-", 1)[1])
    models.append({"label": f"subset_v1__epoch-{ep:02d}", "path": d, "subset": 1, "epoch": ep})
json.dump(models, open(sys.argv[1], "w"), indent=2)
print(f"{len(models)} models to evaluate")
PY
    LIMIT_FLAG=""
    [ -n "$EVAL_LIMIT" ] && LIMIT_FLAG="--limit $EVAL_LIMIT"
    set +e
    # shellcheck disable=SC2086
    run_in_env "$EVAL_ENV" env VLLM_USE_V1=0 HF_ALLOW_CODE_EVAL=1 \
        python "$LCR_ROOT/src/run_eval.py" \
        --models_json "$EVAL_DIR/models.json" \
        --eval_dir "$EVAL_DIR" \
        --suite "$EVAL_SUITE" \
        --humaneval_mode "$HUMANEVAL_MODE" \
        --gpus "$GPU_LIST" \
        --gpu_mem_util "$EVAL_GPU_MEM_UTIL" \
        --max_model_len "$EVAL_MAX_MODEL_LEN" \
        --seed "$SEED" \
        $LIMIT_FLAG \
        2>&1 | tee -a "$LOG_DIR/evaluate.log"
    EVAL_RC=${PIPESTATUS[0]}
    set -e

    if [ "$EVAL_RC" -eq 0 ] && [ "$DELETE_CKPT_AFTER_EVAL" = "1" ]; then
        log "deleting checkpoints (all evals succeeded)"; rm -rf "$RUN_DIR"/epoch-*
    fi

    run_in_env "$EVAL_ENV" python "$LCR_ROOT/src/collect_results.py" \
        --eval_dir "$EVAL_DIR" \
        --train_dir "$TRAIN_DIR" \
        --results_dir "$RESULTS_DIR" \
        --temperature "$TEMPERATURE" \
        --extra_file "$SUBSET_DIR/sampling_report.json"
    BUNDLE="$RUN_ROOT/results_bundle_${TEMP_TAG}.tar.gz"
    tar -czf "$BUNDLE" -C "$RUN_ROOT" results
    log "bundle -> $BUNDLE"

    if [ "$EVAL_RC" -ne 0 ]; then
        log "ERROR: some eval jobs failed. Re-run this script to retry only those."; exit 1
    fi
    mark_done "3_evaluate"
fi
step_time "[3] evaluate"

echo "==== ALL DONE $(date -u) ===="
echo "Send back:  $RUN_ROOT/results_bundle_${TEMP_TAG}.tar.gz"
