#!/usr/bin/env bash
# =============================================================================
#  Stage 1 — generate the synthetic corpus.
#
#  Runs inside the TRAIN env (needs vllm).
#
#  One vLLM worker per GPU, each sampling GEN_SAMPLES_PER_GPU_ROUND texts
#  from the start token at TEMPERATURE. Rounds repeat until the corpus holds
#  at least TARGET_TOKENS tokens. Shards are then merged into
#      $SYNTH_DIR/corpus.jsonl          one {"text", "num_tokens"} per line
#      $SYNTH_DIR/corpus_metadata.json  counts, mean length, settings
#
#  Output is skipped entirely if the stage marker exists.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
source "$LCR_ROOT/storage.sh"

STAGE="1_generate"
if is_done "$STAGE"; then log "already done, skipping (delete $MARKER_DIR/$STAGE.done to redo)"; exit 0; fi

# vLLM engine choice. V0 is the conservative path for min_tokens/stop handling
# with the pinned vllm version. Remove this line for vllm >= 0.10 (V0 is gone).
export VLLM_USE_V1=0

mkdir -p "$SYNTH_DIR"
GEN_PY="$LCR_ROOT/src/generate_synthetic.py"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

# Resume support: if a merged corpus exists remotely but not locally, pull it.
storage_restore_if_missing "$TEMP_TAG/1_synthetic/corpus.jsonl" "$SYNTH_DIR/corpus.jsonl"

if [ ! -f "$SYNTH_DIR/corpus.jsonl" ]; then
    round=0
    while :; do
        have=$(python "$GEN_PY" --count --out_dir "$SYNTH_DIR")
        log "corpus has $have / $TARGET_TOKENS tokens"
        [ "$have" -ge "$TARGET_TOKENS" ] && break
        if [ "$round" -ge "$MAX_GEN_ROUNDS" ]; then
            log "ERROR: still short after $MAX_GEN_ROUNDS rounds; raise GEN_SAMPLES_PER_GPU_ROUND"; exit 1
        fi
        log "round $round: launching $NUM_GPUS workers x $GEN_SAMPLES_PER_GPU_ROUND samples"
        pids=()
        for i in "${!GPUS[@]}"; do
            gpu="${GPUS[$i]}"
            CUDA_VISIBLE_DEVICES="$gpu" python "$GEN_PY" \
                --model "$MODEL" \
                --temperature "$TEMPERATURE" \
                --start_token "$START_TOKEN" \
                --num_samples "$GEN_SAMPLES_PER_GPU_ROUND" \
                --min_tokens "$MIN_TOKENS" \
                --max_tokens "$MAX_TOKENS" \
                --gpu_mem_util "$GEN_GPU_MEM_UTIL" \
                --seed "$SEED" --round "$round" --worker_id "$i" \
                --out_dir "$SYNTH_DIR" \
                > "$LOG_DIR/gen_round${round}_gpu${gpu}.log" 2>&1 &
            pids+=($!)
        done
        fail=0
        for p in "${pids[@]}"; do wait "$p" || fail=1; done
        if [ "$fail" = "1" ]; then
            log "ERROR: a generation worker failed. See $LOG_DIR/gen_round${round}_gpu*.log"; exit 1
        fi
        round=$((round + 1))
    done

    log "merging shards"
    python "$GEN_PY" --merge --out_dir "$SYNTH_DIR" \
        --model "$MODEL" --temperature "$TEMPERATURE" --start_token "$START_TOKEN" --seed "$SEED"
fi

storage_upload "$SYNTH_DIR/corpus.jsonl"          "$TEMP_TAG/1_synthetic/corpus.jsonl"
storage_upload "$SYNTH_DIR/corpus_metadata.json"  "$TEMP_TAG/1_synthetic/corpus_metadata.json"
mark_done "$STAGE"
log "done -> $SYNTH_DIR"
