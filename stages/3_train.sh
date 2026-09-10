#!/usr/bin/env bash
# =============================================================================
#  Stage 3 — continued pretraining, one run per subset.
#
#  Runs inside the TRAIN env (torch, deepspeed, transformers).
#  Uses ALL GPUs in GPU_LIST for a single run (ZeRO-3). With NUM_SUBSETS=1
#  this is one run writing  $TRAIN_DIR/subset_v1/epoch-02 ... epoch-40
#
#  ONLY_SUBSET=N bash stages/3_train.sh    -> train just subset N (if NUM_SUBSETS > 1)
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
source "$LCR_ROOT/storage.sh"

ONLY_SUBSET="${ONLY_SUBSET:-}"
if [ -n "$ONLY_SUBSET" ]; then SUBSETS="$ONLY_SUBSET"; else SUBSETS="$(seq 1 "$NUM_SUBSETS")"; fi

storage_restore_if_missing "$TEMP_TAG/2_subsets" "$SUBSET_DIR"

for s in $SUBSETS; do
    STAGE="3_train_subset_v${s}"
    RUN_DIR="$TRAIN_DIR/subset_v${s}"
    TRAIN_FILE="$SUBSET_DIR/subset_v${s}.jsonl"
    if is_done "$STAGE"; then log "subset_v$s already trained, skipping"; continue; fi
    [ -f "$TRAIN_FILE" ] || { log "ERROR: $TRAIN_FILE missing — run stage 2"; exit 1; }

    # A half-finished run cannot be resumed (weights-only checkpoints, no
    # optimizer state), so start clean.
    if [ -d "$RUN_DIR" ]; then log "removing incomplete run $RUN_DIR"; rm -rf "$RUN_DIR"; fi
    mkdir -p "$RUN_DIR"

    GC_FLAG="--gradient_checkpointing $GRADIENT_CHECKPOINTING"
    log "training subset_v$s  ->  $RUN_DIR   (log: $LOG_DIR/train_subset_v${s}.log)"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_LIST" torchrun --standalone --nproc_per_node="$NUM_GPUS" \
        "$LCR_ROOT/src/train.py" \
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
        $GC_FLAG \
        2>&1 | tee "$LOG_DIR/train_subset_v${s}.log"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ] || [ ! -f "$RUN_DIR/TRAIN_DONE" ]; then
        log "ERROR: training subset_v$s failed (rc=$rc). See $LOG_DIR/train_subset_v${s}.log"; exit 1
    fi

    n_ckpt=$(ls -d "$RUN_DIR"/epoch-* 2>/dev/null | wc -l)
    log "subset_v$s finished with $n_ckpt checkpoints"

    if [ "$UPLOAD_CHECKPOINTS" = "1" ]; then
        storage_upload "$RUN_DIR" "$TEMP_TAG/3_train/subset_v${s}"
    fi
    # the small files are always worth mirroring
    for f in train_log.jsonl trainer_state.json run_metadata.json checkpoint_epochs.json; do
        [ -f "$RUN_DIR/$f" ] && storage_upload "$RUN_DIR/$f" "$TEMP_TAG/3_train/subset_v${s}/$f"
    done
    mark_done "$STAGE"
done
log "done"
