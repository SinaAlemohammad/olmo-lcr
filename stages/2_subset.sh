#!/usr/bin/env bash
# =============================================================================
#  Stage 2 — cut NUM_SUBSETS (default 1) subsets of TOKENS_PER_SUBSET tokens
#  from the merged corpus. No audit, no stratification: shuffle once (seeded),
#  then take samples in shuffled order until 5M tokens. Nothing else.
#
#  Runs inside the TRAIN env (only needs python + numpy).
#  Outputs: $SUBSET_DIR/subset_v1.jsonl and subsets_summary.json
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
source "$LCR_ROOT/storage.sh"

STAGE="2_subset"
if is_done "$STAGE"; then log "already done, skipping"; exit 0; fi

storage_restore_if_missing "$TEMP_TAG/1_synthetic/corpus.jsonl" "$SYNTH_DIR/corpus.jsonl"
[ -f "$SYNTH_DIR/corpus.jsonl" ] || { log "ERROR: $SYNTH_DIR/corpus.jsonl missing — run stage 1"; exit 1; }

mkdir -p "$SUBSET_DIR"
python "$LCR_ROOT/src/create_subsets.py" \
    --corpus "$SYNTH_DIR/corpus.jsonl" \
    --out_dir "$SUBSET_DIR" \
    --num_subsets "$NUM_SUBSETS" \
    --tokens_per_subset "$TOKENS_PER_SUBSET" \
    --seed "$SEED"

storage_upload "$SUBSET_DIR" "$TEMP_TAG/2_subsets"
mark_done "$STAGE"
log "done -> $SUBSET_DIR"
