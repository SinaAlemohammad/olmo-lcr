#!/usr/bin/env bash
# =============================================================================
#  Stage 4 — evaluate the base model and every saved checkpoint.
#
#  Runs inside the EVAL env (lm-eval + vllm). One lm_eval process per GPU.
#  Idempotent: any (checkpoint, task-group) that already has results is skipped,
#  so it is safe to re-run after a crash or after more training finishes.
#
#  ONLY_SUBSET=N bash stages/4_evaluate.sh   -> evaluate base + subset N only
#                                              (only relevant if NUM_SUBSETS > 1)
#
#  Output tree:  $EVAL_DIR/<label>/<group>/.../results_*.json
#  Bundle:       $RESULTS_DIR/results_bundle_<TEMP_TAG>.tar.gz   <- send this back
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
source "$LCR_ROOT/storage.sh"

ONLY_SUBSET="${ONLY_SUBSET:-}"
STAGE="4_evaluate"
if [ -z "$ONLY_SUBSET" ] && is_done "$STAGE"; then log "already done, skipping"; exit 0; fi

# vLLM V0 is the safe engine for lm-eval's loglikelihood requests with the
# pinned vllm. Remove for vllm >= 0.10.
export VLLM_USE_V1=0
export HF_ALLOW_CODE_EVAL=1     # humaneval executes generated code

mkdir -p "$EVAL_DIR" "$RESULTS_DIR"
if [ -n "$ONLY_SUBSET" ]; then SUBSETS="$ONLY_SUBSET"; else SUBSETS="$(seq 1 "$NUM_SUBSETS")"; fi

# ---- 1. list what to evaluate ----------------------------------------------
MODELS_JSON="$EVAL_DIR/models${ONLY_SUBSET:+_subset_v$ONLY_SUBSET}.json"
python - "$MODELS_JSON" $SUBSETS <<'PY'
import glob, json, os, sys
out = sys.argv[1]; subsets = [int(s) for s in sys.argv[2:]]
models = []
if os.environ.get("EVAL_BASE_MODEL", "1") == "1":
    models.append({"label": "base", "path": os.environ["MODEL"], "subset": 0, "epoch": 0})
for s in subsets:
    for d in sorted(glob.glob(f"{os.environ['TRAIN_DIR']}/subset_v{s}/epoch-*")):
        ep = int(d.rsplit("epoch-", 1)[1])
        models.append({"label": f"subset_v{s}__epoch-{ep:02d}", "path": d, "subset": s, "epoch": ep})
with open(out, "w") as f:
    json.dump(models, f, indent=2)
print(f"{len(models)} models listed in {out}")
PY
# the union list used by collect_results (base + all subsets found so far)
if [ -z "$ONLY_SUBSET" ]; then cp "$MODELS_JSON" "$EVAL_DIR/models.json"; else
python - "$EVAL_DIR/models.json" "$MODELS_JSON" <<'PY'
import json, os, sys
union, new = sys.argv[1], sys.argv[2]
cur = json.load(open(union)) if os.path.exists(union) else []
seen = {m["label"] for m in cur}
cur += [m for m in json.load(open(new)) if m["label"] not in seen]
json.dump(cur, open(union, "w"), indent=2)
PY
fi

# ---- 2. run the jobs --------------------------------------------------------
LIMIT_FLAG=""
[ -n "$EVAL_LIMIT" ] && LIMIT_FLAG="--limit $EVAL_LIMIT"
set +e
# shellcheck disable=SC2086
python "$LCR_ROOT/src/run_eval.py" \
    --models_json "$MODELS_JSON" \
    --eval_dir "$EVAL_DIR" \
    --suite "$EVAL_SUITE" \
    --humaneval_mode "$HUMANEVAL_MODE" \
    --gpus "$GPU_LIST" \
    --gpu_mem_util "$EVAL_GPU_MEM_UTIL" \
    --max_model_len "$EVAL_MAX_MODEL_LEN" \
    --seed "$SEED" \
    $LIMIT_FLAG \
    2>&1 | tee -a "$LOG_DIR/evaluate${ONLY_SUBSET:+_subset_v$ONLY_SUBSET}.log"
EVAL_RC=${PIPESTATUS[0]}
set -e

# ---- 3. optionally free disk ------------------------------------------------
if [ "$EVAL_RC" -eq 0 ] && [ "$DELETE_CKPT_AFTER_EVAL" = "1" ]; then
    for s in $SUBSETS; do
        log "deleting checkpoints of subset_v$s (all evals succeeded)"
        rm -rf "$TRAIN_DIR/subset_v${s}"/epoch-*
    done
fi

# ---- 4. bundle --------------------------------------------------------------
# Always (re)build the bundle from whatever exists, even if some jobs failed,
# so partial numbers can be sent back.
python "$LCR_ROOT/src/collect_results.py" \
    --eval_dir "$EVAL_DIR" \
    --train_dir "$TRAIN_DIR" \
    --results_dir "$RESULTS_DIR" \
    --temperature "$TEMPERATURE" \
    --extra_file "$SYNTH_DIR/corpus_metadata.json" \
    --extra_file "$SUBSET_DIR/subsets_summary.json"
BUNDLE="$RUN_ROOT/results_bundle_${TEMP_TAG}.tar.gz"
tar -czf "$BUNDLE" -C "$RUN_ROOT" results
storage_upload "$BUNDLE" "$TEMP_TAG/results/$(basename "$BUNDLE")"
log "bundle -> $BUNDLE"

if [ "$EVAL_RC" -ne 0 ]; then
    log "ERROR: some eval jobs failed (see above). Re-run this stage to retry only the failed ones."
    exit 1
fi
[ -z "$ONLY_SUBSET" ] && mark_done "$STAGE"
log "done"
