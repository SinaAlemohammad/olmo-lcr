#!/usr/bin/env bash
# =============================================================================
#  storage.sh  —  the ONLY place that knows about remote storage.
#
#  All stages read and write on the node's LOCAL disk under $RUN_ROOT.
#  At a few stage boundaries they call the three functions below so that
#  artifacts can be mirrored to a remote store (object storage, "online disk",
#  anything that needs explicit download/upload).
#
#  DEFAULT BEHAVIOUR: no-ops. If the node's local disk (or a POSIX shared
#  filesystem mounted at WORK_DIR) is where you want everything to live, you
#  do not need to touch this file at all.
#
#  TO ADAPT: replace the bodies of storage_upload / storage_download /
#  storage_exists with your own tool (e.g. `mycloud cp`, `rclone`, an S3 CLI,
#  a python client). Keep the signatures. Nothing else in the repo needs to
#  change.
#
#  What gets uploaded, and when (see the stage scripts for the exact calls):
#    stage 1  ->  <prefix>/<T>/1_synthetic/corpus.jsonl (+ corpus_metadata.json)
#    stage 2  ->  <prefix>/<T>/2_subsets/               (small)
#    stage 3  ->  <prefix>/<T>/3_train/subset_v1/        (ONLY if UPLOAD_CHECKPOINTS=1; ~520 GB)
#    stage 4  ->  <prefix>/<T>/results/results_bundle_<T>.tar.gz   (a few MB — this is what Sina needs)
# =============================================================================

# Remote key prefix. Keys passed to the functions look like "T1_25/results/...".
REMOTE_PREFIX="${REMOTE_PREFIX:-lcr-olmo}"

# Upload full checkpoint directories after each training run? Default off:
# evaluation happens on the same node, so the weights never need to leave it.
UPLOAD_CHECKPOINTS="${UPLOAD_CHECKPOINTS:-0}"

# ---------------------------------------------------------------------------
# storage_upload <local_path> <remote_key>
#   Copy a local file OR directory to the remote store under <remote_key>.
#   Must be idempotent (called again on re-runs). Return non-zero on failure.
# ---------------------------------------------------------------------------
storage_upload() {
    local local_path="$1" remote_key="$2"
    # --- default: no-op -------------------------------------------------------
    echo "[storage] (no-op) upload   $local_path  ->  $REMOTE_PREFIX/$remote_key"
    return 0
    # --- example: POSIX mirror -------------------------------------------------
    # mkdir -p "$(dirname "/mnt/remote/$REMOTE_PREFIX/$remote_key")"
    # cp -r "$local_path" "/mnt/remote/$REMOTE_PREFIX/$remote_key"
    # --- example: S3-like CLI --------------------------------------------------
    # if [ -d "$local_path" ]; then
    #     mycli cp --recursive "$local_path" "store://bucket/$REMOTE_PREFIX/$remote_key"
    # else
    #     mycli cp "$local_path" "store://bucket/$REMOTE_PREFIX/$remote_key"
    # fi
}

# ---------------------------------------------------------------------------
# storage_download <remote_key> <local_path>
#   Fetch <remote_key> (file or directory) from the remote store to <local_path>.
#   Only used when a stage's local output is missing but storage_exists says the
#   remote copy is there (i.e. resuming on a fresh node). Return non-zero on failure.
# ---------------------------------------------------------------------------
storage_download() {
    local remote_key="$1" local_path="$2"
    # --- default: no-op -------------------------------------------------------
    echo "[storage] (no-op) download $REMOTE_PREFIX/$remote_key  ->  $local_path"
    return 1
    # --- example: S3-like CLI --------------------------------------------------
    # mkdir -p "$(dirname "$local_path")"
    # mycli cp --recursive "store://bucket/$REMOTE_PREFIX/$remote_key" "$local_path"
}

# ---------------------------------------------------------------------------
# storage_exists <remote_key>
#   Return 0 if <remote_key> exists in the remote store, non-zero otherwise.
# ---------------------------------------------------------------------------
storage_exists() {
    local remote_key="$1"
    # --- default: nothing is remote -----------------------------------------
    return 1
    # --- example ------------------------------------------------------------
    # mycli ls "store://bucket/$REMOTE_PREFIX/$remote_key" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Helper used by stages: if <local_path> is missing but the remote has it,
# download it. Silent no-op otherwise.
# ---------------------------------------------------------------------------
storage_restore_if_missing() {
    local remote_key="$1" local_path="$2"
    if [ ! -e "$local_path" ] && storage_exists "$remote_key"; then
        echo "[storage] restoring $remote_key -> $local_path"
        storage_download "$remote_key" "$local_path"
    fi
}
