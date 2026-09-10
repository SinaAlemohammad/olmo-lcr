#!/usr/bin/env bash
# =============================================================================
#  env/setup_train_env.sh — conda env for stages 1-3 (generation + training).
#
#  Contents (pinned to a set known to work together):
#      python 3.11
#      torch 2.6.0 (CUDA 12.4 build, pulled in by vllm)   needs NVIDIA driver >= 525 (12.x)
#      vllm 0.8.5.post1            generation
#      transformers 4.51.3         OLMo-2 support (>= 4.47 needed)
#      deepspeed 0.16.5            ZeRO-3 (no compiled CUDA ops needed: we use torch AdamW)
#      datasets / accelerate / numpy
#
#  Usage:   bash env/setup_train_env.sh            (env name from config.sh: lcr-train)
#           TRAIN_ENV=myname bash env/setup_train_env.sh
#
#  If your cluster mandates a different torch/CUDA build, change the pins in
#  the block below; the code itself has no hard dependency on these exact
#  versions beyond transformers >= 4.47 (OLMo-2) and vllm >= 0.6.4 (OLMo-2).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${TRAIN_ENV:-lcr-train}"

# ---------------------------------------------------------------- pins
PY_VER="3.11"
VLLM_VER="0.8.5.post1"          # installs torch==2.6.0 (cu124 wheel) as a dependency
TRANSFORMERS_VER="4.51.3"
DEEPSPEED_VER="0.16.5"
DATASETS_VER="3.6.0"
ACCELERATE_VER="1.6.0"
# ----------------------------------------------------------------

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found" >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "== creating conda env '$ENV_NAME' (python $PY_VER)"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "   env exists; reusing (delete with: conda env remove -n $ENV_NAME)"
else
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"

echo "== installing packages"
python -m pip install --upgrade pip wheel setuptools
python -m pip install "vllm==$VLLM_VER"
python -m pip install "transformers==$TRANSFORMERS_VER" \
                      "datasets==$DATASETS_VER" \
                      "accelerate==$ACCELERATE_VER" \
                      "deepspeed==$DEEPSPEED_VER" \
                      "numpy" "tqdm" "sentencepiece" "protobuf"

echo "== smoke test"
python - <<'PY'
import importlib, sys
for m in ["torch", "vllm", "transformers", "deepspeed", "datasets", "accelerate", "numpy"]:
    mod = importlib.import_module(m)
    print(f"   {m:<13} {getattr(mod, '__version__', '?')}")
import torch
print(f"   torch cuda      {torch.version.cuda}   available={torch.cuda.is_available()}   gpus={torch.cuda.device_count()}")
if torch.cuda.is_available():
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    print(f"   bf16 matmul     ok ({(x @ x).sum().item():.1f})")
from transformers import AutoConfig
try:
    AutoConfig.for_model("olmo2")
    print("   olmo2 arch      ok")
except Exception as e:
    print(f"   olmo2 arch      MISSING ({e}) -> transformers too old"); sys.exit(1)
import shutil
for cli in ["torchrun"]:
    print(f"   {cli:<13} {shutil.which(cli) or 'NOT ON PATH'}")
PY

echo
echo "== done. Activate with:  conda activate $ENV_NAME"
