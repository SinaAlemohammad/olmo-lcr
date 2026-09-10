#!/usr/bin/env bash
# =============================================================================
#  env/setup_eval_env.sh — conda env for stage 4 (lm-evaluation-harness).
#
#  Kept SEPARATE from the training env on purpose: lm-eval's math extras
#  (sympy, antlr4-python3-runtime==4.11 for minerva_math answer parsing) and
#  its own transformers/datasets requirements tend to fight with training
#  stacks. Nothing from the training env is needed here except vllm (the
#  inference backend), which is installed again.
#
#  Contents:
#      python 3.11, torch 2.6.0 (via vllm 0.8.5.post1)
#      lm_eval 0.4.8 [math]     + evaluate (for humaneval's code_eval metric)
#      transformers 4.51.3, datasets 3.6.0
#
#  Usage:   bash env/setup_eval_env.sh         (env name from config.sh: lcr-eval)
# =============================================================================
set -euo pipefail
ENV_NAME="${EVAL_ENV:-lcr-eval}"

# ---------------------------------------------------------------- pins
PY_VER="3.11"
VLLM_VER="0.8.5.post1"
LM_EVAL_VER="0.4.8"
TRANSFORMERS_VER="4.51.3"
DATASETS_VER="3.6.0"
EVALUATE_VER="0.4.3"
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
python -m pip install "lm_eval[math]==$LM_EVAL_VER" \
                      "evaluate==$EVALUATE_VER" \
                      "transformers==$TRANSFORMERS_VER" \
                      "datasets==$DATASETS_VER" \
                      "sentencepiece" "protobuf"

echo "== smoke test"
python - <<'PY'
import importlib, shutil, sys
for m in ["torch", "vllm", "lm_eval", "evaluate", "sympy", "antlr4", "transformers", "datasets"]:
    try:
        mod = importlib.import_module(m)
        print(f"   {m:<13} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"   {m:<13} FAIL {e}"); sys.exit(1)
import torch
print(f"   torch cuda      {torch.version.cuda}   available={torch.cuda.is_available()}   gpus={torch.cuda.device_count()}")
print(f"   lm_eval CLI     {shutil.which('lm_eval') or 'NOT ON PATH'}")
from lm_eval.tasks import TaskManager
tm = TaskManager()
missing = [t for t in ["mmlu","arc_challenge","hellaswag","truthfulqa_mc2","gsm8k","minerva_math","humaneval"]
           if t not in tm.all_tasks]
print("   tasks           " + ("ok" if not missing else f"MISSING {missing}"))
PY

echo
echo "== done. Activate with:  conda activate $ENV_NAME"
