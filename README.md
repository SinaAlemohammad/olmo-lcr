# olmo-lcr — prompt-free self-training on OLMo-2-13B

Continued pretraining of a base LM on text it sampled from itself, then
evaluation of every checkpoint. One node runs one sampling temperature end to end.

```
[stage 1] sample ~10M tokens from the model, unconditionally, at temperature T   (vLLM, 8 GPUs, ~30 min)
[stage 2] take a random 5M-token subset of that                                   (seconds)
[stage 3] one training run, 40 epochs, weights saved every 2 epochs (20 ckpts)   (~6-8 h)
[stage 4] evaluate base model + 20 checkpoints, write a small results bundle      (~4-7 h)
```

Roughly one day per node. The bundle at the end is a few MB and contains
every number needed; no checkpoints have to leave the machine.

## Requirements

- One node with 8 x A100-80GB (or similar), NVIDIA driver >= 525.
- conda on PATH.
- ~600 GB of **local** disk per node (20 checkpoints x ~26 GB + model cache).
- Internet for the first run (model weights ~26 GB + benchmark datasets), or
  run `stages/0_download.sh` once on a machine that has it (see "No internet").

## Setup (once)

```bash
bash env/setup_train_env.sh     # conda env lcr-train : torch, vllm, deepspeed, transformers
bash env/setup_eval_env.sh      # conda env lcr-eval  : torch, vllm, lm-eval[math]
```

Two environments on purpose: lm-eval's math extras (sympy / antlr4) and the
training stack do not coexist well. Both scripts end with a smoke test; if it
prints `ok` for everything you're set. Versions are pinned in the header of
each script; change them there if your cluster requires a different
torch/CUDA build.

## Smoke test (~30 min) — do this first

```bash
SMOKE=1 TEMPERATURE=1.25 bash run_all.sh
```

Same code path with tiny sizes (120k tokens generated, one 60k-token subset,
2 epochs, 8 items per benchmark). Output goes to `work/T1_25_smoke/`.
If it ends with `ALL DONE` and `work/T1_25_smoke/results/summary.md` exists,
the infrastructure works. To smoke-test with less GPU time you can also use a
smaller model: `SMOKE=1 MODEL=allenai/OLMo-2-1124-7B bash run_all.sh`.

## Full run (three nodes)

```bash
TEMPERATURE=0.75 bash run_all.sh   # node 1
TEMPERATURE=1.0  bash run_all.sh   # node 2
TEMPERATURE=1.25 bash run_all.sh   # node 3
```

Run it under `nohup`, `tmux`, or your scheduler; it prints its own progress
and also logs to `work/T<temp>/logs/run_all.log`.

Useful overrides (all in `config.sh`, all overridable from the shell):

| variable | default | meaning |
|---|---|---|
| `WORK_DIR` | `./work` | where everything is written — put it on fast local disk |
| `HF_HOME` | `$WORK_DIR/hf_cache` | HuggingFace cache (model + datasets); can be shared across nodes |
| `DELETE_CKPT_AFTER_EVAL` | 0 | 1 = delete the checkpoints once all their evals succeeded |
| `NUM_SUBSETS` | 1 | more than 1 = several disjoint 5M subsets, trained and evaluated in sequence |
| `HF_OFFLINE` | 0 | 1 = never touch the network (after `stages/0_download.sh`) |
| `EVAL_SUITE` | `A` | `A` zero-shot 7 tasks, `B` few-shot 4 tasks, `AB` both |
| `PER_DEVICE_BS` | 2 | lower to 1 if training OOMs (grad accumulation is recomputed to keep batch 64; see below) |

Everything else (learning rate, epochs, temperatures, token budgets) is the
paper's protocol and should stay as is.

## Where things end up

```
work/T1_25/
├── 1_synthetic/   corpus.jsonl, corpus_metadata.json           (~50 MB, ~10M tokens)
├── 2_subsets/     subset_v1.jsonl                               (~25 MB, 5M tokens)
├── 3_train/
│   └── subset_v1/ epoch-02/ epoch-04/ … epoch-40/               (20 x ~26 GB, HF format)
│                  train_log.jsonl  trainer_state.json  run_metadata.json  checkpoint_epochs.json
├── 4_evals/       <label>/<task-group>/…/results_*.json          (lm-eval raw output)
├── results/       results.csv  summary.md  train_logs/  …        <- the bundle contents
├── logs/          one log per stage / worker
├── .markers/      <stage>.done files (see "Resuming")
└── results_bundle_T1_25.tar.gz                                    <- SEND THIS BACK
```

## What to send back

`work/T<temp>/results_bundle_T<temp>.tar.gz` for each temperature. It contains:

- `results.csv` — every metric of every task for the base model and all 20 checkpoints
- `summary.md` — one table per task: accuracy by epoch, delta vs base
- `train_logs/subset_v1/train_log.jsonl` — loss / lr / grad-norm every 10 steps
- `corpus_metadata.json`, `subsets_summary.json`, `config_snapshot.txt`

If you want to send an early sanity check before the eval finishes,
`work/T<temp>/3_train/subset_v1/train_log.jsonl` is enough. A healthy run:
loss starts around the model's own per-token NLL on its samples (higher for
T=1.25 than for T=0.75), decreases smoothly, no spikes or NaNs. A full run is
about 1,560 optimizer steps (≈39 per epoch), so "after 1k iterations" is
roughly epoch 26.

## Adapting the I/O to a non-POSIX store

Every stage reads and writes on local disk under `WORK_DIR`. The only contact
with remote storage is through three shell functions in **`storage.sh`**:

```
storage_upload   <local_path> <remote_key>
storage_download <remote_key> <local_path>
storage_exists   <remote_key>
```

They default to no-ops. Replace their bodies with your download/upload tool
and nothing else changes. They are called only at stage boundaries:

| when | what | size |
|---|---|---|
| after stage 1 | `corpus.jsonl` | ~50 MB |
| after stage 2 | `2_subsets/` | ~25 MB |
| after training | `train_log.jsonl` etc. (checkpoints only if `UPLOAD_CHECKPOINTS=1`) | KB (or 520 GB) |
| after stage 4 | `results_bundle_T<temp>.tar.gz` | a few MB |

Because evaluation runs on the same node right after training, the
checkpoints never need to be uploaded. `DELETE_CKPT_AFTER_EVAL=1` removes
them once every eval job has succeeded.

## Resuming / re-running

Each stage writes `work/T<temp>/.markers/<stage>.done` when it finishes and
skips itself next time. If a node dies, just launch the same command again:
it resumes at the first unfinished stage. Inside stage 3 a half-finished
training run is restarted from scratch (checkpoints are weights-only, there
is no optimizer state to resume from). Inside stage 4 only the
(checkpoint, task) pairs without results are re-run.

To force a stage to redo its work, delete its marker and its output directory,
e.g. `rm -rf work/T1_25/.markers/4_evaluate.done work/T1_25/4_evals`.

## No internet on the compute nodes

On a machine with internet and both conda envs, with `HF_HOME` pointing at a
location the compute nodes can read:

```bash
bash stages/0_download.sh
```

Then run everything with `HF_OFFLINE=1`.

## Modifying the code

- `config.sh` — all knobs, heavily commented.
- `stages/*.sh` — thin drivers, one per stage; each is runnable on its own.
- `src/generate_synthetic.py` — vLLM sampler. The prompt is a single token id
  (EOS for OLMo, which has no BOS), `min_tokens=50`, `max_tokens=1024`, no
  top-p/top-k, `stop_token_ids=[]`.
- `src/train.py` — plain HF `Trainer` + DeepSpeed ZeRO-3. Saving every N
  epochs is a 30-line `TrainerCallback` at the top of the file.
- `src/run_eval.py` — job pool around the `lm_eval` CLI, one process per GPU.
- `src/collect_results.py` — turns lm-eval JSONs into `results.csv` / `summary.md`.

## If training runs out of GPU memory: batch size and gradient accumulation

The paper's protocol fixes the **effective batch at 64 sequences of 2048
tokens per optimizer step**. That number must not change. What you can change
freely is how those 64 sequences are split across GPUs and micro-steps:

```
effective batch (64)  =  PER_DEVICE_BS  x  GRAD_ACCUM  x  NUM_GPUS
```

`config.sh` computes `GRAD_ACCUM = 64 / (PER_DEVICE_BS x NUM_GPUS)` for you, so
lowering `PER_DEVICE_BS` automatically raises the accumulation and the
optimizer sees exactly the same batch. The only cost is wall-clock time.

| GPUs | `PER_DEVICE_BS` | `GRAD_ACCUM` (auto) | GPU memory | speed |
|---|---|---|---|---|
| 8 | 2 (default) | 4 | highest | fastest |
| 8 | 1 | 8 | ~half the activation memory | slightly slower |
| 4 | 2 | 8 | same per GPU as row 1 | 2x slower |
| 4 | 1 | 16 | | |

What is using the memory on a 13B model with ZeRO-3 over 8 GPUs: the
parameters, gradients and Adam states are sharded, ~26 GB per GPU regardless
of batch size. Everything else is activations, which scale with
`PER_DEVICE_BS x CUTOFF_LEN`. Gradient checkpointing is already on
(`GRADIENT_CHECKPOINTING=1`), which is what makes `PER_DEVICE_BS=2` fit in
80 GB.

Steps, in order. Stop at the first one that works.

1. **`PER_DEVICE_BS=1`** — this is almost always enough.
   ```bash
   PER_DEVICE_BS=1 TEMPERATURE=1.25 bash run_all.sh
   ```
   The summary printed at start should show `batch: 1/gpu x 8 accum x 8 gpus = 64`.

2. **Optimizer states to CPU RAM** — if `PER_DEVICE_BS=1` still OOMs (e.g. 40 GB
   GPUs). Needs ~160 GB free host RAM on the node; roughly 2x slower.
   ```bash
   PER_DEVICE_BS=1 DS_CONFIG=ds_z3_offload.json TEMPERATURE=1.25 bash run_all.sh
   ```

3. **Fewer GPUs than 8** — nothing to do; `NUM_GPUS` is detected and
   `GRAD_ACCUM` scales up. Just make sure `PER_DEVICE_BS x NUM_GPUS` divides 64
   (1, 2, 4, 8 GPUs are fine; 3 or 6 are not). `config.sh` prints a `WARNING`
   at start if the product is not 64.

Do **not** change `EFFECTIVE_BS` or `CUTOFF_LEN` to fix memory; both change
the experiment.

Where to verify the batch actually used: the `batch:` line of the
configuration summary at the top of `logs/run_all.log`, and
`3_train/subset_v1/run_metadata.json` → `"effective_batch": 64`.

If you must set the accumulation by hand for some reason, `GRAD_ACCUM=<n>`
overrides the computed value, and the warning above tells you if the product
no longer equals 64.

## Troubleshooting

- **Training OOM** → see the section above: `PER_DEVICE_BS=1`, then `DS_CONFIG=ds_z3_offload.json`.
- **vLLM OOM at generation** → lower `GEN_GPU_MEM_UTIL` in `config.sh`.
- **`conda env 'lcr-train' not found`** → run the two `env/setup_*_env.sh` scripts.
- **A single eval job failed** → look at `work/T<temp>/4_evals/<label>/<group>/eval.log`,
  fix, re-run `bash stages/4_evaluate.sh` (only the failed pairs run again).
- **Driver too old for the torch build** → change `VLLM_VER` / add a torch
  index URL in `env/setup_*_env.sh`; the code needs `transformers >= 4.47`
  and `vllm >= 0.6.4` for OLMo-2, nothing else version-specific.
