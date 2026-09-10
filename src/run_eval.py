#!/usr/bin/env python3
"""
run_eval.py — run lm-evaluation-harness (vLLM backend) over many checkpoints,
one job per GPU, skipping anything that already has results.

Input:  --models_json  a list of {"label", "path", "subset", "epoch"}
        (written by stages/4_evaluate.sh; "base" has subset=0, epoch=0)

For every model x task-group, runs

    lm_eval --model vllm --model_args pretrained=<path>,... \
            --tasks <group tasks> --num_fewshot <n> --gen_kwargs <...> \
            --output_path <eval_dir>/<label>/<group>

Task groups
  Suite A (zero-shot): one invocation with all 7 tasks (model loaded once):
      mmlu, arc_challenge, hellaswag, truthfulqa_mc2, gsm8k, minerva_math, humaneval
      (humaneval is split into its own job if HUMANEVAL_MODE=sample)
  Suite B (few-shot): one invocation per task because few-shot counts differ:
      arc_challenge 25, gsm8k 8, hellaswag 5, minerva_math 4

A job is "done" when a results_*.json exists under its output dir. Each job's
stdout/stderr goes to <output dir>/eval.log. Failed jobs are listed at the end
and the script exits non-zero (so the stage is not marked done), but every
other job still runs.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOGLIK_A = ["mmlu", "arc_challenge", "hellaswag", "truthfulqa_mc2"]
GEN_A = ["gsm8k", "minerva_math"]
SUITE_B = [("arc_challenge", 25), ("gsm8k", 8), ("hellaswag", 5), ("minerva_math", 4)]
GREEDY = "temperature=0,top_p=1.0"


def build_groups(suite: str, humaneval_mode: str):
    suite = suite.upper()
    groups = []
    if "A" in suite:
        if humaneval_mode == "greedy":
            groups.append(dict(name="suiteA_fs0", tasks=LOGLIK_A + GEN_A + ["humaneval"],
                               fewshot=0, gen_kwargs=GREEDY, unsafe=True))
        else:
            groups.append(dict(name="suiteA_fs0", tasks=LOGLIK_A + GEN_A,
                               fewshot=0, gen_kwargs=GREEDY, unsafe=False))
            groups.append(dict(name="humaneval_fs0_sample", tasks=["humaneval"], fewshot=0,
                               gen_kwargs="do_sample=True,temperature=0.2,top_p=0.95", unsafe=True))
    if "B" in suite:
        for task, fs in SUITE_B:
            groups.append(dict(name=f"{task}_fs{fs}", tasks=[task], fewshot=fs,
                               gen_kwargs=GREEDY, unsafe=False))
    if not groups:
        sys.exit(f"unknown EVAL_SUITE '{suite}' (use A, B or AB)")
    return groups


def has_results(out_dir: Path) -> bool:
    return out_dir.is_dir() and any(out_dir.rglob("results_*.json"))


def make_cmd(model_path, group, out_dir, args):
    model_args = (f"pretrained={model_path},dtype=bfloat16,tensor_parallel_size=1,"
                  f"gpu_memory_utilization={args.gpu_mem_util},max_model_len={args.max_model_len},"
                  f"seed={args.seed},trust_remote_code=True")
    cmd = ["lm_eval", "--model", "vllm", "--model_args", model_args,
           "--tasks", ",".join(group["tasks"]),
           "--num_fewshot", str(group["fewshot"]),
           "--batch_size", "auto",
           "--gen_kwargs", group["gen_kwargs"],
           "--seed", str(args.seed),
           "--output_path", str(out_dir)]
    if group["unsafe"]:
        cmd.append("--confirm_run_unsafe_code")   # humaneval executes generated code
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models_json", required=True)
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--suite", default="A")
    p.add_argument("--humaneval_mode", default="greedy")
    p.add_argument("--gpus", default="0", help="comma-separated GPU ids")
    p.add_argument("--gpu_mem_util", type=float, default=0.80)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N items (smoke)")
    args = p.parse_args()

    with open(args.models_json) as f:
        models = json.load(f)
    groups = build_groups(args.suite, args.humaneval_mode)
    eval_dir = Path(args.eval_dir)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    # ---- build job list, skipping done ones ----
    jobs, skipped = [], 0
    for m in models:
        for g in groups:
            out_dir = eval_dir / m["label"] / g["name"]
            if has_results(out_dir):
                skipped += 1
                continue
            jobs.append((m, g, out_dir))
    print(f"{len(models)} models x {len(groups)} groups: {len(jobs)} to run, {skipped} already done, "
          f"{len(gpus)} GPUs")
    if not jobs:
        return 0

    # ---- simple GPU pool ----
    running = {}          # gpu -> (Popen, job, log handle, t0)
    pending = list(jobs)
    failed, done = [], 0
    env_base = os.environ.copy()
    env_base["HF_ALLOW_CODE_EVAL"] = "1"

    def launch(gpu, job):
        m, g, out_dir = job
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = make_cmd(m["path"], g, out_dir, args)
        env = dict(env_base, CUDA_VISIBLE_DEVICES=gpu)
        logf = open(out_dir / "eval.log", "w")
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        running[gpu] = (proc, job, logf, time.time())
        print(f"[gpu {gpu}] START {m['label']} / {g['name']}", flush=True)

    while pending or running:
        for gpu in gpus:
            if gpu not in running and pending:
                launch(gpu, pending.pop(0))
        time.sleep(15)
        for gpu in list(running):
            proc, job, logf, t0 = running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            logf.close()
            del running[gpu]
            m, g, out_dir = job
            mins = (time.time() - t0) / 60
            if rc == 0 and has_results(out_dir):
                done += 1
                print(f"[gpu {gpu}] DONE  {m['label']} / {g['name']}  ({mins:.1f} min)  "
                      f"[{done}/{len(jobs)}]", flush=True)
            else:
                failed.append(job)
                print(f"[gpu {gpu}] FAIL  {m['label']} / {g['name']}  rc={rc}  see {out_dir/'eval.log'}",
                      flush=True)

    print(f"finished: {done} ok, {len(failed)} failed")
    if failed:
        for m, g, out_dir in failed:
            print(f"  FAILED {m['label']} / {g['name']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
