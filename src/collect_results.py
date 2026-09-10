#!/usr/bin/env python3
"""
collect_results.py — turn the eval tree + training logs into one small bundle.

Reads   <eval_dir>/<label>/<group>/**/results_*.json   (lm-eval output)
        <eval_dir>/models.json                          (label -> subset/epoch)
        <train_dir>/subset_vN/{train_log.jsonl, trainer_state.json,
                               run_metadata.json, checkpoint_epochs.json}
Writes  <results_dir>/results.csv      every metric of every task of every checkpoint
        <results_dir>/summary.md       one table per (task, fewshot): rows = epoch,
                                       columns = subsets + mean + delta vs base
        <results_dir>/train_logs/subset_vN/*   copies of the training logs
        <results_dir>/config_snapshot.txt      the environment variables from config.sh
        <results_dir>/{corpus_metadata.json, subsets_summary.json}

This bundle (a few MB) is everything Sina needs. No weights.
"""
import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

# The "headline" metric per task, in priority order (lm-eval key names).
PRIMARY = {
    "mmlu":           ["acc,none"],
    "arc_challenge":  ["acc_norm,none", "acc,none"],
    "arc_easy":       ["acc_norm,none", "acc,none"],
    "hellaswag":      ["acc_norm,none", "acc,none"],
    "winogrande":     ["acc,none"],
    "truthfulqa_mc2": ["acc,none"],
    "gsm8k":          ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none"],
    "minerva_math":   ["exact_match,none", "exact_match,strict-match"],
    "humaneval":      ["pass@1,create_test", "pass@1,none"],
}


def load_results_file(out_dir: Path):
    files = sorted(out_dir.rglob("results_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--temperature", required=True)
    p.add_argument("--extra_file", action="append", default=[], help="small files to copy into the bundle")
    args = p.parse_args()

    eval_dir, train_dir, results_dir = Path(args.eval_dir), Path(args.train_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_dir / "models.json") as f:
        models = {m["label"]: m for m in json.load(f)}

    # ---------------------------------------------------------------- rows
    rows = []
    for label_dir in sorted(d for d in eval_dir.iterdir() if d.is_dir()):
        label = label_dir.name
        meta = models.get(label, {"subset": None, "epoch": None})
        for group_dir in sorted(d for d in label_dir.iterdir() if d.is_dir()):
            data = load_results_file(group_dir)
            if data is None:
                continue
            fewshot_cfg = data.get("n-shot", {})
            for task, metrics in data.get("results", {}).items():
                if task not in PRIMARY:          # skip mmlu_* / minerva_math_* subtasks
                    continue
                fewshot = fewshot_cfg.get(task, None)
                primary = next((k for k in PRIMARY[task] if k in metrics), None)
                for key, val in metrics.items():
                    if key == "alias" or key.endswith("_stderr,none") or "_stderr," in key:
                        continue
                    if not isinstance(val, (int, float)):
                        continue
                    stderr_key = key.replace(",", "_stderr,", 1)
                    rows.append({
                        "temperature": args.temperature,
                        "label": label,
                        "subset": meta["subset"],
                        "epoch": meta["epoch"],
                        "group": group_dir.name,
                        "task": task,
                        "fewshot": fewshot,
                        "metric": key,
                        "primary": int(key == primary),
                        "value": val,
                        "stderr": metrics.get(stderr_key, ""),
                    })

    fields = ["temperature", "label", "subset", "epoch", "group", "task", "fewshot",
              "metric", "primary", "value", "stderr"]
    with open(results_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"results.csv: {len(rows)} rows")

    # ---------------------------------------------------------------- summary.md
    # table[(task, fewshot)][epoch][subset] = primary value
    table = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["primary"]:
            table[(r["task"], r["fewshot"])][r["epoch"]][r["subset"]] = r["value"]

    subsets = sorted({r["subset"] for r in rows if r["subset"]} - {None})
    lines = [f"# Results — T={args.temperature}", "",
             "Primary metric per task (acc_norm / acc / exact_match strict / pass@1). "
             "Epoch 0 = untouched base model. Δ = mean − base.", ""]
    for (task, fs) in sorted(table):
        lines += [f"## {task} ({fs}-shot)", ""]
        header = "| epoch | " + " | ".join(f"subset_v{s}" for s in subsets) + " | mean | Δ vs base |"
        lines += [header, "|" + "---|" * (len(subsets) + 3)]
        base = table[(task, fs)].get(0, {}).get(0)
        for epoch in sorted(e for e in table[(task, fs)] if e is not None):
            vals = table[(task, fs)][epoch]
            if epoch == 0:
                cells = ["—"] * len(subsets)
                mean = vals.get(0)
            else:
                cells = [f"{vals[s]*100:.2f}" if s in vals else "—" for s in subsets]
                present = [vals[s] for s in subsets if s in vals]
                mean = sum(present) / len(present) if present else None
            mean_s = f"{mean*100:.2f}" if mean is not None else "—"
            delta_s = f"{(mean-base)*100:+.2f}" if (mean is not None and base is not None) else "—"
            lines.append(f"| {epoch} | " + " | ".join(cells) + f" | {mean_s} | {delta_s} |")
        lines.append("")
    with open(results_dir / "summary.md", "w") as f:
        f.write("\n".join(lines))
    print("summary.md written")

    # ---------------------------------------------------------------- training logs
    for run in sorted(train_dir.glob("subset_v*")):
        dst = results_dir / "train_logs" / run.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in ["train_log.jsonl", "trainer_state.json", "run_metadata.json", "checkpoint_epochs.json"]:
            if (run / name).exists():
                shutil.copy2(run / name, dst / name)

    for extra in args.extra_file:
        if os.path.exists(extra):
            shutil.copy2(extra, results_dir / os.path.basename(extra))

    # snapshot of the configuration actually used
    keys = ["MODEL", "TEMPERATURE", "START_TOKEN", "SEED", "TARGET_TOKENS", "NUM_SUBSETS",
            "TOKENS_PER_SUBSET", "LR", "EPOCHS", "SAVE_EVERY_EPOCHS", "PER_DEVICE_BS",
            "EFFECTIVE_BS", "GRAD_ACCUM", "NUM_GPUS", "CUTOFF_LEN", "WARMUP_RATIO",
            "GRADIENT_CHECKPOINTING", "EVAL_SUITE", "HUMANEVAL_MODE", "EVAL_LIMIT", "SMOKE"]
    with open(results_dir / "config_snapshot.txt", "w") as f:
        for k in keys:
            f.write(f"{k}={os.environ.get(k, '')}\n")
    print(f"bundle -> {results_dir}")


if __name__ == "__main__":
    main()
