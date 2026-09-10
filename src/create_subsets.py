#!/usr/bin/env python3
"""
create_subsets.py — cut K non-overlapping training subsets from corpus.jsonl.

Procedure
  1. read every sample (text + its exact token count from generation)
  2. shuffle the sample order once with --seed
  3. walk the shuffled order: add samples to subset 1 until it holds at least
     --tokens_per_subset tokens, then continue (from where subset 1 stopped)
     into subset 2, etc.  Subsets are therefore disjoint by construction.
  4. fail loudly if the corpus runs out before all K subsets are full

Outputs (in --out_dir)
  subset_v1.jsonl ... subset_vK.jsonl     one {"text": ...} per line
  subsets_summary.json                    sizes, seed, sample counts
"""
import argparse
import json
import random
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="corpus.jsonl from stage 1")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_subsets", type=int, default=3)
    p.add_argument("--tokens_per_subset", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.corpus}")
    samples = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                samples.append((rec["text"], int(rec["num_tokens"])))
    total = sum(n for _, n in samples)
    need = args.num_subsets * args.tokens_per_subset
    print(f"  {len(samples):,} samples, {total:,} tokens; need {need:,} for "
          f"{args.num_subsets} x {args.tokens_per_subset:,}")
    if total < need:
        sys.exit(f"ERROR: corpus too small ({total:,} < {need:,}). "
                 f"Raise TARGET_TOKENS / GEN_SAMPLES_PER_GPU_ROUND and rerun stage 1.")

    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)

    ptr = 0
    summary = {"seed": args.seed, "tokens_per_subset": args.tokens_per_subset,
               "corpus_samples": len(samples), "corpus_tokens": total, "subsets": []}
    for k in range(1, args.num_subsets + 1):
        path = out_dir / f"subset_v{k}.jsonl"
        n_samp, n_tok = 0, 0
        with open(path, "w", encoding="utf-8") as f:
            while n_tok < args.tokens_per_subset:
                if ptr >= len(order):
                    sys.exit(f"ERROR: ran out of samples while filling subset {k}")
                text, n = samples[order[ptr]]
                ptr += 1
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n_samp += 1
                n_tok += n
        summary["subsets"].append({"subset": k, "file": path.name,
                                   "num_samples": n_samp, "num_tokens": n_tok})
        print(f"  subset_v{k}: {n_samp:,} samples, {n_tok:,} tokens")

    with open(out_dir / "subsets_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
