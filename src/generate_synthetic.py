#!/usr/bin/env python3
"""
generate_synthetic.py — unconditional sampling from a base LM with vLLM.

Three modes (chosen by flags):

  (default)  generate one shard on the current GPU
             writes  <out_dir>/shard_r<round>_w<worker>.jsonl
  --count    print the total number of tokens across all shards (an integer,
             nothing else — the bash driver parses it)
  --merge    concatenate all shards into <out_dir>/corpus.jsonl and write
             <out_dir>/corpus_metadata.json

Design notes
  * The prompt is passed as TOKEN IDS ([start_id]), not as a string. Passing
    the BOS/EOS *string* makes vLLM tokenize it and prepend another special
    token, i.e. a double start token. Token ids avoid that.
  * stop_token_ids=[] overrides any extra stop tokens a model's
    generation_config.json declares; vLLM's min_tokens only blocks EOS, so
    those extras would otherwise end samples before MIN_TOKENS.
  * Token counts come from len(output.token_ids), i.e. the model's own
    tokenizer, including the final EOS. Training appends EOS to each sample
    too, so subset token budgets match what the trainer sees.
  * Each worker gets its own seed: SEED + round*100000 + worker*1000.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path


def resolve_start_token(tokenizer, kind: str):
    """Return (token_id, human_readable_name) for the requested start token."""
    kind = kind.strip().lower()
    if kind == "eos":
        tid = tokenizer.eos_token_id
        name = tokenizer.eos_token
    elif kind == "bos":
        tid = tokenizer.bos_token_id
        name = tokenizer.bos_token
    elif kind == "newline":
        ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'"\\n" is not a single token for this tokenizer: {ids}')
        tid, name = ids[0], "\\n"
    elif kind.isdigit():
        tid, name = int(kind), f"id:{kind}"
    else:
        raise ValueError(f"unknown START_TOKEN '{kind}' (use eos | bos | newline | <int>)")
    if tid is None:
        raise ValueError(f"tokenizer has no '{kind}' token (bos_token={tokenizer.bos_token!r}, "
                         f"eos_token={tokenizer.eos_token!r}). Pick another START_TOKEN.")
    return tid, name


def shard_paths(out_dir: Path):
    return sorted(Path(p) for p in glob.glob(str(out_dir / "shard_r*_w*.jsonl")))


def count_tokens(out_dir: Path) -> int:
    total = 0
    for p in shard_paths(out_dir):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += json.loads(line)["num_tokens"]
    return total


def generate(args):
    # Imported here so --count / --merge work without a GPU.
    from vllm import LLM, SamplingParams

    seed = args.seed + args.round * 100_000 + args.worker_id * 1_000
    out_path = Path(args.out_dir) / f"shard_r{args.round}_w{args.worker_id}.jsonl"
    if out_path.exists():
        print(f"shard {out_path.name} already exists, skipping")
        return

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        seed=seed,
        max_model_len=args.max_tokens + 8,      # 1 prompt token + max_tokens + slack
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    start_id, start_name = resolve_start_token(tokenizer, args.start_token)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,               # no nucleus truncation (paper)
        top_k=-1,                # no top-k truncation (paper)
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        stop_token_ids=[],       # see module docstring
        n=1,
        skip_special_tokens=True,
    )

    print("=" * 70)
    print(f"model={args.model}  T={args.temperature}  start_token={start_name} (id {start_id})")
    print(f"round={args.round} worker={args.worker_id} seed={seed} samples={args.num_samples}")
    print(f"min_tokens={args.min_tokens} max_tokens={args.max_tokens}")
    print("=" * 70)

    prompts = [{"prompt_token_ids": [start_id]}] * args.num_samples
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    kept, dropped, total_tokens = 0, 0, 0
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for o in outputs:
            comp = o.outputs[0]
            text = comp.text.strip()
            n_tok = len(comp.token_ids)
            if not text or n_tok < args.min_tokens:
                dropped += 1
                continue
            f.write(json.dumps({"text": text, "num_tokens": n_tok}, ensure_ascii=False) + "\n")
            kept += 1
            total_tokens += n_tok
    os.replace(tmp_path, out_path)   # atomic: a shard is either complete or absent

    meta = {
        "model": args.model, "temperature": args.temperature,
        "start_token": start_name, "start_token_id": start_id,
        "round": args.round, "worker_id": args.worker_id, "seed": seed,
        "num_samples_requested": args.num_samples, "num_kept": kept, "num_dropped": dropped,
        "total_tokens": total_tokens,
        "mean_tokens_per_sample": (total_tokens / kept) if kept else 0.0,
    }
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"kept {kept} samples ({dropped} dropped), {total_tokens:,} tokens -> {out_path.name}")


def merge(args):
    import numpy as np
    out_dir = Path(args.out_dir)
    shards = shard_paths(out_dir)
    if not shards:
        sys.exit("no shards to merge")

    lengths = []
    corpus_path = out_dir / "corpus.jsonl"
    tmp_path = out_dir / "corpus.jsonl.tmp"
    with open(tmp_path, "w", encoding="utf-8") as out:
        for p in shards:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    lengths.append(rec["num_tokens"])
                    out.write(line if line.endswith("\n") else line + "\n")
    os.replace(tmp_path, corpus_path)

    lengths = np.asarray(lengths)
    meta = {
        "model": args.model, "temperature": args.temperature,
        "start_token": args.start_token, "seed": args.seed,
        "num_shards": len(shards),
        "num_samples": int(len(lengths)),
        "total_tokens": int(lengths.sum()),
        "mean_tokens_per_sample": float(lengths.mean()),
        "std_tokens_per_sample": float(lengths.std()),
        "min_tokens": int(lengths.min()), "max_tokens": int(lengths.max()),
        "frac_hit_max_len": float((lengths >= args.max_tokens).mean()),
    }
    with open(out_dir / "corpus_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--count", action="store_true")
    p.add_argument("--merge", action="store_true")
    # generation / metadata args
    p.add_argument("--model", default="allenai/OLMo-2-1124-13B")
    p.add_argument("--temperature", type=float, default=1.25)
    p.add_argument("--start_token", default="eos")
    p.add_argument("--num_samples", type=int, default=6000)
    p.add_argument("--min_tokens", type=int, default=50)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--gpu_mem_util", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--worker_id", type=int, default=0)
    args = p.parse_args()

    if args.count:
        print(count_tokens(Path(args.out_dir)))   # the ONLY output in this mode
    elif args.merge:
        merge(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
