#!/usr/bin/env python3
"""
sample_pilecc.py — draw one random ~5M-token training set from Pile-CC for the
OLMo-2-13B generic-text control (the OLMo analogue of the Qwen Common Corpus
baseline). Output drops straight into src/train.py.

Filters, applied in this order to every document:
  1. meta.pile_set_name == "Pile-CC"
  2. MIN_DOC_TOKENS <= OLMo-2 tokens (no special tokens) <= MAX_DOC_TOKENS
     (defaults 50 / 2048; 50 is the synthetic runs' min_tokens)
  3. English: langid with normalized probabilities, lang == "en" and
     prob >= --lang_min_prob
  4. 8-gram decontamination against the 7 Figure-4 benchmarks, with the same
     normalization, n-gram definition, splits, text fields and rule as
     latent-capability-resurfacing/src/audit.py (lowercase, strip punctuation,
     whitespace tokens; drop a document that shares >= 1 8-gram with any
     benchmark). Unlike audit.py, a benchmark that fails to load is a hard
     error, not a silent skip.
Documents that pass all four are kept until the set holds >= --tokens tokens.
Tokens are counted as len(ids) + 1 for the EOS that train.py appends, the same
convention as num_tokens in generate_synthetic.py. Text is .strip()-ed, as the
synthetic samples were.

Randomness (all from --seed, one random.Random): the order in which Pile shards
are read, how many raw documents to skip at the start of the first shard, and a
shuffle buffer over Pile-CC documents. The Pile's shards were shuffled when it
was built, so a random offset in a random shard plus a buffer shuffle gives a
random sample of Pile-CC; different seeds give effectively disjoint sets.

Source: --files '<glob>' for local .jsonl / .jsonl.zst shards (e.g. the Pile
copy used for the Pythia runs); otherwise the HF dataset
monology/pile-uncopyrighted is streamed (nothing is downloaded in full).

Outputs in --out_dir:
  subset_v1.jsonl        one {"text", "num_tokens"} per line (train.py reads "text")
  sampling_report.json   seed, shard, skip, filter counts, token totals
"""
import argparse
import glob
import io
import json
import random
import re
import sys
import time
from pathlib import Path

PILE_HF_REPO = "monology/pile-uncopyrighted"


# ============================================================ normalization
# Verbatim from latent-capability-resurfacing/src/audit.py.
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_ngrams(text: str, n: int) -> set:
    tokens = text.split()
    if len(tokens) < n:
        return set()
    return set(zip(*[tokens[i:] for i in range(n)]))


# ============================================================ benchmarks
# Same splits and text fields as audit.py. Each benchmark lists candidate HF ids:
# the namespaced ids lm-eval uses first (already in HF_HOME after an eval run or
# stages/0_download.sh), then the legacy ids audit.py used.
def _load(candidates, split):
    from datasets import load_dataset
    errors = []
    for path, cfg in candidates:
        try:
            return load_dataset(path, cfg, split=split, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001 — try the next id
            errors.append(f"{path}[{cfg}]: {type(e).__name__}: {e}")
    raise RuntimeError("could not load benchmark from any of:\n  " + "\n  ".join(errors))


MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]


def load_benchmark_texts():
    out = {}
    ds = _load([("allenai/ai2_arc", "ARC-Challenge"), ("ai2_arc", "ARC-Challenge")],
               "test+validation+train")
    out["arc_challenge"] = [ex["question"] + " " + " ".join(ex["choices"]["text"]) for ex in ds]

    ds = _load([("openai/gsm8k", "main"), ("gsm8k", "main")], "test+train")
    out["gsm8k"] = [ex["question"] + " " + ex["answer"] for ex in ds]

    ds = _load([("Rowan/hellaswag", None), ("hellaswag", None)], "validation+train")
    out["hellaswag"] = [ex["activity_label"] + " " + ex["ctx"] + " " + " ".join(ex["endings"])
                        for ex in ds]

    # hails/mmlu_no_train = cais/mmlu without auxiliary_train; it is what lm-eval caches
    ds = _load([("cais/mmlu", "all"), ("hails/mmlu_no_train", "all")], "test+validation+dev")
    out["mmlu"] = [ex["question"] + " " + " ".join(ex["choices"]) for ex in ds]

    ds = _load([("truthfulqa/truthful_qa", "multiple_choice"), ("truthful_qa", "multiple_choice")],
               "validation")
    out["truthfulqa_mc2"] = [ex["question"] + " " + " ".join(ex["mc1_targets"]["choices"]) for ex in ds]

    # hendrycks/competition_math (audit.py's id) is no longer on the Hub; the same
    # problems are in EleutherAI/hendrycks_math (what lm-eval's minerva_math reads).
    try:
        ds = _load([("hendrycks/competition_math", None)], "test+train")
        out["minerva_math"] = [ex["problem"] + " " + ex["solution"] for ex in ds]
    except RuntimeError:
        texts = []
        for cfg in MATH_CONFIGS:
            ds = _load([("EleutherAI/hendrycks_math", cfg)], "test+train")
            texts += [ex["problem"] + " " + ex["solution"] for ex in ds]
        out["minerva_math"] = texts

    ds = _load([("openai/openai_humaneval", None), ("openai_humaneval", None)], "test")
    out["humaneval"] = [ex["prompt"] + " " + ex["canonical_solution"] for ex in ds]
    return out


def build_fingerprints(n: int):
    global_set, sizes = set(), {}
    for name, texts in load_benchmark_texts().items():
        s = set()
        for t in texts:
            s.update(get_ngrams(normalize(t), n))
        global_set |= s
        sizes[name] = {"examples": len(texts), "ngrams": len(s)}
        print(f"  {name:15s} {len(texts):7,} examples -> {len(s):10,} {n}-grams", flush=True)
    print(f"  total unique {n}-grams: {len(global_set):,}", flush=True)
    return global_set, sizes


# ============================================================ Pile source
def list_shards(files_glob):
    if files_glob:
        shards = sorted(glob.glob(files_glob))
    else:
        from huggingface_hub import HfFileSystem
        shards = sorted(HfFileSystem().glob(f"datasets/{PILE_HF_REPO}/train/*.jsonl.zst"))
    if not shards:
        sys.exit(f"ERROR: no Pile shards found ({files_glob or PILE_HF_REPO})")
    return shards


def open_shard(path: str, from_hub: bool):
    if from_hub:
        from huggingface_hub import HfFileSystem
        raw = HfFileSystem().open(path, "rb", block_size=16 * 1024 * 1024)
    else:
        raw = open(path, "rb")
    if path.endswith(".zst"):
        import zstandard
        raw = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def pilecc_stream(shards, from_hub, rng, skip, buffer_size, stats):
    """Yield Pile-CC texts in shuffle-buffer order."""
    pool = []
    for k, shard in enumerate(shards):
        print(f"reading shard {shard}" + (f" (skipping first {skip:,} documents)" if k == 0 else ""),
              flush=True)
        with open_shard(shard, from_hub) as f:
            for i, line in enumerate(f):
                if k == 0 and i < skip:
                    continue
                stats["raw_docs_read"] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                if (rec.get("meta") or {}).get("pile_set_name") != "Pile-CC":
                    continue
                stats["pilecc_docs_seen"] += 1
                text = rec.get("text", "")
                if len(pool) < buffer_size:
                    pool.append(text)
                    continue
                j = rng.randrange(buffer_size)
                out, pool[j] = pool[j], text
                yield out
    rng.shuffle(pool)
    yield from pool


# ============================================================ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", default="allenai/OLMo-2-1124-13B", help="tokenizer used for counting")
    p.add_argument("--tokens", type=int, default=5_000_000)
    p.add_argument("--min_doc_tokens", type=int, default=50)
    p.add_argument("--max_doc_tokens", type=int, default=2048)
    p.add_argument("--lang_min_prob", type=float, default=0.9)
    p.add_argument("--ngram", type=int, default=8)
    p.add_argument("--files", default=None, help="glob of local Pile shards; default: stream HF")
    p.add_argument("--shuffle_buffer", type=int, default=20_000)
    p.add_argument("--max_skip", type=int, default=500_000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    t0 = time.time()

    from langid.langid import LanguageIdentifier, model as langid_model
    from transformers import AutoTokenizer
    lid = LanguageIdentifier.from_modelstring(langid_model, norm_probs=True)
    # fail fast if the language identifier is broken in this environment
    en = lid.classify("The committee will meet on Tuesday to review the annual budget and staffing plan.")
    de = lid.classify("Der Ausschuss trifft sich am Dienstag, um den Jahreshaushalt zu prüfen.")
    if en[0] != "en" or de[0] == "en":
        sys.exit(f"ERROR: langid self-test failed (en -> {en}, de -> {de})")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"building {args.ngram}-gram benchmark fingerprints", flush=True)
    bench, bench_sizes = build_fingerprints(args.ngram)

    from_hub = args.files is None
    shards = list_shards(args.files)
    order = shards[:]
    rng.shuffle(order)
    skip = rng.randrange(args.max_skip + 1) if args.max_skip > 0 else 0

    stats = {k: 0 for k in ["raw_docs_read", "bad_json", "pilecc_docs_seen", "considered", "empty",
                            "too_short", "too_long", "not_english", "contaminated", "kept"]}
    kept_tokens = 0
    tmp = out_dir / "subset_v1.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as fout:
        for text in pilecc_stream(order, from_hub, rng, skip, args.shuffle_buffer, stats):
            stats["considered"] += 1
            if stats["considered"] % 2000 == 0:
                print(f"  considered {stats['considered']:,}  kept {stats['kept']:,}  "
                      f"tokens {kept_tokens:,}/{args.tokens:,}  ({(time.time()-t0)/60:.1f} min)", flush=True)
            text = text.strip()
            if not text:
                stats["empty"] += 1
                continue
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            if n < args.min_doc_tokens:
                stats["too_short"] += 1
                continue
            if n > args.max_doc_tokens:
                stats["too_long"] += 1
                continue
            lang, prob = lid.classify(text)
            if lang != "en" or prob < args.lang_min_prob:
                stats["not_english"] += 1
                continue
            if get_ngrams(normalize(text), args.ngram) & bench:
                stats["contaminated"] += 1
                continue
            fout.write(json.dumps({"text": text, "num_tokens": n + 1}, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            kept_tokens += n + 1
            if kept_tokens >= args.tokens:
                break
    if kept_tokens < args.tokens:
        sys.exit(f"ERROR: Pile ran out at {kept_tokens:,} tokens (< {args.tokens:,})")
    tmp.replace(out_dir / "subset_v1.jsonl")

    report = {
        "seed": args.seed, "source": args.files or f"hf://{PILE_HF_REPO}",
        "first_shard": order[0], "skip_raw_docs": skip, "shuffle_buffer": args.shuffle_buffer,
        "tokenizer": args.model, "token_budget": args.tokens, "tokens_kept": kept_tokens,
        "mean_tokens_per_doc": kept_tokens / stats["kept"],
        "filters": {"min_doc_tokens": args.min_doc_tokens, "max_doc_tokens": args.max_doc_tokens,
                    "lang": "en", "lang_min_prob": args.lang_min_prob, "ngram": args.ngram},
        "benchmarks": bench_sizes, "counts": stats,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(out_dir / "sampling_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
