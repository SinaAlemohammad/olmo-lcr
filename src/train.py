#!/usr/bin/env python3
"""
train.py — continued pretraining of a base LM on ONE synthetic subset.

Plain HuggingFace Trainer + DeepSpeed ZeRO-3, full-parameter, bf16.
Launched by stages/3_train.sh as

    torchrun --standalone --nproc_per_node=<NUM_GPUS> src/train.py \
        --model allenai/OLMo-2-1124-13B --train_file .../subset_v1.jsonl \
        --output_dir .../3_train/subset_v1 --deepspeed configs/ds_z3.json ...

What it does
  1. tokenize every sample, append EOS, concatenate everything, cut into
     fixed blocks of --cutoff_len tokens (classic "group_texts" packing; the
     ragged tail is dropped)
  2. train with AdamW, cosine schedule, linear warmup over --warmup_ratio of
     the steps, decaying to 0. Optimizer defaults are HF's (betas 0.9/0.999,
     eps 1e-8, weight_decay 0, grad clip 1.0) — same as the paper.
  3. save WEIGHTS ONLY (no optimizer state) at the end of every
     --save_every_epochs epochs. With 40 epochs / every 2 you get 20 folders:

         <output_dir>/epoch-02/  epoch-04/ ... epoch-40/

     each a normal HF checkpoint (config.json, model-*.safetensors,
     tokenizer files, trainer_state.json with the loss history so far).

Loss logging
  <output_dir>/train_log.jsonl   one line per logging step: step, epoch,
                                 loss, learning_rate, grad_norm
  <output_dir>/trainer_state.json  full HF log history at the end

Reproducibility note: DeepSpeed ZeRO-3 all-reduce order is not bitwise
deterministic across runs; expect tiny loss differences between identical
launches. This is normal.
"""
import argparse
import json
import math
import os
import shutil
import time
from itertools import chain

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local path of the BASE model")
    p.add_argument("--train_file", required=True, help="subset_vN.jsonl with a 'text' field per line")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--deepspeed", required=True, help="DeepSpeed json config (configs/ds_z3.json)")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--save_every_epochs", type=int, default=2)
    p.add_argument("--per_device_bs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--cutoff_len", type=int, default=2048)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--num_proc", type=int, default=8, help="CPU processes for tokenization")
    return p.parse_args()


# --------------------------------------------------------------------------- callbacks
class SaveEveryNEpochs(TrainerCallback):
    """
    Ask the Trainer for a checkpoint at the end of every N-th epoch, then rename
    checkpoint-<global_step> -> epoch-<NN> so nobody downstream has to do step
    arithmetic. Also records the mapping in checkpoint_epochs.json.
    """

    def __init__(self, every: int, output_dir: str):
        self.every = every
        self.output_dir = output_dir
        self.map_path = os.path.join(output_dir, "checkpoint_epochs.json")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch))
        if epoch % self.every == 0:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch))
        src = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        dst = os.path.join(self.output_dir, f"epoch-{epoch:02d}")
        # every rank reaches on_save; make sure rank 0 finished writing first
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if state.is_world_process_zero:
            if os.path.isdir(src):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                os.rename(src, dst)
                mapping = {}
                if os.path.exists(self.map_path):
                    with open(self.map_path) as f:
                        mapping = json.load(f)
                mapping[f"epoch-{epoch:02d}"] = {"global_step": state.global_step, "epoch": state.epoch}
                with open(self.map_path, "w") as f:
                    json.dump(mapping, f, indent=2)
                print(f"[save] epoch {epoch} -> {dst}")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return control


class JsonlLossLogger(TrainerCallback):
    """Append every Trainer log record (loss, lr, grad_norm, epoch) to train_log.jsonl."""

    def __init__(self, path: str):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            rec = {"step": state.global_step, "epoch": state.epoch, "time": time.time()}
            rec.update(logs)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return control


# --------------------------------------------------------------------------- data
def build_dataset(train_file: str, tokenizer, block: int, num_proc: int):
    raw = load_dataset("json", data_files=train_file, split="train")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no EOS token; cannot delimit documents")

    def tokenize(batch):
        enc = tokenizer(batch["text"], add_special_tokens=False)
        return {"input_ids": [ids + [eos] for ids in enc["input_ids"]]}

    tokenized = raw.map(tokenize, batched=True, num_proc=num_proc,
                        remove_columns=raw.column_names, desc="tokenize")

    def group(batch):
        # concatenate this batch of documents and cut into fixed blocks
        concat = list(chain.from_iterable(batch["input_ids"]))
        total = (len(concat) // block) * block
        blocks = [concat[i:i + block] for i in range(0, total, block)]
        return {"input_ids": blocks, "labels": [b[:] for b in blocks]}

    lm = tokenized.map(group, batched=True, batch_size=1000, num_proc=num_proc, desc="pack")
    return lm


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # TrainingArguments must exist BEFORE the model is loaded: the DeepSpeed
    # integration reads the ZeRO-3 config here and then loads the model
    # already partitioned across GPUs (no full 13B copy per rank).
    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        deepspeed=args.deepspeed,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="no",            # saving is driven by SaveEveryNEpochs
        save_only_model=True,          # weights only: ~26 GB per 13B checkpoint, no optimizer state
        save_safetensors=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        ddp_timeout=10800,
        disable_tqdm=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # not used (packed data) but keeps HF quiet

    # rank 0 tokenizes and packs; other ranks wait, then read the cache
    with targs.main_process_first(desc="tokenize+pack"):
        train_ds = build_dataset(args.train_file, tokenizer, args.cutoff_len, args.num_proc)

    n_blocks = len(train_ds)
    if targs.process_index == 0:
        world = max(1, targs.world_size)
        per_rank = math.ceil(n_blocks / world)
        micro_per_epoch = math.ceil(per_rank / args.per_device_bs)
        est_updates_per_epoch = math.ceil(micro_per_epoch / args.grad_accum)
        meta = {
            "model": args.model, "train_file": args.train_file,
            "cutoff_len": args.cutoff_len, "num_blocks": n_blocks,
            "num_tokens_used": n_blocks * args.cutoff_len,
            "world_size": world, "per_device_bs": args.per_device_bs,
            "grad_accum": args.grad_accum,
            "effective_batch": args.per_device_bs * args.grad_accum * world,
            "est_updates_per_epoch": est_updates_per_epoch,
            "est_total_updates": est_updates_per_epoch * args.epochs,
            "lr": args.lr, "epochs": args.epochs, "save_every_epochs": args.save_every_epochs,
            "warmup_ratio": args.warmup_ratio, "seed": args.seed,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
        }
        with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(json.dumps(meta, indent=2))

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False   # incompatible with gradient checkpointing

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=default_data_collator,
        processing_class=tokenizer,     # so every checkpoint folder also gets the tokenizer
        callbacks=[
            SaveEveryNEpochs(args.save_every_epochs, args.output_dir),
            JsonlLossLogger(os.path.join(args.output_dir, "train_log.jsonl")),
        ],
    )

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))
        with open(os.path.join(args.output_dir, "TRAIN_DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        print("training finished")


if __name__ == "__main__":
    main()
