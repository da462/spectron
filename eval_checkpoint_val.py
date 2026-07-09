#!/usr/bin/env python3
"""Evaluate a Spectron/TitanGPT checkpoint on binary validation shards."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from low_rank_linear import replace_linear_with_lowrank
from simple_gpt_dataloader import BinaryDataset
from titan_gpt import TitanGPT, TitanModelArgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint validation loss")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_files", required=True)
    parser.add_argument("--run_mode", choices=["fullrank", "lowrank_all", "lowrank_attention", "lowrank_ffn"], required=True)
    parser.add_argument("--low_rank_ratio", type=float, default=0.25)
    parser.add_argument("--low_rank_w2_same_rank_as_w1w3", action="store_true")
    parser.add_argument("--max_val_samples", type=int, default=12208)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=100,
        help="Print eval progress every N batches; set 0 to disable.",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def lowrank_exclusions(run_mode: str) -> list[str]:
    if run_mode == "fullrank":
        return []
    if run_mode == "lowrank_all":
        return ["tok_embeddings", "output"]
    if run_mode == "lowrank_attention":
        return ["tok_embeddings", "output", "feed_forward"]
    if run_mode == "lowrank_ffn":
        return ["tok_embeddings", "output", "attention"]
    raise ValueError(f"Unsupported run_mode: {run_mode}")


def build_model(
    checkpoint: dict,
    run_mode: str,
    low_rank_ratio: float,
    low_rank_w2_same_rank_as_w1w3: bool,
) -> TitanGPT:
    model_args = dict(checkpoint["model_args"])
    model_args["use_flex_attn"] = True
    args = TitanModelArgs(**model_args)
    model = TitanGPT(args)
    if run_mode != "fullrank":
        model = replace_linear_with_lowrank(
            model,
            rank_ratio=low_rank_ratio,
            method="random",
            exclude_modules=lowrank_exclusions(run_mode),
            disable_c=True,
            w2_same_rank_as_w1w3=low_rank_w2_same_rank_as_w1w3,
        )
    state = checkpoint["model_state_dict"]
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this eval script")

    device = torch.device("cuda")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(
        checkpoint,
        args.run_mode,
        args.low_rank_ratio,
        args.low_rank_w2_same_rank_as_w1w3,
    ).to(device)
    model.eval()

    dataset = BinaryDataset(
        args.val_files,
        args.seq_len,
        rank=0,
        world_size=1,
        device="cpu",
        max_samples=args.max_val_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0
    start_time = time.monotonic()
    total_batches = len(loader)
    print(
        "Starting eval: "
        f"batches={total_batches}, batch_size={args.batch_size}, "
        f"seq_len={args.seq_len}, max_val_samples={args.max_val_samples}",
        flush=True,
    )
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            if args.bf16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids, input_batch=input_ids)
                    loss = F.cross_entropy(
                        logits.float().view(-1, logits.size(-1)),
                        labels.view(-1),
                        reduction="sum",
                    )
            else:
                logits = model(input_ids, input_batch=input_ids)
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )
            total_loss += float(loss.item())
            total_tokens += int(labels.numel())
            if (
                args.progress_interval > 0
                and (batch_idx % args.progress_interval == 0 or batch_idx == total_batches)
            ):
                elapsed = time.monotonic() - start_time
                running_loss = total_loss / max(total_tokens, 1)
                print(
                    "Eval progress: "
                    f"batch={batch_idx}/{total_batches}, "
                    f"tokens={total_tokens:,}, "
                    f"loss={running_loss:.6f}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    val_loss = total_loss / total_tokens
    result = {
        "checkpoint": args.checkpoint,
        "run_mode": args.run_mode,
        "low_rank_ratio": args.low_rank_ratio,
        "max_val_samples": args.max_val_samples,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "tokens": total_tokens,
        "val_loss": val_loss,
        "perplexity": math.exp(val_loss),
        "checkpoint_step": checkpoint.get("step"),
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
