#!/usr/bin/env python3
"""Create the immutable first-sequence batch used by mechanistic runs."""

from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simple_gpt_dataloader import _load_data_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-files", required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(args.train_files))
    if not files:
        raise FileNotFoundError(f"No files match {args.train_files!r}")
    required = args.batch_size * args.sequence_length + 1
    tokens = _load_data_shard(Path(files[0]))
    if tokens.numel() < required:
        raise ValueError(f"First shard has {tokens.numel()} tokens; need {required}")
    sequences = []
    labels = []
    for index in range(args.batch_size):
        start = index * args.sequence_length
        window = tokens[start : start + args.sequence_length + 1]
        sequences.append(window[:-1].to(torch.int32))
        labels.append(window[1:].to(torch.int64))
    input_ids = torch.stack(sequences)
    label_tensor = torch.stack(labels)
    digest = hashlib.sha256(
        input_ids.numpy().tobytes() + label_tensor.numpy().tobytes()
    ).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"input_ids": input_ids, "labels": label_tensor, "sha256": digest},
        output,
    )
    print(f"saved={output} shape={tuple(input_ids.shape)} sha256={digest}")


if __name__ == "__main__":
    main()
