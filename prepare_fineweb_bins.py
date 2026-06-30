"""
Convert FineWeb text shards to Spectron/modded-nanogpt uint16 .bin shards.

This is intended for JZ-local FineWeb assets:

  DATASET_PATH=/lustre/fsmisc/dataset/HuggingFace/fineweb/data
  TOKENIZER_PATH=/lustre/fswork/projects/rech/qps/ulf36rc/assets/hf/Llama-2-7b-hf

The output format is the same format consumed by simple_gpt_dataloader.py:
256 int32 header values followed by uint16 token IDs.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


MAGIC = 20240520
VERSION = 1


def write_datafile(filename: Path, toks: np.ndarray) -> None:
    """Write one binary shard with the modded-nanogpt header."""
    assert toks.dtype == np.uint16
    assert len(toks) < 2**31, "token count too large"
    filename.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(toks)
    print(f"Writing {len(toks):,} tokens to {filename}")
    with filename.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def find_local_files(dataset_path: Path, file_limit: int | None) -> tuple[str, list[str]]:
    """Find a local FineWeb shard format under dataset_path."""
    parquet_files = sorted(glob.glob(str(dataset_path / "**" / "*.parquet"), recursive=True))
    if parquet_files:
        return "parquet", parquet_files[:file_limit]

    arrow_files = sorted(glob.glob(str(dataset_path / "**" / "*.arrow"), recursive=True))
    if arrow_files:
        return "arrow", arrow_files[:file_limit]

    json_files = sorted(glob.glob(str(dataset_path / "**" / "*.json*"), recursive=True))
    if json_files:
        return "json", json_files[:file_limit]

    raise FileNotFoundError(f"No parquet, arrow, or json shards found under {dataset_path}")


def load_text_dataset(args: argparse.Namespace) -> Iterable[dict]:
    """Load either JZ-local shards or a remote HF dataset in streaming mode."""
    dataset_path = Path(args.dataset_path).expanduser()
    if dataset_path.exists():
        try:
            return load_from_disk(str(dataset_path))
        except Exception:
            fmt, files = find_local_files(dataset_path, args.file_limit)
            if not files:
                raise FileNotFoundError(f"No data files found under {dataset_path}")
            print(f"Loading {len(files):,} local {fmt} files from {dataset_path}")
            return load_dataset(
                fmt,
                data_files={"train": files},
                split="train",
                streaming=args.streaming,
            )

    print(f"Loading remote dataset {args.dataset_path}")
    return load_dataset(
        args.dataset_path,
        name=args.dataset_config,
        split=args.split,
        streaming=args.streaming,
    )


def sample_text(sample: dict, text_key: str) -> str:
    if text_key in sample:
        return sample[text_key]
    if "text" in sample:
        return sample["text"]
    if "content" in sample:
        return sample["content"]
    raise KeyError(f"Sample has no '{text_key}', 'text', or 'content' field")


def build_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size} cannot fit uint16 bin files"
        )
    return tokenizer


def tokenize_text(tokenizer, text: str, add_bos: bool, add_eos: bool) -> np.ndarray:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if add_bos and tokenizer.bos_token_id is not None:
        if not token_ids or token_ids[0] != tokenizer.bos_token_id:
            token_ids.insert(0, tokenizer.bos_token_id)
    if add_eos and tokenizer.eos_token_id is not None:
        if not token_ids or token_ids[-1] != tokenizer.eos_token_id:
            token_ids.append(tokenizer.eos_token_id)
    return np.asarray(token_ids, dtype=np.uint16)


def shard_name(out_dir: Path, shard_index: int, val_shards: int) -> Path:
    split = "val" if shard_index < val_shards else "train"
    return out_dir / f"fineweb_{split}_{shard_index:06d}.bin"


def convert(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(args)
    dataset = load_text_dataset(args)

    shard_buffer = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    shard_index = 0
    docs_seen = 0
    tokens_written = 0

    progress = tqdm(dataset, desc="Tokenizing FineWeb docs", unit="doc")
    for sample in progress:
        if args.max_docs is not None and docs_seen >= args.max_docs:
            break

        text = sample_text(sample, args.text_key)
        tokens = tokenize_text(tokenizer, text, args.add_bos, args.add_eos)
        docs_seen += 1

        offset = 0
        while offset < len(tokens):
            remaining = args.shard_size - token_count
            take = min(remaining, len(tokens) - offset)
            shard_buffer[token_count : token_count + take] = tokens[offset : offset + take]
            token_count += take
            offset += take

            if token_count == args.shard_size:
                write_datafile(shard_name(out_dir, shard_index, args.val_shards), shard_buffer)
                tokens_written += token_count
                shard_index += 1
                token_count = 0
                if args.max_shards is not None and shard_index >= args.max_shards:
                    progress.close()
                    print(
                        f"Stopped at max_shards={args.max_shards}; "
                        f"docs_seen={docs_seen:,}, tokens_written={tokens_written:,}"
                    )
                    return

        progress.set_postfix(docs=docs_seen, shards=shard_index)

    if token_count:
        write_datafile(shard_name(out_dir, shard_index, args.val_shards), shard_buffer[:token_count])
        tokens_written += token_count
        shard_index += 1

    print(
        f"Done: docs_seen={docs_seen:,}, shards={shard_index:,}, "
        f"tokens_written={tokens_written:,}, out_dir={out_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        default="/lustre/fsmisc/dataset/HuggingFace/fineweb/data",
        help="Local FineWeb shard directory or remote HF dataset name",
    )
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--tokenizer_path",
        default="/lustre/fswork/projects/rech/qps/ulf36rc/assets/hf/Llama-2-7b-hf",
        help="HF tokenizer directory, for example the JZ Llama-2-7b-hf assets",
    )
    parser.add_argument(
        "--out_dir",
        default="/lustre/fswork/projects/rech/qps/ulf36rc/spectron_data/fineweb_llama2",
    )
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--shard_size", type=int, default=100_000_000)
    parser.add_argument("--val_shards", type=int, default=1)
    parser.add_argument("--file_limit", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_eos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
