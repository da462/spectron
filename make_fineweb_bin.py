"""
make_fineweb_bin.py

Download and preprocess the FineWeb dataset into .bin files compatible with modded-nanogpt.

Usage:
  pip install datasets tiktoken tqdm numpy transformers torch sentencepiece
  python make_fineweb_bin.py --tokenizer llama --version 10B --shard_size 100000000 --out_dir data/fineweb10B_llama
  python make_fineweb_bin.py --tokenizer gpt2 --version 10B --shard_size 100000000 --out_dir data/fineweb10B_gpt2

Arguments:
  --tokenizer    Which tokenizer to use: llama or gpt2 (default: llama)
  --version      Which FineWeb version to use: 10B or 100B (default: 10B)
  --shard_size   Number of tokens per .bin file (default: 100_000_000)
  --out_dir      Output directory for .bin files (default: ./fineweb10B)

The first shard is used for validation, the rest for training.
"""
import os
import argparse
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
from transformers import AutoTokenizer


def write_datafile(filename, toks):
    """Write tokens to .bin file with modded-nanogpt header."""
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(toks) # number of tokens
    toks_np = np.array(toks, dtype=np.uint16)
    print(f"Writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="llama", choices=["llama", "gpt2"])
    parser.add_argument("--version", type=str, default="10B", choices=["10B", "100B"])
    parser.add_argument("--shard_size", type=int, default=100_000_000)
    parser.add_argument("--out_dir", type=str, default="fineweb10B")
    args = parser.parse_args()

    if args.version == "10B":
        remote_name = "sample-10BT"
    else:
        remote_name = "sample-100BT"

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Downloading FineWeb {args.version} from HuggingFace...")
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

    if args.tokenizer == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        def tokenize(doc):
            tokens = [enc._special_tokens['<|endoftext|>']]
            tokens.extend(enc.encode_ordinary(doc["text"]))
            return np.array(tokens, dtype=np.uint16)
    else: # llama
        enc = AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K")
        def tokenize(doc):
            tokens = [enc.eos_token_id]
            tokens.extend(enc.encode(doc["text"], add_special_tokens=False))
            return np.array(tokens, dtype=np.uint16)


    nprocs = os.cpu_count() or 2
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    with tqdm(total=len(fw), desc="Tokenizing docs") as pbar:
        for doc in fw:
            tokens = tokenize(doc)
            pbar.update(1)
            if token_count + len(tokens) < args.shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(args.out_dir, f"fineweb_{split}_{shard_index:06d}.bin")
                remainder = args.shard_size - token_count
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                # Start new shard with leftover tokens
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder
        # Write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(args.out_dir, f"fineweb_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])
    print(f"Done. Shards written to {args.out_dir}")


if __name__ == "__main__":
    main() 