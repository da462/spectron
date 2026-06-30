#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASET_PATH="${DATASET_PATH:-/lustre/fsmisc/dataset/HuggingFace/fineweb/data}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/lustre/fswork/projects/rech/qps/ulf36rc/assets/hf/Llama-2-7b-hf}"
OUT_DIR="${OUT_DIR:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_data/fineweb_llama2}"
SHARD_SIZE="${SHARD_SIZE:-100000000}"
VAL_SHARDS="${VAL_SHARDS:-1}"

python prepare_fineweb_bins.py \
  --dataset_path "$DATASET_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --out_dir "$OUT_DIR" \
  --shard_size "$SHARD_SIZE" \
  --val_shards "$VAL_SHARDS" \
  "$@"
