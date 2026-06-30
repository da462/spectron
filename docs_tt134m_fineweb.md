# TorchTitan-Matched 134M FineWeb Runs

This branch adds a narrow Spectron setup for comparing against the TorchTitan
134M RoPE/FineWeb AdamW runs.

## Data Conversion On JZ

The Spectron dataloader reads pretokenized `.bin` shards. Convert the JZ-local
FineWeb Parquet copy with the JZ-local Llama tokenizer assets:

```bash
cd /lustre/fswork/projects/rech/qps/ulf36rc/spectron
source /etc/profile.d/proxy.sh

./bin/prepare_jz_fineweb_bins.sh
```

Defaults:

```bash
DATASET_PATH=/lustre/fsmisc/dataset/HuggingFace/fineweb/data
TOKENIZER_PATH=/lustre/fswork/projects/rech/qps/ulf36rc/assets/hf/Llama-2-7b-hf
OUT_DIR=/lustre/fswork/projects/rech/qps/ulf36rc/spectron_data/fineweb_llama2
SHARD_SIZE=100000000
VAL_SHARDS=1
```

For a small smoke test:

```bash
OUT_DIR=/tmp/spectron_fineweb_smoke ./bin/prepare_jz_fineweb_bins.sh \
  --file_limit 1 \
  --max_docs 32 \
  --shard_size 4096 \
  --max_shards 2
```

The converter uses `AutoTokenizer.from_pretrained(TOKENIZER_PATH)` and appends
BOS/EOS explicitly, matching TorchTitan's FineWeb dataset path that calls
`encode(..., add_bos=True, add_eos=True)`.

## 134M AdamW Command

The run wrapper uses:

- `dim=768`
- `n_layers=12`
- `n_heads=12`
- `n_kv_heads=12` (MHA)
- `ffn_hidden_dim=2048` via the standard Llama `multiple_of=256` rule
- `vocab_size=32000`
- `max_seq_len=2048`
- `rope_theta=500000`
- untied token embeddings and lm head
- AdamW `lr=5e-3`, `weight_decay=0.1`, betas `(0.9, 0.95)`
- cosine schedule over `2555` steps with `5%` warmup and `min_lr=0`
- global batch `512`, local batch `16` on 4 GPUs
- `seed=1234`

Low-rank all, AB factorization, rank ratio 0.25:

```bash
cd /lustre/fswork/projects/rech/qps/ulf36rc/spectron
./bin/run_tt134m_fineweb_adamw.sh lowrank_all
```

Full-rank baseline with the same hyperparameters:

```bash
cd /lustre/fswork/projects/rech/qps/ulf36rc/spectron
./bin/run_tt134m_fineweb_adamw.sh fullrank
```

Useful overrides:

```bash
DATA_ROOT=/path/to/fineweb_bins \
CHECKPOINT_ROOT=/path/to/checkpoints \
WANDB_PROJECT=sl_moe \
WANDB_ENTITY=da462 \
NPROC_PER_NODE=4 \
RUN_NAME=my_run_name \
./bin/run_tt134m_fineweb_adamw.sh lowrank_all
```

The wrapper intentionally does not set `--use_flex_attn`; it uses the model's
default SDPA path with continuous RoPE positions, matching Spectron's existing
TitanGPT attention behavior.
