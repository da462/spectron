# TorchTitan-Matched Spectron RoPE Runs

These wrappers run Spectron on JZ FineWeb bins with the TorchTitan model shapes
we want to compare against and TorchTitan-style AdamW hyperparameters, while
keeping Spectron's RoPE theta (`10000`).

## Model Presets

`134m` matches the attnrank TorchTitan `llama3._134m_rope` shape, except RoPE
theta is set back to Spectron's `10000`:

- `dim=768`
- `n_layers=12`
- `n_heads=12`
- MHA (`n_kv_heads=12`)
- `ffn_hidden_dim=2048` via `multiple_of=256`
- `vocab_size=32000`
- untied token embedding/output weights

`500m` uses the TorchTitan `llama3._500m` matrix shapes, but keeps Spectron's
untied embedding/output behavior:

- `dim=1280`
- `n_layers=20`
- `n_heads=20`
- MHA (`n_kv_heads=20`)
- `ffn_hidden_dim=5120` via `multiple_of=1024`, `ffn_dim_multiplier=1.5`
- `vocab_size=32000`
- untied token embedding/output weights

Both presets use:

- `rope_theta=10000`
- `seq_len=2048`
- AdamW `lr=5e-3`, `weight_decay=0.1`, betas `(0.9, 0.95)`
- cosine decay over `2555` steps with `5%` warmup and `min_lr=0`
- global batch `512`, micro batch `16`
- `seed=1234`
- WandB offline by default

## Run Commands

Use `lowrank_all` to reproduce the Spectron-style AB low-rank setup on all
eligible linear layers except embeddings/output:

```bash
cd /lustre/fswork/projects/rech/qps/ulf36rc/spectron

./bin/run_ttmatched_spectron_rope_adamw.sh 134m lowrank_all
./bin/run_ttmatched_spectron_rope_adamw.sh 500m lowrank_all
```

Full-rank controls:

```bash
./bin/run_ttmatched_spectron_rope_adamw.sh 134m fullrank
./bin/run_ttmatched_spectron_rope_adamw.sh 500m fullrank
```

JZ Slurm helper examples:

```bash
./bin/submit_jz_ttmatched_spectron.sh h100_4_dev2h_cpu30_whj 134m lowrank_all
./bin/submit_jz_ttmatched_spectron.sh h100_4_dev2h_cpu30_whj 500m lowrank_all
```

For a short A100 sanity run, override steps and use the dev profile:

```bash
TOTAL_STEPS=20 MAX_VAL_SAMPLES=8 \
./bin/submit_jz_ttmatched_spectron.sh a100_dev_20m 134m lowrank_all
```

## Data

The wrappers expect pretokenized FineWeb bins at:

```bash
/lustre/fswork/projects/rech/qps/ulf36rc/spectron_data/fineweb_llama2
```

If they are missing, generate them first:

```bash
./bin/prepare_jz_fineweb_bins.sh
```

## Difference From Regular Spectron Runs

This setup changes the regular Spectron script in these ways:

- Uses JZ FineWeb converted to Spectron/modded-nanogpt `.bin` shards with the
  JZ Llama tokenizer assets.
- Uses the requested TorchTitan model shapes instead of Spectron's older ad hoc
  script defaults.
- Uses TorchTitan-style AdamW settings: `lr=5e-3`, `weight_decay=0.1`, 5% warmup,
  cosine decay to zero, batch `512`, sequence length `2048`.
- Keeps Spectron's untied embedding/output behavior, matching the attnrank
  `134M_rope` runs.
- Keeps Spectron's RoPE theta (`10000`), not the attnrank diagnostic
  `134M_rope` theta (`500000`).
- Uses Spectron's low-rank replacement for `lowrank_all`: SVD-initialized AB
  factors with rank ratio `0.25`, excluding embeddings/output.

Remaining differences from TorchTitan training:

- Spectron reads contiguous pretokenized `.bin` streams; it does not use
  TorchTitan's streaming FineWeb dataloader or packed-document masks.
- Spectron's attention path is causal SDPA over the contiguous stream.
- Spectron init and training loop details remain Spectron's implementation.
- The 500M shape is not weight-tied, so it is shape-matched rather than an exact
  TorchTitan `_500m` parameterization.
- The wrapper does not add TorchTitan attnrank spectral diagnostics.
