# TorchTitan-Shaped 134M Spectron RoPE 500-Step Run

These wrappers run a 500-step Spectron diagnostic on JZ FineWeb bins with the
134M TorchTitan/attnrank model shape and TorchTitan-style AdamW hyperparameters,
while keeping Spectron's RoPE theta (`10000`).

## Model Preset

The run matches the attnrank TorchTitan `llama3._134m_rope` shape, except RoPE
theta is set to Spectron's `10000`:

- `dim=768`
- `n_layers=12`
- `n_heads=12`
- MHA (`n_kv_heads=12`)
- `ffn_hidden_dim=2048` via `multiple_of=256`
- `vocab_size=32000`
- untied token embedding/output weights

The wrapper also uses:

- `rope_theta=10000`
- `seq_len=2048`
- AdamW `lr=5e-3`, `weight_decay=0.1`, betas `(0.9, 0.95)`
- run length `500` steps
- LR schedule horizon `2555` steps: warmup is `5%` of `2555` steps, then cosine
  decay toward zero at step `2555`
- global batch `512`, micro batch `16`
- `seed=1234`
- WandB offline by default

## Run Commands

Use `lowrank_all` to reproduce the Spectron-style AB low-rank setup on all
eligible linear layers except embeddings/output:

```bash
cd /lustre/fswork/projects/rech/qps/ulf36rc/spectron

./bin/run_ttmatched_spectron_rope_adamw.sh lowrank_all
```

Full-rank controls:

```bash
./bin/run_ttmatched_spectron_rope_adamw.sh fullrank
```

JZ Slurm helper examples:

```bash
./bin/submit_jz_ttmatched_spectron.sh h100_4_dev2h_cpu30_whj lowrank_all
```

For a short A100 sanity run, override steps and use the dev profile:

```bash
TOTAL_STEPS=20 MAX_VAL_SAMPLES=8 \
./bin/submit_jz_ttmatched_spectron.sh a100_dev_20m lowrank_all
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
- Defaults to `500` training steps while preserving the `2555`-step LR schedule.
  Override `TOTAL_STEPS` and `LR_SCHEDULE_STEPS` separately only when
  intentionally changing those semantics.

Remaining differences from TorchTitan training:

- Spectron reads contiguous pretokenized `.bin` streams; it does not use
  TorchTitan's streaming FineWeb dataloader or packed-document masks.
- Spectron's attention path uses its original `--use_flex_attn` backend from
  the `torchtitan` Python package.
- Spectron init and training loop details remain Spectron's implementation.
- The wrapper does not add TorchTitan attnrank spectral diagnostics.
