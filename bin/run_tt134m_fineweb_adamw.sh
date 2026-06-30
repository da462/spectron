#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

RUN_MODE="${1:-${RUN_MODE:-lowrank_all}}"
DATA_ROOT="${DATA_ROOT:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_data/fineweb_llama2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-spectron_attnrank}"
WANDB_ENTITY="${WANDB_ENTITY:-da462}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RUN_NAME="${RUN_NAME:-spectron_tt134m_fineweb_adamw_lr5e3_${RUN_MODE}}"

LOW_RANK_FLAGS=()
case "$RUN_MODE" in
  fullrank)
    ;;
  lowrank_all)
    LOW_RANK_FLAGS=(
      --low_rank
      --low_rank_ratio 0.25
      --disable_c
      --exclude_modules tok_embeddings output
    )
    ;;
  *)
    echo "Unknown RUN_MODE '$RUN_MODE'. Use fullrank or lowrank_all." >&2
    exit 2
    ;;
esac

exec torchrun --nproc_per_node="$NPROC_PER_NODE" simple_gpt_training.py \
  --seed 1234 \
  --hidden_size 768 \
  --num_layers 12 \
  --num_heads 12 \
  --n_kv_heads 12 \
  --vocab_size 32000 \
  --max_position_embeddings 2048 \
  --train_seq_len 2048 \
  --val_seq_len 2048 \
  --multiple_of 256 \
  --rope_theta 500000 \
  --optimizer adamw \
  --max_lr 5e-3 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --scheduler cosine \
  --min_lr_factor 0.0 \
  --batch_size 512 \
  --micro_batch_size 16 \
  --total_steps 2555 \
  --warmup_ratio 0.05 \
  --log_interval 10 \
  --bf16 \
  --virtual_workers_per_gpu 1 \
  --max_val_samples 100 \
  --train_files "$DATA_ROOT/fineweb_train_*.bin" \
  --val_files "$DATA_ROOT/fineweb_val_*.bin" \
  --checkpoint_dir "$CHECKPOINT_ROOT/$RUN_NAME" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --run_name "$RUN_NAME" \
  "${LOW_RANK_FLAGS[@]}" \
  "${@:2}"
