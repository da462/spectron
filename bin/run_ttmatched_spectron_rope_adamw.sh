#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

RUN_MODE="${RUN_MODE:-lowrank_all}"

if [[ $# -gt 0 && "$1" != --* ]]; then
  # Backward-compatible no-op for the first version of this wrapper.
  if [[ "$1" == "134m" ]]; then
    shift
  fi
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_MODE="$1"
  shift
fi

MODEL_TAG="tt134m"
HIDDEN_SIZE=768
NUM_LAYERS=12
NUM_HEADS=12
N_KV_HEADS=12
MULTIPLE_OF=256
FFN_DIM_MULTIPLIER=""

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

DATA_ROOT="${DATA_ROOT:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_data/fineweb_llama2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-spectron_attnrank}"
WANDB_ENTITY="${WANDB_ENTITY:-da462}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
ROPE_THETA="${ROPE_THETA:-10000}"
SEQ_LEN="${SEQ_LEN:-2048}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-2555}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LR="${MAX_LR:-5e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-100}"
CHECKPOINT_INTERVAL_HOURS="${CHECKPOINT_INTERVAL_HOURS:-2.8}"

ROPE_TAG="${ROPE_THETA//./p}"
RUN_NAME="${RUN_NAME:-spectron_${MODEL_TAG}_fineweb_adamw_lr5e3_wd0p1_seq${SEQ_LEN}_steps${TOTAL_STEPS}_sched${LR_SCHEDULE_STEPS}_rope${ROPE_TAG}_${RUN_MODE}}"

FFN_DIM_MULTIPLIER_FLAGS=()
if [[ -n "$FFN_DIM_MULTIPLIER" ]]; then
  FFN_DIM_MULTIPLIER_FLAGS=(--ffn_dim_multiplier "$FFN_DIM_MULTIPLIER")
fi

echo "Spectron TT-matched AdamW run"
echo "  model_size=134m hidden=$HIDDEN_SIZE layers=$NUM_LAYERS heads=$NUM_HEADS ffn_multiplier=$FFN_DIM_MULTIPLIER"
echo "  run_mode=$RUN_MODE rope_theta=$ROPE_THETA seq_len=$SEQ_LEN total_steps=$TOTAL_STEPS lr_schedule_steps=$LR_SCHEDULE_STEPS"
echo "  lr=$MAX_LR weight_decay=$WEIGHT_DECAY batch=$GLOBAL_BATCH_SIZE micro_batch=$MICRO_BATCH_SIZE"
echo "  data_root=$DATA_ROOT"
echo "  run_name=$RUN_NAME"
echo "  wandb_mode=$WANDB_MODE"

exec torchrun --nproc_per_node="$NPROC_PER_NODE" simple_gpt_training.py \
  --seed 1234 \
  --hidden_size "$HIDDEN_SIZE" \
  --num_layers "$NUM_LAYERS" \
  --num_heads "$NUM_HEADS" \
  --n_kv_heads "$N_KV_HEADS" \
  --vocab_size 32000 \
  --max_position_embeddings "$SEQ_LEN" \
  --train_seq_len "$SEQ_LEN" \
  --val_seq_len "$SEQ_LEN" \
  --multiple_of "$MULTIPLE_OF" \
  "${FFN_DIM_MULTIPLIER_FLAGS[@]}" \
  --rope_theta "$ROPE_THETA" \
  --optimizer adamw \
  --max_lr "$MAX_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --scheduler cosine \
  --min_lr_factor 0.0 \
  --batch_size "$GLOBAL_BATCH_SIZE" \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  --total_steps "$TOTAL_STEPS" \
  --lr_schedule_steps "$LR_SCHEDULE_STEPS" \
  --warmup_ratio "$WARMUP_RATIO" \
  --log_interval "$LOG_INTERVAL" \
  --use_flex_attn \
  --bf16 \
  --virtual_workers_per_gpu 1 \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --checkpoint_interval_hours "$CHECKPOINT_INTERVAL_HOURS" \
  --train_files "$DATA_ROOT/fineweb_train_*.bin" \
  --val_files "$DATA_ROOT/fineweb_val_*.bin" \
  --checkpoint_dir "$CHECKPOINT_ROOT/$RUN_NAME" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY" \
  --run_name "$RUN_NAME" \
  "${LOW_RANK_FLAGS[@]}" \
  "$@"
