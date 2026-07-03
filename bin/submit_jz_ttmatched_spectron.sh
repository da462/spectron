#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-h100_4_dev2h_cpu30_whj}"
RUN_MODE="${2:-lowrank_all}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_DIR/jz_logs}"
JOB_DIR="${JOB_DIR:-$REPO_DIR/jobs}"
mkdir -p "$LOG_DIR" "$JOB_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
JOB_STEPS="${TOTAL_STEPS:-500}"
JOB_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-2555}"
SUBMIT_MAX_LR="${MAX_LR:-5e-3}"
SUBMIT_OPTIMIZER="${OPTIMIZER:-adamw}"
SUBMIT_LR_TAG="${SUBMIT_MAX_LR//./p}"
SUBMIT_LR_TAG="${SUBMIT_LR_TAG//e-/em}"
SUBMIT_LR_TAG="${SUBMIT_LR_TAG//e+/ep}"
SUBMIT_LR_TAG="${SUBMIT_LR_TAG//-/m}"
JOB_NAME="spectron_tt134m_${SUBMIT_OPTIMIZER}_${RUN_MODE}_lr${SUBMIT_LR_TAG}_steps${JOB_STEPS}_sched${JOB_SCHEDULE_STEPS}"
JOB_SCRIPT="$JOB_DIR/${JOB_NAME}_${PROFILE}_${RUN_STAMP}.slurm"

case "$PROFILE" in
  h100_4_dev2h_cpu30_whj)
    ACCOUNT="whj@h100"
    PARTITION="gpu_p6"
    QOS="qos_gpu_h100-dev"
    CONSTRAINT="h100"
    GPUS=4
    CPUS_PER_TASK=30
    TIME_LIMIT="02:00:00"
    ;;
  h100_4_dev2h_cpu30_qps)
    ACCOUNT="qps@h100"
    PARTITION="gpu_p6"
    QOS="qos_gpu_h100-dev"
    CONSTRAINT="h100"
    GPUS=4
    CPUS_PER_TASK=30
    TIME_LIMIT="02:00:00"
    ;;
  a100_4_dev2h_cpu30_whj)
    ACCOUNT="whj@a100"
    PARTITION="gpu_p5"
    QOS="qos_gpu_a100-dev"
    CONSTRAINT="a100"
    GPUS=4
    CPUS_PER_TASK=30
    TIME_LIMIT="02:00:00"
    ;;
  a100_dev_20m)
    ACCOUNT="qps@a100"
    PARTITION="gpu_p5"
    QOS="qos_gpu_a100-dev"
    CONSTRAINT="a100"
    GPUS=1
    CPUS_PER_TASK=10
    TIME_LIMIT="00:20:00"
    ;;
  *)
    echo "Unknown PROFILE '$PROFILE'. Use h100_4_dev2h_cpu30_whj, h100_4_dev2h_cpu30_qps, a100_4_dev2h_cpu30_whj, or a100_dev_20m." >&2
    exit 2
    ;;
esac

case "$RUN_MODE" in
  fullrank|lowrank_all|lowrank_attention|lowrank_ffn)
    ;;
  *)
    echo "Unknown RUN_MODE '$RUN_MODE'. Use fullrank, lowrank_all, lowrank_attention, or lowrank_ffn." >&2
    exit 2
    ;;
esac

case "$SUBMIT_OPTIMIZER" in
  adamw|muon)
    ;;
  *)
    echo "Unknown OPTIMIZER '$SUBMIT_OPTIMIZER'. Use adamw or muon." >&2
    exit 2
    ;;
esac

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$GPUS
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --time=$TIME_LIMIT
#SBATCH --constraint=$CONSTRAINT
#SBATCH --output=$LOG_DIR/${JOB_NAME}_%j.txt
#SBATCH --error=$LOG_DIR/${JOB_NAME}_%j.txt

set -euo pipefail

cd "$REPO_DIR"
source /etc/profile.d/proxy.sh || true

if [[ -n "\${JZ_VENV:-}" ]]; then
  source "\$JZ_VENV/bin/activate"
elif [[ -d "$REPO_DIR/.venv_spectron" ]]; then
  source "$REPO_DIR/.venv_spectron/bin/activate"
else
  echo "No Spectron venv found. Set JZ_VENV or create $REPO_DIR/.venv_spectron." >&2
  exit 1
fi

export WANDB_MODE="\${WANDB_MODE:-offline}"
export NPROC_PER_NODE="\${NPROC_PER_NODE:-$GPUS}"

RUN_MODE="$RUN_MODE"
DATA_ROOT="\${DATA_ROOT:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_data/fineweb_llama2}"
CHECKPOINT_ROOT="\${CHECKPOINT_ROOT:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_checkpoints}"
WANDB_PROJECT="\${WANDB_PROJECT:-spectron_attnrank}"
WANDB_ENTITY="\${WANDB_ENTITY:-da462}"
WANDB_DIR="\${WANDB_DIR:-\$CHECKPOINT_ROOT/wandb_offline}"
TOTAL_STEPS="\${TOTAL_STEPS:-$JOB_STEPS}"
LR_SCHEDULE_STEPS="\${LR_SCHEDULE_STEPS:-$JOB_SCHEDULE_STEPS}"
GLOBAL_BATCH_SIZE="\${GLOBAL_BATCH_SIZE:-512}"
MICRO_BATCH_SIZE="\${MICRO_BATCH_SIZE:-16}"
LOG_INTERVAL="\${LOG_INTERVAL:-1}"
EVAL_INTERVAL="\${EVAL_INTERVAL:-0}"
MAX_VAL_SAMPLES="\${MAX_VAL_SAMPLES:-100}"
CHECKPOINT_INTERVAL_HOURS="\${CHECKPOINT_INTERVAL_HOURS:-2.8}"
CHECKPOINT_INTERVAL_STEPS="\${CHECKPOINT_INTERVAL_STEPS:-500}"
CHECKPOINT_KEEP_LATEST_K="\${CHECKPOINT_KEEP_LATEST_K:-2}"
MAX_LR="\${MAX_LR:-$SUBMIT_MAX_LR}"
OPTIMIZER="\${OPTIMIZER:-$SUBMIT_OPTIMIZER}"
WARMUP_START_FACTOR="\${WARMUP_START_FACTOR:-0.0}"
SKIP_FINAL_EVAL="\${SKIP_FINAL_EVAL:-1}"
LR_TAG="\${MAX_LR//./p}"
LR_TAG="\${LR_TAG//e-/em}"
LR_TAG="\${LR_TAG//e+/ep}"
LR_TAG="\${LR_TAG//-/m}"
RUN_NAME="\${RUN_NAME:-spectron_tt134m_fineweb_\${OPTIMIZER}_lr\${LR_TAG}_wd0p1_seq2048_steps\${TOTAL_STEPS}_sched\${LR_SCHEDULE_STEPS}_rope10000_\${RUN_MODE}}"
mkdir -p "\$WANDB_DIR"
export WANDB_DIR

LOW_RANK_FLAGS=()
case "\$RUN_MODE" in
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
  lowrank_attention)
    LOW_RANK_FLAGS=(
      --low_rank
      --low_rank_ratio 0.25
      --disable_c
      --exclude_modules tok_embeddings output feed_forward
    )
    ;;
  lowrank_ffn)
    LOW_RANK_FLAGS=(
      --low_rank
      --low_rank_ratio 0.25
      --disable_c
      --exclude_modules tok_embeddings output attention
    )
    ;;
esac

FINAL_EVAL_FLAGS=()
if [[ "\$SKIP_FINAL_EVAL" == "1" ]]; then
  FINAL_EVAL_FLAGS=(--skip_final_eval)
fi

echo "Spectron TT-matched run"
echo "  model_size=134m hidden=768 layers=12 heads=12"
echo "  optimizer=\$OPTIMIZER run_mode=\$RUN_MODE rope_theta=10000 seq_len=2048 total_steps=\$TOTAL_STEPS lr_schedule_steps=\$LR_SCHEDULE_STEPS"
echo "  lr=\$MAX_LR warmup_start_factor=\$WARMUP_START_FACTOR weight_decay=0.1 batch=\$GLOBAL_BATCH_SIZE micro_batch=\$MICRO_BATCH_SIZE"
echo "  log_interval=\$LOG_INTERVAL eval_interval=\$EVAL_INTERVAL skip_final_eval=\$SKIP_FINAL_EVAL"
echo "  data_root=\$DATA_ROOT"
echo "  run_name=\$RUN_NAME"
echo "  checkpoint_interval_steps=\$CHECKPOINT_INTERVAL_STEPS checkpoint_keep_latest_k=\$CHECKPOINT_KEEP_LATEST_K"
echo "  wandb_mode=\$WANDB_MODE"
echo "  wandb_dir=\$WANDB_DIR"

exec torchrun --nproc_per_node="\$NPROC_PER_NODE" simple_gpt_training.py \\
  --seed 1234 \\
  --hidden_size 768 \\
  --num_layers 12 \\
  --num_heads 12 \\
  --n_kv_heads 12 \\
  --vocab_size 32000 \\
  --max_position_embeddings 2048 \\
  --train_seq_len 2048 \\
  --val_seq_len 2048 \\
  --multiple_of 256 \\
  --rope_theta 10000 \\
  --optimizer "\$OPTIMIZER" \\
  --max_lr "\$MAX_LR" \\
  --weight_decay 0.1 \\
  --adam_beta1 0.9 \\
  --adam_beta2 0.95 \\
  --scheduler cosine \\
  --min_lr_factor 0.0 \\
  --batch_size "\$GLOBAL_BATCH_SIZE" \\
  --micro_batch_size "\$MICRO_BATCH_SIZE" \\
  --total_steps "\$TOTAL_STEPS" \\
  --lr_schedule_steps "\$LR_SCHEDULE_STEPS" \\
  --warmup_ratio 0.05 \\
  --warmup_start_factor "\$WARMUP_START_FACTOR" \\
  --log_interval "\$LOG_INTERVAL" \\
  --eval_interval "\$EVAL_INTERVAL" \\
  --use_flex_attn \\
  --bf16 \\
  --virtual_workers_per_gpu 1 \\
  --max_val_samples "\$MAX_VAL_SAMPLES" \\
  --checkpoint_interval_hours "\$CHECKPOINT_INTERVAL_HOURS" \\
  --checkpoint_interval_steps "\$CHECKPOINT_INTERVAL_STEPS" \\
  --checkpoint_keep_latest_k "\$CHECKPOINT_KEEP_LATEST_K" \\
  --train_files "\$DATA_ROOT/fineweb_train_*.bin" \\
  --val_files "\$DATA_ROOT/fineweb_val_*.bin" \\
  --checkpoint_dir "\$CHECKPOINT_ROOT/\$RUN_NAME" \\
  --wandb_project "\$WANDB_PROJECT" \\
  --wandb_entity "\$WANDB_ENTITY" \\
  --run_name "\$RUN_NAME" \\
  "\${FINAL_EVAL_FLAGS[@]}" \\
  "\${LOW_RANK_FLAGS[@]}"
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Prepared $JOB_SCRIPT"
  echo "DRY_RUN=1 set, not submitting."
else
  echo "Submitting $JOB_SCRIPT"
  sbatch "$JOB_SCRIPT"
fi
