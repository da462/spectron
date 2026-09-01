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
SUBMIT_USE_FLEX_ATTN="${USE_FLEX_ATTN:-1}"
SUBMIT_TT_STYLE_INIT="${TT_STYLE_INIT:-0}"
SUBMIT_EMBEDDING_INIT_STD="${EMBEDDING_INIT_STD:-}"
SUBMIT_MECHANISTIC_DIAGNOSTICS="${MECHANISTIC_DIAGNOSTICS:-0}"
SUBMIT_MECHANISTIC_DIAGNOSTIC_BATCH="${MECHANISTIC_DIAGNOSTIC_BATCH:-}"
SUBMIT_CHECKPOINT_INTERVAL_HOURS="${CHECKPOINT_INTERVAL_HOURS:-2.8}"
SUBMIT_CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-500}"
SUBMIT_CHECKPOINT_KEEP_LATEST_K="${CHECKPOINT_KEEP_LATEST_K:-2}"
SUBMIT_SPECTRAL_LR_SCALING="${SPECTRAL_LR_SCALING:-0}"
SUBMIT_SPECTRAL_LR_SCALING_OFFSET="${SPECTRAL_LR_SCALING_OFFSET:-1.0}"
SUBMIT_SPECTRAL_LR_TARGET="${SPECTRAL_LR_TARGET:-all}"
SUBMIT_LOWRANK_FFN_LR_MULTIPLIER="${LOWRANK_FFN_LR_MULTIPLIER:-1.0}"
SUBMIT_LOWRANK_FFN_WEIGHT_DECAY="${LOWRANK_FFN_WEIGHT_DECAY:-}"
SUBMIT_LOWRANK_ATTENTION_WEIGHT_DECAY="${LOWRANK_ATTENTION_WEIGHT_DECAY:-}"
SUBMIT_TRACK_LOWRANK_LR_INTERVAL="${TRACK_LOWRANK_LR_INTERVAL:-0}"
SUBMIT_TRACK_LOWRANK_LR_MODULE_TYPE="${TRACK_LOWRANK_LR_MODULE_TYPE:-ffn}"
SUBMIT_LOWRANK_LR_LOG_PATH="${LOWRANK_LR_LOG_PATH:-}"
SUBMIT_SPECTRAL_WEIGHT_DECAY="${SPECTRAL_WEIGHT_DECAY:-0.0}"
SUBMIT_WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SUBMIT_NH_WEIGHT_DECAY="${NH_WEIGHT_DECAY:-$SUBMIT_WEIGHT_DECAY}"
SUBMIT_MIN_LR_FACTOR="${MIN_LR_FACTOR:-0.0}"
SUBMIT_LOW_RANK_RATIO="${LOW_RANK_RATIO:-0.25}"
SUBMIT_LOW_RANK_W2_SAME_RANK="${LOW_RANK_W2_SAME_RANK_AS_W1W3:-0}"
SUBMIT_LOW_RANK_ATTENTION_FACTORIZATION="${LOW_RANK_ATTENTION_FACTORIZATION:-whole}"
SUBMIT_LOW_RANK_ATTENTION_PER_HEAD_RANK="${LOW_RANK_ATTENTION_PER_HEAD_RANK:-}"
SUBMIT_MODEL_TAG="${MODEL_TAG:-tt134m}"
SUBMIT_MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-134m}"
SUBMIT_HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
SUBMIT_NUM_LAYERS="${NUM_LAYERS:-12}"
SUBMIT_NUM_HEADS="${NUM_HEADS:-12}"
SUBMIT_N_KV_HEADS="${N_KV_HEADS:-$SUBMIT_NUM_HEADS}"
SUBMIT_VOCAB_SIZE="${VOCAB_SIZE:-32000}"
SUBMIT_SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
SUBMIT_MULTIPLE_OF="${MULTIPLE_OF:-256}"
SUBMIT_ROPE_THETA="${ROPE_THETA:-10000}"
SUBMIT_ADJUST_MUON_LR="${ADJUST_MUON_LR:-original}"
SUBMIT_LR_TAG="${SUBMIT_MAX_LR//./p}"
SUBMIT_LR_TAG="${SUBMIT_LR_TAG//e-/em}"
SUBMIT_LR_TAG="${SUBMIT_LR_TAG//e+/ep}"
SUBMIT_LR_TAG="${SUBMIT_LR_TAG//-/m}"
SUBMIT_WD_TAG="${SUBMIT_WEIGHT_DECAY//./p}"
SUBMIT_WD_TAG="${SUBMIT_WD_TAG//e-/em}"
SUBMIT_WD_TAG="${SUBMIT_WD_TAG//e+/ep}"
SUBMIT_WD_TAG="${SUBMIT_WD_TAG//-/m}"
JOB_SUFFIX=""
if [[ "$SUBMIT_USE_FLEX_ATTN" == "0" ]]; then
  JOB_SUFFIX="${JOB_SUFFIX}_sdpa"
fi
if [[ "$SUBMIT_TT_STYLE_INIT" == "1" ]]; then
  JOB_SUFFIX="${JOB_SUFFIX}_ttinit"
fi
if [[ -n "$SUBMIT_EMBEDDING_INIT_STD" ]]; then
  EMBSTD_TAG="${SUBMIT_EMBEDDING_INIT_STD//./p}"
  JOB_SUFFIX="${JOB_SUFFIX}_embstd${EMBSTD_TAG}"
fi
if [[ "$SUBMIT_MECHANISTIC_DIAGNOSTICS" == "1" ]]; then
  JOB_SUFFIX="${JOB_SUFFIX}_mechanistic"
fi
if [[ "$SUBMIT_SPECTRAL_LR_SCALING" == "1" ]]; then
  JOB_SUFFIX="${JOB_SUFFIX}_spectron"
fi
if [[ "$SUBMIT_SPECTRAL_LR_TARGET" != "all" ]]; then
  TARGET_TAG="${SUBMIT_SPECTRAL_LR_TARGET//[^a-zA-Z0-9]/}"
  JOB_SUFFIX="${JOB_SUFFIX}_spectral${TARGET_TAG}"
fi
if [[ "$SUBMIT_LOWRANK_FFN_LR_MULTIPLIER" != "1" && "$SUBMIT_LOWRANK_FFN_LR_MULTIPLIER" != "1.0" && "$SUBMIT_LOWRANK_FFN_LR_MULTIPLIER" != "1.00" ]]; then
  FFNLR_TAG="${SUBMIT_LOWRANK_FFN_LR_MULTIPLIER//./p}"
  FFNLR_TAG="${FFNLR_TAG//e-/em}"
  FFNLR_TAG="${FFNLR_TAG//e+/ep}"
  FFNLR_TAG="${FFNLR_TAG//-/m}"
  JOB_SUFFIX="${JOB_SUFFIX}_ffnlr${FFNLR_TAG}"
fi
if [[ -n "$SUBMIT_LOWRANK_FFN_WEIGHT_DECAY" ]]; then
  FFNWD_TAG="${SUBMIT_LOWRANK_FFN_WEIGHT_DECAY//./p}"
  FFNWD_TAG="${FFNWD_TAG//e-/em}"
  FFNWD_TAG="${FFNWD_TAG//e+/ep}"
  FFNWD_TAG="${FFNWD_TAG//-/m}"
  JOB_SUFFIX="${JOB_SUFFIX}_ffnwd${FFNWD_TAG}"
fi
if [[ -n "$SUBMIT_LOWRANK_ATTENTION_WEIGHT_DECAY" ]]; then
  ATTNWD_TAG="${SUBMIT_LOWRANK_ATTENTION_WEIGHT_DECAY//./p}"
  ATTNWD_TAG="${ATTNWD_TAG//e-/em}"
  ATTNWD_TAG="${ATTNWD_TAG//e+/ep}"
  ATTNWD_TAG="${ATTNWD_TAG//-/m}"
  JOB_SUFFIX="${JOB_SUFFIX}_attnwd${ATTNWD_TAG}"
fi
if [[ "$SUBMIT_SPECTRAL_LR_SCALING_OFFSET" == "0" || "$SUBMIT_SPECTRAL_LR_SCALING_OFFSET" == "0.0" ]]; then
  JOB_SUFFIX="${JOB_SUFFIX}_no_plus_one"
fi
if [[ "$SUBMIT_SPECTRAL_WEIGHT_DECAY" != "0" && "$SUBMIT_SPECTRAL_WEIGHT_DECAY" != "0.0" ]]; then
  SWD_TAG="${SUBMIT_SPECTRAL_WEIGHT_DECAY//./p}"
  SWD_TAG="${SWD_TAG//e-/em}"
  SWD_TAG="${SWD_TAG//e+/ep}"
  SWD_TAG="${SWD_TAG//-/m}"
  JOB_SUFFIX="${JOB_SUFFIX}_swd${SWD_TAG}"
fi
if [[ "$SUBMIT_MIN_LR_FACTOR" != "0" && "$SUBMIT_MIN_LR_FACTOR" != "0.0" ]]; then
  MINLR_TAG="${SUBMIT_MIN_LR_FACTOR//./p}"
  MINLR_TAG="${MINLR_TAG//e-/em}"
  MINLR_TAG="${MINLR_TAG//e+/ep}"
  MINLR_TAG="${MINLR_TAG//-/m}"
  JOB_SUFFIX="${JOB_SUFFIX}_minlr${MINLR_TAG}"
fi
if [[ "$RUN_MODE" != "fullrank" && "$SUBMIT_LOW_RANK_RATIO" != "0.25" && "$SUBMIT_LOW_RANK_RATIO" != "0.250" ]]; then
  LRANK_TAG="${SUBMIT_LOW_RANK_RATIO//./p}"
  LRANK_TAG="${LRANK_TAG//e-/em}"
  LRANK_TAG="${LRANK_TAG//e+/ep}"
  LRANK_TAG="${LRANK_TAG//-/m}"
  JOB_SUFFIX="${JOB_SUFFIX}_rr${LRANK_TAG}"
fi
if [[ "$RUN_MODE" != "fullrank" && "$SUBMIT_LOW_RANK_W2_SAME_RANK" == "1" ]]; then
  JOB_SUFFIX="${JOB_SUFFIX}_w2same"
fi
if [[ "$RUN_MODE" != "fullrank" && "$SUBMIT_LOW_RANK_ATTENTION_FACTORIZATION" == "per_head" ]]; then
  JOB_SUFFIX="${JOB_SUFFIX}_attnperhead"
  if [[ -n "$SUBMIT_LOW_RANK_ATTENTION_PER_HEAD_RANK" ]]; then
    JOB_SUFFIX="${JOB_SUFFIX}r${SUBMIT_LOW_RANK_ATTENTION_PER_HEAD_RANK}"
  fi
fi
JOB_NAME="spectron_${SUBMIT_MODEL_TAG}_${SUBMIT_OPTIMIZER}_${RUN_MODE}_lr${SUBMIT_LR_TAG}_wd${SUBMIT_WD_TAG}_steps${JOB_STEPS}_sched${JOB_SCHEDULE_STEPS}${JOB_SUFFIX}"
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
CHECKPOINT_INTERVAL_HOURS="\${CHECKPOINT_INTERVAL_HOURS:-$SUBMIT_CHECKPOINT_INTERVAL_HOURS}"
CHECKPOINT_INTERVAL_STEPS="\${CHECKPOINT_INTERVAL_STEPS:-$SUBMIT_CHECKPOINT_INTERVAL_STEPS}"
CHECKPOINT_KEEP_LATEST_K="\${CHECKPOINT_KEEP_LATEST_K:-$SUBMIT_CHECKPOINT_KEEP_LATEST_K}"
REQUIRE_RESUME="\${REQUIRE_RESUME:-0}"
MAX_LR="\${MAX_LR:-$SUBMIT_MAX_LR}"
OPTIMIZER="\${OPTIMIZER:-$SUBMIT_OPTIMIZER}"
WEIGHT_DECAY="\${WEIGHT_DECAY:-$SUBMIT_WEIGHT_DECAY}"
NH_WEIGHT_DECAY="\${NH_WEIGHT_DECAY:-$SUBMIT_NH_WEIGHT_DECAY}"
USE_FLEX_ATTN="\${USE_FLEX_ATTN:-$SUBMIT_USE_FLEX_ATTN}"
TT_STYLE_INIT="\${TT_STYLE_INIT:-$SUBMIT_TT_STYLE_INIT}"
EMBEDDING_INIT_STD="\${EMBEDDING_INIT_STD:-$SUBMIT_EMBEDDING_INIT_STD}"
MECHANISTIC_DIAGNOSTICS="\${MECHANISTIC_DIAGNOSTICS:-$SUBMIT_MECHANISTIC_DIAGNOSTICS}"
MECHANISTIC_DIAGNOSTIC_BATCH="\${MECHANISTIC_DIAGNOSTIC_BATCH:-$SUBMIT_MECHANISTIC_DIAGNOSTIC_BATCH}"
SPECTRAL_LR_SCALING="\${SPECTRAL_LR_SCALING:-$SUBMIT_SPECTRAL_LR_SCALING}"
SPECTRAL_LR_SCALING_OFFSET="\${SPECTRAL_LR_SCALING_OFFSET:-$SUBMIT_SPECTRAL_LR_SCALING_OFFSET}"
SPECTRAL_LR_TARGET="\${SPECTRAL_LR_TARGET:-$SUBMIT_SPECTRAL_LR_TARGET}"
LOWRANK_FFN_LR_MULTIPLIER="\${LOWRANK_FFN_LR_MULTIPLIER:-$SUBMIT_LOWRANK_FFN_LR_MULTIPLIER}"
LOWRANK_FFN_WEIGHT_DECAY="\${LOWRANK_FFN_WEIGHT_DECAY:-$SUBMIT_LOWRANK_FFN_WEIGHT_DECAY}"
LOWRANK_ATTENTION_WEIGHT_DECAY="\${LOWRANK_ATTENTION_WEIGHT_DECAY:-$SUBMIT_LOWRANK_ATTENTION_WEIGHT_DECAY}"
TRACK_LOWRANK_LR_INTERVAL="\${TRACK_LOWRANK_LR_INTERVAL:-$SUBMIT_TRACK_LOWRANK_LR_INTERVAL}"
TRACK_LOWRANK_LR_MODULE_TYPE="\${TRACK_LOWRANK_LR_MODULE_TYPE:-$SUBMIT_TRACK_LOWRANK_LR_MODULE_TYPE}"
LOWRANK_LR_LOG_PATH="\${LOWRANK_LR_LOG_PATH:-$SUBMIT_LOWRANK_LR_LOG_PATH}"
SPECTRAL_WEIGHT_DECAY="\${SPECTRAL_WEIGHT_DECAY:-$SUBMIT_SPECTRAL_WEIGHT_DECAY}"
SWD_TYPE="\${SWD_TYPE:-standard}"
MIN_LR_FACTOR="\${MIN_LR_FACTOR:-$SUBMIT_MIN_LR_FACTOR}"
LOW_RANK_RATIO="\${LOW_RANK_RATIO:-$SUBMIT_LOW_RANK_RATIO}"
LOW_RANK_W2_SAME_RANK_AS_W1W3="\${LOW_RANK_W2_SAME_RANK_AS_W1W3:-$SUBMIT_LOW_RANK_W2_SAME_RANK}"
LOW_RANK_ATTENTION_FACTORIZATION="\${LOW_RANK_ATTENTION_FACTORIZATION:-$SUBMIT_LOW_RANK_ATTENTION_FACTORIZATION}"
LOW_RANK_ATTENTION_PER_HEAD_RANK="\${LOW_RANK_ATTENTION_PER_HEAD_RANK:-$SUBMIT_LOW_RANK_ATTENTION_PER_HEAD_RANK}"
MODEL_SIZE_LABEL="\${MODEL_SIZE_LABEL:-$SUBMIT_MODEL_SIZE_LABEL}"
HIDDEN_SIZE="\${HIDDEN_SIZE:-$SUBMIT_HIDDEN_SIZE}"
NUM_LAYERS="\${NUM_LAYERS:-$SUBMIT_NUM_LAYERS}"
NUM_HEADS="\${NUM_HEADS:-$SUBMIT_NUM_HEADS}"
N_KV_HEADS="\${N_KV_HEADS:-$SUBMIT_N_KV_HEADS}"
VOCAB_SIZE="\${VOCAB_SIZE:-$SUBMIT_VOCAB_SIZE}"
SEQUENCE_LENGTH="\${SEQUENCE_LENGTH:-$SUBMIT_SEQUENCE_LENGTH}"
MULTIPLE_OF="\${MULTIPLE_OF:-$SUBMIT_MULTIPLE_OF}"
ROPE_THETA="\${ROPE_THETA:-$SUBMIT_ROPE_THETA}"
ADJUST_MUON_LR="\${ADJUST_MUON_LR:-$SUBMIT_ADJUST_MUON_LR}"
WARMUP_START_FACTOR="\${WARMUP_START_FACTOR:-0.0}"
SKIP_FINAL_EVAL="\${SKIP_FINAL_EVAL:-1}"
LR_TAG="\${MAX_LR//./p}"
LR_TAG="\${LR_TAG//e-/em}"
LR_TAG="\${LR_TAG//e+/ep}"
LR_TAG="\${LR_TAG//-/m}"
WD_TAG="\${WEIGHT_DECAY//./p}"
WD_TAG="\${WD_TAG//e-/em}"
WD_TAG="\${WD_TAG//e+/ep}"
WD_TAG="\${WD_TAG//-/m}"
RUN_SUFFIX=""
if [[ "\$USE_FLEX_ATTN" == "0" ]]; then
  RUN_SUFFIX="\${RUN_SUFFIX}_sdpa"
fi
if [[ "\$TT_STYLE_INIT" == "1" ]]; then
  RUN_SUFFIX="\${RUN_SUFFIX}_ttinit"
fi
if [[ -n "\$EMBEDDING_INIT_STD" ]]; then
  EMBSTD_TAG="\${EMBEDDING_INIT_STD//./p}"
  RUN_SUFFIX="\${RUN_SUFFIX}_embstd\${EMBSTD_TAG}"
fi
if [[ "\$MECHANISTIC_DIAGNOSTICS" == "1" ]]; then
  RUN_SUFFIX="\${RUN_SUFFIX}_mechanistic"
fi
if [[ "\$SPECTRAL_LR_SCALING" == "1" ]]; then
  RUN_SUFFIX="\${RUN_SUFFIX}_spectron"
fi
if [[ "\$SPECTRAL_LR_TARGET" != "all" ]]; then
  TARGET_TAG="\${SPECTRAL_LR_TARGET//[^a-zA-Z0-9]/}"
  RUN_SUFFIX="\${RUN_SUFFIX}_spectral\${TARGET_TAG}"
fi
if [[ "\$LOWRANK_FFN_LR_MULTIPLIER" != "1" && "\$LOWRANK_FFN_LR_MULTIPLIER" != "1.0" && "\$LOWRANK_FFN_LR_MULTIPLIER" != "1.00" ]]; then
  FFNLR_TAG="\${LOWRANK_FFN_LR_MULTIPLIER//./p}"
  FFNLR_TAG="\${FFNLR_TAG//e-/em}"
  FFNLR_TAG="\${FFNLR_TAG//e+/ep}"
  FFNLR_TAG="\${FFNLR_TAG//-/m}"
  RUN_SUFFIX="\${RUN_SUFFIX}_ffnlr\${FFNLR_TAG}"
fi
if [[ -n "\$LOWRANK_FFN_WEIGHT_DECAY" ]]; then
  FFNWD_TAG="\${LOWRANK_FFN_WEIGHT_DECAY//./p}"
  FFNWD_TAG="\${FFNWD_TAG//e-/em}"
  FFNWD_TAG="\${FFNWD_TAG//e+/ep}"
  FFNWD_TAG="\${FFNWD_TAG//-/m}"
  RUN_SUFFIX="\${RUN_SUFFIX}_ffnwd\${FFNWD_TAG}"
fi
if [[ -n "\$LOWRANK_ATTENTION_WEIGHT_DECAY" ]]; then
  ATTNWD_TAG="\${LOWRANK_ATTENTION_WEIGHT_DECAY//./p}"
  ATTNWD_TAG="\${ATTNWD_TAG//e-/em}"
  ATTNWD_TAG="\${ATTNWD_TAG//e+/ep}"
  ATTNWD_TAG="\${ATTNWD_TAG//-/m}"
  RUN_SUFFIX="\${RUN_SUFFIX}_attnwd\${ATTNWD_TAG}"
fi
if [[ "\$SPECTRAL_LR_SCALING_OFFSET" == "0" || "\$SPECTRAL_LR_SCALING_OFFSET" == "0.0" ]]; then
  RUN_SUFFIX="\${RUN_SUFFIX}_no_plus_one"
fi
if [[ "\$SPECTRAL_WEIGHT_DECAY" != "0" && "\$SPECTRAL_WEIGHT_DECAY" != "0.0" ]]; then
  SWD_TAG="\${SPECTRAL_WEIGHT_DECAY//./p}"
  SWD_TAG="\${SWD_TAG//e-/em}"
  SWD_TAG="\${SWD_TAG//e+/ep}"
  SWD_TAG="\${SWD_TAG//-/m}"
  RUN_SUFFIX="\${RUN_SUFFIX}_swd\${SWD_TAG}"
fi
if [[ "\$MIN_LR_FACTOR" != "0" && "\$MIN_LR_FACTOR" != "0.0" ]]; then
  MINLR_TAG="\${MIN_LR_FACTOR//./p}"
  MINLR_TAG="\${MINLR_TAG//e-/em}"
  MINLR_TAG="\${MINLR_TAG//e+/ep}"
  MINLR_TAG="\${MINLR_TAG//-/m}"
  RUN_SUFFIX="\${RUN_SUFFIX}_minlr\${MINLR_TAG}"
fi
if [[ "\$RUN_MODE" != "fullrank" && "\$LOW_RANK_RATIO" != "0.25" && "\$LOW_RANK_RATIO" != "0.250" ]]; then
  LRANK_TAG="\${LOW_RANK_RATIO//./p}"
  LRANK_TAG="\${LRANK_TAG//e-/em}"
  LRANK_TAG="\${LRANK_TAG//e+/ep}"
  LRANK_TAG="\${LRANK_TAG//-/m}"
  RUN_SUFFIX="\${RUN_SUFFIX}_rr\${LRANK_TAG}"
fi
if [[ "\$RUN_MODE" != "fullrank" && "\$LOW_RANK_W2_SAME_RANK_AS_W1W3" == "1" ]]; then
  RUN_SUFFIX="\${RUN_SUFFIX}_w2same"
fi
if [[ "\$RUN_MODE" != "fullrank" && "\$LOW_RANK_ATTENTION_FACTORIZATION" == "per_head" ]]; then
  RUN_SUFFIX="\${RUN_SUFFIX}_attnperhead"
  if [[ -n "\$LOW_RANK_ATTENTION_PER_HEAD_RANK" ]]; then
    RUN_SUFFIX="\${RUN_SUFFIX}r\${LOW_RANK_ATTENTION_PER_HEAD_RANK}"
  fi
fi
RUN_NAME="\${RUN_NAME:-spectron_${SUBMIT_MODEL_TAG}_fineweb_\${OPTIMIZER}_lr\${LR_TAG}_wd\${WD_TAG}_seq\${SEQUENCE_LENGTH}_steps\${TOTAL_STEPS}_sched\${LR_SCHEDULE_STEPS}_rope\${ROPE_THETA}_\${RUN_MODE}\${RUN_SUFFIX}}"
CHECKPOINT_DIR="\$CHECKPOINT_ROOT/\$RUN_NAME"
RESUME_FLAGS=()
if [[ -z "\${RESUME_FROM:-}" && -d "\$CHECKPOINT_DIR" ]]; then
  LATEST_CKPT="\$(find "\$CHECKPOINT_DIR" -type f \\( -name 'checkpoint_final.pt' -o -name 'checkpoint_step_*.pt' \\) -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2- || true)"
  if [[ -n "\$LATEST_CKPT" ]]; then
    RESUME_FROM="\$LATEST_CKPT"
  fi
fi
if [[ -n "\${RESUME_FROM:-}" ]]; then
  RESUME_FLAGS=(--resume_from "\$RESUME_FROM")
fi
if [[ "\$REQUIRE_RESUME" == "1" && \${#RESUME_FLAGS[@]} -eq 0 ]]; then
  echo "REQUIRE_RESUME=1 but no completed checkpoint exists in \$CHECKPOINT_DIR" >&2
  exit 1
fi
mkdir -p "\$WANDB_DIR"
export WANDB_DIR

LOW_RANK_FLAGS=()
case "\$RUN_MODE" in
  fullrank)
    ;;
  lowrank_all)
    LOW_RANK_FLAGS=(
      --low_rank
      --low_rank_ratio "\$LOW_RANK_RATIO"
      --disable_c
      --exclude_modules tok_embeddings output
    )
    ;;
  lowrank_attention)
    LOW_RANK_FLAGS=(
      --low_rank
      --low_rank_ratio "\$LOW_RANK_RATIO"
      --disable_c
      --exclude_modules tok_embeddings output feed_forward
    )
    ;;
  lowrank_ffn)
    LOW_RANK_FLAGS=(
      --low_rank
      --low_rank_ratio "\$LOW_RANK_RATIO"
      --disable_c
      --exclude_modules tok_embeddings output attention
    )
    ;;
esac

if [[ "\$LOW_RANK_W2_SAME_RANK_AS_W1W3" == "1" ]]; then
  LOW_RANK_FLAGS+=(--low_rank_w2_same_rank_as_w1w3)
fi
if [[ "\$LOW_RANK_ATTENTION_FACTORIZATION" != "whole" ]]; then
  LOW_RANK_FLAGS+=(--low_rank_attention_factorization "\$LOW_RANK_ATTENTION_FACTORIZATION")
fi
if [[ -n "\$LOW_RANK_ATTENTION_PER_HEAD_RANK" ]]; then
  LOW_RANK_FLAGS+=(--low_rank_attention_per_head_rank "\$LOW_RANK_ATTENTION_PER_HEAD_RANK")
fi

FINAL_EVAL_FLAGS=()
if [[ "\$SKIP_FINAL_EVAL" == "1" ]]; then
  FINAL_EVAL_FLAGS=(--skip_final_eval)
fi

ATTN_FLAGS=()
if [[ "\$USE_FLEX_ATTN" == "1" ]]; then
  ATTN_FLAGS=(--use_flex_attn)
fi

INIT_FLAGS=()
if [[ "\$TT_STYLE_INIT" == "1" ]]; then
  INIT_FLAGS=(--tt_style_init)
fi
if [[ -n "\$EMBEDDING_INIT_STD" ]]; then
  INIT_FLAGS+=(--embedding_init_std "\$EMBEDDING_INIT_STD")
fi

MECHANISTIC_FLAGS=()
if [[ "\$MECHANISTIC_DIAGNOSTICS" == "1" ]]; then
  if [[ -z "\$MECHANISTIC_DIAGNOSTIC_BATCH" ]]; then
    echo "MECHANISTIC_DIAGNOSTIC_BATCH is required" >&2
    exit 1
  fi
  MECHANISTIC_FLAGS=(
    --mechanistic_diagnostics
    --mechanistic_diagnostic_batch "\$MECHANISTIC_DIAGNOSTIC_BATCH"
    --mechanistic_output_dir "\$CHECKPOINT_DIR/mechanistic_diagnostics"
  )
fi

SPECTRAL_FLAGS=()
if [[ "\$SPECTRAL_LR_SCALING" == "1" ]]; then
  SPECTRAL_FLAGS+=(--spectral_lr_scaling)
fi
if [[ "\$SPECTRAL_WEIGHT_DECAY" != "0" && "\$SPECTRAL_WEIGHT_DECAY" != "0.0" ]]; then
  SPECTRAL_FLAGS+=(--spectral_weight_decay "\$SPECTRAL_WEIGHT_DECAY" --swd_type "\$SWD_TYPE")
fi
if [[ -n "\$LOWRANK_FFN_WEIGHT_DECAY" ]]; then
  SPECTRAL_FLAGS+=(--lowrank_ffn_weight_decay "\$LOWRANK_FFN_WEIGHT_DECAY")
fi
if [[ -n "\$LOWRANK_ATTENTION_WEIGHT_DECAY" ]]; then
  SPECTRAL_FLAGS+=(--lowrank_attention_weight_decay "\$LOWRANK_ATTENTION_WEIGHT_DECAY")
fi

LOWRANK_LR_TRACK_FLAGS=()
if [[ "\$TRACK_LOWRANK_LR_INTERVAL" != "0" ]]; then
  LOWRANK_LR_TRACK_FLAGS+=(
    --track_lowrank_lr_interval "\$TRACK_LOWRANK_LR_INTERVAL"
    --track_lowrank_lr_module_type "\$TRACK_LOWRANK_LR_MODULE_TYPE"
  )
  if [[ -n "\$LOWRANK_LR_LOG_PATH" ]]; then
    LOWRANK_LR_TRACK_FLAGS+=(--track_lowrank_lr_jsonl "\$LOWRANK_LR_LOG_PATH")
  fi
fi

echo "Spectron TT-matched run"
echo "  model_size=\$MODEL_SIZE_LABEL hidden=\$HIDDEN_SIZE layers=\$NUM_LAYERS heads=\$NUM_HEADS kv_heads=\$N_KV_HEADS"
echo "  optimizer=\$OPTIMIZER adjust_muon_lr=\$ADJUST_MUON_LR run_mode=\$RUN_MODE rope_theta=\$ROPE_THETA seq_len=\$SEQUENCE_LENGTH total_steps=\$TOTAL_STEPS lr_schedule_steps=\$LR_SCHEDULE_STEPS"
echo "  lr=\$MAX_LR min_lr_factor=\$MIN_LR_FACTOR warmup_start_factor=\$WARMUP_START_FACTOR weight_decay=\$WEIGHT_DECAY nh_weight_decay=\$NH_WEIGHT_DECAY batch=\$GLOBAL_BATCH_SIZE micro_batch=\$MICRO_BATCH_SIZE"
echo "  use_flex_attn=\$USE_FLEX_ATTN tt_style_init=\$TT_STYLE_INIT embedding_init_std=\${EMBEDDING_INIT_STD:-default}"
echo "  mechanistic_diagnostics=\$MECHANISTIC_DIAGNOSTICS diagnostic_batch=\${MECHANISTIC_DIAGNOSTIC_BATCH:-none}"
echo "  spectral_lr_scaling=\$SPECTRAL_LR_SCALING spectral_lr_scaling_offset=\$SPECTRAL_LR_SCALING_OFFSET spectral_lr_target=\$SPECTRAL_LR_TARGET spectral_weight_decay=\$SPECTRAL_WEIGHT_DECAY swd_type=\$SWD_TYPE"
echo "  lowrank_ffn_lr_multiplier=\$LOWRANK_FFN_LR_MULTIPLIER"
echo "  lowrank_ffn_weight_decay=\${LOWRANK_FFN_WEIGHT_DECAY:-base} lowrank_attention_weight_decay=\${LOWRANK_ATTENTION_WEIGHT_DECAY:-base}"
echo "  track_lowrank_lr_interval=\$TRACK_LOWRANK_LR_INTERVAL track_lowrank_lr_module_type=\$TRACK_LOWRANK_LR_MODULE_TYPE lowrank_lr_log_path=\${LOWRANK_LR_LOG_PATH:-default}"
echo "  low_rank_ratio=\$LOW_RANK_RATIO low_rank_w2_same_rank_as_w1w3=\$LOW_RANK_W2_SAME_RANK_AS_W1W3"
echo "  low_rank_attention_factorization=\$LOW_RANK_ATTENTION_FACTORIZATION low_rank_attention_per_head_rank=\${LOW_RANK_ATTENTION_PER_HEAD_RANK:-auto}"
echo "  log_interval=\$LOG_INTERVAL eval_interval=\$EVAL_INTERVAL skip_final_eval=\$SKIP_FINAL_EVAL"
echo "  data_root=\$DATA_ROOT"
echo "  run_name=\$RUN_NAME"
echo "  checkpoint_dir=\$CHECKPOINT_DIR"
echo "  resume_from=\${RESUME_FROM:-none}"
echo "  require_resume=\$REQUIRE_RESUME"
echo "  checkpoint_interval_steps=\$CHECKPOINT_INTERVAL_STEPS checkpoint_keep_latest_k=\$CHECKPOINT_KEEP_LATEST_K"
echo "  wandb_mode=\$WANDB_MODE"
echo "  wandb_dir=\$WANDB_DIR"

MASTER_PORT="\${MASTER_PORT:-\$((20000 + SLURM_JOB_ID % 40000))}"
echo "  master_port=\$MASTER_PORT"

exec torchrun --nproc_per_node="\$NPROC_PER_NODE" --master_port "\$MASTER_PORT" simple_gpt_training.py \\
  --seed 1234 \\
  --hidden_size "\$HIDDEN_SIZE" \\
  --num_layers "\$NUM_LAYERS" \\
  --num_heads "\$NUM_HEADS" \\
  --n_kv_heads "\$N_KV_HEADS" \\
  --vocab_size "\$VOCAB_SIZE" \\
  --max_position_embeddings "\$SEQUENCE_LENGTH" \\
  --train_seq_len "\$SEQUENCE_LENGTH" \\
  --val_seq_len "\$SEQUENCE_LENGTH" \\
  --multiple_of "\$MULTIPLE_OF" \\
  --rope_theta "\$ROPE_THETA" \\
  --optimizer "\$OPTIMIZER" \\
  --adjust_muon_lr "\$ADJUST_MUON_LR" \\
  --max_lr "\$MAX_LR" \\
  --weight_decay "\$WEIGHT_DECAY" \\
  --nh_weight_decay "\$NH_WEIGHT_DECAY" \\
  --adam_beta1 0.9 \\
  --adam_beta2 0.95 \\
  --scheduler cosine \\
  --min_lr_factor "\$MIN_LR_FACTOR" \\
  --batch_size "\$GLOBAL_BATCH_SIZE" \\
  --micro_batch_size "\$MICRO_BATCH_SIZE" \\
  --total_steps "\$TOTAL_STEPS" \\
  --lr_schedule_steps "\$LR_SCHEDULE_STEPS" \\
  --warmup_ratio 0.05 \\
  --warmup_start_factor "\$WARMUP_START_FACTOR" \\
  --log_interval "\$LOG_INTERVAL" \\
  --eval_interval "\$EVAL_INTERVAL" \\
  "\${ATTN_FLAGS[@]}" \\
  "\${INIT_FLAGS[@]}" \\
  "\${MECHANISTIC_FLAGS[@]}" \\
  "\${SPECTRAL_FLAGS[@]}" \\
  "\${LOWRANK_LR_TRACK_FLAGS[@]}" \\
  "\${LOW_RANK_FLAGS[@]}" \\
  --spectral_lr_scaling_offset "\$SPECTRAL_LR_SCALING_OFFSET" \\
  --spectral_lr_target "\$SPECTRAL_LR_TARGET" \\
  --lowrank_ffn_lr_multiplier "\$LOWRANK_FFN_LR_MULTIPLIER" \\
  --bf16 \\
  --virtual_workers_per_gpu 1 \\
  --max_val_samples "\$MAX_VAL_SAMPLES" \\
  --checkpoint_interval_hours "\$CHECKPOINT_INTERVAL_HOURS" \\
  --checkpoint_interval_steps "\$CHECKPOINT_INTERVAL_STEPS" \\
  --checkpoint_keep_latest_k "\$CHECKPOINT_KEEP_LATEST_K" \\
  --train_files "\$DATA_ROOT/fineweb_train_*.bin" \\
  --val_files "\$DATA_ROOT/fineweb_val_*.bin" \\
  --checkpoint_dir "\$CHECKPOINT_DIR" \\
  --wandb_project "\$WANDB_PROJECT" \\
  --wandb_entity "\$WANDB_ENTITY" \\
  --run_name "\$RUN_NAME" \\
  "\${RESUME_FLAGS[@]}" \\
  "\${FINAL_EVAL_FLAGS[@]}"
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Prepared $JOB_SCRIPT"
  echo "DRY_RUN=1 set, not submitting."
else
  echo "Submitting $JOB_SCRIPT"
  sbatch "$JOB_SCRIPT"
fi
