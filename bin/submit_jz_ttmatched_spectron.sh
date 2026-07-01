#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-h100_4_dev2h_cpu30_whj}"
RUN_MODE="${2:-lowrank_all}"
if [[ "$RUN_MODE" == "134m" ]]; then
  RUN_MODE="${3:-lowrank_all}"
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_DIR/jz_logs}"
JOB_DIR="${JOB_DIR:-$REPO_DIR/jobs}"
mkdir -p "$LOG_DIR" "$JOB_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
JOB_STEPS="${TOTAL_STEPS:-500}"
JOB_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-2555}"
JOB_NAME="spectron_tt134m_${RUN_MODE}_steps${JOB_STEPS}_sched${JOB_SCHEDULE_STEPS}"
JOB_SCRIPT="$JOB_DIR/${JOB_NAME}_${PROFILE}_${RUN_STAMP}.slurm"

case "$PROFILE" in
  h100_4_dev2h_cpu30_whj)
    ACCOUNT="qps@h100"
    PARTITION="gpu_p6"
    QOS="qos_gpu_h100-dev"
    CONSTRAINT="h100"
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
    echo "Unknown PROFILE '$PROFILE'. Use h100_4_dev2h_cpu30_whj or a100_dev_20m." >&2
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

./bin/run_ttmatched_spectron_rope_adamw.sh "$RUN_MODE"
EOF

echo "Submitting $JOB_SCRIPT"
sbatch "$JOB_SCRIPT"
