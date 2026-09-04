#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-a100_4_dev2h_cpu30_whj}"
RUN_MODE="${2:-lowrank_ffn}"

case "$RUN_MODE" in
  fullrank)
    DEFAULT_STEPS=1786
    ;;
  lowrank_ffn)
    DEFAULT_STEPS=2234
    ;;
  *)
    echo "Keller cooldown supports fullrank or lowrank_ffn, got '$RUN_MODE'." >&2
    exit 2
    ;;
esac

export MODEL_TAG="${MODEL_TAG:-keller_cooldown}"
export TOTAL_STEPS="${TOTAL_STEPS:-$DEFAULT_STEPS}"
export LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-$TOTAL_STEPS}"
export MAX_LR="${MAX_LR:-5e-2}"
export OPTIMIZER=muon
export LOWRANK_OPTIMIZER=factor_muon
export ADJUST_MUON_LR="${ADJUST_MUON_LR:-original}"
export AUX_ADAMW_LR_MULTIPLIER="${AUX_ADAMW_LR_MULTIPLIER:-0.1}"
export SCHEDULER=stable_linear_decay
export WARMUP_RATIO=0
export STABLE_DECAY_FRACTION="${STABLE_DECAY_FRACTION:-0.3}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export NH_WEIGHT_DECAY="${NH_WEIGHT_DECAY:-$WEIGHT_DECAY}"
export EMBEDDING_INIT_STD="${EMBEDDING_INIT_STD:-0.02}"
export SPECTRAL_LR_SCALING=0
export LOWRANK_FFN_LR_MULTIPLIER=1.0
export LOW_RANK_RATIO="${LOW_RANK_RATIO:-0.25}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
export SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
export MECHANISTIC_DIAGNOSTICS="${MECHANISTIC_DIAGNOSTICS:-1}"
export MECHANISTIC_DIAGNOSTIC_BATCH="${MECHANISTIC_DIAGNOSTIC_BATCH:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_data/mechanistic/fineweb_first4_seq2048.pt}"
export LIGHTWEIGHT_DIAGNOSTICS="${LIGHTWEIGHT_DIAGNOSTICS:-1}"
export LIGHTWEIGHT_PRODUCT_INTERVAL="${LIGHTWEIGHT_PRODUCT_INTERVAL:-5}"
export CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-500}"
export CHECKPOINT_KEEP_LATEST_K="${CHECKPOINT_KEEP_LATEST_K:-2}"
export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

exec "$SCRIPT_DIR/submit_jz_ttmatched_spectron.sh" "$PROFILE" "$RUN_MODE"
