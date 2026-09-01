#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-h100_4_dev2h_cpu30_whj}"
DIAGNOSTIC_BATCH="${MECHANISTIC_DIAGNOSTIC_BATCH:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_data/mechanistic/fineweb_first4_seq2048.pt}"
MATRIX_CHECKPOINT_INTERVAL_STEPS="${MATRIX_CHECKPOINT_INTERVAL_STEPS:-0}"
MATRIX_CHECKPOINT_KEEP_LATEST_K="${MATRIX_CHECKPOINT_KEEP_LATEST_K:-2}"
MATRIX_LIGHTWEIGHT_PRODUCT_INTERVAL="${MATRIX_LIGHTWEIGHT_PRODUCT_INTERVAL:-5}"
export JZ_VENV="${JZ_VENV:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron/.venv_spectron}"

submit_condition() {
  local mode="$1"
  local embedding_std="$2"
  local steps="$3"
  local spectral="$4"
  local factor_multiplier="$5"
  local spectral_target="all"
  if [[ "$spectral" == "1" ]]; then
    spectral_target="ffn"
  fi

  TOTAL_STEPS="$steps" \
  LR_SCHEDULE_STEPS="$steps" \
  MAX_LR=5e-2 \
  OPTIMIZER=muon \
  WEIGHT_DECAY=0.01 \
  NH_WEIGHT_DECAY=0.01 \
  ADJUST_MUON_LR=original \
  EMBEDDING_INIT_STD="$embedding_std" \
  TT_STYLE_INIT=0 \
  SPECTRAL_LR_SCALING="$spectral" \
  SPECTRAL_LR_SCALING_OFFSET=1.0 \
  SPECTRAL_LR_TARGET="$spectral_target" \
  SPECTRAL_WEIGHT_DECAY=0.0 \
  LOWRANK_FFN_LR_MULTIPLIER="$factor_multiplier" \
  LOW_RANK_RATIO=0.25 \
  LOW_RANK_W2_SAME_RANK_AS_W1W3=0 \
  GLOBAL_BATCH_SIZE=512 \
  MICRO_BATCH_SIZE=16 \
  SEQUENCE_LENGTH=2048 \
  MECHANISTIC_DIAGNOSTICS=1 \
  MECHANISTIC_DIAGNOSTIC_BATCH="$DIAGNOSTIC_BATCH" \
  LIGHTWEIGHT_DIAGNOSTICS=1 \
  LIGHTWEIGHT_PRODUCT_INTERVAL="$MATRIX_LIGHTWEIGHT_PRODUCT_INTERVAL" \
  CHECKPOINT_INTERVAL_STEPS="$MATRIX_CHECKPOINT_INTERVAL_STEPS" \
  CHECKPOINT_KEEP_LATEST_K="$MATRIX_CHECKPOINT_KEEP_LATEST_K" \
  SKIP_FINAL_EVAL=1 \
  WANDB_MODE=offline \
  "$SCRIPT_DIR/submit_jz_ttmatched_spectron.sh" "$PROFILE" "$mode"
}

submit_condition fullrank 1.0 1786 0 1.0
submit_condition fullrank 0.02 1786 0 1.0
submit_condition lowrank_ffn 1.0 2234 0 1.0
submit_condition lowrank_ffn 0.02 2234 0 1.0
submit_condition lowrank_ffn 1.0 2234 1 1.0
submit_condition lowrank_ffn 0.02 2234 1 1.0
submit_condition lowrank_ffn 1.0 2234 1 14.0
submit_condition lowrank_ffn 0.02 2234 1 14.0
