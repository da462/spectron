#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-h100_4_dev2h_cpu30_whj}"
DIAGNOSTIC_BATCH="${MECHANISTIC_DIAGNOSTIC_BATCH:-/lustre/fswork/projects/rech/qps/ulf36rc/spectron_data/mechanistic/fineweb_first4_seq2048.pt}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-20}"
BENCHMARK_TAG="${BENCHMARK_TAG:-v2}"

submit_mode() {
  local label="$1"
  local lightweight="$2"
  local product_interval="$3"
  local heavy="$4"

  MODEL_TAG="mechbench_${BENCHMARK_TAG}_${label}" \
  TOTAL_STEPS="$BENCHMARK_STEPS" \
  LR_SCHEDULE_STEPS=2234 \
  MAX_LR=5e-2 \
  OPTIMIZER=muon \
  WEIGHT_DECAY=0.01 \
  NH_WEIGHT_DECAY=0.01 \
  ADJUST_MUON_LR=original \
  EMBEDDING_INIT_STD=0.02 \
  TT_STYLE_INIT=0 \
  SPECTRAL_LR_SCALING=1 \
  SPECTRAL_LR_TARGET=ffn \
  LOWRANK_FFN_LR_MULTIPLIER=14.0 \
  LOW_RANK_RATIO=0.25 \
  GLOBAL_BATCH_SIZE=512 \
  MICRO_BATCH_SIZE=16 \
  SEQUENCE_LENGTH=2048 \
  LIGHTWEIGHT_DIAGNOSTICS="$lightweight" \
  LIGHTWEIGHT_PRODUCT_INTERVAL="$product_interval" \
  MECHANISTIC_DIAGNOSTICS="$heavy" \
  MECHANISTIC_DIAGNOSTIC_BATCH="$DIAGNOSTIC_BATCH" \
  CHECKPOINT_INTERVAL_STEPS=0 \
  SKIP_FINAL_EVAL=1 \
  WANDB_MODE=offline \
  "$SCRIPT_DIR/submit_jz_ttmatched_spectron.sh" "$PROFILE" lowrank_ffn
}

submit_mode disabled 0 0 0
submit_mode cheap 1 0 0
submit_mode product_every_step 1 1 0
submit_mode product_every5 1 5 0
submit_mode heavy 0 0 1
