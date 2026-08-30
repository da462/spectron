#!/usr/bin/env bash
set -euo pipefail

# Prepare the paper-shape 500M FFN-only low-rank Muon/Spectron comparison.
# Jobs are dry-run by default; set DRY_RUN=0 explicitly to submit them.

PROFILE="${1:-a100_4_dev2h_cpu30_whj}"
PAIR_MODE="${2:-both}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$PAIR_MODE" in
  both|muon|spectron)
    ;;
  *)
    echo "Unknown PAIR_MODE '$PAIR_MODE'. Use both, muon, or spectron." >&2
    exit 2
    ;;
esac

# Spectron paper 500M backbone. FFN-only rank-1/4 factorization keeps dense
# attention, so this is a 362.47M-parameter model rather than the paper's
# 296.93M all-matrix-factorized model.
export MODEL_TAG="tt500m"
export MODEL_SIZE_LABEL="500m-ffn-lowrank"
export HIDDEN_SIZE=1280
export NUM_LAYERS=20
export NUM_HEADS=20
export N_KV_HEADS=20
export VOCAB_SIZE=32000
export SEQUENCE_LENGTH=2048
export MULTIPLE_OF=256
export ROPE_THETA=10000

# Match the OAR dense-500M 9,307-step compute budget with this repository's
# detailed causal FlashAttention accounting. The result is 12,368 optimizer
# steps, or 12,968,787,968 training tokens at the 1M-token batch.
export TOTAL_STEPS="${TOTAL_STEPS:-12368}"
export LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-12368}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
export MAX_LR="${MAX_LR:-0.01}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export NH_WEIGHT_DECAY="${NH_WEIGHT_DECAY:-$WEIGHT_DECAY}"
export OPTIMIZER=muon
export ADJUST_MUON_LR=original
export TT_STYLE_INIT=0
export USE_FLEX_ATTN=1
export LOW_RANK_RATIO=0.25
export LOW_RANK_W2_SAME_RANK_AS_W1W3=0
export SPECTRAL_LR_SCALING_OFFSET=1.0
export SPECTRAL_WEIGHT_DECAY=0.0
export CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-250}"
export CHECKPOINT_KEEP_LATEST_K="${CHECKPOINT_KEEP_LATEST_K:-2}"
export TRACK_LOWRANK_LR_INTERVAL="${TRACK_LOWRANK_LR_INTERVAL:-100}"
export TRACK_LOWRANK_LR_MODULE_TYPE=ffn
export DRY_RUN="${DRY_RUN:-1}"

prepare_muon() {
  SPECTRAL_LR_SCALING=0 \
  SPECTRAL_LR_TARGET=all \
  LOWRANK_FFN_LR_MULTIPLIER=1.0 \
    "$SCRIPT_DIR/submit_jz_ttmatched_spectron.sh" "$PROFILE" lowrank_ffn
}

prepare_spectron() {
  local factor_lr="${SPECTRON_FFN_LR:-0.05}"
  local multiplier
  multiplier="$(python3 -c "print(float('$factor_lr') / float('$MAX_LR'))")"
  SPECTRAL_LR_SCALING=1 \
  SPECTRAL_LR_TARGET=ffn \
  LOWRANK_FFN_LR_MULTIPLIER="$multiplier" \
    "$SCRIPT_DIR/submit_jz_ttmatched_spectron.sh" "$PROFILE" lowrank_ffn
}

if [[ "$PAIR_MODE" == "both" || "$PAIR_MODE" == "muon" ]]; then
  prepare_muon
fi
if [[ "$PAIR_MODE" == "both" || "$PAIR_MODE" == "spectron" ]]; then
  prepare_spectron
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Prepared 500M FFN-only protocol without submission."
  echo "Set DRY_RUN=0 to submit; override SPECTRON_FFN_LR to test another factor LR."
fi
