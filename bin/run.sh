#!/bin/bash


export WANDB_MODE="disabled"
# Set data path
export DATA_PATH="$HOME/data/fineweb"

# Set environment variables for multi-GPU training
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

# Set wandb tags
export WANDB_TAGS="tags-wd"

echo "Launching LLM training with torchrun"
echo "Working directory: $(pwd)"
echo "Data path: $DATA_PATH"
echo "Wandb tags: $WANDB_TAGS"





# Check for existing checkpoints and auto-resume if found
CHECKPOINT_DIR="$HOME/checkpoints/low_rank/150m-wsd-test_first_max_lr0p07_total_steps2600"
RESUME_FROM=""

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Checking for existing checkpoints in $CHECKPOINT_DIR..."

    # Find all checkpoint files sorted by modification time (newest first)
    CHECKPOINTS=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)

    if [ -n "$CHECKPOINTS" ]; then
        # Try to find a valid checkpoint by testing them with Python
        echo "Found checkpoint files, validating..."
        for CKPT in $CHECKPOINTS; do
            # Quick validation test: try to load the checkpoint with torch
            if python3 -c "import torch; torch.load('$CKPT', map_location='cpu'); print('✓ Valid')" 2>/dev/null; then
                export RESUME_FROM="$CKPT"
                echo "Found valid checkpoint: $RESUME_FROM"
                echo "Will resume training from this checkpoint"
                break
            else
                echo "  Skipping corrupted checkpoint: $CKPT"
            fi
        done

        if [ -z "$RESUME_FROM" ]; then
            echo "WARNING: No valid checkpoints found (all appear corrupted)"
            echo "Starting fresh training"
        fi
    else
        echo "No checkpoint files found in $CHECKPOINT_DIR"
    fi
else
    echo "No checkpoint directory found at $CHECKPOINT_DIR"
    echo "Starting fresh training"
fi


# Add resume_from parameter if checkpoint was found
if [ -n "$RESUME_FROM" ]; then
    export RESUME_ARG="--resume_from $RESUME_FROM"
else
    RESUME_ARG=""
fi


torchrun --nproc_per_node=2 \
    simple_gpt_training.py \
    --batch_size 64 \
    --bf16 \
    --checkpoint_dir "$HOME/checkpoints/low_rank/150m-cosine-test_first_max_lr0p07_total_steps2600" \
    --checkpoint_interval_hours 2.5 \
    --disable_c \
    --embed_dropout 0.2 \
    --frobenius_coef 0.0 \
    --hidden_size 768 \
    --log_interval 10 \
    --self_guided \
    --low_rank_ratio 0.25 \
    --max_lr 0.07 \
    --max_position_embeddings 2048 \
    --max_val_samples 100 \
    --micro_batch_size 16 \
    --multiple_of 256 \
    --n_kv_heads 12 \
    --num_heads 12 \
    --num_layers 12 \
    --optimizer adamw \
    --resid_dropout 0.2 \
    --run_name 150m-wsd-test_first_max_lr0p07_total_steps2600 \
    --scheduler cosine \
    --total_steps 2600 \
    --train_files "$DATA_PATH/fineweb_train_*.bin" \
    --train_seq_len 2048 \
    --use_flex_attn \
    --val_files "$DATA_PATH/fineweb_val_*.bin" \
    --val_seq_len 2048 \
    --virtual_workers_per_gpu 1 \
    --vocab_size 32000 \
    --wandb_entity your_entity \
    --wandb_project your_project \
    --warmup_ratio 0.096 \
    --weight_decay 0.0 $RESUME_ARG