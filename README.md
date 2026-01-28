# Spectron - Stablizing Native low rank LLM pretraining

This repository contains the implementation for training llama style large language models with low-rank matrix factorizations. The codebase supports both baseline dense training and low-rank variants for fair FLOP-matched comparisons.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Running Experiments](#running-experiments)
  - [Local Training](#local-training)
  - [SLURM Cluster Training](#slurm-cluster-training)
- [Training Configurations](#training-configurations)
- [Experiment Scripts](#experiment-scripts)
- [Monitoring and Checkpoints](#monitoring-and-checkpoints)

## Overview

This codebase implements:

1. **Baseline Training**: Standard dense transformer models (Llama architecture)
2. **Low-Rank Training**: Matrix factorization (AB decomposition) with reduced parameters
3. **Self-Guided Training**: Progressive transition from full-rank guide layer to low-rank target layer

### Key Features

- Distributed training with PyTorch DDP
- Multiple model sizes (60M to 2.7B parameters)
- Efficient binary data format for fast loading
- Automatic checkpointing and resumption
- WandB integration for experiment tracking
- Support for multiple optimizers (AdamW, Muon)
- Spectral regularization and adaptive learning rates

## Installation

### 1. Create Python Environment

```bash
# Install uv for faster package management
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Sync with uv
uv sync
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Dataset Preparation

The codebase uses FineWeb dataset, which needs to be downloaded and converted to binary format.

### Download and Preprocess FineWeb

```bash
python parallel_make_fineweb_bin.py \
    --dataset_name HuggingFaceFW/fineweb \
    --subset sample-10BT \
    --output_dir ./data/fineweb \
    --tokenizer_name llama \
    --num_workers 48 \
    --chunks_per_worker 21
```

**Parameters:**
- `--dataset_name`: HuggingFace dataset identifier
- `--subset`: Dataset subset (e.g., `sample-10BT` for 10B token sample, `sample-100BT` for 100B)
- `--output_dir`: Where to save binary files
- `--tokenizer_name`: Tokenizer to use (`gpt2` or `llama`)
- `--num_workers`: Parallel workers (default: 48)
- `--chunks_per_worker`: Chunks per worker (default: 21, ~100M tokens each)

**Output Structure:**
```
data/fineweb/
├── fineweb_train_00.bin
├── fineweb_train_01.bin
├── ...
├── fineweb_val_00.bin
└── fineweb_val_01.bin
```

**Estimated Time:** ~2-4 hours for 10BT sample with 48 workers

### Binary Format Details

Each `.bin` file contains:
- Header (256 int32): magic number (20240520), version (1), token count, reserved fields
- Token data: uint16 array of token IDs

This format enables:
- Fast memory-mapped access
- Direct GPU loading without CPU bottleneck
- Efficient distributed shard assignment

## Running Experiments

### Local Training

For single-GPU or small-scale experiments:

```bash
# Baseline dense training (134M model)
python simple_gpt_training.py \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --n_kv_heads 12 \
    --train_files "data/fineweb/fineweb_train_*.bin" \
    --val_files "data/fineweb/fineweb_val_*.bin" \
    --batch_size 512 \
    --micro_batch_size 32 \
    --train_seq_len 2048 \
    --max_lr 0.001 \
    --weight_decay 0.1 \
    --total_steps 8000 \
    --warmup_ratio 0.1 \
    --checkpoint_dir ./checkpoints/baseline_134M \
    --wandb_project low_rank_experiments

# Low-rank training (134 model, 25% rank ratio)
python simple_gpt_training.py \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --n_kv_heads 12 \
    --low_rank \
    --low_rank_ratio 0.25 \
    --train_files "data/fineweb/fineweb_train_*.bin" \
    --val_files "data/fineweb/fineweb_val_*.bin" \
    --batch_size 512 \
    --micro_batch_size 32 \
    --train_seq_len 2048 \
    --max_lr 0.001 \
    --weight_decay 0.1 \
    --total_steps 8000 \
    --warmup_ratio 0.1 \
    --checkpoint_dir ./checkpoints/low_rank134M \
    --wandb_project low_rank_experiments

# Self-guided low-rank training
python simple_gpt_training.py \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --n_kv_heads 12 \
    --low_rank_ratio 0.25 \
    --self_guided  \
    --guided_steps_ratio 0.5 \
    --train_files "data/fineweb/fineweb_train_*.bin" \
    --val_files "data/fineweb/fineweb_val_*.bin" \
    --batch_size 512 \
    --micro_batch_size 32 \
    --train_seq_len 2048 \
    --max_lr 0.001 \
    --weight_decay 0.1 \
    --spectral_weight_decay true \
    --total_steps 8000 \
    --warmup_ratio 0.1 \
    --checkpoint_dir ./checkpoints/self_guided134M \
    --wandb_project low_rank_experiments
```

### Multi-GPU Training (Single Node)

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch with torchrun
torchrun --nproc_per_node=4 simple_gpt_training.py \
    --batch_size 64 \
    --bf16 \
    --checkpoint_dir "$HOME/checkpoints/chkptfolder/" \
    --checkpoint_interval_hours 2.5 \
    --disable_c \
    --embed_dropout 0.2 \
    --hidden_size 768 \
    --low_rank \
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
    --run_name name_you_want \
    --scheduler cosine \
    --total_steps 2600 \
    --train_files "data/fineweb_train_*.bin" \
    --train_seq_len 2048 \
    --use_flex_attn \
    --val_files "data/fineweb_val_*.bin" \
    --val_seq_len 2048 \
    --virtual_workers_per_gpu 1 \
    --vocab_size 32000 \
    --wandb_entity yourentity \
    --wandb_project low_rank \
    --warmup_ratio 0.096 \
```

### SLURM Cluster Training

The `bin/slurm_submitter.py` script provides a comprehensive interface for submitting jobs to SLURM clusters.

#### Basic Usage

```bash
# Single job submission
    python bin/slurm_submitter.py \
        --cluster your_cluster \
        --flavor model_flavor \
        --chain-length 1 \
        --multi-node \
        --job-name vanil-adamw_flavor \
        --time 3:00:00 \
        --grid optimizer=adamw \
        --param self_guided=false \
        --param low_rank=true \
        --param low_rank_ratio=0.25 \
        --param reduce_flop=true \
        --param wandb_project=your_project \
        --param wandb_entity=your_entity \
        --param scheduler=cosine \
        --grid max_lr=0.0001 \
        --param frobenius_coef=0.0 \
        --param spectral_weight_decay=0.0 \
        --param spectral_lr_scaling=false \
        --param total_steps=$steps \
        --param micro_batch_size=4 \
        --param checkpoint_interval_hours=2.5 \
        --param track_stable_rank=false \
        --param nh_weight_decay=0.1 \
        --param weight_decay=0.1
```

#### Parameter Grid Search

```bash
# Sweep over learning rates and weight decay
python bin/slurm_submitter.py \
    --cluster cluster \
    --job-name lr_sweep \
    --flavor 220M \
    --grid max_lr=0.001,0.002,0.004 \
    --grid weight_decay=0.01,0.1,0.2 \
```

This will submit 9 jobs (3 learning rates × 3 weight decay values).

#### Job Chaining (Long Training)

For training that exceeds cluster time limits, use job chaining:

```bash
python bin/slurm_submitter.py \
    --cluster cluster \
    --job-name long_training \
    --flavor 500M \
    --chain-length 5 \
    other args
```

Each job will automatically resume from the last checkpoint.

#### Multi-Node Training

```bash
python bin/slurm_submitter.py \
    --cluster cluster \
    --job-name multinode_1.4B \
    --flavor 1.4B \
    --multi-node \
    other args
```

#### Available Model Flavors

Pre-configured model sizes (use with `--flavor`):

| Flavor | Parameters | Hidden Size | Layers | Heads | Rank (25%) |
|--------|------------|-------------|--------|-------|------------|
| 60M    | 60M        | 512         | 8      | 8     | 128        |
| 92M    | 92M        | 640         | 10     | 10    | 160        |
| 134M   | 134M       | 768         | 12     | 12    | 192        |
| 150M   | 150M       | 768         | 14     | 12    | 192        |
| 220M   | 220M       | 1024        | 12     | 16    | 256        |
| 325M   | 325M       | 1152        | 18     | 18    | 288        |
| 500M   | 500M       | 1536        | 16     | 16    | 384        |
| 780M   | 780M       | 1792        | 20     | 14    | 448        |
| 1.4B   | 1.4B       | 2304        | 24     | 18    | 576        |
| 2.7B   | 2.7B       | 2816        | 32     | 22    | 704        |

## Training Configurations

### Core Arguments

**Model Architecture:**
```bash
--flavor <SIZE>                    # Use predefined model size (60M, 220M, 500M, etc.)
# OR manually specify:
--hidden_size <DIM>                # Hidden dimension
--num_layers <N>                   # Number of transformer layers
--num_heads <N>                    # Number of attention heads
--n_kv_heads <N>                   # Grouped-Query Attention (GQA) heads
```

**Low-Rank Configuration:**
```bash
--low_rank <true|false>            # Enable low-rank factorization (default: false)
--low_rank_ratio <RATIO>           # Rank as ratio of input features (e.g., 0.25 = 25%)
--disable_c <true|false>           # Use AB factorization instead of ACB (default: false)
--frobenius_coef <FLOAT>           # Frobenius norm regularization weight (default: 0.0)
--spectral_weight_decay <true|false> # Use spectral norm-based weight decay (default: false)
--spectral_lr_scaling <true|false> # Scale learning rate by spectral norms (default: false)
```

**Self-Guided Training:**
```bash
--self_guided <true|false>         # Enable self-guided training (default: false)
--guided_steps_ratio <RATIO>       # Fraction of training with guide layer (default: 0.5)
--reduce_flop <true|false>         # Stochastic guide computation to save FLOPs (default: false)
```

**Optimization:**
```bash
--optimizer <adamw|muon>           # Optimizer choice (default: adamw)
--max_lr <FLOAT>                   # Peak learning rate (e.g., 0.001)
--weight_decay <FLOAT>             # Weight decay coefficient (e.g., 0.1)
--scheduler <cosine|wsd>           # LR scheduler (default: cosine)
--warmup_ratio <RATIO>             # Warmup steps as fraction of total (default: 0.1)
--total_steps <N>                  # Total training steps
```

**Data and Batching:**
```bash
--train_files <GLOB>               # Training data glob pattern
--val_files <GLOB>                 # Validation data glob pattern
--batch_size <N>                   # Global batch size in tokens (e.g., 524288)
--micro_batch_size <N>             # Per-GPU micro batch size (e.g., 32)
--train_seq_len <N>                # Training sequence length (default: 1024)
--val_seq_len <N>                  # Validation sequence length (default: 1024)
```

**Checkpointing:**
```bash
--checkpoint_dir <PATH>            # Directory for checkpoints
--checkpoint_interval_hours <HOURS> # Auto-save frequency (default: 1.0)
--resume_from <PATH>               # Resume from specific checkpoint
```

**Logging:**
```bash
--project_name <NAME>              # WandB project name
--run_name <NAME>                  # WandB run name
--log_interval <N>                 # Steps between logging (default: 10)
```

### Example Configurations

#### Baseline Dense Training

```bash
python simple_gpt_training.py \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --n_kv_heads 12 \
    --train_files "data/fineweb/fineweb_train_*.bin" \
    --val_files "data/fineweb/fineweb_val_*.bin" \
    --batch_size 512 \
    --micro_batch_size 32 \
    --max_lr 0.001 \
    --weight_decay 0.1 \
    --optimizer adamw \
    --scheduler cosine \
    --total_steps 10000 \
    --warmup_ratio 0.1 \
    --checkpoint_dir ./checkpoints/baseline_220M
```


#### Low-Rank Training (AB Factorization)

```bash
python simple_gpt_training.py \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --n_kv_heads 12 \
    --low_rank true \
    --low_rank_ratio 0.25 \
    --disable_c true \
    --train_files "data/fineweb/fineweb_train_*.bin" \
    --val_files "data/fineweb/fineweb_val_*.bin" \
    --batch_size 512 \
    --micro_batch_size 32 \
    --max_lr 0.001 \
    --weight_decay 0.1 \
    --optimizer adamw \
    --scheduler cosine \
    --total_steps 10000 \
    --warmup_ratio 0.1 \
    --checkpoint_dir ./checkpoints/lowrank_ab_220M
```

#### Self-Guided Low-Rank Training

```bash
python simple_gpt_training.py \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --n_kv_heads 12 \
    --low_rank_ratio 0.25 \
    --self_guided true \
    --guided_steps_ratio 0.5 \
    --train_files "data/fineweb/fineweb_train_*.bin" \
    --val_files "data/fineweb/fineweb_val_*.bin" \
    --batch_size 512 \
    --micro_batch_size 32 \
    --max_lr 0.001 \
    --weight_decay 0.1 \
    --optimizer adamw \
    --scheduler cosine \
    --total_steps 10000 \
    --warmup_ratio 0.1 \
    --checkpoint_dir ./checkpoints/self_guided_220M
```


## Experiment Scripts

Pre-configured experiment scripts are available in the `bin/` directory:

### Scaling Experiments

```bash
# Different compute budgets
bash bin/scaling/second_budget.sh    # Medium-scale experiments
bash bin/scaling/third_budget.sh     # Large-scale experiments
```

### Self-Guided Training Comparisons

```bash
# Compare self-guided vs standard low-rank
bash bin/self_guided/comparisons.sh
```


### Ablation Studies

```bash
# Various ablations (rank ratios, regularization, etc.)
bash bin/ablations/rank_ablation.sh
bash bin/ablations/spectral_ablation.sh
```

## Monitoring and Checkpoints

### WandB Integration

All experiments automatically log to Weights & Biases:

```bash
# Login to WandB (first time only)
wandb login

# Training will log:
# - Training loss
# - Validation loss
# - Learning rate
# - Gradient norms
# - Spectral norms (if enabled)
# - Stable rank metrics
# - Throughput (tokens/sec)
```

### Checkpoint Structure

Checkpoints are saved as:
```
checkpoints/<experiment_name>/
├── checkpoint_step_1000.pt
├── checkpoint_step_2000.pt
└── ...
```

Each checkpoint contains:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'step': ...,
    'args': ...,  # Full configuration
    'rng_states': ...,  # For reproducibility
}
```

### Automatic Resumption

Training automatically resumes from the latest checkpoint:

```bash
# First run
python simple_gpt_training.py --checkpoint_dir ./checkpoints/exp1 --total_steps 10000

# If interrupted, just rerun the same command:
python simple_gpt_training.py --checkpoint_dir ./checkpoints/exp1 --total_steps 10000
# Will automatically detect and resume from latest checkpoint
```

Or specify a specific checkpoint:

```bash
python simple_gpt_training.py \
    --resume_from ./checkpoints/exp1/checkpoint_step_5000.pt \
    --total_steps 10000
```

## Key Implementation Details

### Low-Rank Factorization

The low-rank linear layers implement matrix factorization:

**ACB Factorization (default):**
```
W ≈ A @ C @ B
where:
  A: (out_features, rank)
  C: (rank, rank)
  B: (rank, in_features)
```

**AB Factorization (`--disable_c true`):**
```
W ≈ A @ B
where:
  A: (out_features, rank)
  B: (rank, in_features)
```

Forward pass: `output = x @ B^T @ C^T @ A^T` (no full matrix materialization)

Initialization: SVD of random Gaussian matrix with Xavier scaling

### Self-Guided Training

Self-guided training uses a curriculum approach:

1. **Phase 1 (Guided, steps 0 to guided_steps_ratio × total_steps):**
   ```
   output = α(t) · W_guide @ x + (1 - α(t)) · U(Vx)
   where α(t) decays from 1.0 to 0.0 (cosine schedule)
   ```

2. **Phase 2 (Low-Rank, remaining steps):**
   ```
   output = U(Vx)
   ```

The guide layer W_guide provides a stable training signal that gradually transitions to the low-rank factorization.


## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{anonymous2024lowrank,
  title={Low-Rank Self-Guided Training for Language Models},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This code is released for academic research purposes. See LICENSE file for details.
