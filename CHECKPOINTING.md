# Checkpointing Guide

This guide explains how to use the checkpointing features in the SimpleSubnet training framework.

## Features

- **Time-based checkpointing**: Automatically saves checkpoints at regular intervals (default: every 2 hours)
- **Resume training**: Load a checkpoint and continue training from where you left off
- **Final checkpoint**: Automatically saves a checkpoint when training completes
- **Complete state preservation**: Saves model, optimizers, scheduler, training progress, and metrics

## Quick Start

### Basic Usage with Checkpointing

```bash
# Train with default checkpointing (saves every 2 hours)
torchrun --nproc_per_node=4 simple_gpt_training.py \
    --checkpoint_dir ./my_checkpoints \
    --batch_size 32
```

### Resume from Checkpoint

```bash
# Resume training from a saved checkpoint
torchrun --nproc_per_node=4 simple_gpt_training.py \
    --resume_from ./my_checkpoints/checkpoint_step_1000_20240106_143022.pt \
    --checkpoint_dir ./my_checkpoints \
    --batch_size 32
```

### Custom Checkpoint Interval

```bash
# Save checkpoints every 30 minutes
torchrun --nproc_per_node=4 simple_gpt_training.py \
    --checkpoint_dir ./my_checkpoints \
    --checkpoint_interval_hours 0.5 \
    --batch_size 32
```

## Command Line Arguments

### Checkpoint Configuration

- `--checkpoint_dir`: Directory where checkpoints will be saved (default: `checkpoints`)
- `--resume_from`: Path to a checkpoint file to resume training from (default: `None`)
- `--checkpoint_interval_hours`: How often to save checkpoints in hours (default: `2.0`)
- `--save_final_checkpoint`: Whether to save a final checkpoint when training completes (default: `True`)

## Checkpoint Contents

Each checkpoint file (`.pt`) contains:

- **Model state**: Complete model weights and architecture
- **Optimizer states**:
  - Shared parameter optimizer state
  - Private parameter optimizer states (for stochastic depth training)
- **Scheduler state**: Learning rate scheduler state
- **Training progress**:
  - Current step number
  - Total tokens processed (rank 0 and world)
  - Total FLOPs computed (forward, backward, and total)
- **Configuration**: All training arguments and model configuration
- **Timestamp**: When the checkpoint was created
- **Private parameter store**: Saved states for virtual workers (if using stochastic depth)

## Checkpoint File Naming

Checkpoints are saved with descriptive filenames:

- **Regular checkpoints**: `checkpoint_step_{step}_{timestamp}.pt`
  - Example: `checkpoint_step_1000_20240106_143022.pt`
- **Final checkpoint**: `checkpoint_final.pt`

## Examples

### Example 1: Long Training with Automatic Checkpointing

```bash
# Train for many steps with checkpoints every 2 hours
torchrun --nproc_per_node=8 simple_gpt_training.py \
    --checkpoint_dir ./training_run_1 \
    --checkpoint_interval_hours 2.0 \
    --total_steps 10000 \
    --batch_size 64 \
    --wandb_project my_project
```

### Example 2: Resume After Interruption

If training was interrupted at step 3500:

```bash
# Find the latest checkpoint
ls -lt ./training_run_1/

# Resume from that checkpoint
torchrun --nproc_per_node=8 simple_gpt_training.py \
    --resume_from ./training_run_1/checkpoint_step_3500_20240106_153045.pt \
    --checkpoint_dir ./training_run_1 \
    --total_steps 10000 \
    --batch_size 64 \
    --wandb_project my_project
```

### Example 3: Frequent Checkpointing for Debugging

```bash
# Save checkpoints every 10 minutes for debugging
torchrun --nproc_per_node=4 simple_gpt_training.py \
    --checkpoint_dir ./debug_checkpoints \
    --checkpoint_interval_hours 0.167 \
    --total_steps 500 \
    --batch_size 16
```

### Example 4: Using Checkpoints with Stochastic Depth

```bash
# Train with stochastic depth and checkpointing
torchrun --nproc_per_node=4 simple_gpt_training.py \
    --checkpoint_dir ./subnet_checkpoints \
    --stochastic_depth \
    --num_overlaps 8 \
    --stochastic_depth_mode backward_nograd \
    --virtual_workers_per_gpu 2 \
    --checkpoint_interval_hours 1.5
```

## Best Practices

### 1. Checkpoint Directory Organization

Create separate directories for different experiments:

```bash
mkdir -p checkpoints/experiment_1
mkdir -p checkpoints/experiment_2
```

### 2. Checkpoint Interval Selection

- **Short jobs (< 4 hours)**: `--checkpoint_interval_hours 1.0`
- **Medium jobs (4-12 hours)**: `--checkpoint_interval_hours 2.0` (default)
- **Long jobs (> 12 hours)**: `--checkpoint_interval_hours 3.0` or `4.0`
- **Debugging**: `--checkpoint_interval_hours 0.167` (10 minutes)

### 3. Managing Disk Space

Checkpoints can be large (especially for big models). To manage disk space:

```bash
# Keep only the 3 most recent checkpoints
cd checkpoints/experiment_1
ls -t checkpoint_step_*.pt | tail -n +4 | xargs rm -f

# Or use a simple script
find . -name "checkpoint_step_*.pt" -type f -mtime +7 -delete  # Delete checkpoints older than 7 days
```

### 4. Verifying Checkpoint Integrity

```python
# Quick script to inspect a checkpoint
import torch

checkpoint = torch.load('checkpoint_step_1000.pt', map_location='cpu')
print(f"Step: {checkpoint['step']}")
print(f"Tokens processed: {checkpoint['total_tokens_world']:,}")
print(f"TFLOPs: {checkpoint['total_flops']/1e12:.2f}")
print(f"Saved at: {checkpoint['timestamp']}")
```

## Troubleshooting

### Checkpoint Not Found Error

```
FileNotFoundError: Checkpoint not found at ./checkpoints/checkpoint.pt
```

**Solution**: Verify the checkpoint path exists:
```bash
ls -lh ./checkpoints/
```

### Out of Disk Space

If you run out of disk space due to checkpoints:

1. Reduce checkpoint frequency: `--checkpoint_interval_hours 4.0`
2. Clean up old checkpoints periodically
3. Use a different storage location: `--checkpoint_dir /large_storage/checkpoints`

### Resume from Different GPU Count

If you resume with a different number of GPUs than the checkpoint was saved with:

- Model weights will load correctly
- Optimizer states will load correctly
- **Important**: Make sure `--virtual_workers_per_gpu` matches the original training

### WandB Resume

When resuming from a checkpoint, WandB will automatically attempt to resume the run:

```python
wandb.init(..., resume="allow")
```

This ensures your training curves are continuous.

## Advanced Usage

### Using Checkpoints for Evaluation

You can use saved checkpoints with the evaluation script:

```bash
python evaluation/evaluate_lm_harness.py \
    --checkpoint ./checkpoints/checkpoint_final.pt \
    --model_type titan \
    --tasks hellaswag,arc_easy
```

### Checkpoint Conversion

To convert a checkpoint to a standalone model file:

```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoint_step_1000.pt')

# Save just the model
torch.save({
    'model_state_dict': checkpoint['model_state_dict'],
    'model_args': checkpoint['model_args'],
}, 'model_only.pt')
```

### Manual Checkpoint Trigger

While the training script doesn't have a signal handler for manual checkpoints, you can:

1. Send SIGTERM to gracefully stop after current step
2. The final checkpoint will be saved automatically
3. Resume from that checkpoint

## FAQ

**Q: How much disk space does a checkpoint use?**

A: Depends on model size:
- Small model (100M params): ~400-500 MB per checkpoint
- Medium model (300M params): ~1.2-1.5 GB per checkpoint
- Large model (1B params): ~4-5 GB per checkpoint

**Q: Can I resume training on a different machine?**

A: Yes, as long as:
- The machine has the same dependencies installed
- The checkpoint file is accessible
- You use the same number of GPUs and virtual workers

**Q: What happens if training crashes between checkpoints?**

A: You'll lose progress since the last checkpoint. This is why choosing an appropriate `checkpoint_interval_hours` is important.

**Q: Can I change hyperparameters when resuming?**

A: Some hyperparameters can be changed (like `total_steps`), but others should match the original training (like model architecture, `num_layers`, `hidden_size`, etc.).

## Related Documentation

- [EVALUATION.md](EVALUATION.md) - Using checkpoints for model evaluation
- [README.md](README.md) - General training guide
