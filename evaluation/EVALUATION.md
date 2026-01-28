# Model Evaluation Guide

This guide explains how to use the `evaluation/evaluate_lm_harness.py` script to evaluate your trained SimpleSubnet models on standard language model benchmarks.

## Prerequisites

Install the lm-evaluation-harness library:

```bash
pip install lm-eval>=0.4.0
```

You'll also need the standard dependencies:
```bash
pip install torch tiktoken transformers
```

## Quick Start

### Basic Usage

Evaluate a TitanGPT model on HellaSwag:

```bash
python evaluation/evaluate_lm_harness.py \
    --checkpoint path/to/your/checkpoint.pt \
    --model_type titan \
    --tasks hellaswag
```

### Multiple Benchmarks

Evaluate on several benchmarks at once:

```bash
python evaluation/evaluate_lm_harness.py \
    --checkpoint path/to/your/checkpoint.pt \
    --model_type titan \
    --tasks hellaswag,arc_easy,arc_challenge,winogrande,piqa
```

## Supported Benchmarks

The script supports all benchmarks available in lm-evaluation-harness, including:

- **hellaswag**: Commonsense reasoning about physical situations
- **arc_easy/arc_challenge**: Science questions (easy and challenging)
- **winogrande**: Commonsense reasoning with pronoun resolution
- **piqa**: Physical interaction reasoning
- **openbookqa**: Open-book science questions
- **boolq**: Yes/No questions
- **lambada**: Reading comprehension
- **mmlu**: Massive multi-task language understanding
- Many more! See [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for full list

## Command Line Arguments

### Required Arguments

- `--checkpoint`: Path to your model checkpoint file (`.pt` file)
- `--model_type`: Type of model architecture (`titan` or `simple`)

### Optional Arguments

#### Evaluation Configuration
- `--tasks`: Comma-separated list of tasks (default: `hellaswag`)
- `--num_fewshot`: Number of few-shot examples (default: `0` for zero-shot)
- `--batch_size`: Batch size for evaluation (default: `1`)
- `--limit`: Limit number of examples for quick testing (default: `None` for full eval)
- `--output_path`: Path to save JSON results (default: `None`)

#### Tokenizer Configuration
- `--tokenizer`: Tokenizer type (`gpt2` or `llama`, default: `gpt2`)

#### Device Configuration
- `--device`: Device to run on (`cuda` or `cpu`, default: `cuda`)

#### Model Configuration (for Simple GPT only)
If using `--model_type simple`, you may need to specify:
- `--hidden_size`: Hidden dimension (default: `768`)
- `--num_heads`: Number of attention heads (default: `12`)
- `--num_layers`: Number of transformer layers (default: `12`)
- `--vocab_size`: Vocabulary size (default: `50257`)
- `--max_position_embeddings`: Max sequence length (default: `2048`)

## Usage Examples

### Example 1: Quick Test on Limited Examples

Test your model on 100 examples of HellaSwag:

```bash
python evaluation/evaluate_lm_harness.py \
    --checkpoint checkpoints/model_step_1000.pt \
    --model_type titan \
    --tasks hellaswag \
    --limit 100
```

### Example 2: Comprehensive Evaluation

Evaluate on multiple benchmarks and save results:

```bash
python evaluation/evaluate_lm_harness.py \
    --checkpoint checkpoints/final_model.pt \
    --model_type titan \
    --tokenizer gpt2 \
    --tasks hellaswag,arc_easy,arc_challenge,winogrande,piqa \
    --num_fewshot 0 \
    --batch_size 8 \
    --output_path results/eval_results.json
```

### Example 3: Few-Shot Evaluation

Evaluate with 5-shot prompting:

```bash
python evaluation/evaluate_lm_harness.py \
    --checkpoint checkpoints/model.pt \
    --model_type titan \
    --tasks hellaswag \
    --num_fewshot 5
```

### Example 4: Evaluating Simple GPT Model

For the simple GPT architecture:

```bash
python evaluation/evaluate_lm_harness.py \
    --checkpoint checkpoints/simple_gpt.pt \
    --model_type simple \
    --hidden_size 768 \
    --num_heads 12 \
    --num_layers 12 \
    --vocab_size 50257 \
    --tasks hellaswag,arc_easy
```

### Example 5: Using Llama Tokenizer

If your model was trained with the Llama tokenizer:

```bash
python evaluation/evaluate_lm_harness.py \
    --checkpoint checkpoints/llama_tokenizer_model.pt \
    --model_type titan \
    --tokenizer llama \
    --tasks hellaswag
```

## Understanding the Output

The script will print results for each task with relevant metrics:

```
================================================================================
EVALUATION RESULTS
================================================================================

HELLASWAG:
  acc: 0.3125
  acc_norm: 0.4250
  acc_stderr: 0.0234

ARC_EASY:
  acc: 0.5234
  acc_norm: 0.5123
  ...
```

Common metrics:
- **acc**: Raw accuracy
- **acc_norm**: Length-normalized accuracy (better for multiple choice)
- **acc_stderr**: Standard error of accuracy

## Checkpoint Format

The script expects PyTorch checkpoint files (`.pt`) with one of these formats:

### Format 1: With metadata
```python
{
    'model_state_dict': model.state_dict(),
    'model_args': model_args,  # TitanModelArgs or config dict
    # ... other training metadata
}
```

### Format 2: Direct state dict
```python
model.state_dict()  # Just the state dict
```

### Format 3: With 'model' key
```python
{
    'model': model.state_dict(),
    # ... other data
}
```

## Tips for Best Results

1. **Use larger batch sizes** when possible for faster evaluation (if you have GPU memory)
2. **Start with `--limit 100`** to quickly test if everything works before full evaluation
3. **Save results to JSON** for later analysis and comparison across checkpoints
4. **Use zero-shot (num_fewshot=0)** first, then try few-shot if needed
5. **Match the tokenizer** to what was used during training

## Troubleshooting

### OOM (Out of Memory) Errors
- Reduce `--batch_size` to 1
- Evaluate on fewer tasks at once
- Use CPU with `--device cpu` (slower but works)

### Tokenizer Issues
- Make sure the tokenizer matches your training setup
- For GPT-2 style models, use `--tokenizer gpt2`
- For Llama style models, use `--tokenizer llama`

### Checkpoint Loading Errors
- Verify the checkpoint file exists and is not corrupted
- Check that model configuration arguments match the checkpoint
- For simple GPT, explicitly provide model dimensions

### Low Scores
- Random/untrained models typically get ~25% on 4-choice tasks (random guessing)
- Small models (<100M params) may struggle on harder tasks
- Make sure the model is actually trained (not just initialized)

## Integration with Training

To save checkpoints during training that work with this script, use:

```python
# Save with metadata (recommended)
torch.save({
    'model_state_dict': model.state_dict(),
    'model_args': model_args,
    'step': step,
    'optimizer_state_dict': optimizer.state_dict(),
}, f'checkpoint_step_{step}.pt')

# Or minimal save
torch.save(model.state_dict(), f'model_step_{step}.pt')
```

## Advanced: Custom Model Wrapper

If you need to modify how the model is evaluated, you can customize the `SimpleSubnetLM` class in `evaluation/evaluate_lm_harness.py`. Key methods:

- `loglikelihood()`: For multiple-choice questions
- `loglikelihood_rolling()`: For perplexity evaluation
- `generate_until()`: For generation tasks

## Example Batch Script

For evaluating multiple checkpoints:

```bash
#!/bin/bash

CHECKPOINTS=(
    "checkpoints/step_1000.pt"
    "checkpoints/step_2000.pt"
    "checkpoints/step_3000.pt"
)

TASKS="hellaswag,arc_easy,arc_challenge,winogrande"

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "Evaluating $ckpt..."
    python evaluation/evaluate_lm_harness.py \
        --checkpoint "$ckpt" \
        --model_type titan \
        --tasks "$TASKS" \
        --output_path "results/$(basename $ckpt .pt)_results.json"
done
```

## References

- [lm-evaluation-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness)
- [HellaSwag paper](https://arxiv.org/abs/1905.07830)
- [AI2 Reasoning Challenge (ARC)](https://arxiv.org/abs/1803.05457)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your checkpoint and model configuration
3. Test with `--limit 10` to quickly debug issues
4. Check lm-evaluation-harness documentation for task-specific details
