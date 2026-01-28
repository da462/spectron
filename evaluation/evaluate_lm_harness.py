#!/usr/bin/env python3
"""
Evaluation script for SimpleSubnet models using lm_evaluation_harness.

This script loads trained GPT/TitanGPT models and evaluates them on benchmarks
like hellaswag, arc, winogrande, etc. using the lm_evaluation_harness library.

Usage:
    # Evaluate TitanGPT model on hellaswag
    python evaluate_lm_harness.py --checkpoint path/to/checkpoint.pt \
        --model_type titan --tasks hellaswag

    # Evaluate on multiple tasks
    python evaluate_lm_harness.py --checkpoint path/to/checkpoint.pt \
        --model_type titan --tasks hellaswag,arc_easy,arc_challenge,winogrande

    # For simple GPT model
    python evaluate_lm_harness.py --checkpoint path/to/checkpoint.pt \
        --model_type simple --tasks hellaswag

Prerequisites:
    pip install lm-eval>=0.4.0
"""

import torch
import argparse
import os
import sys
import csv
from typing import Optional, Union, List
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import model implementations
from titan_gpt import TitanGPT, TitanModelArgs
from simple_gpt_model import GPT
from low_rank_linear import replace_linear_with_lowrank
from self_guided_linear import replace_linear_with_selfguided, LowRankWithSelfGuided

# lm_eval imports
try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    print("Warning: lm_eval not available. Install with: pip install lm-eval>=0.4.0")


class SimpleSubnetLM(LM):
    """
    Wrapper class to make SimpleSubnet models compatible with lm_evaluation_harness.

    This class implements the LM interface required by lm_eval to evaluate
    our custom GPT models on various benchmarks.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 2048,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        self._batch_size = batch_size
        self._max_length = max_length

        # Get vocab size from model
        if hasattr(model, 'output'):  # TitanGPT
            self.vocab_size = model.output.out_features
        elif hasattr(model, 'lm_head'):  # Simple GPT
            self.vocab_size = model.lm_head.out_features
        else:
            raise ValueError("Unknown model type")

    @property
    def eot_token_id(self):
        """End of text token ID."""
        if hasattr(self.tokenizer, 'eos_token_id'):
            return self.tokenizer.eos_token_id
        return 0  # Default fallback

    @property
    def max_length(self):
        """Maximum sequence length the model can handle."""
        return self._max_length

    @property
    def max_gen_toks(self):
        """Maximum number of tokens to generate."""
        return 256

    @property
    def batch_size(self):
        """Batch size for evaluation."""
        return self._batch_size

    @property
    def device(self):
        """Device the model is on."""
        return self._device

    def tok_encode(self, string: str, **kwargs):
        """Encode a string into tokens."""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(string, **kwargs)
        else:
            # For tiktoken tokenizers
            return self.tokenizer.encode(string, allowed_special="all")

    def tok_decode(self, tokens: List[int], **kwargs):
        """Decode tokens into a string."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(tokens, **kwargs)
        else:
            # For tiktoken tokenizers
            return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        """
        Compute log-likelihood for each request.

        Args:
            requests: List of Instance objects with (context, continuation) in args

        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []

        with torch.no_grad():
            for request in requests:
                context, continuation = request.args
                # Encode context and continuation
                context_tokens = self.tok_encode(context)
                continuation_tokens = self.tok_encode(continuation)

                # Combine for full input
                full_tokens = context_tokens + continuation_tokens

                # Truncate if too long
                if len(full_tokens) > self.max_length:
                    full_tokens = full_tokens[-self.max_length:]

                # Convert to tensor
                input_ids = torch.tensor([full_tokens], device=self.device)

                # Get logits
                logits = self.model(input_ids)

                # Calculate log probabilities for continuation
                # We need logits at positions corresponding to continuation tokens
                context_len = len(context_tokens)

                # Make sure we have the right positions
                if len(full_tokens) > len(continuation_tokens):
                    # Get logits for positions before each continuation token
                    logits_for_cont = logits[0, context_len-1:context_len-1+len(continuation_tokens), :]

                    # Get log probabilities
                    log_probs = torch.nn.functional.log_softmax(logits_for_cont, dim=-1)

                    # Extract log probs for actual continuation tokens
                    cont_tensor = torch.tensor(continuation_tokens, device=self.device)
                    token_log_probs = log_probs[range(len(continuation_tokens)), cont_tensor]

                    # Sum log probabilities
                    log_likelihood = token_log_probs.sum().item()

                    # Check if greedy (most likely token at each position)
                    greedy_tokens = logits_for_cont.argmax(dim=-1)
                    is_greedy = (greedy_tokens == cont_tensor).all().item()
                else:
                    # Edge case: continuation is as long as or longer than context
                    log_likelihood = 0.0
                    is_greedy = False

                results.append((log_likelihood, is_greedy))

        return results

    def loglikelihood_rolling(self, requests):
        """
        Compute rolling log-likelihood for sequences.

        This is used for perplexity evaluation where the entire sequence
        is both context and continuation.
        """
        results = []

        with torch.no_grad():
            for request in requests:
                (string,) = request.args
                tokens = self.tok_encode(string)

                # Truncate if too long
                if len(tokens) > self.max_length:
                    tokens = tokens[-self.max_length:]

                if len(tokens) < 2:
                    results.append(0.0)
                    continue

                # Convert to tensor
                input_ids = torch.tensor([tokens], device=self.device)

                # Get logits
                logits = self.model(input_ids)

                # Calculate log probabilities
                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

                # Get log prob of each token given previous context
                target_tokens = torch.tensor(tokens[1:], device=self.device)
                token_log_probs = log_probs[:-1, :][range(len(target_tokens)), target_tokens]

                # Sum log probabilities
                log_likelihood = token_log_probs.sum().item()
                results.append(log_likelihood)

        return results

    def generate_until(self, requests):
        """
        Generate tokens until a stopping condition is met.

        This is used for generative tasks.
        """
        results = []

        with torch.no_grad():
            for request in requests:
                context, gen_kwargs = request.args
                # Parse generation kwargs
                max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
                until = gen_kwargs.get("until", [self.eot_token_id])

                # Encode context
                context_tokens = self.tok_encode(context)

                # Truncate context if too long
                if len(context_tokens) > self.max_length - max_gen_toks:
                    context_tokens = context_tokens[-(self.max_length - max_gen_toks):]

                # Convert to tensor
                input_ids = torch.tensor([context_tokens], device=self.device)

                # Generate
                generated_tokens = []
                for _ in range(max_gen_toks):
                    # Get logits
                    logits = self.model(input_ids)

                    # Get next token (greedy)
                    next_token = logits[0, -1, :].argmax().item()
                    generated_tokens.append(next_token)

                    # Check stopping conditions
                    if isinstance(until, list):
                        if next_token in until:
                            break
                        # Also check string matches
                        decoded = self.tok_decode(generated_tokens)
                        if any(stop_str in decoded for stop_str in until if isinstance(stop_str, str)):
                            break
                    else:
                        if next_token == until:
                            break

                    # Append to input for next iteration
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)

                    # Truncate if exceeds max length
                    if input_ids.shape[1] > self.max_length:
                        input_ids = input_ids[:, -self.max_length:]

                # Decode generated tokens
                generated_text = self.tok_decode(generated_tokens)
                results.append(generated_text)

        return results


def load_tokenizer(tokenizer_type: str = "gpt2"):
    """
    Load the appropriate tokenizer.

    Args:
        tokenizer_type: Type of tokenizer to load ("gpt2" or "llama")

    Returns:
        Tokenizer instance
    """
    if tokenizer_type == "gpt2":
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding("gpt2")
            return tokenizer
        except ImportError:
            print("tiktoken not available, trying transformers")
            from transformers import GPT2Tokenizer
            return GPT2Tokenizer.from_pretrained("gpt2")

    elif tokenizer_type == "llama":
        try:
            return AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K")
            import tiktoken
            # Try to load Llama tokenizer
            tokenizer = tiktoken.get_encoding("cl100k_base")  # Closest to Llama
            return tokenizer
        except:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K")

    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def load_titan_model(
    checkpoint_path: str,
    device: str = "cuda",
    use_low_rank: bool = False,
    use_self_guided: bool = False,
    low_rank_ratio: float = 0.5,
    exclude_modules: list = None,
    max_layers: int = None,
    layer_indices: list = None,
    disable_c: bool = False,
) -> TitanGPT:
    """
    Load a TitanGPT model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on
        use_low_rank: Whether to apply low-rank decomposition
        use_self_guided: Whether to apply self-guided training structure (for evaluation, guide is disabled)
        low_rank_ratio: Ratio for low-rank decomposition
        exclude_modules: List of module names to exclude from low-rank
        max_layers: Maximum number of layers to apply low-rank to
        layer_indices: Specific layer indices to apply low-rank to
        disable_c: If True, disable C matrix gradient in ACB factorization

    Returns:
        Loaded TitanGPT model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model args from checkpoint or use defaults
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
        if not isinstance(model_args, TitanModelArgs):
            # Convert dict to TitanModelArgs if needed
            model_args = TitanModelArgs(**model_args)
    else:
        # Use default args - you may need to adjust these
        print("Warning: No model_args found in checkpoint, using defaults")
        model_args = TitanModelArgs(
            vocab_size=32000,
            n_layers=12,
            n_heads=12,
            dim=768,
            max_seq_len=2048,
        )

    # Disable FlexAttention for evaluation to handle variable sequence lengths
    # FlexAttention uses pre-created block masks that don't work well with varying lengths
    if model_args.use_flex_attn:
        print("Note: Disabling FlexAttention for evaluation (use_flex_attn: True -> False)")
        model_args.use_flex_attn = False

    # Create model
    model = TitanGPT(model_args)

    # Apply low-rank or self-guided decomposition if requested (before loading weights)
    if use_self_guided:
        print(f"Applying self-guided structure with ratio={low_rank_ratio}")
        print(f"Original model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        model = replace_linear_with_selfguided(
            model,
            rank_ratio=low_rank_ratio,
            method="svd",
            exclude_modules=exclude_modules if exclude_modules else [],
            reduce_flop=False  # No need for stochastic computation during eval
        )

        print(f"Self-guided model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    elif use_low_rank:
        print(f"Applying low-rank decomposition with ratio={low_rank_ratio}")
        print(f"Original model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        model = replace_linear_with_lowrank(
            model,
            rank_ratio=low_rank_ratio,
            method="svd",
            exclude_modules=exclude_modules if exclude_modules else [],
            max_layers=max_layers,
            layer_indices=layer_indices,
            disable_c=disable_c,
            sanity_check=False
        )

        print(f"Low-rank model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)

    # If self-guided model, disable guide for evaluation (use only low-rank part)
    if use_self_guided:
        print("Disabling self-guided mode for evaluation (using only low-rank factorization)")
        for module in model.modules():
            if isinstance(module, LowRankWithSelfGuided):
                module.disable_self_guided()

    model.eval()
    return model


def load_simple_gpt_model(checkpoint_path: str, config: dict, device: str = "cuda") -> GPT:
    """
    Load a simple GPT model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        config: Model configuration dict
        device: Device to load model on

    Returns:
        Loaded GPT model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with config
    model = GPT(config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def save_results_to_csv(results: dict, csv_path: str, checkpoint_name: str, model_info: dict):
    """
    Save evaluation results to a CSV file.

    Args:
        results: Results dictionary from lm_eval
        csv_path: Path to the CSV file
        checkpoint_name: Name/identifier for the checkpoint
        model_info: Dictionary with model information (type, low_rank, etc.)
    """
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    
    # Prepare rows to write
    rows = []
    
    for task_name, task_results in results["results"].items():
        # Extract accuracy metrics (most common metrics)
        row = {
            "checkpoint": checkpoint_name,
            "model_type": model_info.get("model_type", "unknown"),
            "is_low_rank": model_info.get("is_low_rank", False),
            "low_rank_ratio": model_info.get("low_rank_ratio", "N/A"),
            "num_parameters": model_info.get("num_parameters", "N/A"),
            "task": task_name,
        }
        
        # Add all metrics from the task
        for metric_name, metric_value in task_results.items():
            if isinstance(metric_value, (int, float)):
                row[metric_name] = metric_value
        
        rows.append(row)
    
    # Write to CSV
    if rows:
        # Get all unique field names
        all_fields = set()
        for row in rows:
            all_fields.update(row.keys())
        
        # Define field order (common fields first, then alphabetically sorted metrics)
        common_fields = ["checkpoint", "model_type", "is_low_rank", "is_self_guided", "low_rank_ratio",
                        "num_parameters", "task"]
        metric_fields = sorted([f for f in all_fields if f not in common_fields])
        fieldnames = common_fields + metric_fields
        
        # Write to CSV
        mode = 'a' if file_exists else 'w'
        with open(csv_path, mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for row in rows:
                # Fill in missing fields with empty string
                complete_row = {field: row.get(field, '') for field in fieldnames}
                writer.writerow(complete_row)
        
        print(f"\nResults appended to {csv_path}")
    else:
        print("\nWarning: No results to save to CSV")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SimpleSubnet models using lm_evaluation_harness"
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["titan", "simple"],
        default="titan",
        help="Type of model (titan or simple GPT)"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Name/identifier for this checkpoint (used in CSV output). If not provided, uses checkpoint filename."
    )

    # Tokenizer arguments
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        choices=["gpt2", "llama"],
        help="Tokenizer to use"
    )

    # Evaluation arguments
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag",
        help="Comma-separated list of tasks to evaluate on (e.g., 'hellaswag,arc_easy,winogrande')"
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (for testing)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON format)"
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default="evaluation_results.csv",
        help="Path to CSV file for saving results (default: evaluation_results.csv)"
    )

    # Simple GPT specific arguments
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Hidden size for simple GPT model"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=2048,
        help="Maximum position embeddings"
    )

    # Low-rank arguments
    parser.add_argument(
        "--low_rank",
        action="store_true",
        help="Enable low-rank decomposition for the model"
    )
    parser.add_argument(
        "--self_guided",
        action="store_true",
        help="Enable self-guided model loading (mutually exclusive with --low_rank)"
    )
    parser.add_argument(
        "--low_rank_ratio",
        type=float,
        default=0.5,
        help="Ratio for low-rank decomposition (default: 0.5)"
    )
    parser.add_argument(
        "--exclude_modules",
        type=str,
        nargs="+",
        default=["tok_embeddings", "output"],
        help="List of module names to exclude from low-rank decomposition"
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=None,
        help="Maximum number of layers to apply low-rank to"
    )
    parser.add_argument(
        "--layer_indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific layer indices to apply low-rank to (space-separated)"
    )
    parser.add_argument(
        "--disable_c",
        action="store_true",
        help="Disable C matrix gradient in ACB factorization"
    )

    args = parser.parse_args()

    if not LM_EVAL_AVAILABLE:
        print("Error: lm_eval not available. Install with: pip install lm-eval>=0.4.0")
        return

    # Validate that self-guided and low-rank are mutually exclusive
    if args.self_guided and args.low_rank:
        print("Error: --self_guided and --low_rank are mutually exclusive. Choose one.")
        return

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # Determine checkpoint name
    if args.checkpoint_name is None:
        args.checkpoint_name = os.path.basename(os.path.dirname(args.checkpoint))
        if not args.checkpoint_name:
            args.checkpoint_name = os.path.basename(args.checkpoint)

    # Load tokenizer
    print(f"Loading {args.tokenizer} tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)

    # Load model
    print(f"Loading {args.model_type} model from {args.checkpoint}...")
    if args.model_type == "titan":
        model = load_titan_model(
            checkpoint_path=args.checkpoint,
            device=args.device,
            use_low_rank=args.low_rank,
            use_self_guided=args.self_guided,
            low_rank_ratio=args.low_rank_ratio,
            exclude_modules=args.exclude_modules,
            max_layers=args.max_layers,
            layer_indices=args.layer_indices,
            disable_c=args.disable_c,
        )
        max_length = model.model_args.max_seq_len
    else:
        config = {
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "vocab_size": args.vocab_size,
            "max_position_embeddings": args.max_position_embeddings,
            "embed_dropout": 0.1,
            "resid_dropout": 0.1,
            "attention_dropout": 0.1,
            "layer_norm_epsilon": 1e-5,
        }
        model = load_simple_gpt_model(args.checkpoint, config, args.device)
        max_length = args.max_position_embeddings

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"Model parameters: {num_params:,}")

    # Create LM wrapper
    lm = SimpleSubnetLM(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        max_length=max_length,
    )

    # Parse tasks
    tasks = [task.strip() for task in args.tasks.split(",")]
    print(f"\nEvaluating on tasks: {tasks}")

    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    for task in tasks:
        if task in results["results"]:
            task_results = results["results"][task]
            print(f"\n{task.upper()}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")

    # Save results to JSON if output path specified
    if args.output_path:
        import json
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON results saved to {args.output_path}")

    # Save results to CSV
    model_info = {
        "model_type": args.model_type,
        "is_low_rank": args.low_rank,
        "is_self_guided": args.self_guided,
        "low_rank_ratio": args.low_rank_ratio if (args.low_rank or args.self_guided) else "N/A",
        "num_parameters": num_params,
    }

    save_results_to_csv(
        results=results,
        csv_path=args.csv_output,
        checkpoint_name=args.checkpoint_name,
        model_info=model_info
    )

    print("\n" + "="*80)


if __name__ == "__main__":
    main()