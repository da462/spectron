import argparse
import math
import torch
from titan_gpt import TitanGPT, TitanModelArgs
from low_rank_linear import replace_linear_with_lowrank


def calculate_transformer_flops(batch_size, seq_len, vocab_size, hidden_size, num_layers, num_heads,
                               ffn_hidden_size=None, use_low_rank=False, rank_ratio=0.25,
                               exclude_modules=None, disable_c=False, n_kv_heads=None):
    """
    Manually calculate FLOPs for a transformer model forward pass.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of query attention heads
        ffn_hidden_size: FFN intermediate size (defaults to 4 * hidden_size)
        use_low_rank: Whether low-rank decomposition is used
        rank_ratio: Rank ratio for low-rank decomposition
        exclude_modules: List of module names excluded from low-rank decomposition
        disable_c: Whether C matrix is disabled in ACB factorization (uses AB instead)
        n_kv_heads: Number of key/value heads (defaults to num_heads)

    Returns:
        forward_flops: FLOPs for forward pass
    """
    if ffn_hidden_size is None:
        ffn_hidden_size = 4 * hidden_size

    if exclude_modules is None:
        exclude_modules = []
    if n_kv_heads is None:
        n_kv_heads = num_heads

    def module_uses_low_rank(module_name):
        if not use_low_rank:
            return False
        return not any(exclude in module_name for exclude in exclude_modules)

    def linear_flops(module_name, in_features, out_features, batch_tokens):
        """Calculate FLOPs for a linear layer: y = x @ W^T where x is (batch_tokens, in_features), W is (out_features, in_features)"""
        if module_uses_low_rank(module_name):
            rank = max(1, int(rank_ratio * in_features))
            if disable_c:
                # AB factorization: W = A @ B
                # x @ (A @ B)^T = x @ B^T @ A^T
                # x @ B^T: (batch_tokens, in_features) @ (in_features, rank) = batch_tokens * in_features * rank * 2
                # result @ A^T: (batch_tokens, rank) @ (rank, out_features) = batch_tokens * rank * out_features * 2
                return batch_tokens * in_features * rank * 2 + batch_tokens * rank * out_features * 2
            else:
                # ACB factorization: W = A @ C @ B
                # x @ (A @ C @ B)^T = x @ B^T @ C^T @ A^T
                # Three matrix multiplications:
                flops = batch_tokens * in_features * rank * 2  # x @ B^T
                flops += batch_tokens * rank * rank * 2  # result @ C^T
                flops += batch_tokens * rank * out_features * 2  # result @ A^T
                return flops
        else:
            # Regular linear: (batch_tokens, in_features) @ (in_features, out_features)
            return batch_tokens * in_features * out_features * 2

    flops = 0
    batch_tokens = batch_size * seq_len

    # Embedding lookup (no FLOPs, just memory access)
    head_dim = hidden_size // num_heads
    q_out_features = num_heads * head_dim
    kv_out_features = n_kv_heads * head_dim

    # Per-layer calculations
    for _ in range(num_layers):
        # 1. Attention projections.
        flops += linear_flops("attention.wq", hidden_size, q_out_features, batch_tokens)
        flops += linear_flops("attention.wk", hidden_size, kv_out_features, batch_tokens)
        flops += linear_flops("attention.wv", hidden_size, kv_out_features, batch_tokens)

        # 2. Attention scores: Q @ K^T
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        flops += batch_size * num_heads * seq_len * seq_len * head_dim * 2

        # 3. Attention softmax: ~5 ops per element
        # flops += batch_size * num_heads * seq_len * seq_len * 5

        # 4. Attention weighted sum: softmax @ V
        flops += batch_size * num_heads * seq_len * seq_len * head_dim * 2

        # 5. Attention output projection: (hidden_size -> hidden_size)
        flops += linear_flops("attention.wo", q_out_features, hidden_size, batch_tokens)

        # 6. SwiGLU FFN projections: W1 and W3 up, W2 down.
        flops += linear_flops("feed_forward.w1", hidden_size, ffn_hidden_size, batch_tokens)
        flops += linear_flops("feed_forward.w3", hidden_size, ffn_hidden_size, batch_tokens)

        # 7. FFN activation (SwiGLU or GeLU): ~8 ops per element
        # flops += batch_tokens * ffn_hidden_size * 8

        # 8. FFN down projection: (ffn_hidden_size -> hidden_size)
        flops += linear_flops("feed_forward.w2", ffn_hidden_size, hidden_size, batch_tokens)

        # 9. Layer norms: ~5 ops per element (mean, variance, normalize, scale, shift)
        # Two layer norms per transformer layer
        # flops += 2 * batch_tokens * hidden_size * 5

    # Final layer norm
    # flops += batch_tokens * hidden_size * 5

    # Output projection: (hidden_size -> vocab_size)
    flops += linear_flops("output", hidden_size, vocab_size, batch_tokens)

    # # Cross-entropy loss: softmax (~5 ops) + log (~3 ops) per element
    # flops += batch_tokens * vocab_size * 8

    return flops


flavors = {
    '60M': {
        'hidden_size': 512,
        'num_layers': 8,
        'num_heads': 8,
        'n_kv_heads': 8,
        'low_rank': False,
    },
    '92M': {
        'hidden_size': 640,
        'num_layers': 10,
        'num_heads': 10,
        'n_kv_heads': 10,
        'low_rank': False,
    },
    '134M': {
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'n_kv_heads': 12,
        'low_rank': False,
    },
    '150M': {
        'hidden_size': 768,
        'num_layers': 14,
        'num_heads': 12,
        'n_kv_heads': 12,
        'low_rank': False,
    },
    '220M': {
        'hidden_size': 896,
        'num_layers': 16,
        'num_heads': 14,
        'n_kv_heads': 14,
        'low_rank': False,
    },
    '325M': {
        'hidden_size': 1024,
        'num_layers': 20,
        'num_heads': 16,
        'n_kv_heads': 16,
        'low_rank': False,
    },
    '500M': {
        'hidden_size': 1280,
        'num_layers': 20,
        'num_heads': 20,
        'n_kv_heads': 20,
        'low_rank': False,
    },
    '780M': {
        'hidden_size': 1536,
        'num_layers': 24,
        'num_heads': 24,
        'n_kv_heads': 24,
        'low_rank': False,
    },
    '835M': {
        'hidden_size': 1536,
        'num_layers': 26,
        'num_heads': 24,
        'n_kv_heads': 24,
        'low_rank': False,
    },
    '1.4B': {
        'hidden_size': 2048,
        'num_layers': 24,
        'num_heads': 16,
        'n_kv_heads': 16,
        'low_rank': False,
    },
    '1.7B': {
        'hidden_size': 2048,
        'num_layers': 30,
        'num_heads': 16,
        'n_kv_heads': 16,
        'low_rank': False,
    },
    '2.7B': {
    'hidden_size': 2560,
    'num_layers': 32,
    'num_heads': 20,
    'n_kv_heads': 20,
    'low_rank': False,
    }
}


def main():
    parser = argparse.ArgumentParser(description='GPT Model Analysis - Parameters and FLOPs')

    # Flavor selection
    parser.add_argument('--flavor', type=str, default=None,
                        choices=list(flavors.keys()),
                        help='Model flavor to use (overrides individual model config args)')

    # Model config (used if --flavor is not specified)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--max_position_embeddings', type=int, default=1024)
    parser.add_argument('--embed_dropout', type=float, default=0.0)
    parser.add_argument('--resid_dropout', type=float, default=0.0)
    parser.add_argument('--attention_dropout', type=float, default=0.0)
    parser.add_argument('--layer_norm_epsilon', type=float, default=1e-5)
    parser.add_argument('--n_kv_heads', type=int, default=None)
    parser.add_argument('--multiple_of', type=int, default=256)
    parser.add_argument('--use_flex_attn', action='store_true', help="Enable flex attention.")

    # Low-rank training config
    parser.add_argument('--low_rank', action='store_true', help='Enable low-rank linear layer decomposition')
    parser.add_argument('--low_rank_ratio', type=float, default=0.25, help='Low-rank ratio for calculating rank as ratio * in_features (default: 0.25)')
    parser.add_argument('--max_layers', type=int, default=None, help='Maximum number of layers to apply low-rank decomposition (default: all eligible layers)')
    parser.add_argument('--layer_indices', type=int, nargs='+', default=None, help='Specific layer indices to apply low-rank decomposition (0-based indexing)')
    parser.add_argument('--exclude_modules', type=str, nargs='+', default=['tok_embeddings','output'], help='Modules to exclude from low-rank decomposition')
    parser.add_argument('--disable_c', action='store_true', help='Disable C matrix in ACB factorization (use AB factorization instead)')
    parser.add_argument('--sanity_check_lowrank', action='store_true', help='Run low-rank sanity check with A=W, C=B=I')

    # FLOP calculation config
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for FLOP calculation')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length for FLOP calculation')
    parser.add_argument(
        '--flop_match_steps',
        type=int,
        default=None,
        help='Dense full-rank step budget to match when reporting low-rank equivalent steps',
    )

    args = parser.parse_args()

    # Apply flavor configuration if specified
    if args.flavor:
        flavor_config = flavors[args.flavor]
        args.hidden_size = flavor_config['hidden_size']
        args.num_layers = flavor_config['num_layers']
        args.num_heads = flavor_config['num_heads']
        args.n_kv_heads = flavor_config['n_kv_heads']
        # Note: low_rank from flavor is ignored; use --low_rank and --low_rank_ratio instead

    print("\n" + "="*80)
    print("GPT MODEL ANALYSIS - PARAMETERS AND FLOPS")
    print("="*80 + "\n")

    # Create model
    if args.flavor:
        print(f"Using flavor: {args.flavor}")
        print()
    print("Creating model with configuration:")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Vocabulary size: {args.vocab_size}")
    print(f"  Max position embeddings: {args.max_position_embeddings}")
    print(f"  Flex attention: {args.use_flex_attn}")
    print(f"  n_kv_heads: {args.n_kv_heads}")
    print(f"  multiple_of: {args.multiple_of}")
    print()

    model_args = TitanModelArgs(
        vocab_size=args.vocab_size,
        n_layers=args.num_layers,
        n_heads=args.num_heads,
        dim=args.hidden_size,
        max_seq_len=args.max_position_embeddings,
        norm_eps=args.layer_norm_epsilon,
        use_flex_attn=args.use_flex_attn,
        n_kv_heads=args.n_kv_heads,
        multiple_of=args.multiple_of,
    )
    model = TitanGPT(model_args)

    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())

    # Count non-embedding parameters (exclude tok_embeddings and output)
    original_non_embedding_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'tok_embeddings' not in name and 'output' not in name
    )

    print(f"Original model parameters: {original_params:,} ({original_params/1e6:.2f}M)")
    print(f"  Non-embedding parameters: {original_non_embedding_params:,} ({original_non_embedding_params/1e6:.2f}M)")
    print()

    # Apply low-rank decomposition if enabled
    if args.low_rank:
        print("-" * 80)
        print("APPLYING LOW-RANK DECOMPOSITION")
        print("-" * 80)
        print(f"  Low-rank ratio: {args.low_rank_ratio}")
        print(f"  Disable C matrix (AB factorization): {args.disable_c}")
        print(f"  Exclude modules: {args.exclude_modules}")
        if args.max_layers:
            print(f"  Max layers: {args.max_layers}")
        if args.layer_indices:
            print(f"  Layer indices: {args.layer_indices}")
        if args.sanity_check_lowrank:
            print(f"  Sanity check mode: ENABLED")
        print()

        model = replace_linear_with_lowrank(
            model,
            rank_ratio=args.low_rank_ratio,
            method="svd",
            exclude_modules=args.exclude_modules,
            max_layers=args.max_layers,
            layer_indices=args.layer_indices,
            disable_c=args.disable_c,
            sanity_check=args.sanity_check_lowrank
        )

        low_rank_params = sum(p.numel() for p in model.parameters())
        reduction = (1 - low_rank_params / original_params) * 100

        # Count non-embedding parameters after low-rank
        low_rank_non_embedding_params = sum(
            p.numel() for name, p in model.named_parameters()
            if 'tok_embeddings' not in name and 'output' not in name
        )

        print(f"Low-rank model parameters: {low_rank_params:,} ({low_rank_params/1e6:.2f}M)")
        print(f"  Non-embedding parameters: {low_rank_non_embedding_params:,} ({low_rank_non_embedding_params/1e6:.2f}M)")
        print(f"Parameter reduction: {reduction:.1f}%")
        non_embedding_reduction = (1 - low_rank_non_embedding_params / original_non_embedding_params) * 100
        print(f"Non-embedding parameter reduction: {non_embedding_reduction:.1f}%")
        print()

    # Calculate FFN hidden dimension (same logic as FeedForward class)
    ffn_hidden_dim = int(2 * args.hidden_size / 3) * 4
    if args.multiple_of:
        ffn_hidden_dim = args.multiple_of * ((ffn_hidden_dim + args.multiple_of - 1) // args.multiple_of)

    print("-" * 80)
    print("CALCULATING FLOPS")
    print("-" * 80)
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  FFN hidden dimension: {ffn_hidden_dim}")
    print()

    # Calculate forward pass FLOPs (non-embedding only, as per the commented out lines)
    flops_per_step_forward = calculate_transformer_flops(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_size=ffn_hidden_dim,
        use_low_rank=args.low_rank,
        rank_ratio=args.low_rank_ratio if args.low_rank else 0.25,
        exclude_modules=args.exclude_modules if args.low_rank else [],
        disable_c=args.disable_c if args.low_rank else False,
        n_kv_heads=args.n_kv_heads,
    )

    # Estimate backward pass as 2x forward pass
    flops_per_step_backward = 2 * flops_per_step_forward
    flops_per_step_total = flops_per_step_forward + flops_per_step_backward

    # Convert to TFLOPs for readable output
    tflops_forward = flops_per_step_forward / 1e12
    tflops_backward = flops_per_step_backward / 1e12
    tflops_total = flops_per_step_total / 1e12

    # Get current non-embedding parameters
    current_non_embedding_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'tok_embeddings' not in name and 'output' not in name
    )

    # Calculate ratio of forward FLOPs to non-embedding parameters
    flops_per_param_ratio = flops_per_step_forward / (current_non_embedding_params * args.batch_size * args.seq_len)

    print(f"Forward FLOPs per batch: {tflops_forward:.3f} TFLOPs ({flops_per_step_forward:,})")
    print(f"Estimated Backward FLOPs per batch: {tflops_backward:.3f} TFLOPs ({flops_per_step_backward:,})")
    print(f"Total FLOPs per batch: {tflops_total:.3f} TFLOPs ({flops_per_step_total:,})")
    print()
    print(f"Non-embedding parameters: {current_non_embedding_params:,} ({current_non_embedding_params/1e6:.2f}M)")
    print(f"Forward FLOPs per token / Non-embedding Params ratio: {flops_per_param_ratio:.2f}")
    print()

    # Chinchilla optimal training calculation
    print("-" * 80)
    print("CHINCHILLA OPTIMAL SCALING")
    print("-" * 80)

    # Calculate tokens per step
    tokens_per_step = args.batch_size * args.seq_len

    if args.low_rank:
        # For low-rank: compute based on FULL-RANK non-embedding model parameters
        print("Computing Chinchilla optimal for FULL-RANK model (using non-embedding params):")
        print(f"  Full-rank parameters: {original_params:,} ({original_params/1e6:.2f}M)")
        print(f"  Full-rank non-embedding parameters: {original_non_embedding_params:,} ({original_non_embedding_params/1e6:.2f}M)")

        if args.flop_match_steps is not None:
            optimal_tokens_full = args.flop_match_steps * tokens_per_step
            optimal_steps_full = args.flop_match_steps
        else:
            # Chinchilla optimal for full-rank model: 20 * full_rank_non_embedding_params
            optimal_tokens_full = 20 * original_non_embedding_params
            optimal_steps_full = int(optimal_tokens_full / tokens_per_step)

        if args.flop_match_steps is not None:
            print(f"  Full-rank reference tokens: {optimal_tokens_full:,} ({optimal_tokens_full/1e9:.2f}B)")
            print(f"  Full-rank reference steps: {optimal_steps_full:,}")
        else:
            print(f"  Chinchilla optimal tokens: {optimal_tokens_full:,} ({optimal_tokens_full/1e9:.2f}B)")
            print(f"  Optimal training steps: {optimal_steps_full:,}")
        print()

        # Calculate full-rank FLOPs
        flops_per_step_full_forward = calculate_transformer_flops(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_hidden_size=ffn_hidden_dim,
            use_low_rank=False,  # Full-rank
            rank_ratio=0.25,
            exclude_modules=[],
            disable_c=False,
            n_kv_heads=args.n_kv_heads,
        )
        flops_per_step_full_total = flops_per_step_full_forward * 3  # forward + 2x backward

        # Calculate full-rank FLOPs/param ratio
        flops_per_param_ratio_full = flops_per_step_full_forward / (
            original_non_embedding_params * args.batch_size * args.seq_len
        )

        # Total FLOPs for full-rank Chinchilla optimal training
        total_flops_full_training = flops_per_step_full_total * optimal_steps_full

        # Calculate equivalent steps for low-rank model with same compute budget
        equivalent_steps_low_rank_exact = total_flops_full_training / flops_per_step_total
        equivalent_steps_low_rank_floor = math.floor(equivalent_steps_low_rank_exact)
        equivalent_steps_low_rank_ceil = math.ceil(equivalent_steps_low_rank_exact)
        equivalent_steps_low_rank = equivalent_steps_low_rank_ceil
        matched_flops_integer = equivalent_steps_low_rank * flops_per_step_total
        integer_match_error = (
            matched_flops_integer - total_flops_full_training
        ) / total_flops_full_training

        print("FLOP-Matched Training (Low-Rank Model):")
        print(f"  Full-rank FLOPs per step: {flops_per_step_full_total/1e12:.3f} TFLOPs")
        print(f"  Full-rank Forward FLOPs / Non-embedding Params: {flops_per_param_ratio_full:.2f}")
        print(f"  Total FLOPs for full-rank optimal training: {total_flops_full_training/1e12:.2f} TFLOPs")
        print(f"  Low-rank FLOPs per step: {flops_per_step_total/1e12:.3f} TFLOPs")
        print(f"  Low-rank Forward FLOPs / Non-embedding Params: {flops_per_param_ratio:.2f}")
        print(f"  Equivalent low-rank training steps exact: {equivalent_steps_low_rank_exact:,.6f}")
        print(f"  Equivalent low-rank training steps floor/ceil: {equivalent_steps_low_rank_floor:,} / {equivalent_steps_low_rank_ceil:,}")
        print(f"  Recommended integer low-rank steps: {equivalent_steps_low_rank:,}")
        print(f"  Integer-step FLOP match error: {integer_match_error:+.4%}")

        # Calculate speedup
        speedup = equivalent_steps_low_rank / optimal_steps_full
        print(f"  Speedup factor: {speedup:.2f}x more steps for same compute budget")
        print()
    else:
        # For full-rank model: standard Chinchilla calculation using non-embedding params
        current_params = sum(p.numel() for p in model.parameters())
        current_non_embedding_params = sum(
            p.numel() for name, p in model.named_parameters()
            if 'tok_embeddings' not in name and 'output' not in name
        )

        # Chinchilla optimal: train for 20 * non_embedding_params tokens
        optimal_tokens = 20 * current_non_embedding_params

        # Calculate number of steps needed
        optimal_steps = int(optimal_tokens / tokens_per_step)

        print(f"Model parameters: {current_params:,} ({current_params/1e6:.2f}M)")
        print(f"Non-embedding parameters: {current_non_embedding_params:,} ({current_non_embedding_params/1e6:.2f}M)")
        print(f"Chinchilla optimal tokens: {optimal_tokens:,} ({optimal_tokens/1e9:.2f}B)")
        print(f"Tokens per step (bs={args.batch_size} × seq_len={args.seq_len}): {tokens_per_step:,}")
        print(f"Optimal training steps: {optimal_steps:,}")
        print()

    # Summary table
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Configuration: {args.num_layers} layers, {args.hidden_size} hidden size, {args.num_heads} heads")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Non-Embedding Parameters: {current_non_embedding_params/1e6:.2f}M")
    if args.low_rank:
        print(f"  Original Total: {original_params/1e6:.2f}M")
        print(f"  Original Non-Embedding: {original_non_embedding_params/1e6:.2f}M")
        print(f"  Total Reduction: {reduction:.1f}%")
        print(f"  Non-Embedding Reduction: {non_embedding_reduction:.1f}%")
    print(f"FLOPs per batch (bs={args.batch_size}, seq_len={args.seq_len}): {tflops_total:.3f} TFLOPs")
    print(f"  Forward: {tflops_forward:.3f} TFLOPs")
    print(f"  Backward: {tflops_backward:.3f} TFLOPs")
    print(f"  Forward FLOPs / Non-Embedding Params: {flops_per_param_ratio:.2f}")

    if args.low_rank:
        if args.flop_match_steps is not None:
            print(f"Full-Rank Reference Training:")
            print(f"  Total tokens: {optimal_tokens_full/1e9:.2f}B")
            print(f"  Training steps: {optimal_steps_full:,}")
        else:
            print(f"Chinchilla Optimal Training (Full-Rank Model, Non-Embedding):")
            print(f"  Total tokens: {optimal_tokens_full/1e9:.2f}B")
            print(f"  Training steps: {optimal_steps_full:,}")
        print(f"FLOP-Matched Training (Low-Rank Model):")
        print(f"  Equivalent training steps exact: {equivalent_steps_low_rank_exact:,.6f}")
        print(f"  Recommended integer training steps: {equivalent_steps_low_rank:,}")
        print(f"  Speedup: {speedup:.2f}x")
    else:
        print(f"Chinchilla Optimal Training (Non-Embedding):")
        print(f"  Total tokens: {optimal_tokens/1e9:.2f}B")
        print(f"  Training steps: {optimal_steps:,}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
