"""
Self-Guided Training for Low-Rank Linear Layers

This module implements self-guided training, a baseline method where low-rank
linear layers are trained alongside a dense guide layer. Training occurs in two phases:

Phase 1 (guided): output = α·Wx + (1-α)·U(Vx) where α: 1.0→0.0
Phase 2 (low-rank): output = U(Vx)

The guide layer provides better optimization dynamics early in training, while
the low-rank factorization learns to approximate the task independently.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List


class CosineTempDecay:
    """
    Cosine temperature decay scheduler for α (guide ratio).

    Decays α from 1.0 to 0.0 using a cosine schedule over guided_steps.
    Optionally supports a linear warmup phase.

    Args:
        guided_steps: Total number of guided training steps
        warmup_steps: Number of warmup steps (linear 0→1)
        device: Device for tensor computation
    """

    def __init__(self, guided_steps: int, warmup_steps: int = 0, device='cuda'):
        self.guided_steps = guided_steps
        self.warmup_steps = warmup_steps
        self.device = device

    def get_alpha(self, step: int) -> torch.Tensor:
        """
        Get α value for current training step.

        Args:
            step: Current training step

        Returns:
            Alpha value as CUDA tensor (0.0 to 1.0)
        """
        if step < self.warmup_steps:
            # Linear warmup: 0→1
            alpha = step / self.warmup_steps
        elif step < self.guided_steps:
            # Cosine decay: 1→0
            progress = (step - self.warmup_steps) / (self.guided_steps - self.warmup_steps)
            alpha = 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Phase 2: guide disabled
            alpha = 0.0

        return torch.tensor(alpha, device=self.device)


class LowRankWithSelfGuided(nn.Module):
    """
    Low-rank linear layer with self-guided training support.

    This layer maintains both a dense guide layer (W) and low-rank factorization (UV).
    During Phase 1, outputs are mixed: α·Wx + (1-α)·U(Vx)
    During Phase 2, only low-rank is used: U(Vx)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of low-rank factorization
        bias: Whether to include bias term
        reduce_flop: If True, use stochastic guide computation (saves ~50% FLOPs)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        reduce_flop: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.reduce_flop = reduce_flop

        # Guide layer (dense weight matrix)
        self.guide_linear = nn.Linear(in_features, out_features, bias=False)

        # Low-rank factorization: W ≈ lr2 @ lr1
        # lr1 = V (rank × in_features)
        # lr2 = U (out_features × rank)
        self.lr1 = nn.Parameter(torch.empty(rank, in_features))
        self.lr2 = nn.Parameter(torch.empty(out_features, rank))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # State buffers (automatically saved in checkpoints)
        self.register_buffer('current_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('current_alpha', torch.tensor(1.0))
        self.register_buffer('self_guided_enabled', torch.tensor(True))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        # Guide layer
        nn.init.kaiming_uniform_(self.guide_linear.weight, a=math.sqrt(5))

        # Low-rank matrices
        bound_lr1 = math.sqrt(6.0 / (self.rank + self.in_features))
        bound_lr2 = math.sqrt(6.0 / (self.out_features + self.rank))
        nn.init.uniform_(self.lr1, -bound_lr1, bound_lr1)
        nn.init.uniform_(self.lr2, -bound_lr2, bound_lr2)

        # Bias
        if self.bias is not None:
            bound_bias = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound_bias, bound_bias)

    @property
    def weight(self):
        """Reconstructed low-rank weight for compatibility with analysis tools."""
        return self.lr2 @ self.lr1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-guided training.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        if not self.self_guided_enabled:
            # Phase 2: Pure low-rank
            # Compute U(Vx) using associative property
            out = x @ self.lr1.t() @ self.lr2.t()
            if self.bias is not None:
                out = out + self.bias
            return out

        # Phase 1: Self-guided training
        alpha = self.current_alpha

        if self.reduce_flop and self.training:
            # Stochastic guide: compute with probability α to save FLOPs
            if torch.rand(1, device=x.device).item() < alpha.item():
                guide_out = self.guide_linear(x)
                lowrank_out = x @ self.lr1.t() @ self.lr2.t()
                out = alpha * guide_out + (1 - alpha) * lowrank_out
            else:
                # Skip guide computation
                out = x @ self.lr1.t() @ self.lr2.t()
        else:
            # Always compute both (eval mode or reduce_flop=False)
            guide_out = self.guide_linear(x)
            lowrank_out = x @ self.lr1.t() @ self.lr2.t()
            out = alpha * guide_out + (1 - alpha) * lowrank_out

        if self.bias is not None:
            out = out + self.bias
        return out

    def update_alpha(self, alpha: torch.Tensor):
        """Update current alpha value."""
        self.current_alpha.copy_(alpha)

    def disable_self_guided(self):
        """Disable self-guided mode (phase transition to Phase 2)."""
        self.self_guided_enabled.fill_(False)
        self.current_alpha.fill_(0.0)

    @classmethod
    def from_linear(
        cls,
        linear_layer: nn.Linear,
        rank: int,
        method: str = "svd",
        reduce_flop: bool = True
    ):
        """
        Create LowRankWithSelfGuided layer from existing Linear layer.

        Args:
            linear_layer: The original Linear layer
            rank: Target rank for low-rank factorization
            method: Initialization method ('svd' or 'random')
            reduce_flop: Whether to use stochastic guide computation

        Returns:
            LowRankWithSelfGuided layer initialized from linear_layer
        """
        sg_layer = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            bias=linear_layer.bias is not None,
            reduce_flop=reduce_flop
        )

        if method == "svd":
            with torch.no_grad():
                # Perform SVD on original weight
                U, S, Vh = torch.linalg.svd(linear_layer.weight.data, full_matrices=False)
                r = min(rank, len(S))

                # Initialize guide layer with full original weight
                sg_layer.guide_linear.weight.data.copy_(linear_layer.weight.data)

                # Initialize low-rank factorization with SVD
                # Distribute singular values: U√S and √S V^T
                sqrt_s = torch.sqrt(S[:r])
                sg_layer.lr2.data = U[:, :r] * sqrt_s.unsqueeze(0)  # U√S
                sg_layer.lr1.data = sqrt_s.unsqueeze(1) * Vh[:r, :]  # √S V^T

                # Ensure contiguous memory layout
                sg_layer.lr2.data = sg_layer.lr2.data.contiguous()
                sg_layer.lr1.data = sg_layer.lr1.data.contiguous()

        # Copy bias if it exists
        if linear_layer.bias is not None:
            sg_layer.bias.data.copy_(linear_layer.bias.data)

        return sg_layer


def replace_linear_with_selfguided(
    model: nn.Module,
    rank: Optional[int] = None,
    rank_ratio: Optional[float] = None,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    method: str = "svd",
    reduce_flop: bool = True,
) -> nn.Module:
    """
    Replace Linear layers in model with LowRankWithSelfGuided layers.

    This function follows the pattern of replace_linear_with_lowrank() from
    low_rank_linear.py for consistency.

    Args:
        model: Model to modify
        rank: Fixed rank for all layers (mutually exclusive with rank_ratio)
        rank_ratio: Ratio of rank to in_features (mutually exclusive with rank)
        target_modules: List of module name patterns to replace (None = all Linear)
        exclude_modules: List of module name patterns to exclude
        method: Initialization method ('svd' or 'random')
        reduce_flop: Whether to use stochastic guide computation

    Returns:
        Modified model with LowRankWithSelfGuided layers
    """
    if target_modules is None:
        target_modules = []
    if exclude_modules is None:
        exclude_modules = []

    # Validate rank parameters (same logic as low_rank_linear.py:365-370)
    if rank is not None and rank_ratio is not None:
        raise ValueError("Cannot specify both 'rank' and 'rank_ratio'.")
    if rank is None and rank_ratio is None:
        raise ValueError("Must specify either 'rank' or 'rank_ratio'.")
    if rank_ratio is not None and (rank_ratio <= 0 or rank_ratio > 1):
        raise ValueError("rank_ratio must be between 0 and 1.")

    def should_replace(name: str, module: nn.Module) -> bool:
        """Determine if module should be replaced."""
        if not isinstance(module, nn.Linear):
            return False

        # Check exclusions
        for exclude in exclude_modules:
            if exclude in name:
                return False

        # Check targets
        if target_modules:
            return any(target in name for target in target_modules)

        return True

    # Collect eligible modules
    eligible_modules = []
    for name, module in model.named_modules():
        if should_replace(name, module):
            eligible_modules.append((name, module))

    modules_to_replace = {name: module for name, module in eligible_modules}

    print(f"Total eligible Linear layers: {len(eligible_modules)}")
    print(f"Layers to replace with self-guided: {len(modules_to_replace)}")

    if eligible_modules:
        print("Eligible layers:")
        for i, (name, module) in enumerate(eligible_modules):
            if rank_ratio is not None:
                calc_rank = max(1, int(rank_ratio * module.in_features))
                print(f"  [{i:2d}] ✓ {name}: {module.weight.shape} -> rank {calc_rank}")
            else:
                print(f"  [{i:2d}] ✓ {name}: {module.weight.shape} -> rank {rank}")

    # Replace modules
    for name, linear_module in modules_to_replace.items():
        *parent_path, attr_name = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        # Calculate rank
        if rank_ratio is not None:
            calc_rank = max(1, int(rank_ratio * linear_module.in_features))
        else:
            calc_rank = rank

        # Create self-guided layer
        sg_module = LowRankWithSelfGuided.from_linear(
            linear_module,
            calc_rank,
            method,
            reduce_flop
        )

        # Replace in model
        setattr(parent, attr_name, sg_module)

    return model
