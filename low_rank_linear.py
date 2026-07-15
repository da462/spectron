import torch
import torch.nn as nn
import math
from typing import Optional, Union
from torch.nn import functional as F


def l2_normalize(v, eps=1e-12):
    """L2 normalize a tensor along last dimension."""
    return v / (v.norm() + eps)


def frobgrad_acb(matrices):
    """
    Compute frobenius norm gradients for ACB factorization: W = A @ C @ B
    Adapted from frobenius_factorization.py for 3-matrix decomposition
    Uses optimization technique to avoid redundant multiplications and tensor size mismatches
    """
    assert type(matrices) == list
    assert len(matrices) == 3, "ACB factorization requires exactly 3 matrices"

    A, C, B = matrices
    # Use intermediate results to avoid redundant multiplication (optimization from frobenius_factorization.py)
    AC, CB = torch.matmul(A, C), torch.matmul(C, B)

    # Compute gradients for each matrix based on frobenius norm of A @ C @ B
    # Using optimized approach siyour_clusterr to frobenius_factorization.py lines 21-24
    grad_A = torch.linalg.multi_dot([A, CB, CB.T])
    grad_C = torch.linalg.multi_dot([A.T, A, C, B, B.T])
    grad_B = torch.linalg.multi_dot([AC.T, AC, B])

    return [grad_A, grad_C, grad_B]


def apply_frobdecay(name, module, skiplist=[]):
    """Check if module should have frobenius decay applied"""
    if hasattr(module, "frobgrad"):
        return not any(name[-len(entry) :] == entry for entry in skiplist)
    return False


def frobdecay(model, coef=0.0, skiplist=[], local_rank=-1, world_size=1):
    """
    Apply Frobenius decay regularization to all modules with frobgrad method

    Args:
        model: The model to apply regularization to
        coef: Regularization coefficient
        skiplist: List of module name endings to skip
        local_rank: Current process rank (-1 if not distributed)
        world_size: Total number of processes
    """
    modules_with_frobgrad = []
    for name, module in model.named_modules():
        if apply_frobdecay(name, module, skiplist=skiplist):
            modules_with_frobgrad.append((name, module))

    # Distribute work across processes if in distributed setting
    for i, (name, module) in enumerate(modules_with_frobgrad):
        if local_rank == -1 or local_rank == i % world_size:
            module.frobgrad(coef=coef)


def get_frobenius_regularization_grads(
    model, coef=0.0, skiplist=[], local_rank=-1, world_size=1
):
    """
    Compute Frobenius decay regularization gradients for decoupled weight decay.
    Returns a dictionary mapping parameter names to regularization gradients.

    Args:
        model: The model to compute regularization for
        coef: Regularization coefficient
        skiplist: List of module name endings to skip
        local_rank: Current process rank (-1 if not distributed)
        world_size: Total number of processes

    Returns:
        dict: Mapping from parameter name to regularization gradient
    """
    regularization_grads = {}

    modules_with_frobgrad = []
    for name, module in model.named_modules():
        if apply_frobdecay(name, module, skiplist=skiplist):
            modules_with_frobgrad.append((name, module))

    # Distribute work across processes if in distributed setting
    for i, (name, module) in enumerate(modules_with_frobgrad):
        if local_rank == -1 or local_rank == i % world_size:
            # Compute Frobenius regularization gradients
            if coef > 0:
                if module.disable_c:
                    # For AB factorization
                    grad_A = coef * torch.matmul(
                        module.A, torch.matmul(module.B, module.B.T)
                    )
                    grad_B = coef * torch.matmul(
                        torch.matmul(module.A.T, module.A), module.B
                    )

                    a_param_name = f"{name}.A" if name else "A"
                    b_param_name = f"{name}.B" if name else "B"
                    regularization_grads[a_param_name] = grad_A.detach()
                    regularization_grads[b_param_name] = grad_B.detach()
                else:
                    # For ACB factorization
                    grad_A, grad_C, grad_B = frobgrad_acb(
                        [module.A.data, module.C.data, module.B.data]
                    )

                    a_param_name = f"{name}.A" if name else "A"
                    c_param_name = f"{name}.C" if name else "C"
                    b_param_name = f"{name}.B" if name else "B"
                    regularization_grads[a_param_name] = (coef * grad_A).detach()
                    regularization_grads[c_param_name] = (coef * grad_C).detach()
                    regularization_grads[b_param_name] = (coef * grad_B).detach()

    return regularization_grads


class LowRankLinear(nn.Module):
    """
    Low-rank decomposition of Linear layer: W = A @ C @ B
    where A: (out_features, r), C: (r, r), B: (r, in_features)
    When disable_c=True, the computation becomes W = A @ B (C matrix is not used)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        sanity_check: bool = False,
        disable_c: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sanity_check = sanity_check
        self.disable_c = disable_c

        if self.sanity_check:
            self.rank = in_features
            self.A = nn.Parameter(torch.empty(out_features, in_features))
            self.register_buffer("C", torch.eye(in_features))
            self.register_buffer("B", torch.eye(in_features))
            self.C.requires_grad = False
            self.B.requires_grad = False
        else:
            self.rank = rank
            # Initialize the low-rank matrices
            self.A = nn.Parameter(torch.empty(out_features, rank))
            if not disable_c:
                self.C = nn.Parameter(torch.empty(rank, rank))
            else:
                # When C is disabled, don't create it as a parameter at all
                self.register_buffer("C", torch.eye(rank))
                self.C.requires_grad = False
            self.B = nn.Parameter(torch.empty(rank, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Register u vectors for power iteration spectral norm computation
        # u_A for matrix A of shape (out_features, rank)
        # u_B for matrix B of shape (rank, in_features)
        self.register_buffer("u_A", l2_normalize(torch.randn(1, out_features)))
        self.register_buffer("u_B", l2_normalize(torch.randn(1, rank)))

        self.reset_parameters()

    @property
    def weight(self):
        """Property to compute the full weight matrix on-demand for compatibility"""
        if self.disable_c:
            return self.A @ self.B
        else:
            return self.A @ self.C @ self.B

    def reset_parameters(self):
        if self.sanity_check:
            # In sanity check mode, A is the full weight matrix
            bound_A = math.sqrt(6.0 / (self.out_features + self.in_features))
            nn.init.uniform_(self.A, -bound_A, bound_A)
        else:
            # Initialize using Kaiming uniform (siyour_clusterr to nn.Linear)
            bound_A = math.sqrt(6.0 / (self.out_features + self.rank))
            bound_B = math.sqrt(6.0 / (self.rank + self.in_features))

            nn.init.uniform_(self.A, -bound_A, bound_A)
            nn.init.uniform_(self.B, -bound_B, bound_B)

            # Only initialize C if it's not disabled
            if not self.disable_c:
                bound_C = math.sqrt(6.0 / (2 * self.rank))
                nn.init.uniform_(self.C, -bound_C, bound_C)

        if self.bias is not None:
            bound_bias = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound_bias, bound_bias)

    def forward(self, x):
        # Use associative computation to avoid materializing the full weight matrix
        # x @ (A @ C @ B)^T = ((x @ B^T) @ C^T) @ A^T (for ACB factorization)
        # x @ (A @ B)^T = (x @ B^T) @ A^T (for AB factorization)

        if self.disable_c:
            # W = A @ B, so x @ W^T = x @ B^T @ A^T
            out = x @ self.B.t() @ self.A.t()
        else:
            # W = A @ C @ B, so x @ W^T = x @ B^T @ C^T @ A^T
            out = x @ self.B.t() @ self.C.t() @ self.A.t()

        if self.bias is not None:
            out = out + self.bias

        return out

    def get_full_weight(self):
        """Return the full weight matrix W = A @ C @ B or W = A @ B when C is disabled"""
        if self.disable_c:
            return self.A @ self.B
        else:
            return self.A @ self.C @ self.B

    @torch.no_grad()
    def frobgrad(self, coef=0.0):
        """
        Apply frobenius norm regularization to the ACB factorization
        When disable_c=True, applies regularization to AB factorization only
        """
        if coef:
            if self.disable_c:
                # For AB factorization, compute gradients for A and B only
                # Frobenius norm of A @ B, siyour_clusterr to 2-matrix case
                grad_A = torch.matmul(self.A, torch.matmul(self.B, self.B.T))
                grad_B = torch.matmul(torch.matmul(self.A.T, self.A), self.B)

                # Add gradients to existing gradients or create new ones
                if self.A.grad is None:
                    self.A.grad = coef * grad_A.detach()
                else:
                    self.A.grad += coef * grad_A.detach()

                if self.B.grad is None:
                    self.B.grad = coef * grad_B.detach()
                else:
                    self.B.grad += coef * grad_B.detach()
            else:
                # Original ACB factorization
                grad_A, grad_C, grad_B = frobgrad_acb(
                    [self.A.data, self.C.data, self.B.data]
                )

                # Add gradients to existing gradients or create new ones
                if self.A.grad is None:
                    self.A.grad = coef * grad_A.detach()
                else:
                    self.A.grad += coef * grad_A.detach()

                if self.C.grad is None:
                    self.C.grad = coef * grad_C.detach()
                else:
                    self.C.grad += coef * grad_C.detach()

                if self.B.grad is None:
                    self.B.grad = coef * grad_B.detach()
                else:
                    self.B.grad += coef * grad_B.detach()

    @classmethod
    def from_linear(
        cls,
        linear_layer: nn.Linear,
        rank: int,
        method: str = "svd",
        sanity_check: bool = False,
        rank_ratio: float = None,
        disable_c: bool = False,
    ):
        """
        Create a LowRankLinear layer from an existing Linear layer

        Args:
            linear_layer: The original Linear layer
            rank: Target rank
            method: 'svd', 'random', or 'only_a' initialization
            sanity_check: If True, configure for sanity checking (A=W, C=B=I)
            rank_ratio: Ratio used for rank calculation (needed for only_a assertion)
            disable_c: If True, don't use C matrix in computation (AB factorization)
        """
        lr_layer = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            bias=linear_layer.bias is not None,
            sanity_check=sanity_check,
            disable_c=disable_c,
        )

        if sanity_check:
            # In sanity check mode, A is the original weight matrix
            lr_layer.A.data.copy_(linear_layer.weight.data)
        elif method == "only_a":
            # only_a initialization: A = weight, B and C are identity
            # This only works when rank_ratio == 1
            assert rank_ratio == 1.0, "only_a initialization requires rank_ratio == 1.0"

            with torch.no_grad():
                # A gets the original weight
                lr_layer.A.data.copy_(linear_layer.weight.data)
                # B and C are identity matrices
                lr_layer.B.data = torch.eye(rank, linear_layer.in_features)
                lr_layer.C.data = torch.eye(rank)
        elif method == "svd":
            # Use SVD to initialize low-rank approximation
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(
                    linear_layer.weight.data, full_matrices=False
                )

                # Take top-r components
                r = min(rank, len(S))

                if disable_c:
                    # For AB factorization, distribute singular values between A and B
                    sqrt_s = torch.sqrt(S[:r])
                    lr_layer.A.data = U[:, :r] * sqrt_s.unsqueeze(0)
                    # lr_layer.A.data = U[:, :r]
                    lr_layer.B.data = sqrt_s.unsqueeze(1) * Vh[:r, :]
                    # lr_layer.B.data = Vh[:r, :]
                    lr_layer.A.data = lr_layer.A.data.contiguous()
                    lr_layer.B.data = lr_layer.B.data.contiguous()
                else:
                    # For ACB factorization, keep C as identity-like
                    sqrt_s = torch.sqrt(S[:r])
                    lr_layer.A.data = U[:, :r] * sqrt_s.unsqueeze(0)
                    lr_layer.B.data = sqrt_s.unsqueeze(1) * Vh[:r, :]
                    # Initialize C as identity
                    lr_layer.C.data = torch.eye(r)

        # Copy bias if it exists
        if linear_layer.bias is not None:
            lr_layer.bias.data.copy_(linear_layer.bias.data)

        return lr_layer


class PerHeadLowRankLinear(nn.Module):
    """
    Per-head AB low-rank decomposition for attention projections.

    Row-block projections such as Wq/Wk/Wv store one low-rank matrix per output
    head block:
      A: (num_heads, head_dim, rank), B: (num_heads, rank, in_features)

    Column-block projections such as Wo store one low-rank matrix per input head
    block:
      A: (num_heads, out_features, rank), B: (num_heads, rank, head_dim)

    The module deliberately uses parameter names A/B so existing Spectron/Muon
    parameter routing can treat it like LowRankLinear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        num_heads: int,
        head_dim: int,
        block_axis: str,
        bias: bool = True,
    ):
        super().__init__()
        if block_axis not in {"row", "column"}:
            raise ValueError(f"block_axis must be 'row' or 'column', got {block_axis!r}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if block_axis == "row" and out_features != num_heads * head_dim:
            raise ValueError(
                "row-block per-head low-rank requires "
                "out_features == num_heads * head_dim"
            )
        if block_axis == "column" and in_features != num_heads * head_dim:
            raise ValueError(
                "column-block per-head low-rank requires "
                "in_features == num_heads * head_dim"
            )
        if rank > min(in_features, out_features, head_dim):
            raise ValueError(
                f"rank {rank} is too large for per-head dimensions "
                f"in={in_features}, out={out_features}, head_dim={head_dim}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_axis = block_axis
        self.disable_c = True

        if block_axis == "row":
            self.A = nn.Parameter(torch.empty(num_heads, head_dim, rank))
            self.B = nn.Parameter(torch.empty(num_heads, rank, in_features))
        else:
            self.A = nn.Parameter(torch.empty(num_heads, out_features, rank))
            self.B = nn.Parameter(torch.empty(num_heads, rank, head_dim))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        left_dim = self.A.shape[-2]
        u_A = torch.randn(num_heads, 1, left_dim)
        u_B = torch.randn(num_heads, 1, rank)
        self.register_buffer("u_A", u_A / (u_A.norm(dim=-1, keepdim=True) + 1e-12))
        self.register_buffer("u_B", u_B / (u_B.norm(dim=-1, keepdim=True) + 1e-12))

        self.reset_parameters()

    @property
    def weight(self):
        return self.get_full_weight()

    def reset_parameters(self):
        if self.block_axis == "row":
            bound_A = math.sqrt(6.0 / (self.head_dim + self.rank))
            bound_B = math.sqrt(6.0 / (self.rank + self.in_features))
        else:
            bound_A = math.sqrt(6.0 / (self.out_features + self.rank))
            bound_B = math.sqrt(6.0 / (self.rank + self.head_dim))
        nn.init.uniform_(self.A, -bound_A, bound_A)
        nn.init.uniform_(self.B, -bound_B, bound_B)

        if self.bias is not None:
            bound_bias = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound_bias, bound_bias)

    def get_full_weight(self):
        if self.block_axis == "row":
            head_weights = torch.einsum("hdr,hri->hdi", self.A, self.B)
            weight = head_weights.reshape(self.out_features, self.in_features)
        else:
            head_weights = torch.einsum("hor,hrd->hod", self.A, self.B)
            weight = (
                head_weights.permute(1, 0, 2)
                .contiguous()
                .reshape(self.out_features, self.in_features)
            )
        expected = (self.out_features, self.in_features)
        assert weight.shape == expected, (
            f"effective weight shape {tuple(weight.shape)} does not match {expected}"
        )
        return weight

    def forward(self, x):
        if self.block_axis == "row":
            hidden = torch.einsum("...i,hri->...hr", x, self.B)
            out = torch.einsum("...hr,hdr->...hd", hidden, self.A)
            out = out.reshape(*x.shape[:-1], self.out_features)
        else:
            x_by_head = x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)
            hidden = torch.einsum("...hd,hrd->...hr", x_by_head, self.B)
            out_by_head = torch.einsum("...hr,hor->...ho", hidden, self.A)
            out = out_by_head.sum(dim=-2)

        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_linear(
        cls,
        linear_layer: nn.Linear,
        rank: int,
        num_heads: int,
        head_dim: int,
        block_axis: str,
        method: str = "svd",
    ):
        layer = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            num_heads,
            head_dim,
            block_axis,
            bias=linear_layer.bias is not None,
        )

        if method == "random":
            pass
        elif method == "svd":
            with torch.no_grad():
                if block_axis == "row":
                    blocks = linear_layer.weight.data.view(
                        num_heads, head_dim, linear_layer.in_features
                    )
                else:
                    blocks = (
                        linear_layer.weight.data.view(
                            linear_layer.out_features, num_heads, head_dim
                        )
                        .permute(1, 0, 2)
                        .contiguous()
                    )

                U, S, Vh = torch.linalg.svd(blocks, full_matrices=False)
                sqrt_s = torch.sqrt(S[:, :rank])
                layer.A.data.copy_(U[:, :, :rank] * sqrt_s.unsqueeze(1))
                layer.B.data.copy_(sqrt_s.unsqueeze(2) * Vh[:, :rank, :])
        else:
            raise ValueError(
                f"Unsupported per-head low-rank initialization method {method!r}"
            )

        if linear_layer.bias is not None:
            layer.bias.data.copy_(linear_layer.bias.data)

        return layer


def replace_linear_with_lowrank(
    model: nn.Module,
    rank: int = None,
    rank_ratio: float = None,
    target_modules: Optional[list] = None,
    exclude_modules: Optional[list] = None,
    method: str = "svd",
    max_layers: Optional[int] = None,
    layer_indices: Optional[list] = None,
    disable_c: bool = False,
    sanity_check: bool = False,
    w2_same_rank_as_w1w3: bool = False,
    attention_factorization: str = "whole",
    attention_per_head_rank: Optional[int] = None,
) -> nn.Module:
    """
    Replace Linear layers in a model with LowRankLinear layers

    Args:
        model: The model to modify
        rank: Target rank for decomposition (deprecated, use rank_ratio instead)
        rank_ratio: Ratio for calculating rank as rank_ratio * in_features
        target_modules: List of module names to target (if None, targets all Linear)
        exclude_modules: List of module names to exclude
        method: Initialization method ('svd' or 'random')
        max_layers: Maximum number of layers to replace (if None, replaces all eligible)
        layer_indices: List of specific layer indices to replace (0-based indexing of eligible layers)
        disable_c: If True, set requires_grad=False for C matrices in ACB factorization
        sanity_check: If True, configure for sanity checking (A=W, C=B=I)
        w2_same_rank_as_w1w3: If True, feed_forward.w2 uses
            rank_ratio * out_features instead of rank_ratio * in_features.
        attention_factorization: "whole" keeps existing behavior. "per_head"
            replaces attention Wq/Wk/Wv/Wo with PerHeadLowRankLinear.
        attention_per_head_rank: Fixed rank per attention head. If None, uses
            rank_ratio * head_dim.

    Returns:
        Modified model with low-rank Linear layers
    """
    if attention_factorization not in {"whole", "per_head"}:
        raise ValueError(
            "attention_factorization must be 'whole' or 'per_head', got "
            f"{attention_factorization!r}"
        )
    if attention_factorization == "per_head" and not disable_c:
        raise ValueError(
            "per-head attention low-rank currently supports AB factorization only; "
            "pass disable_c=True"
        )
    if attention_per_head_rank is not None and attention_per_head_rank <= 0:
        raise ValueError("attention_per_head_rank must be positive when set")

    if target_modules is None:
        target_modules = []
    if exclude_modules is None:
        exclude_modules = []

    # Validate rank vs rank_ratio parameters
    if rank is not None and rank_ratio is not None:
        raise ValueError(
            "Cannot specify both 'rank' and 'rank_ratio'. Please use only 'rank_ratio'."
        )
    if rank is None and rank_ratio is None:
        raise ValueError("Must specify either 'rank' or 'rank_ratio'.")
    if rank_ratio is not None and (rank_ratio <= 0 or rank_ratio > 1):
        raise ValueError(
            "rank_ratio must be between 0 and 1 (exclusive of 0, inclusive of 1)."
        )

    def get_parent_module(name: str) -> tuple[nn.Module, str]:
        *parent_path, attr_name = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        return parent, attr_name

    def get_per_head_attention_spec(name: str, module: nn.Linear):
        if attention_factorization != "per_head" or ".attention." not in name:
            return None
        parent, attr_name = get_parent_module(name)
        if attr_name not in {"wq", "wk", "wv", "wo"}:
            return None
        if not all(
            hasattr(parent, attr) for attr in ("n_heads", "n_kv_heads", "head_dim")
        ):
            raise ValueError(
                f"Cannot infer attention head geometry for {name}; parent lacks "
                "n_heads/n_kv_heads/head_dim"
            )

        if attr_name in {"wq", "wo"}:
            num_heads = int(parent.n_heads)
        else:
            num_heads = int(parent.n_kv_heads)
        head_dim = int(parent.head_dim)
        block_axis = "column" if attr_name == "wo" else "row"
        expected = num_heads * head_dim
        actual = module.in_features if block_axis == "column" else module.out_features
        if actual != expected:
            raise ValueError(
                f"{name} has incompatible shape for per-head factorization: "
                f"expected {'in' if block_axis == 'column' else 'out'}_features "
                f"{expected}, got {actual}"
            )
        per_head_rank = attention_per_head_rank
        if per_head_rank is None:
            if rank_ratio is None:
                per_head_rank = rank
            else:
                per_head_rank = max(1, int(rank_ratio * head_dim))
        per_head_rank = max(
            1,
            min(int(per_head_rank), head_dim, module.in_features, module.out_features),
        )
        return {
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_axis": block_axis,
            "rank": per_head_rank,
        }

    def calculate_layer_rank(name: str, module: nn.Linear) -> int:
        per_head_spec = get_per_head_attention_spec(name, module)
        if per_head_spec is not None:
            return per_head_spec["rank"]
        if rank_ratio is None:
            return rank
        rank_features = module.in_features
        if w2_same_rank_as_w1w3 and "feed_forward.w2" in name:
            rank_features = module.out_features
        return max(1, int(rank_ratio * rank_features))

    def should_replace(name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False

        # Check exclusions
        for exclude in exclude_modules:
            if exclude in name:
                return False

        # If target_modules specified, only replace those
        if target_modules:
            return any(target in name for target in target_modules)

        return True

    # Collect all eligible modules first
    eligible_modules = []
    for name, module in model.named_modules():
        if should_replace(name, module):
            eligible_modules.append((name, module))

    # Apply layer selection logic
    modules_to_replace = {}

    if layer_indices is not None:
        # Replace only specific layer indices
        for idx in layer_indices:
            if 0 <= idx < len(eligible_modules):
                name, module = eligible_modules[idx]
                modules_to_replace[name] = module
            else:
                print(
                    f"Warning: Layer index {idx} is out of range (0-{len(eligible_modules)-1})"
                )
        print(f"Selected {len(modules_to_replace)} layers by indices: {layer_indices}")

    elif max_layers is not None:
        # Replace up to max_layers (first N eligible layers)
        for i, (name, module) in enumerate(eligible_modules[:max_layers]):
            modules_to_replace[name] = module
        print(
            f"Selected first {len(modules_to_replace)} layers (max_layers={max_layers})"
        )

    else:
        # Replace all eligible modules (original behavior)
        for name, module in eligible_modules:
            modules_to_replace[name] = module
        print(f"Selected all {len(modules_to_replace)} eligible layers")

    print(f"Total eligible Linear layers found: {len(eligible_modules)}")
    print(f"Layers to be replaced: {len(modules_to_replace)}")

    if eligible_modules:
        print("Eligible layers:")
        for i, (name, module) in enumerate(eligible_modules):
            marker = "✓" if name in modules_to_replace else " "
            if rank_ratio is not None:
                calculated_rank = calculate_layer_rank(name, module)
                per_head_spec = get_per_head_attention_spec(name, module)
                if per_head_spec is not None:
                    rank_source = (
                        f"per_head_{per_head_spec['block_axis']}_head_dim"
                    )
                else:
                    rank_source = "out_features" if (
                    w2_same_rank_as_w1w3 and "feed_forward.w2" in name
                    ) else "in_features"
                print(
                    f"  [{i:2d}] {marker} {name}: {module.weight.shape} -> rank {calculated_rank} (ratio={rank_ratio:.3f}, source={rank_source})"
                )
            else:
                print(
                    f"  [{i:2d}] {marker} {name}: {module.weight.shape} -> rank {rank}"
                )

    # Replace modules
    for name, linear_module in modules_to_replace.items():
        # Navigate to parent module
        parent, attr_name = get_parent_module(name)

        # Calculate rank for this specific layer
        if rank_ratio is not None:
            calculated_rank = calculate_layer_rank(name, linear_module)
        else:
            calculated_rank = rank

        # Create and set low-rank replacement
        per_head_spec = get_per_head_attention_spec(name, linear_module)
        if per_head_spec is not None:
            if sanity_check:
                raise ValueError(
                    "sanity_check_lowrank is not supported with per-head attention "
                    "factorization"
                )
            lr_module = PerHeadLowRankLinear.from_linear(
                linear_module,
                rank=calculated_rank,
                num_heads=per_head_spec["num_heads"],
                head_dim=per_head_spec["head_dim"],
                block_axis=per_head_spec["block_axis"],
                method=method,
            )
        else:
            lr_module = LowRankLinear.from_linear(
                linear_module,
                calculated_rank,
                method,
                sanity_check=sanity_check,
                rank_ratio=rank_ratio,
                disable_c=disable_c,
            )

        setattr(parent, attr_name, lr_module)

    return model


def count_parameters(model: nn.Module) -> tuple:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
