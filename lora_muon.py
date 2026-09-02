"""Product-aware LoRA-Muon updates for native low-rank training."""

from __future__ import annotations

import math

import torch


POLAR_EXPRESS_COEFFICIENTS = (
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
    (1.8648, -1.2224, 0.3577),
)

INVERSE_ROOT_COEFFICIENTS = (
    (7.424865680309214, -18.39581635618996, 12.896720413604342),
    (3.4877256051546017, -2.3300436563986993, 0.4404692168431095),
    (2.7766085124882527, -2.070643152532662, 0.46302261050004967),
    (1.9913142104341506, -1.373936700681269, 0.3875934979568538),
    (1.8754637749479246, -1.2505152090010534, 0.37505152463617264),
    (1.874999066623701, -1.2499981332141676, 0.37499906659046633),
    (1.875, -1.25, 0.375),
)


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.mT)


def matrix_sign_newton_schulz(matrix: torch.Tensor) -> torch.Tensor:
    """Approximate the rectangular matrix sign with Polar Express steps."""
    if matrix.ndim != 2:
        raise ValueError(f"matrix sign expects a matrix, got shape {matrix.shape}")
    original_dtype = matrix.dtype
    work = matrix.float()
    transpose = work.shape[0] > work.shape[1]
    if transpose:
        work = work.mT
    work = work / work.norm().clamp_min(1e-20)
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        gram = work @ work.mT
        work = a * work + (b * gram + c * gram @ gram) @ work
    if transpose:
        work = work.mT
    return work.to(original_dtype)


def matrix_inverse_sqrt_newton_schulz(
    matrix: torch.Tensor,
    *,
    epsilon: float = 1e-5,
    gamma: float = 1.001,
) -> torch.Tensor:
    """Approximate a PSD inverse square root using LoRA-Muon Algorithm 4."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"inverse square root expects square matrix, got {matrix.shape}")
    original_dtype = matrix.dtype
    work = _symmetrize(matrix.float())
    scale = work.norm().clamp_min(1e-20)
    identity = torch.eye(work.shape[0], dtype=work.dtype, device=work.device)
    p = work / scale + epsilon * identity
    inverse_root = identity
    for a, b, c in INVERSE_ROOT_COEFFICIENTS:
        p2 = p @ p
        update = (
            (a / gamma) * identity
            + (b / gamma**3) * p
            + (c / gamma**5) * p2
        )
        inverse_root = inverse_root @ update
        p = _symmetrize(p @ update @ update)
    return (inverse_root / scale.sqrt()).to(original_dtype)


def matrix_inverse_sqrt_eigh(
    matrix: torch.Tensor,
    *,
    relative_floor: float = 1e-12,
) -> torch.Tensor:
    """Reference PSD inverse square root for tests and diagnostics."""
    work = _symmetrize(matrix.double())
    eigenvalues, eigenvectors = torch.linalg.eigh(work)
    floor = eigenvalues.max().clamp_min(1.0) * relative_floor
    inverse_values = eigenvalues.clamp_min(floor).rsqrt()
    result = (eigenvectors * inverse_values.unsqueeze(0)) @ eigenvectors.mT
    return result.to(matrix.dtype)


def lora_muon_factor_directions(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    moment_a: torch.Tensor,
    moment_b: torch.Tensor,
    *,
    exact: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-LR LoRA-Muon directions for ``W = factor_a @ factor_b``.

    ``factor_b`` is stored in ``(rank, in_features)`` orientation. The paper's
    second factor is therefore ``factor_b.T``.
    """
    if factor_a.ndim != 2 or factor_b.ndim != 2:
        raise ValueError("LoRA-Muon requires two matrix factors")
    if factor_a.shape[1] != factor_b.shape[0]:
        raise ValueError(
            f"incompatible factors {factor_a.shape} and {factor_b.shape}"
        )
    if moment_a.shape != factor_a.shape or moment_b.shape != factor_b.shape:
        raise ValueError("moment shapes must match their factors")

    inverse_root = (
        matrix_inverse_sqrt_eigh if exact else matrix_inverse_sqrt_newton_schulz
    )
    work_a = factor_a if exact else factor_a.float()
    work_b = factor_b if exact else factor_b.float()
    work_moment_a = moment_a if exact else moment_a.float()
    work_moment_b = moment_b if exact else moment_b.float()
    gram_a = work_a.mT @ work_a
    gram_b = work_b @ work_b.mT
    inverse_a = inverse_root(gram_a)
    inverse_b = inverse_root(gram_b)

    whitened_a = work_moment_a @ inverse_b
    whitened_b_transpose = work_moment_b.mT @ inverse_a
    if exact:
        sign_a = _exact_matrix_sign(whitened_a)
        sign_b_transpose = _exact_matrix_sign(whitened_b_transpose)
    else:
        sign_a = matrix_sign_newton_schulz(whitened_a)
        sign_b_transpose = matrix_sign_newton_schulz(whitened_b_transpose)

    direction_a = -0.5 * sign_a @ inverse_b
    direction_b = (-0.5 * sign_b_transpose @ inverse_a).mT
    return direction_a.to(factor_a.dtype), direction_b.to(factor_b.dtype)


def _exact_matrix_sign(matrix: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    return (u @ vh).to(matrix.dtype)


@torch.no_grad()
def apply_lora_muon_step(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
) -> None:
    """Apply Algorithm 1's split product decay and paired factor update."""
    decay = 1.0 - weight_decay * lr
    if decay <= 0.0:
        raise ValueError(
            f"LoRA-Muon split decay requires weight_decay * lr < 1, got "
            f"{weight_decay} * {lr}"
        )
    scale = math.sqrt(decay)
    inverse_scale = 1.0 / scale
    factor_a.mul_(scale).add_(direction_a, alpha=lr * inverse_scale)
    factor_b.mul_(scale).add_(direction_b, alpha=lr * inverse_scale)
