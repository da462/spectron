"""Product-space calibrations for paired low-rank factor-Muon updates."""

from __future__ import annotations

import torch


EPS = 1e-12


def _frob_inner(
    left_a: torch.Tensor,
    right_a: torch.Tensor,
    left_b: torch.Tensor,
    right_b: torch.Tensor,
) -> torch.Tensor:
    """Return <left_a @ right_a, left_b @ right_b> without dense products."""
    left_gram = left_a.mT @ left_b
    right_gram = right_b @ right_a.mT
    return torch.sum(left_gram * right_gram.mT)


def lowrank_sum_frobenius_squared(
    terms: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Return the squared Frobenius norm of a sum of low-rank products."""
    total = torch.zeros((), device=terms[0][0].device, dtype=torch.float32)
    float_terms = [(left.float(), right.float()) for left, right in terms]
    for index, (left_a, right_a) in enumerate(float_terms):
        total = total + _frob_inner(left_a, right_a, left_a, right_a)
        for left_b, right_b in float_terms[index + 1 :]:
            total = total + 2.0 * _frob_inner(
                left_a, right_a, left_b, right_b
            )
    return total.clamp_min(0.0)


def product_adamrms_directions(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    raw_direction_a: torch.Tensor,
    raw_direction_b: torch.Tensor,
    *,
    target_rms: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Calibrate one shared factor scale to a product first-order RMS target."""
    output_features = factor_a.shape[0]
    input_features = factor_b.shape[1]
    first_order_terms = [
        (raw_direction_a, factor_b),
        (factor_a, raw_direction_b),
    ]
    pre_sumsq = lowrank_sum_frobenius_squared(first_order_terms)
    pre_rms = torch.sqrt(
        pre_sumsq / float(output_features * input_features)
    ).clamp_min(EPS)
    multiplier = torch.as_tensor(
        target_rms, device=pre_rms.device, dtype=pre_rms.dtype
    ) / pre_rms
    direction_a = (raw_direction_a.float() * multiplier).to(factor_a.dtype)
    direction_b = (raw_direction_b.float() * multiplier).to(factor_b.dtype)
    applied_sumsq = lowrank_sum_frobenius_squared(
        [(direction_a, factor_b), (factor_a, direction_b)]
    )
    applied_rms = torch.sqrt(
        applied_sumsq / float(output_features * input_features)
    )
    return direction_a, direction_b, {
        "pre_first_order_direction_rms": pre_rms,
        "product_adamrms_multiplier": multiplier,
        "target_first_order_direction_rms": torch.as_tensor(
            target_rms, device=pre_rms.device, dtype=pre_rms.dtype
        ),
        "first_order_direction_rms": applied_rms,
    }


def rankaware_product_adamrms_directions(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    raw_direction_a: torch.Tensor,
    raw_direction_b: torch.Tensor,
    *,
    dense_target_rms: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Calibrate product RMS per singular direction available at rank ``r``."""
    output_features, rank = factor_a.shape
    factor_rank, input_features = factor_b.shape
    if factor_rank != rank:
        raise ValueError(
            f"Incompatible factor ranks: A is {tuple(factor_a.shape)}, "
            f"B is {tuple(factor_b.shape)}"
        )
    q = min(output_features, input_features)
    k = min(2 * rank, q)
    target_rms = dense_target_rms * (k / q) ** 0.5
    direction_a, direction_b, metrics = product_adamrms_directions(
        factor_a,
        factor_b,
        raw_direction_a,
        raw_direction_b,
        target_rms=target_rms,
    )
    metrics.update(
        {
            "dense_product_target_rms": torch.as_tensor(
                dense_target_rms,
                device=factor_a.device,
                dtype=torch.float32,
            ),
            "rankaware_product_target_rms": torch.as_tensor(
                target_rms,
                device=factor_a.device,
                dtype=torch.float32,
            ),
            "rankaware_dense_min_dimension": torch.as_tensor(
                q, device=factor_a.device, dtype=torch.float32
            ),
            "rankaware_effective_rank_cap": torch.as_tensor(
                k, device=factor_a.device, dtype=torch.float32
            ),
            "rankaware_target_scale": torch.as_tensor(
                (k / q) ** 0.5,
                device=factor_a.device,
                dtype=torch.float32,
            ),
        }
    )
    return direction_a, direction_b, metrics


def _update_matmul(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    return delta_a @ ((factor_b + delta_b) @ vector) + factor_a @ (
        delta_b @ vector
    )


def _update_transpose_matmul(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    return (factor_b + delta_b).mT @ (delta_a.mT @ vector) + delta_b.mT @ (
        factor_a.mT @ vector
    )


def _deterministic_basis(
    rows: int,
    columns: int,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        rows, columns, device=device, dtype=torch.float32, generator=generator
    )


def approximate_update_top2(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    *,
    left_basis: torch.Tensor | None = None,
    power_iterations: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate the top two singular triplets of the implicit product update."""
    a = factor_a.float()
    b = factor_b.float()
    da = delta_a.float()
    db = delta_b.float()
    if left_basis is None or left_basis.shape != (a.shape[0], 2):
        left = _deterministic_basis(
            a.shape[0], 2, device=a.device, seed=seed
        )
    else:
        left = left_basis.float()
    left = torch.linalg.qr(left, mode="reduced").Q

    right = None
    for _ in range(max(1, power_iterations)):
        right = _update_transpose_matmul(a, b, da, db, left)
        right = torch.linalg.qr(right, mode="reduced").Q
        left = _update_matmul(a, b, da, db, right)
        left = torch.linalg.qr(left, mode="reduced").Q
    assert right is not None
    core = left.mT @ _update_matmul(a, b, da, db, right)
    core_u, singular_values, core_vh = torch.linalg.svd(
        core, full_matrices=False
    )
    left = left @ core_u
    right = right @ core_vh.mT
    return singular_values[:2], left[:, 0], right[:, 0], left[:, :2]


def product_update_metrics(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return exact Gram-based first, quadratic, and total update metrics."""
    first_sumsq = lowrank_sum_frobenius_squared(
        [(delta_a, factor_b), (factor_a, delta_b)]
    )
    quadratic_sumsq = lowrank_sum_frobenius_squared([(delta_a, delta_b)])
    total_sumsq = lowrank_sum_frobenius_squared(
        [(delta_a, factor_b), (factor_a, delta_b), (delta_a, delta_b)]
    )
    numel = float(factor_a.shape[0] * factor_b.shape[1])
    first_norm = torch.sqrt(first_sumsq)
    total_norm = torch.sqrt(total_sumsq)
    return {
        "first_order_update_rms": first_norm / numel**0.5,
        "actual_update_rms": total_norm / numel**0.5,
        "quadratic_to_first_frobenius": torch.sqrt(quadratic_sumsq)
        / first_norm.clamp_min(EPS),
        "actual_update_frobenius": total_norm,
    }


def _regularized_solve(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    scale = torch.diagonal(gram).mean().clamp_min(EPS)
    ridge = torch.finfo(gram.dtype).eps * scale * gram.shape[0]
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + ridge * identity, rhs)


def headclip_directions(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    baseline_direction_a: torch.Tensor,
    baseline_direction_b: torch.Tensor,
    *,
    learning_rate: float,
    left_basis: torch.Tensor | None = None,
    power_iterations: int = 4,
    seed: int = 0,
    collect_post_metrics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply one tangent-space correction that clips sigma1(D) to sigma2(D)."""
    if learning_rate < 0:
        raise ValueError("HeadClip requires a non-negative learning rate")
    if learning_rate == 0:
        if left_basis is None or left_basis.shape != (factor_a.shape[0], 2):
            left_basis = _deterministic_basis(
                factor_a.shape[0],
                2,
                device=factor_a.device,
                seed=seed,
            )
        left_basis = torch.linalg.qr(left_basis.float(), mode="reduced").Q
        zero = torch.zeros((), device=factor_a.device, dtype=torch.float32)
        metrics = {
            "pre_sigma1": zero,
            "pre_sigma2": zero,
            "pre_sigma1_to_sigma2": zero,
            "target_tau": zero,
            "head_beta": zero,
            "head_fraction_removed": zero,
            "pre_update_frobenius": zero,
            "post_update_frobenius": zero,
            "relative_frobenius_change": zero,
        }
        if collect_post_metrics:
            metrics.update(
                {
                    "post_sigma1": zero,
                    "post_sigma2": zero,
                    "post_sigma1_to_sigma2": zero,
                }
            )
        return (
            baseline_direction_a,
            baseline_direction_b,
            left_basis,
            metrics,
        )
    a = factor_a.float()
    b = factor_b.float()
    delta_a = -learning_rate * baseline_direction_a.float()
    delta_b = -learning_rate * baseline_direction_b.float()
    singular, top_left, top_right, next_basis = approximate_update_top2(
        a,
        b,
        delta_a,
        delta_b,
        left_basis=left_basis,
        power_iterations=power_iterations,
        seed=seed,
    )
    sigma1, sigma2 = singular[0], singular[1]
    beta = (sigma1 - sigma2).clamp_min(0.0)

    endpoint_a = a + delta_a
    endpoint_b = b + delta_b
    coefficient_a = _regularized_solve(
        endpoint_a.mT @ endpoint_a, endpoint_a.mT @ top_left
    )
    coefficient_b = _regularized_solve(
        endpoint_b @ endpoint_b.mT, endpoint_b @ top_right
    )
    correction_b = -beta * torch.outer(coefficient_a, top_right)
    residual_left = top_left - endpoint_a @ coefficient_a
    correction_a = -beta * torch.outer(residual_left, coefficient_b)
    corrected_delta_a = (delta_a + correction_a).to(factor_a.dtype)
    corrected_delta_b = (delta_b + correction_b).to(factor_b.dtype)
    direction_a = (-corrected_delta_a.float() / learning_rate).to(factor_a.dtype)
    direction_b = (-corrected_delta_b.float() / learning_rate).to(factor_b.dtype)

    pre_sumsq = lowrank_sum_frobenius_squared(
        [(delta_a, b), (a, delta_b), (delta_a, delta_b)]
    )
    post_sumsq = lowrank_sum_frobenius_squared(
        [
            (corrected_delta_a, b),
            (a, corrected_delta_b),
            (corrected_delta_a, corrected_delta_b),
        ]
    )
    metrics: dict[str, torch.Tensor] = {
        "pre_sigma1": sigma1,
        "pre_sigma2": sigma2,
        "pre_sigma1_to_sigma2": sigma1 / sigma2.clamp_min(EPS),
        "target_tau": sigma2,
        "head_beta": beta,
        "head_fraction_removed": beta / sigma1.clamp_min(EPS),
        "pre_update_frobenius": torch.sqrt(pre_sumsq),
        "post_update_frobenius": torch.sqrt(post_sumsq),
        "relative_frobenius_change": (
            torch.sqrt(post_sumsq) - torch.sqrt(pre_sumsq)
        ).abs()
        / torch.sqrt(pre_sumsq).clamp_min(EPS),
    }
    if collect_post_metrics:
        post_singular, _, _, _ = approximate_update_top2(
            a,
            b,
            corrected_delta_a,
            corrected_delta_b,
            left_basis=next_basis,
            power_iterations=power_iterations,
            seed=seed,
        )
        metrics.update(
            {
                "post_sigma1": post_singular[0],
                "post_sigma2": post_singular[1],
                "post_sigma1_to_sigma2": post_singular[0]
                / post_singular[1].clamp_min(EPS),
            }
        )
    return direction_a, direction_b, next_basis, metrics


def metrics_to_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in metrics.items()}
