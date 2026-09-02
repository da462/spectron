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


def _lowrank_sum_matmul(
    terms: list[tuple[torch.Tensor, torch.Tensor]], vector: torch.Tensor
) -> torch.Tensor:
    result = terms[0][0] @ (terms[0][1] @ vector)
    for left, right in terms[1:]:
        result = result + left @ (right @ vector)
    return result


def _lowrank_sum_transpose_matmul(
    terms: list[tuple[torch.Tensor, torch.Tensor]], vector: torch.Tensor
) -> torch.Tensor:
    result = terms[0][1].mT @ (terms[0][0].mT @ vector)
    for left, right in terms[1:]:
        result = result + right.mT @ (left.mT @ vector)
    return result


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
    return approximate_lowrank_sum_top2(
        [
            (delta_a, factor_b),
            (factor_a, delta_b),
            (delta_a, delta_b),
        ],
        left_basis=left_basis,
        power_iterations=power_iterations,
        seed=seed,
    )


def approximate_lowrank_sum_top2(
    terms: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    left_basis: torch.Tensor | None = None,
    power_iterations: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate the top two singular triplets of an implicit low-rank sum."""
    float_terms = [(left.float(), right.float()) for left, right in terms]
    output_features = float_terms[0][0].shape[0]
    if left_basis is None or left_basis.shape != (output_features, 2):
        left = _deterministic_basis(
            output_features,
            2,
            device=float_terms[0][0].device,
            seed=seed,
        )
    else:
        left = left_basis.float()
    left = torch.linalg.qr(left, mode="reduced").Q

    right = None
    for _ in range(max(1, power_iterations)):
        right = _lowrank_sum_transpose_matmul(float_terms, left)
        right = torch.linalg.qr(right, mode="reduced").Q
        left = _lowrank_sum_matmul(float_terms, right)
        left = torch.linalg.qr(left, mode="reduced").Q
    assert right is not None
    core = left.mT @ _lowrank_sum_matmul(float_terms, right)
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


def top_singular_pin_directions(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    momentum_a: torch.Tensor,
    momentum_b: torch.Tensor,
    *,
    target_rms: float = 0.2,
    left_basis: torch.Tensor | None = None,
    power_iterations: int = 6,
    seed: int = 0,
    collect_spectral_metrics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Pin the original product-momentum head to one, then map it tangentially."""
    a = factor_a.float()
    b = factor_b.float()
    ma = momentum_a.float()
    mb = momentum_b.float()
    momentum_terms = [(ma, b), (a, mb)]
    singular, top_left, top_right, next_basis = approximate_lowrank_sum_top2(
        momentum_terms,
        left_basis=left_basis,
        power_iterations=power_iterations,
        seed=seed,
    )
    sigma1, sigma2 = singular[0], singular[1]
    head_shift = sigma1 - 1.0

    coefficient_a = _regularized_solve(
        a.mT @ a, a.mT @ top_left
    )
    coefficient_b = _regularized_solve(
        b @ b.mT, b @ top_right
    )
    residual_left = top_left - a @ coefficient_a
    residual_right = top_right - b.mT @ coefficient_b
    mapped_a = ma - head_shift * torch.outer(residual_left, coefficient_b)
    mapped_b = mb - head_shift * torch.outer(coefficient_a, top_right)

    direction_a, direction_b, metrics = product_adamrms_directions(
        factor_a,
        factor_b,
        mapped_a,
        mapped_b,
        target_rms=target_rms,
    )
    metrics.update(
        {
            "original_product_momentum_sigma1": sigma1,
            "original_product_momentum_sigma2": sigma2,
            "original_product_momentum_sigma1_to_sigma2": sigma1
            / sigma2.clamp_min(EPS),
            "intended_original_top_coefficient": torch.ones_like(sigma1),
            "intended_second_singular_value": sigma2,
            "top_component_shift": head_shift,
        }
    )
    if collect_spectral_metrics:
        mapped_terms = [(mapped_a, b), (a, mapped_b)]
        mapped_singular, _, _, _ = approximate_lowrank_sum_top2(
            mapped_terms,
            left_basis=next_basis,
            power_iterations=power_iterations,
            seed=seed,
        )
        mapped_top_coefficient = torch.dot(
            top_left,
            _lowrank_sum_matmul(mapped_terms, top_right),
        )
        desired_sumsq = lowrank_sum_frobenius_squared(
            momentum_terms
            + [(-head_shift * top_left[:, None], top_right[None, :])]
        )
        tangent_error = (
            head_shift.abs()
            * torch.linalg.vector_norm(residual_left)
            * torch.linalg.vector_norm(residual_right)
        )
        multiplier = metrics["product_adamrms_multiplier"]
        left_residual = torch.linalg.vector_norm(
            _lowrank_sum_matmul(momentum_terms, top_right)
            - sigma1 * top_left
        ) / sigma1.clamp_min(EPS)
        right_residual = torch.linalg.vector_norm(
            _lowrank_sum_transpose_matmul(momentum_terms, top_left)
            - sigma1 * top_right
        ) / sigma1.clamp_min(EPS)
        metrics.update(
            {
                "top_left_singular_residual_relative": left_residual,
                "top_right_singular_residual_relative": right_residual,
                "mapped_sigma1": mapped_singular[0],
                "mapped_sigma2": mapped_singular[1],
                "mapped_sigma1_to_sigma2": mapped_singular[0]
                / mapped_singular[1].clamp_min(EPS),
                "mapped_original_top_coefficient": mapped_top_coefficient,
                "mapped_original_top_coefficient_error": (
                    mapped_top_coefficient - 1.0
                ).abs(),
                "desired_to_mapped_relative_frobenius_error": tangent_error
                / torch.sqrt(desired_sumsq).clamp_min(EPS),
                "scaled_mapped_sigma1": mapped_singular[0] * multiplier,
                "scaled_mapped_sigma2": mapped_singular[1] * multiplier,
                "scaled_original_top_coefficient": mapped_top_coefficient
                * multiplier,
            }
        )
    return direction_a, direction_b, next_basis, metrics


def metrics_to_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in metrics.items()}
