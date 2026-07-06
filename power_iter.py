import torch

def _l2_normalize(v, eps=1e-12):
    """L2 normalize a tensor."""
    return v / (v.norm() + eps)


def _max_singular_value_power_iter_old(W, u, Ip=1):
    """
    Apply power iteration to estimate the maximum singular value of W.

    Args:
        W: Weight matrix of shape (out_features, in_features)
        u: Left singular vector estimate of shape (1, out_features)
        Ip: Number of power iterations

    Returns:
        sigma: Maximum singular value
        u: Updated left singular vector
    """
    _u = u
    for _ in range(Ip):
        # v = W^T u / ||W^T u||
        _v = _l2_normalize(torch.mm(_u, W))
        # u = W v / ||W v||
        _u = _l2_normalize(torch.mm(_v, W.t()))

    # Compute sigma = u^T W v
    sigma = torch.sum(_u @ W * _v)
    return sigma, _u


def _max_singular_value_power_iter(W, u, Ip=1):
    _u = u
    for _ in range(Ip):
        _v = _l2_normalize(_u @ W)
        _u = _l2_normalize(_v @ W.t())

    sigma = (_u @ W @ _v.t()).squeeze()
    return sigma, _u, _v


def get_lowrank_spectral_norm_scaling(
    model,
    power_iter_steps=1,
    spectral_weight_decay=0.0,
    swd_type='standard',
    spectral_lr_scaling_offset=1.0,
):
    """
    Compute spectral norm scaling factors for A and B parameters in LowRankLinear modules.

    Uses power iteration for efficient spectral norm estimation, updating the stored
    u vectors in each LowRankLinear module for subsequent calls.

    Returns a dict mapping parameter name to the scaling factor 1/(spectral_norm(A) + spectral_norm(B))
    where A and B come from the same LowRankLinear module.

    Args:
        model: The model containing LowRankLinear modules
        power_iter_steps: Number of power iteration steps (default 1)
        spectral_weight_decay: Spectral weight decay coefficient (default 0.0, disabled)
        swd_type: Type of spectral weight decay - 'standard' or 'product' (default: 'standard')
        spectral_lr_scaling_offset: Additive offset in
            1 / (spectral_norm(A) + spectral_norm(B) + offset). Current default
            preserves existing behavior with offset=1.0.

    Returns:
        tuple: (scaling_factors, regularization_grads)
            - scaling_factors: dict mapping parameter name to scaling factor
            - regularization_grads: dict mapping parameter name to regularization gradient
    """
    scaling_factors = {}
    regularization_grads = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            # Check if this is a LowRankLinear module with u vectors
            if (
                hasattr(module, "A")
                and hasattr(module, "B")
                and isinstance(module.A, torch.nn.Parameter)
                and hasattr(module, "u_A")
                and hasattr(module, "u_B")
            ):
                # Validate and fix u_A shape if needed (handles checkpoint mismatches)
                # A has shape (out_features, rank), so u_A should be (1, out_features)
                if False:
                    expected_u_A_shape = (1, module.A.shape[0])
                    if module.u_A.shape != expected_u_A_shape:
                        from low_rank_linear import l2_normalize
                        module.u_A = l2_normalize(
                            torch.randn(expected_u_A_shape, device=module.A.device, dtype=module.u_A.dtype)
                        )

                    # Validate and fix u_B shape if needed (handles checkpoint mismatches)
                    # B has shape (rank, in_features), so u_B should be (1, rank)
                    expected_u_B_shape = (1, module.B.shape[0])
                    if module.u_B.shape != expected_u_B_shape:
                        from low_rank_linear import l2_normalize
                        module.u_B = l2_normalize(
                            torch.randn(expected_u_B_shape, device=module.B.device, dtype=module.u_B.dtype)
                        )

                # Compute spectral norm of A using power iteration
                A_float = module.A.float()
                sigma_A, new_u_A, v_A = _max_singular_value_power_iter(
                    A_float, module.u_A.float(), Ip=power_iter_steps
                )
                # Update u_A for next iteration
                module.u_A.copy_(new_u_A)
                A_spectral_norm = sigma_A.item()

                # Compute spectral norm of B using power iteration
                B_float = module.B.float()
                sigma_B, new_u_B, v_B = _max_singular_value_power_iter(
                    B_float, module.u_B.float(), Ip=power_iter_steps
                )
                # Update u_B for next iteration
                module.u_B.copy_(new_u_B)
                B_spectral_norm = sigma_B.item()

                # Compute regularization gradients if spectral weight decay is enabled
                # Store them separately for decoupled weight decay
                if spectral_weight_decay > 0:
                    # Map both A and B parameter names for regularization
                    a_param_name = f"{name}.A" if name else "A"
                    b_param_name = f"{name}.B" if name else "B"

                    if swd_type == 'standard':
                        # Standard type: gradient scaled by own spectral norm
                        # Regularization: λ * (σ_A² / 2 + σ_B² / 2)
                        # Gradient for A: λ * σ_A * (u_A^T @ v_A)
                        # Gradient for B: λ * σ_B * (u_B^T @ v_B)

                        grad_A = spectral_weight_decay * A_spectral_norm * new_u_A.t() @ v_A
                        grad_A = grad_A.to(module.A.dtype)
                        regularization_grads[a_param_name] = grad_A

                        grad_B = spectral_weight_decay * B_spectral_norm * new_u_B.t() @ v_B
                        grad_B = grad_B.to(module.B.dtype)
                        regularization_grads[b_param_name] = grad_B

                    elif swd_type == 'product':
                        # Product type: gradient scaled by paired matrix's spectral norm
                        # Regularization: λ * (σ_A^2 * σ_B^2)
                        # Gradient for A: λ * σ_B^2 * (∂σ_A/∂A)
                        # Gradient for B: λ * σ_A^2 * (∂σ_B/∂B)

                        # Use the same power iteration results, but swap the scaling
                        # grad_A scaled by σ_B (instead of σ_A)
                        # grad_B scaled by σ_A (instead of σ_B)

                        grad_A = spectral_weight_decay * B_spectral_norm**2 * new_u_A.t() @ v_A  # Scaled by σ_B^2
                        grad_A = grad_A.to(module.A.dtype)
                        regularization_grads[a_param_name] = grad_A

                        grad_B = spectral_weight_decay * A_spectral_norm**2 * new_u_B.t() @ v_B  # Scaled by σ_A^2
                        grad_B = grad_B.to(module.B.dtype)
                        regularization_grads[b_param_name] = grad_B

                # Compute scaling factor:
                # 1 / (spectral_norm(A) + spectral_norm(B) + offset)
                sum_spectral_norms = (
                    A_spectral_norm + B_spectral_norm + spectral_lr_scaling_offset
                )

                # Avoid division by zero
                if sum_spectral_norms > 1e-8:
                    scaling_factor = 1.0 / sum_spectral_norms
                else:
                    scaling_factor = 1.0

                # Map both A and B parameter names to this scaling factor
                a_param_name = f"{name}.A" if name else "A"
                b_param_name = f"{name}.B" if name else "B"

                scaling_factors[a_param_name] = scaling_factor
                scaling_factors[b_param_name] = scaling_factor

    return scaling_factors, regularization_grads
