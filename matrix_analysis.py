
import torch

def compute_stable_rank(weight_matrix):
    """Compute stable rank of a 2D weight matrix.

    Stable rank = (sum of squared singular values) / (max squared singular value)
    """
    if weight_matrix.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got {weight_matrix.dim()}D tensor")

    # Compute SVD
    with torch.no_grad():
        _, singular_values, _ = torch.linalg.svd(weight_matrix.float(), full_matrices=False)

        # Compute squared singular values
        sq_singular_values = singular_values ** 2

        # Stable rank = sum(s^2) / max(s^2)
        stable_rank = sq_singular_values.sum() / sq_singular_values[0]

    return stable_rank.item()


def compute_matrix_metrics(weight_matrix):
    """Compute stable rank, spectral norm, and condition number of a 2D weight matrix.

    Returns:
        dict: Dictionary containing 'stable_rank', 'spectral_norm', and 'condition_number'
    """
    if weight_matrix.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got {weight_matrix.dim()}D tensor")

    # Compute SVD
    with torch.no_grad():
        _, singular_values, _ = torch.linalg.svd(weight_matrix.float(), full_matrices=False)

        # Compute squared singular values
        sq_singular_values = singular_values ** 2

        # Stable rank = sum(s^2) / max(s^2)
        stable_rank = sq_singular_values.sum() / sq_singular_values[0]

        # Spectral norm = largest singular value
        spectral_norm = singular_values[0]

        # Condition number = largest singular value / smallest singular value
        # Avoid division by zero
        if singular_values[-1] > 1e-10:
            condition_number = singular_values[0] / singular_values[-1]
        else:
            condition_number = float('inf')

    return {
        'stable_rank': stable_rank.item(),
        'spectral_norm': spectral_norm.item(),
        'condition_number': condition_number.item() if condition_number != float('inf') else condition_number
    }