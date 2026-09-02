"""Small, testable helpers for optimizer and learning-rate protocols."""

from __future__ import annotations

import math


def stable_linear_decay_factor(
    update_index: int,
    schedule_steps: int,
    decay_fraction: float,
) -> float:
    """Return a flat-then-linear LR multiplier for a zero-based update index.

    The first ``schedule_steps - ceil(schedule_steps * decay_fraction)``
    optimizer updates use the peak LR. The remaining updates decay linearly,
    with the final scheduled update using zero LR.
    """
    if update_index < 0:
        raise ValueError("update_index must be non-negative")
    if schedule_steps <= 0:
        raise ValueError("schedule_steps must be positive")
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError("decay_fraction must be in (0, 1]")

    if schedule_steps == 1:
        return 1.0 if update_index == 0 else 0.0

    decay_steps = max(1, math.ceil(schedule_steps * decay_fraction))
    decay_start = schedule_steps - decay_steps
    if update_index < decay_start:
        return 1.0
    if update_index >= schedule_steps:
        return 0.0

    return (schedule_steps - 1 - update_index) / decay_steps


def auxiliary_adamw_lr(muon_lr: float, multiplier: float) -> float:
    """Return the auxiliary AdamW LR paired with a Muon base LR."""
    if muon_lr < 0.0:
        raise ValueError("muon_lr must be non-negative")
    if multiplier <= 0.0:
        raise ValueError("multiplier must be positive")
    return muon_lr * multiplier
