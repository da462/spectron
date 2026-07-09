"""Optimizer parameter routing helpers for low-rank Spectron/Muon runs."""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence


SPECTRAL_LR_TARGET_CHOICES = ("all", "attention", "ffn", "none")


def lowrank_module_type(param_name: str) -> str:
    """Classify a low-rank factor parameter by its owning transformer module."""
    if ".attention." in param_name:
        return "attention"
    if ".feed_forward." in param_name:
        return "ffn"
    return "other"


def spectral_lr_target_matches(param_name: str, target: str) -> bool:
    """Return whether a low-rank parameter should receive Spectron LR scaling."""
    if target not in SPECTRAL_LR_TARGET_CHOICES:
        raise ValueError(
            f"Unknown spectral LR target {target!r}; expected one of "
            f"{SPECTRAL_LR_TARGET_CHOICES}"
        )
    if target == "all":
        return True
    if target == "none":
        return False
    return lowrank_module_type(param_name) == target


def lowrank_lr_multiplier_for_param(param_name: str, ffn_multiplier: float) -> float:
    """Return the base LR multiplier for an individual low-rank factor."""
    if lowrank_module_type(param_name) == "ffn":
        return ffn_multiplier
    return 1.0


def apply_lowrank_lr_overrides(
    param_groups: MutableSequence[dict],
    *,
    base_lr: float,
    spectral_scaling: Mapping[str, float] | None,
    spectral_lr_target: str,
    lowrank_ffn_lr_multiplier: float,
    apply_spectral_lr_scaling: bool,
) -> int:
    """Apply per-factor LR routing to low-rank optimizer param groups.

    Spectron scaling is applied only to groups selected by ``spectral_lr_target``.
    Non-selected low-rank groups stay on plain Muon LR, with the optional FFN
    multiplier applied to FFN factors.
    """
    scaling = spectral_scaling or {}
    updated = 0
    for param_group in param_groups:
        if not param_group.get("is_lowrank", False) or "param_name" not in param_group:
            continue
        param_name = param_group["param_name"]
        lr_multiplier = param_group.get(
            "lr_multiplier",
            lowrank_lr_multiplier_for_param(param_name, lowrank_ffn_lr_multiplier),
        )
        lr = base_lr * lr_multiplier
        if apply_spectral_lr_scaling and spectral_lr_target_matches(
            param_name, spectral_lr_target
        ):
            if param_name not in scaling:
                raise KeyError(f"Missing spectral LR scale for low-rank param {param_name}")
            lr *= scaling[param_name]
        param_group["lr"] = lr
        updated += 1
    return updated


def lowrank_decoupled_weight_decay_lr(
    param_group: dict,
    *,
    base_lr: float,
    spectral_lr_target: str,
) -> float:
    """LR to use for manual decoupled WD on a low-rank factor group.

    Existing Spectron behavior decays spectrally-scaled low-rank factors with the
    unscaled base LR. For low-rank factors that are deliberately routed to plain
    Muon, use the group's plain LR multiplier.
    """
    param_name = param_group.get("param_name")
    if param_name is None:
        return base_lr
    if spectral_lr_target_matches(param_name, spectral_lr_target):
        return base_lr
    return base_lr * param_group.get("lr_multiplier", 1.0)
