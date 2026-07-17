"""Optimizer parameter routing helpers for low-rank Spectron/Muon runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSequence
import re


SPECTRAL_LR_TARGET_CHOICES = ("all", "attention", "ffn", "none")
LOWRANK_LR_TRACK_MODULE_CHOICES = ("all", "attention", "ffn")
LOWRANK_FACTOR_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\."
    r"(?P<owner>attention|feed_forward)\."
    r"(?P<matrix>[A-Za-z0-9_]+)\."
    r"(?P<factor>A|B)$"
)


def lowrank_module_type(param_name: str) -> str:
    """Classify a low-rank factor parameter by its owning transformer module."""
    if ".attention." in param_name:
        return "attention"
    if ".feed_forward." in param_name:
        return "ffn"
    return "other"


def lowrank_factor_metadata(param_name: str) -> dict | None:
    """Parse a Transformer low-rank factor parameter name into stable fields."""
    match = LOWRANK_FACTOR_RE.match(param_name)
    if match is None:
        return None
    owner = match.group("owner")
    return {
        "layer": int(match.group("layer")),
        "module_type": "ffn" if owner == "feed_forward" else "attention",
        "matrix": match.group("matrix"),
        "factor": match.group("factor"),
    }


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


def lowrank_weight_decay_for_param(
    param_name: str,
    *,
    default_weight_decay: float,
    ffn_weight_decay: float | None = None,
    attention_weight_decay: float | None = None,
) -> float:
    """Return decoupled WD for a low-rank factor, with optional owner overrides."""
    module_type = lowrank_module_type(param_name)
    if module_type == "ffn" and ffn_weight_decay is not None:
        return ffn_weight_decay
    if module_type == "attention" and attention_weight_decay is not None:
        return attention_weight_decay
    return default_weight_decay


def _to_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def collect_lowrank_lr_records(
    named_parameters: Iterable[tuple[str, object]],
    param_groups: Iterable[dict],
    *,
    base_lr: float,
    lowrank_ffn_lr_multiplier: float,
    spectral_scaling: Mapping[str, float] | None,
    spectral_lr_target: str,
    apply_spectral_lr_scaling: bool,
    module_type_filter: str = "ffn",
) -> list[dict]:
    """Collect actual optimizer LRs for low-rank factor parameters.

    This inspects the optimizer group containing each model parameter, so it
    works both when factors have individual Spectron/split-LR groups and when
    plain Muon keeps all hidden 2D tensors in one shared group.
    """
    if module_type_filter not in LOWRANK_LR_TRACK_MODULE_CHOICES:
        raise ValueError(
            f"Unknown module_type_filter {module_type_filter!r}; expected one of "
            f"{LOWRANK_LR_TRACK_MODULE_CHOICES}"
        )

    param_to_group: dict[int, tuple[int, dict]] = {}
    for group_index, param_group in enumerate(param_groups):
        for param in param_group.get("params", ()):
            param_to_group[id(param)] = (group_index, param_group)

    scaling = spectral_scaling or {}
    records: list[dict] = []
    for param_name, param in named_parameters:
        metadata = lowrank_factor_metadata(param_name)
        if metadata is None:
            continue
        if (
            module_type_filter != "all"
            and metadata["module_type"] != module_type_filter
        ):
            continue

        group_entry = param_to_group.get(id(param))
        if group_entry is None:
            continue
        group_index, param_group = group_entry

        lr_multiplier = _to_float(
            param_group.get(
                "lr_multiplier",
                lowrank_lr_multiplier_for_param(
                    param_name, lowrank_ffn_lr_multiplier
                ),
            )
        )
        nominal_lr = _to_float(base_lr) * lr_multiplier
        spectral_scale = (
            _to_float(scaling[param_name]) if param_name in scaling else None
        )
        spectral_targeted = (
            apply_spectral_lr_scaling
            and spectral_lr_target_matches(param_name, spectral_lr_target)
        )
        expected_lr = nominal_lr
        if spectral_targeted:
            expected_lr = None if spectral_scale is None else nominal_lr * spectral_scale

        weight_decay = param_group.get(
            "lowrank_weight_decay", param_group.get("weight_decay")
        )
        shape = list(param.shape) if hasattr(param, "shape") else None
        numel = int(param.numel()) if hasattr(param, "numel") else None

        records.append(
            {
                "param_name": param_name,
                "layer": metadata["layer"],
                "module_type": metadata["module_type"],
                "matrix": metadata["matrix"],
                "factor": metadata["factor"],
                "shape": shape,
                "numel": numel,
                "group_index": group_index,
                "actual_lr": _to_float(param_group.get("lr", float("nan"))),
                "base_lr": _to_float(base_lr),
                "lr_multiplier": lr_multiplier,
                "nominal_lr": nominal_lr,
                "spectral_targeted": spectral_targeted,
                "spectral_scale": spectral_scale,
                "expected_lr": expected_lr,
                "weight_decay": None if weight_decay is None else _to_float(weight_decay),
                "use_muon": bool(param_group.get("use_muon", False)),
                "is_individual_lowrank_group": bool(
                    param_group.get("is_lowrank", False)
                    and param_group.get("param_name") == param_name
                ),
            }
        )

    return sorted(
        records,
        key=lambda row: (
            row["module_type"],
            row["layer"],
            row["matrix"],
            row["factor"],
            row["param_name"],
        ),
    )


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
