"""Low-overhead diagnostics collected from the real training step.

This module deliberately keeps the high-frequency path separate from
``mechanistic_diagnostics``. Forward hooks reduce activations immediately to
scalar sufficient statistics. Optimizer hooks inspect the update tensor while
it already exists and never retain parameters, gradients, updates, or training
activations.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist

from optimizer_routing import lowrank_factor_metadata


_UPDATE_FIELDS = (
    "present",
    "learning_rate",
    "weight_decay",
    "weight_decay_lr",
    "external_decay_multiplier",
    "decay_multiplier",
    "shape_adjustment_multiplier",
    "numel",
    "old_parameter_sumsq",
    "adjusted_direction_sumsq",
    "applied_direction_sumsq",
    "total_update_sumsq",
)

_PAIR_METRIC_FIELDS = (
    "present",
    "pre_first_order_direction_rms",
    "product_adamrms_multiplier",
    "target_first_order_direction_rms",
    "first_order_direction_rms",
    "first_order_update_rms",
    "actual_update_rms",
    "quadratic_to_first_frobenius",
    "actual_update_frobenius",
    "dense_product_target_rms",
    "rankaware_product_target_rms",
    "rankaware_dense_min_dimension",
    "rankaware_effective_rank_cap",
    "rankaware_target_scale",
    "pre_sigma1",
    "pre_sigma2",
    "pre_sigma1_to_sigma2",
    "target_tau",
    "head_beta",
    "head_fraction_removed",
    "post_sigma1",
    "post_sigma2",
    "post_sigma1_to_sigma2",
    "pre_update_frobenius",
    "post_update_frobenius",
    "relative_frobenius_change",
)

_PAIR_FIELDS_BY_VARIANT = {
    "product_adamrms": (
        "pre_first_order_direction_rms",
        "product_adamrms_multiplier",
        "target_first_order_direction_rms",
        "first_order_direction_rms",
        "first_order_update_rms",
        "actual_update_rms",
        "quadratic_to_first_frobenius",
        "actual_update_frobenius",
        "post_sigma1",
        "post_sigma2",
        "post_sigma1_to_sigma2",
    ),
    "rankaware_product_adamrms": (
        "pre_first_order_direction_rms",
        "product_adamrms_multiplier",
        "target_first_order_direction_rms",
        "first_order_direction_rms",
        "first_order_update_rms",
        "actual_update_rms",
        "quadratic_to_first_frobenius",
        "actual_update_frobenius",
        "dense_product_target_rms",
        "rankaware_product_target_rms",
        "rankaware_dense_min_dimension",
        "rankaware_effective_rank_cap",
        "rankaware_target_scale",
        "post_sigma1",
        "post_sigma2",
        "post_sigma1_to_sigma2",
    ),
    "headclip": (
        "pre_sigma1",
        "pre_sigma2",
        "pre_sigma1_to_sigma2",
        "target_tau",
        "head_beta",
        "head_fraction_removed",
        "post_sigma1",
        "post_sigma2",
        "post_sigma1_to_sigma2",
        "pre_update_frobenius",
        "post_update_frobenius",
        "relative_frobenius_change",
    ),
}


def _shape_adjustment_multiplier(mode: str, shape: torch.Size) -> float:
    if mode == "none":
        return 1.0
    rows, columns = shape[:2]
    if mode == "original":
        return math.sqrt(max(1.0, rows / columns))
    if mode == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, columns))
    raise ValueError(f"Unknown Muon LR adjustment mode: {mode}")


def _safe_sqrt(value: float) -> float:
    return math.sqrt(max(0.0, value))


def _summary(values: list[float]) -> dict[str, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(finite),
        "median": statistics.median(finite),
        "max": max(finite),
    }


def _trace_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return tr(left @ right) without an r x r matrix multiplication."""
    return torch.sum(left * right.mT)


def factor_gram_terms(
    current_gram: torch.Tensor,
    current_direction_cross: torch.Tensor,
    direction_gram: torch.Tensor,
    *,
    state_scale: float,
    internal_decay_multiplier: float,
    learning_rate: float,
    optimizer_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return old Gram, old/delta cross Gram, and delta Gram.

    ``current`` is the parameter value observed inside the optimizer. It may
    already include an externally applied Spectron decay with multiplier
    ``state_scale``. The optimizer subsequently applies
    ``internal_decay_multiplier`` and ``-learning_rate * direction``.
    """
    if state_scale == 0:
        raise ValueError("state_scale must be nonzero")
    old_gram = current_gram / (state_scale * state_scale)
    if optimizer_only:
        alpha = 0.0
    else:
        alpha = internal_decay_multiplier - 1.0 / state_scale
    cross = (
        (alpha / state_scale) * current_gram
        - (learning_rate / state_scale) * current_direction_cross
    )
    delta_gram = (
        alpha * alpha * current_gram
        - alpha
        * learning_rate
        * (current_direction_cross + current_direction_cross.mT)
        + learning_rate * learning_rate * direction_gram
    )
    return old_gram, cross, delta_gram


def product_metrics_from_gram_terms(
    a_terms: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    b_terms: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    output_features: int,
    input_features: int,
) -> dict[str, float]:
    """Compute exact ||AB update||_F and ||AB||_F from rank-sized Grams.

    The repository stores ``A`` as ``[out, rank]`` and ``B`` as
    ``[rank, in]``, so the represented matrix is ``A @ B``. The B-side terms
    are row Grams (``B @ B.T``), while A-side terms are column Grams.
    """
    old_a, cross_a, delta_a = a_terms
    old_b, cross_b, delta_b = b_terms

    new_a = old_a + cross_a + cross_a.mT + delta_a
    new_b = old_b + cross_b + cross_b.mT + delta_b
    new_a_old_a = old_a + cross_a.mT
    old_a_new_a = old_a + cross_a
    new_b_old_b = old_b + cross_b.mT
    old_b_new_b = old_b + cross_b

    delta_sumsq = (
        _trace_product(new_a, new_b)
        - _trace_product(new_a_old_a, old_b_new_b)
        - _trace_product(old_a_new_a, new_b_old_b)
        + _trace_product(old_a, old_b)
    )
    old_sumsq = _trace_product(old_a, old_b)
    delta_sumsq_value = max(0.0, float(delta_sumsq))
    old_sumsq_value = max(0.0, float(old_sumsq))
    return {
        "update_frobenius_norm": _safe_sqrt(delta_sumsq_value),
        "update_rms": _safe_sqrt(
            delta_sumsq_value / (output_features * input_features)
        ),
        "weight_frobenius_norm": _safe_sqrt(old_sumsq_value),
        "relative_update": _safe_sqrt(
            delta_sumsq_value / max(old_sumsq_value, 1e-30)
        ),
    }


class ActivationScalarCollector:
    """Forward hooks that retain only scalar reductions from real microbatches."""

    def __init__(self, model: torch.nn.Module, *, enabled: bool) -> None:
        self.enabled = enabled
        self.active = False
        self._stats: dict[str, dict[str, Any]] = {}
        self._handles: list[Any] = []
        if enabled:
            self._register(model)

    def _register(self, model: torch.nn.Module) -> None:
        self._handles.append(
            model.tok_embeddings.register_forward_hook(self._output_hook("embedding"))
        )
        for layer_name, layer in model.layers.items():
            prefix = f"layer_{layer_name}"
            self._handles.append(
                layer.attention_norm.register_forward_pre_hook(
                    self._input_hook(f"{prefix}.pre_attention")
                )
            )
            self._handles.append(
                layer.attention_norm.register_forward_hook(
                    self._output_hook(f"{prefix}.post_attention_norm")
                )
            )
            self._handles.append(
                layer.attention.register_forward_hook(
                    self._output_hook(f"{prefix}.attention_branch")
                )
            )
            self._handles.append(
                layer.ffn_norm.register_forward_pre_hook(
                    self._input_hook(f"{prefix}.post_attention")
                )
            )
            self._handles.append(
                layer.ffn_norm.register_forward_hook(
                    self._output_hook(f"{prefix}.post_ffn_norm")
                )
            )
            self._handles.append(
                layer.feed_forward.w1.register_forward_hook(
                    self._output_hook(f"{prefix}.ffn_u1")
                )
            )
            self._handles.append(
                layer.feed_forward.w3.register_forward_hook(
                    self._output_hook(f"{prefix}.ffn_u3")
                )
            )
            self._handles.append(
                layer.feed_forward.w2.register_forward_pre_hook(
                    self._input_hook(f"{prefix}.ffn_g")
                )
            )
            self._handles.append(
                layer.feed_forward.w2.register_forward_hook(
                    self._output_hook(f"{prefix}.ffn_y")
                )
            )
            self._handles.append(
                layer.register_forward_hook(
                    self._output_hook(f"{prefix}.post_ffn")
                )
            )
        self._handles.append(
            model.norm.register_forward_pre_hook(self._input_hook("final_residual"))
        )
        self._handles.append(
            model.norm.register_forward_hook(self._output_hook("final_normalized"))
        )
        self._handles.append(
            model.output.register_forward_hook(
                self._output_hook(
                    "logits", include_sum=True, deactivate_after=True
                )
            )
        )

    def _input_hook(self, key: str):
        def hook(module, inputs):
            if self.active:
                self._accumulate(key, inputs[0])

        return hook

    def _output_hook(
        self,
        key: str,
        *,
        include_sum: bool = False,
        deactivate_after: bool = False,
    ):
        def hook(module, inputs, output):
            if self.active:
                self._accumulate(key, output, include_sum=include_sum)
                if deactivate_after:
                    self.active = False

        return hook

    @torch.no_grad()
    def _accumulate(
        self, key: str, tensor: torch.Tensor, *, include_sum: bool = False
    ) -> None:
        value = tensor.detach()
        sumsq = torch.linalg.vector_norm(value, dtype=torch.float32).square()
        minimum, maximum = torch.aminmax(value)
        max_abs = torch.maximum(maximum.float(), -minimum.float())
        entry = self._stats.get(key)
        if entry is None:
            entry = {
                "sumsq": torch.zeros((), device=value.device, dtype=torch.float32),
                "max_abs": torch.zeros((), device=value.device, dtype=torch.float32),
                "sum": torch.zeros((), device=value.device, dtype=torch.float32),
                "count": 0,
                "include_sum": include_sum,
            }
            self._stats[key] = entry
        entry["sumsq"].add_(sumsq)
        entry["max_abs"].copy_(torch.maximum(entry["max_abs"], max_abs))
        if include_sum:
            entry["sum"].add_(torch.sum(value, dtype=torch.float32))
        entry["count"] += value.numel()

    def begin_step(self) -> None:
        self._stats.clear()
        self.active = self.enabled

    def end_forward(self) -> None:
        self.active = False

    def snapshot(self) -> dict[str, dict[str, float]]:
        if not self._stats:
            return {}
        keys = sorted(self._stats)
        packed = torch.stack(
            [
                scalar
                for key in keys
                for scalar in (
                    self._stats[key]["sumsq"],
                    self._stats[key]["max_abs"],
                    self._stats[key]["sum"],
                )
            ]
        ).cpu()
        result = {}
        for index, key in enumerate(keys):
            sumsq, max_abs, value_sum = packed[3 * index : 3 * index + 3].tolist()
            count = self._stats[key]["count"]
            row = {
                "rms": _safe_sqrt(sumsq / max(count, 1)),
                "max_abs": max_abs,
                "count": count,
            }
            if self._stats[key]["include_sum"]:
                mean = value_sum / max(count, 1)
                row["mean"] = mean
                row["std"] = _safe_sqrt(sumsq / max(count, 1) - mean * mean)
            result[key] = row
        self._stats.clear()
        return result

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._stats.clear()


class LightweightDiagnostics:
    """Coordinate activation and optimizer diagnostics with one collective."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        output_dir: str,
        run_name: str,
        rank: int,
        world_size: int,
        product_interval: int,
        adjust_muon_lr: str,
        scheduler: str = "unknown",
        aux_adamw_lr_multiplier: float = 1.0,
        stable_decay_fraction: float = 0.3,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.rank = rank
        self.world_size = world_size
        self.product_interval = product_interval
        self.adjust_muon_lr = adjust_muon_lr
        self.scheduler = scheduler
        self.aux_adamw_lr_multiplier = aux_adamw_lr_multiplier
        self.stable_decay_fraction = stable_decay_fraction
        self.output_dir = Path(output_dir)
        if rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.activation = ActivationScalarCollector(model, enabled=rank == 0)
        self._param_names = {id(param): name for name, param in model.named_parameters()}
        self._params = {name: param for name, param in model.named_parameters()}
        self._observed_names = self._select_observed_names()
        self._row_by_name = {
            name: index for index, name in enumerate(self._observed_names)
        }
        self._pairs = self._select_lowrank_ffn_pairs()
        self._pair_by_name = {pair["base_name"]: pair for pair in self._pairs}
        self._pair_variants = {
            str(group["pair_name"]): str(group["factor_update_variant"])
            for group in optimizer.param_groups
            if group.get("factor_update_variant") is not None
        }
        self._pair_metric_layout = {
            name: index * len(_PAIR_METRIC_FIELDS)
            for index, name in enumerate(sorted(self._pair_variants))
        }
        self._pair_metric_values = len(self._pair_metric_layout) * len(
            _PAIR_METRIC_FIELDS
        )
        self._gram_layout, self._gram_values = self._build_gram_layout()
        self._scalar_values = len(self._observed_names) * len(_UPDATE_FIELDS)

        self._buffer: torch.Tensor | None = None
        self._read_buffer: torch.Tensor | None = None
        self._manual_decay: dict[int, tuple[float, float, float]] = {}
        self._product_due = False
        self._active = False
        self._step_started_at = 0.0
        self._update_number = 0
        self._base_lr = 0.0
        self._files: dict[str, Any] = {}
        if rank == 0:
            self._write_metadata(run_name)
        optimizer.update_observer = self.observe_update
        optimizer.pair_update_observer = self.observe_pair_update

    def _select_observed_names(self) -> list[str]:
        names = []
        for name, parameter in self.model.named_parameters():
            metadata = lowrank_factor_metadata(name)
            is_ffn_factor = metadata is not None and metadata["module_type"] == "ffn"
            is_fullrank_matrix = (
                parameter.ndim == 2
                and name.endswith(".weight")
                and (".attention." in name or ".feed_forward." in name)
            )
            if (
                is_ffn_factor
                or is_fullrank_matrix
                or name in {"tok_embeddings.weight", "output.weight"}
            ):
                names.append(name)
        return sorted(names)

    def _select_lowrank_ffn_pairs(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = defaultdict(dict)
        for name in self._observed_names:
            metadata = lowrank_factor_metadata(name)
            if metadata is None or metadata["module_type"] != "ffn":
                continue
            base_name = name.rsplit(".", 1)[0]
            grouped[base_name][metadata["factor"]] = name
            grouped[base_name]["layer"] = metadata["layer"]
            grouped[base_name]["matrix"] = metadata["matrix"]
        pairs = []
        for base_name, values in sorted(grouped.items()):
            if "A" not in values or "B" not in values:
                continue
            a = self._params[values["A"]]
            b = self._params[values["B"]]
            if a.ndim != 2 or b.ndim != 2:
                continue
            pairs.append(
                {
                    "base_name": base_name,
                    "a_name": values["A"],
                    "b_name": values["B"],
                    "layer": values["layer"],
                    "matrix": values["matrix"],
                    "rank": a.shape[1],
                    "output_features": a.shape[0],
                    "input_features": b.shape[1],
                }
            )
        return pairs

    def _build_gram_layout(self) -> tuple[dict[str, tuple[int, int]], int]:
        layout = {}
        offset = 0
        pair_names = {
            factor_name
            for pair in self._pairs
            for factor_name in (pair["a_name"], pair["b_name"])
        }
        for name in sorted(pair_names):
            parameter = self._params[name]
            rank = min(parameter.shape)
            size = 3 * rank * rank
            layout[name] = (offset, size)
            offset += size
        return layout, offset

    def _write_metadata(self, run_name: str) -> None:
        metadata = {
            "run_name": run_name,
            "activation_scope": "rank0_first_real_training_microbatch_each_step",
            "activation_retention": "scalar_sums_sumsq_max_count_only",
            "optimizer_collectives_per_step": 1,
            "product_interval": self.product_interval,
            "product_representation": "A_at_B_repo_orientation",
            "muon_adjustment": self.adjust_muon_lr,
            "aux_adamw_lr_multiplier": self.aux_adamw_lr_multiplier,
            "scheduler": self.scheduler,
            "stable_decay_fraction": self.stable_decay_fraction,
            "muon_adjustment_label": {
                "original": "keller_original",
                "match_rms_adamw": "moonshot_match_rms_adamw",
                "none": "none",
            }.get(self.adjust_muon_lr, self.adjust_muon_lr),
            "heavy_diagnostics_unchanged": True,
            "paired_factor_variants": self._pair_variants,
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    def begin_training_step(self, update_number: int) -> None:
        self._step_started_at = time.perf_counter()
        self._update_number = update_number
        self.activation.begin_step()

    def end_training_forward(self) -> None:
        self.activation.end_forward()

    def prepare_optimizer_step(
        self,
        *,
        base_lr: float,
        manual_decay: dict[int, tuple[float, float]],
    ) -> None:
        self._base_lr = float(base_lr)
        self._product_due = (
            self.product_interval > 0
            and self._update_number % self.product_interval == 0
        )
        length = self._scalar_values
        if self._product_due:
            length += self._gram_values + self._pair_metric_values
        device = next(self.model.parameters()).device
        self._buffer = torch.zeros(length, device=device, dtype=torch.float32)
        self._manual_decay = {
            param_id: (coefficient, decay_lr, 1.0 - coefficient * decay_lr)
            for param_id, (coefficient, decay_lr) in manual_decay.items()
        }
        self._active = True
        self.optimizer.pair_metrics_due = self._product_due

    @torch.no_grad()
    def _copy_factor_grams(
        self,
        target: torch.Tensor,
        parameter: torch.Tensor,
        direction: torch.Tensor,
        *,
        factor: str,
        chunk_size: int = 256,
    ) -> None:
        """Accumulate rank-sized Grams without casting a complete factor/update."""
        target.zero_()
        if factor == "A":
            target[0].copy_(parameter.mT @ parameter)
            for start in range(0, parameter.shape[0], chunk_size):
                parameter_chunk = parameter[start : start + chunk_size]
                direction_chunk = direction[start : start + chunk_size].to(
                    parameter.dtype
                )
                target[1].add_(parameter_chunk.mT @ direction_chunk)
                target[2].add_(direction_chunk.mT @ direction_chunk)
        else:
            target[0].copy_(parameter @ parameter.mT)
            for start in range(0, parameter.shape[1], chunk_size):
                parameter_chunk = parameter[:, start : start + chunk_size]
                direction_chunk = direction[:, start : start + chunk_size].to(
                    parameter.dtype
                )
                target[1].add_(parameter_chunk @ direction_chunk.mT)
                target[2].add_(direction_chunk @ direction_chunk.mT)

    @torch.no_grad()
    def observe_update(
        self,
        parameter: torch.Tensor,
        direction: torch.Tensor,
        group: dict,
        optimizer_kind: str,
    ) -> None:
        if not self._active or self._buffer is None:
            return
        name = self._param_names.get(id(parameter))
        if name not in self._row_by_name:
            return
        if optimizer_kind == "adam" and self.rank != 0:
            return

        row_index = self._row_by_name[name]
        start = row_index * len(_UPDATE_FIELDS)
        target = self._buffer[start : start + len(_UPDATE_FIELDS)]
        learning_rate = float(group["lr"])
        internal_weight_decay = float(group.get("weight_decay", 0.0))
        internal_decay = 1.0 - learning_rate * internal_weight_decay
        external_weight_decay, external_decay_lr, state_scale = self._manual_decay.get(
            id(parameter), (0.0, 0.0, 1.0)
        )
        if external_weight_decay:
            weight_decay = external_weight_decay
            weight_decay_lr = external_decay_lr
        else:
            weight_decay = internal_weight_decay
            weight_decay_lr = learning_rate
        total_decay = state_scale * internal_decay

        current_sumsq = torch.sum(parameter * parameter, dtype=torch.float32)
        direction_sumsq = torch.sum(direction * direction, dtype=torch.float32)
        current_direction = torch.sum(parameter * direction, dtype=torch.float32)
        alpha = internal_decay - 1.0 / state_scale
        old_sumsq = current_sumsq / (state_scale * state_scale)
        total_update_sumsq = (
            alpha * alpha * current_sumsq
            - 2.0 * alpha * learning_rate * current_direction
            + learning_rate * learning_rate * direction_sumsq
        )
        adjustment = (
            _shape_adjustment_multiplier(
                str(group.get("adjust_lr_fn", "none")), parameter.shape
            )
            if optimizer_kind == "muon"
            else 1.0
        )
        values = torch.stack(
            (
                torch.ones((), device=parameter.device),
                torch.tensor(learning_rate, device=parameter.device),
                torch.tensor(weight_decay, device=parameter.device),
                torch.tensor(weight_decay_lr, device=parameter.device),
                torch.tensor(state_scale, device=parameter.device),
                torch.tensor(total_decay, device=parameter.device),
                torch.tensor(adjustment, device=parameter.device),
                torch.tensor(parameter.numel(), device=parameter.device),
                old_sumsq,
                direction_sumsq,
                learning_rate * learning_rate * direction_sumsq,
                torch.clamp(total_update_sumsq, min=0.0),
            )
        ).to(torch.float32)
        target.copy_(values)

        if not self._product_due or name not in self._gram_layout:
            return
        offset, size = self._gram_layout[name]
        rank = int(math.sqrt(size // 3))
        gram_target = self._buffer[
            self._scalar_values + offset : self._scalar_values + offset + size
        ].view(3, rank, rank)
        self._copy_factor_grams(
            gram_target,
            parameter,
            direction,
            factor="A" if name.endswith(".A") else "B",
        )

    @torch.no_grad()
    def observe_pair_update(
        self,
        pair_name: str,
        metrics: dict[str, float],
        group: dict,
        optimizer_kind: str,
    ) -> None:
        if (
            not self._active
            or not self._product_due
            or self._buffer is None
            or pair_name not in self._pair_metric_layout
        ):
            return
        start = (
            self._scalar_values
            + self._gram_values
            + self._pair_metric_layout[pair_name]
        )
        target = self._buffer[start : start + len(_PAIR_METRIC_FIELDS)]
        values = {"present": 1.0, **metrics}
        target.copy_(
            torch.tensor(
                [float(values.get(field, 0.0)) for field in _PAIR_METRIC_FIELDS],
                device=target.device,
                dtype=target.dtype,
            )
        )

    def _scalar_record(self, name: str) -> dict[str, float]:
        source = self._read_buffer if self._read_buffer is not None else self._buffer
        if source is None:
            raise RuntimeError("No diagnostic buffer is available")
        row_index = self._row_by_name[name]
        start = row_index * len(_UPDATE_FIELDS)
        values = source[start : start + len(_UPDATE_FIELDS)].tolist()
        return dict(zip(_UPDATE_FIELDS, values))

    def _gram_record(self, name: str) -> tuple[torch.Tensor, ...]:
        source = self._read_buffer if self._read_buffer is not None else self._buffer
        if source is None:
            raise RuntimeError("No diagnostic buffer is available")
        offset, size = self._gram_layout[name]
        rank = int(math.sqrt(size // 3))
        values = source[
            self._scalar_values + offset : self._scalar_values + offset + size
        ].view(3, rank, rank)
        return values[0], values[1], values[2]

    def _pair_metric_record(self, pair_name: str) -> dict[str, float]:
        source = self._read_buffer if self._read_buffer is not None else self._buffer
        if source is None:
            raise RuntimeError("No diagnostic buffer is available")
        start = (
            self._scalar_values
            + self._gram_values
            + self._pair_metric_layout[pair_name]
        )
        values = source[start : start + len(_PAIR_METRIC_FIELDS)].tolist()
        return dict(zip(_PAIR_METRIC_FIELDS, values))

    def _parameter_rows(self) -> list[dict[str, Any]]:
        rows = []
        for name in self._observed_names:
            record = self._scalar_record(name)
            if record["present"] < 0.5:
                continue
            numel = max(record["numel"], 1.0)
            metadata = lowrank_factor_metadata(name)
            old_norm = _safe_sqrt(record["old_parameter_sumsq"])
            direction_rms = _safe_sqrt(record["adjusted_direction_sumsq"] / numel)
            applied_rms = _safe_sqrt(record["applied_direction_sumsq"] / numel)
            row = {
                "step": self._update_number,
                "param_name": name,
                "optimizer_kind": (
                    "adam"
                    if name in {"tok_embeddings.weight", "output.weight"}
                    else "muon"
                ),
                "muon_adjustment": (
                    self.adjust_muon_lr
                    if name not in {"tok_embeddings.weight", "output.weight"}
                    else None
                ),
                "learning_rate": record["learning_rate"],
                "weight_decay": record["weight_decay"],
                "weight_decay_lr": record["weight_decay_lr"],
                "decay_multiplier": record["decay_multiplier"],
                "external_decay_multiplier": record[
                    "external_decay_multiplier"
                ],
                "shape_adjustment_multiplier": record[
                    "shape_adjustment_multiplier"
                ],
                "parameter_rms_pre": old_norm / _safe_sqrt(numel),
                "parameter_frobenius_norm_pre": old_norm,
                "adjusted_direction_rms": direction_rms,
                "applied_direction_rms": applied_rms,
                "applied_direction_rms_per_lr": direction_rms,
                "relative_direction_update": _safe_sqrt(
                    record["applied_direction_sumsq"]
                    / max(record["old_parameter_sumsq"], 1e-30)
                ),
                "relative_total_update_including_weight_decay": _safe_sqrt(
                    record["total_update_sumsq"]
                    / max(record["old_parameter_sumsq"], 1e-30)
                ),
            }
            if metadata is not None:
                row.update(metadata)
            rows.append(row)
        return rows

    def _product_rows(self) -> list[dict[str, Any]]:
        if not self._product_due:
            return []
        rows = []
        for pair in self._pairs:
            a_record = self._scalar_record(pair["a_name"])
            b_record = self._scalar_record(pair["b_name"])
            if min(a_record["present"], b_record["present"]) < 0.5:
                continue
            a_raw = self._gram_record(pair["a_name"])
            b_raw = self._gram_record(pair["b_name"])

            def terms(raw, record, optimizer_only):
                learning_rate = record["learning_rate"]
                decay_multiplier = record["decay_multiplier"]
                weight_decay = record["weight_decay"]
                weight_decay_lr = record["weight_decay_lr"]
                state_scale = record["external_decay_multiplier"]
                internal_decay = decay_multiplier / state_scale
                return factor_gram_terms(
                    *raw,
                    state_scale=state_scale,
                    internal_decay_multiplier=internal_decay,
                    learning_rate=learning_rate,
                    optimizer_only=optimizer_only,
                )

            optimizer_only = product_metrics_from_gram_terms(
                terms(a_raw, a_record, True),
                terms(b_raw, b_record, True),
                output_features=pair["output_features"],
                input_features=pair["input_features"],
            )
            total = product_metrics_from_gram_terms(
                terms(a_raw, a_record, False),
                terms(b_raw, b_record, False),
                output_features=pair["output_features"],
                input_features=pair["input_features"],
            )
            eta_a = a_record["learning_rate"]
            eta_b = b_record["learning_rate"]
            normalization_lr = math.sqrt(max(eta_a * eta_b, 0.0))
            a_norm = _safe_sqrt(a_record["old_parameter_sumsq"])
            b_norm = _safe_sqrt(b_record["old_parameter_sumsq"])
            rows.append(
                {
                    "step": self._update_number,
                    "param_name": pair["base_name"],
                    "layer": pair["layer"],
                    "matrix": pair["matrix"],
                    "rank": pair["rank"],
                    "a_learning_rate": eta_a,
                    "b_learning_rate": eta_b,
                    "normalization_lr": normalization_lr,
                    "a_frobenius_norm_pre": a_norm,
                    "b_frobenius_norm_pre": b_norm,
                    "a_to_b_frobenius_ratio": a_norm / max(b_norm, 1e-30),
                    "effective_product_update_rms": optimizer_only["update_rms"],
                    "effective_product_update_rms_per_lr": optimizer_only[
                        "update_rms"
                    ]
                    / max(normalization_lr, 1e-30),
                    "relative_effective_product_update": optimizer_only[
                        "relative_update"
                    ],
                    "effective_product_update_rms_including_weight_decay": total[
                        "update_rms"
                    ],
                    "relative_effective_product_update_including_weight_decay": total[
                        "relative_update"
                    ],
                }
            )
        return rows

    def _pair_optimizer_rows(self) -> list[dict[str, Any]]:
        if not self._product_due:
            return []
        rows = []
        for pair_name, variant in sorted(self._pair_variants.items()):
            record = self._pair_metric_record(pair_name)
            if record["present"] < 0.5:
                continue
            pair = self._pair_by_name[pair_name]
            row = {
                "step": self._update_number,
                "param_name": pair_name,
                "layer": pair["layer"],
                "matrix": pair["matrix"],
                "rank": pair["rank"],
                "factor_update_variant": variant,
            }
            row.update(
                {
                    field: record[field]
                    for field in _PAIR_FIELDS_BY_VARIANT[variant]
                }
            )
            if variant in {"product_adamrms", "rankaware_product_adamrms"}:
                pair_group = next(
                    group
                    for group in self.optimizer.param_groups
                    if group.get("pair_name") == pair_name
                )
                target = row["target_first_order_direction_rms"] * float(
                    pair_group["lr"]
                )
                row["target_first_order_update_rms"] = target
                row["first_order_target_relative_error"] = abs(
                    row["first_order_update_rms"] - target
                ) / max(target, 1e-30)
            rows.append(row)
        return rows

    @staticmethod
    def _activation_rows(
        update_number: int, stats: dict[str, dict[str, float]]
    ) -> list[dict[str, Any]]:
        layers = sorted(
            {
                int(key.split(".", 1)[0].split("_", 1)[1])
                for key in stats
                if key.startswith("layer_")
            }
        )
        rows = []
        for layer in layers:
            prefix = f"layer_{layer}."
            get = lambda suffix: stats.get(prefix + suffix, {}).get("rms", float("nan"))
            attention_ratio = get("attention_branch") / max(
                get("pre_attention"), 1e-30
            )
            ffn_ratio = get("ffn_y") / max(get("post_attention"), 1e-30)
            row = {
                "step": update_number,
                "layer": layer,
                "pre_attention_rms": get("pre_attention"),
                "post_attention_norm_rms": get("post_attention_norm"),
                "attention_branch_rms": get("attention_branch"),
                "attention_branch_to_residual": attention_ratio,
                "post_attention_residual_rms": get("post_attention"),
                "pre_ffn_rms": get("post_attention"),
                "post_ffn_norm_rms": get("post_ffn_norm"),
                "ffn_branch_rms": get("ffn_y"),
                "ffn_branch_to_residual": ffn_ratio,
                "post_ffn_residual_rms": get("post_ffn"),
                "ffn_to_attention_branch_ratio": ffn_ratio
                / max(attention_ratio, 1e-30),
            }
            for name in ("ffn_u1", "ffn_u3", "ffn_g", "ffn_y"):
                row[f"{name}_rms"] = get(name)
                row[f"{name}_max_abs"] = stats.get(prefix + name, {}).get(
                    "max_abs", float("nan")
                )
            rows.append(row)
        return rows

    def _append_rows(self, filename: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        handle = self._files.get(filename)
        if handle is None:
            handle = (self.output_dir / filename).open("a", buffering=1)
            self._files[filename] = handle
        handle.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    def finish_step(
        self,
        *,
        loss: float,
        grad_norm: float,
        tokens_seen: int,
        tokens_this_step: int,
    ) -> None:
        if self._buffer is None:
            raise RuntimeError("prepare_optimizer_step must precede finish_step")
        self._active = False
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self._buffer, op=dist.ReduceOp.SUM)

        if self.rank == 0:
            self._read_buffer = self._buffer.cpu()
            activation_stats = self.activation.snapshot()
            layer_rows = self._activation_rows(self._update_number, activation_stats)
            parameter_rows = self._parameter_rows()
            product_rows = self._product_rows()
            pair_optimizer_rows = self._pair_optimizer_rows()
            elapsed = time.perf_counter() - self._step_started_at

            attention_ratios = [row["attention_branch_to_residual"] for row in layer_rows]
            ffn_ratios = [row["ffn_branch_to_residual"] for row in layer_rows]
            balance_ratios = [row["ffn_to_attention_branch_ratio"] for row in layer_rows]
            residuals = [row["pre_attention_rms"] for row in layer_rows]
            factor_updates = [
                row["applied_direction_rms_per_lr"]
                for row in parameter_rows
                if row.get("module_type") == "ffn"
            ]
            product_updates = [
                row["effective_product_update_rms_per_lr"] for row in product_rows
            ]
            product_relative = [
                row["relative_effective_product_update"] for row in product_rows
            ]
            logits = activation_stats.get("logits", {})
            step_row = {
                "step": self._update_number,
                "loss": float(loss),
                "tokens_seen": int(tokens_seen),
                "scheduled_base_lr": self._base_lr,
                "grad_norm": float(grad_norm),
                "step_time_seconds": elapsed,
                "tokens_per_second": tokens_this_step / max(elapsed, 1e-30),
                "max_memory_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                ),
                "max_memory_reserved_bytes": (
                    torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
                ),
                "product_grams_recorded": self._product_due,
                "embedding_rms": activation_stats.get("embedding", {}).get("rms"),
                "final_residual_rms": activation_stats.get("final_residual", {}).get("rms"),
                "final_normalized_rms": activation_stats.get("final_normalized", {}).get("rms"),
                "logits_rms": logits.get("rms"),
                "logits_std": logits.get("std"),
                "logits_max_abs": logits.get("max_abs"),
                "residual_rms_across_depth": _summary(residuals),
                "attention_branch_to_residual": _summary(attention_ratios),
                "ffn_branch_to_residual": _summary(ffn_ratios),
                "ffn_to_attention_branch_ratio": _summary(balance_ratios),
                "factor_update_rms_per_lr": _summary(factor_updates),
                "effective_product_update_rms_per_lr": _summary(product_updates),
                "relative_effective_product_update": _summary(product_relative),
            }
            self._append_rows("step_metrics.jsonl", [step_row])
            self._append_rows("layer_metrics.jsonl", layer_rows)
            self._append_rows("matrix_metrics.jsonl", parameter_rows + product_rows)
            self._append_rows(
                "pair_optimizer_metrics.jsonl", pair_optimizer_rows
            )

        self._manual_decay.clear()
        self._read_buffer = None
        self._buffer = None
        self._product_due = False
        self.optimizer.pair_metrics_due = False

    def close(self) -> None:
        self._active = False
        self.activation.close()
        if getattr(self.optimizer, "update_observer", None) == self.observe_update:
            self.optimizer.update_observer = None
        if (
            getattr(self.optimizer, "pair_update_observer", None)
            == self.observe_pair_update
        ):
            self.optimizer.pair_update_observer = None
        self.optimizer.pair_metrics_due = False
        for handle in self._files.values():
            handle.close()
        self._files.clear()
