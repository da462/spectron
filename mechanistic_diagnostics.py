"""Sparse mechanistic diagnostics for dense and low-rank FFN updates.

The diagnostics are deliberately side-effect free with respect to training.
At selected optimizer steps they use an immutable batch, restore RNG and
train/eval state, request activation gradients with ``autograd.grad`` instead
of populating parameter gradients, and write detached statistics after the
real optimizer step. No diagnostic model pass runs on ordinary training steps.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from low_rank_linear import LowRankLinear
from muon_local import _adjust_lr
from optimizer_routing import (
    lowrank_decoupled_weight_decay_lr,
    spectral_lr_target_matches,
)


EPS = 1e-12
FFN_MATRIX_NAMES = ("w1", "w2", "w3")


def normalize_device(device: torch.device | str | int) -> torch.device:
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def diagnostic_step(update_number: int, total_steps: int) -> bool:
    return (
        update_number in {1, 5, 10, 25, 50, 75, 100, 150, 200, 300}
        or update_number % 100 == 0
        or update_number == total_steps
    )


def full_matrix_gradient(inputs: torch.Tensor, output_grads: torch.Tensor) -> torch.Tensor:
    """Return dL/dW for ``y = x @ W.T`` using arbitrary leading dimensions."""
    x = inputs.detach().float().reshape(-1, inputs.shape[-1])
    d = output_grads.detach().float().reshape(-1, output_grads.shape[-1])
    return d.mT @ x


def spectral_steepest_descent_oracle(
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``-UV^T`` and singular values for the instantaneous gradient."""
    u, singular_values, vh = torch.linalg.svd(gradient.float(), full_matrices=False)
    return -(u @ vh), singular_values


def product_update_decomposition(
    a_before: torch.Tensor,
    b_before: torch.Tensor,
    a_after: torch.Tensor,
    b_after: torch.Tensor,
) -> dict[str, torch.Tensor]:
    d_a = a_after - a_before
    d_b = b_after - b_before
    exact = a_after @ b_after - a_before @ b_before
    first = d_a @ b_before + a_before @ d_b
    second = d_a @ d_b
    return {
        "delta_a": d_a,
        "delta_b": d_b,
        "exact": exact,
        "first": first,
        "second": second,
        "reconstruction_error": exact - first - second,
    }


def singular_metrics(delta: torch.Tensor, attainable_rank: int) -> dict[str, Any]:
    values = torch.linalg.svdvals(delta.float())
    sigma_1 = values[0] if values.numel() else delta.new_tensor(0.0).float()
    frobenius = torch.linalg.vector_norm(values)
    stable_rank = frobenius.square() / sigma_1.square().clamp_min(EPS)
    k = max(1, min(int(attainable_rank), delta.shape[-2], delta.shape[-1]))
    normalized = values[:32] / sigma_1.clamp_min(EPS)
    return {
        "spectral_norm": float(sigma_1),
        "frobenius_norm": float(frobenius),
        "stable_rank": float(stable_rank),
        "attainable_rank": k,
        "attainable_rank_flatness": float(stable_rank / k),
        "normalized_singular_values_top32": normalized.cpu().tolist(),
        "singular_values_top32": values[:32].cpu().tolist(),
    }


def lowrank_subspaces(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute compact SVD subspaces for W=AB without materializing a full SVD."""
    q_a, r_a = torch.linalg.qr(a.float(), mode="reduced")
    q_b, r_b = torch.linalg.qr(b.float().mT, mode="reduced")
    u_core, singular_values, vh_core = torch.linalg.svd(
        r_a @ r_b.mT, full_matrices=False
    )
    if singular_values.numel() == 0:
        return q_a[:, :0], q_b[:, :0], singular_values
    keep = singular_values > singular_values[0] * 1e-6
    u = q_a @ u_core[:, keep]
    v = q_b @ vh_core.mT[:, keep]
    return u, v, singular_values[keep]


def tangent_projection(z: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    left = u @ (u.mT @ z)
    right = (z @ v) @ v.mT
    overlap = u @ ((u.mT @ z) @ v) @ v.mT
    return left + right - overlap


def tangent_motion_fractions(
    delta: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> dict[str, float]:
    uv = (u.mT @ delta) @ v
    core = u @ uv @ v.mT
    delta_v = delta @ v
    left = (delta_v - u @ (u.mT @ delta_v)) @ v.mT
    u_delta = u.mT @ delta
    right = u @ (u_delta - (u_delta @ v) @ v.mT)
    normal = delta - core - left - right
    total_energy = delta.float().square().sum().clamp_min(EPS)
    return {
        "core_energy_fraction": float(core.square().sum() / total_energy),
        "left_energy_fraction": float(left.square().sum() / total_energy),
        "right_energy_fraction": float(right.square().sum() / total_energy),
        "normal_energy_fraction": float(normal.square().sum() / total_energy),
    }


def tensor_distribution(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.detach().float().abs().flatten()
    return {
        "rms": float(rms(tensor)),
        "p99_abs": float(torch.quantile(flat, 0.99)),
        "p999_abs": float(torch.quantile(flat, 0.999)),
        "max_abs": float(flat.max()),
    }


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float((a * b).sum() / denominator.clamp_min(EPS))


def factor_condition_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    gram_a = a.mT @ a
    gram_b = b @ b.mT
    singular_a = torch.linalg.svdvals(a)
    singular_b = torch.linalg.svdvals(b)
    denominator = (
        torch.linalg.vector_norm(gram_a) + torch.linalg.vector_norm(gram_b)
    ).clamp_min(EPS)
    return {
        "a_spectral_norm": float(torch.linalg.matrix_norm(a, ord=2)),
        "b_spectral_norm": float(torch.linalg.matrix_norm(b, ord=2)),
        "a_frobenius_norm": float(torch.linalg.vector_norm(a)),
        "b_frobenius_norm": float(torch.linalg.vector_norm(b)),
        "a_gram_condition": float(
            (singular_a[0] / singular_a[-1].clamp_min(EPS)).square()
        ),
        "b_gram_condition": float(
            (singular_b[0] / singular_b[-1].clamp_min(EPS)).square()
        ),
        "balance_defect": float(torch.linalg.vector_norm(gram_a - gram_b) / denominator),
    }


def per_token_linearized_loss_change(
    inputs: torch.Tensor,
    output_grads: torch.Tensor,
    delta: torch.Tensor,
) -> dict[str, float]:
    x = inputs.detach().float().reshape(-1, inputs.shape[-1])
    d = output_grads.detach().float().reshape(-1, output_grads.shape[-1])
    values = (d * (x @ delta.float().mT)).sum(dim=-1)
    abs_values = values.abs()
    return {
        "linearized_loss_change_mean": float(values.mean()),
        "linearized_loss_change_rms": float(rms(values)),
        "linearized_loss_change_p95_abs": float(torch.quantile(abs_values, 0.95)),
        "linearized_loss_change_p99_abs": float(torch.quantile(abs_values, 0.99)),
        "linearized_loss_change_max_abs": float(abs_values.max()),
    }


@contextmanager
def preserve_training_state(model: nn.Module, device: torch.device):
    was_training = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = None
    if device.type == "cuda":
        cuda_rng = torch.cuda.get_rng_state(device)
    try:
        model.eval()
        yield
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)


@dataclass
class _MatrixCapture:
    module: nn.Module
    inputs: torch.Tensor
    output: torch.Tensor


class _ActivationCapture:
    def __init__(self, model: nn.Module, capture_functions: bool):
        self.model = model
        self.capture_functions = capture_functions
        self.matrices: dict[str, _MatrixCapture] = {}
        self.layers: dict[int, dict[str, torch.Tensor]] = {}
        self.model_values: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _matrix_hook(self, name: str, module: nn.Module, inputs, output) -> None:
        self.matrices[name] = _MatrixCapture(module, inputs[0].detach(), output)

    def _layer_pre_hook(self, layer_index: int, key: str, module, inputs) -> None:
        if self.capture_functions:
            self.layers.setdefault(layer_index, {})[key] = inputs[0].detach()

    def _layer_output_hook(self, layer_index: int, key: str, module, inputs, output) -> None:
        if self.capture_functions:
            self.layers.setdefault(layer_index, {})[key] = output.detach()

    def _model_pre_hook(self, key: str, module, inputs) -> None:
        if self.capture_functions:
            self.model_values[key] = inputs[0].detach()

    def _model_output_hook(self, key: str, module, inputs, output) -> None:
        if self.capture_functions:
            self.model_values[key] = output.detach()

    def __enter__(self):
        for layer_name, layer in self.model.layers.items():
            layer_index = int(layer_name)
            for matrix_name in FFN_MATRIX_NAMES:
                matrix = getattr(layer.feed_forward, matrix_name)
                full_name = f"layers.{layer_name}.feed_forward.{matrix_name}"
                self.handles.append(
                    matrix.register_forward_hook(
                        lambda module, inputs, output, name=full_name: self._matrix_hook(
                            name, module, inputs, output
                        )
                    )
                )
            if self.capture_functions:
                self.handles.append(
                    layer.attention_norm.register_forward_pre_hook(
                        lambda module, inputs, index=layer_index: self._layer_pre_hook(
                            index, "attention_residual", module, inputs
                        )
                    )
                )
                self.handles.append(
                    layer.attention_norm.register_forward_hook(
                        lambda module, inputs, output, index=layer_index: self._layer_output_hook(
                            index, "attention_normalized_input", module, inputs, output
                        )
                    )
                )
                self.handles.append(
                    layer.attention.register_forward_hook(
                        lambda module, inputs, output, index=layer_index: self._layer_output_hook(
                            index, "attention_output", module, inputs, output
                        )
                    )
                )
                self.handles.append(
                    layer.ffn_norm.register_forward_pre_hook(
                        lambda module, inputs, index=layer_index: self._layer_pre_hook(
                            index, "residual", module, inputs
                        )
                    )
                )
                self.handles.append(
                    layer.ffn_norm.register_forward_hook(
                        lambda module, inputs, output, index=layer_index: self._layer_output_hook(
                            index, "normalized_input", module, inputs, output
                        )
                    )
                )
                self.handles.append(
                    layer.feed_forward.register_forward_hook(
                        lambda module, inputs, output, index=layer_index: self._layer_output_hook(
                            index, "ffn_output", module, inputs, output
                        )
                    )
                )
        if self.capture_functions:
            self.handles.append(
                self.model.tok_embeddings.register_forward_hook(
                    lambda module, inputs, output: self._model_output_hook(
                        "embedding_output", module, inputs, output
                    )
                )
            )
            self.handles.append(
                self.model.norm.register_forward_pre_hook(
                    lambda module, inputs: self._model_pre_hook(
                        "final_residual", module, inputs
                    )
                )
            )
            self.handles.append(
                self.model.norm.register_forward_hook(
                    lambda module, inputs, output: self._model_output_hook(
                        "final_normalized", module, inputs, output
                    )
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()


class MechanisticDiagnostics:
    """Emit sparse fixed-batch pre/post optimizer diagnostics."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        diagnostic_batch_path: str,
        output_dir: str,
        run_name: str,
        total_steps: int,
        device: torch.device | str | int,
        bf16: bool,
        adjust_muon_lr: str,
        spectral_lr_scaling: bool,
        spectral_lr_target: str,
        weight_decay: float,
        embedding_init_std: float,
    ) -> None:
        normalized_device = normalize_device(device)
        payload = torch.load(diagnostic_batch_path, map_location="cpu", weights_only=True)
        self.tokens = payload["input_ids"].to(normalized_device)
        self.labels = payload["labels"].to(normalized_device)
        self.batch_sha256 = payload.get("sha256", "unknown")
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.total_steps = total_steps
        self.device = normalized_device
        self.bf16 = bf16
        self.adjust_muon_lr = adjust_muon_lr
        self.spectral_lr_scaling = spectral_lr_scaling
        self.spectral_lr_target = spectral_lr_target
        self.weight_decay = weight_decay
        self.embedding_init_std = embedding_init_std
        self.pending: dict[str, Any] | None = None
        self._write_metadata()

    def _write_metadata(self) -> None:
        model_args = self.model.model_args
        metadata = {
            "run_name": self.run_name,
            "diagnostic_batch_sha256": self.batch_sha256,
            "diagnostic_batch_shape": list(self.tokens.shape),
            "total_steps": self.total_steps,
            "embedding_init_std": self.embedding_init_std,
            "adjust_muon_lr": self.adjust_muon_lr,
            "spectral_lr_scaling": self.spectral_lr_scaling,
            "spectral_lr_target": self.spectral_lr_target,
            "weight_decay": self.weight_decay,
            "diagnostic_steps": [
                step
                for step in range(1, self.total_steps + 1)
                if diagnostic_step(step, self.total_steps)
            ],
            "full_matrix_reference": {
                "kind": "instantaneous_spectral_steepest_descent_oracle",
                "gradient": "diagnostic_batch_dL_dW",
                "direction": "negative_exact_compact_svd_matrix_sign",
                "momentum": False,
            },
            "model": {
                "dim": model_args.dim,
                "layers": model_args.n_layers,
                "heads": model_args.n_heads,
                "kv_heads": model_args.n_kv_heads,
                "vocab_size": model_args.vocab_size,
                "sequence_length": model_args.max_seq_len,
            },
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _append(self, filename: str, row: dict[str, Any]) -> None:
        row = {"run_name": self.run_name, **row}
        with (self.output_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    def _autocast(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.bf16,
        )

    def _optimizer_group(self, parameter: nn.Parameter) -> dict[str, Any]:
        for group in self.optimizer.param_groups:
            if any(candidate is parameter for candidate in group["params"]):
                return group
        raise KeyError("Parameter is absent from optimizer groups")

    def _factor_audit(
        self,
        name: str,
        parameter: nn.Parameter,
        base_lr: float,
        spectral_scaling: dict[str, float],
    ) -> dict[str, Any]:
        group = self._optimizer_group(parameter)
        actual_lr = float(group["lr"])
        lowrank_wd = float(group.get("lowrank_weight_decay", group.get("weight_decay", 0.0)))
        is_spectron = self.spectral_lr_scaling and spectral_lr_target_matches(
            name, self.spectral_lr_target
        )
        decay_lr = (
            lowrank_decoupled_weight_decay_lr(
                group,
                base_lr=base_lr,
                spectral_lr_target=self.spectral_lr_target,
            )
            if self.spectral_lr_scaling
            else actual_lr
        )
        scale = spectral_scaling.get(name)
        lr_multiplier = float(group.get("lr_multiplier", 1.0))
        return {
            "actual_lr": actual_lr,
            "nominal_factor_lr": float(base_lr) * lr_multiplier,
            "lr_multiplier": lr_multiplier,
            "weight_decay": lowrank_wd,
            "decay_lr": float(decay_lr),
            "decay_multiplier": 1.0 - float(decay_lr) * lowrank_wd,
            "muon_shape_multiplier": _adjust_lr(
                1.0, self.adjust_muon_lr, parameter.shape
            ),
            "spectron_targeted": is_spectron,
            "spectron_scale": None if scale is None else float(scale),
            "spectron_denominator": None if not scale else 1.0 / float(scale),
        }

    def _dense_audit(self, parameter: nn.Parameter) -> dict[str, Any]:
        group = self._optimizer_group(parameter)
        lr = float(group["lr"])
        wd = float(group.get("weight_decay", 0.0))
        return {
            "actual_lr": lr,
            "weight_decay": wd,
            "decay_lr": lr,
            "decay_multiplier": 1.0 - lr * wd,
            "muon_shape_multiplier": _adjust_lr(
                1.0, self.adjust_muon_lr, parameter.shape
            ),
            "spectron_targeted": False,
            "spectron_scale": None,
            "spectron_denominator": None,
        }

    def _parameter_audit(self, parameter: nn.Parameter) -> dict[str, Any]:
        group = self._optimizer_group(parameter)
        lr = float(group["lr"])
        wd = float(group.get("weight_decay", 0.0))
        return {
            "actual_lr": lr,
            "weight_decay": wd,
            "decay_lr": lr,
            "decay_multiplier": 1.0 - lr * wd,
            "use_muon": bool(group.get("use_muon", False)),
        }

    def _parameter_snapshot(self, parameter: nn.Parameter) -> dict[str, Any]:
        gradient = parameter.grad
        return {
            "parameter": parameter,
            "weight": parameter.detach().float().clone(),
            "gradient": (
                torch.zeros_like(parameter, dtype=torch.float32)
                if gradient is None
                else gradient.detach().float().clone()
            ),
            "audit": self._parameter_audit(parameter),
        }

    def before_optimizer_step(
        self,
        *,
        update_number: int,
        base_lr: float,
        spectral_scaling: dict[str, float],
    ) -> None:
        if not diagnostic_step(update_number, self.total_steps):
            self.pending = None
            return

        with preserve_training_state(self.model, self.device):
            with _ActivationCapture(self.model, capture_functions=True) as capture:
                with self._autocast():
                    logits = self.model(self.tokens, input_batch=self.tokens)
                    loss = self.criterion(
                        logits.reshape(-1, logits.shape[-1]), self.labels.reshape(-1)
                    )
                names = list(capture.matrices)
                outputs = [capture.matrices[name].output for name in names]
                output_grads = torch.autograd.grad(loss, outputs, retain_graph=False)

            oracle: dict[str, dict[str, torch.Tensor]] = {}
            for name, output_grad in zip(names, output_grads):
                matrix_capture = capture.matrices[name]
                gradient = full_matrix_gradient(matrix_capture.inputs, output_grad)
                direction, singular_values = spectral_steepest_descent_oracle(gradient)
                oracle[name] = {
                    "gradient": gradient,
                    "direction": direction,
                    "gradient_singular_values": singular_values,
                    "inputs": matrix_capture.inputs,
                    "output_grads": output_grad.detach(),
                }

            matrices: dict[str, dict[str, Any]] = {}
            for name, matrix_capture in capture.matrices.items():
                module = matrix_capture.module
                if isinstance(module, LowRankLinear):
                    a_name = f"{name}.A"
                    b_name = f"{name}.B"
                    matrices[name] = {
                        "kind": "lowrank",
                        "module": module,
                        "a": module.A.detach().float().clone(),
                        "b": module.B.detach().float().clone(),
                        "a_audit": self._factor_audit(
                            a_name, module.A, base_lr, spectral_scaling
                        ),
                        "b_audit": self._factor_audit(
                            b_name, module.B, base_lr, spectral_scaling
                        ),
                    }
                else:
                    matrices[name] = {
                        "kind": "dense",
                        "module": module,
                        "weight": module.weight.detach().float().clone(),
                        "audit": self._dense_audit(module.weight),
                    }

            detached_layers = {
                index: {key: value.detach().clone() for key, value in values.items()}
                for index, values in capture.layers.items()
            }
            for layer_index, values in detached_layers.items():
                prefix = f"layers.{layer_index}.feed_forward"
                values["gate_preactivation"] = capture.matrices[
                    f"{prefix}.w1"
                ].output.detach().clone()
                values["up_preactivation"] = capture.matrices[
                    f"{prefix}.w3"
                ].output.detach().clone()
                values["gated_intermediate"] = capture.matrices[
                    f"{prefix}.w2"
                ].inputs.detach().clone()
            whole_model = {
                key: value.detach().clone()
                for key, value in capture.model_values.items()
            }
            whole_model["embedding"] = self._parameter_snapshot(
                self.model.tok_embeddings.weight
            )
            whole_model["lm_head"] = self._parameter_snapshot(
                self.model.output.weight
            )
            self.pending = {
                "update_number": update_number,
                "base_lr": float(base_lr),
                "loss_pre": float(loss.detach()),
                "logits_pre": logits.detach(),
                "layers": detached_layers,
                "matrices": matrices,
                "oracle": oracle,
                "whole_model": whole_model,
            }

    def _write_function_metrics(self, pending: dict[str, Any]) -> None:
        for layer_index, values in pending["layers"].items():
            residual = values["residual"]
            h = values["normalized_input"]
            f_pre = values["ffn_output"]
            layer = self.model.layers[str(layer_index)]
            with self._autocast():
                f_post = layer.feed_forward(h)
            delta_f = f_post.float() - f_pre.float()
            post_state = residual.float() + f_post.float()
            pre_state = residual.float() + f_pre.float()
            post_norm = torch.nn.functional.rms_norm(
                post_state, (post_state.shape[-1],), eps=self.model.model_args.norm_eps
            )
            pre_norm = torch.nn.functional.rms_norm(
                pre_state, (pre_state.shape[-1],), eps=self.model.model_args.norm_eps
            )
            attention_residual = values["attention_residual"]
            attention_output = values["attention_output"]
            attention_h = values["attention_normalized_input"]
            with self._autocast():
                attention_post = layer.attention(
                    attention_h,
                    self.model.freqs_cis.to(attention_h.device),
                )
            delta_attention = attention_post.float() - attention_output.float()
            attention_post_state = attention_residual.float() + attention_post.float()
            attention_pre_state = attention_residual.float() + attention_output.float()
            attention_post_norm = torch.nn.functional.rms_norm(
                attention_post_state,
                (attention_post_state.shape[-1],),
                eps=self.model.model_args.norm_eps,
            )
            attention_pre_norm = torch.nn.functional.rms_norm(
                attention_pre_state,
                (attention_pre_state.shape[-1],),
                eps=self.model.model_args.norm_eps,
            )
            ffn_q = rms(delta_f) / rms(residual).clamp_min(EPS)
            attention_q = rms(delta_attention) / rms(attention_residual).clamp_min(EPS)
            row = {
                "step": pending["update_number"],
                "layer": layer_index,
                "residual_rms": float(rms(residual)),
                "branch_rms": float(rms(f_pre)),
                "branch_to_residual": float(rms(f_pre) / rms(residual).clamp_min(EPS)),
                "local_update_u": float(rms(delta_f) / rms(f_pre).clamp_min(EPS)),
                "local_update_q": float(ffn_q),
                "normalized_state_displacement_z": float(rms(post_norm - pre_norm)),
                "attention_residual_rms": float(rms(attention_residual)),
                "attention_branch_rms": float(rms(attention_output)),
                "attention_branch_to_residual": float(
                    rms(attention_output) / rms(attention_residual).clamp_min(EPS)
                ),
                "attention_local_update_u": float(
                    rms(delta_attention) / rms(attention_output).clamp_min(EPS)
                ),
                "attention_local_update_q": float(attention_q),
                "attention_normalized_state_displacement_z": float(
                    rms(attention_post_norm - attention_pre_norm)
                ),
                "ffn_to_attention_q_ratio": float(
                    ffn_q / attention_q.clamp_min(EPS)
                ),
                "gate_preactivation": tensor_distribution(
                    values["gate_preactivation"]
                ),
                "up_preactivation": tensor_distribution(
                    values["up_preactivation"]
                ),
                "gated_intermediate": tensor_distribution(
                    values["gated_intermediate"]
                ),
                "ffn_output": tensor_distribution(f_pre),
            }
            self._append("function_metrics.jsonl", row)

    def _parameter_update_metrics(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        before = snapshot["weight"]
        after = snapshot["parameter"].detach().float()
        update = after - before
        weight_rms = rms(before).clamp_min(EPS)
        return {
            "weight_rms_pre": float(weight_rms),
            "weight_rms_post": float(rms(after)),
            "gradient_rms": float(rms(snapshot["gradient"])),
            "gradient_to_weight_rms": float(rms(snapshot["gradient"]) / weight_rms),
            "update_rms": float(rms(update)),
            "relative_update_rms": float(rms(update) / weight_rms),
            "gradient_source": "training_batch_after_global_clip",
            "optimizer": snapshot["audit"],
        }

    def _write_model_metrics(
        self,
        pending: dict[str, Any],
        logits_post: torch.Tensor,
        loss_post: torch.Tensor,
        output_kl: torch.Tensor,
    ) -> None:
        whole_model = pending["whole_model"]
        logits_pre = pending["logits_pre"].float()
        log_probs_pre = torch.log_softmax(logits_pre, dim=-1)
        probabilities_pre = log_probs_pre.exp()
        entropy_pre = -(probabilities_pre * log_probs_pre).sum(dim=-1).mean()
        logits_post_float = logits_post.float()
        log_probs_post = torch.log_softmax(logits_post_float, dim=-1)
        entropy_post = -(log_probs_post.exp() * log_probs_post).sum(dim=-1).mean()
        embedding_output_pre = whole_model["embedding_output"].float()
        embedding_output_post = self.model.tok_embeddings(self.tokens).float()
        self._append(
            "model_metrics.jsonl",
            {
                "step": pending["update_number"],
                "embedding_output_rms_pre": float(rms(embedding_output_pre)),
                "embedding_output_rms_post": float(rms(embedding_output_post)),
                "embedding_output_update_rms": float(
                    rms(embedding_output_post - embedding_output_pre)
                ),
                "embedding": self._parameter_update_metrics(
                    whole_model["embedding"]
                ),
                "final_residual_rms": float(rms(whole_model["final_residual"])),
                "final_normalized_rms": float(
                    rms(whole_model["final_normalized"])
                ),
                "lm_head": self._parameter_update_metrics(whole_model["lm_head"]),
                "logits_rms_pre": float(rms(logits_pre)),
                "logits_rms_post": float(rms(logits_post_float)),
                "logits_std_pre": float(logits_pre.std(unbiased=False)),
                "logits_std_post": float(logits_post_float.std(unbiased=False)),
                "logits_max_abs_pre": float(logits_pre.abs().max()),
                "logits_max_abs_post": float(logits_post_float.abs().max()),
                "prediction_entropy_pre": float(entropy_pre),
                "prediction_entropy_post": float(entropy_post),
                "diagnostic_ce_pre": pending["loss_pre"],
                "diagnostic_ce_post": float(loss_post),
                "logit_update_rms": float(rms(logits_post_float - logits_pre)),
                "output_kl_pre_to_post": float(output_kl),
            },
        )

    def _write_matrix_metrics(self, pending: dict[str, Any]) -> None:
        for name, snapshot in pending["matrices"].items():
            module = snapshot["module"]
            oracle = pending["oracle"][name]
            if snapshot["kind"] == "lowrank":
                a_before = snapshot["a"]
                b_before = snapshot["b"]
                a_after = module.A.detach().float()
                b_after = module.B.detach().float()
                decomposition = product_update_decomposition(
                    a_before, b_before, a_after, b_after
                )
                delta = decomposition["exact"]
                attainable_rank = min(
                    2 * module.rank, module.out_features, module.in_features
                )
                a_scale = snapshot["a_audit"]["decay_multiplier"]
                b_scale = snapshot["b_audit"]["decay_multiplier"]
                decay_weight = (a_before * a_scale) @ (b_before * b_scale)
                before_weight = a_before @ b_before
                after_weight = a_after @ b_after
                decay_delta = decay_weight - before_weight
                gradient_delta = after_weight - decay_weight
                second_f = torch.linalg.vector_norm(decomposition["second"])
                exact_f = torch.linalg.vector_norm(delta).clamp_min(EPS)
                second_s = torch.linalg.matrix_norm(
                    decomposition["second"], ord=2
                )
                exact_s = torch.linalg.matrix_norm(delta, ord=2).clamp_min(EPS)
                u, v, _ = lowrank_subspaces(a_before, b_before)
                projected = tangent_projection(oracle["gradient"], u, v)
                capture = torch.linalg.vector_norm(projected) / torch.linalg.vector_norm(
                    oracle["gradient"]
                ).clamp_min(EPS)
                row = {
                    "step": pending["update_number"],
                    "matrix": name,
                    "kind": "lowrank",
                    "shape": list(delta.shape),
                    "rank": module.rank,
                    "decomposition_relative_error": float(
                        torch.linalg.vector_norm(decomposition["reconstruction_error"])
                        / exact_f
                    ),
                    "second_order_frobenius_fraction": float(second_f / exact_f),
                    "second_order_spectral_fraction": float(second_s / exact_s),
                    "first_second_cosine": cosine_similarity(
                        decomposition["first"], decomposition["second"]
                    ),
                    "decay_product_update_frobenius": float(
                        torch.linalg.vector_norm(decay_delta)
                    ),
                    "gradient_product_update_frobenius": float(
                        torch.linalg.vector_norm(gradient_delta)
                    ),
                    "instantaneous_gradient_tangent_capture": float(capture),
                    "instantaneous_tangent_gradient_alignment": cosine_similarity(
                        delta, -projected
                    ),
                    **tangent_motion_fractions(delta, u, v),
                    **factor_condition_metrics(a_after, b_after),
                    "factor_a": snapshot["a_audit"],
                    "factor_b": snapshot["b_audit"],
                }
            else:
                before_weight = snapshot["weight"]
                after_weight = module.weight.detach().float()
                delta = after_weight - before_weight
                attainable_rank = min(delta.shape)
                scale = snapshot["audit"]["decay_multiplier"]
                decay_weight = before_weight * scale
                row = {
                    "step": pending["update_number"],
                    "matrix": name,
                    "kind": "dense",
                    "shape": list(delta.shape),
                    "decay_update_frobenius": float(
                        torch.linalg.vector_norm(decay_weight - before_weight)
                    ),
                    "gradient_update_frobenius": float(
                        torch.linalg.vector_norm(after_weight - decay_weight)
                    ),
                    "optimizer": snapshot["audit"],
                }

            spectrum = singular_metrics(delta, attainable_rank)
            oracle_direction = oracle["direction"]
            instantaneous_gradient = oracle["gradient"]
            gradient_singular = oracle["gradient_singular_values"]
            spectral_efficiency = -(
                instantaneous_gradient * delta
            ).sum() / (
                gradient_singular.sum()
                * torch.linalg.matrix_norm(delta.float(), ord=2).clamp_min(EPS)
            ).clamp_min(EPS)
            row.update(
                {
                    **spectrum,
                    "full_matrix_reference": (
                        "instantaneous_spectral_steepest_descent_oracle"
                    ),
                    "instantaneous_spectral_oracle_alignment": cosine_similarity(
                        delta, oracle_direction
                    ),
                    "instantaneous_gradient_descent_alignment": cosine_similarity(
                        delta, -instantaneous_gradient
                    ),
                    "instantaneous_spectral_descent_efficiency": float(
                        spectral_efficiency
                    ),
                    "instantaneous_gradient_nuclear_norm": float(
                        gradient_singular.sum()
                    ),
                    "instantaneous_gradient_spectral_norm": float(
                        gradient_singular[0]
                    ),
                    "instantaneous_gradient_singular_values_top32": (
                        gradient_singular[:32].cpu().tolist()
                    ),
                    **per_token_linearized_loss_change(
                        oracle["inputs"], oracle["output_grads"], delta
                    ),
                }
            )
            self._append("matrix_metrics.jsonl", row)

    def after_optimizer_step(self) -> None:
        if self.pending is None:
            return
        pending = self.pending
        with preserve_training_state(self.model, self.device):
            with torch.no_grad():
                self._write_function_metrics(pending)
                self._write_matrix_metrics(pending)
                with self._autocast():
                    logits_post = self.model(self.tokens, input_batch=self.tokens)
                    loss_post = self.criterion(
                        logits_post.reshape(-1, logits_post.shape[-1]),
                        self.labels.reshape(-1),
                    )
                pre_log_probs = torch.log_softmax(pending["logits_pre"].float(), dim=-1)
                post_log_probs = torch.log_softmax(logits_post.float(), dim=-1)
                pre_probs = pre_log_probs.exp()
                kl = (pre_probs * (pre_log_probs - post_log_probs)).sum(dim=-1).mean()
                self._write_model_metrics(pending, logits_post, loss_post, kl)
                self._append(
                    "step_metrics.jsonl",
                    {
                        "step": pending["update_number"],
                        "base_lr": pending["base_lr"],
                        "diagnostic_loss_pre": pending["loss_pre"],
                        "diagnostic_loss_post": float(loss_post),
                        "diagnostic_loss_delta": float(loss_post) - pending["loss_pre"],
                        "logit_update_rms": float(
                            rms(logits_post.float() - pending["logits_pre"].float())
                        ),
                        "output_kl_pre_to_post": float(kl),
                    },
                )
        self.pending = None
