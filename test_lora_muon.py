import copy
import unittest

import torch

from lora_muon import (
    apply_lora_muon_step,
    lora_muon_factor_directions,
    matrix_inverse_sqrt_newton_schulz,
)
from muon_local import SingleDeviceMuonWithAuxAdam, muon_update


def reference_lora_muon_product_direction(
    gradient: torch.Tensor,
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
) -> torch.Tensor:
    """Exact product direction from the LoRA-Muon tangent projectors."""
    q_a, _ = torch.linalg.qr(factor_a.double(), mode="reduced")
    q_b, _ = torch.linalg.qr(factor_b.mT.double(), mode="reduced")

    def matrix_sign(matrix: torch.Tensor) -> torch.Tensor:
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u @ vh

    work_gradient = gradient.double()
    return -0.5 * (
        matrix_sign(work_gradient @ q_b) @ q_b.mT
        + q_a @ matrix_sign(q_a.mT @ work_gradient)
    )


class LoRAMuonMathTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.a = torch.randn(11, 4, dtype=torch.float64)
        self.b = torch.randn(4, 9, dtype=torch.float64)
        self.gradient = torch.randn(11, 9, dtype=torch.float64)
        self.grad_a = self.gradient @ self.b.mT
        self.grad_b = self.a.mT @ self.gradient

    def test_factor_formula_matches_projector_product_direction(self) -> None:
        direction_a, direction_b = lora_muon_factor_directions(
            self.a,
            self.b,
            self.grad_a,
            self.grad_b,
            exact=True,
        )
        actual = direction_a @ self.b + self.a @ direction_b
        expected = reference_lora_muon_product_direction(
            self.gradient, self.a, self.b
        )
        torch.testing.assert_close(
            actual, expected.to(actual.dtype), rtol=1e-6, atol=1e-6
        )

    def test_production_direction_tracks_exact_formula(self) -> None:
        factor_a = self.a.float()
        factor_b = self.b.float()
        grad_a = self.grad_a.float()
        grad_b = self.grad_b.float()
        exact_a, exact_b = lora_muon_factor_directions(
            factor_a, factor_b, grad_a, grad_b, exact=True
        )
        actual_a, actual_b = lora_muon_factor_directions(
            factor_a, factor_b, grad_a, grad_b
        )
        exact_product = exact_a @ factor_b + factor_a @ exact_b
        actual_product = actual_a @ factor_b + factor_a @ actual_b
        cosine = torch.nn.functional.cosine_similarity(
            actual_product.flatten(), exact_product.flatten(), dim=0
        )
        self.assertGreater(float(cosine), 0.99)

    def test_scalar_gauge_covariance(self) -> None:
        reference_a, reference_b = lora_muon_factor_directions(
            self.a,
            self.b,
            self.grad_a,
            self.grad_b,
            exact=True,
        )
        for scale in (0.25, 3.0, 11.0):
            actual_a, actual_b = lora_muon_factor_directions(
                scale * self.a,
                self.b / scale,
                self.grad_a / scale,
                scale * self.grad_b,
                exact=True,
            )
            torch.testing.assert_close(
                actual_a, scale * reference_a, rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                actual_b, reference_b / scale, rtol=1e-6, atol=1e-6
            )
            actual_product = (
                actual_a @ (self.b / scale) + (scale * self.a) @ actual_b
            )
            reference_product = reference_a @ self.b + self.a @ reference_b
            torch.testing.assert_close(
                actual_product, reference_product, rtol=1e-6, atol=1e-6
            )

    def test_general_gauge_covariance(self) -> None:
        reference_a, reference_b = lora_muon_factor_directions(
            self.a,
            self.b,
            self.grad_a,
            self.grad_b,
            exact=True,
        )
        gauge = torch.randn(4, 4, dtype=torch.float64) + 3.0 * torch.eye(
            4, dtype=torch.float64
        )
        inverse_gauge = torch.linalg.inv(gauge)
        actual_a, actual_b = lora_muon_factor_directions(
            self.a @ gauge,
            inverse_gauge @ self.b,
            self.grad_a @ inverse_gauge.mT,
            gauge.mT @ self.grad_b,
            exact=True,
        )
        torch.testing.assert_close(
            actual_a, reference_a @ gauge, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            actual_b, inverse_gauge @ reference_b, rtol=1e-6, atol=1e-6
        )

    def test_newton_schulz_inverse_root_is_finite_and_accurate(self) -> None:
        factor = torch.randn(16, 8)
        gram = factor.mT @ factor
        inverse_root = matrix_inverse_sqrt_newton_schulz(gram)
        identity = inverse_root @ gram @ inverse_root
        self.assertTrue(torch.isfinite(inverse_root).all())
        torch.testing.assert_close(identity, torch.eye(8), rtol=5e-2, atol=5e-2)

    def test_split_weight_decay_decays_product_once(self) -> None:
        factor_a = self.a.clone()
        factor_b = self.b.clone()
        product = factor_a @ factor_b
        lr = 0.005
        weight_decay = 0.01
        apply_lora_muon_step(
            factor_a,
            factor_b,
            torch.zeros_like(factor_a),
            torch.zeros_like(factor_b),
            lr=lr,
            weight_decay=weight_decay,
        )
        torch.testing.assert_close(
            factor_a @ factor_b,
            (1.0 - lr * weight_decay) * product,
            rtol=1e-12,
            atol=1e-12,
        )


class LoRAMuonOptimizerTests(unittest.TestCase):
    def test_pair_group_can_coexist_with_dense_and_adam_groups(self) -> None:
        factor_a = torch.nn.Parameter(torch.randn(12, 4))
        factor_b = torch.nn.Parameter(torch.randn(4, 10))
        dense = torch.nn.Parameter(torch.randn(8, 8))
        bias = torch.nn.Parameter(torch.randn(8))
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [dense],
                    "use_muon": True,
                    "use_lora_muon": False,
                    "lr": 0.005,
                },
                {
                    "params": [bias],
                    "use_muon": False,
                    "use_lora_muon": False,
                    "lr": 0.005,
                },
                {
                    "params": [factor_a, factor_b],
                    "use_muon": False,
                    "use_lora_muon": True,
                    "lr": 0.005,
                    "pair_param_names": ("ffn.A", "ffn.B"),
                },
            ]
        )
        for parameter in (factor_a, factor_b, dense, bias):
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in (factor_a, factor_b)))

    def test_one_step_updates_both_factors_once_and_stays_finite(self) -> None:
        torch.manual_seed(23)
        factor_a = torch.nn.Parameter(torch.randn(12, 4))
        factor_b = torch.nn.Parameter(torch.randn(4, 10))
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [factor_a, factor_b],
                    "use_muon": False,
                    "use_lora_muon": True,
                    "lr": 0.005,
                    "momentum": 0.95,
                    "weight_decay": 0.01,
                }
            ]
        )
        before_a = factor_a.detach().clone()
        before_b = factor_b.detach().clone()
        factor_a.grad = torch.randn_like(factor_a)
        factor_b.grad = torch.randn_like(factor_b)
        updates = []
        optimizer.update_observer = (
            lambda parameter, update, group, kind: updates.append((parameter, kind))
        )
        optimizer.step()

        self.assertFalse(torch.equal(factor_a, before_a))
        self.assertFalse(torch.equal(factor_b, before_b))
        self.assertTrue(torch.isfinite(factor_a).all())
        self.assertTrue(torch.isfinite(factor_b).all())
        self.assertEqual([kind for _, kind in updates], ["lora_muon", "lora_muon"])
        self.assertEqual(len(optimizer.state[factor_a]), 1)
        self.assertEqual(len(optimizer.state[factor_b]), 1)

    def test_default_factor_muon_path_matches_reference_update(self) -> None:
        torch.manual_seed(29)
        parameter = torch.nn.Parameter(torch.randn(8, 5))
        before = parameter.detach().clone()
        gradient = torch.randn_like(parameter)
        lr = 0.02
        weight_decay = 0.01
        momentum = torch.zeros_like(parameter)
        update = muon_update(
            gradient.clone(),
            momentum,
            beta=0.95,
            adjust_lr_fn="match_rms_adamw",
        )
        expected = before.clone()
        expected.mul_(1.0 - lr * weight_decay)
        expected.add_(update, alpha=-lr)
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [parameter],
                    "use_muon": True,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "adjust_lr_fn": "match_rms_adamw",
                }
            ]
        )
        parameter.grad = gradient
        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=0.0, atol=0.0)

    def test_checkpoint_round_trip_preserves_pair_momenta(self) -> None:
        factor_a = torch.nn.Parameter(torch.randn(7, 3))
        factor_b = torch.nn.Parameter(torch.randn(3, 6))
        group = {
            "params": [factor_a, factor_b],
            "use_muon": False,
            "use_lora_muon": True,
            "lr": 0.005,
            "weight_decay": 0.01,
        }
        optimizer = SingleDeviceMuonWithAuxAdam([group])
        factor_a.grad = torch.randn_like(factor_a)
        factor_b.grad = torch.randn_like(factor_b)
        optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())

        restored_a = torch.nn.Parameter(factor_a.detach().clone())
        restored_b = torch.nn.Parameter(factor_b.detach().clone())
        restored = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [restored_a, restored_b],
                    "use_muon": False,
                    "use_lora_muon": True,
                    "lr": 0.005,
                    "weight_decay": 0.01,
                }
            ]
        )
        restored.load_state_dict(state)
        torch.testing.assert_close(
            restored.state[restored_a]["momentum_buffer"],
            optimizer.state[factor_a]["momentum_buffer"],
        )
        torch.testing.assert_close(
            restored.state[restored_b]["momentum_buffer"],
            optimizer.state[factor_b]["momentum_buffer"],
        )


if __name__ == "__main__":
    unittest.main()
