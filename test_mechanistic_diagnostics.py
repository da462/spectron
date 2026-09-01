import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from low_rank_linear import replace_linear_with_lowrank
from mechanistic_diagnostics import (
    MechanisticDiagnostics,
    diagnostic_step,
    full_matrix_gradient,
    lowrank_subspaces,
    normalize_device,
    product_update_decomposition,
    singular_metrics,
    tangent_motion_fractions,
    tangent_projection,
)
from muon_local import SingleDeviceMuonWithAuxAdam
from titan_gpt import TitanGPT, TitanModelArgs


def _assert_finite_tree(testcase: unittest.TestCase, value) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_finite_tree(testcase, nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_finite_tree(testcase, nested)
    elif isinstance(value, float):
        testcase.assertTrue(math.isfinite(value), value)


class TestMechanisticNumerics(unittest.TestCase):
    def test_integer_local_rank_normalizes_to_cuda_device(self) -> None:
        self.assertEqual(normalize_device(0), torch.device("cuda", 0))
        self.assertEqual(normalize_device("cpu"), torch.device("cpu"))

    def test_product_update_decomposition_is_exact(self) -> None:
        generator = torch.Generator().manual_seed(7)
        a = torch.randn(11, 4, generator=generator)
        b = torch.randn(4, 9, generator=generator)
        d_a = 0.01 * torch.randn(11, 4, generator=generator)
        d_b = 0.01 * torch.randn(4, 9, generator=generator)
        result = product_update_decomposition(a, b, a + d_a, b + d_b)
        torch.testing.assert_close(
            result["exact"], result["first"] + result["second"], rtol=1e-5, atol=1e-6
        )
        self.assertLess(result["reconstruction_error"].norm().item(), 1e-5)

    def test_normalized_spectrum_is_scale_invariant(self) -> None:
        matrix = torch.randn(13, 9, generator=torch.Generator().manual_seed(8))
        first = singular_metrics(matrix, 9)
        second = singular_metrics(14 * matrix, 9)
        torch.testing.assert_close(
            torch.tensor(first["normalized_singular_values_top32"]),
            torch.tensor(second["normalized_singular_values_top32"]),
        )
        self.assertAlmostEqual(first["stable_rank"], second["stable_rank"], places=5)

    def test_shadow_full_matrix_gradient_matches_autograd(self) -> None:
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(2, 5, 7, generator=generator)
        weight = torch.randn(6, 7, generator=generator, requires_grad=True)
        output = x @ weight.mT
        coefficients = torch.randn(output.shape, generator=generator)
        loss = (output * coefficients).sum()
        expected = torch.autograd.grad(loss, weight)[0]
        actual = full_matrix_gradient(x, coefficients)
        torch.testing.assert_close(actual, expected)

    def test_tangent_projection_and_motion_decomposition(self) -> None:
        generator = torch.Generator().manual_seed(10)
        a = torch.randn(12, 4, generator=generator)
        b = torch.randn(4, 9, generator=generator)
        u, v, _ = lowrank_subspaces(a, b)
        z = torch.randn(12, 9, generator=generator)
        projected = tangent_projection(z, u, v)
        normal = z - projected
        torch.testing.assert_close(u.mT @ normal @ v, torch.zeros_like(u.mT @ normal @ v), atol=2e-5, rtol=2e-5)
        fractions = tangent_motion_fractions(z, u, v)
        self.assertAlmostEqual(sum(fractions.values()), 1.0, places=5)

    def test_requested_diagnostic_schedule(self) -> None:
        selected = [step for step in range(1, 251) if diagnostic_step(step, 250)]
        self.assertEqual(selected, [1, 10, 50, 100, 200, 250])


class TestMechanisticTrajectoryParity(unittest.TestCase):
    @staticmethod
    def _model() -> TitanGPT:
        model = TitanGPT(
            TitanModelArgs(
                vocab_size=64,
                n_layers=2,
                n_heads=4,
                n_kv_heads=4,
                dim=16,
                max_seq_len=8,
                multiple_of=8,
            )
        )
        return replace_linear_with_lowrank(
            model,
            rank_ratio=0.25,
            exclude_modules=["tok_embeddings", "output", "attention"],
            disable_c=True,
        )

    @staticmethod
    def _optimizer(model: TitanGPT) -> SingleDeviceMuonWithAuxAdam:
        matrices = []
        auxiliary = []
        for name, parameter in model.named_parameters():
            if "layers." in name and parameter.ndim >= 2:
                matrices.append(parameter)
            else:
                auxiliary.append(parameter)
        return SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": matrices,
                    "lr": 0.01,
                    "momentum": 0.95,
                    "weight_decay": 0.01,
                    "adjust_lr_fn": "original",
                    "use_muon": True,
                },
                {
                    "params": auxiliary,
                    "lr": 0.01,
                    "betas": (0.9, 0.95),
                    "eps": 1e-10,
                    "weight_decay": 0.01,
                    "use_muon": False,
                },
            ]
        )

    @staticmethod
    def _train_backward(model, optimizer, tokens, labels, criterion) -> None:
        optimizer.zero_grad(set_to_none=False)
        logits = model(tokens, input_batch=tokens)
        criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).backward()

    def test_diagnostics_do_not_change_one_step_trajectory(self) -> None:
        torch.manual_seed(11)
        baseline = self._model()
        instrumented = self._model()
        instrumented.load_state_dict(baseline.state_dict())
        baseline_optimizer = self._optimizer(baseline)
        instrumented_optimizer = self._optimizer(instrumented)
        tokens = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(12))
        labels = torch.roll(tokens.to(torch.long), shifts=-1, dims=1)
        criterion = torch.nn.CrossEntropyLoss()

        with tempfile.TemporaryDirectory() as directory:
            batch_path = Path(directory) / "batch.pt"
            torch.save(
                {"input_ids": tokens.to(torch.int32), "labels": labels, "sha256": "test"},
                batch_path,
            )
            diagnostics = MechanisticDiagnostics(
                model=instrumented,
                optimizer=instrumented_optimizer,
                criterion=criterion,
                diagnostic_batch_path=str(batch_path),
                output_dir=str(Path(directory) / "metrics"),
                run_name="test",
                total_steps=1,
                device=torch.device("cpu"),
                bf16=False,
                adjust_muon_lr="original",
                spectral_lr_scaling=False,
                spectral_lr_target="ffn",
                weight_decay=0.01,
                embedding_init_std=1.0,
            )

            self._train_backward(
                baseline, baseline_optimizer, tokens, labels, criterion
            )
            self._train_backward(
                instrumented, instrumented_optimizer, tokens, labels, criterion
            )
            grads_before = {
                name: parameter.grad.clone()
                for name, parameter in instrumented.named_parameters()
            }
            rng_before = torch.get_rng_state().clone()
            diagnostics.before_optimizer_step(
                update_number=1, base_lr=0.01, spectral_scaling={}
            )
            torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
            for name, parameter in instrumented.named_parameters():
                torch.testing.assert_close(
                    parameter.grad, grads_before[name], rtol=0, atol=0
                )

            baseline_optimizer.step()
            instrumented_optimizer.step()
            diagnostics.after_optimizer_step()
            for baseline_parameter, instrumented_parameter in zip(
                baseline.parameters(), instrumented.parameters()
            ):
                torch.testing.assert_close(
                    baseline_parameter, instrumented_parameter, rtol=0, atol=0
                )

            metric_dir = Path(directory) / "metrics"
            expected_counts = {
                "function_metrics.jsonl": 2,
                "matrix_metrics.jsonl": 6,
                "model_metrics.jsonl": 1,
                "step_metrics.jsonl": 1,
            }
            for filename, expected_count in expected_counts.items():
                rows = [
                    json.loads(line)
                    for line in (metric_dir / filename).read_text().splitlines()
                ]
                self.assertEqual(len(rows), expected_count)
                for row in rows:
                    _assert_finite_tree(self, row)

            function_row = json.loads(
                (metric_dir / "function_metrics.jsonl").read_text().splitlines()[0]
            )
            for key in (
                "attention_residual_rms",
                "attention_branch_rms",
                "attention_branch_to_residual",
                "attention_local_update_q",
                "attention_normalized_state_displacement_z",
                "ffn_to_attention_q_ratio",
            ):
                self.assertIn(key, function_row)

            model_row = json.loads(
                (metric_dir / "model_metrics.jsonl").read_text().splitlines()[0]
            )
            for key in (
                "embedding_output_rms_pre",
                "embedding",
                "final_residual_rms",
                "final_normalized_rms",
                "lm_head",
                "logits_rms_pre",
                "logits_std_pre",
                "logits_max_abs_pre",
                "prediction_entropy_pre",
                "diagnostic_ce_pre",
                "logit_update_rms",
                "output_kl_pre_to_post",
            ):
                self.assertIn(key, model_row)
            self.assertIn("relative_update_rms", model_row["embedding"])
            self.assertIn("relative_update_rms", model_row["lm_head"])


if __name__ == "__main__":
    unittest.main()
