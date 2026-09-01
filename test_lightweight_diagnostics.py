import gc
import json
import math
from pathlib import Path
import tempfile
import unittest
import weakref

import torch

from lightweight_diagnostics import (
    ActivationScalarCollector,
    LightweightDiagnostics,
    factor_gram_terms,
    product_metrics_from_gram_terms,
)
from low_rank_linear import replace_linear_with_lowrank
from muon_local import SingleDeviceMuonWithAuxAdam
from titan_gpt import TitanGPT, TitanModelArgs


class TestExactProductGrams(unittest.TestCase):
    def test_gram_product_matches_dense_update_with_both_decay_paths(self) -> None:
        generator = torch.Generator().manual_seed(31)
        a_old = torch.randn(9, 4, generator=generator, dtype=torch.float64)
        b_old = torch.randn(4, 7, generator=generator, dtype=torch.float64)
        direction_a = torch.randn(a_old.shape, generator=generator, dtype=torch.float64)
        direction_b = torch.randn(b_old.shape, generator=generator, dtype=torch.float64)
        learning_rate = 0.07
        external_scale = 0.997
        internal_scale = 0.993
        current_a = external_scale * a_old
        current_b = external_scale * b_old

        def raw_terms(parameter, direction, factor):
            if factor == "A":
                return (
                    parameter.mT @ parameter,
                    parameter.mT @ direction,
                    direction.mT @ direction,
                )
            return (
                parameter @ parameter.mT,
                parameter @ direction.mT,
                direction @ direction.mT,
            )

        for optimizer_only in (True, False):
            a_terms = factor_gram_terms(
                *raw_terms(current_a, direction_a, "A"),
                state_scale=external_scale,
                internal_decay_multiplier=internal_scale,
                learning_rate=learning_rate,
                optimizer_only=optimizer_only,
            )
            b_terms = factor_gram_terms(
                *raw_terms(current_b, direction_b, "B"),
                state_scale=external_scale,
                internal_decay_multiplier=internal_scale,
                learning_rate=learning_rate,
                optimizer_only=optimizer_only,
            )
            actual = product_metrics_from_gram_terms(
                a_terms,
                b_terms,
                output_features=a_old.shape[0],
                input_features=b_old.shape[1],
            )
            if optimizer_only:
                a_new = a_old - learning_rate * direction_a
                b_new = b_old - learning_rate * direction_b
            else:
                a_new = internal_scale * current_a - learning_rate * direction_a
                b_new = internal_scale * current_b - learning_rate * direction_b
            dense_delta = a_new @ b_new - a_old @ b_old
            expected_frobenius = float(torch.linalg.vector_norm(dense_delta))
            expected_relative = expected_frobenius / float(
                torch.linalg.vector_norm(a_old @ b_old)
            )
            self.assertAlmostEqual(
                actual["update_frobenius_norm"], expected_frobenius, places=10
            )
            self.assertAlmostEqual(
                actual["relative_update"], expected_relative, places=10
            )


class TestActivationRetention(unittest.TestCase):
    @staticmethod
    def _model() -> TitanGPT:
        return TitanGPT(
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

    def test_hooks_retain_only_scalars(self) -> None:
        model = self._model()
        collector = ActivationScalarCollector(model, enabled=True)
        tokens = torch.randint(0, 64, (3, 8))
        collector.begin_step()
        with torch.no_grad():
            logits = model(tokens, input_batch=tokens)
            ignored_logits = model(tokens, input_batch=tokens)
        logits_reference = weakref.ref(logits)
        collector.end_forward()

        self.assertTrue(collector._stats)
        for entry in collector._stats.values():
            for key in ("sumsq", "max_abs", "sum"):
                self.assertEqual(entry[key].numel(), 1)
                self.assertIsNone(entry[key].grad_fn)
        del logits
        del ignored_logits
        gc.collect()
        self.assertIsNone(logits_reference())

        snapshot = collector.snapshot()
        self.assertIn("embedding", snapshot)
        self.assertEqual(snapshot["embedding"]["count"], tokens.numel() * 16)
        self.assertIn("layer_0.ffn_g", snapshot)
        self.assertIn("logits", snapshot)
        self.assertEqual(collector._stats, {})
        collector.close()


class TestLightweightTrajectoryParity(unittest.TestCase):
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

    def test_every_step_observer_does_not_change_training(self) -> None:
        torch.manual_seed(32)
        baseline = self._model()
        instrumented = self._model()
        instrumented.load_state_dict(baseline.state_dict())
        baseline_optimizer = self._optimizer(baseline)
        instrumented_optimizer = self._optimizer(instrumented)
        tokens = torch.randint(0, 64, (4, 8))
        labels = torch.roll(tokens, shifts=-1, dims=1)
        criterion = torch.nn.CrossEntropyLoss()

        with tempfile.TemporaryDirectory() as directory:
            diagnostics = LightweightDiagnostics(
                model=instrumented,
                optimizer=instrumented_optimizer,
                output_dir=directory,
                run_name="unit",
                rank=0,
                world_size=1,
                product_interval=1,
                adjust_muon_lr="original",
            )
            baseline_optimizer.zero_grad(set_to_none=False)
            baseline_logits = baseline(tokens, input_batch=tokens)
            baseline_loss = criterion(
                baseline_logits.reshape(-1, 64), labels.reshape(-1)
            )
            baseline_loss.backward()

            diagnostics.begin_training_step(1)
            instrumented_optimizer.zero_grad(set_to_none=False)
            instrumented_logits = instrumented(tokens, input_batch=tokens)
            instrumented_loss = criterion(
                instrumented_logits.reshape(-1, 64), labels.reshape(-1)
            )
            instrumented_loss.backward()
            diagnostics.end_training_forward()
            diagnostics.prepare_optimizer_step(base_lr=0.01, manual_decay={})

            baseline_optimizer.step()
            instrumented_optimizer.step()
            diagnostics.finish_step(
                loss=float(instrumented_loss),
                grad_norm=1.0,
                tokens_seen=tokens.numel(),
                tokens_this_step=tokens.numel(),
            )
            diagnostics.close()

            for baseline_parameter, instrumented_parameter in zip(
                baseline.parameters(), instrumented.parameters()
            ):
                torch.testing.assert_close(
                    baseline_parameter, instrumented_parameter, rtol=0, atol=0
                )

            output = Path(directory)
            step_row = json.loads(
                (output / "step_metrics.jsonl").read_text().splitlines()[0]
            )
            self.assertTrue(step_row["product_grams_recorded"])
            self.assertEqual(step_row["step"], 1)
            self.assertTrue(math.isfinite(step_row["embedding_rms"]))

            matrix_rows = [
                json.loads(line)
                for line in (output / "matrix_metrics.jsonl").read_text().splitlines()
            ]
            product_rows = [
                row for row in matrix_rows if "effective_product_update_rms" in row
            ]
            self.assertEqual(len(product_rows), 6)
            for row in product_rows:
                self.assertGreater(row["effective_product_update_rms_per_lr"], 0)
                self.assertGreaterEqual(row["relative_effective_product_update"], 0)

            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(
                metadata["activation_scope"],
                "rank0_first_real_training_microbatch_each_step",
            )
            self.assertEqual(metadata["muon_adjustment_label"], "keller_original")


if __name__ == "__main__":
    unittest.main()
