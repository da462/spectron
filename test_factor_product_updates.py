import math
import json
from pathlib import Path
import tempfile
import unittest

import torch

from factor_product_updates import (
    approximate_update_top2,
    lowrank_sum_frobenius_squared,
    product_adamrms_directions,
    product_update_metrics,
    rankaware_product_adamrms_directions,
    top_singular_pin_directions,
)
from muon_local import (
    SingleDeviceMuonWithAuxAdam,
    muon_momentum_proposal,
    muon_update,
)
from lightweight_diagnostics import LightweightDiagnostics
from low_rank_linear import replace_linear_with_lowrank
from titan_gpt import TitanGPT, TitanModelArgs


class FactorProductPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)

    def test_lowrank_frobenius_matches_materialized_sum(self) -> None:
        terms = [
            (torch.randn(9, 3), torch.randn(3, 7)),
            (torch.randn(9, 4), torch.randn(4, 7)),
            (torch.randn(9, 2), torch.randn(2, 7)),
        ]
        dense = sum(left @ right for left, right in terms)
        actual = lowrank_sum_frobenius_squared(terms)
        torch.testing.assert_close(actual, dense.square().sum(), rtol=2e-5, atol=2e-5)

    def test_product_adamrms_hits_dense_first_order_target(self) -> None:
        factor_a = torch.randn(11, 4)
        factor_b = torch.randn(4, 13)
        raw_a = torch.randn_like(factor_a)
        raw_b = torch.randn_like(factor_b)
        direction_a, direction_b, metrics = product_adamrms_directions(
            factor_a, factor_b, raw_a, raw_b
        )
        dense = direction_a @ factor_b + factor_a @ direction_b
        dense_rms = dense.norm() / math.sqrt(dense.numel())
        torch.testing.assert_close(dense_rms, torch.tensor(0.2), rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(
            metrics["first_order_direction_rms"], dense_rms, rtol=2e-5, atol=2e-5
        )

    def test_rankaware_product_adamrms_hits_available_rank_target(self) -> None:
        factor_a = torch.randn(11, 2)
        factor_b = torch.randn(2, 13)
        direction_a, direction_b, metrics = (
            rankaware_product_adamrms_directions(
                factor_a,
                factor_b,
                torch.randn_like(factor_a),
                torch.randn_like(factor_b),
            )
        )
        dense = direction_a @ factor_b + factor_a @ direction_b
        expected = 0.2 * math.sqrt(4 / 11)
        actual = dense.norm() / math.sqrt(dense.numel())
        torch.testing.assert_close(
            actual, torch.tensor(expected), rtol=2e-5, atol=2e-5
        )
        self.assertEqual(float(metrics["rankaware_dense_min_dimension"]), 11.0)
        self.assertEqual(float(metrics["rankaware_effective_rank_cap"]), 4.0)
        torch.testing.assert_close(
            metrics["rankaware_product_target_rms"], actual
        )

    def test_rankaware_target_saturates_at_dense_target(self) -> None:
        factor_a = torch.randn(7, 4)
        factor_b = torch.randn(4, 9)
        direction_a, direction_b, metrics = (
            rankaware_product_adamrms_directions(
                factor_a,
                factor_b,
                torch.randn_like(factor_a),
                torch.randn_like(factor_b),
            )
        )
        dense = direction_a @ factor_b + factor_a @ direction_b
        actual = dense.norm() / math.sqrt(dense.numel())
        torch.testing.assert_close(actual, torch.tensor(0.2), rtol=2e-5, atol=2e-5)
        self.assertEqual(float(metrics["rankaware_effective_rank_cap"]), 7.0)

    def test_product_update_metrics_match_dense_update(self) -> None:
        factor_a = torch.randn(8, 3)
        factor_b = torch.randn(3, 10)
        delta_a = 0.01 * torch.randn_like(factor_a)
        delta_b = 0.01 * torch.randn_like(factor_b)
        dense_first = delta_a @ factor_b + factor_a @ delta_b
        dense_quadratic = delta_a @ delta_b
        dense = dense_first + dense_quadratic
        metrics = product_update_metrics(factor_a, factor_b, delta_a, delta_b)
        torch.testing.assert_close(
            metrics["first_order_update_rms"],
            dense_first.norm() / math.sqrt(dense.numel()),
        )
        torch.testing.assert_close(
            metrics["actual_update_rms"], dense.norm() / math.sqrt(dense.numel())
        )
        torch.testing.assert_close(
            metrics["quadratic_to_first_frobenius"],
            dense_quadratic.norm() / dense_first.norm(),
        )

    def test_implicit_top2_matches_dense_svd(self) -> None:
        factor_a = torch.randn(12, 4)
        factor_b = torch.randn(4, 10)
        delta_a = 0.03 * torch.randn_like(factor_a)
        delta_b = 0.03 * torch.randn_like(factor_b)
        dense = (
            delta_a @ factor_b
            + factor_a @ delta_b
            + delta_a @ delta_b
        )
        expected = torch.linalg.svdvals(dense)[:2]
        actual, _, _, _ = approximate_update_top2(
            factor_a,
            factor_b,
            delta_a,
            delta_b,
            power_iterations=20,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    def test_top_singular_pin_changes_only_the_original_head(self) -> None:
        factor_a = torch.randn(5, 5)
        factor_b = torch.randn(5, 7)
        momentum_a = 3.0 * torch.randn_like(factor_a)
        momentum_b = 3.0 * torch.randn_like(factor_b)
        original = momentum_a @ factor_b + factor_a @ momentum_b
        u, singular, vh = torch.linalg.svd(original, full_matrices=False)
        expected = original - (singular[0] - 1.0) * torch.outer(u[:, 0], vh[0])

        direction_a, direction_b, _, metrics = top_singular_pin_directions(
            factor_a,
            factor_b,
            momentum_a,
            momentum_b,
            power_iterations=40,
            collect_spectral_metrics=True,
        )
        multiplier = metrics["product_adamrms_multiplier"]
        mapped = (
            direction_a.float() @ factor_b / multiplier
            + factor_a @ direction_b.float() / multiplier
        )
        torch.testing.assert_close(mapped, expected, rtol=2e-4, atol=2e-4)
        achieved_coefficient = u[:, 0] @ mapped @ vh[0]
        torch.testing.assert_close(
            achieved_coefficient, torch.tensor(1.0), rtol=2e-4, atol=2e-4
        )
        scaled = direction_a @ factor_b + factor_a @ direction_b
        torch.testing.assert_close(
            scaled.norm() / math.sqrt(scaled.numel()),
            torch.tensor(0.2),
            rtol=2e-5,
            atol=2e-5,
        )
        self.assertLess(
            float(metrics["desired_to_mapped_relative_frobenius_error"]),
            2e-4,
        )

    def test_top_singular_pin_maps_the_target_to_the_factor_tangent(self) -> None:
        factor_a = torch.randn(9, 3)
        factor_b = torch.randn(3, 7)
        momentum_a = 2.0 * torch.randn_like(factor_a)
        momentum_b = 2.0 * torch.randn_like(factor_b)
        original = momentum_a @ factor_b + factor_a @ momentum_b
        u, singular, vh = torch.linalg.svd(original, full_matrices=False)
        desired = original - (singular[0] - 1.0) * torch.outer(u[:, 0], vh[0])
        left = torch.linalg.qr(factor_a, mode="reduced").Q
        right = torch.linalg.qr(factor_b.mT, mode="reduced").Q
        expected = (
            left @ (left.mT @ desired)
            + desired @ right @ right.mT
            - left @ (left.mT @ desired @ right) @ right.mT
        )
        actual_a, actual_b, _, metrics = top_singular_pin_directions(
            factor_a,
            factor_b,
            momentum_a,
            momentum_b,
            power_iterations=40,
            collect_spectral_metrics=True,
        )
        multiplier = metrics["product_adamrms_multiplier"]
        mapped = (
            actual_a.float() @ factor_b / multiplier
            + factor_a @ actual_b.float() / multiplier
        )
        torch.testing.assert_close(mapped, expected, rtol=4e-4, atol=4e-4)
        expected_error = (desired - expected).norm() / desired.norm()
        torch.testing.assert_close(
            metrics["desired_to_mapped_relative_frobenius_error"],
            expected_error,
            rtol=5e-4,
            atol=5e-4,
        )


class PairedFactorOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5678)

    def test_product_adamrms_optimizer_keeps_baseline_factor_decay(self) -> None:
        initial_a = torch.randn(7, 3)
        initial_b = torch.randn(3, 9)
        grad_a = torch.randn_like(initial_a)
        grad_b = torch.randn_like(initial_b)
        lr = 0.007
        wd = 0.1

        reference_grad_a = grad_a.clone()
        reference_grad_b = grad_b.clone()
        raw_a = muon_update(
            reference_grad_a, torch.zeros_like(initial_a), adjust_lr_fn="none"
        )
        raw_b = muon_update(
            reference_grad_b, torch.zeros_like(initial_b), adjust_lr_fn="none"
        )
        expected_a_direction, expected_b_direction, _ = product_adamrms_directions(
            initial_a, initial_b, raw_a, raw_b
        )

        factor_a = torch.nn.Parameter(initial_a.clone())
        factor_b = torch.nn.Parameter(initial_b.clone())
        factor_a.grad = grad_a.clone()
        factor_b.grad = grad_b.clone()
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [factor_a, factor_b],
                    "use_muon": False,
                    "factor_update_variant": "product_adamrms",
                    "pair_name": "layers.0.feed_forward.w1",
                    "lr": lr,
                    "momentum": 0.95,
                    "weight_decay": wd,
                }
            ]
        )
        optimizer.step()
        torch.testing.assert_close(
            factor_a,
            initial_a * (1.0 - lr * wd) - lr * expected_a_direction,
        )
        torch.testing.assert_close(
            factor_b,
            initial_b * (1.0 - lr * wd) - lr * expected_b_direction,
        )

    def test_pair_observer_reports_product_target(self) -> None:
        factor_a = torch.nn.Parameter(torch.randn(6, 3))
        factor_b = torch.nn.Parameter(torch.randn(3, 8))
        factor_a.grad = torch.randn_like(factor_a)
        factor_b.grad = torch.randn_like(factor_b)
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [factor_a, factor_b],
                    "use_muon": False,
                    "factor_update_variant": "product_adamrms",
                    "pair_name": "layers.0.feed_forward.w1",
                    "lr": 0.007,
                    "weight_decay": 0.1,
                }
            ]
        )
        rows = []
        optimizer.pair_update_observer = lambda *args: rows.append(args)
        optimizer.pair_metrics_due = True
        optimizer.step()
        self.assertEqual(len(rows), 1)
        metrics = rows[0][1]
        self.assertAlmostEqual(metrics["first_order_update_rms"], 0.0014, places=6)
        self.assertTrue(math.isfinite(metrics["actual_update_rms"]))

    def test_rankaware_pair_observer_reports_rank_scaled_target(self) -> None:
        factor_a = torch.nn.Parameter(torch.randn(8, 2))
        factor_b = torch.nn.Parameter(torch.randn(2, 10))
        factor_a.grad = torch.randn_like(factor_a)
        factor_b.grad = torch.randn_like(factor_b)
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [factor_a, factor_b],
                    "use_muon": False,
                    "factor_update_variant": "rankaware_product_adamrms",
                    "pair_name": "layers.0.feed_forward.w1",
                    "lr": 0.007,
                    "weight_decay": 0.1,
                }
            ]
        )
        rows = []
        optimizer.pair_update_observer = lambda *args: rows.append(args)
        optimizer.pair_metrics_due = True
        optimizer.step()
        metrics = rows[0][1]
        expected_direction_rms = 0.2 * math.sqrt(4 / 8)
        self.assertAlmostEqual(
            metrics["first_order_update_rms"],
            0.007 * expected_direction_rms,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["rankaware_product_target_rms"],
            expected_direction_rms,
            places=6,
        )

    def test_top_singular_pin_optimizer_uses_pre_ns_momentum(self) -> None:
        initial_a = torch.randn(7, 3)
        initial_b = torch.randn(3, 9)
        grad_a = torch.randn_like(initial_a)
        grad_b = torch.randn_like(initial_b)
        lr = 0.007
        wd = 0.1
        beta = 0.95

        buffer_a = torch.zeros_like(initial_a)
        buffer_b = torch.zeros_like(initial_b)
        momentum_a = muon_momentum_proposal(grad_a.clone(), buffer_a, beta=beta)
        momentum_b = muon_momentum_proposal(grad_b.clone(), buffer_b, beta=beta)
        expected_a, expected_b, _, _ = top_singular_pin_directions(
            initial_a,
            initial_b,
            momentum_a,
            momentum_b,
            seed=0,
        )

        factor_a = torch.nn.Parameter(initial_a.clone())
        factor_b = torch.nn.Parameter(initial_b.clone())
        factor_a.grad = grad_a.clone()
        factor_b.grad = grad_b.clone()
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [factor_a, factor_b],
                    "use_muon": False,
                    "factor_update_variant": "top_singular_pin",
                    "factor_pair_index": 0,
                    "pair_name": "layers.0.feed_forward.w1",
                    "lr": lr,
                    "momentum": beta,
                    "weight_decay": wd,
                }
            ]
        )
        optimizer.step()
        torch.testing.assert_close(
            factor_a,
            initial_a * (1.0 - lr * wd) - lr * expected_a,
        )
        torch.testing.assert_close(
            factor_b,
            initial_b * (1.0 - lr * wd) - lr * expected_b,
        )
        torch.testing.assert_close(
            optimizer.state[factor_a]["momentum_buffer"], buffer_a
        )
        torch.testing.assert_close(
            optimizer.state[factor_b]["momentum_buffer"], buffer_b
        )

    def test_top_singular_pin_observer_reports_target_and_mapping(self) -> None:
        factor_a = torch.nn.Parameter(torch.randn(8, 3))
        factor_b = torch.nn.Parameter(torch.randn(3, 10))
        factor_a.grad = torch.randn_like(factor_a)
        factor_b.grad = torch.randn_like(factor_b)
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [factor_a, factor_b],
                    "use_muon": False,
                    "factor_update_variant": "top_singular_pin",
                    "factor_pair_index": 0,
                    "pair_name": "layers.0.feed_forward.w1",
                    "lr": 0.007,
                    "weight_decay": 0.1,
                }
            ]
        )
        rows = []
        optimizer.pair_update_observer = lambda *args: rows.append(args)
        optimizer.pair_metrics_due = True
        optimizer.step()
        self.assertEqual(len(rows), 1)
        metrics = rows[0][1]
        self.assertAlmostEqual(metrics["first_order_update_rms"], 0.0014, places=6)
        self.assertEqual(metrics["intended_original_top_coefficient"], 1.0)
        self.assertTrue(math.isfinite(metrics["mapped_sigma1"]))
        self.assertTrue(
            math.isfinite(metrics["desired_to_mapped_relative_frobenius_error"])
        )

    def test_rankaware_pair_metrics_flow_through_lightweight_diagnostics(self) -> None:
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
        model = replace_linear_with_lowrank(
            model,
            rank_ratio=0.25,
            exclude_modules=["tok_embeddings", "output", "attention"],
            disable_c=True,
        )
        pairs = {}
        dense = []
        auxiliary = []
        for name, parameter in model.named_parameters():
            if name.endswith(".A") or name.endswith(".B"):
                pair_name, factor = name.rsplit(".", 1)
                pairs.setdefault(pair_name, {})[factor] = (name, parameter)
            elif "layers." in name and parameter.ndim >= 2:
                dense.append(parameter)
            else:
                auxiliary.append(parameter)
        groups = [
            {
                "params": dense,
                "use_muon": True,
                "lr": 0.007,
                "weight_decay": 0.1,
                "adjust_lr_fn": "match_rms_adamw",
            },
            {
                "params": auxiliary,
                "use_muon": False,
                "lr": 0.007,
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
            },
        ]
        for pair_index, (pair_name, factors) in enumerate(sorted(pairs.items())):
            groups.append(
                {
                    "params": [factors["A"][1], factors["B"][1]],
                    "use_muon": False,
                    "factor_update_variant": "rankaware_product_adamrms",
                    "factor_pair_index": pair_index,
                    "pair_name": pair_name,
                    "pair_param_names": (factors["A"][0], factors["B"][0]),
                    "lr": 0.007,
                    "weight_decay": 0.1,
                }
            )
        optimizer = SingleDeviceMuonWithAuxAdam(groups)
        tokens = torch.randint(0, 64, (4, 8))
        labels = torch.roll(tokens, shifts=-1, dims=1)
        criterion = torch.nn.CrossEntropyLoss()
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = LightweightDiagnostics(
                model=model,
                optimizer=optimizer,
                output_dir=directory,
                run_name="product-test",
                rank=0,
                world_size=1,
                product_interval=1,
                adjust_muon_lr="match_rms_adamw",
            )
            diagnostics.begin_training_step(1)
            logits = model(tokens, input_batch=tokens)
            loss = criterion(logits.reshape(-1, 64), labels.reshape(-1))
            loss.backward()
            diagnostics.end_training_forward()
            diagnostics.prepare_optimizer_step(base_lr=0.007, manual_decay={})
            optimizer.step()
            diagnostics.finish_step(
                loss=float(loss),
                grad_norm=1.0,
                tokens_seen=tokens.numel(),
                tokens_this_step=tokens.numel(),
            )
            diagnostics.close()
            rows = [
                json.loads(line)
                for line in (
                    Path(directory) / "pair_optimizer_metrics.jsonl"
                ).read_text().splitlines()
            ]
        self.assertEqual(len(rows), 6)
        self.assertTrue(
            all(
                row["factor_update_variant"] == "rankaware_product_adamrms"
                for row in rows
            )
        )
        self.assertTrue(all(row["first_order_target_relative_error"] < 1e-4 for row in rows))
        self.assertTrue(
            all(
                row["rankaware_product_target_rms"]
                == row["target_first_order_direction_rms"]
                for row in rows
            )
        )


class ProductVariantLauncherTests(unittest.TestCase):
    def test_launcher_pins_the_confirmed_baseline_and_only_two_variants(self) -> None:
        launcher = (
            Path(__file__).parent / "bin" / "submit_jz_factor_product_variants.sh"
        ).read_text()
        for expected in (
            "MAX_LR=7e-3",
            "WEIGHT_DECAY=0.1",
            "NH_WEIGHT_DECAY=0.1",
            "ADJUST_MUON_LR=match_rms_adamw",
            "EMBEDDING_INIT_STD=0.02",
            "GLOBAL_BATCH_SIZE=512",
            "MICRO_BATCH_SIZE=16",
            "SEQUENCE_LENGTH=2048",
            "TOTAL_STEPS=2234",
            "LR_SCHEDULE_STEPS=2234",
            "MECHANISTIC_DIAGNOSTICS=1",
            "LIGHTWEIGHT_DIAGNOSTICS=1",
        ):
            self.assertIn(expected, launcher)
        self.assertEqual(launcher.count("submit_variant "), 2)
        self.assertIn("submit_variant rankaware_product_adamrms", launcher)
        self.assertIn("submit_variant top_singular_pin", launcher)

    def test_rankaware_launcher_submits_only_the_replacement_job(self) -> None:
        launcher = (
            Path(__file__).parent
            / "bin"
            / "submit_jz_rankaware_product_adamrms.sh"
        ).read_text()
        self.assertIn("MODEL_TAG=factor_muon_rankaware_product_adamrms", launcher)
        self.assertIn("LOWRANK_OPTIMIZER=rankaware_product_adamrms", launcher)
        self.assertNotIn("LOWRANK_OPTIMIZER=top_singular_pin", launcher)

    def test_top_singular_pin_launcher_pins_the_baseline_protocol(self) -> None:
        launcher = (
            Path(__file__).parent / "bin" / "submit_jz_top_singular_pin.sh"
        ).read_text()
        self.assertIn("LOWRANK_OPTIMIZER=top_singular_pin", launcher)
        self.assertIn("MAX_LR=7e-3", launcher)
        self.assertIn("WEIGHT_DECAY=0.1", launcher)
        self.assertIn("ADJUST_MUON_LR=match_rms_adamw", launcher)
        self.assertIn("EMBEDDING_INIT_STD=0.02", launcher)
        self.assertIn('MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"', launcher)


if __name__ == "__main__":
    unittest.main()
