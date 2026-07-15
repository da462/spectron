from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from low_rank_linear import LowRankLinear, PerHeadLowRankLinear, replace_linear_with_lowrank
from muon_local import SingleDeviceMuonWithAuxAdam
from power_iter import get_lowrank_spectral_norm_scaling
from titan_gpt import TitanGPT, TitanModelArgs


def module_by_name(model: nn.Module, name: str) -> nn.Module:
    return dict(model.named_modules())[name]


class PerHeadLowRankLinearTests(unittest.TestCase):
    def test_row_block_forward_matches_effective_weight(self) -> None:
        torch.manual_seed(0)
        layer = PerHeadLowRankLinear(
            in_features=8,
            out_features=12,
            rank=3,
            num_heads=3,
            head_dim=4,
            block_axis="row",
            bias=True,
        )
        with torch.no_grad():
            layer.A.copy_(torch.randn_like(layer.A))
            layer.B.copy_(torch.randn_like(layer.B))
            layer.bias.copy_(torch.randn_like(layer.bias))

        x = torch.randn(2, 5, 8)
        weight = layer.get_full_weight()
        self.assertEqual(tuple(weight.shape), (12, 8))
        torch.testing.assert_close(layer(x), F.linear(x, weight, layer.bias))

    def test_column_block_forward_matches_effective_weight(self) -> None:
        torch.manual_seed(1)
        layer = PerHeadLowRankLinear(
            in_features=12,
            out_features=8,
            rank=3,
            num_heads=3,
            head_dim=4,
            block_axis="column",
            bias=False,
        )
        with torch.no_grad():
            layer.A.copy_(torch.randn_like(layer.A))
            layer.B.copy_(torch.randn_like(layer.B))

        x = torch.randn(2, 5, 12)
        weight = layer.get_full_weight()
        self.assertEqual(tuple(weight.shape), (8, 12))
        torch.testing.assert_close(layer(x), F.linear(x, weight, None))

    def test_full_rank_svd_reconstructs_per_head_blocks(self) -> None:
        torch.manual_seed(2)
        row_linear = nn.Linear(8, 12, bias=False)
        row = PerHeadLowRankLinear.from_linear(
            row_linear,
            rank=4,
            num_heads=3,
            head_dim=4,
            block_axis="row",
            method="svd",
        )
        torch.testing.assert_close(row.weight, row_linear.weight, rtol=1e-5, atol=1e-6)

        col_linear = nn.Linear(12, 8, bias=False)
        col = PerHeadLowRankLinear.from_linear(
            col_linear,
            rank=4,
            num_heads=3,
            head_dim=4,
            block_axis="column",
            method="svd",
        )
        torch.testing.assert_close(col.weight, col_linear.weight, rtol=1e-5, atol=1e-6)

    def test_replace_per_head_attention_keeps_ffn_on_existing_lowrank_class(self) -> None:
        model = TitanGPT(
            TitanModelArgs(
                vocab_size=128,
                n_layers=2,
                n_heads=4,
                n_kv_heads=4,
                dim=32,
                max_seq_len=16,
                multiple_of=16,
                use_flex_attn=False,
            )
        )
        replace_linear_with_lowrank(
            model,
            rank_ratio=0.25,
            method="random",
            exclude_modules=["tok_embeddings", "output"],
            disable_c=True,
            attention_factorization="per_head",
        )

        for layer_idx in range(2):
            prefix = f"layers.{layer_idx}"
            for name in ("wq", "wk", "wv"):
                module = module_by_name(model, f"{prefix}.attention.{name}")
                self.assertIsInstance(module, PerHeadLowRankLinear)
                self.assertEqual(module.block_axis, "row")
                self.assertEqual(module.rank, 2)
                self.assertEqual(tuple(module.A.shape), (4, 8, 2))
                self.assertEqual(tuple(module.B.shape), (4, 2, 32))
            wo = module_by_name(model, f"{prefix}.attention.wo")
            self.assertIsInstance(wo, PerHeadLowRankLinear)
            self.assertEqual(wo.block_axis, "column")
            self.assertEqual(tuple(wo.A.shape), (4, 32, 2))
            self.assertEqual(tuple(wo.B.shape), (4, 2, 8))

            for name in ("w1", "w2", "w3"):
                module = module_by_name(model, f"{prefix}.feed_forward.{name}")
                self.assertIsInstance(module, LowRankLinear)
                self.assertTrue(module.disable_c)

        self.assertIsInstance(model.output, nn.Linear)
        self.assertIsInstance(model.tok_embeddings, nn.Embedding)

    def test_ttmatched_per_head_rank16_parameter_count(self) -> None:
        model = TitanGPT(
            TitanModelArgs(
                vocab_size=32000,
                n_layers=12,
                n_heads=12,
                n_kv_heads=12,
                dim=768,
                max_seq_len=2048,
                multiple_of=256,
                use_flex_attn=False,
            )
        )
        replace_linear_with_lowrank(
            model,
            rank_ratio=0.25,
            method="random",
            exclude_modules=["tok_embeddings", "output"],
            disable_c=True,
            attention_factorization="per_head",
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 87_116_544)
        self.assertEqual(module_by_name(model, "layers.0.attention.wq").rank, 16)
        self.assertEqual(module_by_name(model, "layers.0.feed_forward.w1").rank, 192)
        self.assertEqual(module_by_name(model, "layers.0.feed_forward.w2").rank, 512)

    def test_spectron_scaling_produces_finite_keys_for_per_head_factors(self) -> None:
        torch.manual_seed(3)
        model = nn.Module()
        model.attention = nn.Module()
        model.attention.wq = PerHeadLowRankLinear(
            in_features=8,
            out_features=12,
            rank=3,
            num_heads=3,
            head_dim=4,
            block_axis="row",
            bias=False,
        )

        scaling, regularization = get_lowrank_spectral_norm_scaling(
            model,
            power_iter_steps=2,
            spectral_lr_scaling_offset=1.0,
        )
        self.assertFalse(regularization)
        self.assertEqual(
            set(scaling),
            {"attention.wq.A", "attention.wq.B"},
        )
        self.assertTrue(math.isfinite(scaling["attention.wq.A"]))
        self.assertGreater(scaling["attention.wq.A"], 0.0)
        self.assertEqual(scaling["attention.wq.A"], scaling["attention.wq.B"])

        _, regularization = get_lowrank_spectral_norm_scaling(
            model,
            power_iter_steps=2,
            spectral_weight_decay=1e-7,
            spectral_lr_scaling_offset=1.0,
        )
        self.assertEqual(tuple(regularization["attention.wq.A"].shape), tuple(model.attention.wq.A.shape))
        self.assertEqual(tuple(regularization["attention.wq.B"].shape), tuple(model.attention.wq.B.shape))
        self.assertTrue(torch.isfinite(regularization["attention.wq.A"]).all())
        self.assertTrue(torch.isfinite(regularization["attention.wq.B"]).all())

    def test_single_device_muon_updates_3d_per_head_factor_tensor(self) -> None:
        param = torch.nn.Parameter(torch.zeros(3, 4, 2))
        param.grad = torch.randn_like(param)
        opt = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [param],
                    "use_muon": True,
                    "lr": 0.05,
                    "weight_decay": 0.0,
                    "momentum": 0.0,
                    "adjust_lr_fn": "none",
                }
            ]
        )
        before = param.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(before, param.detach()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
