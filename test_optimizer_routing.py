from __future__ import annotations

import unittest

import torch

from optimizer_routing import (
    collect_lowrank_lr_records,
    lowrank_decoupled_weight_decay_lr,
    lowrank_factor_metadata,
    lowrank_weight_decay_for_param,
)


class OptimizerRoutingTest(unittest.TestCase):
    def test_lowrank_factor_metadata_parses_ffn_and_attention(self) -> None:
        self.assertEqual(
            lowrank_factor_metadata("layers.11.feed_forward.w2.B"),
            {
                "layer": 11,
                "module_type": "ffn",
                "matrix": "w2",
                "factor": "B",
            },
        )
        self.assertEqual(
            lowrank_factor_metadata("layers.3.attention.wq.A"),
            {
                "layer": 3,
                "module_type": "attention",
                "matrix": "wq",
                "factor": "A",
            },
        )
        self.assertIsNone(lowrank_factor_metadata("tok_embeddings.weight"))

    def test_collect_lowrank_lr_records_reads_spectron_factor_group_lrs(self) -> None:
        w1_a = torch.nn.Parameter(torch.zeros(8, 2))
        w1_b = torch.nn.Parameter(torch.zeros(2, 8))
        attn_a = torch.nn.Parameter(torch.zeros(8, 2))
        records = collect_lowrank_lr_records(
            [
                ("layers.0.feed_forward.w1.A", w1_a),
                ("layers.0.feed_forward.w1.B", w1_b),
                ("layers.0.attention.wq.A", attn_a),
            ],
            [
                {
                    "params": [w1_a],
                    "lr": 0.20,
                    "is_lowrank": True,
                    "param_name": "layers.0.feed_forward.w1.A",
                    "lr_multiplier": 20.0,
                    "lowrank_weight_decay": 0.1,
                    "use_muon": True,
                },
                {
                    "params": [w1_b],
                    "lr": 0.40,
                    "is_lowrank": True,
                    "param_name": "layers.0.feed_forward.w1.B",
                    "lr_multiplier": 20.0,
                    "lowrank_weight_decay": 0.1,
                    "use_muon": True,
                },
                {
                    "params": [attn_a],
                    "lr": 0.05,
                    "is_lowrank": True,
                    "param_name": "layers.0.attention.wq.A",
                    "lr_multiplier": 1.0,
                    "lowrank_weight_decay": 0.1,
                    "use_muon": True,
                },
            ],
            base_lr=0.05,
            lowrank_ffn_lr_multiplier=20.0,
            spectral_scaling={
                "layers.0.feed_forward.w1.A": 0.2,
                "layers.0.feed_forward.w1.B": 0.4,
            },
            spectral_lr_target="ffn",
            apply_spectral_lr_scaling=True,
            module_type_filter="ffn",
        )

        self.assertEqual([record["factor"] for record in records], ["A", "B"])
        self.assertEqual([record["actual_lr"] for record in records], [0.20, 0.40])
        self.assertEqual([record["expected_lr"] for record in records], [0.20, 0.40])
        self.assertTrue(all(record["is_individual_lowrank_group"] for record in records))
        self.assertTrue(all(record["spectral_targeted"] for record in records))

    def test_collect_lowrank_lr_records_handles_plain_shared_muon_group(self) -> None:
        w1_a = torch.nn.Parameter(torch.zeros(8, 2))
        w1_b = torch.nn.Parameter(torch.zeros(2, 8))
        records = collect_lowrank_lr_records(
            [
                ("layers.0.feed_forward.w1.A", w1_a),
                ("layers.0.feed_forward.w1.B", w1_b),
            ],
            [
                {
                    "params": [w1_a, w1_b],
                    "lr": 0.05,
                    "weight_decay": 0.01,
                    "use_muon": True,
                    "is_lowrank": False,
                },
            ],
            base_lr=0.05,
            lowrank_ffn_lr_multiplier=1.0,
            spectral_scaling=None,
            spectral_lr_target="none",
            apply_spectral_lr_scaling=False,
            module_type_filter="ffn",
        )

        self.assertEqual(len(records), 2)
        self.assertEqual([record["actual_lr"] for record in records], [0.05, 0.05])
        self.assertEqual([record["expected_lr"] for record in records], [0.05, 0.05])
        self.assertTrue(all(record["use_muon"] for record in records))
        self.assertFalse(any(record["is_individual_lowrank_group"] for record in records))

    def test_lowrank_weight_decay_uses_owner_specific_overrides(self) -> None:
        self.assertEqual(
            lowrank_weight_decay_for_param(
                "layers.0.feed_forward.w1.A",
                default_weight_decay=0.01,
                ffn_weight_decay=0.1,
                attention_weight_decay=0.2,
            ),
            0.1,
        )
        self.assertEqual(
            lowrank_weight_decay_for_param(
                "layers.0.attention.wq.B",
                default_weight_decay=0.01,
                ffn_weight_decay=0.1,
                attention_weight_decay=0.2,
            ),
            0.2,
        )
        self.assertEqual(
            lowrank_weight_decay_for_param(
                "layers.0.feed_forward.w2.B",
                default_weight_decay=0.01,
                ffn_weight_decay=None,
                attention_weight_decay=0.2,
            ),
            0.01,
        )

    def test_spectron_decay_lr_keeps_targeted_factors_on_base_lr(self) -> None:
        ffn_group = {
            "param_name": "layers.0.feed_forward.w3.A",
            "lr_multiplier": 20.0,
        }
        attn_group = {
            "param_name": "layers.0.attention.wv.B",
            "lr_multiplier": 1.0,
        }

        self.assertEqual(
            lowrank_decoupled_weight_decay_lr(
                ffn_group,
                base_lr=0.05,
                spectral_lr_target="ffn",
            ),
            0.05,
        )
        self.assertEqual(
            lowrank_decoupled_weight_decay_lr(
                ffn_group,
                base_lr=0.05,
                spectral_lr_target="attention",
            ),
            1.0,
        )
        self.assertEqual(
            lowrank_decoupled_weight_decay_lr(
                attn_group,
                base_lr=0.05,
                spectral_lr_target="ffn",
            ),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
