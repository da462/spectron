from __future__ import annotations

import unittest

from optimizer_routing import (
    lowrank_decoupled_weight_decay_lr,
    lowrank_weight_decay_for_param,
)


class OptimizerRoutingTest(unittest.TestCase):
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
