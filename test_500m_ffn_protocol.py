import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from low_rank_linear import LowRankLinear, replace_linear_with_lowrank
from model_analysis import calculate_transformer_training_flops
from muon_local import _adjust_lr
from optimizer_routing import (
    apply_lowrank_lr_overrides,
    lowrank_decoupled_weight_decay_lr,
)
from titan_gpt import TitanGPT, TitanModelArgs


class Test500MFFNProtocol(unittest.TestCase):
    def test_paper_shape_and_ffn_only_geometry(self) -> None:
        args = TitanModelArgs(
            vocab_size=32_000,
            n_layers=20,
            n_heads=20,
            n_kv_heads=20,
            dim=1_280,
            max_seq_len=2_048,
            multiple_of=256,
            rope_theta=10_000,
        )

        with torch.device("meta"):
            dense = TitanGPT(args)
        self.assertEqual(sum(p.numel() for p in dense.parameters()), 488_295_680)

        with torch.device("meta"):
            lowrank = TitanGPT(args)
            lowrank = replace_linear_with_lowrank(
                lowrank,
                rank_ratio=0.25,
                method="random",
                exclude_modules=["tok_embeddings", "output", "attention"],
                disable_c=True,
            )

        self.assertEqual(sum(p.numel() for p in lowrank.parameters()), 362_466_560)
        layer = lowrank.layers["0"]
        self.assertEqual(layer.feed_forward.w1.rank, 320)
        self.assertEqual(layer.feed_forward.w2.rank, 896)
        self.assertEqual(layer.feed_forward.w3.rank, 320)
        self.assertIsInstance(layer.feed_forward.w1, LowRankLinear)
        self.assertNotIsInstance(layer.attention.wq, LowRankLinear)

    def test_svd_initialization_is_balanced(self) -> None:
        torch.manual_seed(7)
        dense = torch.nn.Linear(12, 8, bias=False)
        lowrank = LowRankLinear.from_linear(
            dense,
            rank=4,
            method="svd",
            disable_c=True,
        )
        singular_values = torch.linalg.svdvals(dense.weight)[:4]

        torch.testing.assert_close(
            lowrank.A.T @ lowrank.A,
            torch.diag(singular_values),
        )
        torch.testing.assert_close(
            lowrank.B @ lowrank.B.T,
            torch.diag(singular_values),
        )

    def test_flop_matched_horizon(self) -> None:
        common = dict(
            batch_size=1,
            seq_len=2_048,
            vocab_size=32_000,
            hidden_size=1_280,
            num_layers=20,
            num_heads=20,
            n_kv_heads=20,
            ffn_hidden_size=3_584,
            attention_flop_accounting="flash_causal_7",
        )
        dense_total = calculate_transformer_training_flops(
            **common,
            use_low_rank=False,
        )[2]
        ffn_lowrank_total = calculate_transformer_training_flops(
            **common,
            use_low_rank=True,
            rank_ratio=0.25,
            exclude_modules=["tok_embeddings", "output", "attention"],
            disable_c=True,
        )[2]

        matched_steps = round(9_307 * dense_total / ffn_lowrank_total)
        self.assertEqual(matched_steps, 12_368)
        self.assertEqual(matched_steps * 512 * 2_048, 12_968_787_968)

    def test_spectron_factor_lr_and_decay_lr_are_independent(self) -> None:
        param_group = {
            "params": [torch.nn.Parameter(torch.zeros(2, 2))],
            "lr": 0.01,
            "is_lowrank": True,
            "param_name": "layers.0.feed_forward.w1.A",
            "lr_multiplier": 5.0,
            "lowrank_weight_decay": 0.1,
            "weight_decay": 0.0,
        }
        updated = apply_lowrank_lr_overrides(
            [param_group],
            base_lr=0.01,
            spectral_scaling={"layers.0.feed_forward.w1.A": 0.25},
            spectral_lr_target="ffn",
            lowrank_ffn_lr_multiplier=5.0,
            apply_spectral_lr_scaling=True,
        )

        self.assertEqual(updated, 1)
        self.assertAlmostEqual(param_group["lr"], 0.05 * 0.25)
        self.assertAlmostEqual(
            lowrank_decoupled_weight_decay_lr(
                param_group,
                base_lr=0.01,
                spectral_lr_target="ffn",
            ),
            0.01,
        )

    def test_500m_protocol_uses_keller_original_muon_adjustment(self) -> None:
        # The OAR 500M dense reference uses Keller/original Muon, not
        # Moonshot/Adam-RMS matching. Pin the concrete scaling so the 500M
        # positive-control run cannot silently switch optimizer conventions.
        self.assertAlmostEqual(
            _adjust_lr(0.01, "original", torch.Size([3_584, 320])),
            0.01 * (3_584 / 320) ** 0.5,
        )
        self.assertAlmostEqual(
            _adjust_lr(0.01, "match_rms_adamw", torch.Size([3_584, 320])),
            0.01 * 0.2 * (3_584 ** 0.5),
        )
        self.assertNotAlmostEqual(
            _adjust_lr(0.01, "original", torch.Size([3_584, 320])),
            _adjust_lr(0.01, "match_rms_adamw", torch.Size([3_584, 320])),
        )

    def test_launcher_prepares_matched_muon_and_spectron_jobs(self) -> None:
        root = Path(__file__).resolve().parent
        launcher = root / "bin" / "prepare_jz_500m_ffn_spectron.sh"
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "JOB_DIR": str(Path(tmp) / "jobs"),
                    "LOG_DIR": str(Path(tmp) / "logs"),
                }
            )
            subprocess.run(
                ["bash", str(launcher), "a100_4_dev2h_cpu30_whj", "both"],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            jobs = sorted((Path(tmp) / "jobs").glob("*.slurm"))
            scripts = [job.read_text() for job in jobs]

        self.assertEqual(len(scripts), 2)
        muon = next(
            script
            for script in scripts
            if 'SPECTRAL_LR_SCALING="${SPECTRAL_LR_SCALING:-0}"' in script
        )
        spectron = next(
            script
            for script in scripts
            if 'SPECTRAL_LR_SCALING="${SPECTRAL_LR_SCALING:-1}"' in script
        )

        for script in scripts:
            self.assertIn('HIDDEN_SIZE="${HIDDEN_SIZE:-1280}"', script)
            self.assertIn('NUM_LAYERS="${NUM_LAYERS:-20}"', script)
            self.assertIn('NUM_HEADS="${NUM_HEADS:-20}"', script)
            self.assertIn(
                'source "/lustre/fswork/projects/rech/qps/ulf36rc/spectron/.venv_spectron/bin/activate"',
                script,
            )
            self.assertIn('TOTAL_STEPS="${TOTAL_STEPS:-12368}"', script)
            self.assertIn('LR_SCHEDULE_STEPS="${LR_SCHEDULE_STEPS:-12368}"', script)
            self.assertIn('GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"', script)
            self.assertIn('MAX_LR="${MAX_LR:-0.01}"', script)
            self.assertIn('WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"', script)
            self.assertIn('ADJUST_MUON_LR="${ADJUST_MUON_LR:-original}"', script)
            self.assertIn("--exclude_modules tok_embeddings output attention", script)
            self.assertIn('CHECKPOINT_DIR="$CHECKPOINT_ROOT/$RUN_NAME"', script)
            self.assertIn('RESUME_FLAGS=(--resume_from "$RESUME_FROM")', script)
            self.assertIn('"${RESUME_FLAGS[@]}"', script)

        self.assertIn(
            'LOWRANK_FFN_LR_MULTIPLIER="${LOWRANK_FFN_LR_MULTIPLIER:-1.0}"',
            muon,
        )
        self.assertIn('SPECTRAL_LR_TARGET="${SPECTRAL_LR_TARGET:-all}"', muon)
        self.assertIn(
            'LOWRANK_FFN_LR_MULTIPLIER="${LOWRANK_FFN_LR_MULTIPLIER:-5.0}"',
            spectron,
        )
        self.assertIn('SPECTRAL_LR_TARGET="${SPECTRAL_LR_TARGET:-ffn}"', spectron)
        self.assertIn("--spectral_lr_scaling", spectron)

    def test_mechanistic_matrix_checkpoint_knobs_are_plumbed(self) -> None:
        root = Path(__file__).resolve().parent
        launcher = root / "bin" / "submit_jz_mechanistic_matrix.sh"
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "JOB_DIR": str(Path(tmp) / "jobs"),
                    "LOG_DIR": str(Path(tmp) / "logs"),
                    "MATRIX_CHECKPOINT_INTERVAL_STEPS": "500",
                    "MATRIX_CHECKPOINT_KEEP_LATEST_K": "1",
                }
            )
            subprocess.run(
                ["bash", str(launcher), "a100_4_dev2h_cpu30_whj"],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            jobs = sorted((Path(tmp) / "jobs").glob("*.slurm"))
            scripts = [job.read_text() for job in jobs]

        self.assertEqual(len(scripts), 8)
        for script in scripts:
            self.assertIn(
                'CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-500}"',
                script,
            )
            self.assertIn(
                'CHECKPOINT_KEEP_LATEST_K="${CHECKPOINT_KEEP_LATEST_K:-1}"',
                script,
            )


if __name__ == "__main__":
    unittest.main()
