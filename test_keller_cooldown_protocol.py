import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from muon_local import SingleDeviceMuonWithAuxAdam, _adjust_lr
from training_protocols import auxiliary_adamw_lr, stable_linear_decay_factor


class TestKellerCooldownProtocol(unittest.TestCase):
    def test_stable_linear_decay_is_flat_then_reaches_zero(self) -> None:
        factors = [
            stable_linear_decay_factor(step, 10, 0.3) for step in range(10)
        ]
        self.assertEqual(factors[:7], [1.0] * 7)
        self.assertAlmostEqual(factors[7], 2.0 / 3.0)
        self.assertAlmostEqual(factors[8], 1.0 / 3.0)
        self.assertEqual(factors[9], 0.0)
        self.assertEqual(stable_linear_decay_factor(10, 10, 0.3), 0.0)

    def test_keller_adjustment_and_auxiliary_lr_are_independent(self) -> None:
        self.assertAlmostEqual(
            _adjust_lr(0.05, "original", (12, 4)), 0.05 * 3**0.5
        )
        self.assertAlmostEqual(_adjust_lr(0.05, "original", (4, 12)), 0.05)
        self.assertAlmostEqual(auxiliary_adamw_lr(0.05, 0.1), 0.005)

    def test_scheduler_preserves_auxiliary_lr_ratio(self) -> None:
        matrix = torch.nn.Parameter(torch.zeros(8, 4))
        scalar = torch.nn.Parameter(torch.zeros(4))
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": [matrix],
                    "use_muon": True,
                    "lr": 0.05,
                    "weight_decay": 0.01,
                    "adjust_lr_fn": "original",
                },
                {
                    "params": [scalar],
                    "use_muon": False,
                    "lr": 0.005,
                    "weight_decay": 0.01,
                },
            ]
        )
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                lambda step: stable_linear_decay_factor(step, 10, 0.3)
            ]
            * 2,
        )

        observed = []
        for _ in range(10):
            muon_lr, adam_lr = (group["lr"] for group in optimizer.param_groups)
            observed.append((muon_lr, adam_lr))
            if muon_lr:
                self.assertAlmostEqual(adam_lr / muon_lr, 0.1)
            optimizer.step()
            schedule.step()

        self.assertEqual(observed[0], (0.05, 0.005))
        self.assertAlmostEqual(observed[7][0], 0.05 * 2.0 / 3.0)
        self.assertAlmostEqual(observed[8][0], 0.05 / 3.0)
        self.assertEqual(observed[9], (0.0, 0.0))

    def test_dedicated_launcher_selects_exact_protocol_and_diagnostics(self) -> None:
        root = Path(__file__).resolve().parent
        launcher = root / "bin" / "submit_jz_keller_cooldown.sh"
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
                [
                    "bash",
                    str(launcher),
                    "a100_4_dev2h_cpu30_whj",
                    "lowrank_ffn",
                ],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            jobs = list((Path(tmp) / "jobs").glob("*.slurm"))
            self.assertEqual(len(jobs), 1)
            script = jobs[0].read_text()

        expected = (
            'MAX_LR="${MAX_LR:-5e-2}"',
            'ADJUST_MUON_LR="${ADJUST_MUON_LR:-original}"',
            'AUX_ADAMW_LR_MULTIPLIER="${AUX_ADAMW_LR_MULTIPLIER:-0.1}"',
            'SCHEDULER="${SCHEDULER:-stable_linear_decay}"',
            'WARMUP_RATIO="${WARMUP_RATIO:-0}"',
            'STABLE_DECAY_FRACTION="${STABLE_DECAY_FRACTION:-0.3}"',
            'MECHANISTIC_DIAGNOSTICS="${MECHANISTIC_DIAGNOSTICS:-1}"',
            'LIGHTWEIGHT_DIAGNOSTICS="${LIGHTWEIGHT_DIAGNOSTICS:-1}"',
            'SPECTRAL_LR_SCALING="${SPECTRAL_LR_SCALING:-0}"',
            '--exclude_modules tok_embeddings output attention',
            'RUN_SUFFIX="${RUN_SUFFIX}_sched${SCHED_TAG}"',
            'RUN_SUFFIX="${RUN_SUFFIX}_auxlr${AUXLR_TAG}"',
        )
        for fragment in expected:
            self.assertIn(fragment, script)


if __name__ == "__main__":
    unittest.main()
