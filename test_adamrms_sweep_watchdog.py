import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "bin" / "watch_adamrms_sweep.py"
SPEC = importlib.util.spec_from_file_location("watch_adamrms_sweep", SCRIPT)
assert SPEC and SPEC.loader
WATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCH)


class AdamRMSSweepWatchdogTests(unittest.TestCase):
    def test_parse_tail_ce_uses_complete_four_rank_steps(self) -> None:
        lines = []
        for step in range(3):
            for rank in range(4):
                lines.append(f"Step {step}, Rank {rank}: Loss={step + rank / 10:.4f}, LR=0")
        lines.append("Step 3, Rank 0: Loss=99, LR=0")
        metrics = WATCH.parse_tail_ce("\n".join(lines), tail_steps=2)
        self.assertEqual(metrics["last_step"], 2)
        self.assertEqual(metrics["complete_steps"], 3)
        self.assertAlmostEqual(metrics["tail_ce"], (1.15 + 2.15) / 2)

    def test_one_e_minus_three_is_already_bracketed(self) -> None:
        decision = WATCH.next_lr_action(
            {"7e-3": 3.2, "5e-3": 3.1, "1e-3": 3.0, "7e-4": 3.05}
        )
        self.assertEqual(decision["action"], "bracketed")
        self.assertEqual(decision["lrs"], ["5e-3", "1e-3", "7e-4"])

    def test_seven_e_minus_four_requests_five_e_minus_four(self) -> None:
        decision = WATCH.next_lr_action(
            {"7e-3": 3.3, "5e-3": 3.2, "1e-3": 3.1, "7e-4": 3.0}
        )
        self.assertEqual(decision["action"], "extend")
        self.assertEqual(decision["lr"], "5e-4")

    def test_five_e_minus_four_requests_one_e_minus_four(self) -> None:
        decision = WATCH.next_lr_action(
            {
                "7e-3": 3.4,
                "5e-3": 3.3,
                "1e-3": 3.2,
                "7e-4": 3.1,
                "5e-4": 3.0,
            }
        )
        self.assertEqual(decision["action"], "extend")
        self.assertEqual(decision["lr"], "1e-4")

    def test_wd_grid_has_six_missing_runs_after_lr_row(self) -> None:
        lrs = ["5e-3", "1e-3", "7e-4"]
        state = {"runs": {}}
        for lr in lrs:
            state["runs"][WATCH.run_key(lr, "0.01")] = {"lr": lr, "wd": "0.01"}
        missing = WATCH.missing_wd_runs(state, lrs)
        self.assertEqual(len(missing), 6)
        self.assertEqual({wd for _, wd in missing}, {"0.001", "0.1"})

    def test_completed_lr_results_reads_wd_point_zero_one(self) -> None:
        state = {
            "runs": {
                WATCH.run_key("1e-3", "0.01"): {
                    "lr": "1e-3",
                    "wd": "0.01",
                    "status": "COMPLETED",
                    "tail_ce": 3.0,
                }
            }
        }
        self.assertEqual(WATCH.completed_lr_results(state), {"1e-3": 3.0})

    def test_lr_and_wd_use_separate_canonical_grids(self) -> None:
        self.assertEqual(WATCH.canonical_lr("0.0007"), "7e-4")
        self.assertEqual(WATCH.canonical_lr("0.005"), "5e-3")
        self.assertEqual(WATCH.canonical_wd("0.01"), "0.01")
        self.assertEqual(WATCH.run_key("1e-3", "0.01"), "lr=1e-3|wd=0.01")


if __name__ == "__main__":
    unittest.main()
