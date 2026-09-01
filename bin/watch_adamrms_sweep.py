#!/usr/bin/env python3
"""Advance the std-0.02 AdamRMS LR/WD sweep without duplicate submissions."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


LR_GRID = (
    "7e-2",
    "5e-2",
    "1e-2",
    "7e-3",
    "5e-3",
    "1e-3",
    "7e-4",
    "5e-4",
    "1e-4",
    "7e-5",
    "5e-5",
    "1e-5",
)
WD_GRID = ("0.001", "0.01", "0.1")
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
FAILED_STATES = TERMINAL_STATES - {"COMPLETED"}
LOSS_RE = re.compile(
    r"Step\s+(\d+),\s+Rank\s+(\d+):\s+Loss=([0-9.eE+-]+)"
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def canonical_grid_value(value: str | float, grid: tuple[str, ...]) -> str:
    number = float(value)
    for candidate in grid:
        if math.isclose(number, float(candidate), rel_tol=0.0, abs_tol=1e-15):
            return candidate
    return f"{number:.12g}"


def canonical_lr(value: str | float) -> str:
    return canonical_grid_value(value, LR_GRID)


def canonical_wd(value: str | float) -> str:
    return canonical_grid_value(value, WD_GRID)


def run_key(lr: str, wd: str) -> str:
    return f"lr={canonical_lr(lr)}|wd={canonical_wd(wd)}"


def parse_tail_ce(text: str, tail_steps: int = 100) -> dict[str, float | int]:
    rows: dict[int, dict[int, float]] = {}
    for step, rank, loss in LOSS_RE.findall(text):
        rows.setdefault(int(step), {})[int(rank)] = float(loss)
    complete = [
        (step, sum(rank_losses.values()) / 4.0)
        for step, rank_losses in sorted(rows.items())
        if len(rank_losses) == 4
    ]
    if not complete:
        raise ValueError("log has no complete four-rank loss rows")
    tail = complete[-tail_steps:]
    ce = sum(loss for _, loss in tail) / len(tail)
    return {
        "last_step": complete[-1][0],
        "complete_steps": len(complete),
        "tail_steps": len(tail),
        "tail_ce": ce,
        "tail_ppl": math.exp(ce),
    }


def completed_lr_results(state: dict[str, Any], wd: str = "0.01") -> dict[str, float]:
    results = {}
    for run in state["runs"].values():
        if (
            run["wd"] == canonical_wd(wd)
            and run.get("status") == "COMPLETED"
            and "tail_ce" in run
        ):
            results[run["lr"]] = float(run["tail_ce"])
    return results


def next_lr_action(results: dict[str, float]) -> dict[str, Any]:
    if not results:
        return {"action": "wait", "reason": "no completed LR results"}
    unknown = set(results) - set(LR_GRID)
    if unknown:
        raise ValueError(f"LRs absent from LR_GRID: {sorted(unknown)}")
    best = min(results, key=lambda lr: (results[lr], LR_GRID.index(lr)))
    index = LR_GRID.index(best)
    if index == 0 or index == len(LR_GRID) - 1:
        raise ValueError(f"cannot bracket terminal LR grid point {best}")
    upper = LR_GRID[index - 1]
    lower = LR_GRID[index + 1]
    missing = [lr for lr in (upper, lower) if lr not in results]
    if missing:
        # Usually only one side is absent. Submit one point per decision tick so a
        # newly observed direction can change the next extension.
        return {
            "action": "extend",
            "best_lr": best,
            "best_ce": results[best],
            "lr": missing[0],
            "upper": upper,
            "lower": lower,
        }
    return {
        "action": "bracketed",
        "best_lr": best,
        "best_ce": results[best],
        "upper": upper,
        "lower": lower,
        "lrs": [upper, best, lower],
    }


def missing_wd_runs(state: dict[str, Any], lrs: list[str]) -> list[tuple[str, str]]:
    missing = []
    for wd in WD_GRID:
        for lr in lrs:
            if run_key(lr, wd) not in state["runs"]:
                missing.append((lr, wd))
    return missing


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "version": 1,
        "experiment": "adamrms_lowrank_ffn_embstd0p02",
        "embedding_std": "0.02",
        "selection_metric": "tail-100 four-rank mean training CE",
        "phase": "lr_sweep",
        "created_at": now(),
        "updated_at": now(),
        "runs": {},
        "events": [],
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def event(state: dict[str, Any], message: str, **fields: Any) -> None:
    state["events"].append({"time": now(), "message": message, **fields})


def seed_jobs(state: dict[str, Any], seeds: list[str]) -> None:
    for seed in seeds:
        try:
            lr, job_id = seed.split("=", 1)
        except ValueError as error:
            raise ValueError(f"invalid --seed {seed!r}; expected LR=JOB_ID") from error
        lr = canonical_lr(lr)
        key = run_key(lr, "0.01")
        existing = state["runs"].get(key)
        if existing and existing["job_id"] != job_id:
            raise ValueError(f"{key} already belongs to job {existing['job_id']}")
        if not existing:
            state["runs"][key] = {
                "lr": lr,
                "wd": "0.01",
                "embedding_std": "0.02",
                "job_id": job_id,
                "status": "UNKNOWN",
                "submitted_at": now(),
            }
            event(state, "seeded run", key=key, job_id=job_id)


def normalize_slurm_state(raw: str) -> str:
    return raw.strip().split()[0].split("+")[0]


def slurm_states(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    command = [
        "sacct",
        "-j",
        ",".join(job_ids),
        "-X",
        "-n",
        "-P",
        "--format=JobIDRaw,State",
    ]
    output = subprocess.run(command, text=True, capture_output=True, check=True).stdout
    states = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, raw_state = line.split("|", 1)
        states[job_id] = normalize_slurm_state(raw_state)
    return states


def find_log(log_dir: Path, job_id: str) -> Path | None:
    candidates = sorted(log_dir.glob(f"*_{job_id}.txt"))
    return candidates[-1] if candidates else None


def refresh_runs(state: dict[str, Any], log_dir: Path, total_steps: int) -> None:
    jobs = [run["job_id"] for run in state["runs"].values()]
    states = slurm_states(jobs)
    for key, run in state["runs"].items():
        previous = run.get("status", "UNKNOWN")
        current = states.get(run["job_id"], previous)
        run["status"] = current
        if current != previous:
            event(state, "job state changed", key=key, job_id=run["job_id"], status=current)
        log_path = find_log(log_dir, run["job_id"])
        if log_path:
            run["log"] = str(log_path)
        if current == "COMPLETED" and "tail_ce" not in run:
            if not log_path:
                raise RuntimeError(f"completed job {run['job_id']} has no log")
            metrics = parse_tail_ce(log_path.read_text(errors="replace"))
            if metrics["last_step"] < total_steps - 1:
                raise RuntimeError(
                    f"completed job {run['job_id']} ended at step {metrics['last_step']}, "
                    f"expected {total_steps - 1}"
                )
            run.update(metrics)
            event(
                state,
                "recorded completed result",
                key=key,
                job_id=run["job_id"],
                tail_ce=metrics["tail_ce"],
                tail_ppl=metrics["tail_ppl"],
            )


def active_a100_jobs() -> int:
    output = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-p", "gpu_p5", "-o", "%i"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return sum(bool(line.strip()) for line in output.splitlines())


def submit_run(
    repo: Path,
    lr: str,
    wd: str,
    dry_run: bool,
) -> str:
    lr = canonical_lr(lr)
    wd = canonical_wd(wd)
    if dry_run:
        return f"dry-{lr}-{wd}"
    environment = {
        **os.environ,
        "MODEL_TAG": "tt134m_adamrms_embstd0p02_watch_v1",
        "TOTAL_STEPS": "2234",
        "LR_SCHEDULE_STEPS": "2234",
        "MAX_LR": lr,
        "OPTIMIZER": "muon",
        "WEIGHT_DECAY": wd,
        "NH_WEIGHT_DECAY": wd,
        "ADJUST_MUON_LR": "match_rms_adamw",
        "EMBEDDING_INIT_STD": "0.02",
        "TT_STYLE_INIT": "0",
        "LOW_RANK_RATIO": "0.25",
        "LOW_RANK_W2_SAME_RANK_AS_W1W3": "0",
        "GLOBAL_BATCH_SIZE": "512",
        "MICRO_BATCH_SIZE": "16",
        "SEQUENCE_LENGTH": "2048",
        "CHECKPOINT_INTERVAL_STEPS": "500",
        "CHECKPOINT_KEEP_LATEST_K": "1",
        "SKIP_FINAL_EVAL": "1",
        "WANDB_MODE": "offline",
        "JZ_VENV": "/lustre/fswork/projects/rech/qps/ulf36rc/spectron/.venv_spectron",
    }
    result = subprocess.run(
        [
            "bash",
            str(repo / "bin" / "submit_jz_ttmatched_spectron.sh"),
            "a100_4_dev2h_cpu30_whj",
            "lowrank_ffn",
        ],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"could not parse sbatch output:\n{result.stdout}\n{result.stderr}")
    return match.group(1)


def add_submitted_run(
    state: dict[str, Any], lr: str, wd: str, job_id: str, dry_run: bool
) -> None:
    key = run_key(lr, wd)
    if key in state["runs"]:
        raise ValueError(f"refusing duplicate run {key}")
    state["runs"][key] = {
        "lr": canonical_lr(lr),
        "wd": canonical_wd(wd),
        "embedding_std": "0.02",
        "job_id": job_id,
        "status": "DRY_RUN" if dry_run else "PENDING",
        "submitted_at": now(),
    }
    event(state, "submitted run", key=key, job_id=job_id, dry_run=dry_run)


def nonterminal_runs(state: dict[str, Any], keys: list[str] | None = None) -> list[dict[str, Any]]:
    selected = state["runs"] if keys is None else {key: state["runs"][key] for key in keys}
    return [
        run
        for run in selected.values()
        if run.get("status") not in TERMINAL_STATES and run.get("status") != "DRY_RUN"
    ]


def failed_runs(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [run for run in state["runs"].values() if run.get("status") in FAILED_STATES]


def render_report(path: Path, state: dict[str, Any]) -> None:
    lines = [
        "# AdamRMS std-0.02 sweep watchdog",
        "",
        f"- Updated: {state['updated_at']}",
        f"- Phase: `{state['phase']}`",
        f"- Selection metric: {state['selection_metric']}",
        "",
        "| LR | WD | Job | State | Last step | Tail-100 CE | Implied PPL |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for run in sorted(
        state["runs"].values(),
        key=lambda item: (float(item["wd"]), -float(item["lr"])),
    ):
        lines.append(
            "| {lr} | {wd} | {job} | {status} | {step} | {ce} | {ppl} |".format(
                lr=run["lr"],
                wd=run["wd"],
                job=run["job_id"],
                status=run.get("status", "UNKNOWN"),
                step=run.get("last_step", ""),
                ce=f"{run['tail_ce']:.6f}" if "tail_ce" in run else "",
                ppl=f"{run['tail_ppl']:.4f}" if "tail_ppl" in run else "",
            )
        )
    if state.get("bracket"):
        lines.extend(["", f"LR bracket: `{state['bracket']}`"])
    if state.get("recommendation"):
        lines.extend(["", f"Recommendation: `{state['recommendation']}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def tick(args: argparse.Namespace, state: dict[str, Any]) -> None:
    refresh_runs(state, args.log_dir, args.total_steps)
    failures = failed_runs(state)
    if failures:
        state["phase"] = "blocked"
        state["blocked_reason"] = "one or more jobs failed"
        return

    if state["phase"] == "lr_sweep":
        current_keys = [key for key, run in state["runs"].items() if run["wd"] == "0.01"]
        if nonterminal_runs(state, current_keys):
            return
        decision = next_lr_action(completed_lr_results(state))
        state["lr_decision"] = decision
        if decision["action"] == "extend":
            key = run_key(decision["lr"], "0.01")
            if key not in state["runs"] and active_a100_jobs() < args.max_a100_jobs:
                job_id = submit_run(args.repo, decision["lr"], "0.01", args.dry_run)
                add_submitted_run(state, decision["lr"], "0.01", job_id, args.dry_run)
            return
        if decision["action"] != "bracketed":
            return
        state["bracket"] = decision["lrs"]
        state["phase"] = "wd_grid"
        event(state, "LR bracket complete", lrs=decision["lrs"], best_lr=decision["best_lr"])

    if state["phase"] == "wd_grid":
        lrs = state["bracket"]
        for lr, wd in missing_wd_runs(state, lrs):
            if active_a100_jobs() >= args.max_a100_jobs:
                break
            job_id = submit_run(args.repo, lr, wd, args.dry_run)
            add_submitted_run(state, lr, wd, job_id, args.dry_run)
        grid_keys = [run_key(lr, wd) for wd in WD_GRID for lr in lrs]
        if any(key not in state["runs"] for key in grid_keys):
            return
        if nonterminal_runs(state, grid_keys):
            return
        completed = [state["runs"][key] for key in grid_keys]
        if any(run.get("status") != "COMPLETED" for run in completed):
            state["phase"] = "blocked"
            state["blocked_reason"] = "WD grid did not complete cleanly"
            return
        best = min(completed, key=lambda run: float(run["tail_ce"]))
        state["recommendation"] = {
            "lr": best["lr"],
            "wd": best["wd"],
            "tail_ce": best["tail_ce"],
            "tail_ppl": best["tail_ppl"],
        }
        state["phase"] = "awaiting_std1_approval"
        event(state, "std-0.02 grid complete", recommendation=state["recommendation"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(
            "/lustre/fswork/projects/rech/qps/ulf36rc/spectron_sweep_state/"
            "adamrms_embstd0p02.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "/lustre/fswork/projects/rech/qps/ulf36rc/spectron_sweep_state/"
            "adamrms_embstd0p02.md"
        ),
    )
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--total-steps", type=int, default=2234)
    parser.add_argument("--max-a100-jobs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo = args.repo.resolve()
    if args.log_dir is None:
        args.log_dir = args.repo / "jz_logs"
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.state.with_suffix(args.state.suffix + ".lock")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another watchdog tick owns {lock_path}", file=sys.stderr)
            raise SystemExit(2)
        state = load_state(args.state)
        seed_jobs(state, args.seed)
        tick(args, state)
        save_state(args.state, state)
        render_report(args.report, state)
    print(
        json.dumps(
            {
                "phase": state["phase"],
                "runs": len(state["runs"]),
                "state": str(args.state),
                "report": str(args.report),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
