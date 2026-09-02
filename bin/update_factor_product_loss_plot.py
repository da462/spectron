#!/usr/bin/env python3
"""Fetch JZ logs and plot the matched factor-product optimizer runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


LOSS_RE = re.compile(
    r"Step\s+(\d+),\s+Rank\s+(\d+):\s+Loss=([0-9.eE+-]+)"
)


@dataclass(frozen=True)
class Run:
    label: str
    job_id: str
    color: str


RUNS = (
    Run("Factor-Muon", "1656518", "#222222"),
    Run("Rank-aware Product-AdamRMS", "1673742", "#167d9a"),
    Run("HeadClip", "1673961", "#b94c3a"),
)


def run_command(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def jz_command(gateway: str, target: str, command: list[str]) -> list[str]:
    return [
        "ssh",
        gateway,
        "ssh",
        "-i",
        "~/.ssh/id_ed25519",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "IdentityAgent=none",
        target,
        *command,
    ]


def job_record(gateway: str, target: str, job_id: str) -> tuple[str, str]:
    output = run_command(
        jz_command(
            gateway,
            target,
            [
                "sacct",
                "-j",
                job_id,
                "-X",
                "-n",
                "-P",
                "-o",
                "JobIDRaw,State,StdOut",
            ],
        )
    )
    for line in output.splitlines():
        fields = line.split("|", 2)
        if len(fields) == 3 and fields[0] == job_id:
            state = fields[1].split()[0].split("+")[0]
            return state, fields[2].replace("%j", job_id)
    return "UNKNOWN", ""


def fetch_log(
    gateway: str,
    target: str,
    remote_path: str,
    destination: Path,
) -> bool:
    if not remote_path:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        result = subprocess.run(
            jz_command(gateway, target, ["cat", remote_path]),
            stdout=handle,
            stderr=subprocess.PIPE,
        )
    return result.returncode == 0 and destination.stat().st_size > 0


def parse_four_rank_loss(path: Path) -> list[tuple[int, float]]:
    losses: dict[int, dict[int, float]] = {}
    for step, rank, loss in LOSS_RE.findall(path.read_text(errors="replace")):
        losses.setdefault(int(step), {})[int(rank)] = float(loss)
    return [
        (step, sum(by_rank.values()) / len(by_rank))
        for step, by_rank in sorted(losses.items())
        if len(by_rank) == 4
    ]


def ema(values: list[float], decay: float) -> list[float]:
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(decay * smoothed[-1] + (1.0 - decay) * value)
    return smoothed


def tail_mean(rows: list[tuple[int, float]], count: int = 100) -> float:
    tail = rows[-count:]
    return sum(value for _, value in tail) / len(tail)


def write_csv(path: Path, curves: dict[str, list[tuple[int, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("condition", "step", "four_rank_mean_training_ce"))
        for label, rows in curves.items():
            writer.writerows((label, step, loss) for step, loss in rows)


def plot(
    path: Path,
    curves: dict[str, list[tuple[int, float]]],
    states: dict[str, str],
    ema_decay: float,
) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.84)
    for run in RUNS:
        rows = curves.get(run.label, [])
        if not rows:
            continue
        steps = [step for step, _ in rows]
        values = [value for _, value in rows]
        tail = tail_mean(rows)
        state = states[run.label]
        label = (
            f"{run.label} ({state.lower()}, step {steps[-1]}, "
            f"tail-100 {tail:.4f})"
        )
        smoothed = ema(values, ema_decay)
        axis.plot(
            steps,
            values,
            color=run.color,
            alpha=0.11,
            linewidth=0.55,
        )
        axis.plot(
            steps,
            smoothed,
            color=run.color,
            linewidth=2.0,
            label=label,
        )

    axis.set_xlim(0, 2234)
    axis.set_ylim(top=3.4)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Training cross-entropy")
    axis.grid(True, alpha=0.20)
    axis.legend(frameon=False, fontsize=9)
    subtitle = "LR and schedule 7e-3 | WD 0.1 | embedding std 0.02 | 4-rank mean"
    fig.suptitle(
        "Low-rank FFN product-update comparison\n" + subtitle,
        fontsize=12,
        y=0.97,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="gpu")
    parser.add_argument("--target", default="ulf36rc@jean-zay.idris.fr")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/factor_product_variants"),
    )
    parser.add_argument("--ema-decay", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1)")
    raw_dir = args.output_dir / "raw"
    curves: dict[str, list[tuple[int, float]]] = {}
    states: dict[str, str] = {}
    metadata: dict[str, object] = {
        "protocol": {
            "steps": 2234,
            "lr": 7e-3,
            "lr_schedule_steps": 2234,
            "weight_decay": 0.1,
            "embedding_init_std": 0.02,
        },
        "runs": {},
    }
    for run in RUNS:
        state, remote_path = job_record(args.gateway, args.target, run.job_id)
        states[run.label] = state
        local_path = raw_dir / f"{run.job_id}.txt"
        fetched = fetch_log(
            args.gateway,
            args.target,
            remote_path,
            local_path,
        )
        rows = parse_four_rank_loss(local_path) if fetched else []
        if rows:
            curves[run.label] = rows
        metadata["runs"][run.label] = {
            "job_id": run.job_id,
            "state": state,
            "remote_log": remote_path,
            "last_complete_step": rows[-1][0] if rows else None,
            "tail_100_ce": tail_mean(rows) if rows else None,
            "tail_100_ppl": math.exp(tail_mean(rows)) if rows else None,
        }

    if not curves:
        raise RuntimeError("no complete four-rank loss rows are available")
    write_csv(args.output_dir / "training_ce.csv", curves)
    plot(
        args.output_dir / "training_ce.png",
        curves,
        states,
        args.ema_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
