#!/usr/bin/env python3
"""Plot the AdamRMS cooldown run against a temporary Factor-Muon reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from update_factor_product_loss_plot import (
    LOSS_RE,
    ema,
    fetch_log,
    job_record,
    tail_mean,
)


COOLDOWN_START = round(0.70 * 2234)


def read_factor_muon(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == "Factor-Muon 7e-3":
                rows.append(
                    (int(row["step"]), float(row["four_rank_mean_training_ce"]))
                )
    if not rows:
        raise RuntimeError(f"Factor-Muon 7e-3 is absent from {path}")
    return rows


def read_rank_zero(path: Path) -> list[tuple[int, float]]:
    by_step: dict[int, float] = {}
    for step, rank, loss in LOSS_RE.findall(path.read_text(errors="replace")):
        if int(rank) == 0:
            by_step[int(step)] = float(loss)
    if not by_step:
        raise RuntimeError(f"no rank-zero losses found in {path}")
    return sorted(by_step.items())


def write_csv(
    path: Path,
    cooldown: list[tuple[int, float]],
    factor_muon: list[tuple[int, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("condition", "step", "training_ce"))
        for label, rows in (
            ("AdamRMS cooldown", cooldown),
            ("Factor-Muon temporary reference", factor_muon),
        ):
            writer.writerows((label, step, loss) for step, loss in rows)


def plot(
    path: Path,
    cooldown: list[tuple[int, float]],
    factor_muon: list[tuple[int, float]],
    ema_decay: float,
) -> None:
    fig, axis = plt.subplots(figsize=(11.5, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.78)

    series = (
        (
            "AdamRMS cooldown | LR 1e-3, WD 0.1 | flat 70%, linear decay 30%",
            cooldown,
            "#c45a16",
            "-",
        ),
        (
            "Factor-Muon (temporary reference) | LR 7e-3, WD 0.1",
            factor_muon,
            "#282828",
            "--",
        ),
    )
    for label, rows, color, linestyle in series:
        steps = [step for step, _ in rows]
        losses = [loss for _, loss in rows]
        tail = tail_mean(rows)
        axis.plot(steps, losses, color=color, alpha=0.10, linewidth=0.55)
        axis.plot(
            steps,
            ema(losses, ema_decay),
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{label} | tail-100 {tail:.4f}",
        )

    axis.axvline(
        COOLDOWN_START,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
        label=f"Cooldown begins (step {COOLDOWN_START})",
    )
    axis.set_xlim(400, 2234)
    visible = [
        loss
        for rows in (cooldown, factor_muon)
        for step, loss in rows
        if step >= 400
    ]
    axis.set_ylim(max(0.0, min(visible) - 0.04), min(4.25, max(visible) + 0.08))
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Training cross-entropy")
    axis.grid(True, alpha=0.20)
    axis.legend(
        frameon=False,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.025),
        borderaxespad=0.0,
    )
    fig.suptitle(
        "AdamRMS cooldown trajectory\n"
        "Embedding std 0.02 | 2,234 steps | EMA 0.95",
        fontsize=13,
        y=0.97,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="gpu")
    parser.add_argument("--target", default="ulf36rc@jean-zay.idris.fr")
    parser.add_argument("--cooldown-job", default="1758082")
    parser.add_argument(
        "--factor-csv",
        type=Path,
        default=Path(
            "../spectron-mechanistic/reports/factor_product_variants/final_ce.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/adamrms_cooldown"),
    )
    parser.add_argument("--ema-decay", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state, remote_path = job_record(args.gateway, args.target, args.cooldown_job)
    raw_path = args.output_dir / "raw" / f"{args.cooldown_job}.txt"
    if not fetch_log(args.gateway, args.target, remote_path, raw_path):
        raise RuntimeError(f"could not fetch cooldown log {remote_path}")

    cooldown = read_rank_zero(raw_path)
    factor_muon = read_factor_muon(args.factor_csv)
    write_csv(args.output_dir / "training_ce.csv", cooldown, factor_muon)
    plot(args.output_dir / "cooldown_vs_factor_muon.png", cooldown, factor_muon, args.ema_decay)

    metadata = {
        "cooldown": {
            "job_id": args.cooldown_job,
            "state": state,
            "last_step": cooldown[-1][0],
            "last_ce": cooldown[-1][1],
            "tail_100_ce": tail_mean(cooldown),
            "tail_100_ppl": math.exp(tail_mean(cooldown)),
        },
        "factor_muon_temporary_reference": {
            "last_step": factor_muon[-1][0],
            "tail_100_ce": tail_mean(factor_muon),
            "tail_100_ppl": math.exp(tail_mean(factor_muon)),
            "source": str(args.factor_csv),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
