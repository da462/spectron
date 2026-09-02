#!/usr/bin/env python3
"""Validate product-space invariants from a top-singular-pin smoke run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED_FINITE = (
    "original_product_momentum_sigma1",
    "original_product_momentum_sigma2",
    "top_left_singular_residual_relative",
    "top_right_singular_residual_relative",
    "mapped_sigma1",
    "mapped_sigma2",
    "mapped_original_top_coefficient",
    "desired_to_mapped_relative_frobenius_error",
    "pre_first_order_direction_rms",
    "product_adamrms_multiplier",
    "first_order_direction_rms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--rms-tolerance", type=float, default=5e-3)
    parser.add_argument("--minimum-rows", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.metrics.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) < args.minimum_rows:
        raise SystemExit(
            f"expected at least {args.minimum_rows} rows, found {len(rows)}"
        )

    max_rms_error = 0.0
    max_identity_error = 0.0
    max_left_residual = 0.0
    max_right_residual = 0.0
    for index, row in enumerate(rows):
        if row.get("factor_update_variant") != "top_singular_pin":
            raise SystemExit(f"row {index} has the wrong optimizer variant")
        for field in REQUIRED_FINITE:
            if not math.isfinite(float(row[field])):
                raise SystemExit(f"row {index} has non-finite {field}")

        intended_top = float(row["intended_original_top_coefficient"])
        original_sigma1 = float(row["original_product_momentum_sigma1"])
        original_sigma2 = float(row["original_product_momentum_sigma2"])
        intended_sigma2 = float(row["intended_second_singular_value"])
        shift = float(row["top_component_shift"])
        identity_error = max(
            abs(intended_top - 1.0),
            abs(intended_sigma2 - original_sigma2),
            abs(shift - (original_sigma1 - 1.0)),
        )
        max_identity_error = max(max_identity_error, identity_error)

        target = float(row["target_first_order_direction_rms"])
        achieved = float(row["first_order_direction_rms"])
        rms_error = abs(achieved - target) / max(abs(target), 1e-30)
        max_rms_error = max(max_rms_error, rms_error)
        max_left_residual = max(
            max_left_residual,
            float(row["top_left_singular_residual_relative"]),
        )
        max_right_residual = max(
            max_right_residual,
            float(row["top_right_singular_residual_relative"]),
        )

    if max_identity_error > 1e-6:
        raise SystemExit(
            f"top-only spectrum identity error is {max_identity_error:.3e}"
        )
    if max_rms_error > args.rms_tolerance:
        raise SystemExit(
            f"shared product RMS relative error is {max_rms_error:.3e}"
        )

    print(f"rows={len(rows)}")
    print(f"max_top_only_identity_error={max_identity_error:.6e}")
    print(f"max_shared_product_rms_relative_error={max_rms_error:.6e}")
    print(f"max_left_singular_residual={max_left_residual:.6e}")
    print(f"max_right_singular_residual={max_right_residual:.6e}")


if __name__ == "__main__":
    main()
