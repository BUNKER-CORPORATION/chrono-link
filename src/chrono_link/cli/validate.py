"""Run leakage-safe synthetic decoder validation without saving a model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from chrono_link.cli.common import (
    EXIT_CONFIG,
    EXIT_RUNTIME,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    integer_csv,
    print_error,
)
from chrono_link.validation import DecoderKind, ValidationReport, validate_decoder
from chrono_link.visualization import write_strict_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chrono-validate",
        description="Validate Chrono Link synthetic baselines with grouped trial folds.",
    )
    parser.add_argument(
        "--decoder",
        choices=("riemann", "vector", "both"),
        default="both",
    )
    parser.add_argument("--seeds", type=integer_csv, default=(7, 42, 20260709))
    parser.add_argument("--out", type=Path, default=Path("artifacts/validation.json"))
    parser.add_argument(
        "--permutations",
        type=int,
        default=25,
        help="Trial-label permutation controls per seed (normative default: 25).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.permutations < 0:
        print_error("configuration", ValueError("--permutations cannot be negative"))
        return EXIT_CONFIG
    kinds: tuple[DecoderKind, ...] = (
        ("riemann", "vector") if args.decoder == "both" else (args.decoder,)
    )
    try:
        reports: dict[str, ValidationReport] = {
            kind: validate_decoder(
                kind,
                seeds=args.seeds,
                permutation_count=args.permutations,
            )
            for kind in kinds
        }
        payload = {
            "schema_version": 1,
            "reports": {
                kind: json.loads(report.to_json()) for kind, report in reports.items()
            },
            "passed": all(report.passed for report in reports.values()),
        }
        write_strict_json(args.out, payload)
    except FileExistsError as exc:
        print_error("output", exc)
        return EXIT_CONFIG
    except Exception as exc:
        print_error("validation_runtime", exc)
        return EXIT_RUNTIME

    for kind, report in reports.items():
        print(
            f"{kind}: grand_mean={report.grand_mean_balanced_accuracy:.6f} "
            f"passed={str(report.passed).lower()}"
        )
    return EXIT_SUCCESS if payload["passed"] else EXIT_VALIDATION


if __name__ == "__main__":
    raise SystemExit(main())
