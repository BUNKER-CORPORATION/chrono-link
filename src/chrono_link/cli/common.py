"""Shared CLI parsing, strict JSON loading, and exit-code constants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chrono_link.validation import ValidationError, ValidationReport

EXIT_SUCCESS = 0
EXIT_CONFIG = 2
EXIT_RUNTIME = 3
EXIT_VALIDATION = 4
EXIT_ARTIFACT = 5


def csv_tuple(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("value must be a comma-separated unique list")
    return items


def integer_csv(value: str) -> tuple[int, ...]:
    try:
        items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not items or len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique list")
    return items


def strict_json_file(path: Path) -> object:
    def reject_constant(token: str) -> None:
        raise ValueError(f"invalid JSON constant {token}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValidationError(f"cannot read strict JSON {path}: {exc}") from exc


def load_validation_report(path: Path, decoder_kind: str) -> ValidationReport:
    value = strict_json_file(path)
    if isinstance(value, dict) and "reports" in value:
        reports = value["reports"]
        if not isinstance(reports, dict) or decoder_kind not in reports:
            raise ValidationError(
                f"combined report does not contain decoder {decoder_kind!r}"
            )
        value = reports[decoder_kind]
    report = ValidationReport.from_dict(value)
    if report.decoder_kind != decoder_kind:
        raise ValidationError("validation report decoder kind does not match request")
    return report


def print_error(context: str, exc: BaseException) -> None:
    """Print structured failure context without biosignal samples."""
    payload: dict[str, Any] = {
        "status": "error",
        "context": context,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
