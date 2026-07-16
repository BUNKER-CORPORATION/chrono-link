"""Enforce Track A aggregate and critical-module branch coverage gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

OVERALL_MINIMUM = 85.0
CRITICAL_MINIMUM = 90.0
CRITICAL_MODULES = (
    "src/chrono_link/acquisition.py",
    "src/chrono_link/filters.py",
    "src/chrono_link/windows.py",
    "src/chrono_link/features.py",
    "src/chrono_link/recording.py",
    "src/chrono_link/decoders.py",
    "src/chrono_link/model_store.py",
)


def main() -> int:
    report_path = Path("coverage.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    overall = float(payload["totals"]["percent_covered"])
    if overall < OVERALL_MINIMUM:
        failures.append(f"overall {overall:.2f}% < {OVERALL_MINIMUM:.2f}%")
    files = {name.replace("\\", "/"): value for name, value in payload["files"].items()}
    for module in CRITICAL_MODULES:
        covered = float(files[module]["summary"]["percent_covered"])
        if covered < CRITICAL_MINIMUM:
            failures.append(f"{module} {covered:.2f}% < {CRITICAL_MINIMUM:.2f}%")
    if failures:
        print("coverage gates failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    print(f"coverage gates passed: overall={overall:.2f}% critical>={CRITICAL_MINIMUM:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
