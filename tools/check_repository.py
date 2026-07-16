"""Enforce repository-wide legal, metadata, and supply-chain policy."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LICENSE = "LicenseRef-Proprietary"
EXPECTED_SETUPTOOLS = "setuptools==80.9.0"
ACTION_USE = re.compile(r"^\s*-?\s*uses:\s*([^#\s]+)", re.MULTILINE)
IMMUTABLE_ACTION = re.compile(r"^[^@]+@[0-9a-f]{40}$")
VERSION = re.compile(r'^__version__\s*=\s*"([^"]+)"$', re.MULTILINE)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

REQUIRED_PATHS = (
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "LICENSE",
    "NOTICE.md",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    ".editorconfig",
    ".gitattributes",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/workflows/ci.yml",
    "docs/index.md",
    "docs/architecture.md",
    "docs/configuration.md",
    "docs/development.md",
    "docs/operations.md",
    "docs/release-process.md",
    "docs/reproducibility.md",
    "docs/security-and-privacy.md",
    "docs/scope-report.md",
    "examples/config.synthetic.json",
    "locks/README.md",
)


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _check_required_files(failures: list[str]) -> None:
    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).is_file():
            failures.append(f"required file is missing: {relative}")


def _check_legal_metadata(project: dict[str, object], failures: list[str]) -> None:
    license_text = _read("LICENSE")
    if "PROPRIETARY SOFTWARE NOTICE" not in license_text:
        failures.append("LICENSE is not the Chrono Link proprietary notice")
    if "MIT License" in license_text:
        failures.append("LICENSE still contains MIT terms")
    if project.get("license") != EXPECTED_LICENSE:
        failures.append(f"project.license must be {EXPECTED_LICENSE}")

    readme = _read("README.md")
    if "Proprietary" not in readme or "[LICENSE](LICENSE)" not in readme:
        failures.append("README must expose the proprietary LICENSE boundary")


def _check_version(project: dict[str, object], failures: list[str]) -> None:
    source = _read("src/chrono_link/__init__.py")
    matched = VERSION.search(source)
    source_version = matched.group(1) if matched else None
    if source_version != project.get("version"):
        failures.append(
            "package version drift: "
            f"pyproject={project.get('version')!r}, source={source_version!r}"
        )


def _check_dependency_boundary(project: dict[str, object], failures: list[str]) -> None:
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or EXPECTED_SETUPTOOLS not in dependencies:
        failures.append(f"project dependencies must pin {EXPECTED_SETUPTOOLS}")
    for relative in (
        "locks/windows-py311-dev.txt",
        "locks/windows-py311-runtime.txt",
    ):
        if f"{EXPECTED_SETUPTOOLS} \\" not in _read(relative):
            failures.append(f"{relative} does not pin {EXPECTED_SETUPTOOLS}")


def _check_action_pins(failures: list[str]) -> None:
    workflow_root = ROOT / ".github" / "workflows"
    for workflow in sorted(workflow_root.glob("*.yml")):
        content = workflow.read_text(encoding="utf-8")
        for action in ACTION_USE.findall(content):
            if action.startswith("./"):
                continue
            if not IMMUTABLE_ACTION.fullmatch(action):
                failures.append(
                    f"{workflow.relative_to(ROOT).as_posix()} uses mutable action ref {action}"
                )


def _check_document_links(failures: list[str]) -> None:
    documentation = [*ROOT.glob("*.md")]
    for directory in (".github", "docs", "locks", "specs"):
        documentation.extend((ROOT / directory).rglob("*.md"))
    for document in sorted(documentation):
        content = document.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(content):
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative = target.split("#", maxsplit=1)[0]
            if not relative:
                continue
            destination = (document.parent / relative).resolve()
            if not destination.exists():
                failures.append(f"{document.relative_to(ROOT).as_posix()} has broken link {target}")


def _check_example_config(failures: list[str]) -> None:
    try:
        payload = json.loads(_read("examples/config.synthetic.json"))
    except json.JSONDecodeError as exc:
        failures.append(f"synthetic example is invalid JSON: {exc}")
        return
    if not isinstance(payload, dict) or set(payload) != {"board", "output_root", "signal"}:
        failures.append("synthetic example must contain board, output_root, and signal objects")


def main() -> int:
    failures: list[str] = []
    _check_required_files(failures)
    payload = tomllib.loads(_read("pyproject.toml"))
    project = payload.get("project")
    if not isinstance(project, dict):
        failures.append("pyproject.toml is missing [project]")
    else:
        _check_legal_metadata(project, failures)
        _check_version(project, failures)
        _check_dependency_boundary(project, failures)
    _check_action_pins(failures)
    _check_document_links(failures)
    _check_example_config(failures)

    if failures:
        print("repository policy failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(
        "repository policy passed: proprietary metadata, required files, "
        "dependency boundary, immutable Actions, docs, and examples"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
