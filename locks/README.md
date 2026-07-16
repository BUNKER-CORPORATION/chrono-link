# Dependency locks

These pip-tools files are the exact, hash-verified Python 3.11 reference environments:

- `windows-py311-runtime.txt`: dependencies needed to install and run the wheel.
- `windows-py311-dev.txt`: runtime dependencies plus build, test, lint, typing, and lock tooling.

The historical `windows-` name identifies the host on which resolution is performed. Every recorded
hash file is installed unchanged on both Windows and Ubuntu CI; it is not considered cross-platform
evidence until both jobs pass.

## Install contract

```powershell
python -m pip install --require-hashes -r locks\windows-py311-dev.txt
python -m pip install --no-deps -e .
python -m pip check
```

Never omit `--require-hashes` for an evidence environment and never let editable/wheel installation
resolve dependencies again.

## Regeneration

Run from the repository root in the existing pinned Python 3.11 development environment:

```powershell
python -m piptools compile --allow-unsafe --generate-hashes --strip-extras `
  --extra=dev --output-file locks\windows-py311-dev.txt pyproject.toml
python -m piptools compile --allow-unsafe --generate-hashes --strip-extras `
  --output-file locks\windows-py311-runtime.txt pyproject.toml
```

Pip-tools reuses compatible pins from an existing output file by default. Use an intentional upgrade
flag/package only when the pull request is explicitly upgrading dependencies.

## Review checklist

- Every changed version is explained.
- Hash changes correspond to the intended package/version and trusted package index.
- Both locks contain `setuptools==80.9.0` while BrainFlow 5.22.2 uses `pkg_resources`.
- Development and runtime graphs contain only expected dependencies.
- A fresh venv installs each lock with hashes and passes `pip check`.
- Windows and Ubuntu tests, package isolation, and smoke pass.
- Numeric/model fingerprints and normative evidence are refreshed if the scientific stack changes.

Do not hand-edit generated package blocks. Change `pyproject.toml` and regenerate the locks.
