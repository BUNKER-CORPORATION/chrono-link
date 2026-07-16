# Contributing to Chrono Link

Chrono Link is proprietary. Contributions are accepted only from collaborators authorized by Bunker
Corporation and covered by an applicable employment, contractor, or contribution agreement. Do not
submit code, data, recordings, model files, or confidential material unless you have the right and
authorization to do so.

## Development setup

Use CPython 3.11.9. The committed hash lock is the reproducible reference environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r locks\windows-py311-dev.txt
.venv\Scripts\python -m pip install --no-deps -e .
```

Linux:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r locks/windows-py311-dev.txt
.venv/bin/python -m pip install --no-deps -e .
```

The lock filename records where it is generated; CI validates that same artifact on Windows and
Ubuntu. See [locks/README.md](locks/README.md) before changing dependencies.

## Required local gates

Run the narrow test for the code you changed first. Before requesting review, run the complete gate:

```powershell
.venv\Scripts\ruff format --check src tests tools
.venv\Scripts\ruff check src tests tools
.venv\Scripts\mypy src
.venv\Scripts\python -m pytest --cov=chrono_link --cov-branch --cov-report=json:coverage.json
.venv\Scripts\python tools\check_coverage.py
.venv\Scripts\python tools\check_repository.py
.venv\Scripts\python -m build
.venv\Scripts\python -m pip check
```

For runtime or decoder changes, also run:

```powershell
.venv\Scripts\chrono-smoke --out-dir artifacts\smoke --windows 200
```

## Engineering rules

- Preserve consume-once acquisition: one component owns and drains each source.
- Filter samples once before overlapping window assembly; never refilter overlapping windows.
- Reset all causal state at continuity boundaries.
- Keep raw EEG out of ordinary logs, exceptions, metrics, and prediction callbacks.
- Treat recordings as sensitive data and model bundles as trusted-code artifacts.
- Keep public contracts typed and fail closed on unknown configuration or schema fields.
- Add deterministic tests for non-trivial numeric, lifecycle, persistence, and error behavior.
- Do not weaken coverage, validation, integrity, latency, or package-isolation gates to land a change.

The complete ownership and data-flow rules are in [docs/architecture.md](docs/architecture.md).

## Git and review workflow

1. Branch from current `main` using `feature/`, `fix/`, `docs/`, or `chore/` naming.
2. Keep commits focused and use an imperative summary of at most 72 characters.
3. Rebase or merge current `main` before final review; never rewrite shared history without approval.
4. Open a pull request with motivation, risk, evidence, and explicit scientific-scope impact.
5. Obtain CODEOWNER approval and green required checks before merge.
6. Use squash or rebase merging when it preserves a clear, intentional history.

Generated recordings, models, validation reports, plots, and benchmark outputs do not belong in Git.
Attach only sanitized evidence through the approved artifact channel.

## Reporting vulnerabilities

Do not open a public issue for a suspected vulnerability, privacy leak, unsafe model artifact, or
exposed recording. Follow [SECURITY.md](SECURITY.md).
