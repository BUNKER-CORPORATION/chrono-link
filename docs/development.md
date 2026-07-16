# Development and quality model

## Prerequisites

- CPython 3.11.9, 64-bit.
- Git and a shell appropriate to the platform.
- Access authorized under the proprietary license and company policy.
- No hardware is required for the default synthetic/replay workflow.

Bootstrap commands are maintained in [CONTRIBUTING.md](../CONTRIBUTING.md). Always install a committed
hash lock first and install the project with `--no-deps` afterward.

## Change workflow

1. Refresh `main` and confirm the worktree is clean.
2. Create one focused branch.
3. Read the relevant specification, callers, tests, and persisted contracts.
4. Run the narrowest reproducer or test before editing.
5. Implement at the owning boundary; avoid parallel code paths for the same responsibility.
6. Add deterministic success and failure-path tests.
7. Run the full local gate and update docs/changelog.
8. Open a pull request using the repository template and wait for supported-platform CI.

## Test tiers

| Tier | Purpose | Typical command |
| --- | --- | --- |
| Focused | Fast feedback on one module/behavior | `python -m pytest tests/test_filters.py -q` |
| Core suite | All deterministic unit/integration tests | `python -m pytest -q` |
| Coverage | Branch and critical-module enforcement | `pytest --cov=chrono_link --cov-branch --cov-report=json:coverage.json` then `tools/check_coverage.py` |
| Runtime smoke | Two decoder paths and deadline budget | `chrono-smoke --windows 200 ...` |
| Normative evidence | Three seeds, grouped five-fold CV, 25 permutations | `chrono-validate --decoder both ...` |
| Package isolation | Build, clean venv, locked runtime deps, wheel, CLI help/smoke | CI package job |

Numeric tests use deterministic seeds and explicit tolerances. Do not loosen a tolerance before
identifying whether the change is expected mathematics, platform noise, or a regression.

## Static policy

- Ruff is the only formatter and linter. The configured rules include correctness, modernization,
  simplification, and security checks.
- Production correctness must not depend on `assert`; optimized Python removes assertions.
- Strict mypy applies to the `chrono_link` package. Avoid broad `Any`, unchecked casts, and ignored
  errors unless an external untyped boundary is documented narrowly.
- Public data crossing modules uses typed immutable contracts where practical.
- `tools/check_repository.py` verifies legal metadata, required governance/docs, version sync,
  dependency compatibility, and immutable Action references.

## Coverage policy

Overall branch coverage must remain at least 85%. Acquisition, filters, windows, features, recording,
decoders, and model storage each remain at least 90%. Coverage is a floor, not evidence that numeric
or scientific behavior is correct; critical invariants still require direct assertions and negative
tests.

## Dependency changes

Read [locks/README.md](../locks/README.md). A dependency change must include:

- a reason and compatibility impact;
- updated `pyproject.toml` bounds or exact compatibility pin;
- regenerated development and runtime hash locks;
- a clean locked installation and `pip check`;
- package isolation, relevant runtime smoke, and normative evidence when math may change;
- changelog/reproducibility updates.

BrainFlow 5.22.2 imports `pkg_resources` at runtime. `setuptools==80.9.0` is therefore an intentional
runtime compatibility pin, not unused build tooling. Removing or moving that boundary requires both
supported-platform synthetic-board tests.

## Documentation and claims

Update documentation in the same pull request as behavior. Keep configured, locally observed, CI
observed, hardware observed, and human-subject evidence distinct. Never infer a scientific or safety
claim from passing software tests.

Generated artifacts, recordings, and model payloads are not committed. If a reviewer needs evidence,
attach sanitized CI artifacts or provide deterministic reproduction commands.
