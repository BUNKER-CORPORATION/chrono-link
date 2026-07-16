# Changelog

All notable Chrono Link changes are recorded here. The project follows semantic versioning once a
version is released; the current package remains pre-release.

## Unreleased

### Changed

- Replaced the MIT license with proprietary Bunker Corporation terms and aligned package metadata,
  notices, contribution rules, and release policy with that legal boundary.
- Rebuilt the README and documentation as role-oriented product, architecture, configuration,
  development, operations, reproducibility, security/privacy, and release manuals.
- Hardened CI with least-privilege permissions, credential isolation, concurrency cancellation,
  timeouts, immutable current Action SHAs, named evidence steps, package metadata checks, and a clean
  installed-wheel smoke run.
- Adopted Ruff formatting plus broader correctness, simplification, and security lint rules across
  production code, tests, and repository tools.
- Replaced production assertions with explicit invariant handling so optimized Python cannot remove
  runtime safeguards.
- Added repository ownership, governance, security disclosure, support, issue/PR templates,
  Dependabot policy, line-ending policy, and an executable repository-policy gate.

### Fixed

- Pinned `setuptools==80.9.0` in project metadata and both hash locks because BrainFlow 5.22.2 imports
  `pkg_resources`; unconstrained setuptools 83.0.0 broke real synthetic-board tests on Windows and
  Ubuntu.
- Replaced yanked development dependency `build 1.5.1` with non-yanked `build 1.5.0`.
- Made `tests` an importable package so cross-test fixtures collect consistently in CI.
- Corrected invalid `.venv` paths in the original Windows onboarding commands.

### Track A foundation

- Added strict JSON/environment/CLI configuration and runtime board-fact resolution.
- Added consume-once BrainFlow acquisition, package-gap splitting, bounded recording, and replay.
- Added causal/offline filter banks, warm-up gating, exact rolling windows, and quality checks.
- Added stable temporal features, Riemannian and vector LDA baselines, grouped validation controls,
  immutable hashed model bundles, and explicit local-model trust.
- Added live/headless visualization and `chrono-stream`, `chrono-validate`, `chrono-train`, and
  `chrono-smoke` commands.
- Added hash-locked reference environments, Windows/Linux CI, package isolation checks, numeric
  fixtures, and branch-coverage enforcement.
- Deferred optional batch session recentering; it is not required for fixed Track A completion.
