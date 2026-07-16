# Track A scope and evidence report

Evidence refreshed: 2026-07-16. Local reference host: Windows, CPython 3.11.9, CPU-only. Generated
reports and package artifacts remain gitignored; the commands in `docs/reproducibility.md` reproduce
them.

## Demonstrated locally on the current hardened branch

- A clean hash-only development environment installs with `setuptools 80.9.0`, exposes the
  BrainFlow-required `pkg_resources`, and passes `pip check`.
- 121 unit, integration, numeric, lifecycle, negative-path, CLI, and package tests pass, including a
  real BrainFlow synthetic-board lifecycle and first settled window within 6.5 seconds.
- Overall branch coverage is 91.49%; every required critical module remains above 90%.
- Ruff formatting, correctness/security lint, strict mypy, and the repository legal/supply-chain/docs
  policy pass.
- Exact three-seed, five-fold, 25-permutation validation passes both decoders: Riemann grand balanced
  accuracy 0.9875 and vector 0.9917; the minimum observed fold is 0.9375 and maximum permutation p95
  is 0.575.
- Both immutable model kinds save/load with prediction parity; contract, version, and content
  tampering are rejected before unpickle.
- Both locked 200-window workloads pass with zero 256 ms deadline misses. Total p50/p95 is
  1.17/1.80 ms for Riemann and 1.75/2.68 ms for the temporal-vector decoder on this host.
- Source and wheel distributions build under the proprietary license metadata. A clean runtime-only
  lock installs the wheel, passes dependency/version/license checks, exposes all four commands, and
  passes packaged 200-window smoke.
- The wheel contains `py.typed` and the proprietary LICENSE and excludes tests, recordings, models,
  and generated local artifacts.

## Configured but not yet observed for this branch

- Current hardened CI is configured for Windows and Ubuntu with immutable Action SHAs, read-only
  permissions, hash installs, full tests/coverage, smoke evidence, and isolated package verification.
- Cross-platform acceptance remains pending until an actual current-branch CI run is observed.

## Not demonstrated

- Cyton hardware streaming or any other person-connected acquisition.
- Electrode contact quality, signal safety, or real EEG validity.
- Human motor-imagery separability, clinical efficacy, medical use, or the neuroscience thesis.
- GPU execution, LSL interoperability, deep learning, a service API, or a desktop GUI.
- Signed model provenance, sandboxed model execution, encrypted recording storage, or remote telemetry.

## Explicit deferral

Optional batch Riemannian session recentering (FR-041 / AC-011) is deferred. Fixed Track A does not
require it, and no incremental/adaptive calibration claim is made.
