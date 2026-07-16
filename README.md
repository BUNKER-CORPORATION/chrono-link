# Chrono Link

[![CI](https://github.com/BUNKER-CORPORATION/chrono-link/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BUNKER-CORPORATION/chrono-link/actions/workflows/ci.yml)

Chrono Link is a typed, CPU-only Python foundation for deterministic EEG-style acquisition, causal
signal processing, leakage-safe decoder validation, immutable model artifacts, and real-time
operation. Track A runs end to end with BrainFlow's synthetic board or validated local replay so the
software contract can be proved before person-connected hardware work begins.

> **Proprietary:** Copyright (c) 2026 Bunker Corporation. All rights reserved. Source visibility does
> not grant reuse or redistribution rights. See [LICENSE](LICENSE).
>
> **Research boundary:** Synthetic-data performance is software evidence only. It does not establish
> valid Cyton streaming, electrode contact, real EEG quality, motor-imagery efficacy, safety, medical
> utility, or clinical fitness.

## Status

| Dimension | Current contract |
| --- | --- |
| Product stage | Track A software foundation, pre-release `0.1.0` |
| Python | CPython 3.11.9; `>=3.11,<3.12` |
| Reference platforms | Windows and Ubuntu Linux CI |
| Compute | CPU-only |
| Live source | BrainFlow synthetic board; Cyton configuration is present but hardware is unproven |
| Offline source | Strict, pickle-free Chrono Link NPZ recordings |
| Decoders | Riemannian tangent-space LDA and temporal-vector shrinkage LDA |
| License | Proprietary, all rights reserved |

## What the foundation provides

### Acquisition and continuity

- A consume-once `BrainFlowSource` lifecycle with one owner for prepare, start, drain, stop, and
  release.
- Runtime board-fact resolution for sampling rate, data rows, timestamps, markers, counters, and
  channel names.
- Package-counter gap detection, invalid-timestamp handling, stall detection, and explicit reset
  events.
- Bounded, atomic recording and deterministic replay with strict schema/type validation.

### Signal and feature pipeline

- Stateful causal high-pass and mains-notch filtering with independent broadband, MI, mu, and beta
  branches.
- A 1,000-sample settling gate followed by exact 250-sample windows at a 64-sample hop.
- Chunk-invariant processing: samples are filtered once before overlap is assembled.
- Window quality checks, Welch power, Hilbert amplitude/phase summaries, spectral entropy, and a
  stable temporal feature schema.

### Decoder and artifact controls

- Fold-local Riemannian and vector decoder pipelines with trial-grouped cross-validation.
- Trial-level scoring, deterministic multi-seed evidence, and grouped label permutations.
- Immutable model directories with manifest, contract, payload hashes, dependency versions, and a
  pickle-free self-test.
- Explicit `--trust-model` acknowledgement before any joblib payload is deserialized.

### Engineering controls

- Strict configuration precedence and rejection of unknown `CHRONO_` environment keys.
- Ruff formatting/linting, security rules, strict mypy, branch coverage, package isolation, and
  latency smoke gates.
- Hash-locked runtime/development environments and immutable GitHub Action references.
- Repository policy, ownership, security disclosure, support, contribution, and release rules.

## Runtime architecture

```text
BrainFlow drain or validated replay
  |
  +--> raw recorder fan-out (optional, bounded, atomic)
  +--> explicit local visualizer callback
  |
  `--> causal high-pass + 50/100 Hz notch cascade
         `--> independent broadband / MI / mu / beta branches
                `--> 1,000-sample settling gate
                       `--> W=250, H=64 exact windows
                              `--> quality gate
                                     +--> plot-only callbacks
                                     `--> contract-bound decoder
                                            `--> non-raw prediction events
```

Continuity gaps reset filters, settling state, and window buffers before new samples can contribute
to a window. The complete ownership, lifecycle, and persistence model is in
[docs/architecture.md](docs/architecture.md).

## Quick start

The committed dependency lock is the evidence environment. Install it with hashes, then install
Chrono Link without invoking the resolver a second time.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r locks\windows-py311-dev.txt
.venv\Scripts\python -m pip install --no-deps -e .
.venv\Scripts\python -m pip check
```

### Ubuntu Linux

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r locks/windows-py311-dev.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
```

The lock filename records its generation host; CI installs the same hashes on both supported
platforms. See [locks/README.md](locks/README.md) for the update protocol and the intentionally pinned
BrainFlow/setuptools compatibility boundary.

Run a two-second headless synthetic session:

```powershell
.venv\Scripts\chrono-stream --board synthetic --seconds 2 --headless `
  --out-dir artifacts\quickstart
```

Expected outputs:

```text
artifacts/quickstart/
  metrics.json       strict runtime and timing summary
  predictions.json   empty in plot-only mode
  signal.png         deterministic headless visualization
```

Every output path refuses accidental replacement.

## Commands

| Command | Purpose | Important boundary |
| --- | --- | --- |
| `chrono-stream` | Stream BrainFlow or replay a recording through the real-time pipeline | Recording is opt-in; model loading requires trust |
| `chrono-validate` | Run grouped deterministic decoder evidence | Synthetic evidence only |
| `chrono-train` | Validate, fit, and publish an immutable model bundle | Training requires a passing normative report |
| `chrono-smoke` | Exercise both decoders against latency/deadline gates | Minimum 200 measured windows |

Equivalent module forms are available through
`python -m chrono_link stream|validate|train|smoke`.

### Record and replay

Recording requires a finite duration and is never enabled implicitly:

```powershell
.venv\Scripts\chrono-stream --board synthetic --seconds 10 `
  --record recordings\synthetic-session.npz `
  --headless --out-dir artifacts\record-run

.venv\Scripts\chrono-stream --replay recordings\synthetic-session.npz `
  --headless --out-dir artifacts\replay-run
```

### Validate both decoder families

The default is the normative three-seed, five-fold, 25-permutation protocol:

```powershell
.venv\Scripts\chrono-validate --decoder both `
  --seeds 7,42,20260709 `
  --out artifacts\validation.json
```

Validation groups all overlapping windows from one trial, averages probabilities per trial before
scoring, and permutes labels at trial granularity. It fails if folds leak a trial, the deterministic
protocol changes, or accuracy/permutation thresholds are missed.

### Train and use a trusted model

```powershell
.venv\Scripts\chrono-train --decoder riemann `
  --validation-report artifacts\validation.json `
  --out-dir models

.venv\Scripts\chrono-stream --replay recordings\synthetic-session.npz `
  --model models\<model-id> --trust-model `
  --headless --out-dir artifacts\model-run
```

Model bundles contain joblib data, which can execute code. Only load an artifact you obtained through
an approved trusted channel. `--trust-model` is an acknowledgement, not a sandbox.

### Run the performance gate

```powershell
.venv\Scripts\chrono-smoke --out-dir artifacts\smoke --windows 200
```

The smoke command includes causal preprocessing, both decoder paths, persisted JSON metrics, and the
256 ms hop-deadline check.

## Configuration

Precedence is:

```text
CLI argument > CHRONO_* environment variable > strict JSON > profile/default
```

Only these environment keys are recognized:

| Key | Meaning |
| --- | --- |
| `CHRONO_BOARD` | `synthetic` or `cyton` |
| `CHRONO_SERIAL_PORT` | Serial port required before Cyton acquisition |
| `CHRONO_MAINS_HZ` | `50` or `60` |
| `CHRONO_OUTPUT_ROOT` | Local output root |

Unknown keys beginning with `CHRONO_` fail closed. Start from
[examples/config.synthetic.json](examples/config.synthetic.json); the complete field, precedence, and
validation contract is in [docs/configuration.md](docs/configuration.md).

## Artifact and data policy

The following root directories are intentionally ignored:

| Directory | Contents | Handling |
| --- | --- | --- |
| `recordings/` | Raw selected channels, timestamps, counters, markers, config, events | Sensitive; local/approved storage only |
| `models/` | Trusted-code model payloads, manifests, self-tests | Verify source and hashes before trust |
| `artifacts/` | Plots, metrics, validation, smoke, test evidence | Sanitize before sharing |

Ordinary metrics, errors, and prediction events are designed not to contain raw sample arrays. This
does not make generated files anonymous or safe to publish. Follow
[docs/security-and-privacy.md](docs/security-and-privacy.md).

## Quality gates

```powershell
.venv\Scripts\ruff format --check src tests tools
.venv\Scripts\ruff check src tests tools
.venv\Scripts\mypy src
.venv\Scripts\python -m pytest --cov=chrono_link --cov-branch `
  --cov-report=json:coverage.json
.venv\Scripts\python tools\check_coverage.py
.venv\Scripts\python tools\check_repository.py
.venv\Scripts\python -m build
.venv\Scripts\python -m pip check
```

The coverage gate requires at least 85% overall branch coverage and at least 90% for acquisition,
filters, windows, features, recording, decoders, and model storage. CI additionally installs the
wheel in a clean environment, exercises all installed commands, uploads sanitized smoke evidence,
and runs the supported Windows/Ubuntu matrix.

The current locked local Track A evidence includes 121 tests, 91.49% branch coverage, exact
three-seed decoder validation, immutable model round-trips, and two 200-window deadline workloads.
Numbers and non-claims are maintained in [docs/scope-report.md](docs/scope-report.md); reproduction
instructions are in [docs/reproducibility.md](docs/reproducibility.md).

## Repository layout

```text
chrono-link/
  .github/                 ownership, templates, dependency updates, CI
  docs/                    architecture, operations, security, quality, release
  examples/                safe configuration examples
  locks/                   hash-pinned Python 3.11 environments
  specs/                   reviewed requirements and implementation contract
  src/chrono_link/         typed production package
    cli/                   four installed command surfaces
  tests/                   unit, integration, numeric, lifecycle, package tests
  tools/                   coverage and repository-policy gates
  CHANGELOG.md             user-visible change history
  CONTRIBUTING.md          authorized engineering workflow
  GOVERNANCE.md            decision, ownership, branch, release policy
  LICENSE                  proprietary terms
  SECURITY.md              private disclosure and supported versions
  SUPPORT.md               supported support channels and inputs
```

## Documentation

Start with [docs/index.md](docs/index.md), or go directly to:

- [Architecture and invariants](docs/architecture.md)
- [Configuration reference](docs/configuration.md)
- [Developer workflow and quality model](docs/development.md)
- [Operator runbooks and troubleshooting](docs/operations.md)
- [Reproducibility protocol](docs/reproducibility.md)
- [Security, privacy, and threat model](docs/security-and-privacy.md)
- [Release process](docs/release-process.md)
- [Scope and observed evidence](docs/scope-report.md)
- [Reviewed Track A implementation specification](specs/chrono-link-track-a-software-foundation-implementation-spec.md)

## Contributing, support, and security

Only authorized proprietary contributions are accepted; see [CONTRIBUTING.md](CONTRIBUTING.md) and
[GOVERNANCE.md](GOVERNANCE.md). Use [SUPPORT.md](SUPPORT.md) for software support boundaries. Report
vulnerabilities and privacy issues privately according to [SECURITY.md](SECURITY.md).
