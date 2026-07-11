# Chrono Link

Chrono Link Track A is a CPU-only Python 3.11 foundation for developing an EEG-style signal and
decoder pipeline with BrainFlow's synthetic board or deterministic local replay. It implements
consume-once acquisition, causal stateful filtering, sample-indexed windows, stable temporal and
Riemannian decoder contracts, local recording/replay, immutable model bundles, visualization, and
four installed commands.

This repository's positive results are software evidence on synthetic data. They are not evidence of
successful Cyton streaming, valid electrode contact, real EEG, human motor-imagery separability,
medical utility, or the broader neuroscience thesis.

## Install

The reference environment is Python 3.11.9. On Windows, install the exact hash-locked dependencies,
then install the project without resolving them again:

~~~powershell
py -3.11 -m venv .venv
..venv\Scripts\python -m pip install --require-hashes -r locks\windows-py311-dev.txt
..venv\Scripts\python -m pip install --no-deps -e .
~~~

For a runtime-only wheel installation:

~~~powershell
..venv\Scripts\python -m pip install --require-hashes -r locks\windows-py311-runtime.txt
..venv\Scripts\python -m pip install --no-deps dist\chrono_link-0.1.0-py3-none-any.whl
~~~

The bounds in `pyproject.toml` define supported dependency families; the lock files define the exact
reference versions and hashes used for evidence.

## Commands

Run a two-second synthetic plot-only session and write a PNG plus strict JSON:

~~~powershell
chrono-stream --board synthetic --seconds 2 --headless --out-dir artifacts\run
~~~

Recording is opt-in, requires a finite duration, and never overwrites an existing file:

~~~powershell
chrono-stream --board synthetic --seconds 10 `
  --record recordings\session.npz --headless --out-dir artifacts\record-run
chrono-stream --replay recordings\session.npz --headless --out-dir artifacts\replay-run
~~~

Run the normative leakage-safe validation, train immutable models, or execute the locked performance
smoke workload:

~~~powershell
chrono-validate --decoder both --seeds 7,42,20260709 --out artifacts\validation.json
chrono-train --decoder riemann --validation-report artifacts\validation.json --out-dir models
chrono-smoke --out-dir artifacts\smoke --windows 200
~~~

Model bundles contain joblib data, which can execute code when loaded. Loading is refused unless the
caller explicitly supplies `--trust-model`, and manifest/contract/payload hashes are checked before
unpickling:

~~~powershell
chrono-stream --replay recordings\session.npz `
  --model models\<model-id> --trust-model --headless --out-dir artifacts\model-run
~~~

Equivalent module forms are available as `python -m chrono_link stream|validate|train|smoke`.

## Runtime contract

The default chain is:

~~~text
BrainFlow drain or replay
  -> optional raw recorder fan-out
  -> causal 0.5 Hz high-pass + 50/100 Hz notches
  -> broadband / MI / mu / beta stateful band branches
  -> 1,000-sample warm-up gate
  -> W=250, H=64 exact windows
  -> quality gate
  -> optional bound decoder
  -> callbacks, visualization, and strict metrics
~~~

The synthetic and Cyton profiles resolve sampling rate, board rows, channel names, timestamps,
markers, and counters at runtime. Cyton configuration is supported, but a serial port is required
before session preparation and no person-connected hardware claim is made.

Configuration precedence is CLI > environment > strict JSON > profile/default. The only environment
keys are `CHRONO_BOARD`, `CHRONO_SERIAL_PORT`, `CHRONO_MAINS_HZ`, and `CHRONO_OUTPUT_ROOT`; unknown
`CHRONO_` keys fail closed.

## Development gates

~~~powershell
..venv\Scripts\ruff check src tests tools
..venv\Scripts\mypy src
..venv\Scripts\python -m pytest --cov=chrono_link --cov-branch --cov-report=json
..venv\Scripts\python tools\check_coverage.py
..venv\Scripts\python -m build
..venv\Scripts\python -m pip check
~~~

See [architecture](docs/architecture.md), [reproducibility](docs/reproducibility.md), and the
[scope/evidence report](docs/scope-report.md). The implementation follows the
[reviewed Track A specification](specs/chrono-link-track-a-software-foundation-implementation-spec.md).
