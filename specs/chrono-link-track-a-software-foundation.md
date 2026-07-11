# Chrono Link — Track A Software Foundation (Synthetic-Data BCI Pipeline)

Status: Draft
Owner: Chrono Link (user) — R&D
Created: 2026-07-09
Last updated: 2026-07-09

## 1. Executive Summary

Chrono Link is a ground-up, minimal-channel EEG brain–computer interface (BCI). Its
core thesis is to **decode time, not space**: the skull destroys fine spatial detail
through volume conduction but passes temporal, phase, and rhythmic structure nearly
intact, so the decoder targets self-generated *temporal codes* over ~2 sensorimotor
channels (C3/C4) rather than a spatial motor map.

This spec covers **Track A only: the real-time software foundation, developed entirely
against BrainFlow's synthetic board** — no electrodes, no hardware purchase required.
Because BrainFlow exposes an identical API across its synthetic board and the OpenBCI
Cyton (ADS1299) we plan to buy, ~90% of Chrono Link's software can be built, tested, and
validated now; switching to real hardware later is a one-line board-ID change.

Track A delivers five components:

1. A **board-agnostic acquisition interface** (ring-buffered, sliding-window).
2. A **filter chain**: mains notch (50/60 Hz) + per-band Butterworth bandpass, with both
   zero-phase (offline) and causal streaming (online) modes.
3. A **temporal/phase feature-extraction layer** — the thesis-critical piece: band power,
   Hilbert instantaneous phase/amplitude, inter-channel phase-locking value (PLV),
   spectral entropy, and SPD covariance matrices.
4. A **pluggable decoder scaffold**: a Riemannian tangent-space + shrinkage-LDA baseline
   and a band-power + LDA baseline, with a co-adaptive online-recentering hook.
5. A **live signal plotter** (time series + PSD) plus a **synthetic motor-imagery (MI)
   generator** that fabricates a two-class dataset with known band-power structure, so the
   decoder can be validated to separate classes *above chance* on data that isn't just
   fixed sine waves.

The environment has already been provisioned and verified (see §5).

## 2. Background & Problem Statement

Real EEG hardware introduces two coupled problems at once: electrode/contact noise and
biological signal validity. Debugging a full BCI pipeline against real electrodes conflates
"my code is wrong" with "my contact is bad" with "the paradigm doesn't separate." That is
the wrong place to start a ground-up build.

The synthetic board removes the hardware variable entirely. It produces deterministic,
reproducible multi-channel signals through the *exact same BrainFlow API* the Cyton uses.
This lets us:

- Build and unit-test the entire acquisition → filter → feature → decode → visualize chain
  deterministically.
- Validate decoder logic against fabricated data with *known* class structure (something
  the fixed-sine synthetic board cannot itself provide).
- Arrive at the day the Cyton ships with a trusted, tested software stack, so first contact
  with real signal debugs *only* the electrode/biology layer.

Non-goals of Track A (hardware front-end design, dry-electrode physics, the language-model
output layer, real motor-imagery decoding accuracy) are deliberately deferred to later
tracks; see §4.

## 3. Goals

- G1: Stream from BrainFlow's synthetic board through a board-agnostic interface that swaps
  to the Cyton via a single config change.
- G2: Implement a correct, testable notch + bandpass filter chain in both zero-phase and
  causal-streaming forms.
- G3: Implement a temporal/phase feature layer (band power, Hilbert phase/amplitude, PLV,
  spectral entropy, SPD covariance) that operates on any window of any channel subset.
- G4: Provide a pluggable decoder scaffold with two working baselines (Riemannian TS+LDA;
  band-power+LDA) and an online-adaptation hook.
- G5: Render a live trace + PSD on screen from the synthetic board, with a headless
  PNG-export mode for CI/verification.
- G6: Provide a synthetic MI generator and prove, in an automated test, that the decoder
  classifies its two classes well above chance (target ≥ 80% cross-validated).
- G7: Ship as an installable, importable Python package with a pytest suite and a runnable
  CLI/entry-point per component.

## 4. Non-Goals

- N1: Any analog front-end / amplifier / electrode hardware design (that is the deferred
  "custom silicon" decision, only revisited after the paradigm proves out).
- N2: Real motor-imagery decoding accuracy on human subjects — Track A validates *plumbing
  and decoder logic*, not neurophysiological separability.
- N3: The language-model output/composition layer (few-bits → fluent control).
- N4: EMG/jaw-clench or any muscle-based control input (explicitly rejected by the project;
  do not reintroduce).
- N5: Multi-user accounts, cloud sync, networking beyond an optional local LSL stream.
- N6: Production packaging/distribution, installers, or GUI application shell.

## 5. Repository Findings

The repository is currently **empty** (`X:\code\bunker-corporation\chrono-link`, not a git
repo). No existing specs, docs, CLAUDE.md, or source. This spec establishes the initial
conventions.

**Environment already provisioned and verified on this machine (2026-07-09):**

- Interpreters available: Python 3.14.3 (default) and Python 3.11.9. **Python 3.11 chosen**
  for the project venv (`.venv/`) — mature wheels for the whole scientific stack; 3.14 is
  wheel-roulette for scipy/scikit-learn/matplotlib.
- Installed into `.venv` (verified import + install success): `brainflow 5.22.2`,
  `numpy 2.4.6`, `scipy 1.17.1`, `scikit-learn 1.9.0`, `pyriemann 0.12`, `matplotlib 3.11.0`.
- **Synthetic board probed and confirmed streaming:** `BoardIds.SYNTHETIC_BOARD` (id `-1`),
  sampling rate **250 Hz**, **16 EEG channels** on rows `1..16`, named
  `['Fz','C3','Cz','C4','Pz','PO7','Oz','PO8','F5','F7','F3','F1','F2','F4','F6','F8']` —
  crucially, **`C3` (row 2) and `C4` (row 4)** exist, matching the project's target
  montage. Timestamp channel is row 30; total 32 rows; a 1.5 s capture returned shape
  `(32, 376)`.

Constraints imposed by these findings:

- Sampling rate is 250 Hz on both synthetic and Cyton — window/FFT sizing can assume 250 Hz
  as the default but must read it from `BoardShim.get_sampling_rate(board_id)`.
- Channel-row indices differ per board; never hardcode row numbers — resolve channels by
  name/index through `BoardShim.get_eeg_channels` / `get_eeg_names`.
- With only ~2 decode channels, covariance matrices are 2×2 (tangent-space vector length 3);
  covariance estimation must use shrinkage/OAS to stay well-conditioned on short windows.

## 6. Research Notes & References

Facts (F), best-practice recommendations (R), and assumptions (A) are labelled.

- **BrainFlow buffering (F/R).** BrainFlow maintains an internal ring buffer.
  `get_board_data()` returns and **removes** all buffered data; `get_current_board_data(n)`
  returns the last ≤ n samples and **does not remove** them — the correct primitive for
  overlapping sliding windows in a real-time loop. At stream start the buffer may be shorter
  than requested, so code must tolerate short first windows.
  ([BrainFlow User API](https://brainflow.readthedocs.io/en/stable/UserAPI.html),
  [Realtime visualization](https://brainflow.org/2021-01-04-data-visualization/))
- **Riemannian MI pipeline (F/R).** The pyRiemann canonical MI pipeline is
  `Covariances() → TangentSpace() → classifier`; tangent-space mapping turns SPD covariance
  matrices into Euclidean vectors, and TS + LDA is a strong, data-efficient baseline that
  has won BCI competitions. Chosen baseline: `Covariances(estimator='oas') →
  TangentSpace(metric='riemann') → LDA(solver='lsqr', shrinkage='auto')`.
  ([pyRiemann MI example](https://pyriemann.readthedocs.io/en/latest/auto_examples/biosignal-mi/plot_single.html),
  [TangentSpace docs](https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.tangentspace.TangentSpace.html))
- **Non-stationarity / co-adaptation (F/R).** Riemannian **recentering (RCT)** matches
  covariate shift on the manifold and can run online, unsupervised, without labels — the
  right mechanism for day-to-day drift and warm-start personalization. pyRiemann exposes
  this via its `transfer` module (`TLCenter` and related). This is the concrete hook behind
  the project's "decoder and brain train each other" goal.
  ([Riemannian EEG review, arXiv:2407.20250](https://arxiv.org/abs/2407.20250),
  [Riemannian Procrustes Analysis](https://www.researchgate.net/publication/329913851_Riemannian_Procrustes_Analysis_Transfer_Learning_for_Brain-Computer_Interfaces))
- **Phase connectivity features (F).** Phase-Locking Value (PLV) in sensor space is a strong
  MI feature — reported as the best-performing connectivity feature and competitive with
  single-channel features, and combining amplitude + phase features improves decoding. This
  directly supports making inter-channel PLV a first-class Chrono Link feature, consistent
  with the "decode time/phase" thesis.
  ([Phase-based connectivity decoding, PMC9279670](https://pmc.ncbi.nlm.nih.gov/articles/PMC9279670/),
  [Connectivity + amplitude, MDPI](https://www.mdpi.com/2306-5354/12/6/614))
- **Minimal channels + EOG (F).** 2025 literature confirms channel reduction is a legitimate,
  active direction and that a few EEG channels plus EOG can beat many mediocre EEG channels
  — supporting recording extra channels but decoding few. EOG is treated here as an
  artifact/observation channel, **not** a control input.
  ([Reduced-channel MI, PMC11723053](https://pmc.ncbi.nlm.nih.gov/articles/PMC11723053/))
- **Future deep path (A/R).** EEGNet/Braindecode are worth trying once real labelled data
  exists, but are out of scope for Track A; the decoder interface is designed so they can be
  dropped in later without touching acquisition/features.

Research limitation: recommendations above are grounded in current (2024–2026) sources and
the empirically verified local environment; exact library method signatures should be
pinned against the installed versions (`pyriemann 0.12`, `brainflow 5.22.2`) during
implementation.

## 7. Users, Use Cases & User Stories

Primary user is the project author/developer doing R&D; secondary "user" is CI/automation.

- As the developer, I want to stream synthetic EEG and see a live trace, so that I can
  confirm the acquisition + visualization chain works before any hardware exists.
- As the developer, I want to point the same code at the Cyton by changing one config value,
  so that no rewrite is needed when hardware arrives.
- As the developer, I want feature extraction and decoding as isolated, unit-tested modules,
  so that I can iterate on the temporal-code paradigm without touching I/O.
- As the developer, I want an automated test proving the decoder separates a known two-class
  dataset above chance, so that "the pipeline runs" is distinguished from "the pipeline
  decodes."
- As CI, I want a headless mode that exercises the full pipeline and exports artifacts
  (PNG/metrics) without a display, so that regressions are caught automatically.

## 8. Functional Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| FR-001 | Acquisition wraps `BoardShim`; board selected by config (`SYNTHETIC_BOARD` default, `CYTON_BOARD` supported). | Must | Serial port / MAC only needed for real boards. |
| FR-002 | Resolve sampling rate and EEG channel rows from BrainFlow at runtime; never hardcode. | Must | `get_sampling_rate`, `get_eeg_channels`, `get_eeg_names`. |
| FR-003 | Select a decode channel subset by name (default `["C3","C4"]`) with graceful fallback if a name is absent. | Must | Names differ per board. |
| FR-004 | Sliding-window reader using `get_current_board_data(n)`; configurable window length and hop; tolerate short initial windows. | Must | Non-destructive ring-buffer reads. |
| FR-005 | Notch filter at configurable mains frequency (50/60 Hz) + harmonics option. | Must | scipy `iirnotch`; default 50 Hz (EU/user locale). |
| FR-006 | Bandpass filter, configurable band(s); provide mu (8–12 Hz) and beta (13–30 Hz) presets, plus broadband. | Must | Butterworth via `sosfiltfilt` (offline) / stateful `sosfilt` (online). |
| FR-007 | Zero-phase mode for offline/batch and causal stateful mode for streaming, sharing coefficients. | Must | Preserve filter state across windows in streaming mode. |
| FR-008 | DC/detrend removal before filtering. | Should | Constant/linear detrend option. |
| FR-009 | Feature: band power per channel per band (Welch PSD integrated over band; log option). | Must | |
| FR-010 | Feature: Hilbert instantaneous amplitude and phase per band per channel. | Must | scipy `hilbert` on band-limited signal. |
| FR-011 | Feature: inter-channel Phase-Locking Value (PLV) per band. | Must | Thesis-critical temporal/phase feature. |
| FR-012 | Feature: spectral entropy per channel. | Should | Normalized PSD entropy. |
| FR-013 | Feature: SPD covariance matrix per window (shrinkage/OAS). | Must | Feeds Riemannian decoder. |
| FR-014 | Feature layer returns both a flat feature vector (for LDA) and epoch tensors (for covariance/deep). | Must | Two consumption shapes. |
| FR-015 | Decoder abstract interface: `fit(X,y)`, `predict(X)`, `predict_proba(X)`, `partial_fit`/adapt hook. | Must | Pluggable. |
| FR-016 | Baseline decoder A: `Covariances(oas) → TangentSpace → LDA(shrinkage)`. | Must | Primary baseline. |
| FR-017 | Baseline decoder B: band-power/PLV vector → standardize → LDA. | Must | Interpretable baseline. |
| FR-018 | Online co-adaptation hook via Riemannian recentering (unsupervised). | Should | pyRiemann `transfer` (`TLCenter`). |
| FR-019 | Synthetic MI generator producing a labelled two-class dataset with configurable per-class band-power (mu ERD) contrast and SNR. | Must | Enables real separation test. |
| FR-020 | Live plotter: rolling time series (selected channels) + live PSD, from the synthetic board. | Must | matplotlib `FuncAnimation`. |
| FR-021 | Headless mode for plotter/pipeline: no display, export PNG + JSON metrics. | Must | For CI/verification. |
| FR-022 | Optional session recording to disk (numpy `.npz` and/or CSV) and replay. | Should | Reproducible offline dev. |
| FR-023 | Optional LSL output stream of raw/filtered/decoded data. | Could | Interop; behind a flag. |
| FR-024 | Every component runnable via a CLI entry point / `python -m` module. | Must | |

## 9. Non-Functional Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| NFR-001 | End-to-end per-window latency (read→filter→features→predict) < window hop at 250 Hz (target < 50 ms for a 1 s window / 250 ms hop). | Must | Keep real-time headroom. |
| NFR-002 | Board portability: swap synthetic↔Cyton with zero code change beyond config. | Must | Core architectural constraint. |
| NFR-003 | Deterministic tests: fixed RNG seeds; synthetic MI generator reproducible. | Must | CI stability. |
| NFR-004 | CPU-only; no GPU dependency for baselines. | Must | Riemannian + LDA are CPU-light. |
| NFR-005 | Cross-platform code (Windows primary, Linux for later real-time work); no OS-specific calls in core. | Should | Dev is on Windows 11. |
| NFR-006 | Modules decoupled: acquisition, filters, features, decoder, viz independently importable/testable. | Must | |
| NFR-007 | Pinned dependencies (`requirements.txt` / lock) reproducing the verified environment. | Must | brainflow 5.22.2, numpy 2.4.6, scipy 1.17.1, sklearn 1.9.0, pyriemann 0.12, matplotlib 3.11.0. |
| NFR-008 | Config-driven (dataclass + optional file); no magic numbers in logic. | Should | |
| NFR-009 | Type hints + docstrings on public API; lint clean. | Should | |
| NFR-010 | Biosignal data at rest treated as private; recordings default to a gitignored local dir. | Should | Even synthetic; sets the habit for real data. |

## 10. Proposed Solution

### 10.1 Package layout

```
chrono-link/
├── .venv/                       # Python 3.11 (provisioned)
├── requirements.txt             # pinned to verified versions
├── pyproject.toml               # package metadata + entry points
├── README.md
├── specs/
│   └── chrono-link-track-a-software-foundation.md
├── chrono_link/
│   ├── __init__.py
│   ├── config.py                # ChronoConfig dataclass (board, channels, fs, bands, window)
│   ├── acquisition.py           # Acquisition: BoardShim wrapper + sliding-window reader
│   ├── filters.py               # notch + bandpass; zero-phase + stateful streaming
│   ├── features.py              # band power, Hilbert phase/amp, PLV, spectral entropy, covariance
│   ├── decoder.py               # Decoder ABC + RiemannTSLDA + BandPowerLDA + online recenter hook
│   ├── synthetic_mi.py          # fabricated 2-class MI dataset generator
│   ├── pipeline.py              # real-time orchestration loop (read→filter→features→decode)
│   ├── recording.py             # save/replay sessions (.npz/csv)   [Should]
│   └── viz.py                   # live plot (time series + PSD) + headless PNG export
├── scripts/
│   ├── stream_live.py           # FR-020 live plotter entry point
│   ├── smoke_test.py            # end-to-end headless run + artifacts
│   └── validate_decoder.py      # FR-019/G6 above-chance proof
└── tests/
    ├── test_acquisition.py
    ├── test_filters.py
    ├── test_features.py
    ├── test_decoder.py
    └── test_pipeline.py
```

### 10.2 Component responsibilities & flow

1. **config.py** — `ChronoConfig`: `board_id`, `serial_port`, `decode_channels=["C3","C4"]`,
   `record_channels` (superset), `mains_hz=50`, `bands={"mu":(8,12),"beta":(13,30)}`,
   `window_s=1.0`, `hop_s=0.25`, `filter_order`, seeds. Single source of truth.
2. **acquisition.py** — `Acquisition(config)` wraps `BoardShim`; `start()/stop()` manage
   session/stream; `read_window()` returns the latest `(n_channels, n_samples)` array via
   `get_current_board_data`, resolving channel rows by name; `iter_windows()` yields
   overlapping windows at the configured hop.
3. **filters.py** — precompute SOS coefficients from `fs`; `notch()`, `bandpass(band)`;
   `apply_offline()` (`sosfiltfilt`, zero-phase) and `StreamFilter` (stateful `sosfilt`,
   carries `zi` across windows for causal real-time).
4. **features.py** — pure functions on a `(channels, samples)` window:
   `band_power`, `hilbert_phase_amp`, `plv(ch_i, ch_j, band)`, `spectral_entropy`,
   `covariance`. `FeatureExtractor` assembles a named flat vector and/or epoch tensor per
   the configured bands/channels.
5. **decoder.py** — `Decoder` ABC; `RiemannTSLDA` (sklearn Pipeline: `Covariances(oas) →
   TangentSpace(riemann) → LDA(lsqr, shrinkage='auto')`); `BandPowerLDA`
   (`StandardScaler → LDA`); `.adapt(X_unlabeled)` implementing unsupervised Riemannian
   recentering.
6. **synthetic_mi.py** — generates band-limited signals with class-dependent mu power
   (simulated ERD) at C3/C4, additive pink/white noise at configurable SNR, returning
   `(epochs, labels)`; deterministic under seed.
7. **pipeline.py** — `RealtimePipeline` ties it together: `for window in
   acq.iter_windows(): x = stream_filter(window); f = features(x); y = decoder.predict(f)`;
   emits callbacks/LSL; measures per-stage latency.
8. **viz.py / scripts** — live matplotlib animation and a headless variant that renders the
   same figure to PNG and writes a metrics JSON.

### 10.3 Alternatives considered

- **BrainFlow `DataFilter` for filtering** instead of scipy. Rejected as the *primary* path:
  scipy gives transparent, version-stable coefficients and easy zero-phase/stateful control
  that suits the "immaculate channel" thesis and unit testing. BrainFlow `DataFilter`
  remains a documented fallback.
- **Minimum Distance to Mean (MDM)** instead of TS+LDA. Kept as an easy secondary decoder;
  TS+LDA chosen as primary for accuracy and because tangent-space vectors compose with
  standard sklearn tooling.
- **Deep nets (EEGNet) now.** Deferred (N-scope) — no labelled human data yet; the decoder
  interface leaves the door open.
- **Threaded/async acquisition.** BrainFlow already streams on its own thread into the ring
  buffer; a synchronous windowed reader is simpler and sufficient for Track A. Revisit if
  NFR-001 latency budget is threatened.

## 11. API, Interface, CLI, or UX Contract

Public Python API (illustrative signatures; pin to installed versions in code):

```python
cfg = ChronoConfig(board_id=BoardIds.SYNTHETIC_BOARD, decode_channels=["C3","C4"])

with Acquisition(cfg) as acq:                 # context manager: prepare/start/stop/release
    for window in acq.iter_windows():         # np.ndarray (n_ch, n_samp)
        ...

sf = StreamFilter(cfg)                         # stateful causal filter
x = sf(window)                                 # notch -> bandpass, preserves state

fx = FeatureExtractor(cfg)
vec = fx.vector(x)                             # 1-D np.ndarray, named order
cov = fx.covariance(epoch)                     # (n_ch, n_ch) SPD

dec = RiemannTSLDA(cfg).fit(epochs, labels)
proba = dec.predict_proba(epochs)
dec.adapt(unlabeled_epochs)                    # unsupervised recentering
```

CLI entry points (also `python -m chrono_link.<module>`):

- `chrono-stream` (`scripts/stream_live.py`): `--board synthetic|cyton --channels C3,C4
  --seconds N --headless --out plot.png`. Exit 0 on clean shutdown.
- `chrono-smoke` (`scripts/smoke_test.py`): runs the full chain headless, writes
  `artifacts/smoke_<ts>.png` and `artifacts/smoke_<ts>.json` (per-stage latency, shapes).
- `chrono-validate` (`scripts/validate_decoder.py`): builds a synthetic MI set, runs
  stratified CV, prints/writes accuracy; **exit non-zero if accuracy < 0.80** (CI gate).

Configuration keys (env/file optional): `CHRONO_BOARD`, `CHRONO_SERIAL_PORT`,
`CHRONO_MAINS_HZ`, `CHRONO_RECORD_DIR`.

## 12. Data Model & Persistence

- No database. Optional session recording (FR-022): raw + metadata to `.npz`
  (`data`, `channel_names`, `fs`, `board_id`, `created`) and/or CSV for interop; replay
  source implements the same `iter_windows()` contract as `Acquisition` so the pipeline is
  agnostic to live-vs-recorded input.
- Recordings and artifacts default to a **gitignored** `recordings/` and `artifacts/` dir.
- No schema migrations. Forward compatibility: persist `board_id`, `fs`, and channel names
  with every recording so files remain interpretable if defaults change.

## 13. Security, Privacy & Abuse Considerations

Track A is a local R&D tool with synthetic data, so the surface is small, but habits set
here carry to real neural data later:

- **Biosignal privacy (habit-setting).** Treat all recordings as private by default;
  gitignore `recordings/`; never commit captured data. Document that real EEG is sensitive
  personal data.
- **No secrets.** No API keys, no network auth in scope. If LSL output (FR-023) is enabled,
  it binds locally; document that it broadcasts on the local network and default it off.
- **Untrusted external content.** Research sources were treated as untrusted (facts only).
- **Input validation.** Validate config (channel names exist on the board, band within
  Nyquist, window ≥ filter transient); fail fast with clear errors.
- **Electrical safety (informational, out of Track A scope).** Recorded here so it is not
  lost: anything connected to a person must be battery-powered and mains-isolated. The
  Cyton (battery + Bluetooth) satisfies this; no mains-tethered acquisition, ever.

## 14. Error Handling & Edge Cases

| Scenario | Expected Behavior | Notes |
| --- | --- | --- |
| Buffer shorter than requested window at stream start | Skip/accumulate until a full window exists; never pad silently in decode path | FR-004 |
| Requested channel name absent on board | Raise clear `ConfigError` listing available names; optional index fallback | FR-003 |
| Bandpass edge ≥ Nyquist (fs/2) | Reject at config validation with actionable message | 250 Hz → Nyquist 125 Hz |
| Covariance singular on 2 short-window channels | Use OAS/shrinkage estimator; guarantees SPD | FR-013 |
| Filter transient at first window | Warm up `StreamFilter` state; mark first N windows as "settling" | FR-007 |
| Board disconnect / stream error mid-run | Catch BrainFlow error, stop cleanly, release session, non-zero exit | resource safety |
| No display available (headless/CI) | Auto-select non-interactive matplotlib backend (`Agg`), export PNG | FR-021 |
| Decoder `predict` before `fit` | Raise `NotFittedError` | sklearn convention |
| NaN/inf in a window | Detect, drop window, log a counter; do not propagate | robustness |
| Synthetic MI SNR so low classes overlap | Test asserts a documented minimum SNR yields ≥ target accuracy | avoids flaky CI |

## 15. Observability & Operations

- **Logging.** Standard `logging`; per-stage debug logs behind a flag; BrainFlow board
  logger toggled via config (`disable_board_logger()` by default for quiet runs).
- **Metrics.** Pipeline records per-window latency (read/filter/features/decode), dropped-
  window count, and windows/sec; `chrono-smoke` writes them to JSON.
- **Health checks.** `chrono-smoke` doubles as a self-test: green means the full chain works
  end to end on this machine.
- **No production ops** (no dashboards/alerts) — this is local R&D.

## 16. Implementation Plan

### Phase 0: Project scaffolding

- [ ] Add `requirements.txt` (pinned), `pyproject.toml` (package + entry points), `README.md`, `.gitignore` (`.venv/`, `recordings/`, `artifacts/`, `__pycache__/`).
- [ ] `git init` (optional) and create `chrono_link/` + `tests/` + `scripts/` skeleton.

### Phase 1: Acquisition + live trace (the "signal on screen" milestone)

- [ ] `config.py` `ChronoConfig` with verified defaults (fs 250, channels C3/C4).
- [ ] `acquisition.py` context-managed `BoardShim` wrapper + `iter_windows()`.
- [ ] `viz.py` + `scripts/stream_live.py`: rolling time series + live PSD; headless PNG mode.
- [ ] `tests/test_acquisition.py`: synthetic board yields correct shapes; short-window tolerance.

### Phase 2: Filter chain

- [ ] `filters.py`: notch (`iirnotch`) + Butterworth bandpass (SOS); offline `sosfiltfilt` and stateful streaming `StreamFilter`.
- [ ] `tests/test_filters.py`: assert ≥X dB attenuation at mains freq; passband gain ≈ 0 dB; zero-phase vs causal parity on steady state.

### Phase 3: Feature layer (thesis-critical)

- [ ] `features.py`: band power, Hilbert phase/amp, PLV, spectral entropy, covariance; `FeatureExtractor.vector()` and `.covariance()`.
- [ ] `tests/test_features.py`: PLV ≈ 1 for phase-locked synthetic pair and ≈ 0 for independent noise; band power peaks in the injected band; covariance SPD.

### Phase 4: Decoder scaffold + validation

- [ ] `synthetic_mi.py`: labelled two-class generator (mu ERD contrast, seeded).
- [ ] `decoder.py`: `Decoder` ABC, `RiemannTSLDA`, `BandPowerLDA`, `.adapt()` recentering hook.
- [ ] `scripts/validate_decoder.py`: stratified CV, CI gate at ≥ 0.80.
- [ ] `tests/test_decoder.py`: above-chance on synthetic MI; `NotFittedError` guard; adapt() runs.

### Phase 5: Real-time orchestration + polish

- [ ] `pipeline.py`: `RealtimePipeline` read→filter→features→decode with latency metrics.
- [ ] `recording.py`: save/replay; replay implements `iter_windows()`.
- [ ] `scripts/smoke_test.py`: full headless run + artifacts; `tests/test_pipeline.py` latency budget (NFR-001).
- [ ] README quickstart; optional LSL output behind a flag.

## 17. Testing Strategy

| Test Area | Coverage | Notes |
| --- | --- | --- |
| Unit | filters (attenuation/passband), features (PLV/band power/covariance SPD), config validation | Deterministic, seeded |
| Integration | acquisition→filter→features→decoder on synthetic board; replay path parity | Uses SYNTHETIC_BOARD |
| Decoder validity | synthetic MI, stratified CV ≥ 0.80 (G6) | Distinguishes "runs" from "decodes" |
| Performance | per-window latency < hop (NFR-001) | Assert on median over N windows |
| Regression | golden feature vectors for a fixed seeded window | Detects silent numeric drift |
| Headless/CI | `chrono-smoke` exits 0 and writes artifacts with no display | `Agg` backend |
| Manual QA | run `chrono-stream` and eyeball live trace + PSD | The human "signal on screen" check |

Hardware note: the eyes-open/eyes-closed occipital-alpha "hello world" is the first *real*-
hardware validation and is **out of Track A scope** (no electrodes), but the software it
exercises is fully built and tested here.

## 18. Rollout, Migration & Rollback

- No deployment/users. "Rollout" = merge to the repo's main line behind passing tests.
- **Board migration (synthetic → Cyton)** is the one real migration: change `board_id`
  (and set `serial_port`); everything else is unchanged by design (NFR-002). Provide a
  documented `cyton` config profile.
- **Rollback** is trivial (local, version-controlled); recordings are additive and never
  overwritten (timestamped filenames).
- Compatibility: pin dependencies (NFR-007); persist `board_id`/`fs`/channel names in
  recordings so older files stay readable.

## 19. Documentation Updates

- [ ] `README.md`: quickstart (create venv, install, `chrono-stream`, `chrono-validate`).
- [ ] Module docstrings + a short `docs/` note on the "decode time not space" rationale and
      why features emphasize phase/PLV.
- [ ] A "swap to Cyton" how-to (config profile, serial port, safety note).
- [ ] Changelog entry for the Track A foundation.

## 20. Acceptance Criteria

- [ ] AC1: `chrono-stream --board synthetic` renders a live time series + PSD; `--headless`
      writes a valid PNG. (G1, G5)
- [ ] AC2: Switching config to `cyton` requires **no source code change** (verified by a test
      that constructs the pipeline for both board IDs). (G1, NFR-002)
- [ ] AC3: Filter tests show mains-frequency attenuation ≥ threshold and flat passband. (G2)
- [ ] AC4: Feature tests pass: PLV ≈ 1 (locked) / ≈ 0 (independent), band power localizes to
      the injected band, covariance is SPD. (G3)
- [ ] AC5: `chrono-validate` achieves ≥ 0.80 cross-validated accuracy on the synthetic MI
      dataset and gates CI. (G4, G6)
- [ ] AC6: `chrono-smoke` runs the full chain headless, exits 0, and emits latency metrics
      with median per-window latency < hop. (G7, NFR-001)
- [ ] AC7: All modules import independently and the pytest suite passes on the pinned env. (G7)

## 21. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
| --- | --- | --- | --- |
| Synthetic board's fixed sine waves give false confidence ("runs" ≠ "decodes") | High | High | Mandatory synthetic MI generator + above-chance CI gate (AC5); documented explicitly |
| Library API drift (pyRiemann/BrainFlow signatures) | Medium | Medium | Pin versions; thin adapters around library calls; tests catch breakage |
| Real-time latency exceeds hop on weaker hardware | Medium | Low | Latency test (NFR-001); vectorized features; optional larger hop |
| 2-channel covariance ill-conditioned | Medium | Medium | OAS/shrinkage estimator (FR-013) |
| Over-fitting the scaffold to synthetic quirks | Medium | Medium | Keep decoder pluggable; treat synthetic accuracy as a plumbing check, not a science claim |
| Python 3.14 default interpreter accidentally used | Low | Low | Project pinned to `.venv` (3.11); document interpreter in README |
| Windows-specific real-time jitter | Low | Medium | Keep core OS-agnostic; note Linux for later hard-real-time work |

## 22. Open Questions

| Question | Blocking? | Suggested Owner | Notes |
| --- | --- | --- | --- |
| Mains frequency default — 50 Hz (EU) confirmed for user locale? | No | User | Configurable; default 50 Hz assumed. |
| Target window/hop for the paradigm (1.0 s / 0.25 s assumed)? | No | Developer | Tunable; defaults set. |
| Do we want LSL output in Track A or defer? | No | Developer | Marked "Could"; default off. |
| Adopt git now for the repo? | No | User | Recommended; enables rollback/history. |
| Minimum acceptable synthetic-MI SNR/accuracy for the CI gate (0.80 proposed)? | No | Developer | Tune to avoid flakiness. |

## 23. Assumptions

- A1: Mains interference modeled at 50 Hz by default (user locale); trivially switchable to 60 Hz.
- A2: 250 Hz sampling for both synthetic and Cyton (verified for synthetic; standard for Cyton).
- A3: Two decode channels (C3, C4) as default montage, with extra channels recorded for R&D.
- A4: CPU-only baselines are sufficient for Track A; no GPU.
- A5: Development proceeds on Windows 11 with the provisioned `.venv` (Python 3.11).
- A6: Synthetic MI separation is a *software-validity* signal only, not a neurophysiological claim.
- A7: The language-model output layer and dry-electrode physics are separate future tracks.

## 24. Appendix

**Verified synthetic board facts (2026-07-09 probe).** id `-1`; fs 250 Hz; 16 EEG rows
`1..16`; names include `C3` (row 2) and `C4` (row 4); timestamp row 30; 32 total rows.

**Primary baseline pipeline (pseudocode).**

```python
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    Covariances(estimator="oas"),
    TangentSpace(metric="riemann"),
    LDA(solver="lsqr", shrinkage="auto"),
)
# online drift: recenter incoming covariances toward a running reference (unsupervised)
```

**Real-time loop skeleton (pseudocode).**

```python
with Acquisition(cfg) as acq:
    sf = StreamFilter(cfg)
    fx = FeatureExtractor(cfg)
    for window in acq.iter_windows():       # get_current_board_data(n), non-destructive
        x = sf(window)                       # notch -> bandpass, stateful
        f = fx.vector(x)                     # band power / PLV / entropy ...
        y = decoder.predict_proba([f])       # once fitted
        emit(y)
```

**Key references:**
[BrainFlow User API](https://brainflow.readthedocs.io/en/stable/UserAPI.html) ·
[BrainFlow realtime viz](https://brainflow.org/2021-01-04-data-visualization/) ·
[pyRiemann MI example](https://pyriemann.readthedocs.io/en/latest/auto_examples/biosignal-mi/plot_single.html) ·
[Riemannian EEG review (arXiv:2407.20250)](https://arxiv.org/abs/2407.20250) ·
[Phase-based connectivity decoding (PMC9279670)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9279670/) ·
[Connectivity + amplitude features (MDPI)](https://www.mdpi.com/2306-5354/12/6/614) ·
[Reduced-channel MI (PMC11723053)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11723053/)
