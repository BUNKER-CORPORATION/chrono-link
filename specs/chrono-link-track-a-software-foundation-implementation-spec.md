# Chrono Link — Track A Software Foundation Implementation Specification

Status: Draft
Owner: Chrono Link — R&D
Created: 2026-07-10
Last updated: 2026-07-10

This document is the corrected, implementation-ready successor to
[chrono-link-track-a-software-foundation.md](./chrono-link-track-a-software-foundation.md).
The earlier document remains unchanged as historical design context. After this specification is
reviewed and accepted, this document becomes the execution authority for Track A.

## 1. Executive Summary

Chrono Link is a minimal-channel EEG brain-computer-interface research project whose working thesis
is to prioritize temporal, rhythmic, amplitude, and inter-channel phase structure over dense spatial
reconstruction. Track A builds the software foundation without electrodes or human-subject data. It
uses BrainFlow's synthetic board plus an independently generated, labelled synthetic motor-imagery
dataset.

Track A delivers an installable Python 3.11 package with:

1. Board-agnostic BrainFlow acquisition for the synthetic board, plus a configuration-compatible
   Cyton profile that can be exercised without claiming hardware validation.
2. Incremental acquisition that consumes each new sample once, causal stateful preprocessing, and a
   downstream sample-indexed rolling-window assembler.
3. A parallel filter bank for broadband, motor-imagery, mu, and beta signals, plus a separate
   zero-phase offline analysis mode.
4. Stable feature contracts for Welch band power, Hilbert amplitude and phase, relative-phase
   summaries, phase-locking value (PLV), spectral entropy, and regularized covariance matrices.
5. Two typed decoder baselines:
   - Riemannian OAS covariance → SPD regularization → tangent space → shrinkage LDA.
   - Stable temporal-feature vector → standardization → shrinkage LDA.
6. A reproducible synthetic motor-imagery generator and leakage-resistant cross-validation gate.
7. An explicit model lifecycle: validate → train → save → inspect → load → contract-check → infer.
8. Live time-series and PSD visualization, headless PNG/JSON output, deterministic recording/replay,
   and a complete smoke/performance workflow.
9. An optional, batch-calibrated Riemannian recentering profile. Track A does not claim that
   pyRiemann's TLCenter is an incremental online-learning primitive.

The corrected design locks the following decisions:

- The causal path uses BrainFlow's destructive get_board_data() call to obtain new, non-overlapping
  chunks. get_current_board_data() is diagnostic-only and must not feed a stateful filter.
- Every sample passes through each causal filter exactly once. Overlap is created only after
  filtering by a rolling window assembler.
- At 250 Hz, the default window is 250 samples and the default hop is 64 samples (256 ms). The
  earlier 0.25-second hop is 62.5 samples and therefore cannot be an exact fixed-sample cadence.
- Riemannian and vector decoders have distinct, declared input adapters.
- A live decoder must be trained with the same causal preprocessing contract used at inference.
- No prediction occurs before filter warm-up, complete-window availability, model validation, and
  fitted-estimator validation.
- All acceptance criteria below use numeric thresholds and fixed fixtures.

Passing Track A proves software plumbing and known-structure classification. It does not prove that
real motor imagery is separable, that the temporal-code thesis is correct, or that Cyton hardware has
been validated.

## 2. Background & Problem Statement

Real EEG development combines at least three independent failure domains:

- electrical contact and acquisition quality;
- biological task separability and subject variability;
- software correctness across filtering, windowing, features, models, and visualization.

Starting with all three makes failures ambiguous. BrainFlow's synthetic board removes the physical
acquisition variable while preserving the same acquisition API family intended for Cyton. A separate
synthetic motor-imagery generator provides labelled data with controlled class structure, which the
fixed synthetic-board waveform does not.

The predecessor Track A draft established the correct broad scope but cannot safely drive
implementation as written. The audit found three blocking contradictions:

1. It passed overlapping snapshots through a stateful causal filter, which would process repeated
   samples multiple times and corrupt filter state.
2. It defined one generic decoder input even though the Riemannian model requires epoch tensors and
   the vector model requires two-dimensional feature matrices.
3. It invoked prediction without defining how a fitted model is produced, persisted, loaded, and
   verified.

It also left multi-band ownership, feature-vector ordering, adaptation semantics, filter thresholds,
synthetic-data parameters, latency percentiles, persistence metadata, clean-package installation, and
CI gates underspecified. This specification resolves those issues before production code exists.

## 3. Goals

- **G1 — Installable foundation.** Ship a clean-wheel-installable Python 3.11 package with declared
  CLI entry points, locked dependencies, static checks, and Windows/Linux CI.
- **G2 — Correct acquisition.** Resolve board facts at runtime, consume each new sample once, preserve
  timestamps, detect gaps, and release BrainFlow resources on every exit path.
- **G3 — Correct streaming DSP.** Implement chunk-invariant causal filtering with persistent state,
  parallel band branches, explicit warm-up, and sample-indexed window emission.
- **G4 — Offline analysis.** Provide a separate zero-phase filter path using the same coefficient
  design without mislabelling it as live-model compatible.
- **G5 — Temporal feature contract.** Produce deterministic named vectors and temporal arrays for band
  power, Hilbert envelope/phase, relative phase, PLV, spectral entropy, and diagnostic covariance.
- **G6 — Typed decoder baselines.** Implement one epoch-tensor Riemannian baseline and one flat-vector
  baseline without shape ambiguity or duplicate covariance ownership.
- **G7 — Synthetic validity gate.** Generate a fixed, reproducible two-class motor-imagery fixture and
  require both decoders to exceed explicit cross-validation thresholds without trial leakage.
- **G8 — Model lifecycle.** Train, atomically persist, inspect, hash, load, and reject incompatible
  model artifacts before a board starts streaming.
- **G9 — Complete observable pipeline.** Run acquisition/replay → filtering → windows → features →
  inference → metrics in both live and headless modes.
- **G10 — Reproducible data path.** Record raw selected EEG plus interpretation metadata and replay it
  through the same SampleSource contract.
- **G11 — Real-time headroom.** Meet p50, p95, and deadline-miss gates over a fixed replay workload.
- **G12 — Honest scope.** Clearly separate software evidence from future real-hardware and
  neurophysiological evidence.

## 4. Non-Goals

- **N1:** Connecting to or validating a physical Cyton in Track A. Static Cyton metadata and
  configuration compatibility are in scope; prepare_session() against hardware is not.
- **N2:** Real human motor-imagery accuracy, clinical efficacy, medical diagnosis, safety
  certification, or validation of the project's neuroscience thesis.
- **N3:** Analog front-end, electrode, enclosure, custom-silicon, impedance, or dry-electrode design.
- **N4:** EMG, jaw clench, eye movement, or another muscle/ocular signal as a control channel. EOG may
  be recorded later as an artifact observation, never as a Track A command input.
- **N5:** Deep neural networks, EEGNet, GPU dependencies, language-model output, or fluent-composition
  layers.
- **N6:** Per-window classifier-weight updates, pseudo-labelling, generic partial_fit(), or a claim of
  continuously co-adaptive learning.
- **N7:** LSL, cloud sync, network services, accounts, telemetry, or multi-user functionality.
- **N8:** A desktop GUI shell, installer, hosted deployment, or production distribution.
- **N9:** Loading model files obtained from untrusted people, websites, or network locations.

## 5. Repository Findings

Repository inspection on 2026-07-10 produced the following evidence.

| Finding | Evidence | Design consequence |
| --- | --- | --- |
| The repository has no implementation. | Apart from .venv/, specs/, and empty .agents/.codex/.git placeholder directories, there is no chrono_link package, src/, tests/, scripts/, README, pyproject.toml, lock file, CI, or artifacts. | Track A starts from scaffolding; no compatibility layer is required. |
| The directory is not a valid Git repository. | An empty .git directory exists, but git rev-parse and git status both report that this is not a Git repository. | Phase 0 initializes valid Git metadata and creates an initial baseline commit if execution is authorized. |
| A predecessor spec exists. | specs/chrono-link-track-a-software-foundation.md | Preserve it; do not silently rewrite historical intent. |
| Python 3.11 is provisioned. | .venv/pyvenv.cfg and the interpreter report Python 3.11.9. The machine default is Python 3.14.3. | Every documented command uses .venv or an explicit Python 3.11 interpreter. |
| The scientific stack imports and pip check is clean. | brainflow 5.22.2, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, pyriemann 0.12, matplotlib 3.11.0. | These versions are the Track A reference stack and model-artifact compatibility boundary. |
| Test tooling is absent. | importlib cannot find pytest. | Dev/test/static/build dependencies must be declared and locked during scaffolding. |
| Synthetic BrainFlow streaming works locally. | ID -1, 250 Hz, 16 EEG rows 1..16, C3 at row 2, C4 at row 4, timestamp row 30, 32 rows total. | A real synthetic-board integration test is available without hardware. |
| Synthetic timestamps are burst-timed, not uniformly spaced. | A live 299-sample probe produced contiguous package counters but 76 timestamp deltas above 6 ms, with approximately four samples sharing each delivery burst. | Never infer sample loss from a universal 1/fs timestamp-spacing threshold. Use package-counter continuity; timestamp spacing is diagnostic. |
| Cyton static metadata differs. | ID 0, 250 Hz, EEG rows 1..8, names Fp1, Fp2, C3, C4, P7, P8, O1, O2. C3 is row 3. | Never reuse synthetic row numbers. Resolve name-to-row mapping for each selected board. |
| Cyton requires more than a board ID. | BrainFlow requires serial_port for Cyton. | “One-line board-ID swap” is replaced by “configuration-only swap”; live hardware remains unproven. |
| OAS alone is not strict-SPD protection. | Local pyRiemann 0.12 verification returned a zero covariance matrix for a constant epoch. | Reject invalid windows and apply a documented eigenvalue floor before manifold operations. |
| pyRiemann adaptation APIs are batch-oriented. | TLCenter requires target_domain and fit(X, y_enc); it has no partial_fit. TangentSpace(tsupdate=True) is documented as not online compatible. | Use tsupdate=False. Define optional session calibration as a distinct batch lifecycle. |

## 6. Research Notes & References

The following facts and recommendations are normative for implementation.

### 6.1 Acquisition

- BrainFlow returns board data as a two-dimensional array with rows representing EEG, timestamps,
  package counters, and other board fields. Board-specific row metadata must be queried through
  BoardShim. [BrainFlow User API](https://brainflow.readthedocs.io/en/stable/UserAPI.html)
- get_current_board_data(n) returns the latest available samples without removing them.
  get_board_data() returns accumulated data and removes it from the internal ring buffer. The causal
  pipeline therefore uses get_board_data(); a non-destructive snapshot is permitted only for
  diagnostics that do not mutate DSP state.
- BrainFlow's API is board-agnostic, but connection parameters remain board-specific. Cyton requires
  serial_port. [BrainFlow supported boards — Cyton](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#cyton)
- BrainFlow arrays are channel-major: (board_rows, samples). The package preserves this orientation
  internally and names every public shape.

### 6.2 Filtering and spectral analysis

- scipy.signal.sosfilt is the causal primitive. When zi is supplied it returns final state zf, which
  must become the next chunk's state. SOS form reduces numerical problems relative to high-order
  transfer-function form. [SciPy 1.17 sosfilt](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.signal.sosfilt.html)
- scipy.signal.sosfilt_zi provides a step-steady-state initializer. For input shaped (channels,
  samples), per-channel state is shaped (sections, channels, 2).
  [SciPy 1.17 sosfilt_zi](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.signal.sosfilt_zi.html)
- scipy.signal.sosfiltfilt is forward/backward and non-causal; its pad length must be smaller than the
  sample dimension. [SciPy 1.17 sosfiltfilt](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.signal.sosfiltfilt.html)
- scipy.signal.iirnotch returns transfer-function coefficients; convert each notch to SOS before
  composing it with the causal chain. Q is center frequency divided by the -3 dB bandwidth.
  [SciPy 1.17 iirnotch](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.signal.iirnotch.html)
- Welch PSD configuration is part of the feature schema rather than a library-default accident.
  [SciPy 1.17 welch](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.signal.welch.html)

### 6.3 Riemannian decoding and calibration

- pyRiemann Covariances accepts epoch tensors shaped (epochs, channels, samples) and returns covariance
  matrices shaped (epochs, channels, channels). OAS is a supported estimator.
  [pyRiemann 0.12 Covariances](https://pyriemann.readthedocs.io/en/v0.12/generated/pyriemann.estimation.Covariances.html)
- TangentSpace expects SPD matrices, learns a reference during fit(), and maps an n-by-n SPD matrix to
  n(n+1)/2 features. Track A sets tsupdate=False.
  [pyRiemann 0.12 TangentSpace](https://pyriemann.readthedocs.io/en/v0.12/generated/pyriemann.tangentspace.TangentSpace.html)
- TLCenter performs batch/domain recentering and requires encoded domain labels. Its fit_transform()
  semantics differ from fit().transform(); it is not an incremental streaming estimator.
  [pyRiemann 0.12 TLCenter](https://pyriemann.readthedocs.io/en/v0.12/generated/pyriemann.transfer.TLCenter.html)
- LDA shrinkage is supported with lsqr and eigen solvers. Track A uses lsqr with shrinkage="auto".
  [scikit-learn 1.9 LDA](https://scikit-learn.org/1.9/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

### 6.4 Model persistence and packaging

- joblib is pickle-based and can execute arbitrary code during loading. Cross-version scikit-learn
  loading is unsupported. Model artifacts must be local/trusted and must record exact dependency and
  training metadata. [scikit-learn 1.9 model persistence](https://scikit-learn.org/1.9/model_persistence.html)
- Track A uses a src/ package layout so tests cannot accidentally import an uninstalled working-tree
  package. [PyPA src-layout discussion](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- pyproject.toml contains build-system and project metadata; console commands are declared under
  project.scripts. [PyPA pyproject.toml guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

### 6.5 Scientific interpretation boundary

Riemannian covariance models, band power, and phase connectivity are established BCI feature
families, but Track A does not use literature to claim real-subject performance. Synthetic
classification is a controlled software test. PLV is retained as a first-class relative-phase
feature; absolute phase is exposed as temporal diagnostic data and is not blindly flattened into the
LDA vector.

## 7. Users, Use Cases & User Stories

The primary user is the Chrono Link R&D developer. CI is a secondary operational user.

- As the developer, I can install the package into a clean environment and run every public workflow
  from outside the source tree.
- As the developer, I can stream the synthetic board and see C3/C4 traces plus PSD without possessing
  EEG hardware.
- As the developer, I can switch from the synthetic profile to a Cyton profile through configuration
  without editing source, while receiving an honest warning that hardware has not been validated.
- As the developer, I can process irregular BrainFlow chunk sizes without duplicate filtering or
  duplicate windows.
- As the developer, I can inspect stable, named temporal features and know their precise ordering and
  units.
- As the developer, I can validate both decoder families on known synthetic structure before creating
  a loadable model.
- As the developer, I can record a session, replay it with arbitrary chunk boundaries, and obtain
  identical windows and deterministic results.
- As the developer, I can run in plot-only mode when no trained model exists.
- As CI, I can exercise acquisition, filtering, features, decoders, artifacts, packaging, and
  performance without a display or physical board.

## 8. Functional Requirements

### 8.1 Configuration and board resolution

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-001 | Separate requested configuration from ResolvedBoardInfo. Sampling rate, EEG rows, names, timestamp row, and board row count are runtime-derived facts, never mutable defaults. | Must |
| FR-002 | Provide named profiles synthetic and cyton. Synthetic is the default. Cyton requires serial_port before prepare_session(). | Must |
| FR-003 | Resolve decode channels by exact case-sensitive 10-20 name in requested order. Missing names fail closed with requested and available names; no implicit numeric fallback is permitted. | Must |
| FR-004 | Validate all bands against the resolved Nyquist frequency and validate window/hop sample counts before streaming starts. | Must |
| FR-005 | Config serialization must round-trip without executable code and must reject unknown keys by default. Environment overrides are limited to explicitly documented keys. | Should |

### 8.2 Acquisition and source lifecycle

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-006 | Define a SampleSource protocol with prepare(), start(), read_new(), stop(), and release() and explicit lifecycle states NEW, PREPARED, STREAMING, STOPPED, RELEASED, FAILED. | Must |
| FR-007 | BrainFlowSource.read_new() must use get_board_data() to return only samples not previously consumed. get_current_board_data() must not feed a stateful processing path. | Must |
| FR-008 | Return continuity-bounded SampleChunk objects with EEG, timestamps, local sequence, resolved channel names, markers, board package counters when exposed, and zero or more SourceEvent values that must be applied before the chunk. A board read containing a discontinuity is split at that boundary before DSP sees it. | Must |
| FR-009 | Preserve raw selected EEG before filtering for recording and visualization. Select decode channels through resolved indices, never hardcoded rows. | Must |
| FR-010 | Detect sample loss primarily through board package-counter discontinuity (modulo 256 for the synthetic and Cyton profiles), explicit replay discontinuity, or buffer-overrun signals where available. Timestamp spacing is jitter telemetry only and must not trigger a reset by itself. Non-finite/non-monotonic timestamps remain invalid. A streaming source with no new samples for 2.0 seconds is a source-stall runtime error. | Must |
| FR-011 | stop() and release() are idempotent and execute at most once. stop() executes exactly once only if streaming started; release() executes exactly once only if preparation acquired a session. This holds on normal exit, Ctrl+C, board, plotting, and callback errors. | Must |

### 8.3 Filtering and windowing

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-012 | Design all IIR filters from resolved fs. Defaults: scipy.signal.butter N=2 for the 0.5 Hz high-pass (final order 2), Q=30 second-order mains notches, and scipy.signal.butter N=4 for each bandpass (four SOS sections; final digital bandpass order 8). Store SOS coefficients and a deterministic filter fingerprint. | Must |
| FR-013 | Generate notch frequencies mains_hz × k only while frequency is strictly below Nyquist. Defaults: 50 and 100 Hz for 50 Hz mains; 60 and 120 Hz for 60 Hz mains at fs=250. | Must |
| FR-014 | CausalFilterBank must process every new chunk once, preserve independent SOS state per branch/channel, and emit cleaned plus broadband (1–40), mi (8–30), mu (8–12), and beta (13–30) arrays. | Must |
| FR-015 | Causal output must be invariant to how the same chronological signal is chunked. Initial state uses sosfilt_zi scaled to the first sample; reset() clears all state. | Must |
| FR-016 | Mark the first 1,000 samples (4.0 s at 250 Hz) after start/reset/gap as unsettled. No training example, prediction, or performance measurement may include unsettled samples. | Must |
| FR-017 | OfflineFilterBank must use the same coefficient design with sosfiltfilt, validate pad length, and mark its output preprocessing_mode="offline_zero_phase". Such output cannot create a live-compatible model. | Must |
| FR-018 | WindowAssembler must maintain filtered sample-indexed ring buffers and emit a first window after 250 settled samples, then at endpoints advanced by exactly 64 samples. It must emit multiple windows for a large chunk and none for insufficient data. | Must |
| FR-019 | Every WindowBundle must contain aligned branches, timestamps, start/end sequence, channel order, fs, and data-quality flags. No window may be repeated or span a gap/reset boundary. | Must |

### 8.4 Feature contracts

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-020 | Welch band power uses a periodic Hann window, nperseg=min(250, samples), 50% overlap, density scaling, constant detrend, and frequency-bin integration. Log power is 10×log10(max(power, 1e-12)). | Must |
| FR-021 | Hilbert processing returns full amplitude and wrapped phase arrays for each configured band/channel plus a valid central slice that excludes 25 samples from each window edge for summaries. | Must |
| FR-022 | For each ordered channel pair i<j, define phase_difference = phase_i - phase_j and produce its mean cosine, mean sine, and PLV = abs(mean(exp(j×phase_difference))) over the valid slice. | Must |
| FR-023 | Normalized spectral entropy uses only the inclusive 1–40 Hz bins of the broadband Welch PSD, divides by log(number_of_included_bins), returns a finite value in [0,1], and defines a zero-power signal as entropy 0. | Must |
| FR-024 | FeatureVector must contain exactly named values in the stable order defined in Section 10.6. Values and names have equal length and are fingerprinted into model artifacts. | Must |
| FR-025 | TemporalFeatures exposes band/channel/sample amplitude and phase arrays separately; raw instantaneous phase is not silently flattened into FeatureVector. | Must |
| FR-026 | The Riemannian covariance path owns OAS covariance estimation. After covariance estimation, symmetrize and floor eigenvalues to max(1e-12, 1.01e-6 × lambda_max) before tangent-space operations. | Must |
| FR-027 | Before covariance/features, reject a decode channel as flat when std(channel) ≤ max(1e-12, 1e-8 × max(abs(channel))) in native units. Windows with non-finite values, any flat channel, failed Cholesky, or condition number above 1.01e6 after regularization are counted and never predicted. | Must |

### 8.5 Synthetic data and persistence

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-028 | SyntheticMISessionGenerator must implement the fixed two-class trial recipe in Section 10.8, accept an explicit seed/RNG, and return raw trials, labels, trial IDs, timestamps, and generator metadata. | Must |
| FR-029 | Same seed and configuration must produce array-identical data/labels within the pinned environment; different seeds must change data while preserving shapes and class balance. | Must |
| FR-030 | SessionRecorder must atomically write schema-versioned NPZ without object arrays/pickle. Required fields are raw data, timestamps, sequence, package counters, channel names/order, units, fs, board ID, creation UTC, markers, quality/gap/reset events, and canonical config JSON. Recording requires a finite max_samples and uses bounded temporary disk-backed storage rather than retaining the session in RAM. | Must |
| FR-031 | ReplaySource must implement SampleSource, support deterministic or configured chunk sizes, reject corrupt/incompatible records, and reproduce source samples exactly. | Must |

### 8.6 Decoders, validation, and models

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-032 | BoundDecoder must declare input_kind ("epoch" or "vector") and own an input adapter. fit/predict/predict_proba accept WindowBatch rather than an ambiguous ndarray. | Must |
| FR-033 | RiemannTSLDA must select the mi epoch tensor and use Covariances("oas") → SPDRegularizer → TangentSpace(metric="riemann", tsupdate=False) → LDA(solver="lsqr", shrinkage="auto"). | Must |
| FR-034 | TemporalVectorLDA must build the stable FeatureVector and use StandardScaler → LDA(solver="lsqr", shrinkage="auto"). | Must |
| FR-035 | Prediction before fit raises sklearn NotFittedError. Probabilities must be finite, shaped (examples, classes), lie in [0,1], and sum to one within 1e-12. | Must |
| FR-036 | chrono-validate must run the exact three-seed grouped cross-validation protocol in Section 17, emit strict JSON plus human-readable results, and never save/overwrite a model. | Must |
| FR-037 | chrono-train must run or consume a passing validation report, fit the selected final model on all eligible training windows, create an immutable model ID, and atomically save a ModelBundle. | Must |
| FR-038 | ModelBundle must contain bound_decoder.joblib, self_test.npz, and manifest.json with schema/model ID, SHA-256 for both payloads, decoder/input kind, exact channels/order, fs, window/hop, filters and preprocessing fingerprint, feature fingerprint, class order, exact chrono-link distribution version, dependency versions, source commit/wheel hash when available, generator/dataset recipe, seed, and CV metrics. The persisted object is the complete BoundDecoder, including its adapter and fitted estimator. | Must |
| FR-039 | ModelStore must inspect and validate the manifest and both payload hashes before joblib loading, reject dependency/contract mismatches, and load only explicitly trusted local artifacts. | Must |
| FR-040 | The live pipeline may run plot-only without a model. Decoding requires --model, and model validation must finish before BrainFlow start_stream(). | Must |
| FR-041 | Optional Riemannian session calibration must use the CALIBRATING batch lifecycle and numeric guards in Section 10.10. It must not expose generic partial_fit(), update LDA weights, or claim TLCenter is incremental. | Should |

### 8.7 Runtime, visualization, and CLI

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-042 | RealtimePipeline must orchestrate source → recorder fan-out → causal filter bank → window assembler → quality gate → bound decoder → callbacks/metrics with explicit state transitions. | Must |
| FR-043 | Live visualization must show rolling selected-channel time series and PSD, reuse existing artists/axes, and remain usable in plot-only mode. | Must |
| FR-044 | Headless mode must select Agg before pyplot import, write a decodable PNG of at least 640×480 whose normalized RGB pixel standard deviation is ≥0.01 and which contains at least 64 unique RGB colors, and write strict JSON with no NaN/Infinity. | Must |
| FR-045 | Provide installed commands chrono-stream, chrono-validate, chrono-train, and chrono-smoke plus equivalent python -m chrono_link subcommands. main(argv=None) returns an integer and --help performs no board connection or display creation. | Must |
| FR-046 | chrono-smoke must build or load fixed-seed local models for both decoder kinds, run the locked hop-sized replay workload through each, emit artifacts, and require both 200-window runs to pass the performance gates. Internally generated smoke models are trusted by construction. | Must |
| FR-047 | All CLIs use the exit-code contract in Appendix D and log structured failure context without exposing raw biosignal samples by default. | Must |

## 9. Non-Functional Requirements

| ID | Requirement | Acceptance |
| --- | --- | --- |
| NFR-001 | Real-time compute headroom | For each decoder, after the 1,000-sample DSP warm-up and 20 unmeasured benchmark windows, process 200 measured hop-sized steps: p50 ≤20 ms, p95 ≤50 ms, and zero total step computations ≥256 ms. Waiting, plotting, recording, and process startup are excluded; replay read/copy, filtering, window/quality, adapter/features, and prediction are included. |
| NFR-002 | Sample correctness | Arbitrary-chunk causal output matches one-shot causal output at rtol 1e-10, atol 1e-11; reference window slices match exactly. |
| NFR-003 | Determinism | Every stochastic public API requires a seed or Generator. Fixed fixtures use seed 20260709; decoder validation uses seeds 7, 42, and 20260709. |
| NFR-004 | CPU-only | No Torch, TensorFlow, CuPy, JAX, GPU runtime, or device selection dependency. |
| NFR-005 | Platform support | Unit, integration, package, and headless suites pass on Windows and Linux with Python 3.11. |
| NFR-006 | Modularity | Acquisition, DSP, windows, features, models, recording, and visualization import independently. Core signal/model modules do not import matplotlib or BrainFlow. |
| NFR-007 | Reproducibility | Exact reference runtime versions and all transitive runtime/dev dependencies are recorded in lock files; a clean install passes pip check. |
| NFR-008 | Installability | Build sdist and wheel; install wheel into a clean environment from outside the repository; run imports and all CLI help/headless workflows. |
| NFR-009 | Code quality | Ruff, mypy, and public API docstring checks pass. Branch coverage is ≥85% overall and ≥90% for acquisition, filters, windows, features, recording, decoders, calibration, and model_store. |
| NFR-010 | Privacy | recordings/, artifacts/, and models/ are root-gitignored. Recording is opt-in. No raw samples appear in normal logs or metrics. |
| NFR-011 | Numeric safety | Core arrays are float64. Non-finite data is rejected at boundaries. JSON is RFC 8259-compatible strict JSON with allow_nan=False. |
| NFR-012 | Resource safety | Board, file, figure, and temporary-artifact resources close on every tested failure path; final artifacts never expose partial writes. |
| NFR-013 | Honest evidence | Hardware-free tests may claim Cyton configuration compatibility only, not a successful Cyton stream or real EEG validity. |

## 10. Proposed Solution

### 10.1 Repository and package layout

~~~text
chrono-link/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── locks/
│   ├── windows-py311-runtime.txt
│   ├── windows-py311-dev.txt
│   ├── linux-py311-runtime.txt
│   └── linux-py311-dev.txt
├── specs/
│   ├── chrono-link-track-a-software-foundation.md
│   └── chrono-link-track-a-software-foundation-implementation-spec.md
├── src/
│   └── chrono_link/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py
│       ├── contracts.py
│       ├── acquisition.py
│       ├── recording.py
│       ├── filters.py
│       ├── windows.py
│       ├── features.py
│       ├── covariance.py
│       ├── synthetic_mi.py
│       ├── decoders.py
│       ├── calibration.py
│       ├── model_store.py
│       ├── pipeline.py
│       ├── metrics.py
│       ├── viz.py
│       └── cli/
│           ├── __init__.py
│           ├── stream.py
│           ├── validate.py
│           ├── train.py
│           └── smoke.py
└── tests/
    ├── fixtures/
    ├── unit/
    ├── integration/
    ├── performance/
    └── package/
~~~

Use PEP 621 metadata and setuptools with src-layout package discovery. The project requires Python
>=3.11,<3.12 and declares compatible direct-dependency bounds: brainflow==5.22.2, numpy>=2.4,<2.5,
scipy>=1.17,<1.18, scikit-learn>=1.9,<1.10, pyriemann>=0.12,<0.13,
matplotlib>=3.11,<3.12, and joblib>=1.5,<1.6. A dev extra contains pytest, pytest-cov, Ruff, mypy,
build, pip-tools, and metadata-validation tooling.

The four platform-role lock files are generated by pip-compile on their matching operating system and
pin every transitive version and artifact hash. pyproject.toml also pins exact setuptools and wheel
versions in build-system.requires; those exact versions appear in both dev locks. A clean runtime
wheel check first runs python -m pip install --require-hashes -r
locks/<platform>-py311-runtime.txt and then python -m pip install --no-deps <wheel>. CI builds with
python -m build --no-isolation inside the matching locked dev environment. Lock files, not compatible
project-metadata ranges, define reproducibility.

### 10.2 Configuration and resolved facts

Requested values and board facts are distinct immutable objects.

~~~python
@dataclass(frozen=True)
class BoardConfig:
    profile: Literal["synthetic", "cyton"] = "synthetic"
    serial_port: str | None = None
    decode_channels: tuple[str, ...] = ("C3", "C4")
    record_channels: tuple[str, ...] | Literal["all"] = "all"

@dataclass(frozen=True)
class SignalConfig:
    window_samples: int = 250
    hop_samples: int = 64
    warmup_samples: int = 1000
    mains_hz: Literal[50, 60] = 50
    notch_q: float = 30.0
    dc_highpass_hz: float = 0.5
    flat_abs_tol: float = 1e-12
    flat_rel_tol: float = 1e-8
    filter_bands: tuple[BandSpec, ...] = DEFAULT_FILTER_BANDS  # broadband, mi, mu, beta
    vector_feature_bands: tuple[str, ...] = ("mu", "beta")

@dataclass(frozen=True)
class ResolvedBoardInfo:
    board_id: int
    fs: int
    eeg_names: tuple[str, ...]
    eeg_rows: tuple[int, ...]
    timestamp_row: int
    package_num_row: int | None
    marker_row: int | None
    num_rows: int
    decode_rows: tuple[int, ...]
    record_rows: tuple[int, ...]
~~~

Signal durations are derived from integer sample counts and resolved fs. At 250 Hz, 250 samples are
1.0 s and 64 samples are 0.256 s. The model manifest stores both counts and derived durations.

Optional configuration files are strict UTF-8 JSON and reject unknown keys. Precedence is
command-line flag > environment > JSON file > named-profile/default value. The only Track A
environment keys are CHRONO_BOARD, CHRONO_SERIAL_PORT, CHRONO_MAINS_HZ, and CHRONO_OUTPUT_ROOT;
values use the same validators as JSON/CLI input. An unknown CHRONO_ key is a configuration error so
misspellings cannot silently fall back.

### 10.3 Acquisition contract

~~~python
@dataclass(frozen=True)
class SourceEvent:
    code: Literal["PACKAGE_GAP", "REPLAY_GAP", "RESET", "STALL"]
    before_sequence: int
    details: Mapping[str, str | int | float]

@dataclass(frozen=True)
class SampleChunk:
    eeg: NDArray[np.float64]          # (record_channels, new_samples)
    timestamps: NDArray[np.float64]   # (new_samples,)
    sequence: NDArray[np.int64]       # (new_samples,)
    package_counter: NDArray[np.int64] | None  # (new_samples,), board-native
    channel_names: tuple[str, ...]
    fs: int
    markers: NDArray[np.float64] | None
    events_before: tuple[SourceEvent, ...]

class SampleSource(Protocol):
    def prepare(self) -> ResolvedBoardInfo: ...
    def start(self) -> None: ...
    def read_new(self) -> SampleChunk | None: ...
    def stop(self) -> None: ...
    def release(self) -> None: ...
~~~

BrainFlowSource is the single owner of BoardShim. It drains accumulated new samples with
get_board_data(), immediately copies the selected rows, and assigns a monotonically increasing local
sequence. Other components receive fan-out copies/references from this owner; they never call
BoardShim independently.

The poll cadence is not the window cadence. A poll may return zero, fewer than one hop, one hop, or
many hops. Correctness is based on sequence counts, not sleep timing.

For synthetic and Cyton, adjacent package counters must satisfy
(current - previous) mod 256 == 1, including across chunks. A violation declares a data gap and resets
DSP/window state. BrainFlow synthetic timestamps reflect delivery batching and are not expected to
advance by exactly 1/fs; record their deltas as jitter metrics but do not infer loss from delta size.
An empty read is normal until the continuous no-sample duration reaches the 2.0-second stall timeout.

If one get_board_data() result contains a counter discontinuity, BrainFlowSource splits it into
maximal contiguous SampleChunk objects and queues them in order. The first post-gap chunk carries a
PACKAGE_GAP event with before_sequence equal to its first local sequence. RealtimePipeline applies
events and resets state before processing that chunk. SessionRecorder persists these events;
ReplaySource restores the same boundary and cross-checks recorded events against counters.

### 10.4 Causal filter bank

The filter bank processes only decode channels:

~~~text
new chronological chunk
    │
    ├── stateful 0.5 Hz DC high-pass
    │
    ├── stateful mains notch cascade (50/100 or 60/120)
    │       └── cleaned branch
    │
    ├── 1–40 Hz broadband branch
    ├── 8–30 Hz MI branch
    ├── 8–12 Hz mu branch
    └── 13–30 Hz beta branch
~~~

Each branch uses SOS and state shaped (sections, channels, 2). For an input x shaped
(channels, samples), initialize each stage/branch from its own input with
zi = sosfilt_zi(sos)[:, None, :] * x[:, 0][None, :, None]. Later chunks use the previous zf. A reset
occurs on explicit restart, detected gap, channel/config change, or fatal numeric error.

~~~python
@dataclass(frozen=True)
class FilteredChunk:
    cleaned: NDArray[np.float64]                # (decode_channels, new_samples)
    bands: Mapping[str, NDArray[np.float64]]    # each (decode_channels, new_samples)
    timestamps: NDArray[np.float64]
    sequence: NDArray[np.int64]
    settled: NDArray[np.bool_]
~~~

Feature extraction never designs or reapplies filters. OfflineFilterBank is a separate object and
must not share mutable state with CausalFilterBank.

### 10.5 Sample-indexed window assembly

WindowAssembler maintains synchronized ring buffers for every filtered branch and metadata.

Let W=250 and H=64:

- The first eligible endpoint is W settled samples after warm-up/reset.
- Later endpoints are W+H, W+2H, and so on in local settled-sequence coordinates.
- Each emitted window contains [endpoint-W, endpoint).
- If one input chunk crosses several endpoints, push() returns all corresponding windows in order.
- An empty or short chunk produces no output.
- A gap clears ring buffers and restarts warm-up; no window bridges the gap.

~~~python
@dataclass(frozen=True)
class WindowBundle:
    cleaned: NDArray[np.float64]                # (channels, W)
    bands: Mapping[str, NDArray[np.float64]]    # each (channels, W)
    timestamps: NDArray[np.float64]             # (W,)
    start_sequence: int
    end_sequence: int
    channel_names: tuple[str, ...]
    fs: int
    quality: WindowQuality
~~~

This architecture is required by the two most important regression tests:

1. Filtering a continuous signal in arbitrary chunks matches filtering it in one causal call.
2. Emitted windows equal exact reference slices at endpoints W, W+H, W+2H, with no duplicate filter
   calls for overlapping samples.

Batching stacks only quality-approved WindowBundle objects with identical contracts:

~~~python
@dataclass(frozen=True)
class WindowBatch:
    cleaned: NDArray[np.float64]                # (examples, channels, W)
    bands: Mapping[str, NDArray[np.float64]]    # each (examples, channels, W)
    channel_names: tuple[str, ...]
    fs: int
    sequence_ranges: NDArray[np.int64]          # (examples, 2), [start, end)
    quality: tuple[WindowQuality, ...]           # length examples
    trial_ids: NDArray[np.int64] | None          # (examples,), synthetic/training only
~~~

WindowBatch construction rejects mixed fs, channel order, W, branch names, unsettled windows, and
failed quality flags. Trial IDs are never required for live inference but are mandatory for grouped
synthetic validation.

### 10.6 Feature schema

Filter branches and vector-feature bands are separate contracts. DEFAULT_FILTER_BANDS contains
broadband, mi, mu, and beta; the default vector_feature_bands contains only mu and beta in that order.
Channel order is resolved decode-channel order. Pair order is lexicographic by channel position i<j,
not alphabetic channel name.

For every band and channel, append:

1. {band}.{channel}.log_power_db
2. {band}.{channel}.envelope_mean
3. {band}.{channel}.envelope_std

For every band and channel pair, append:

1. {band}.{channel_i}.{channel_j}.phase_cos
2. {band}.{channel_i}.{channel_j}.phase_sin
3. {band}.{channel_i}.{channel_j}.plv

Finally, for every channel append:

1. broadband.{channel}.spectral_entropy

For C3/C4 with mu/beta, the default vector has 20 values:

- 12 per-channel band values;
- 6 pairwise phase values;
- 2 entropy values.

~~~python
@dataclass(frozen=True)
class FeatureVector:
    values: NDArray[np.float64]  # (features,)
    names: tuple[str, ...]       # same length
    fingerprint: str

@dataclass(frozen=True)
class TemporalFeatures:
    amplitude: NDArray[np.float64]  # (bands, channels, W)
    phase: NDArray[np.float64]      # (bands, channels, W), [-pi, pi]
    valid_slice: slice              # 25 : W-25
~~~

Feature fingerprint is SHA-256 over canonical JSON containing names, band order/ranges, Welch
parameters, Hilbert guard, power floor, and implementation schema version.

### 10.7 Decoder input adapters

~~~python
class DecoderInputAdapter(Protocol):
    input_kind: Literal["epoch", "vector"]
    def transform(self, windows: WindowBatch) -> NDArray[np.float64]: ...

@dataclass
class BoundDecoder:
    adapter: DecoderInputAdapter
    estimator: ClassifierMixin
    def fit(self, windows: WindowBatch, y: NDArray[np.int64]) -> Self: ...
    def predict(self, windows: WindowBatch) -> NDArray[np.int64]: ...
    def predict_proba(self, windows: WindowBatch) -> NDArray[np.float64]: ...
~~~

RiemannEpochAdapter returns mi epochs shaped (examples, channels, 250). Covariance estimation occurs
inside its fitted estimator pipeline exactly once.

TemporalVectorAdapter returns vectors shaped (examples, 20). StandardScaler and LDA remain inside the
fold/final fitted estimator.

SPDRegularizer is a package-defined, importable sklearn transformer. It symmetrizes each covariance,
performs eigh, floors eigenvalues with max(1e-12, 1.01e-6 × lambda_max), reconstructs the matrix, and
validates Cholesky and condition ≤1.01e6. It must not be a lambda or script-local object because the
complete estimator is persisted. Its direct unit test accepts a degenerate matrix to prove numeric
hardening; RealtimePipeline's earlier signal-quality gate still rejects a flat EEG window.

### 10.8 Synthetic motor-imagery recipe

The normative validation dataset contains 160 independent raw trials, exactly 80 per class.

| Parameter | Value |
| --- | --- |
| fs | 250 Hz |
| trial duration | 6.0 s |
| filter warm-up prefix | first 4.0 s, never scored |
| scored duration | final 2.0 s, passed through the real W=250/H=64 assembler |
| scored window starts | trial-relative samples 1000, 1064, 1128, and 1192 (four windows/trial) |
| channels | C3, C4 |
| class 0 mu RMS amplitude | C3=0.501, C4=1.0 |
| class 1 mu RMS amplitude | C3=1.0, C4=0.501 |
| nominal mu frequency | one frequency sampled uniformly from 9–11 Hz per trial and shared by C3/C4 |
| phase | independent uniform [0, 2pi) per trial/channel/component |
| beta nuisance | equal 20 Hz component, RMS 0.25 per channel with random phase |
| noise | equal-power normalized mixture of independent white and 1/f noise, defined below |
| requested SNR | -6.0 dB relative to the high-amplitude mu component |
| validation seeds | 7, 42, 20260709 |

For each trial/channel, generate white noise from independent standard-normal samples. Generate pink
noise by drawing an independent real-signal rFFT spectrum, setting DC to zero, multiplying every
positive-frequency coefficient by 1/sqrt(frequency_hz), enforcing the real Nyquist coefficient, and
applying irfft. Subtract each component's mean and normalize each separately to RMS 1 over the full
trial. Combine (white + pink)/sqrt(2), renormalize the mixture to RMS 1, and scale it to
10^(-SNR_dB/20) = 1.995262... RMS for SNR=-6 dB relative to the RMS-1 high mu component. Noise
realizations are independent by trial and channel.

The generator randomizes trial order after exact balancing. The 6 dB lateralized mu contrast is the
only label-dependent rule. Trial ID is the cross-validation group so all four windows from one trial
remain in the same fold.

Each raw trial is processed causally from a reset state. The first four seconds satisfy the exact live
warm-up contract and are dropped. The normal WindowAssembler then emits the four scored windows at the
locked starts above, leaving 58 trailing samples. This produces deployable causal training examples
without resetting a filter at an overlapping scored window.

The generator measures pre-filter realized SNR from the explicit stored mu/noise components and
post-filter lateralized mu contrast from scored windows. Realized SNR must be within 0.1 dB of the
request and measured class-direction mu contrast must be at least 4 dB.

### 10.9 Validation, training, and ModelBundle

~~~text
generate/load trials
    → causal preprocess each trial
    → discard warm-up
    → create grouped WindowBatch
    → grouped fold split
    → fold-local adapter/estimator fit
    → aggregate out-of-fold metrics
    → validation gate
    → final fit on all eligible windows
    → immutable atomic ModelBundle save
~~~

chrono-validate stops after the validation report. chrono-train may either run validation itself or
consume a report whose dataset/config/fingerprint hashes exactly match the current training request.
It refuses to train from a failed/stale report.

ModelBundle layout:

~~~text
models/<model-id>/
├── manifest.json
├── self_test.npz
└── bound_decoder.joblib
~~~

Creation writes a temporary sibling directory, fsyncs files where supported, computes SHA-256 for the
bound decoder and self-test payloads, writes strict manifest JSON last, and atomically renames into a
previously nonexistent final directory. Existing model IDs are never overwritten.

The loader:

1. Reads and schema-validates manifest.json without unpickling.
2. Verifies bound-decoder and self-test SHA-256 values.
3. Compares exact chrono-link distribution, NumPy, SciPy, scikit-learn, pyRiemann, joblib, feature,
   preprocessing, channel/order, fs, W, H, and class contracts. Python must match major.minor; the
   complete training/runtime patch versions remain recorded for diagnostics.
4. Requires an explicit trusted-local acknowledgement.
5. Loads and schema-validates self_test.npz with allow_pickle=False.
6. Loads bound_decoder.joblib.
7. Reconstructs the self-test WindowBatch and requires an array-equal class plus probabilities
   matching the manifest at rtol=1e-12 and atol=1e-12.

### 10.10 Optional session recentering

Track A's optional adaptation is a batch calibration state, not continuous per-window learning.

The recentered Riemannian profile is trained as follows:

1. Estimate and regularize source covariance matrices.
2. Compute the affine-invariant Riemannian source mean.
3. Whiten source covariances by congruence with the inverse square root of that mean.
4. Fit TangentSpace(tsupdate=False) and LDA on the centered source covariances.
5. Store the source center and calibration contract in ModelBundle.

At session start:

1. Complete causal filter warm-up.
2. Collect exactly 32 valid, non-overlapping, unlabeled MI-band windows with a 250-sample calibration
   stride within 120 seconds after warm-up; overlapping H=64 inference windows are not calibration
   observations.
3. Estimate and validate the target Riemannian center.
4. Shadow-compute centered covariances and validate finite values, SPD, condition ≤1.01e6, and
   distance_riemann(source_center, target_center) ≤5.0.
5. Atomically activate the target center for that session.
6. Keep the target center fixed until reset; do not update LDA or use predicted labels.

If calibration fails or does not collect 32 valid windows within 120 seconds, a calibration-required
model produces no predictions, writes diagnostics, and exits 3 after cleanup. A normal fixed model
remains the default Track A artifact. Reset occurs on model, board, sampling-rate, channel-order,
preprocessing, gap, or session changes.

Acceptance requires the deterministic drift fixture to reduce Riemannian-mean distance to identity by
at least 90% while preserving pairwise affine-invariant distances within rtol 1e-6. It does not
require or claim improved human-subject accuracy.

### 10.11 Runtime state and data flow

~~~text
NEW
  → PREPARED
  → STREAMING
      ├──→ PLOT_ONLY_READY                                  # immediate
      ├──→ STREAMING_WARMUP → INFERENCE_READY               # fixed model
      └──→ STREAMING_WARMUP → CALIBRATING → INFERENCE_READY # calibrated model
  → STOPPED
  → RELEASED

Any active state
  → FAILED
  → cleanup
  → RELEASED
~~~

Core processing is synchronous. BrainFlow already owns its acquisition thread. RealtimePipeline.step()
drains new data, processes any returned chunk, and emits zero or more windows. Interactive
matplotlib's timer calls step(); headless mode uses a normal loop. No second consumer may drain the
BrainFlow buffer.

Plot callbacks may render raw/available filtered data in every STREAMING branch, including model
warm-up and calibration. Warm-up gates model windows and predictions, not visualization; therefore a
two-second plot-only run does not wait for the four-second DSP warm-up.

Prediction callbacks receive class, probabilities, window sequence range, model ID, quality flags,
and stage timings. They do not receive raw EEG unless an explicitly privileged local callback is
registered.

## 11. API, Interface, CLI, or UX Contract

### 11.1 Illustrative public Python API

~~~python
cfg = ChronoConfig.synthetic()

with BrainFlowSource(cfg.board) as source:
    info = source.prepare()
    runtime = ResolvedRuntime.from_config(cfg, info)
    filters = CausalFilterBank(runtime)
    windows = WindowAssembler(runtime)
    source.start()
    while not stop_requested():
        chunk = source.read_new()
        if chunk is None:
            continue
        for window in windows.push(filters.process(chunk)):
            feature_frame = FeatureExtractor(runtime).transform(window)

report = validate_synthetic(decoder="riemann", seeds=(7, 42, 20260709))
bundle = train_model(cfg, validation_report=report)

model = ModelStore.load(
    bundle.path,
    runtime_contract=runtime.model_contract,
    trusted_local=True,
)
~~~

The exact implementation may refine names, but shapes, ownership, lifecycle, and failure behavior are
normative.

### 11.2 CLI commands

**chrono-stream**

~~~text
chrono-stream [--config config.json] [--board synthetic|cyton] [--serial-port PORT]
              [--seconds N] [--channels C3,C4]
              [--model models/<id> --trust-model] [--headless] [--out-dir artifacts/run]
              [--record recordings/session.npz]
chrono-stream --replay recordings/session.npz [same model/output flags]
~~~

- Without --model: plot/record only; decoded output is absent, not fabricated.
- With --model: require --trust-model and validate the model before starting the source.
- --board defaults through CLI/environment/JSON/profile precedence; cyton requires --serial-port or
  the equivalent environment/JSON value.
- --replay is mutually exclusive with --board, --serial-port, and --record and never prepares a board.
- --record requires a finite positive --seconds value, capped at 3,600 seconds in Track A.
- --headless writes signal.png and metrics.json.

**chrono-validate**

~~~text
chrono-validate --decoder riemann|vector|both
                --seeds 7,42,20260709
                --out artifacts/validation.json
~~~

- Runs the normative generator/CV gate.
- Never writes a model.
- Exits 4 if any mandatory accuracy/leakage gate fails.

**chrono-train**

~~~text
chrono-train --decoder riemann|vector
             [--validation-report artifacts/validation.json]
             --out-dir models/
~~~

- Revalidates report hashes or performs validation.
- Writes one immutable ModelBundle.
- Prints model ID/path only after atomic completion.

**chrono-smoke**

~~~text
chrono-smoke --out-dir artifacts/smoke [--windows N]
~~~

- Uses deterministic replay and a fitted fixed-seed test model.
- Measures the complete compute path and emits signal.png, metrics.json, and predictions.json.
- Rejects --windows below 200 with configuration exit 2.

**Module form**

~~~text
python -m chrono_link stream ...
python -m chrono_link validate ...
python -m chrono_link train ...
python -m chrono_link smoke ...
~~~

## 12. Data Model & Persistence

### 12.1 SessionRecord schema version 1

NPZ fields:

| Field | Type/shape | Meaning |
| --- | --- | --- |
| schema_version | int scalar | Must equal 1 |
| data | float64 (channels, samples) | Raw selected BrainFlow EEG values |
| timestamps | float64 (samples,) | Board timestamps |
| sequence | int64 (samples,) | Local monotonically increasing sequence |
| package_counter | int64 (samples,) | Board counter, or -1 when the source has none |
| channel_names | fixed-width Unicode (channels,) | Exact recorded order; no object dtype |
| units | fixed-width Unicode (channels,) | Explicit units or brainflow_native if unknown |
| fs | int scalar | Resolved sampling rate |
| board_id | int scalar | Actual/master board ID |
| created_utc | Unicode scalar | RFC 3339 UTC timestamp |
| markers | float64 (samples,) | Marker channel, zero when absent |
| config_json_utf8 | uint8 (bytes,) | Canonical strict JSON bytes |
| events_json_utf8 | uint8 (bytes,) | Canonical list of quality, package-gap, reset, and stall events |

Loading uses numpy.load(..., allow_pickle=False). The loader rejects missing/extra required fields,
object arrays, mismatched lengths, non-finite timestamps/data, duplicate/out-of-order sequence,
unsupported schema, invalid JSON, and channel-name/data-row mismatch.

SessionRecorder requires max_samples at construction. The CLI derives it from --seconds × resolved fs
and refuses recording above 3,600 seconds. Before starting, it checks sufficient free space for the
final NPZ plus temporary data with a 20% margin. Chunks append into a preallocated temporary memmap;
both temporary paths are siblings of the destination on the same filesystem. Finalization creates the
NPZ at the second temporary path and atomically renames it. Disk-full or finalization failure
closes/removes both temporaries and leaves no final file.

The default path is recordings/<UTC>_<board>_<short-id>.npz. Writes never overwrite.

### 12.2 ModelBundle manifest schema version 1

The manifest is strict JSON. Besides FR-038 metadata, it records:

- created_utc and training duration;
- exact chrono-link distribution version, repository commit when available, and installed wheel
  SHA-256 when available;
- validation report SHA-256;
- generator or input-record hashes;
- model persistence warning and trusted-local requirement;
- self-test payload hash, expected class/probabilities, and probability tolerance;
- compatible package schema range;
- calibration mode: none or batch_riemann_center;
- human-readable limitations.

Persisted sklearn/pyRiemann objects are not forward-compatible promises. A version mismatch is a hard
load error followed by retraining instructions.

self_test.npz is non-pickled and contains one quality-approved example: cleaned plus every required
band array shaped (1, channels, W), channel_names, fs, sequence_ranges, and a trial_ids sentinel. The
manifest stores expected class/probabilities and rtol=atol=1e-12. The loader validates its schema/hash
before joblib loading and reconstructs a normal WindowBatch rather than bypassing the adapter.

### 12.3 Metrics schema

metrics.json includes schema version, config/model IDs, window counts, dropped reasons, gaps/resets,
per-stage p50/p95/max milliseconds, deadline misses, source duration, and artifact paths. It contains
no raw EEG samples and uses strict JSON.

## 13. Security, Privacy & Abuse Considerations

- **Biosignal privacy:** Treat synthetic recordings with the same local/private defaults intended for
  future real EEG. Recording is opt-in; paths are gitignored; no automatic uploads or telemetry.
- **Unsafe model deserialization:** joblib loading can execute code. The CLI refuses implicit model
  discovery and requires an explicit path plus trusted-local acknowledgement. Documentation states
  never to load downloaded/untrusted artifacts.
- **Artifact integrity:** Verify SHA-256 before unpickling. Atomic/versioned writes prevent partial
  models or recordings from appearing valid.
- **Input validation:** Reject malformed configuration, bands outside Nyquist, invalid sample counts,
  mismatched model contracts, corrupt NPZ, non-finite arrays, and unsafe output collisions.
- **Path handling:** Resolve output paths, create only within the user-selected/local project path by
  default, and never overwrite an existing recording/model unless a future explicit migration
  operation defines it.
- **No network surface:** Track A has no LSL, sockets, cloud API, auth, or remote model loading.
- **Hardware safety boundary:** Track A makes no electrical-safety claim because no person-connected
  hardware is used or tested. Future hardware work requires its own safety specification.
- **Research-use communication:** CLI/README state that output is experimental research software, not
  a medical device or diagnostic result.

## 14. Error Handling & Edge Cases

| Scenario | Required behavior |
| --- | --- |
| Source has fewer than one complete window | Continue collecting; emit no window and no padding. |
| get_board_data() returns empty | Return None/no-op; do not advance filter/window state. |
| Large chunk contains several hops | Process once and emit every eligible window in sequence order. |
| Package-counter discontinuity or explicit replay gap | Count gap, reset causal filters/windows/warm-up, record event, suppress cross-gap windows. |
| Large positive timestamp delta with continuous package counters | Record timestamp-jitter telemetry only; do not reset or drop data. |
| Non-monotonic/duplicate timestamp | Reject affected chunk, count error, reset; fatal after three consecutive invalid chunks. |
| No new samples for 2.0 seconds while STREAMING | Stop and release source, record stall duration, exit 3. |
| NaN/Inf in a chunk | Drop chunk, count reason, reset state; no prediction. |
| Requested channel absent | Fail during prepare with requested and available names; do not silently substitute. |
| Cyton profile lacks serial port | Configuration exit 2 before prepare_session(). |
| Band edge ≤0 or ≥Nyquist | Configuration exit 2 with band, fs, and valid range. |
| sosfiltfilt input too short | Raise actionable OfflineWindowTooShort; never silently switch to causal. |
| Flat EEG epoch by FR-027 tolerance | Reject before feature/covariance work, count flat channel, and make no prediction. SPDRegularizer's degenerate-matrix unit behavior does not override this runtime gate. |
| Predict before fit | Raise NotFittedError in Python API; CLI maps to runtime failure. |
| Model hash mismatch | Reject before joblib load. |
| Model dependency/feature/config mismatch | Reject before starting source; show all mismatched fields. |
| Untrusted model flag absent | Refuse joblib load with security guidance. |
| Model calibration fails | No prediction for calibration-required profile; preserve diagnostics and clean shutdown. |
| Calibration has fewer than 32 valid windows after 120 seconds | Write calibration diagnostics, stop/release, and exit 3 without predictions. |
| PNG/JSON/model/record target exists | Refuse overwrite unless command contract explicitly provides a separate replace flag; models never replace. |
| Atomic write fails | Remove temporary path; leave no final path. |
| Recording requested without finite --seconds, above 3,600 seconds, or without estimated disk space | Configuration exit 2 before starting the board. |
| Recording disk fills despite preflight | Stop recording/pipeline safely, remove temporary/final partial paths, release source, exit 3. |
| Ctrl+C | Exit 0 after controlled stop if no prior error; release board/figures/files. |
| Board/runtime failure | Log context, stop if streaming and release if prepared, with each call at most once; exit 3. |
| Validation below gate | Write report with failures, exit 4, do not save model. |

## 15. Observability & Operations

- Use standard logging with concise INFO lifecycle events and opt-in DEBUG stage details.
- Call BoardShim.disable_board_logger() during normal startup. A debug flag explicitly enables the
  BrainFlow board logger instead.
- Attach run_id, model_id, board profile, and window sequence to structured log context.
- Counters: chunks, raw samples, windows emitted, predictions, unsettled windows, gaps, resets,
  non-finite drops, flat-window drops, covariance failures, source empty reads, deadline misses.
- Timers: acquisition copy, causal filtering, window/quality, feature/adapter, prediction, total
  compute. Percentiles use NumPy's linear method.
- chrono-smoke and chrono-validate always write reports on both success and gate failure when the
  output path is writable.
- CI uploads validation JSON, smoke metrics, PNG, prediction report, coverage, and test reports on
  success or failure.
- There are no production dashboards, services, alerts, or remote operations in Track A.

## 16. Implementation Plan

Each phase must pass its narrow gate before the next phase begins.

### Phase 0 — Version control and scaffolding

- If git rev-parse still fails, initialize valid Git metadata; preserve both specifications.
- Add src-layout pyproject.toml, README, CHANGELOG, lock files, .gitignore, package skeleton, test
  skeleton, and CI skeleton.
- Declare installed commands and verify clean wheel installation plus --help from outside the repo.
- Gate: static metadata validation, pip check, import checks, and package tests pass.

### Phase 1 — Contracts, configuration, and replay-first fixtures

- Implement immutable contracts, exception taxonomy, configuration parsing/validation, exit codes,
  ResolvedBoardInfo, fixed clock/RNG injection, and deterministic ReplaySource.
- Implement SessionRecord schema and atomic NPZ round-trip before live acquisition.
- Gate: config negative tests and exact 625-sample SessionRecord → ReplaySource sample/chunk parity
  pass. Window-reference assertions begin in Phase 3.

### Phase 2 — BrainFlow acquisition

- Implement BrainFlowSource lifecycle, dynamic channel resolution, new-chunk draining, timestamp/gap
  validation, and guaranteed cleanup.
- Add fake-BoardShim unit tests and real synthetic-board integration tests.
- Gate: irregular chunk shapes, package-counter continuity/wrap, failure cleanup, C3/C4 resolution,
  and a 2-second source-only capture pass.

### Phase 3 — Causal/offline DSP and window assembler

- Implement filter design/fingerprinting, shared causal preprocessing, parallel branches, state reset,
  offline zero-phase mode, warm-up, and sample-indexed windows.
- Gate: exact response thresholds, arbitrary-chunk invariance, no duplicate sample processing, gap
  reset, reference-window tests, and a real synthetic source → causal filter → first settled window
  integration with a 6.5-second timeout pass.

### Phase 4 — Feature layer and quality gates

- Implement Welch power, Hilbert temporal output/summaries, relative phase/PLV, entropy, stable vector
  names/fingerprint, covariance regularization, and invalid-window rejection.
- Gate: every numeric feature fixture in Section 17 passes.

### Phase 5 — Synthetic MI and decoders

- Implement normative generator, trial-group preprocessing, WindowBatch/adapters, both decoders, and
  cross-validation reporting.
- Gate: reproducibility/leakage tests and both three-seed decoder gates pass.

### Phase 6 — Model lifecycle

- Implement chrono-validate, chrono-train, immutable atomic ModelBundle, manifest/hash/self-test,
  trusted/version-checked loading, and mismatched-contract rejection.
- Gate: train/save/load prediction parity and all negative security/compatibility tests pass.

### Phase 7 — Runtime, visualization, and smoke path

- Implement RealtimePipeline, plot-only/model modes, live time series/PSD, Agg export, CLI contracts,
  strict metrics, and chrono-smoke.
- Gate: AC-001 through AC-010 and performance gates pass on the reference Windows environment.

### Phase 8 — Optional batch recentering

- Implement only after the fixed-model path is green.
- Add recentered model profile, calibration collection/state, safety guards, reset behavior, and drift
  fixture.
- Gate: AC-011 passes. Do not block the fixed Track A foundation if this Should item is explicitly
  deferred in the release report.

### Phase 9 — Cross-platform/package completion

- Run Windows/Linux CI matrices, clean-wheel commands, coverage/static gates, docs examples, and final
  evidence report.
- Mark each requirement with test/report evidence. Do not declare Cyton hardware or human EEG proven.

## 17. Testing Strategy

### 17.1 Locked test profile

- Python 3.11.9 reference, float64, fs=250.
- Channels C3, C4.
- Window 250 samples; hop 64 samples; warm-up 1,000 samples.
- Default deterministic seed 20260709.
- Filters: high-pass butter N=2/final order 2; bandpasses butter N=4/four SOS sections/final order
  8; notch Q=30/final order 2 per notch.
- Welch: periodic Hann, nperseg=250, noverlap=125, nfft=250, density scaling.
- Power floor 1e-12.

### 17.2 Numeric DSP and feature fixtures

| Area | Fixture and gate |
| --- | --- |
| DC high-pass response | Evaluate the one-pass causal design with sosfreqz at exact frequencies: 0.5 Hz is -3.01 ±0.15 dB, 0.1 Hz attenuation is ≥27 dB, and 2 Hz loss is ≤0.05 dB. |
| Bandpass response | Evaluate the one-pass causal designs with sosfreqz at exact frequencies. Mu: interior 9/10/11 Hz loss ≤0.5 dB; stop probes 4/20 Hz attenuation ≥25 dB. Beta: interior 16/20/25 Hz loss ≤0.5 dB; stop probes 6/50 Hz ≥25 dB. Broadband: interior 2/10/30 Hz loss ≤0.5 dB; stop probes 0.25/70 Hz ≥25 dB. Each one-pass cutoff is -3.01 ±0.15 dB; the corresponding zero-phase forward/backward magnitude is -6.02 ±0.30 dB. MI: interior 10/20/27 Hz loss ≤1.5 dB; stop probes 3/60 Hz ≥25 dB. |
| Notch response | Evaluate exact frequencies after TF-to-SOS conversion. At each configured notch center attenuation ≥40 dB. Fundamental 50/60 Hz loss at ±5 Hz ≤0.25 dB. Harmonic 100/120 Hz loss at valid ±5 Hz probes ≤1.5 dB. |
| Notch signal | Eight-second equal-amplitude 10+50 Hz signal: ≥30 dB PSD suppression at 50 Hz and ≤0.5 dB change at 10 Hz after one-second edge crop. |
| Causal chunking | Filtering the same 20-second seeded signal as one chunk and with the repeating chunk-size cycle [1,7,64,3,511,2,128,19], truncated at the end, matches at rtol 1e-10, atol 1e-11. Every input sequence appears once. |
| Offline phase | Zero-phase 10 Hz output phase error ≤0.02 rad after one-second edge crop. |
| Warm-up | No window/prediction contains any sample marked unsettled; reset repeats the 1,000-sample suppression. |
| Band power | Four seconds of sqrt(2)×sin(10 Hz) + 0.25×sqrt(2)×sin(20 Hz): mu power 1.0 and beta power 0.0625, each within 5%; dB formula within 1e-10. |
| Hilbert | Unit-test the Hilbert primitive directly on an already band-limited four-second amplitude-2 10 Hz sine, bypassing the causal filter. Against analytic phase 2pi×10t-pi/2 and after the normative 25-sample guard: mean amplitude within 2%, amplitude-error p95 ≤0.05, unwrapped frequency within 0.02 Hz, circular phase-error p95 ≤0.02 rad. A separate integration test checks causal output against the causal filter's measured complex response rather than raw-sine absolute phase. |
| PLV | Fixed-offset 10 Hz pair PLV ≥0.995. Independent uniform phase arrays of length 5,000 PLV ≤0.05. Independent seeded 8–12 Hz band-limited noise over 20 s, one-second edge crop, PLV ≤0.20. For 256 independent one-second windows using the actual 200-sample valid slice, finite-sample bias is expected: median PLV must lie in [0.25,0.55] and p95 must be ≤0.90; the implementation must not treat a single short-window PLV as asymptotically near zero. Positive amplitude scaling changes PLV by ≤1e-12. |
| Entropy | Use only inclusive 1–40 Hz PSD bins and normalize by log(40) at the locked 1 Hz resolution. Output is finite in [0,1]. On an actual settled W=250 window, a 10 Hz sine is ≤0.25 and independently generated broadband-filtered white noise is ≥0.85 for each seed 7, 42, and 20260709; zero signal equals 0. |
| Covariance | The SPDRegularizer unit fixture accepts collinear and constant matrices and produces finite symmetric matrices (normalized symmetry error ≤1e-12), successful Cholesky after flooring, and condition ≤1.01e6. The runtime quality gate separately rejects signals at the exact FR-027 flatness boundary before prediction. |
| Windowing | WindowAssembler unit test: mark seeded 2×625 samples as already settled. W=250/H=64 emits six windows starting at local indices 0,64,128,192,256,320, each array-equal to its reference slice. End-to-end replay is tested separately with raw warm-up. |

### 17.3 Synthetic MI and decoder gate

Use StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed). Group is trial ID. All learned
transforms, tangent references, scalers, and classifiers fit only on training folds. Each trial's four
window probabilities are averaged before choosing its predicted class; fold and aggregate metrics are
computed over the resulting 160 independent trial-level predictions, not 640 correlated windows.

For both decoders and each seed 7, 42, and 20260709:

- mean balanced accuracy ≥0.80;
- no fold balanced accuracy <0.70;
- grand mean across seeds ≥0.85;
- aggregate trial-level out-of-fold exact binomial test against p=0.5 has p<1e-6;
- 25 trial-label permutations generated from seed+10,000 through seed+10,024 have 95th-percentile
  trial-level balanced accuracy ≤0.65; all four windows retain their trial's permuted label;
- no trial ID occurs in both train and test for a fold;
- rerunning the same seed produces the same splits and scores within 1e-12 in the pinned environment.

chrono-validate passes a threshold exactly equal to 0.80 and exits 4 below it.

### 17.4 Model and persistence tests

- Train/save/load both decoder types; class predictions are array-equal and probabilities agree within
  rtol 1e-12, atol 1e-12.
- Mutation of either payload is rejected before joblib.load is called.
- Wrong dependency version, fs, W/H, channel order, preprocessing fingerprint, feature fingerprint,
  decoder kind, schema, or trust flag is rejected.
- Self-test mismatch fails load.
- Injected write failure leaves no final model directory.
- NPZ round-trip is array-equal with allow_pickle=False; corrupt/missing/object-dtype inputs fail.
- End-to-end replay of 1,625 raw samples applies 1,000 warm-up samples and emits six windows starting
  at raw sequences 1000,1064,1128,1192,1256,1320; each scored slice matches the corresponding
  continuous causal reference.
- Gap fixture: generate 3,200 logical samples with package_counter=(250+logical_index) mod 256, omit
  logical sample 1300, and record the remaining 3,199 samples plus the PACKAGE_GAP before local
  sequence 1300. Replay it as one large read, fixed 64-sample reads, and the repeating chunk cycle.
  Every run must produce one identical gap/reset, restart the full 1,000-sample warm-up, emit identical
  sequence ranges, accept all legitimate 255→0 wraps, and emit zero windows spanning sequence 1300.
  A counter/event disagreement makes the recording corrupt and fails before DSP.

### 17.5 Runtime and performance tests

Run separate workloads for the Riemannian and vector decoders. Use a preloaded ReplaySource and the
decoder's fitted fixed-seed model. The first step supplies exactly 1,250 samples (the 1,000-sample DSP
warm-up plus the first W=250 window). Then supply 20 unmeasured H=64 benchmark-warm-up steps followed
by 200 measured H=64 steps. Every post-prime step must emit exactly one window, so its read/filter/
window/adapter/predict work is attributed to that window without amortizing one huge chunk.

Include replay read/copy, filter, window/quality, feature/adapter, and predict. Exclude deliberate wait,
matplotlib rendering, recording IO, logging IO, model load, and process startup.

- both decoder workloads have p50 total ≤20 ms;
- both decoder workloads have p95 total ≤50 ms;
- both decoder workloads have zero totals ≥256 ms;
- zero duplicate/missing output sequences;
- zero non-finite predictions;
- strict metrics JSON reports per-stage p50/p95/max and deadline misses.

### 17.6 Visualization and CLI tests

- Under MPLBACKEND=Agg, PNG decodes, is at least 640×480, has normalized RGB pixel standard deviation
  ≥0.01, and contains at least 64 unique RGB colors.
- Plot contains two named channel traces plus a PSD axis; updates reuse axes/artists.
- Every --help exits 0 without board preparation or display.
- Configuration errors exit 2; runtime errors exit 3; validation gate failures exit 4; model
  incompatibility exits 5.
- Clean wheel install from outside the repository runs all module/console forms.
- From that clean install, a model-backed replay without --trust-model exits 5 before joblib.load;
  the same trusted local bundle with --trust-model completes at least one finite prediction.

### 17.7 CI jobs

1. **static:** locked dev install, pip check, Ruff, mypy, docstrings, metadata validation.
2. **unit:** Windows/Linux Python 3.11 unit matrix with coverage.
3. **integration:** real BrainFlow synthetic board, headless rendering, recording/replay.
4. **decoder:** both decoders, all three seeds, permutation control.
5. **performance:** fixed replay workload and percentile/deadline gates.
6. **package:** build sdist/wheel, clean install outside source tree, imports, help, smoke.
7. **artifacts:** upload reports and images even on failure.

## 18. Rollout, Migration & Rollback

There is no deployed user base.

- **Specification rollout:** Preserve the predecessor. After review, link README to this document as
  Track A's canonical implementation spec.
- **Code rollout:** Implement phase-by-phase on Git with a passing narrow gate at each phase.
- **Model rollout:** Models are immutable versioned directories. Selecting a previous model ID is
  rollback; artifacts are never migrated in place.
- **Recording rollout:** SessionRecord schema starts at version 1. Unknown versions fail with a
  migration-required message; recordings are never overwritten.
- **Dependency upgrade:** Any reference-stack change requires lock regeneration, full validation,
  model retraining, and a manifest schema/compatibility decision.
- **Synthetic-to-Cyton migration:** Track A proves only construction/configuration compatibility.
  Physical connection, serial discovery, signal quality, channel montage, safety, and real-data
  acceptance belong to a later hardware spec.
- **Rollback:** Revert the relevant Git commit and select the last passing model/artifact. Raw
  recordings remain additive.

## 19. Documentation Updates

- README:
  - exact Python 3.11 setup and locked install;
  - synthetic quickstart;
  - plot-only, validate, train, model-backed stream, replay, and smoke examples;
  - strict statement of what synthetic success does not prove.
- docs or README architecture note:
  - why acquisition drains new chunks;
  - why filtering precedes overlap;
  - causal versus zero-phase compatibility;
  - decoder input shapes and model lifecycle.
- Model security note:
  - joblib trusted-local boundary and version limitations.
- Data privacy note:
  - recordings are sensitive/local by default.
- Cyton configuration note:
  - serial_port requirement and unvalidated-hardware warning.
- CHANGELOG:
  - Track A foundation release and evidence summary.
- Public modules/classes/functions:
  - type hints, shape-aware docstrings, exceptions, and small runnable examples.

## 20. Acceptance Criteria

- [x] **AC-001 — Clean package:** sdist/wheel build succeeds; clean Python 3.11 wheel installation
      from outside the repository passes pip check, independent imports, and every --help command.
- [x] **AC-002 — Acquisition:** fake-board lifecycle/error tests, a real two-second source capture,
      and a real synthetic source → causal filter → first settled window within 6.5 seconds pass;
      dynamic C3/C4 mapping and exact cleanup are proven.
- [x] **AC-003 — One-pass streaming:** arbitrary-chunk causal equality and exact W=250/H=64 reference
      windows pass with no repeated sample processing or cross-gap windows.
- [x] **AC-004 — Filters:** every response, signal, state, warm-up, and offline-phase numeric gate in
      Section 17.2 passes.
- [x] **AC-005 — Features:** band-power, Hilbert, PLV, entropy, stable 20-name vector, covariance
      regularization, and invalid-window gates pass.
- [x] **AC-006 — Decoder validity:** both baselines pass all three-seed grouped CV, minimum-fold,
      binomial, permutation, and leakage gates.
- [x] **AC-007 — Model lifecycle:** validate/train separation, immutable atomic save, manifest/hash,
      trust/version/config rejection, self-test, and prediction-parity tests pass.
- [x] **AC-008 — Live/headless UX:** installed chrono-stream runs synthetic plot-only for at least two
      seconds; headless mode exits 0 and emits a valid PNG plus strict metrics JSON.
- [x] **AC-009 — Recording/replay:** NPZ exact round-trip, corrupt-input negatives, atomic-failure
      behavior, the 625-sample already-settled WindowAssembler unit fixture, and the 1,625-raw-sample
      six-window end-to-end replay fixture plus the 3,199-sample wrap/gap/reset fixture pass.
- [x] **AC-010 — Smoke/performance:** installed chrono-smoke produces 200 measured inference windows
      for each decoder with finite predictions, p50 ≤20 ms, p95 ≤50 ms, and zero ≥256 ms deadlines.
- [ ] **AC-011 — Optional calibration:** if delivered in Track A, deterministic congruence drift is
      recentered by ≥90%, distances are preserved within rtol 1e-6, invalid calibration rolls back,
      classifier weights remain unchanged, and an injected clock proves the 120-second insufficient-
      window timeout exits 3 without predictions.
- [x] **AC-012 — Quality/CI:** Windows and Linux Python 3.11 jobs, Ruff, mypy, coverage thresholds,
      strict JSON, resource-cleanup negatives, and package tests all pass.
- [x] **AC-013 — Scope report:** final evidence explicitly marks Cyton streaming, real EEG, real motor
      imagery, human safety, and neuroscience efficacy as untested.

Track A fixed-foundation completion requires AC-001 through AC-010, AC-012, and AC-013. AC-011 is
required only if the optional calibration profile is included in the Track A release.

Local evidence on 2026-07-16 satisfies AC-001 through AC-010 and AC-013. AC-011 is explicitly
deferred. AC-012 is satisfied by pull request #2 GitHub Actions run 29522621249: the Windows and Linux
Python 3.11 test/coverage/smoke jobs and the isolated package job all passed on the hardened
implementation commit `66a5d3c`.

## 21. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
| --- | --- | --- | --- |
| Synthetic accuracy is mistaken for biological validity | High | High | Scope statements, grouped/permutation controls, and AC-013. |
| Repeated samples corrupt causal state | High | Medium | Single get_board_data owner, one-pass DSP, chunk-invariance/reference-window tests. |
| Offline/live preprocessing mismatch | High | Medium | Manifest preprocessing mode; live loader rejects zero-phase models; causal validation recipe. |
| Synthetic generator and decoder are co-tuned | High | Medium | Locked public recipe, multiple seeds, fold groups, permutation control, two decoder families. |
| Degenerate covariance reaches tangent space | Medium | Medium | Flat-window gate, eigenvalue floor, Cholesky/condition checks. |
| Untrusted joblib executes code | High | Low under local scope | Explicit path/trust flag, hash and manifest before load, documentation, no network discovery. |
| Library/API drift invalidates models | Medium | Medium | Exact locks, manifest versions, fail-closed load, retraining after upgrade. |
| Performance tests are flaky | Medium | Medium | Large headroom, fixed replay, exclude IO/waits, p50/p95 plus hard deadline, report raw timings. |
| Cyton compatibility is overstated | Medium | High | Configuration-only claim; physical acceptance deferred. |
| Window/hop duration expectations drift | Medium | Low | Store integer counts as authority and derived seconds in config/model/metrics. |
| Batch calibration harms predictions | Medium | Medium | Optional/default-off, separate profile, minimum batch, shadow validation, fixed session center, rollback. |
| Recordings leak into version control | High | Low | Root-anchored gitignore, opt-in recording, privacy docs, CI check. |

## 22. Open Questions

These questions do not block useful implementation because defaults are locked below.

| Question | Default decision | Change impact |
| --- | --- | --- |
| Should this spec become canonical after review? | Yes; predecessor remains historical. | README/spec links only. |
| Is a 64-sample/256 ms hop acceptable instead of nominal 250 ms? | Yes; exact fixed-sample cadence wins. | Model/window contract and all fixtures. |
| Should optional batch recentering ship with the fixed foundation? | Implement after AC-001–010 are green; may be explicitly deferred. | AC-011 and Phase 8 only. |
| Should physical Cyton validation be appended to Track A later? | No; create a hardware-track spec. | Avoids changing Track A evidence semantics. |
| Is 50 Hz the correct mains default? | Yes for the current locale; 60 Hz remains a tested config. | Configuration default only. |

## 23. Assumptions

- **A1:** The reference development environment remains Python 3.11 with the verified library
  versions until a deliberate lock update.
- **A2:** Synthetic and Cyton profiles both resolve to 250 Hz in BrainFlow 5.22.2.
- **A3:** C3/C4 are the default decode channels and their BrainFlow 10-20 names are available on both
  selected profiles.
- **A4:** Integer sample counts are the authoritative timing representation.
- **A5:** BrainFlow EEG values are preserved without unit conversion in raw recordings; units metadata
  records the board/native interpretation.
- **A6:** Local model artifacts are generated and consumed by the same trusted user.
- **A7:** CPU-only Riemannian and LDA models provide ample Track A latency headroom.
- **A8:** Synthetic MI gates are software tests only and do not estimate future human performance.
- **A9:** The current empty .git placeholder may be initialized into valid local Git metadata during
  implementation; no remote, push, release, or publish action is implied.

## 24. Appendix

### Appendix A — Canonical defaults

| Key | Default |
| --- | --- |
| Python | 3.11.9 |
| Board profile | synthetic |
| Decode channels | C3, C4 |
| fs | runtime resolved; expected 250 Hz |
| Window | 250 samples (1.0 s at 250 Hz) |
| Hop | 64 samples (0.256 s at 250 Hz) |
| Warm-up | 1,000 samples (4.0 s at 250 Hz) |
| Calibration timeout | 120 s after warm-up |
| Source stall timeout | 2.0 s without a new sample |
| Mains | 50 Hz |
| Notch Q | 30 |
| DC high-pass | 0.5 Hz, butter N=2, final order 2 |
| Broadband | 1–40 Hz, butter N=4, four SOS sections, final order 8 |
| MI covariance band | 8–30 Hz, butter N=4, four SOS sections, final order 8 |
| Mu | 8–12 Hz, butter N=4, four SOS sections, final order 8 |
| Beta | 13–30 Hz, butter N=4, four SOS sections, final order 8 |
| Hilbert guard | 25 samples per edge |
| Flat-channel tolerance | std ≤ max(1e-12, 1e-8 × max(abs(channel))) |
| Covariance estimator | OAS + max(1e-12, 1.01e-6 × lambda_max) eigenvalue floor |
| RNG default | 20260709 |
| Decoder validation seeds | 7, 42, 20260709 |

### Appendix B — Canonical shapes

| Object | Shape |
| --- | --- |
| BrainFlow raw | (board_rows, new_samples) |
| SampleChunk.eeg | (record_channels, new_samples) |
| FilteredChunk branch | (decode_channels, new_samples) |
| WindowBundle branch | (decode_channels, 250) |
| WindowBatch MI epochs | (examples, decode_channels, 250) |
| TemporalFeatures | (bands, decode_channels, 250) |
| FeatureVector default | (20,) |
| Vector decoder input | (examples, 20) |
| Covariance | (examples, decode_channels, decode_channels) |
| Two-channel tangent vector | (examples, 3) |
| predict_proba | (examples, 2) |

### Appendix C — Predecessor requirement mapping

| Predecessor area | This specification |
| --- | --- |
| Board wrapper/sliding snapshots | FR-001–011; corrected to new-chunk drain plus downstream windowing |
| Notch/bandpass/offline/online | FR-012–019 with exact ownership, warm-up, and thresholds |
| Band power/Hilbert/PLV/entropy/covariance | FR-020–027 with stable shapes/names and SPD floor |
| Synthetic MI | FR-028–029 and normative grouped recipe |
| Recording/replay | FR-030–031, promoted to Must |
| Generic decoder/adapt | FR-032–041 with typed adapters, model lifecycle, and batch calibration |
| Plot/headless/CLI | FR-042–047 with installed/module commands and exit codes |
| LSL | Explicitly deferred |

### Appendix D — CLI exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Success or controlled user shutdown |
| 2 | CLI/configuration/input validation error |
| 3 | Board/runtime/resource failure |
| 4 | Validation gate failure |
| 5 | Model/recording schema, integrity, trust, or compatibility failure |

### Appendix E — Definition of done

Track A is done only when:

1. Required acceptance criteria pass from a clean wheel installation.
2. Reports/artifacts identify the exact lock, commit, config, seeds, and model IDs.
3. Every Must requirement has a linked automated test or explicit evidence artifact.
4. Any deferred Should item is named rather than silently omitted.
5. No result is described as real EEG, Cyton-stream, human motor-imagery, medical, or
   neuroscience-thesis validation.

### Appendix F — Requirement-to-evidence traceability

Test IDs are stable identifiers used in pytest names/docstrings and emitted reports. A requirement is
not complete until its mapped test/evidence is linked from the final Track A evidence report.

| Test/evidence ID | Requirements | Acceptance | CI job or artifact |
| --- | --- | --- | --- |
| T-CFG-001 | FR-001–005; NFR-007–008 | AC-001, AC-012 | static/unit config report |
| T-ACQ-001 | FR-006–011; NFR-012–013 | AC-002, AC-013 | integration BrainFlow report |
| T-DSP-001 | FR-012–017; NFR-002, NFR-011 | AC-003–004 | unit DSP response/chunk report |
| T-WIN-001 | FR-018–019; NFR-002–003 | AC-003 | unit/replay window-reference report |
| T-FEAT-001 | FR-020–027; NFR-003, NFR-011 | AC-005 | unit numeric-feature report |
| T-SYN-001 | FR-028–029; NFR-003–004 | AC-006 | generator reproducibility report |
| T-REC-001 | FR-030–031; NFR-010–012 | AC-009 | recording/replay and failure report |
| T-DEC-001 | FR-032–036; NFR-003–004 | AC-006 | decoder CV/permutation report |
| T-MODEL-001 | FR-037–040; NFR-007, NFR-011–012 | AC-007 | model security/parity report |
| T-CAL-001 | FR-041 | AC-011 when delivered | calibration drift/rollback report |
| T-PIPE-001 | FR-042; NFR-001–002, NFR-012 | AC-003, AC-010 | integration/performance metrics |
| T-VIZ-001 | FR-043–044 | AC-008 | PNG and artist-data report |
| T-CLI-001 | FR-045, FR-047; NFR-005–009 | AC-001, AC-008, AC-012 | clean-wheel package matrix |
| T-SMOKE-001 | FR-046; NFR-001 | AC-010 | both-decoder smoke metrics |
| T-SCOPE-001 | NFR-013 | AC-013 | final scope/evidence report |
