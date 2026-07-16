# Architecture and invariants

## System context

Chrono Link Track A is a single-process, CPU-only Python system. It accepts a live BrainFlow drain or
a validated local replay, converts source samples into causal overlapping windows, optionally emits
decoder predictions, and publishes local evidence artifacts. It deliberately avoids services,
databases, telemetry backends, GPUs, and network-facing APIs in this phase.

The architecture optimizes for deterministic evidence, strict ownership, failure visibility, and a
small trusted-computing boundary rather than maximum throughput or feature breadth.

## Non-negotiable invariants

1. Exactly one component owns and consumes a source.
2. Live acquisition drains new samples; it never repeatedly reads overlapping snapshots.
3. Samples are filtered once before overlapping windows are assembled.
4. Any continuity boundary resets every causal state that could bridge old and new data.
5. Sampling rate, rows, names, timestamps, markers, and counters come from resolved source facts.
6. Overlapping windows from one synthetic trial stay in one validation group.
7. Persisted recordings are pickle-free; persisted model code is never loaded without explicit trust.
8. Normal logs, errors, metrics, and prediction events do not contain raw signal arrays.
9. Outputs publish atomically where supported and refuse replacement.
10. Configured behavior is not claimed as observed evidence until an actual run proves it.

## Package ownership

| Module | Responsibility | Must not own |
| --- | --- | --- |
| `config.py` | Strict defaults, JSON/environment/CLI merge, fingerprints | Runtime board handles |
| `contracts.py` | Immutable typed data passed between stages | I/O or estimator behavior |
| `acquisition.py` | BrainFlow lifecycle, board facts, drain, continuity events | DSP or persistence |
| `recording.py` | NPZ schema, bounded atomic recorder, deterministic replay | Joblib models |
| `filters.py` | Filter design, causal state, offline reference filtering | Window overlap |
| `windows.py` | Exact sample-indexed overlap and reset behavior | Filtering |
| `features.py` | Quality checks, spectral/temporal feature contracts | Fold splitting |
| `synthetic.py` | Deterministic trial generator and live-path preprocessing | Acceptance decisions |
| `decoders.py` | Input adapters and fitted estimator pipelines | Validation orchestration |
| `validation.py` | Grouped folds, trial scoring, permutations, normative report | Artifact deserialization |
| `model_store.py` | Model contract, hashes, self-test, trusted load/save | Source acquisition |
| `pipeline.py` | Stage orchestration, lifecycle, timing, callbacks | CLI parsing or plotting |
| `visualization.py` | Reused plot artists and strict local evidence output | Core acquisition/DSP |
| `cli/` | User-facing parsing, exit mapping, command composition | New scientific logic |

Dependencies should point inward toward contracts and domain logic. CLI modules may compose core
modules; core modules must not import CLI modules. Acquisition, filtering, features, recording, and
model storage must not import pyplot.

## Runtime data flow

```text
SampleSource
  | raw SampleChunk
  +----> SessionRecorder (optional, bounded and atomic)
  +----> LiveVisualizer (explicit local raw callback)
  `----> CausalFilterBank
          `----> FilteredChunk
                  `----> WindowAssembler
                          `----> WindowBundle + quality
                                  +----> plot-only callback
                                  `----> BoundDecoder
                                          `----> PredictionEvent (no raw EEG)
```

`BrainFlowSource` is the sole owner and consumer of a BrainFlow `BoardShim`. It resolves all board
facts during preparation and drains accumulated data with `get_board_data()`. `ReplaySource`
implements the same lifecycle from a validated, pickle-free `SessionRecord`.

Raw selected EEG fans out to recording and visualization before DSP. That is an explicit local trust
boundary: callers that register a raw callback are responsible for the data they receive.

## Source lifecycle

```text
NEW -> PREPARED -> STREAMING -> STOPPED -> RELEASED
  |         |          |
  +---------+----------+----> FAILED (on fatal source/runtime error)
```

- `prepare()` may resolve facts and acquire resources but does not start streaming.
- `start()` is valid only after preparation.
- `read_new()` is valid only while streaming.
- `stop()` and `release()` are idempotent for normal cleanup.
- Context-manager exit releases acquired resources even when downstream processing fails.
- Inconsistent internal lifecycle state raises an explicit runtime error; correctness never depends
  on Python `assert` statements that disappear under optimization.

## Continuity and causal state

Package-counter discontinuities split a raw drain before processing. The first post-gap chunk carries
a `PACKAGE_GAP` event. Invalid source chunks cause a reset before the next valid chunk, and three
consecutive invalid chunks are fatal. Timestamp jitter is telemetry; non-finite or non-monotonic
timestamps are invalid.

On a reset, `RealtimePipeline` clears filter histories, settling state, and window buffers. No window
may include samples from both sides of a boundary.

`CausalFilterBank` owns independent SOS state for the cleaned cascade and each configured band.
`WindowAssembler` operates only on filtered samples. This makes results invariant to arbitrary source
chunking while avoiding repeated filtering of overlap.

## Decoder and validation boundary

A `BoundDecoder` owns an input adapter and a fitted sklearn-compatible pipeline:

- The Riemann adapter selects MI-band epochs. OAS covariance, tangent-space projection, and shrinkage
  LDA remain inside the fold-local fitted pipeline.
- The temporal-vector adapter constructs the stable feature schema before fold-local scaling and
  shrinkage LDA.

Validation uses `StratifiedGroupKFold`. Every trial owns four overlapping windows; no trial crosses a
fold. Probabilities are averaged at trial level before balanced accuracy and binomial evidence are
computed. Label permutations happen at trial granularity.

## Persistence boundary

### Recordings

`SessionRecord` uses compressed NPZ with fixed dtypes, shapes, strict UTF-8 JSON, and no pickle.
Loading validates schema version, numeric finiteness, timestamps, sequence, counters, events,
channels, configuration, and cross-field consistency before constructing a record.

### Models

Model directories contain a manifest, a joblib payload, and a pickle-free self-test. The loader checks
manifest schema, payload/self-test SHA-256, Python/dependency versions, channel order, sampling rate,
window/hop, filter/feature fingerprints, and explicit trust before deserialization. Prediction parity
against the self-test is checked after load.

Joblib is a trusted-code format. Hashes establish integrity and contract identity, not safety or
provenance.

## Extension rules

- Add a new source by implementing `SampleSource`; do not branch the downstream pipeline.
- Add a new decoder through an adapter plus bound estimator; keep preprocessing inside fold-local
  fitting where leakage is possible.
- Change a persisted schema only with a version increment, migration/compatibility policy, negative
  tests, and documentation.
- Change default filter, feature, window, dependency, or channel contracts only with new fingerprints
  and re-run normative evidence.
- Add external services only through a specification that defines authentication, privacy, retries,
  observability, retention, and rollback.
