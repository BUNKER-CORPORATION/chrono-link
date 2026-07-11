# Architecture

## Ownership and flow

`BrainFlowSource` is the sole owner and consumer of a BrainFlow `BoardShim`. It drains accumulated
samples with `get_board_data()`; no stateful path reads overlapping snapshots. `ReplaySource`
implements the same lifecycle from validated, pickle-free NPZ recordings.

Every source chunk retains raw selected EEG for recorder/visualization fan-out. Package-counter
discontinuities split chunks before DSP. The first post-gap chunk carries an event, and the runtime
resets filters, warm-up state, and window buffers before processing it. Timestamp jitter is
telemetry; non-finite or non-monotonic timestamps are invalid, with three consecutive invalid chunks
treated as fatal.

`CausalFilterBank` owns independent SOS state for the cleaned cascade and each band branch. Samples
are filtered once, then `WindowAssembler` creates overlapping windows from filtered ring buffers.
This preserves causal chunk invariance and avoids refiltering overlap.

~~~text
SampleSource
  | raw SampleChunk
  +----> SessionRecorder (optional, bounded/atomic)
  +----> LiveVisualizer (explicit local raw callback)
  `----> CausalFilterBank
          `----> WindowAssembler
                  `----> quality gate
                          +----> plot-only callbacks
                          `----> BoundDecoder
                                  `----> non-raw PredictionEvent
~~~

## Decoder boundaries

A `BoundDecoder` owns both its input adapter and fitted estimator. The epoch adapter selects the MI
tensor and leaves OAS covariance estimation inside the fitted pyRiemann/sklearn pipeline. The vector
adapter constructs the stable 20-feature default schema before a fold-local scaler and shrinkage LDA.

Model bundles persist the complete bound object plus a pickle-free self-test. The loader validates
the manifest, both SHA-256 hashes, dependency versions, Python major/minor, channels, sampling rate,
window/hop, filter fingerprint, feature fingerprint, and trust acknowledgment before joblib loading.

## Resource rules

- Source stop/release are idempotent and called at most once after acquisition.
- Recordings and models publish from same-filesystem temporary paths and refuse replacement.
- Headless matplotlib selects `Agg` before importing pyplot.
- Core acquisition, DSP, feature, recording, and model modules do not import pyplot.
- Normal metrics and prediction callbacks do not contain raw EEG samples.
