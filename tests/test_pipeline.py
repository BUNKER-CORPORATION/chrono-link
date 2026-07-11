from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from chrono_link.config import ChronoConfig
from chrono_link.contracts import SourceEvent
from chrono_link.decoders import make_riemann_decoder
from chrono_link.pipeline import PipelineState, RealtimePipeline
from chrono_link.recording import ReplaySource, SessionRecord
from chrono_link.synthetic import (
    SyntheticMISessionGenerator,
    preprocess_synthetic_session,
)


def record_from_values(values: np.ndarray) -> SessionRecord:
    samples = values.shape[1]
    sequence = np.arange(samples, dtype=np.int64)
    return SessionRecord(
        data=np.asarray(values, dtype=np.float64),
        timestamps=1_700_000_000.0 + sequence / 250,
        sequence=sequence,
        package_counter=sequence % 256,
        channel_names=("C3", "C4"),
        units=("brainflow_native", "brainflow_native"),
        fs=250,
        board_id=-1,
        created_utc=datetime.now(UTC).isoformat(),
        markers=np.zeros(samples, dtype=np.float64),
        config_json=ChronoConfig.synthetic().canonical_json(),
        events=(),
    )


def test_pipeline_replay_applies_warmup_and_emits_six_exact_windows() -> None:
    rng = np.random.default_rng(20260709)
    record = record_from_values(rng.normal(size=(2, 1625)))
    starts = []
    source = ReplaySource(record, chunk_sizes=(1, 64, 511, 7))
    pipeline = RealtimePipeline(
        source,
        ChronoConfig.synthetic(),
        window_callback=lambda window: starts.append(window.start_sequence),
    )
    metrics = pipeline.run(poll_interval_s=0)
    assert starts == [1000, 1064, 1128, 1192, 1256, 1320]
    assert metrics.windows == 6
    assert metrics.predictions == 0
    assert pipeline.state is PipelineState.RELEASED


def test_pipeline_decodes_valid_synthetic_windows() -> None:
    session = SyntheticMISessionGenerator(7, trial_count=20).generate()
    dataset = preprocess_synthetic_session(session)
    decoder = make_riemann_decoder().fit(dataset.windows, dataset.labels)
    record = record_from_values(session.raw_trials[0])
    source = ReplaySource(record, chunk_sizes=(1250, 64, 64, 64, 64))
    predictions = []
    pipeline = RealtimePipeline(
        source,
        ChronoConfig.synthetic(),
        decoder=decoder,
        prediction_callback=predictions.append,
    )
    metrics = pipeline.run(poll_interval_s=0)
    assert metrics.predictions == 4
    assert len(predictions) == 4
    assert all(np.isfinite(event.probabilities).all() for event in predictions)


def test_callback_failure_still_releases_source() -> None:
    record = record_from_values(np.ones((2, 20), dtype=np.float64))
    source = ReplaySource(record, chunk_sizes=20)

    def fail(chunk) -> None:
        raise RuntimeError("callback failed")

    pipeline = RealtimePipeline(
        source,
        ChronoConfig.synthetic(),
        raw_callback=fail,
    )
    with pytest.raises(RuntimeError, match="callback failed"):
        pipeline.run(poll_interval_s=0)
    assert pipeline.state is PipelineState.RELEASED


def test_gap_replay_patterns_reset_full_warmup_and_never_bridge_boundary() -> None:
    logical = np.delete(np.arange(3200, dtype=np.int64), 1300)
    local = np.arange(3199, dtype=np.int64)
    rng = np.random.default_rng(20260709)
    values = rng.normal(size=(2, 3199))
    previous = int((250 + 1299) % 256)
    current = int((250 + 1301) % 256)
    record = SessionRecord(
        data=values,
        timestamps=1_700_000_000.0 + logical / 250,
        sequence=local,
        package_counter=(250 + logical) % 256,
        channel_names=("C3", "C4"),
        units=("brainflow_native", "brainflow_native"),
        fs=250,
        board_id=-1,
        created_utc=datetime.now(UTC).isoformat(),
        markers=np.zeros(3199, dtype=np.float64),
        config_json=ChronoConfig.synthetic().canonical_json(),
        events=(
            SourceEvent(
                "PACKAGE_GAP",
                1300,
                {"previous_counter": previous, "current_counter": current},
            ),
        ),
    )
    outputs = []
    for sizes in (3199, 64, (1, 7, 64, 3, 511, 2, 128, 19)):
        ranges = []
        pipeline = RealtimePipeline(
            ReplaySource(record, chunk_sizes=sizes),
            ChronoConfig.synthetic(),
            window_callback=lambda window, target=ranges: target.append(
                (window.start_sequence, window.end_sequence)
            ),
        )
        metrics = pipeline.run(poll_interval_s=0)
        assert metrics.gaps == 1 and metrics.resets == 1
        assert not any(start < 1300 < stop for start, stop in ranges)
        outputs.append(ranges)
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0][0] == (1000, 1250)
    assert outputs[0][1][0] == 2300
