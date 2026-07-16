from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chrono_link.config import ChronoConfig
from chrono_link.contracts import ResolvedBoardInfo, SampleChunk, SourceEvent
from chrono_link.recording import (
    REQUIRED_FIELDS,
    RecordingError,
    ReplaySource,
    SessionRecord,
    SessionRecorder,
)


def board_info() -> ResolvedBoardInfo:
    return ResolvedBoardInfo(
        board_id=-1,
        fs=250,
        eeg_names=("C3", "C4"),
        eeg_rows=(1, 2),
        timestamp_row=3,
        package_num_row=0,
        marker_row=4,
        num_rows=5,
        decode_rows=(1, 2),
        record_rows=(1, 2),
    )


def chunk(start: int, stop: int, *, event: bool = False) -> SampleChunk:
    sequence = np.arange(start, stop, dtype=np.int64)
    counters = sequence % 256
    if event:
        counters = counters.copy()
        counters[0:] += 3
    events = ()
    if event:
        events = (
            SourceEvent(
                "PACKAGE_GAP",
                start,
                {"previous_counter": start - 1, "current_counter": start + 3},
            ),
        )
    return SampleChunk(
        eeg=np.vstack((sequence, sequence + 1000)).astype(np.float64),
        timestamps=1_700_000_000.0 + sequence / 250.0,
        sequence=sequence,
        package_counter=counters,
        channel_names=("C3", "C4"),
        fs=250,
        markers=np.zeros(stop - start, dtype=np.float64),
        events_before=events,
    )


def test_625_sample_recording_round_trip_and_replay(tmp_path: Path) -> None:
    path = tmp_path / "session.npz"
    expected = chunk(0, 625)
    recorder = SessionRecorder(
        path,
        board_info(),
        expected.channel_names,
        ChronoConfig.synthetic(),
        max_samples=625,
        created_utc="2026-07-10T12:00:00Z",
    )
    recorder.append(expected)
    assert recorder.finalize() == path
    assert not list(tmp_path.glob("*.tmp"))

    record = SessionRecord.load(path)
    np.testing.assert_array_equal(record.data, expected.eeg)
    np.testing.assert_array_equal(record.sequence, expected.sequence)
    assert record.channel_names == expected.channel_names

    replay = ReplaySource(path, chunk_sizes=(1, 64, 127))
    replay.prepare()
    replay.start()
    parts = []
    while (part := replay.read_new()) is not None:
        parts.append(part)
    np.testing.assert_array_equal(
        np.concatenate([part.eeg for part in parts], axis=1), expected.eeg
    )
    np.testing.assert_array_equal(
        np.concatenate([part.sequence for part in parts]), expected.sequence
    )
    replay.release()


def test_replay_never_crosses_recorded_gap(tmp_path: Path) -> None:
    path = tmp_path / "gap.npz"
    recorder = SessionRecorder(
        path,
        board_info(),
        ("C3", "C4"),
        ChronoConfig.synthetic(),
        max_samples=10,
    )
    recorder.append(chunk(0, 5))
    recorder.append(chunk(5, 10, event=True))
    recorder.finalize()

    replay = ReplaySource(path, chunk_sizes=10)
    replay.prepare()
    replay.start()
    before = replay.read_new()
    after = replay.read_new()
    assert before is not None and after is not None
    assert before.eeg.shape[1] == 5
    assert after.eeg.shape[1] == 5
    assert after.events_before[0].before_sequence == 5


def test_loader_rejects_extra_fields_and_gap_disagreement(tmp_path: Path) -> None:
    good = tmp_path / "good.npz"
    recorder = SessionRecorder(
        good,
        board_info(),
        ("C3", "C4"),
        ChronoConfig.synthetic(),
        max_samples=5,
    )
    recorder.append(chunk(0, 5))
    recorder.finalize()
    with np.load(good, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}

    extra = tmp_path / "extra.npz"
    np.savez(extra, **values, unexpected=np.asarray(1))
    with pytest.raises(RecordingError, match="fields mismatch"):
        SessionRecord.load(extra)
    assert set(values) == REQUIRED_FIELDS

    values["package_counter"] = values["package_counter"].copy()
    values["package_counter"][3] += 4
    corrupt = tmp_path / "corrupt.npz"
    np.savez(corrupt, **values)
    with pytest.raises(RecordingError, match="disagree"):
        SessionRecord.load(corrupt)


def test_recorder_aborts_without_partial_final(tmp_path: Path) -> None:
    path = tmp_path / "never.npz"
    recorder = SessionRecorder(
        path,
        board_info(),
        ("C3", "C4"),
        ChronoConfig.synthetic(),
        max_samples=2,
    )
    with pytest.raises(RecordingError, match="capacity"):
        recorder.append(chunk(0, 3))
    recorder.abort()
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_config_json_is_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "strict.npz"
    recorder = SessionRecorder(
        path,
        board_info(),
        ("C3", "C4"),
        ChronoConfig.synthetic(),
        max_samples=1,
    )
    recorder.append(chunk(0, 1))
    recorder.finalize()
    record = SessionRecord.load(path)
    assert isinstance(json.loads(record.config_json), dict)
