from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import chrono_link.recording as recording_module
from chrono_link.acquisition import SourceError
from chrono_link.config import ChronoConfig, ConfigError
from chrono_link.contracts import LifecycleState, SourceEvent
from chrono_link.recording import (
    RecordingError,
    ReplaySource,
    SessionRecord,
    SessionRecorder,
)
from tests.test_recording import board_info, chunk


def base_record() -> SessionRecord:
    source = chunk(0, 5)
    assert source.package_counter is not None and source.markers is not None
    return SessionRecord(
        data=source.eeg,
        timestamps=source.timestamps,
        sequence=source.sequence,
        package_counter=source.package_counter,
        channel_names=source.channel_names,
        units=("brainflow_native", "brainflow_native"),
        fs=source.fs,
        board_id=-1,
        created_utc="2026-07-10T12:00:00Z",
        markers=source.markers,
        config_json=ChronoConfig.synthetic().canonical_json(),
        events=(),
    )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"schema_version": 2}, "schema"),
        ({"data": np.ones((2, 5), dtype=np.float32)}, "data"),
        ({"timestamps": np.ones(4)}, "timestamps"),
        ({"channel_names": ("C3",)}, "channel_names"),
        ({"units": ("native",)}, "units"),
        ({"channel_names": ("C3", "C3")}, "unique"),
        ({"units": ("", "native")}, "units"),
        ({"fs": 0}, "fs"),
        ({"config_json": "[]"}, "object"),
        ({"config_json": "{bad"}, "invalid"),
        ({"created_utc": "bad"}, "RFC"),
        ({"created_utc": "2026-07-10T12:00:00"}, "offset"),
    ],
)
def test_session_record_schema_validation(changes, match: str) -> None:
    with pytest.raises(RecordingError, match=match):
        replace(base_record(), **changes)


def test_session_record_numeric_and_sequence_validation() -> None:
    record = base_record()
    data = record.data.copy()
    data[0, 0] = np.nan
    with pytest.raises(RecordingError, match="finite"):
        replace(record, data=data)
    markers = record.markers.copy()
    markers[0] = np.inf
    with pytest.raises(RecordingError, match="markers"):
        replace(record, markers=markers)
    timestamps = record.timestamps.copy()
    timestamps[2] = timestamps[1]
    with pytest.raises(RecordingError, match="timestamps"):
        replace(record, timestamps=timestamps)
    sequence = record.sequence.copy()
    sequence[2] = sequence[1]
    with pytest.raises(RecordingError, match="sequence"):
        replace(record, sequence=sequence)
    counters = record.package_counter.copy()
    counters[0] = -2
    with pytest.raises(RecordingError, match="counters"):
        replace(record, package_counter=counters)
    counters[0] = -1
    with pytest.raises(RecordingError, match="mix"):
        replace(record, package_counter=counters)


def test_session_record_event_and_gap_validation() -> None:
    record = base_record()
    with pytest.raises(RecordingError, match="outside"):
        replace(record, events=(SourceEvent("RESET", 99, {"reason": "bad"}),))
    duplicate = SourceEvent(
        "PACKAGE_GAP",
        3,
        {"previous_counter": 2, "current_counter": 7},
    )
    with pytest.raises(RecordingError, match="duplicate"):
        replace(record, events=(duplicate, duplicate))
    with pytest.raises(RecordingError, match="require"):
        replace(
            record,
            package_counter=np.full(5, -1, dtype=np.int64),
            events=(duplicate,),
        )

    counters = np.asarray([0, 1, 2, 7, 8], dtype=np.int64)
    with pytest.raises(RecordingError, match="require integer"):
        replace(
            record,
            package_counter=counters,
            events=(
                SourceEvent(
                    "PACKAGE_GAP",
                    3,
                    {"previous_counter": "x", "current_counter": 7},
                ),
            ),
        )
    with pytest.raises(RecordingError, match="counter details"):
        replace(
            record,
            package_counter=counters,
            events=(SourceEvent("PACKAGE_GAP", 3, {"previous_counter": 2, "current_counter": 3}),),
        )
    with pytest.raises(RecordingError, match="previous"):
        replace(
            record,
            package_counter=counters,
            events=(SourceEvent("PACKAGE_GAP", 3, {"previous_counter": 1, "current_counter": 7}),),
        )
    initial = SourceEvent(
        "PACKAGE_GAP",
        0,
        {"previous_counter": 250, "current_counter": 0},
    )
    assert replace(record, events=(initial,)).events == (initial,)


def write_fields(path: Path, fields: dict[str, object]) -> None:
    with path.open("xb") as output:
        np.savez(output, **fields)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", np.asarray(1, dtype=np.int32), "int64 scalar"),
        ("channel_names", np.asarray([1, 2], dtype=np.int64), "Unicode"),
        ("created_utc", np.asarray(["bad"]), "Unicode scalar"),
        ("config_json_utf8", np.asarray([1], dtype=np.int64), "uint8"),
        ("events_json_utf8", np.asarray([255], dtype=np.uint8), "UTF-8"),
    ],
)
def test_loader_rejects_field_type_corruption(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    fields = base_record().as_npz_fields()
    fields[field] = value
    path = tmp_path / f"{field}.npz"
    write_fields(path, fields)
    with pytest.raises(RecordingError, match=match):
        SessionRecord.load(path)


@pytest.mark.parametrize(
    "events",
    [
        {},
        [{"bad": 1}],
        [{"code": "BAD", "before_sequence": 0, "details": {}}],
        [{"code": "RESET", "before_sequence": True, "details": {}}],
        [{"code": "RESET", "before_sequence": 0, "details": {"bad": True}}],
    ],
)
def test_loader_rejects_event_json_schemas(tmp_path: Path, events: object) -> None:
    fields = base_record().as_npz_fields()
    fields["events_json_utf8"] = np.frombuffer(
        json.dumps(events).encode(), dtype=np.uint8
    )
    path = tmp_path / f"events-{len(list(tmp_path.iterdir()))}.npz"
    write_fields(path, fields)
    with pytest.raises(RecordingError, match="event"):
        SessionRecord.load(path)


def test_recorder_constructor_append_context_and_finalize_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "existing.npz"
    destination.write_bytes(b"exists")
    with pytest.raises(FileExistsError):
        SessionRecorder(
            destination,
            board_info(),
            ("C3", "C4"),
            ChronoConfig.synthetic(),
            max_samples=1,
        )
    for kwargs in (
        {"max_samples": 0},
        {"max_samples": 1, "channel_names": ()},
        {"max_samples": 1, "units": ("one",)},
    ):
        names = kwargs.pop("channel_names", ("C3", "C4"))
        with pytest.raises(ValueError):
            SessionRecorder(
                tmp_path / f"invalid-{len(list(tmp_path.iterdir()))}.npz",
                board_info(),
                names,
                ChronoConfig.synthetic(),
                **kwargs,
            )

    monkeypatch.setattr(
        recording_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=0),
    )
    with pytest.raises(RecordingError, match="disk"):
        SessionRecorder(
            tmp_path / "disk.npz",
            board_info(),
            ("C3", "C4"),
            ChronoConfig.synthetic(),
            max_samples=1,
        )
    monkeypatch.undo()

    success = tmp_path / "context.npz"
    no_metadata = replace(chunk(0, 1), package_counter=None, markers=None)
    with SessionRecorder(
        success,
        board_info(),
        ("C3", "C4"),
        ChronoConfig.synthetic(),
        max_samples=1,
    ) as recorder:
        recorder.append(no_metadata)
        assert recorder.sample_count == 1
    loaded = SessionRecord.load(success)
    assert loaded.package_counter.tolist() == [-1]
    assert loaded.markers.tolist() == [0.0]

    aborted_path = tmp_path / "aborted.npz"
    recorder = SessionRecorder(
        aborted_path,
        board_info(),
        ("C3", "C4"),
        ChronoConfig.synthetic(),
        max_samples=1,
    )
    with pytest.raises(RuntimeError):
        with recorder:
            raise RuntimeError("abort")
    with pytest.raises(RecordingError, match="closed"):
        recorder.append(chunk(0, 1))
    with pytest.raises(RecordingError, match="aborted"):
        recorder.finalize()


def test_recorder_race_and_replay_lifecycle_errors(tmp_path: Path) -> None:
    path = tmp_path / "race.npz"
    recorder = SessionRecorder(
        path,
        board_info(),
        ("C3", "C4"),
        ChronoConfig.synthetic(),
        max_samples=1,
    )
    with pytest.raises(RecordingError, match="channel"):
        recorder.append(replace(chunk(0, 1), channel_names=("X", "Y")))
    path.write_bytes(b"raced")
    with pytest.raises(FileExistsError):
        recorder.finalize()

    record = base_record()
    with pytest.raises(ValueError, match="sizes"):
        ReplaySource(record, chunk_sizes=0)
    replay = ReplaySource(record, decode_channels=("missing",))
    with pytest.raises(SourceError, match="start"):
        replay.start()
    with pytest.raises(SourceError, match="read"):
        replay.read_new()
    with pytest.raises(ConfigError, match="unavailable"):
        replay.prepare()
    released = ReplaySource(record)
    released.release()
    assert released.state is LifecycleState.RELEASED
    with pytest.raises(SourceError, match="prepare"):
        released.prepare()

    with ReplaySource(record, chunk_sizes=2) as active:
        first = active.prepare()
        assert active.prepare() is first
        active.start()
        assert active.read_new() is not None
    assert active.state is LifecycleState.RELEASED
