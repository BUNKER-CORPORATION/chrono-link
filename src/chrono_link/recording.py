"""Bounded, atomic session recording and deterministic replay."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from chrono_link.acquisition import SourceError
from chrono_link.config import ChronoConfig, ConfigError
from chrono_link.contracts import (
    LifecycleState,
    ResolvedBoardInfo,
    SampleChunk,
    SourceEvent,
)

SCHEMA_VERSION = 1
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "data",
        "timestamps",
        "sequence",
        "package_counter",
        "channel_names",
        "units",
        "fs",
        "board_id",
        "created_utc",
        "markers",
        "config_json_utf8",
        "events_json_utf8",
    }
)


class RecordingError(RuntimeError):
    """A recording cannot be created, validated, or replayed."""


def _strict_json_loads(value: str, label: str) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"invalid JSON constant {token}")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RecordingError(f"invalid {label}: {exc}") from exc


def _encode_json(value: Any) -> NDArray[np.uint8]:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8).copy()


def _decode_utf8(value: NDArray[np.uint8], label: str) -> str:
    if value.ndim != 1 or value.dtype != np.uint8:
        raise RecordingError(f"{label} must be a one-dimensional uint8 array")
    try:
        return value.tobytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RecordingError(f"{label} is not valid UTF-8") from exc


def _event_to_dict(event: SourceEvent) -> dict[str, Any]:
    return {
        "code": event.code,
        "before_sequence": event.before_sequence,
        "details": dict(event.details),
    }


def _parse_events(value: Any) -> tuple[SourceEvent, ...]:
    if not isinstance(value, list):
        raise RecordingError("events JSON must contain a list")
    events: list[SourceEvent] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "code",
            "before_sequence",
            "details",
        }:
            raise RecordingError(f"event {index} has an invalid schema")
        code = item["code"]
        if code not in {"PACKAGE_GAP", "REPLAY_GAP", "RESET", "STALL"}:
            raise RecordingError(f"event {index} has invalid code {code!r}")
        before = item["before_sequence"]
        details = item["details"]
        if not isinstance(before, int) or isinstance(before, bool):
            raise RecordingError(f"event {index} before_sequence must be an integer")
        if not isinstance(details, dict) or any(
            not isinstance(key, str)
            or isinstance(detail, bool)
            or not isinstance(detail, (str, int, float))
            or (isinstance(detail, float) and not np.isfinite(detail))
            for key, detail in details.items()
        ):
            raise RecordingError(f"event {index} details are invalid")
        events.append(
            SourceEvent(
                code=cast(Literal["PACKAGE_GAP", "REPLAY_GAP", "RESET", "STALL"], code),
                before_sequence=before,
                details=cast(Mapping[str, str | int | float], details),
            )
        )
    return tuple(events)


@dataclass(frozen=True)
class SessionRecord:
    """Validated, pickle-free representation of one raw source session."""

    data: NDArray[np.float64]
    timestamps: NDArray[np.float64]
    sequence: NDArray[np.int64]
    package_counter: NDArray[np.int64]
    channel_names: tuple[str, ...]
    units: tuple[str, ...]
    fs: int
    board_id: int
    created_utc: str
    markers: NDArray[np.float64]
    config_json: str
    events: tuple[SourceEvent, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    @property
    def sample_count(self) -> int:
        return int(self.data.shape[1])

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RecordingError(f"unsupported recording schema {self.schema_version}")
        if self.data.dtype != np.float64 or self.data.ndim != 2:
            raise RecordingError("data must be float64 with shape (channels, samples)")
        channels, samples = self.data.shape
        expected_vectors = (
            ("timestamps", self.timestamps, np.dtype(np.float64)),
            ("sequence", self.sequence, np.dtype(np.int64)),
            ("package_counter", self.package_counter, np.dtype(np.int64)),
            ("markers", self.markers, np.dtype(np.float64)),
        )
        for name, value, dtype in expected_vectors:
            if value.dtype != dtype or value.shape != (samples,):
                raise RecordingError(f"{name} must have dtype {dtype} and shape ({samples},)")
        if channels == 0 or len(self.channel_names) != channels:
            raise RecordingError("channel_names must match non-empty data rows")
        if len(self.units) != channels:
            raise RecordingError("units must match data rows")
        if not all(self.channel_names) or len(set(self.channel_names)) != channels:
            raise RecordingError("channel_names must be non-empty and unique")
        if not all(self.units):
            raise RecordingError("units must be non-empty")
        if self.fs <= 0:
            raise RecordingError("fs must be positive")
        if not np.isfinite(self.data).all() or not np.isfinite(self.timestamps).all():
            raise RecordingError("recording data and timestamps must be finite")
        if not np.isfinite(self.markers).all():
            raise RecordingError("recording markers must be finite")
        if samples:
            if np.any(np.diff(self.timestamps) <= 0):
                raise RecordingError("timestamps must be strictly increasing")
            if np.any(np.diff(self.sequence) <= 0):
                raise RecordingError("sequence must be strictly increasing")
        if np.any(self.package_counter < -1):
            raise RecordingError("package counters must be -1 or non-negative")
        if np.any(self.package_counter == -1) and not np.all(self.package_counter == -1):
            raise RecordingError("package counters cannot mix missing and present values")
        config = _strict_json_loads(self.config_json, "config JSON")
        if not isinstance(config, dict):
            raise RecordingError("config JSON must contain an object")
        try:
            parsed_created = datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RecordingError("created_utc must be RFC 3339") from exc
        if parsed_created.tzinfo is None:
            raise RecordingError("created_utc must include a UTC offset")

        sequence_positions = {int(value): index for index, value in enumerate(self.sequence)}
        for event in self.events:
            if event.before_sequence not in sequence_positions:
                raise RecordingError(
                    f"event boundary {event.before_sequence} is outside the recording"
                )
        self._validate_gap_events(sequence_positions)

    def _validate_gap_events(self, positions: Mapping[int, int]) -> None:
        package_events = {
            event.before_sequence: event for event in self.events if event.code == "PACKAGE_GAP"
        }
        if len(package_events) != sum(event.code == "PACKAGE_GAP" for event in self.events):
            raise RecordingError("duplicate PACKAGE_GAP event boundary")
        if np.all(self.package_counter == -1):
            if package_events:
                raise RecordingError("PACKAGE_GAP events require package counters")
            return

        expected = {
            int(self.sequence[index])
            for index in range(1, self.sample_count)
            if (int(self.package_counter[index]) - int(self.package_counter[index - 1])) % 256 != 1
        }
        internal_actual = {boundary for boundary in package_events if positions[boundary] > 0}
        if internal_actual != expected:
            raise RecordingError("PACKAGE_GAP events disagree with package-counter discontinuities")
        for boundary, event in package_events.items():
            position = positions[boundary]
            previous = event.details.get("previous_counter")
            current = event.details.get("current_counter")
            if not isinstance(previous, int) or not isinstance(current, int):
                raise RecordingError("PACKAGE_GAP details require integer counters")
            if current != int(self.package_counter[position]) or (current - previous) % 256 == 1:
                raise RecordingError("PACKAGE_GAP counter details are inconsistent")
            if position > 0 and previous != int(self.package_counter[position - 1]):
                raise RecordingError("PACKAGE_GAP previous counter is inconsistent")

    def as_npz_fields(self) -> dict[str, Any]:
        return {
            "schema_version": np.asarray(self.schema_version, dtype=np.int64),
            "data": self.data,
            "timestamps": self.timestamps,
            "sequence": self.sequence,
            "package_counter": self.package_counter,
            "channel_names": np.asarray(self.channel_names, dtype=np.str_),
            "units": np.asarray(self.units, dtype=np.str_),
            "fs": np.asarray(self.fs, dtype=np.int64),
            "board_id": np.asarray(self.board_id, dtype=np.int64),
            "created_utc": np.asarray(self.created_utc, dtype=np.str_),
            "markers": self.markers,
            "config_json_utf8": np.frombuffer(
                self.config_json.encode("utf-8"), dtype=np.uint8
            ).copy(),
            "events_json_utf8": _encode_json([_event_to_dict(event) for event in self.events]),
        }

    @classmethod
    def load(cls, path: Path) -> SessionRecord:
        try:
            with np.load(path, allow_pickle=False) as archive:
                fields = set(archive.files)
                if fields != REQUIRED_FIELDS:
                    missing = sorted(REQUIRED_FIELDS - fields)
                    extra = sorted(fields - REQUIRED_FIELDS)
                    raise RecordingError(
                        f"recording fields mismatch: missing={missing}, extra={extra}"
                    )
                arrays = {name: archive[name] for name in archive.files}
        except RecordingError:
            raise
        except (OSError, ValueError, EOFError) as exc:
            raise RecordingError(f"cannot load recording {path}: {exc}") from exc

        if any(value.dtype.hasobject for value in arrays.values()):
            raise RecordingError("object arrays are forbidden in recordings")
        for name in ("schema_version", "fs", "board_id"):
            if arrays[name].shape != () or arrays[name].dtype != np.int64:
                raise RecordingError(f"{name} must be an int64 scalar")
        for name in ("channel_names", "units"):
            if arrays[name].ndim != 1 or arrays[name].dtype.kind != "U":
                raise RecordingError(f"{name} must be a one-dimensional Unicode array")
        if arrays["created_utc"].shape != () or arrays["created_utc"].dtype.kind != "U":
            raise RecordingError("created_utc must be a Unicode scalar")

        config_json = _decode_utf8(arrays["config_json_utf8"], "config_json_utf8")
        events_json = _decode_utf8(arrays["events_json_utf8"], "events_json_utf8")
        events = _parse_events(_strict_json_loads(events_json, "events JSON"))
        return cls(
            data=arrays["data"],
            timestamps=arrays["timestamps"],
            sequence=arrays["sequence"],
            package_counter=arrays["package_counter"],
            channel_names=tuple(str(value) for value in arrays["channel_names"]),
            units=tuple(str(value) for value in arrays["units"]),
            fs=int(arrays["fs"]),
            board_id=int(arrays["board_id"]),
            created_utc=str(arrays["created_utc"]),
            markers=arrays["markers"],
            config_json=config_json,
            events=events,
            schema_version=int(arrays["schema_version"]),
        )


class SessionRecorder:
    """Append chunks to bounded disk storage and publish one atomic NPZ."""

    def __init__(
        self,
        destination: Path,
        info: ResolvedBoardInfo,
        channel_names: tuple[str, ...],
        config: ChronoConfig,
        *,
        max_samples: int,
        units: tuple[str, ...] | None = None,
        created_utc: str | None = None,
    ) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if destination.exists():
            raise FileExistsError(f"recording already exists: {destination}")
        if not channel_names:
            raise ValueError("channel_names cannot be empty")
        if units is not None and len(units) != len(channel_names):
            raise ValueError("units must match channel_names")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._destination = destination
        self._info = info
        self._channel_names = channel_names
        self._config_json = config.canonical_json()
        self._max_samples = max_samples
        self._units = units or ("brainflow_native",) * len(channel_names)
        self._created_utc = created_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self._events: list[SourceEvent] = []
        self._count = 0
        self._closed = False
        self._published = False
        self._mmap_path = self._temporary_path("mmap")
        self._npz_path = self._temporary_path("npz")
        self._dtype = np.dtype(
            [
                ("data", "<f8", (len(channel_names),)),
                ("timestamp", "<f8"),
                ("sequence", "<i8"),
                ("package_counter", "<i8"),
                ("marker", "<f8"),
            ]
        )
        required = int(self._dtype.itemsize * max_samples * 2 * 1.2) + 1_048_576
        free = shutil.disk_usage(destination.parent).free
        if free < required:
            raise RecordingError(
                f"insufficient disk space: require {required} bytes, available {free}"
            )
        try:
            self._store = np.memmap(
                self._mmap_path,
                dtype=self._dtype,
                mode="w+",
                shape=(max_samples,),
            )
        except Exception:
            self._cleanup_paths()
            raise

    def _temporary_path(self, kind: str) -> Path:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self._destination.name}.",
            suffix=f".{kind}.tmp",
            dir=self._destination.parent,
        )
        os.close(descriptor)
        path = Path(name)
        path.unlink()
        return path

    @property
    def sample_count(self) -> int:
        return self._count

    def append(self, chunk: SampleChunk) -> None:
        if self._closed:
            raise RecordingError("recorder is closed")
        if chunk.channel_names != self._channel_names or chunk.fs != self._info.fs:
            raise RecordingError("chunk channel order or sampling rate changed")
        samples = chunk.eeg.shape[1]
        if self._count + samples > self._max_samples:
            raise RecordingError(
                f"recording capacity {self._max_samples} samples would be exceeded"
            )
        target = slice(self._count, self._count + samples)
        self._store["data"][target] = chunk.eeg.T
        self._store["timestamp"][target] = chunk.timestamps
        self._store["sequence"][target] = chunk.sequence
        self._store["package_counter"][target] = (
            -1 if chunk.package_counter is None else chunk.package_counter
        )
        self._store["marker"][target] = 0.0 if chunk.markers is None else chunk.markers
        self._events.extend(chunk.events_before)
        self._count += samples

    def finalize(self) -> Path:
        if self._published:
            return self._destination
        if self._closed:
            raise RecordingError("recorder was aborted")
        if self._destination.exists():
            self.abort()
            raise FileExistsError(f"recording already exists: {self._destination}")
        try:
            self._store.flush()
            record = SessionRecord(
                data=np.array(self._store["data"][: self._count].T, copy=True),
                timestamps=np.array(self._store["timestamp"][: self._count], copy=True),
                sequence=np.array(self._store["sequence"][: self._count], copy=True),
                package_counter=np.array(self._store["package_counter"][: self._count], copy=True),
                channel_names=self._channel_names,
                units=self._units,
                fs=self._info.fs,
                board_id=self._info.board_id,
                created_utc=self._created_utc,
                markers=np.array(self._store["marker"][: self._count], copy=True),
                config_json=self._config_json,
                events=tuple(self._events),
            )
            with self._npz_path.open("xb") as output:
                np.savez_compressed(output, **record.as_npz_fields())
                output.flush()
                os.fsync(output.fileno())
            os.link(self._npz_path, self._destination)
            self._npz_path.unlink()
            self._published = True
            self._close_store()
            self._mmap_path.unlink(missing_ok=True)
            return self._destination
        except Exception:
            self.abort()
            raise

    def _close_store(self) -> None:
        mmap = getattr(self._store, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self._closed = True

    def _cleanup_paths(self) -> None:
        self._mmap_path.unlink(missing_ok=True)
        self._npz_path.unlink(missing_ok=True)

    def abort(self) -> None:
        if not self._closed:
            self._close_store()
        self._cleanup_paths()

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()


class ReplaySource:
    """Replay a validated SessionRecord in deterministic continuity-bounded chunks."""

    def __init__(
        self,
        record: SessionRecord | Path,
        *,
        chunk_sizes: int | Sequence[int] = 64,
        decode_channels: tuple[str, ...] | None = None,
    ) -> None:
        self._record_or_path = record
        sizes = (chunk_sizes,) if isinstance(chunk_sizes, int) else tuple(chunk_sizes)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("chunk sizes must be positive")
        self._sizes = sizes
        self._size_iterator = cycle(sizes)
        self._decode_channels = decode_channels
        self._record: SessionRecord | None = None
        self._info: ResolvedBoardInfo | None = None
        self._state = LifecycleState.NEW
        self._position = 0
        self._events_by_sequence: dict[int, tuple[SourceEvent, ...]] = {}

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def exhausted(self) -> bool:
        return self._record is not None and self._position >= self._record.sample_count

    def prepare(self) -> ResolvedBoardInfo:
        if self._state is LifecycleState.PREPARED:
            if self._info is None:
                raise SourceError("prepared replay is missing resolved board information")
            return self._info
        if self._state is not LifecycleState.NEW:
            raise SourceError(f"cannot prepare replay in state {self._state.name}")
        record = (
            SessionRecord.load(self._record_or_path)
            if isinstance(self._record_or_path, Path)
            else self._record_or_path
        )
        decode_names = self._decode_channels or record.channel_names
        missing = [name for name in decode_names if name not in record.channel_names]
        if missing:
            raise ConfigError(
                f"decode channels {missing} are unavailable; available={list(record.channel_names)}"
            )
        name_to_row = {name: index for index, name in enumerate(record.channel_names)}
        channel_count = len(record.channel_names)
        self._record = record
        self._info = ResolvedBoardInfo(
            board_id=record.board_id,
            fs=record.fs,
            eeg_names=record.channel_names,
            eeg_rows=tuple(range(channel_count)),
            timestamp_row=channel_count,
            package_num_row=channel_count + 1,
            marker_row=channel_count + 2,
            num_rows=channel_count + 3,
            decode_rows=tuple(name_to_row[name] for name in decode_names),
            record_rows=tuple(range(channel_count)),
        )
        event_groups: dict[int, list[SourceEvent]] = {}
        for event in record.events:
            event_groups.setdefault(event.before_sequence, []).append(event)
        self._events_by_sequence = {
            sequence: tuple(events) for sequence, events in event_groups.items()
        }
        self._state = LifecycleState.PREPARED
        return self._info

    def start(self) -> None:
        if self._state is LifecycleState.STREAMING:
            return
        if self._state is not LifecycleState.PREPARED:
            raise SourceError(f"cannot start replay in state {self._state.name}")
        self._position = 0
        self._size_iterator = cycle(self._sizes)
        self._state = LifecycleState.STREAMING

    def read_new(self) -> SampleChunk | None:
        if self._state is not LifecycleState.STREAMING or self._record is None:
            raise SourceError(f"cannot read replay in state {self._state.name}")
        record = self._record
        if self._position >= record.sample_count:
            return None
        requested_stop = min(
            record.sample_count,
            self._position + next(self._size_iterator),
        )
        start_sequence = int(record.sequence[self._position])
        event_boundaries = [
            int(np.searchsorted(record.sequence, boundary))
            for boundary in self._events_by_sequence
            if boundary > start_sequence
        ]
        internal = [
            boundary for boundary in event_boundaries if self._position < boundary < requested_stop
        ]
        stop = min(internal, default=requested_stop)
        target = slice(self._position, stop)
        events = self._events_by_sequence.get(start_sequence, ())
        package = record.package_counter[target]
        chunk = SampleChunk(
            eeg=np.array(record.data[:, target], copy=True),
            timestamps=np.array(record.timestamps[target], copy=True),
            sequence=np.array(record.sequence[target], copy=True),
            package_counter=(None if np.all(package == -1) else np.array(package, copy=True)),
            channel_names=record.channel_names,
            fs=record.fs,
            markers=np.array(record.markers[target], copy=True),
            events_before=events,
        )
        self._position = stop
        return chunk

    def stop(self) -> None:
        if self._state is LifecycleState.STREAMING:
            self._state = LifecycleState.STOPPED

    def release(self) -> None:
        if self._state is not LifecycleState.RELEASED:
            if self._state is LifecycleState.STREAMING:
                self.stop()
            self._state = LifecycleState.RELEASED

    def __enter__(self) -> ReplaySource:
        self.prepare()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.stop()
        finally:
            self.release()
