"""Sample-source lifecycle and BrainFlow-backed acquisition."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol, cast

import numpy as np

from chrono_link.config import BoardConfig, ConfigError, SignalConfig
from chrono_link.contracts import (
    LifecycleState,
    ResolvedBoardInfo,
    SampleChunk,
    SourceEvent,
)


class SourceError(RuntimeError):
    """A sample source failed or was used in an invalid state."""


class SourceStallError(SourceError):
    """A streaming source produced no samples for the configured timeout."""


class InvalidSampleError(SourceError):
    """A source returned malformed or non-finite sample data."""


class SampleSource(Protocol):
    """Minimal lifecycle for chronological, consume-once samples."""

    @property
    def state(self) -> LifecycleState: ...

    def prepare(self) -> ResolvedBoardInfo: ...

    def start(self) -> None: ...

    def read_new(self) -> SampleChunk | None: ...

    def stop(self) -> None: ...

    def release(self) -> None: ...


class _BoardLike(Protocol):
    def prepare_session(self) -> None: ...

    def start_stream(self) -> None: ...

    def get_board_data(self) -> np.ndarray[Any, Any]: ...

    def stop_stream(self) -> None: ...

    def release_session(self) -> None: ...


class _BoardApi(Protocol):
    @staticmethod
    def get_sampling_rate(board_id: int) -> int: ...

    @staticmethod
    def get_eeg_channels(board_id: int) -> list[int]: ...

    @staticmethod
    def get_eeg_names(board_id: int) -> list[str]: ...

    @staticmethod
    def get_timestamp_channel(board_id: int) -> int: ...

    @staticmethod
    def get_package_num_channel(board_id: int) -> int: ...

    @staticmethod
    def get_marker_channel(board_id: int) -> int: ...

    @staticmethod
    def get_num_rows(board_id: int) -> int: ...


BoardFactory = Callable[[int, Any], _BoardLike]


def _brainflow_dependencies() -> tuple[type[Any], type[Any], type[Any]]:
    """Import BrainFlow lazily so unrelated core modules stay independent."""
    from brainflow.board_shim import (  # type: ignore[import-untyped]
        BoardIds,
        BoardShim,
        BrainFlowInputParams,
    )

    return BoardIds, BoardShim, BrainFlowInputParams


class BrainFlowSource:
    """Own one BrainFlow session and drain every newly accumulated sample once."""

    def __init__(
        self,
        board_config: BoardConfig,
        signal_config: SignalConfig | None = None,
        *,
        stall_timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        board_factory: BoardFactory | None = None,
        board_api: _BoardApi | None = None,
        board_id: int | None = None,
        input_params: Any | None = None,
    ) -> None:
        if stall_timeout_s <= 0:
            raise ValueError("stall_timeout_s must be positive")
        self._config = board_config
        self._signal_config = signal_config or SignalConfig()
        self._stall_timeout_s = stall_timeout_s
        self._clock = clock
        self._factory = board_factory
        self._api = board_api
        self._requested_board_id = board_id
        self._input_params = input_params
        self._board: _BoardLike | None = None
        self._info: ResolvedBoardInfo | None = None
        self._state = LifecycleState.NEW
        self._session_acquired = False
        self._stream_started = False
        self._stop_called = False
        self._release_called = False
        self._next_sequence = 0
        self._previous_counter: int | None = None
        self._previous_timestamp: float | None = None
        self._last_sample_at: float | None = None
        self._pending: deque[SampleChunk] = deque()
        self._consecutive_invalid_chunks = 0
        self._reset_before_next_valid = False

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def info(self) -> ResolvedBoardInfo:
        if self._info is None:
            raise SourceError("source is not prepared")
        return self._info

    def _build_runtime(self) -> tuple[int, _BoardApi, BoardFactory, Any]:
        if self._config.profile == "cyton" and not self._config.serial_port:
            raise ConfigError("cyton profile requires serial_port before prepare_session()")

        if self._api is not None and self._factory is not None:
            if self._requested_board_id is None:
                raise ValueError("injected board dependencies require board_id")
            return (
                self._requested_board_id,
                self._api,
                self._factory,
                self._input_params,
            )

        board_ids, board_shim, params_type = _brainflow_dependencies()
        resolved_id = (
            int(board_ids.SYNTHETIC_BOARD.value)
            if self._config.profile == "synthetic"
            else int(board_ids.CYTON_BOARD.value)
        )
        params = params_type()
        if self._config.serial_port is not None:
            params.serial_port = self._config.serial_port
        factory = cast(BoardFactory, board_shim)
        return resolved_id, cast(_BoardApi, board_shim), factory, params

    def prepare(self) -> ResolvedBoardInfo:
        if self._state is LifecycleState.PREPARED:
            return self.info
        if self._state is not LifecycleState.NEW:
            raise SourceError(f"cannot prepare source in state {self._state.name}")

        try:
            board_id, api, factory, params = self._build_runtime()
            info = self._resolve_info(board_id, api)
            self._signal_config.validate_for(info.fs)
            board = factory(board_id, params)
            board.prepare_session()
        except Exception:
            self._state = LifecycleState.FAILED
            raise

        self._board = board
        self._info = info
        self._session_acquired = True
        self._state = LifecycleState.PREPARED
        return info

    def _resolve_info(self, board_id: int, api: _BoardApi) -> ResolvedBoardInfo:
        eeg_rows = tuple(int(row) for row in api.get_eeg_channels(board_id))
        eeg_names = tuple(str(name) for name in api.get_eeg_names(board_id))
        if len(eeg_rows) != len(eeg_names) or not eeg_rows:
            raise ConfigError("BrainFlow returned inconsistent EEG rows and names")
        name_to_row = dict(zip(eeg_names, eeg_rows, strict=True))

        missing_decode = [name for name in self._config.decode_channels if name not in name_to_row]
        if missing_decode:
            raise ConfigError(
                f"decode channels {missing_decode} are unavailable; available={list(eeg_names)}"
            )
        if self._config.record_channels == "all":
            record_names = eeg_names
        else:
            missing_record = [
                name for name in self._config.record_channels if name not in name_to_row
            ]
            if missing_record:
                raise ConfigError(
                    f"record channels {missing_record} are unavailable; available={list(eeg_names)}"
                )
            record_names = cast(tuple[str, ...], self._config.record_channels)

        return ResolvedBoardInfo(
            board_id=board_id,
            fs=int(api.get_sampling_rate(board_id)),
            eeg_names=eeg_names,
            eeg_rows=eeg_rows,
            timestamp_row=int(api.get_timestamp_channel(board_id)),
            package_num_row=int(api.get_package_num_channel(board_id)),
            marker_row=int(api.get_marker_channel(board_id)),
            num_rows=int(api.get_num_rows(board_id)),
            decode_rows=tuple(name_to_row[name] for name in self._config.decode_channels),
            record_rows=tuple(name_to_row[name] for name in record_names),
        )

    def start(self) -> None:
        if self._state is LifecycleState.STREAMING:
            return
        if self._state is not LifecycleState.PREPARED or self._board is None:
            raise SourceError(f"cannot start source in state {self._state.name}")
        try:
            self._board.start_stream()
        except Exception:
            self._state = LifecycleState.FAILED
            raise
        self._stream_started = True
        self._state = LifecycleState.STREAMING
        self._last_sample_at = self._clock()

    def read_new(self) -> SampleChunk | None:
        if self._state is not LifecycleState.STREAMING or self._board is None:
            raise SourceError(f"cannot read source in state {self._state.name}")
        if self._pending:
            return self._pending.popleft()

        raw = np.asarray(self._board.get_board_data(), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[0] != self.info.num_rows:
            self._handle_invalid_chunk(
                InvalidSampleError(
                    f"BrainFlow data must have shape ({self.info.num_rows}, samples)"
                )
            )
            return None
        if raw.shape[1] == 0:
            started = self._last_sample_at
            if started is None:
                raise SourceError("source timing state is unavailable while streaming")
            elapsed = self._clock() - started
            if elapsed >= self._stall_timeout_s:
                raise SourceStallError(f"source stalled for {elapsed:.3f} seconds")
            return None

        try:
            chunks = self._convert_and_split(raw)
        except InvalidSampleError as exc:
            self._handle_invalid_chunk(exc)
            return None
        self._consecutive_invalid_chunks = 0
        if self._reset_before_next_valid:
            first = chunks[0]
            reset = SourceEvent(
                code="RESET",
                before_sequence=int(first.sequence[0]),
                details={"reason": "invalid_source_chunk"},
            )
            chunks[0] = SampleChunk(
                eeg=first.eeg,
                timestamps=first.timestamps,
                sequence=first.sequence,
                package_counter=first.package_counter,
                channel_names=first.channel_names,
                fs=first.fs,
                markers=first.markers,
                events_before=(reset, *first.events_before),
            )
            self._reset_before_next_valid = False
        self._last_sample_at = self._clock()
        self._pending.extend(chunks)
        return self._pending.popleft()

    def _handle_invalid_chunk(self, error: InvalidSampleError) -> None:
        self._consecutive_invalid_chunks += 1
        self._reset_before_next_valid = True
        self._last_sample_at = self._clock()
        if self._consecutive_invalid_chunks >= 3:
            self._state = LifecycleState.FAILED
            raise error
        return None

    def _convert_and_split(self, raw: np.ndarray[Any, Any]) -> list[SampleChunk]:
        info = self.info
        samples = raw.shape[1]
        eeg = np.array(raw[np.asarray(info.record_rows), :], dtype=np.float64, copy=True)
        timestamps = np.array(raw[info.timestamp_row, :], dtype=np.float64, copy=True)
        if not np.isfinite(eeg).all() or not np.isfinite(timestamps).all():
            raise InvalidSampleError("source returned non-finite EEG or timestamps")
        timestamp_diffs = np.diff(timestamps)
        if np.any(timestamp_diffs <= 0) or (
            self._previous_timestamp is not None and timestamps[0] <= self._previous_timestamp
        ):
            raise InvalidSampleError("source timestamps must be finite and strictly increasing")

        counter_values = raw[info.package_num_row, :]
        if not np.isfinite(counter_values).all() or not np.allclose(
            counter_values, np.rint(counter_values), atol=1e-9, rtol=0.0
        ):
            raise InvalidSampleError("source package counters must be finite integers")
        counters = np.rint(counter_values).astype(np.int64)
        markers = np.array(raw[info.marker_row, :], dtype=np.float64, copy=True)
        if not np.isfinite(markers).all():
            raise InvalidSampleError("source markers must be finite")

        sequence = np.arange(
            self._next_sequence,
            self._next_sequence + samples,
            dtype=np.int64,
        )
        break_indices: list[int] = []
        if (
            self._previous_counter is not None
            and (int(counters[0]) - self._previous_counter) % 256 != 1
        ):
            break_indices.append(0)
        for index in np.flatnonzero((np.diff(counters) % 256) != 1) + 1:
            break_indices.append(int(index))

        record_name_by_row = dict(zip(info.eeg_rows, info.eeg_names, strict=True))
        channel_names = tuple(record_name_by_row[row] for row in info.record_rows)
        starts = sorted({0, *break_indices})
        stops = [*starts[1:], samples]
        chunks: list[SampleChunk] = []
        for start, stop in zip(starts, stops, strict=True):
            event: tuple[SourceEvent, ...] = ()
            if start in break_indices:
                before = int(sequence[start])
                previous = self._previous_counter if start == 0 else int(counters[start - 1])
                if previous is None:
                    raise SourceError("package-gap state is missing the previous counter")
                event = (
                    SourceEvent(
                        code="PACKAGE_GAP",
                        before_sequence=before,
                        details={
                            "previous_counter": int(previous),
                            "current_counter": int(counters[start]),
                        },
                    ),
                )
            chunks.append(
                SampleChunk(
                    eeg=eeg[:, start:stop],
                    timestamps=timestamps[start:stop],
                    sequence=sequence[start:stop],
                    package_counter=counters[start:stop],
                    channel_names=channel_names,
                    fs=info.fs,
                    markers=markers[start:stop],
                    events_before=event,
                )
            )

        self._next_sequence += samples
        self._previous_counter = int(counters[-1])
        self._previous_timestamp = float(timestamps[-1])
        return chunks

    def stop(self) -> None:
        if self._stop_called or not self._stream_started:
            return
        self._stop_called = True
        try:
            board = self._board
            if board is None:
                raise SourceError("stream is active without an acquired board")
            board.stop_stream()
        finally:
            self._stream_started = False
            if self._state is not LifecycleState.RELEASED:
                self._state = LifecycleState.STOPPED

    def release(self) -> None:
        if self._release_called or not self._session_acquired:
            return
        if self._stream_started:
            self.stop()
        self._release_called = True
        try:
            board = self._board
            if board is None:
                raise SourceError("session is acquired without a board instance")
            board.release_session()
        finally:
            self._session_acquired = False
            self._state = LifecycleState.RELEASED

    def __enter__(self) -> BrainFlowSource:
        self.prepare()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.stop()
        finally:
            self.release()
