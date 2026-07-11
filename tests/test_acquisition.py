from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from chrono_link.acquisition import (
    BrainFlowSource,
    InvalidSampleError,
    SourceError,
    SourceStallError,
)
from chrono_link.config import BoardConfig, ConfigError, SignalConfig
from chrono_link.contracts import LifecycleState
from chrono_link.filters import CausalFilterBank
from chrono_link.windows import WindowAssembler


class FakeApi:
    @staticmethod
    def get_sampling_rate(board_id: int) -> int:
        return 250

    @staticmethod
    def get_eeg_channels(board_id: int) -> list[int]:
        return [1, 2]

    @staticmethod
    def get_eeg_names(board_id: int) -> list[str]:
        return ["C3", "C4"]

    @staticmethod
    def get_timestamp_channel(board_id: int) -> int:
        return 3

    @staticmethod
    def get_package_num_channel(board_id: int) -> int:
        return 0

    @staticmethod
    def get_marker_channel(board_id: int) -> int:
        return 4

    @staticmethod
    def get_num_rows(board_id: int) -> int:
        return 5


@dataclass
class FakeBoard:
    reads: list[np.ndarray[Any, Any]]
    prepared: int = 0
    started: int = 0
    stopped: int = 0
    released: int = 0

    def prepare_session(self) -> None:
        self.prepared += 1

    def start_stream(self) -> None:
        self.started += 1

    def get_board_data(self) -> np.ndarray[Any, Any]:
        if self.reads:
            return self.reads.pop(0)
        return np.empty((5, 0), dtype=np.float64)

    def stop_stream(self) -> None:
        self.stopped += 1

    def release_session(self) -> None:
        self.released += 1


@dataclass
class FakeClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


def make_raw(counters: list[int], start_time: float = 1.0) -> np.ndarray[Any, Any]:
    samples = len(counters)
    raw = np.zeros((5, samples), dtype=np.float64)
    raw[0] = counters
    raw[1] = np.arange(samples) + 10.0
    raw[2] = np.arange(samples) + 20.0
    raw[3] = start_time + np.arange(samples) / 250.0
    return raw


def test_brainflow_source_splits_gaps_and_cleans_up_once() -> None:
    board = FakeBoard([make_raw([10, 11, 15, 16])])
    source = BrainFlowSource(
        BoardConfig(),
        board_factory=lambda board_id, params: board,
        board_api=FakeApi(),
        board_id=-1,
        input_params=object(),
    )

    info = source.prepare()
    assert info.decode_rows == (1, 2)
    source.start()
    first = source.read_new()
    second = source.read_new()
    assert first is not None and second is not None
    np.testing.assert_array_equal(first.sequence, [0, 1])
    assert first.events_before == ()
    np.testing.assert_array_equal(second.sequence, [2, 3])
    assert second.events_before[0].code == "PACKAGE_GAP"
    assert second.events_before[0].before_sequence == 2

    source.stop()
    source.stop()
    source.release()
    source.release()
    assert source.state is LifecycleState.RELEASED
    assert (board.prepared, board.started, board.stopped, board.released) == (1, 1, 1, 1)


def test_brainflow_source_detects_cross_read_gap_and_stall() -> None:
    clock = FakeClock()
    board = FakeBoard([make_raw([254, 255]), make_raw([2, 3], start_time=2.0)])
    source = BrainFlowSource(
        BoardConfig(),
        clock=clock,
        stall_timeout_s=2.0,
        board_factory=lambda board_id, params: board,
        board_api=FakeApi(),
        board_id=-1,
    )
    source.prepare()
    source.start()
    assert source.read_new() is not None
    post_gap = source.read_new()
    assert post_gap is not None and post_gap.events_before[0].before_sequence == 2
    assert source.read_new() is None
    clock.value = 2.0
    with pytest.raises(SourceStallError):
        source.read_new()
    source.release()


def test_cyton_serial_is_required_only_when_preparing() -> None:
    config = BoardConfig(profile="cyton")
    source = BrainFlowSource(
        config,
        board_factory=lambda board_id, params: FakeBoard([]),
        board_api=FakeApi(),
        board_id=0,
    )
    with pytest.raises(ConfigError, match="serial_port"):
        source.prepare()
    assert source.state is LifecycleState.FAILED


def test_source_state_errors_and_idempotent_prepare() -> None:
    board = FakeBoard([make_raw([0, 1])])
    source = BrainFlowSource(
        BoardConfig(),
        board_factory=lambda board_id, params: board,
        board_api=FakeApi(),
        board_id=-1,
    )
    with pytest.raises(SourceError, match="start"):
        source.start()
    with pytest.raises(SourceError, match="read"):
        source.read_new()
    first = source.prepare()
    assert source.prepare() is first
    source.start()
    assert source.read_new() is not None
    with pytest.raises(SourceError, match="prepare"):
        source.prepare()
    source.release()
    assert board.stopped == board.released == 1


class MissingApi(FakeApi):
    @staticmethod
    def get_eeg_names(board_id: int) -> list[str]:
        return ["F3", "F4"]


class InconsistentApi(FakeApi):
    @staticmethod
    def get_eeg_names(board_id: int) -> list[str]:
        return ["C3"]


def test_source_resolution_errors_fail_before_session_acquisition() -> None:
    board = FakeBoard([])
    for api, match in ((MissingApi(), "unavailable"), (InconsistentApi(), "inconsistent")):
        source = BrainFlowSource(
            BoardConfig(),
            board_factory=lambda board_id, params: board,
            board_api=api,
            board_id=-1,
        )
        with pytest.raises(ConfigError, match=match):
            source.prepare()
        assert board.prepared == 0


@pytest.mark.parametrize("mutation", ["shape", "eeg", "timestamp", "counter", "marker"])
def test_source_rejects_malformed_board_arrays(mutation: str) -> None:
    raw = make_raw([0, 1])
    if mutation == "shape":
        raw = raw[:4]
    elif mutation == "eeg":
        raw[1, 0] = np.nan
    elif mutation == "timestamp":
        raw[3, 1] = raw[3, 0]
    elif mutation == "counter":
        raw[0, 0] = 0.5
    else:
        raw[4, 0] = np.inf
    board = FakeBoard([raw.copy(), raw.copy(), raw.copy()])
    source = BrainFlowSource(
        BoardConfig(),
        board_factory=lambda board_id, params: board,
        board_api=FakeApi(),
        board_id=-1,
    )
    source.prepare()
    source.start()
    assert source.read_new() is None
    assert source.read_new() is None
    with pytest.raises(InvalidSampleError):
        source.read_new()
    source.release()


def test_valid_chunk_after_invalid_input_carries_reset_event() -> None:
    invalid = make_raw([0, 1])
    invalid[3, 1] = invalid[3, 0]
    board = FakeBoard([invalid, make_raw([2, 3], start_time=2.0)])
    source = BrainFlowSource(
        BoardConfig(),
        board_factory=lambda board_id, params: board,
        board_api=FakeApi(),
        board_id=-1,
    )
    source.prepare()
    source.start()
    assert source.read_new() is None
    recovered = source.read_new()
    assert recovered is not None
    assert recovered.events_before[0].code == "RESET"
    source.release()


def test_real_brainflow_synthetic_source_drains_new_samples() -> None:
    source = BrainFlowSource(BoardConfig())
    with source:
        info = source.prepare()
        assert info.fs == 250
        assert info.decode_rows == (2, 4)
        source.start()
        chunk = None
        deadline = time.monotonic() + 1.0
        while chunk is None and time.monotonic() < deadline:
            chunk = source.read_new()
            if chunk is None:
                time.sleep(0.01)
        assert chunk is not None
        assert chunk.channel_names[:2] == ("Fz", "C3")


def test_real_synthetic_source_reaches_first_settled_window_within_6_5_seconds() -> None:
    signal_config = SignalConfig()
    source = BrainFlowSource(BoardConfig(), signal_config)
    with source:
        info = source.prepare()
        filters = CausalFilterBank(info.fs, ("C3", "C4"), signal_config)
        windows = WindowAssembler(("C3", "C4"), info.fs)
        source.start()
        emitted = []
        deadline = time.monotonic() + 6.5
        while not emitted and time.monotonic() < deadline:
            sample = source.read_new()
            if sample is None:
                time.sleep(0.01)
                continue
            if sample.events_before:
                filters.reset()
                windows.reset()
            emitted.extend(windows.push(filters.process(sample)))
        assert emitted
        assert emitted[0].start_sequence == 1000
