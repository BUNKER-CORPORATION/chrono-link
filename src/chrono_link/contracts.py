"""Shared, shape-explicit data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class LifecycleState(Enum):
    NEW = auto()
    PREPARED = auto()
    STREAMING = auto()
    STOPPED = auto()
    RELEASED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class BandSpec:
    name: str
    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        if not self.name or self.low_hz <= 0 or self.low_hz >= self.high_hz:
            raise ValueError(f"invalid band {self.name!r}: ({self.low_hz}, {self.high_hz})")


@dataclass(frozen=True)
class SourceEvent:
    code: Literal["PACKAGE_GAP", "REPLAY_GAP", "RESET", "STALL"]
    before_sequence: int
    details: Mapping[str, str | int | float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


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


@dataclass(frozen=True)
class SampleChunk:
    eeg: FloatArray
    timestamps: FloatArray
    sequence: IntArray
    package_counter: IntArray | None
    channel_names: tuple[str, ...]
    fs: int
    markers: FloatArray | None = None
    events_before: tuple[SourceEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.eeg.ndim != 2:
            raise ValueError("eeg must have shape (channels, samples)")
        samples = self.eeg.shape[1]
        if self.eeg.shape[0] != len(self.channel_names):
            raise ValueError("channel_names length does not match eeg rows")
        for name, value in (("timestamps", self.timestamps), ("sequence", self.sequence)):
            if value.shape != (samples,):
                raise ValueError(f"{name} must have shape ({samples},)")
        if self.package_counter is not None and self.package_counter.shape != (samples,):
            raise ValueError(f"package_counter must have shape ({samples},)")
        if self.markers is not None and self.markers.shape != (samples,):
            raise ValueError(f"markers must have shape ({samples},)")


@dataclass(frozen=True)
class FilteredChunk:
    cleaned: FloatArray
    bands: Mapping[str, FloatArray]
    timestamps: FloatArray
    sequence: IntArray
    settled: BoolArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "bands", MappingProxyType(dict(self.bands)))
        if self.cleaned.ndim != 2:
            raise ValueError("cleaned must have shape (channels, samples)")
        samples = self.cleaned.shape[1]
        if self.timestamps.shape != (samples,) or self.sequence.shape != (samples,):
            raise ValueError("filtered metadata length must match samples")
        if self.settled.shape != (samples,):
            raise ValueError("settled length must match samples")
        for name, values in self.bands.items():
            if values.shape != self.cleaned.shape:
                raise ValueError(f"band {name} shape must match cleaned")


@dataclass(frozen=True)
class WindowQuality:
    valid: bool = True
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowBundle:
    cleaned: FloatArray
    bands: Mapping[str, FloatArray]
    timestamps: FloatArray
    start_sequence: int
    end_sequence: int
    channel_names: tuple[str, ...]
    fs: int
    quality: WindowQuality = WindowQuality()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bands", MappingProxyType(dict(self.bands)))
        if self.cleaned.ndim != 2:
            raise ValueError("cleaned must have shape (channels, samples)")
        channels, samples = self.cleaned.shape
        if channels != len(self.channel_names):
            raise ValueError("channel_names length must match window channels")
        if self.timestamps.shape != (samples,):
            raise ValueError("timestamps length must match window samples")
        if self.end_sequence - self.start_sequence != samples:
            raise ValueError("window sequence range must match samples")
        if self.fs <= 0:
            raise ValueError("window sampling rate must be positive")
        for name, values in self.bands.items():
            if values.shape != self.cleaned.shape:
                raise ValueError(f"band {name} shape must match cleaned")


@dataclass(frozen=True)
class WindowBatch:
    cleaned: FloatArray
    bands: Mapping[str, FloatArray]
    channel_names: tuple[str, ...]
    fs: int
    sequence_ranges: IntArray
    quality: tuple[WindowQuality, ...]
    trial_ids: IntArray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bands", MappingProxyType(dict(self.bands)))
        examples = self.cleaned.shape[0]
        if self.cleaned.ndim != 3:
            raise ValueError("cleaned must have shape (examples, channels, samples)")
        if self.sequence_ranges.shape != (examples, 2):
            raise ValueError("sequence_ranges must have shape (examples, 2)")
        if len(self.quality) != examples:
            raise ValueError("quality length must match examples")
        if self.trial_ids is not None and self.trial_ids.shape != (examples,):
            raise ValueError("trial_ids must have shape (examples,)")

    @classmethod
    def from_windows(
        cls,
        windows: list[WindowBundle],
        trial_ids: IntArray | None = None,
    ) -> WindowBatch:
        if not windows:
            raise ValueError("at least one window is required")
        first = windows[0]
        branch_names = tuple(first.bands)
        for window in windows:
            if (
                window.fs != first.fs
                or window.channel_names != first.channel_names
                or window.cleaned.shape != first.cleaned.shape
                or tuple(window.bands) != branch_names
                or not window.quality.valid
            ):
                raise ValueError("windows have incompatible contracts or failed quality")
        return cls(
            cleaned=np.stack([window.cleaned for window in windows]),
            bands={
                name: np.stack([window.bands[name] for window in windows])
                for name in branch_names
            },
            channel_names=first.channel_names,
            fs=first.fs,
            sequence_ranges=np.asarray(
                [(window.start_sequence, window.end_sequence) for window in windows],
                dtype=np.int64,
            ),
            quality=tuple(window.quality for window in windows),
            trial_ids=trial_ids,
        )
