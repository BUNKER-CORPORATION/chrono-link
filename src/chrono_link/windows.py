"""Sample-indexed rolling window assembly."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chrono_link.contracts import FilteredChunk, WindowBundle, WindowQuality


class WindowError(RuntimeError):
    """Filtered chunks do not satisfy the windowing contract."""


class WindowAssembler:
    """Emit exact W-sample windows at endpoints W, W+H, W+2H, and so on."""

    def __init__(
        self,
        channel_names: tuple[str, ...],
        fs: int,
        *,
        window_samples: int = 250,
        hop_samples: int = 64,
    ) -> None:
        if not channel_names or fs <= 0:
            raise ValueError("channel_names and a positive fs are required")
        if window_samples <= 0 or hop_samples <= 0:
            raise ValueError("window_samples and hop_samples must be positive")
        self.channel_names = channel_names
        self.fs = fs
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self.reset()

    def reset(self) -> None:
        channels = len(self.channel_names)
        self._cleaned = np.empty((channels, 0), dtype=np.float64)
        self._bands: dict[str, NDArray[np.float64]] = {}
        self._timestamps = np.empty(0, dtype=np.float64)
        self._sequence = np.empty(0, dtype=np.int64)
        self._origin = 0
        self._seen = 0
        self._next_endpoint = self.window_samples
        self._last_sequence: int | None = None

    def _clear_eligible(self) -> None:
        channels = len(self.channel_names)
        self._cleaned = np.empty((channels, 0), dtype=np.float64)
        self._bands = {}
        self._timestamps = np.empty(0, dtype=np.float64)
        self._sequence = np.empty(0, dtype=np.int64)
        self._origin = 0
        self._seen = 0
        self._next_endpoint = self.window_samples
        self._last_sequence = None

    def push(self, chunk: FilteredChunk) -> list[WindowBundle]:
        samples = chunk.cleaned.shape[1]
        if samples == 0:
            return []
        if chunk.cleaned.shape[0] != len(self.channel_names):
            raise WindowError("filtered channel count changed")
        if not np.isfinite(chunk.cleaned).all() or not np.isfinite(chunk.timestamps).all():
            raise WindowError("filtered data and timestamps must be finite")
        if np.any(np.diff(chunk.sequence) != 1):
            raise WindowError("filtered sequence must be contiguous")

        start = 0
        unsettled = np.flatnonzero(~chunk.settled)
        if unsettled.size:
            self._clear_eligible()
            start = int(unsettled[-1]) + 1
        if start == samples:
            return []

        sequence = chunk.sequence[start:]
        if self._last_sequence is not None and int(sequence[0]) != self._last_sequence + 1:
            self._clear_eligible()
        branch_names = tuple(chunk.bands)
        if self._bands and tuple(self._bands) != branch_names:
            raise WindowError("filtered branch names changed")

        self._cleaned = np.concatenate((self._cleaned, chunk.cleaned[:, start:]), axis=1)
        if not self._bands:
            self._bands = {
                name: np.array(values[:, start:], copy=True)
                for name, values in chunk.bands.items()
            }
        else:
            for name, values in chunk.bands.items():
                self._bands[name] = np.concatenate(
                    (self._bands[name], values[:, start:]), axis=1
                )
        self._timestamps = np.concatenate((self._timestamps, chunk.timestamps[start:]))
        self._sequence = np.concatenate((self._sequence, sequence))
        self._seen += len(sequence)
        self._last_sequence = int(sequence[-1])

        windows: list[WindowBundle] = []
        while self._next_endpoint <= self._seen:
            window_start = self._next_endpoint - self.window_samples
            relative_start = window_start - self._origin
            relative_stop = self._next_endpoint - self._origin
            target = slice(relative_start, relative_stop)
            window_sequence = self._sequence[target]
            windows.append(
                WindowBundle(
                    cleaned=np.array(self._cleaned[:, target], copy=True),
                    bands={
                        name: np.array(values[:, target], copy=True)
                        for name, values in self._bands.items()
                    },
                    timestamps=np.array(self._timestamps[target], copy=True),
                    start_sequence=int(window_sequence[0]),
                    end_sequence=int(window_sequence[-1]) + 1,
                    channel_names=self.channel_names,
                    fs=self.fs,
                    quality=WindowQuality(),
                )
            )
            self._next_endpoint += self.hop_samples

        earliest_needed = self._next_endpoint - self.window_samples
        trim = earliest_needed - self._origin
        if trim > 0:
            self._cleaned = self._cleaned[:, trim:]
            self._bands = {name: values[:, trim:] for name, values in self._bands.items()}
            self._timestamps = self._timestamps[trim:]
            self._sequence = self._sequence[trim:]
            self._origin = earliest_needed
        return windows
