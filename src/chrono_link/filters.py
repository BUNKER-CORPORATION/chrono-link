"""Deterministic causal and offline EEG filter banks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal  # type: ignore[import-untyped]

from chrono_link.config import SignalConfig
from chrono_link.contracts import FilteredChunk, SampleChunk


class FilterError(RuntimeError):
    """Signal data violates the filter-bank contract."""


class OfflineWindowTooShort(FilterError):
    """A zero-phase filter cannot pad the supplied signal."""


@dataclass(frozen=True)
class FilterDesign:
    """Immutable SOS coefficients and their compatibility fingerprint."""

    fs: int
    cleaned_sos: NDArray[np.float64]
    band_sos: MappingProxyType[str, NDArray[np.float64]]
    notch_frequencies: tuple[float, ...]
    fingerprint: str


def _as_float64_sos(values: NDArray[Any]) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def design_filters(fs: int, config: SignalConfig | None = None) -> FilterDesign:
    """Design the normative high-pass, notch cascade, and band-pass branches."""
    selected = config or SignalConfig()
    selected.validate_for(fs)
    highpass = _as_float64_sos(
        signal.butter(
            2,
            selected.dc_highpass_hz,
            btype="highpass",
            fs=fs,
            output="sos",
        )
    )
    notch_frequencies = tuple(
        float(selected.mains_hz * harmonic)
        for harmonic in range(1, int((fs / 2) // selected.mains_hz) + 1)
        if selected.mains_hz * harmonic < fs / 2
    )
    notch_sections: list[NDArray[np.float64]] = []
    for frequency in notch_frequencies:
        numerator, denominator = signal.iirnotch(
            frequency,
            selected.notch_q,
            fs=fs,
        )
        notch_sections.append(_as_float64_sos(signal.tf2sos(numerator, denominator)))
    cleaned_sos = np.concatenate([highpass, *notch_sections], axis=0)
    bands = {
        band.name: _as_float64_sos(
            signal.butter(
                4,
                (band.low_hz, band.high_hz),
                btype="bandpass",
                fs=fs,
                output="sos",
            )
        )
        for band in selected.filter_bands
    }
    metadata = {
        "schema": 1,
        "fs": fs,
        "dc_highpass_hz": selected.dc_highpass_hz,
        "mains_hz": selected.mains_hz,
        "notch_q": selected.notch_q,
        "notches": notch_frequencies,
        "bands": [
            {"name": band.name, "low_hz": band.low_hz, "high_hz": band.high_hz}
            for band in selected.filter_bands
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(cleaned_sos.astype("<f8", copy=False).tobytes())
    for name in sorted(bands):
        digest.update(name.encode("utf-8"))
        digest.update(bands[name].astype("<f8", copy=False).tobytes())
    return FilterDesign(
        fs=fs,
        cleaned_sos=cleaned_sos,
        band_sos=MappingProxyType(bands),
        notch_frequencies=notch_frequencies,
        fingerprint=digest.hexdigest(),
    )


class CausalFilterBank:
    """Stateful one-pass filtering whose output is invariant to chunk boundaries."""

    preprocessing_mode = "causal"

    def __init__(
        self,
        fs: int,
        channel_names: tuple[str, ...],
        config: SignalConfig | None = None,
    ) -> None:
        if not channel_names or len(set(channel_names)) != len(channel_names):
            raise ValueError("channel_names must be non-empty and unique")
        self.config = config or SignalConfig()
        self.design = design_filters(fs, self.config)
        self.fs = fs
        self.channel_names = channel_names
        self._cleaned_state: NDArray[np.float64] | None = None
        self._band_states: dict[str, NDArray[np.float64] | None] = {
            name: None for name in self.design.band_sos
        }
        self._processed_since_reset = 0

    def reset(self) -> None:
        self._cleaned_state = None
        self._band_states = {name: None for name in self.design.band_sos}
        self._processed_since_reset = 0

    @staticmethod
    def _apply(
        sos: NDArray[np.float64],
        values: NDArray[np.float64],
        state: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if state is None:
            state = signal.sosfilt_zi(sos)[:, None, :] * values[:, 0][None, :, None]
        output, final_state = signal.sosfilt(sos, values, axis=1, zi=state)
        return (
            np.asarray(output, dtype=np.float64),
            np.asarray(final_state, dtype=np.float64),
        )

    def process(self, chunk: SampleChunk) -> FilteredChunk:
        if chunk.fs != self.fs:
            raise FilterError(f"chunk fs {chunk.fs} does not match filter fs {self.fs}")
        missing = [name for name in self.channel_names if name not in chunk.channel_names]
        if missing:
            raise FilterError(
                f"decode channels {missing} are unavailable in {list(chunk.channel_names)}"
            )
        indices = [chunk.channel_names.index(name) for name in self.channel_names]
        values = np.asarray(chunk.eeg[indices, :], dtype=np.float64)
        samples = values.shape[1]
        if samples == 0:
            return FilteredChunk(
                cleaned=values.copy(),
                bands={name: values.copy() for name in self.design.band_sos},
                timestamps=chunk.timestamps.copy(),
                sequence=chunk.sequence.copy(),
                settled=np.empty(0, dtype=np.bool_),
            )
        if not np.isfinite(values).all():
            raise FilterError("filter input must be finite")

        cleaned, self._cleaned_state = self._apply(
            self.design.cleaned_sos,
            values,
            self._cleaned_state,
        )
        bands: dict[str, NDArray[np.float64]] = {}
        for name, sos in self.design.band_sos.items():
            bands[name], self._band_states[name] = self._apply(
                sos,
                cleaned,
                self._band_states[name],
            )
        settled = (
            np.arange(
                self._processed_since_reset,
                self._processed_since_reset + samples,
                dtype=np.int64,
            )
            >= self.config.warmup_samples
        )
        self._processed_since_reset += samples
        return FilteredChunk(
            cleaned=cleaned,
            bands=bands,
            timestamps=chunk.timestamps.copy(),
            sequence=chunk.sequence.copy(),
            settled=settled,
        )


@dataclass(frozen=True)
class OfflineFiltered:
    """Zero-phase output that is intentionally incompatible with live models."""

    cleaned: NDArray[np.float64]
    bands: MappingProxyType[str, NDArray[np.float64]]
    preprocessing_mode: str = "offline_zero_phase"


class OfflineFilterBank:
    """Apply the same coefficient design with forward/backward zero-phase filtering."""

    preprocessing_mode = "offline_zero_phase"

    def __init__(self, fs: int, config: SignalConfig | None = None) -> None:
        self.design = design_filters(fs, config)

    @staticmethod
    def _apply(sos: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            return np.asarray(signal.sosfiltfilt(sos, values, axis=1), dtype=np.float64)
        except ValueError as exc:
            raise OfflineWindowTooShort(
                f"signal with {values.shape[1]} samples is too short for "
                f"zero-phase filtering: {exc}"
            ) from exc

    def process(self, values: NDArray[np.float64]) -> OfflineFiltered:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] == 0:
            raise FilterError("offline input must have shape (channels, nonzero samples)")
        if not np.isfinite(array).all():
            raise FilterError("offline input must be finite")
        cleaned = self._apply(self.design.cleaned_sos, array)
        bands = {
            name: self._apply(sos, cleaned)
            for name, sos in self.design.band_sos.items()
        }
        return OfflineFiltered(cleaned, MappingProxyType(bands))
