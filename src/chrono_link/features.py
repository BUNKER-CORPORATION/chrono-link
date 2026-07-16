"""Stable temporal, spectral, phase, and quality feature contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy import signal  # type: ignore[import-untyped]

from chrono_link.config import SignalConfig
from chrono_link.contracts import BandSpec, WindowBundle, WindowQuality

POWER_FLOOR = 1e-12
HILBERT_GUARD = 25
FEATURE_SCHEMA_VERSION = 1


class FeatureError(RuntimeError):
    """A window cannot produce valid features."""


@dataclass(frozen=True)
class FeatureVector:
    """One finite, named vector with a compatibility fingerprint."""

    values: NDArray[np.float64]
    names: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if self.values.dtype != np.float64 or self.values.ndim != 1:
            raise ValueError("feature values must be a one-dimensional float64 array")
        if self.values.shape[0] != len(self.names):
            raise ValueError("feature names and values must have equal length")
        if not np.isfinite(self.values).all():
            raise ValueError("feature values must be finite")


@dataclass(frozen=True)
class TemporalFeatures:
    """Full Hilbert amplitude/phase arrays kept separate from summary vectors."""

    amplitude: NDArray[np.float64]
    phase: NDArray[np.float64]
    valid_slice: slice
    band_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.amplitude.shape != self.phase.shape or self.amplitude.ndim != 3:
            raise ValueError("amplitude and phase must have shape (bands, channels, samples)")
        if self.amplitude.shape[0] != len(self.band_names):
            raise ValueError("band_names must match temporal feature bands")
        if not np.isfinite(self.amplitude).all() or not np.isfinite(self.phase).all():
            raise ValueError("temporal features must be finite")


def welch_psd(
    values: NDArray[np.float64],
    fs: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the normative periodic-Hann Welch density along the final axis."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 2:
        raise FeatureError("Welch PSD requires at least two samples")
    if not np.isfinite(array).all() or fs <= 0:
        raise FeatureError("Welch input must be finite with a positive fs")
    nperseg = min(250, int(array.shape[-1]))
    frequencies, density = signal.welch(
        array,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nperseg,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
    )
    return (
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(density, dtype=np.float64),
    )


def integrate_band_power(
    frequencies: NDArray[np.float64],
    density: NDArray[np.float64],
    low_hz: float,
    high_hz: float,
) -> NDArray[np.float64]:
    """Integrate inclusive Welch bins within one band."""
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if np.count_nonzero(mask) < 2:
        raise FeatureError(f"band {low_hz:g}-{high_hz:g} Hz has fewer than two bins")
    return np.asarray(np.trapezoid(density[..., mask], frequencies[mask], axis=-1))


def log_power_db(power: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Convert power to decibels with the normative numeric floor."""
    return np.asarray(10.0 * np.log10(np.maximum(power, POWER_FLOOR)), dtype=np.float64)


def hilbert_features(
    bands: NDArray[np.float64],
    *,
    guard: int = HILBERT_GUARD,
) -> tuple[NDArray[np.float64], NDArray[np.float64], slice]:
    """Return full amplitude/wrapped phase plus the guarded summary slice."""
    values = np.asarray(bands, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] <= 2 * guard or guard < 0:
        raise FeatureError("Hilbert input is too short for the requested edge guard")
    if not np.isfinite(values).all():
        raise FeatureError("Hilbert input must be finite")
    analytic = signal.hilbert(values, axis=-1)
    return (
        np.asarray(np.abs(analytic), dtype=np.float64),
        np.asarray(np.angle(analytic), dtype=np.float64),
        slice(guard, values.shape[-1] - guard),
    )


def phase_summary(
    phase_i: NDArray[np.float64],
    phase_j: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Return mean cosine, mean sine, and PLV for phase_i - phase_j."""
    if phase_i.shape != phase_j.shape or phase_i.size == 0:
        raise FeatureError("phase arrays must have equal nonzero shape")
    difference = phase_i - phase_j
    mean_complex = np.mean(np.exp(1j * difference))
    return float(mean_complex.real), float(mean_complex.imag), float(abs(mean_complex))


def normalized_spectral_entropy(
    frequencies: NDArray[np.float64],
    density: NDArray[np.float64],
    *,
    low_hz: float = 1.0,
    high_hz: float = 40.0,
) -> NDArray[np.float64]:
    """Compute inclusive-band Shannon entropy normalized by log(bin count)."""
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    bins = int(np.count_nonzero(mask))
    if bins < 2:
        raise FeatureError("entropy band has fewer than two bins")
    selected = np.asarray(density[..., mask], dtype=np.float64)
    totals = np.sum(selected, axis=-1, keepdims=True)
    probabilities = np.divide(
        selected,
        totals,
        out=np.zeros_like(selected),
        where=totals > 0,
    )
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    entropy = -np.sum(terms, axis=-1) / np.log(bins)
    return np.asarray(np.clip(entropy, 0.0, 1.0), dtype=np.float64)


def assess_window_quality(
    window: WindowBundle,
    *,
    flat_abs_tol: float = 1e-12,
    flat_rel_tol: float = 1e-8,
) -> WindowQuality:
    """Reject non-finite or flat decode channels before features/covariance."""
    reasons: list[str] = []
    if not np.isfinite(window.cleaned).all() or any(
        not np.isfinite(values).all() for values in window.bands.values()
    ):
        reasons.append("non_finite")
    for index, channel in enumerate(window.channel_names):
        values = window.cleaned[index]
        threshold = max(flat_abs_tol, flat_rel_tol * float(np.max(np.abs(values))))
        if float(np.std(values)) <= threshold:
            reasons.append(f"flat_channel:{channel}")
    return WindowQuality(valid=not reasons, reasons=tuple(reasons))


class FeatureExtractor:
    """Build temporal arrays and the stable named summary vector."""

    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()
        by_name = {band.name: band for band in self.config.filter_bands}
        self.vector_bands = tuple(
            by_name[name] for name in self.config.vector_feature_bands
        )

    def temporal(self, window: WindowBundle) -> TemporalFeatures:
        missing = [band.name for band in self.vector_bands if band.name not in window.bands]
        if missing:
            raise FeatureError(f"window is missing vector bands: {missing}")
        values = np.stack([window.bands[band.name] for band in self.vector_bands])
        amplitude, phase, valid_slice = hilbert_features(values)
        return TemporalFeatures(
            amplitude=amplitude,
            phase=phase,
            valid_slice=valid_slice,
            band_names=tuple(band.name for band in self.vector_bands),
        )

    def _names(self, channel_names: tuple[str, ...]) -> tuple[str, ...]:
        names: list[str] = []
        for band in self.vector_bands:
            for channel in channel_names:
                names.extend(
                    (
                        f"{band.name}.{channel}.log_power_db",
                        f"{band.name}.{channel}.envelope_mean",
                        f"{band.name}.{channel}.envelope_std",
                    )
                )
            for left, right in combinations(channel_names, 2):
                names.extend(
                    (
                        f"{band.name}.{left}.{right}.phase_cos",
                        f"{band.name}.{left}.{right}.phase_sin",
                        f"{band.name}.{left}.{right}.plv",
                    )
                )
        names.extend(
            f"broadband.{channel}.spectral_entropy" for channel in channel_names
        )
        return tuple(names)

    def names(self, channel_names: tuple[str, ...]) -> tuple[str, ...]:
        """Return the stable ordered feature schema for a channel order."""
        return self._names(channel_names)

    def fingerprint(self, channel_names: tuple[str, ...]) -> str:
        payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "names": self._names(channel_names),
            "bands": [
                {
                    "name": band.name,
                    "low_hz": band.low_hz,
                    "high_hz": band.high_hz,
                }
                for band in self.vector_bands
            ],
            "welch": {
                "window": "hann_periodic",
                "nperseg_max": 250,
                "overlap_fraction": 0.5,
                "scaling": "density",
                "detrend": "constant",
            },
            "hilbert_guard": HILBERT_GUARD,
            "power_floor": POWER_FLOOR,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def vector(self, window: WindowBundle) -> FeatureVector:
        quality = assess_window_quality(
            window,
            flat_abs_tol=self.config.flat_abs_tol,
            flat_rel_tol=self.config.flat_rel_tol,
        )
        if not quality.valid:
            raise FeatureError(f"window failed quality: {quality.reasons}")
        temporal = self.temporal(window)
        values: list[float] = []
        for band_index, band in enumerate(self.vector_bands):
            frequencies, density = welch_psd(window.bands[band.name], window.fs)
            powers = integrate_band_power(
                frequencies,
                density,
                band.low_hz,
                band.high_hz,
            )
            for channel_index in range(len(window.channel_names)):
                amplitude = temporal.amplitude[
                    band_index, channel_index, temporal.valid_slice
                ]
                values.extend(
                    (
                        float(log_power_db(powers[channel_index])),
                        float(np.mean(amplitude)),
                        float(np.std(amplitude)),
                    )
                )
            for left, right in combinations(range(len(window.channel_names)), 2):
                values.extend(
                    phase_summary(
                        temporal.phase[band_index, left, temporal.valid_slice],
                        temporal.phase[band_index, right, temporal.valid_slice],
                    )
                )

        if "broadband" not in window.bands:
            raise FeatureError("window is missing broadband for spectral entropy")
        broadband_frequency, broadband_density = welch_psd(
            window.bands["broadband"], window.fs
        )
        values.extend(
            float(value)
            for value in normalized_spectral_entropy(
                broadband_frequency,
                broadband_density,
            )
        )
        names = self._names(window.channel_names)
        return FeatureVector(
            values=np.asarray(values, dtype=np.float64),
            names=names,
            fingerprint=self.fingerprint(window.channel_names),
        )


def band_by_name(config: SignalConfig, name: str) -> BandSpec:
    """Resolve one configured band by exact name."""
    for band in config.filter_bands:
        if band.name == name:
            return band
    raise FeatureError(f"unknown configured band {name!r}")
