"""Deterministic two-class synthetic motor-imagery fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chrono_link.config import SignalConfig
from chrono_link.contracts import SampleChunk, WindowBatch
from chrono_link.features import integrate_band_power, welch_psd
from chrono_link.filters import CausalFilterBank
from chrono_link.windows import WindowAssembler

DEFAULT_SEED = 20260709


@dataclass(frozen=True)
class SyntheticMISession:
    """Raw balanced trials and their generation provenance."""

    raw_trials: NDArray[np.float64]
    labels: NDArray[np.int64]
    trial_ids: NDArray[np.int64]
    timestamps: NDArray[np.float64]
    channel_names: tuple[str, ...]
    fs: int
    metadata: MappingProxyType[str, Any]

    def __post_init__(self) -> None:
        trials, channels, samples = self.raw_trials.shape
        if self.raw_trials.dtype != np.float64 or self.raw_trials.ndim != 3:
            raise ValueError("raw_trials must be float64 (trials, channels, samples)")
        if channels != len(self.channel_names):
            raise ValueError("channel_names must match raw trial channels")
        if self.labels.shape != (trials,) or self.trial_ids.shape != (trials,):
            raise ValueError("labels and trial_ids must match trials")
        if self.timestamps.shape != (trials, samples):
            raise ValueError("timestamps must match trials and samples")
        if not np.isfinite(self.raw_trials).all() or not np.isfinite(self.timestamps).all():
            raise ValueError("synthetic arrays must be finite")


@dataclass(frozen=True)
class SyntheticWindowDataset:
    """Causally preprocessed grouped windows and their labels."""

    windows: WindowBatch
    labels: NDArray[np.int64]
    metadata: MappingProxyType[str, Any]


def _normalize_rms(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = values - np.mean(values)
    rms = float(np.sqrt(np.mean(centered**2)))
    if not np.isfinite(rms) or rms <= 0:
        raise RuntimeError("cannot normalize degenerate synthetic noise")
    return np.asarray(centered / rms, dtype=np.float64)


def _pink_noise(rng: np.random.Generator, samples: int, fs: int) -> NDArray[np.float64]:
    seed_signal = rng.normal(size=samples)
    spectrum = np.fft.rfft(seed_signal)
    frequencies = np.fft.rfftfreq(samples, d=1 / fs)
    spectrum[0] = 0.0
    spectrum[1:] *= 1.0 / np.sqrt(frequencies[1:])
    if samples % 2 == 0:
        spectrum[-1] = spectrum[-1].real + 0j
    return _normalize_rms(np.asarray(np.fft.irfft(spectrum, n=samples), dtype=np.float64))


class SyntheticMISessionGenerator:
    """Generate the locked, balanced Track A synthetic MI session."""

    def __init__(
        self,
        seed_or_rng: int | np.random.Generator = DEFAULT_SEED,
        *,
        trial_count: int = 160,
        fs: int = 250,
        duration_s: float = 6.0,
        snr_db: float = -6.0,
    ) -> None:
        if trial_count <= 0 or trial_count % 2:
            raise ValueError("trial_count must be a positive even number")
        if fs <= 0 or duration_s <= 0 or int(fs * duration_s) != fs * duration_s:
            raise ValueError("fs and duration must yield a positive integer sample count")
        self._rng = (
            seed_or_rng
            if isinstance(seed_or_rng, np.random.Generator)
            else np.random.default_rng(seed_or_rng)
        )
        self._seed = seed_or_rng if isinstance(seed_or_rng, int) else None
        self.trial_count = trial_count
        self.fs = fs
        self.samples = int(fs * duration_s)
        self.duration_s = duration_s
        self.snr_db = snr_db

    def generate(self) -> SyntheticMISession:
        labels = np.repeat(np.asarray([0, 1], dtype=np.int64), self.trial_count // 2)
        self._rng.shuffle(labels)
        trial_ids = np.arange(self.trial_count, dtype=np.int64)
        time = np.arange(self.samples, dtype=np.float64) / self.fs
        trials = np.empty((self.trial_count, 2, self.samples), dtype=np.float64)
        measured_snr: list[float] = []
        noise_rms = 10 ** (-self.snr_db / 20.0)

        for trial_index, label in enumerate(labels):
            mu_frequency = self._rng.uniform(9.0, 11.0)
            mu_rms = (0.501, 1.0) if label == 0 else (1.0, 0.501)
            for channel_index in range(2):
                mu_phase = self._rng.uniform(0.0, 2 * np.pi)
                beta_phase = self._rng.uniform(0.0, 2 * np.pi)
                mu = (
                    mu_rms[channel_index]
                    * np.sqrt(2.0)
                    * np.sin(2 * np.pi * mu_frequency * time + mu_phase)
                )
                beta = 0.25 * np.sqrt(2.0) * np.sin(2 * np.pi * 20.0 * time + beta_phase)
                white = _normalize_rms(
                    np.asarray(self._rng.normal(size=self.samples), dtype=np.float64)
                )
                pink = _pink_noise(self._rng, self.samples, self.fs)
                noise = _normalize_rms((white + pink) / np.sqrt(2.0)) * noise_rms
                trials[trial_index, channel_index] = mu + beta + noise
                if mu_rms[channel_index] == 1.0:
                    measured_snr.append(
                        20
                        * np.log10(
                            float(np.sqrt(np.mean(mu**2))) / float(np.sqrt(np.mean(noise**2)))
                        )
                    )

        realized_snr = float(np.mean(measured_snr))
        if abs(realized_snr - self.snr_db) > 0.1:
            raise RuntimeError(
                f"realized SNR {realized_snr:.3f} dB misses requested {self.snr_db:.3f} dB"
            )
        timestamps = np.broadcast_to(time, (self.trial_count, self.samples)).copy()
        metadata = MappingProxyType(
            {
                "schema_version": 1,
                "seed": self._seed if self._seed is not None else "external_generator",
                "trial_count": self.trial_count,
                "fs": self.fs,
                "duration_s": self.duration_s,
                "requested_snr_db": self.snr_db,
                "realized_snr_db": realized_snr,
                "class_mu_rms": {
                    "0": {"C3": 0.501, "C4": 1.0},
                    "1": {"C3": 1.0, "C4": 0.501},
                },
                "beta_rms": 0.25,
            }
        )
        return SyntheticMISession(
            raw_trials=trials,
            labels=labels,
            trial_ids=trial_ids,
            timestamps=timestamps,
            channel_names=("C3", "C4"),
            fs=self.fs,
            metadata=metadata,
        )


def preprocess_synthetic_session(
    session: SyntheticMISession,
    signal_config: SignalConfig | None = None,
) -> SyntheticWindowDataset:
    """Reset and causally preprocess every trial through the live DSP/window path."""
    config = signal_config or SignalConfig()
    windows = []
    labels: list[int] = []
    trial_ids: list[int] = []
    expected_per_trial = 4
    for trial_index in range(session.raw_trials.shape[0]):
        sequence_start = trial_index * session.raw_trials.shape[2]
        sequence = np.arange(
            sequence_start,
            sequence_start + session.raw_trials.shape[2],
            dtype=np.int64,
        )
        chunk = SampleChunk(
            eeg=session.raw_trials[trial_index],
            timestamps=session.timestamps[trial_index]
            + trial_index * session.raw_trials.shape[2] / session.fs,
            sequence=sequence,
            package_counter=sequence % 256,
            channel_names=session.channel_names,
            fs=session.fs,
            markers=np.zeros(session.raw_trials.shape[2], dtype=np.float64),
        )
        filter_bank = CausalFilterBank(session.fs, session.channel_names, config)
        assembler = WindowAssembler(
            session.channel_names,
            session.fs,
            window_samples=config.window_samples,
            hop_samples=config.hop_samples,
        )
        trial_windows = assembler.push(filter_bank.process(chunk))
        if len(trial_windows) != expected_per_trial:
            raise RuntimeError(
                f"trial {trial_index} emitted {len(trial_windows)} windows, "
                f"expected {expected_per_trial}"
            )
        windows.extend(trial_windows)
        labels.extend([int(session.labels[trial_index])] * expected_per_trial)
        trial_ids.extend([int(session.trial_ids[trial_index])] * expected_per_trial)

    batch = WindowBatch.from_windows(
        windows,
        trial_ids=np.asarray(trial_ids, dtype=np.int64),
    )
    contrast = _lateralized_mu_contrast(batch, np.asarray(labels, dtype=np.int64))
    if contrast < 4.0:
        raise RuntimeError(f"post-filter lateralized mu contrast is only {contrast:.3f} dB")
    metadata = dict(session.metadata)
    metadata["post_filter_lateralized_mu_contrast_db"] = contrast
    metadata["windows_per_trial"] = expected_per_trial
    return SyntheticWindowDataset(
        windows=batch,
        labels=np.asarray(labels, dtype=np.int64),
        metadata=MappingProxyType(metadata),
    )


def _lateralized_mu_contrast(
    windows: WindowBatch,
    labels: NDArray[np.int64],
) -> float:
    frequencies, density = welch_psd(windows.bands["mu"], windows.fs)
    power = integrate_band_power(frequencies, density, 8.0, 12.0)
    directions = np.where(
        labels == 0,
        10 * np.log10(np.maximum(power[:, 1], 1e-12) / np.maximum(power[:, 0], 1e-12)),
        10 * np.log10(np.maximum(power[:, 0], 1e-12) / np.maximum(power[:, 1], 1e-12)),
    )
    return float(np.mean(directions))
