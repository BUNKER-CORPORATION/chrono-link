from __future__ import annotations

import numpy as np
import pytest
from scipy import signal  # type: ignore[import-untyped]

from chrono_link.contracts import SampleChunk, WindowBundle
from chrono_link.features import (
    FeatureError,
    FeatureExtractor,
    assess_window_quality,
    hilbert_features,
    integrate_band_power,
    log_power_db,
    normalized_spectral_entropy,
    phase_summary,
    welch_psd,
)
from chrono_link.filters import CausalFilterBank


def window_from_bands(
    cleaned: np.ndarray,
    bands: dict[str, np.ndarray],
) -> WindowBundle:
    samples = cleaned.shape[1]
    return WindowBundle(
        cleaned=np.asarray(cleaned, dtype=np.float64),
        bands={name: np.asarray(values, dtype=np.float64) for name, values in bands.items()},
        timestamps=np.arange(samples, dtype=np.float64) / 250,
        start_sequence=0,
        end_sequence=samples,
        channel_names=("C3", "C4"),
        fs=250,
    )


def test_welch_band_power_and_decibel_formula() -> None:
    time = np.arange(1000) / 250
    values = np.sqrt(2) * np.sin(2 * np.pi * 10 * time)
    values += 0.25 * np.sqrt(2) * np.sin(2 * np.pi * 20 * time)
    frequency, density = welch_psd(values, 250)
    mu = float(integrate_band_power(frequency, density, 8, 12))
    beta = float(integrate_band_power(frequency, density, 13, 30))
    assert mu == pytest.approx(1.0, rel=0.05)
    assert beta == pytest.approx(0.0625, rel=0.05)
    assert float(log_power_db(beta)) == pytest.approx(10 * np.log10(beta), abs=1e-10)


def test_hilbert_amplitude_frequency_and_phase() -> None:
    time = np.arange(1000) / 250
    values = 2 * np.sin(2 * np.pi * 10 * time)
    amplitude, phase, valid = hilbert_features(values)
    assert np.mean(amplitude[valid]) == pytest.approx(2.0, rel=0.02)
    unwrapped = np.unwrap(phase[valid])
    frequency = np.mean(np.diff(unwrapped)) * 250 / (2 * np.pi)
    assert frequency == pytest.approx(10.0, abs=0.02)
    expected = 2 * np.pi * 10 * time[valid] - np.pi / 2
    circular_error = np.angle(np.exp(1j * (phase[valid] - expected)))
    assert np.quantile(np.abs(circular_error), 0.95) <= 0.02


def test_phase_summary_fixed_and_independent() -> None:
    time = np.arange(5000) / 250
    left = 2 * np.pi * 10 * time
    right = left + 0.7
    cosine, sine, plv = phase_summary(left, right)
    assert cosine == pytest.approx(np.cos(-0.7), abs=1e-12)
    assert sine == pytest.approx(np.sin(-0.7), abs=1e-12)
    assert plv >= 0.995

    rng = np.random.default_rng(20260709)
    _, _, independent = phase_summary(
        rng.uniform(-np.pi, np.pi, 5000),
        rng.uniform(-np.pi, np.pi, 5000),
    )
    assert independent <= 0.05


def test_band_limited_noise_plv_bias_and_scale_invariance() -> None:
    rng = np.random.default_rng(20260709)
    sos = signal.butter(4, (8, 12), btype="bandpass", fs=250, output="sos")
    long_noise = signal.sosfiltfilt(sos, rng.normal(size=(2, 5000)), axis=1)
    _, long_phase, _ = hilbert_features(long_noise, guard=250)
    assert phase_summary(long_phase[0, 250:-250], long_phase[1, 250:-250])[2] <= 0.20

    window_plv = []
    for _ in range(256):
        filtered = signal.sosfiltfilt(sos, rng.normal(size=(2, 250)), axis=1)
        _, phase, valid = hilbert_features(filtered)
        window_plv.append(phase_summary(phase[0, valid], phase[1, valid])[2])
    assert 0.25 <= float(np.median(window_plv)) <= 0.55
    assert float(np.quantile(window_plv, 0.95)) <= 0.90

    scaled = long_noise.copy()
    scaled[0] *= 7.0
    _, scaled_phase, _ = hilbert_features(scaled, guard=250)
    original = phase_summary(long_phase[0, 250:-250], long_phase[1, 250:-250])[2]
    changed = phase_summary(scaled_phase[0, 250:-250], scaled_phase[1, 250:-250])[2]
    assert changed == pytest.approx(original, abs=1e-12)


def test_entropy_sine_noise_and_zero() -> None:
    time = np.arange(250) / 250
    sine = np.sin(2 * np.pi * 10 * time)
    frequency, sine_density = welch_psd(sine, 250)
    assert float(normalized_spectral_entropy(frequency, sine_density)) <= 0.25
    zero_frequency, zero_density = welch_psd(np.zeros(250), 250)
    assert float(normalized_spectral_entropy(zero_frequency, zero_density)) == 0.0
    for seed in (7, 42, 20260709):
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=(2, 1250))
        sequence = np.arange(1250, dtype=np.int64)
        source = SampleChunk(
            eeg=noise,
            timestamps=sequence / 250,
            sequence=sequence,
            package_counter=sequence % 256,
            channel_names=("C3", "C4"),
            fs=250,
            markers=np.zeros(1250),
        )
        broadband = CausalFilterBank(250, ("C3", "C4")).process(source).bands["broadband"][:, -250:]
        noise_frequency, noise_density = welch_psd(broadband, 250)
        entropy = normalized_spectral_entropy(noise_frequency, noise_density)
        assert np.all(entropy >= 0.85)


def test_default_vector_has_stable_twenty_names() -> None:
    time = np.arange(250) / 250
    c3 = np.sin(2 * np.pi * 10 * time) + 0.2 * np.sin(2 * np.pi * 20 * time)
    c4 = 0.7 * np.sin(2 * np.pi * 10 * time + 0.3) + 0.2 * np.sin(2 * np.pi * 20 * time + 0.5)
    cleaned = np.vstack((c3, c4))
    bands = {name: cleaned.copy() for name in ("broadband", "mi", "mu", "beta")}
    extractor = FeatureExtractor()
    vector = extractor.vector(window_from_bands(cleaned, bands))
    assert vector.values.shape == (20,)
    assert len(vector.names) == 20
    assert vector.names[:3] == (
        "mu.C3.log_power_db",
        "mu.C3.envelope_mean",
        "mu.C3.envelope_std",
    )
    assert vector.names[-2:] == (
        "broadband.C3.spectral_entropy",
        "broadband.C4.spectral_entropy",
    )
    assert vector.fingerprint == extractor.fingerprint(("C3", "C4"))


def test_flat_channel_is_rejected_at_exact_tolerance() -> None:
    cleaned = np.vstack((np.ones(250), np.linspace(-1, 1, 250)))
    bands = {name: cleaned.copy() for name in ("broadband", "mi", "mu", "beta")}
    window = window_from_bands(cleaned, bands)
    quality = assess_window_quality(window)
    assert not quality.valid
    assert quality.reasons == ("flat_channel:C3",)
    with pytest.raises(FeatureError, match="flat_channel:C3"):
        FeatureExtractor().vector(window)
