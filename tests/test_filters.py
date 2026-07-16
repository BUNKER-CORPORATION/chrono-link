from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
from scipy import signal  # type: ignore[import-untyped]

from chrono_link.config import SignalConfig
from chrono_link.contracts import SampleChunk
from chrono_link.filters import (
    CausalFilterBank,
    FilterError,
    OfflineFilterBank,
    OfflineWindowTooShort,
    design_filters,
)


def sample_chunk(values: np.ndarray, start: int = 0) -> SampleChunk:
    samples = values.shape[1]
    sequence = np.arange(start, start + samples, dtype=np.int64)
    return SampleChunk(
        eeg=np.asarray(values, dtype=np.float64),
        timestamps=1_700_000_000.0 + sequence / 250.0,
        sequence=sequence,
        package_counter=sequence % 256,
        channel_names=("C3", "C4"),
        fs=250,
        markers=np.zeros(samples, dtype=np.float64),
    )


def chunks(values: np.ndarray) -> Iterator[SampleChunk]:
    sizes = (1, 7, 64, 3, 511, 2, 128, 19)
    start = 0
    size_index = 0
    while start < values.shape[1]:
        stop = min(start + sizes[size_index % len(sizes)], values.shape[1])
        yield sample_chunk(values[:, start:stop], start)
        start = stop
        size_index += 1


def magnitude_db(sos: np.ndarray, frequencies: list[float]) -> np.ndarray:
    _, response = signal.sosfreqz(sos, worN=np.asarray(frequencies), fs=250)
    return 20 * np.log10(np.maximum(np.abs(response), 1e-15))


def test_filter_design_response_and_fingerprint() -> None:
    design = design_filters(250)
    assert design.notch_frequencies == (50.0, 100.0)
    assert all(sos.shape == (4, 6) for sos in design.band_sos.values())
    assert design.fingerprint == design_filters(250).fingerprint

    highpass = design.cleaned_sos[:1]
    hp_db = magnitude_db(highpass, [0.1, 0.5, 2.0])
    assert hp_db[0] <= -27
    assert hp_db[1] == pytest.approx(-3.0103, abs=0.15)
    assert hp_db[2] >= -0.05

    for offset, center in enumerate(design.notch_frequencies, start=1):
        notch = design.cleaned_sos[offset : offset + 1]
        assert magnitude_db(notch, [center])[0] <= -40

    mu_db = magnitude_db(design.band_sos["mu"], [4, 8, 9, 10, 11, 12, 20])
    assert mu_db[0] <= -25 and mu_db[-1] <= -25
    assert mu_db[1] == pytest.approx(-3.0103, abs=0.15)
    assert mu_db[-2] == pytest.approx(-3.0103, abs=0.15)
    assert np.all(mu_db[2:5] >= -0.5)


def test_all_normative_band_and_notch_response_gates() -> None:
    design = design_filters(250)
    probes = {
        "broadband": {
            "inside": ([2, 10, 30], 0.5),
            "outside": [0.25, 70],
            "cutoffs": [1, 40],
        },
        "mi": {
            "inside": ([10, 20, 27], 1.5),
            "outside": [3, 60],
            "cutoffs": [8, 30],
        },
        "mu": {
            "inside": ([9, 10, 11], 0.5),
            "outside": [4, 20],
            "cutoffs": [8, 12],
        },
        "beta": {
            "inside": ([16, 20, 25], 0.5),
            "outside": [6, 50],
            "cutoffs": [13, 30],
        },
    }
    for name, gate in probes.items():
        inside, maximum_loss = gate["inside"]
        assert np.all(magnitude_db(design.band_sos[name], inside) >= -maximum_loss)
        assert np.all(magnitude_db(design.band_sos[name], gate["outside"]) <= -25)
        cutoff_db = magnitude_db(design.band_sos[name], gate["cutoffs"])
        np.testing.assert_allclose(cutoff_db, -3.0103, atol=0.15)
        np.testing.assert_allclose(2 * cutoff_db, -6.0206, atol=0.30)

    for section, center in enumerate(design.notch_frequencies, start=1):
        notch = design.cleaned_sos[section : section + 1]
        assert magnitude_db(notch, [center])[0] <= -40
        side_loss = -magnitude_db(notch, [center - 5, center + 5])
        limit = 0.25 if center == 50 else 1.5
        assert np.all(side_loss <= limit)


def test_notch_signal_suppression_preserves_ten_hz() -> None:
    time = np.arange(8 * 250) / 250
    values = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 50 * time)
    source = np.vstack((values, values))
    cleaned = CausalFilterBank(250, ("C3", "C4")).process(sample_chunk(source)).cleaned[0]
    frequencies, input_psd = signal.welch(values[250:-250], fs=250, nperseg=250)
    _, output_psd = signal.welch(cleaned[250:-250], fs=250, nperseg=250)
    bin_10 = int(np.argmin(np.abs(frequencies - 10)))
    bin_50 = int(np.argmin(np.abs(frequencies - 50)))
    suppression_50 = 10 * np.log10(input_psd[bin_50] / output_psd[bin_50])
    change_10 = 10 * np.log10(output_psd[bin_10] / input_psd[bin_10])
    assert suppression_50 >= 30
    assert abs(change_10) <= 0.5


def test_causal_output_is_invariant_to_chunking_and_tracks_warmup() -> None:
    rng = np.random.default_rng(20260709)
    values = rng.normal(size=(2, 5000))
    one_bank = CausalFilterBank(250, ("C3", "C4"))
    one = one_bank.process(sample_chunk(values))

    chunk_bank = CausalFilterBank(250, ("C3", "C4"))
    outputs = [chunk_bank.process(part) for part in chunks(values)]
    np.testing.assert_allclose(
        np.concatenate([output.cleaned for output in outputs], axis=1),
        one.cleaned,
        rtol=1e-10,
        atol=1e-11,
    )
    for name in one.bands:
        np.testing.assert_allclose(
            np.concatenate([output.bands[name] for output in outputs], axis=1),
            one.bands[name],
            rtol=1e-10,
            atol=1e-11,
        )
    settled = np.concatenate([output.settled for output in outputs])
    assert not settled[:1000].any()
    assert settled[1000:].all()

    chunk_bank.reset()
    after_reset = chunk_bank.process(sample_chunk(values[:, :1001]))
    assert not after_reset.settled[:1000].any()
    assert after_reset.settled[-1]


def test_offline_filter_is_zero_phase_and_rejects_short_input() -> None:
    fs = 250
    time = np.arange(fs * 8) / fs
    values = np.vstack((np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 10 * time)))
    filtered = OfflineFilterBank(fs).process(values)
    target = values[0, fs:-fs]
    actual = filtered.bands["mu"][0, fs:-fs]
    analytic_target = signal.hilbert(target)
    analytic_actual = signal.hilbert(actual)
    phase_error = np.angle(np.vdot(analytic_target, analytic_actual))
    assert abs(phase_error) <= 0.02
    assert filtered.preprocessing_mode == "offline_zero_phase"

    with pytest.raises(OfflineWindowTooShort):
        OfflineFilterBank(fs).process(np.ones((2, 4), dtype=np.float64))


def test_invalid_nyquist_configuration_fails_before_filtering() -> None:
    config = SignalConfig()
    with pytest.raises(ValueError, match="Nyquist"):
        design_filters(60, config)


def test_filter_bank_rejects_invalid_runtime_inputs() -> None:
    with pytest.raises(ValueError, match="channel_names"):
        CausalFilterBank(250, ())
    bank = CausalFilterBank(250, ("C3", "C4"))
    values = np.ones((2, 4), dtype=np.float64)
    with pytest.raises(FilterError, match="fs"):
        bank.process(replace(sample_chunk(values), fs=200))
    with pytest.raises(FilterError, match="unavailable"):
        CausalFilterBank(250, ("C3", "Cz")).process(sample_chunk(values))
    empty = sample_chunk(np.empty((2, 0), dtype=np.float64))
    assert bank.process(empty).cleaned.shape == (2, 0)
    invalid = values.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(FilterError, match="finite"):
        bank.process(sample_chunk(invalid))
    offline = OfflineFilterBank(250)
    with pytest.raises(FilterError, match="shape"):
        offline.process(np.ones(4))
    with pytest.raises(FilterError, match="finite"):
        offline.process(np.full((2, 100), np.nan))
