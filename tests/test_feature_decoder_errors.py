from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from chrono_link.contracts import WindowBatch
from chrono_link.decoders import (
    BoundDecoder,
    DecoderError,
    RiemannEpochAdapter,
    SPDRegularizer,
    TemporalVectorAdapter,
)
from chrono_link.features import (
    FeatureError,
    FeatureExtractor,
    FeatureVector,
    TemporalFeatures,
    band_by_name,
    hilbert_features,
    integrate_band_power,
    normalized_spectral_entropy,
    phase_summary,
    welch_psd,
)
from chrono_link.synthetic import (
    SyntheticMISessionGenerator,
    preprocess_synthetic_session,
)


def batch() -> WindowBatch:
    return preprocess_synthetic_session(
        SyntheticMISessionGenerator(7, trial_count=10).generate()
    ).windows


def test_feature_value_and_temporal_contract_errors() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        FeatureVector(np.ones((1, 1)), ("x",), "fp")
    with pytest.raises(ValueError, match="equal"):
        FeatureVector(np.ones(1), (), "fp")
    with pytest.raises(ValueError, match="finite"):
        FeatureVector(np.asarray([np.nan]), ("x",), "fp")
    with pytest.raises(ValueError, match="shape"):
        TemporalFeatures(np.ones((1, 2)), np.ones((1, 2)), slice(0, 1), ("mu",))
    with pytest.raises(ValueError, match="band_names"):
        TemporalFeatures(np.ones((1, 2, 3)), np.ones((1, 2, 3)), slice(0, 1), ())
    with pytest.raises(ValueError, match="finite"):
        TemporalFeatures(
            np.full((1, 2, 3), np.nan),
            np.ones((1, 2, 3)),
            slice(0, 1),
            ("mu",),
        )


def test_feature_primitives_reject_invalid_shapes_and_bins() -> None:
    with pytest.raises(FeatureError, match="at least two"):
        welch_psd(np.ones(1), 250)
    with pytest.raises(FeatureError, match="finite"):
        welch_psd(np.asarray([1.0, np.nan]), 250)
    frequency = np.asarray([0.0, 1.0])
    density = np.ones(2)
    with pytest.raises(FeatureError, match="fewer"):
        integrate_band_power(frequency, density, 1.0, 1.0)
    with pytest.raises(FeatureError, match="short"):
        hilbert_features(np.ones(50))
    with pytest.raises(FeatureError, match="finite"):
        hilbert_features(np.full(100, np.nan))
    with pytest.raises(FeatureError, match="equal"):
        phase_summary(np.ones(2), np.ones(3))
    with pytest.raises(FeatureError, match="fewer"):
        normalized_spectral_entropy(frequency, density, low_hz=1, high_hz=1)


def test_feature_extractor_missing_branches_and_unknown_band() -> None:
    windows = batch()
    start, stop = windows.sequence_ranges[0]
    from chrono_link.contracts import WindowBundle

    window = WindowBundle(
        cleaned=windows.cleaned[0],
        bands={"mu": windows.bands["mu"][0]},
        timestamps=np.arange(250) / 250,
        start_sequence=int(start),
        end_sequence=int(stop),
        channel_names=windows.channel_names,
        fs=windows.fs,
    )
    extractor = FeatureExtractor()
    with pytest.raises(FeatureError, match="vector bands"):
        extractor.temporal(window)
    complete_except_broadband = replace(
        window,
        bands={name: values[0] for name, values in windows.bands.items() if name != "broadband"},
    )
    with pytest.raises(FeatureError, match="broadband"):
        extractor.vector(complete_except_broadband)
    with pytest.raises(FeatureError, match="unknown"):
        band_by_name(extractor.config, "theta")


def test_spd_and_adapter_error_contracts() -> None:
    regularizer = SPDRegularizer(absolute_floor=0)
    with pytest.raises(ValueError, match="positive"):
        regularizer.fit(np.ones((1, 2, 2)))
    for values in (np.ones((2, 2)), np.ones((1, 2, 3)), np.full((1, 2, 2), np.nan)):
        with pytest.raises(DecoderError):
            SPDRegularizer().fit(values)
    windows = batch()
    missing_mi = replace(
        windows,
        bands={name: value for name, value in windows.bands.items() if name != "mi"},
    )
    with pytest.raises(DecoderError, match="missing"):
        RiemannEpochAdapter().transform(missing_mi)
    empty = WindowBatch(
        cleaned=np.empty((0, 2, 250), dtype=np.float64),
        bands={
            name: np.empty((0, 2, 250), dtype=np.float64)
            for name in windows.bands
        },
        channel_names=windows.channel_names,
        fs=windows.fs,
        sequence_ranges=np.empty((0, 2), dtype=np.int64),
        quality=(),
    )
    with pytest.raises(DecoderError, match="at least"):
        TemporalVectorAdapter().transform(empty)


class BadClassifier:
    def fit(self, values, y):
        return self

    def predict(self, values):
        return np.zeros(2, dtype=np.int64)

    def predict_proba(self, values):
        return np.full((len(values), 2), np.nan)


def test_bound_decoder_rejects_labels_predictions_and_probabilities() -> None:
    windows = batch()
    decoder = BoundDecoder(TemporalVectorAdapter(), BadClassifier(), "bad")
    with pytest.raises(DecoderError, match="labels"):
        decoder.fit(windows, np.zeros(1, dtype=np.int64))
    with pytest.raises(DecoderError, match="two classes"):
        decoder.fit(windows, np.zeros(windows.cleaned.shape[0], dtype=np.int64))
    labels = np.tile([0, 1], windows.cleaned.shape[0] // 2).astype(np.int64)
    decoder.fit(windows, labels)
    with pytest.raises(DecoderError, match="prediction"):
        decoder.predict(windows)
    with pytest.raises(DecoderError, match="normalized"):
        decoder.predict_proba(windows)
