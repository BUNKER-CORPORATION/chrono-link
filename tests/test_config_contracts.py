from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chrono_link.config import (
    BoardConfig,
    ChronoConfig,
    ConfigError,
    SignalConfig,
)
from chrono_link.contracts import (
    BandSpec,
    FilteredChunk,
    SampleChunk,
    SourceEvent,
    WindowBatch,
    WindowBundle,
    WindowQuality,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": "unknown"},
        {"decode_channels": ()},
        {"decode_channels": ("C3", "C3")},
        {"record_channels": ()},
        {"record_channels": ("C3", "C3")},
        {"record_channels": ("C3",)},
    ],
)
def test_board_configuration_rejects_invalid_contracts(kwargs) -> None:
    with pytest.raises(ConfigError):
        BoardConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_samples": 0},
        {"mains_hz": 55},
        {"notch_q": 0},
        {"dc_highpass_hz": 0},
        {
            "filter_bands": (
                BandSpec("same", 1, 2),
                BandSpec("same", 2, 3),
            ),
            "vector_feature_bands": ("same",),
        },
        {"vector_feature_bands": ("missing",)},
    ],
)
def test_signal_configuration_rejects_invalid_contracts(kwargs) -> None:
    with pytest.raises(ConfigError):
        SignalConfig(**kwargs)


def test_signal_runtime_validation_rejects_sampling_and_nyquist() -> None:
    with pytest.raises(ConfigError, match="sampling"):
        SignalConfig().validate_for(0)
    with pytest.raises(ConfigError, match="dc_highpass"):
        SignalConfig(dc_highpass_hz=10).validate_for(20)
    with pytest.raises(ConfigError, match="broadband"):
        SignalConfig().validate_for(80)


def test_configuration_sources_precedence_conversion_and_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "board": {
                    "profile": "synthetic",
                    "decode_channels": ["C3", "C4"],
                    "record_channels": ["C3", "C4"],
                },
                "signal": {
                    "mains_hz": 50,
                    "filter_bands": [
                        {"name": "broadband", "low_hz": 1, "high_hz": 40},
                        {"name": "mi", "low_hz": 8, "high_hz": 30},
                        {"name": "mu", "low_hz": 8, "high_hz": 12},
                        {"name": "beta", "low_hz": 13, "high_hz": 30},
                    ],
                    "vector_feature_bands": ["mu", "beta"],
                },
                "output_root": "json-output",
            }
        ),
        encoding="utf-8",
    )
    config = ChronoConfig.from_sources(
        path,
        environ={
            "CHRONO_BOARD": "cyton",
            "CHRONO_SERIAL_PORT": "COM3",
            "CHRONO_MAINS_HZ": "60",
            "CHRONO_OUTPUT_ROOT": "env-output",
        },
        cli={"profile": "synthetic", "mains_hz": 50, "output_root": "cli-output"},
    )
    assert config.board.profile == "synthetic"
    assert config.board.serial_port == "COM3"
    assert config.board.record_channels == ("C3", "C4")
    assert config.signal.mains_hz == 50
    assert config.output_root == Path("cli-output")
    assert config.fingerprint() == config.fingerprint()
    changed = config.with_board(serial_port="COM4")
    assert changed.board.serial_port == "COM4"
    assert changed.fingerprint() != config.fingerprint()


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("not json", "cannot read"),
        ("[]", "root"),
        ('{"unknown":1}', "unknown configuration"),
        ('{"board":{"unknown":1}}', "unknown board"),
        ('{"signal":{"unknown":1}}', "unknown signal"),
    ],
)
def test_configuration_file_failures(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        ChronoConfig.from_sources(path, environ={})
    with pytest.raises(ConfigError, match="environment"):
        ChronoConfig.from_sources(environ={"CHRONO_MIAN_HZ": "50"})


def valid_chunk() -> SampleChunk:
    return SampleChunk(
        eeg=np.ones((2, 4), dtype=np.float64),
        timestamps=np.arange(4, dtype=np.float64),
        sequence=np.arange(4, dtype=np.int64),
        package_counter=np.arange(4, dtype=np.int64),
        channel_names=("C3", "C4"),
        fs=250,
        markers=np.zeros(4, dtype=np.float64),
    )


def valid_window() -> WindowBundle:
    cleaned = np.arange(8, dtype=np.float64).reshape(2, 4)
    return WindowBundle(
        cleaned=cleaned,
        bands={"mi": cleaned.copy()},
        timestamps=np.arange(4, dtype=np.float64),
        start_sequence=0,
        end_sequence=4,
        channel_names=("C3", "C4"),
        fs=250,
    )


def test_band_event_and_sample_chunk_validation() -> None:
    with pytest.raises(ValueError):
        BandSpec("", 1, 2)
    with pytest.raises(ValueError):
        BandSpec("bad", 2, 1)
    event = SourceEvent("RESET", 1, {"reason": "test"})
    with pytest.raises(TypeError):
        event.details["reason"] = "mutate"  # type: ignore[index]
    chunk = valid_chunk()
    with pytest.raises(ValueError, match="eeg"):
        replace(chunk, eeg=np.ones(4))
    with pytest.raises(ValueError, match="channel_names"):
        replace(chunk, channel_names=("C3",))
    with pytest.raises(ValueError, match="timestamps"):
        replace(chunk, timestamps=np.ones(3))
    with pytest.raises(ValueError, match="package_counter"):
        replace(chunk, package_counter=np.ones(3, dtype=np.int64))
    with pytest.raises(ValueError, match="markers"):
        replace(chunk, markers=np.ones(3))


def test_filtered_window_and_batch_validation_branches() -> None:
    cleaned = np.ones((2, 4), dtype=np.float64)
    filtered = FilteredChunk(
        cleaned=cleaned,
        bands={"mi": cleaned.copy()},
        timestamps=np.arange(4, dtype=np.float64),
        sequence=np.arange(4, dtype=np.int64),
        settled=np.ones(4, dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="cleaned"):
        replace(filtered, cleaned=np.ones(4))
    with pytest.raises(ValueError, match="metadata"):
        replace(filtered, timestamps=np.ones(3))
    with pytest.raises(ValueError, match="settled"):
        replace(filtered, settled=np.ones(3, dtype=np.bool_))
    with pytest.raises(ValueError, match="band"):
        replace(filtered, bands={"mi": np.ones((2, 3))})

    window = valid_window()
    with pytest.raises(ValueError, match="cleaned"):
        replace(window, cleaned=np.ones(4))
    with pytest.raises(ValueError, match="channel_names"):
        replace(window, channel_names=("C3",))
    with pytest.raises(ValueError, match="timestamps"):
        replace(window, timestamps=np.ones(3))
    with pytest.raises(ValueError, match="sequence"):
        replace(window, end_sequence=5)
    with pytest.raises(ValueError, match="sampling"):
        replace(window, fs=0)
    with pytest.raises(ValueError, match="band"):
        replace(window, bands={"mi": np.ones((2, 3))})

    batch = WindowBatch.from_windows([window])
    with pytest.raises(ValueError, match="cleaned"):
        replace(batch, cleaned=np.ones((2, 4)))
    with pytest.raises(ValueError, match="sequence_ranges"):
        replace(batch, sequence_ranges=np.ones((1, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="quality"):
        replace(batch, quality=())
    with pytest.raises(ValueError, match="trial_ids"):
        replace(batch, trial_ids=np.ones(2, dtype=np.int64))
    with pytest.raises(ValueError, match="at least"):
        WindowBatch.from_windows([])
    invalid = replace(window, quality=WindowQuality(False, ("bad",)))
    with pytest.raises(ValueError, match="incompatible"):
        WindowBatch.from_windows([invalid])
