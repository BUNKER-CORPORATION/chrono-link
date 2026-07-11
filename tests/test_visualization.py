from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from chrono_link.contracts import SampleChunk
from chrono_link.visualization import LiveVisualizer, render_headless


def chunk(start: int, samples: int) -> SampleChunk:
    sequence = np.arange(start, start + samples, dtype=np.int64)
    time = sequence / 250
    eeg = np.vstack(
        (
            np.sin(2 * np.pi * 10 * time) + 0.3 * np.sin(2 * np.pi * 22 * time),
            0.7 * np.sin(2 * np.pi * 12 * time + 0.4),
        )
    )
    return SampleChunk(
        eeg=eeg.astype(np.float64),
        timestamps=1_700_000_000 + time,
        sequence=sequence,
        package_counter=sequence % 256,
        channel_names=("C3", "C4"),
        fs=250,
        markers=np.zeros(samples, dtype=np.float64),
    )


def test_live_updates_reuse_artists_and_headless_png_is_informative(tmp_path: Path) -> None:
    visualizer = LiveVisualizer(("C3", "C4"), 250, headless=True)
    initial_axes = tuple(visualizer.figure.axes)
    initial_artists = visualizer.artist_count
    visualizer.update(chunk(0, 250))
    visualizer.update(chunk(250, 250))
    assert tuple(visualizer.figure.axes) == initial_axes
    assert visualizer.artist_count == initial_artists == 4
    path = tmp_path / "live.png"
    visualizer.save(path)
    visualizer.close()

    image = mpimg.imread(path)[..., :3]
    assert image.shape[1] >= 640 and image.shape[0] >= 480
    assert float(np.std(image)) >= 0.01
    unique = np.unique((image * 255).astype(np.uint8).reshape(-1, 3), axis=0)
    assert len(unique) >= 64


def test_render_headless_writes_strict_json(tmp_path: Path) -> None:
    png, metrics = render_headless(
        [chunk(0, 250), chunk(250, 250)],
        tmp_path,
        {"finite": 1.0, "windows": 0},
    )
    assert png.exists()
    assert json.loads(metrics.read_text(encoding="utf-8")) == {
        "finite": 1.0,
        "windows": 0,
    }
