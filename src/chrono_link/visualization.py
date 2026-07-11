"""Reusable live artists and deterministic headless artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from chrono_link.contracts import SampleChunk
from chrono_link.features import welch_psd


class VisualizationError(RuntimeError):
    """A visualization artifact cannot be produced safely."""


def write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    """Write RFC 8259-compatible JSON atomically without replacing a target."""
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class LiveVisualizer:
    """Update existing time-series and PSD artists for selected channels."""

    def __init__(
        self,
        channel_names: tuple[str, ...],
        fs: int,
        *,
        history_samples: int = 1000,
        headless: bool = False,
    ) -> None:
        if not channel_names or fs <= 0 or history_samples <= 1:
            raise ValueError("channels, positive fs, and history_samples > 1 are required")
        if headless:
            import matplotlib

            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self._plt = plt
        self.channel_names = channel_names
        self.fs = fs
        self.history_samples = history_samples
        self.figure, (self.time_axis, self.psd_axis) = plt.subplots(
            2,
            1,
            figsize=(8, 6),
            dpi=100,
            constrained_layout=True,
        )
        self.time_axis.set_title("Chrono Link selected-channel signal")
        self.time_axis.set_xlabel("Time (s)")
        self.time_axis.set_ylabel("BrainFlow native units")
        self.psd_axis.set_title("Welch power spectral density")
        self.psd_axis.set_xlabel("Frequency (Hz)")
        self.psd_axis.set_ylabel("Power / Hz")
        self.psd_axis.set_yscale("log")
        self.time_axis.grid(alpha=0.2)
        self.psd_axis.grid(alpha=0.2)
        self._time_lines = {
            name: self.time_axis.plot([], [], label=name, linewidth=1.0)[0]
            for name in channel_names
        }
        self._psd_lines = {
            name: self.psd_axis.plot([], [], label=name, linewidth=1.2)[0]
            for name in channel_names
        }
        self.time_axis.legend(loc="upper right")
        self.psd_axis.legend(loc="upper right")
        self._data = np.empty((len(channel_names), 0), dtype=np.float64)
        self._samples_seen = 0
        self._closed = False

    @property
    def artist_count(self) -> int:
        return len(self._time_lines) + len(self._psd_lines)

    def update(self, chunk: SampleChunk) -> None:
        if self._closed:
            raise VisualizationError("visualizer is closed")
        missing = [name for name in self.channel_names if name not in chunk.channel_names]
        if missing:
            raise VisualizationError(f"visualization channels are unavailable: {missing}")
        indices = [chunk.channel_names.index(name) for name in self.channel_names]
        selected = np.asarray(chunk.eeg[indices, :], dtype=np.float64)
        if not np.isfinite(selected).all():
            raise VisualizationError("visualization input must be finite")
        self._data = np.concatenate((self._data, selected), axis=1)[
            :, -self.history_samples :
        ]
        self._samples_seen += selected.shape[1]
        end_seconds = self._samples_seen / self.fs
        start_seconds = end_seconds - self._data.shape[1] / self.fs
        time_values = start_seconds + np.arange(self._data.shape[1]) / self.fs
        for index, name in enumerate(self.channel_names):
            self._time_lines[name].set_data(time_values, self._data[index])
        self.time_axis.relim()
        self.time_axis.autoscale_view()
        if self._data.shape[1] >= 2:
            frequencies, density = welch_psd(self._data, self.fs)
            positive = frequencies > 0
            for index, name in enumerate(self.channel_names):
                self._psd_lines[name].set_data(
                    frequencies[positive],
                    np.maximum(density[index, positive], 1e-16),
                )
            self.psd_axis.set_xlim(0, min(50, self.fs / 2))
            self.psd_axis.relim()
            self.psd_axis.autoscale_view(scalex=False, scaley=True)
        self.figure.canvas.draw_idle()

    def pause(self, seconds: float = 0.001) -> None:
        self._plt.pause(seconds)

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            self.figure.savefig(temporary, format="png", dpi=100)
            with temporary.open("r+b") as output:
                output.flush()
                os.fsync(output.fileno())
            os.link(temporary, path)
            temporary.unlink()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def close(self) -> None:
        if not self._closed:
            self._plt.close(self.figure)
            self._closed = True

    def __enter__(self) -> LiveVisualizer:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def render_headless(
    chunks: list[SampleChunk],
    output_dir: Path,
    metrics: dict[str, Any],
) -> tuple[Path, Path]:
    """Render supplied raw chunks to signal.png and strict metrics.json."""
    if not chunks:
        raise VisualizationError("headless rendering requires at least one chunk")
    first = chunks[0]
    png_path = output_dir / "signal.png"
    metrics_path = output_dir / "metrics.json"
    with LiveVisualizer(first.channel_names, first.fs, headless=True) as visualizer:
        for chunk in chunks:
            visualizer.update(chunk)
        visualizer.save(png_path)
    write_strict_json(metrics_path, metrics)
    return png_path, metrics_path
