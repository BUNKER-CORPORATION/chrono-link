"""Synchronous real-time orchestration with explicit cleanup and metrics."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from chrono_link.acquisition import SampleSource
from chrono_link.config import ChronoConfig
from chrono_link.contracts import ResolvedBoardInfo, SampleChunk, WindowBatch, WindowBundle
from chrono_link.decoders import BoundDecoder
from chrono_link.features import assess_window_quality
from chrono_link.filters import CausalFilterBank
from chrono_link.recording import SessionRecorder
from chrono_link.windows import WindowAssembler


class PipelineState(Enum):
    NEW = auto()
    PREPARED = auto()
    STREAMING = auto()
    PLOT_ONLY_READY = auto()
    STREAMING_WARMUP = auto()
    INFERENCE_READY = auto()
    STOPPED = auto()
    RELEASED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class PredictionEvent:
    """Non-raw callback payload for one valid decoded window."""

    predicted_class: int
    probabilities: NDArray[np.float64]
    sequence_range: tuple[int, int]
    model_id: str | None
    quality_reasons: tuple[str, ...]
    stage_timings_s: MappingProxyType[str, float]


@dataclass(frozen=True)
class PipelineStep:
    """Observable result of one source drain and processing pass."""

    had_chunk: bool
    windows: tuple[WindowBundle, ...] = ()
    predictions: tuple[PredictionEvent, ...] = ()
    timings_s: MappingProxyType[str, float] = field(default_factory=lambda: MappingProxyType({}))


@dataclass
class PipelineMetrics:
    reads: int = 0
    empty_reads: int = 0
    samples: int = 0
    gaps: int = 0
    resets: int = 0
    windows: int = 0
    invalid_windows: int = 0
    predictions: int = 0
    timings_s: dict[str, list[float]] = field(default_factory=dict)

    def observe(self, stage: str, seconds: float) -> None:
        self.timings_s.setdefault(stage, []).append(seconds)

    def summary(self) -> dict[str, object]:
        timing_summary: dict[str, dict[str, float]] = {}
        for stage, values in self.timings_s.items():
            array = np.asarray(values, dtype=np.float64)
            timing_summary[stage] = {
                "count": float(len(values)),
                "p50_ms": float(np.percentile(array, 50) * 1000),
                "p95_ms": float(np.percentile(array, 95) * 1000),
                "max_ms": float(np.max(array) * 1000),
            }
        return {
            "reads": self.reads,
            "empty_reads": self.empty_reads,
            "samples": self.samples,
            "gaps": self.gaps,
            "resets": self.resets,
            "windows": self.windows,
            "invalid_windows": self.invalid_windows,
            "predictions": self.predictions,
            "timings": timing_summary,
        }


RawCallback = Callable[[SampleChunk], None]
WindowCallback = Callable[[WindowBundle], None]
PredictionCallback = Callable[[PredictionEvent], None]


class RealtimePipeline:
    """Orchestrate source -> record -> causal DSP -> windows -> quality -> decode."""

    def __init__(
        self,
        source: SampleSource,
        config: ChronoConfig,
        *,
        decoder: BoundDecoder | None = None,
        recorder: SessionRecorder | None = None,
        model_id: str | None = None,
        raw_callback: RawCallback | None = None,
        window_callback: WindowCallback | None = None,
        prediction_callback: PredictionCallback | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.source = source
        self.config = config
        self.decoder = decoder
        self.recorder = recorder
        self.model_id = model_id
        self.raw_callback = raw_callback
        self.window_callback = window_callback
        self.prediction_callback = prediction_callback
        self.clock = clock
        self.state = PipelineState.NEW
        self.info: ResolvedBoardInfo | None = None
        self.filter_bank: CausalFilterBank | None = None
        self.window_assembler: WindowAssembler | None = None
        self.metrics = PipelineMetrics()

    def prepare(self) -> ResolvedBoardInfo:
        if self.state is PipelineState.PREPARED and self.info is not None:
            return self.info
        if self.state is not PipelineState.NEW:
            raise RuntimeError(f"cannot prepare pipeline in state {self.state.name}")
        try:
            info = self.source.prepare()
            self.config.signal.validate_for(info.fs)
            self.filter_bank = CausalFilterBank(
                info.fs,
                self.config.board.decode_channels,
                self.config.signal,
            )
            self.window_assembler = WindowAssembler(
                self.config.board.decode_channels,
                info.fs,
                window_samples=self.config.signal.window_samples,
                hop_samples=self.config.signal.hop_samples,
            )
            self.info = info
            self.state = PipelineState.PREPARED
            return info
        except Exception:
            self.state = PipelineState.FAILED
            self.source.release()
            self.state = PipelineState.RELEASED
            raise

    def start(self) -> None:
        if self.state is not PipelineState.PREPARED:
            raise RuntimeError(f"cannot start pipeline in state {self.state.name}")
        try:
            self.source.start()
            self.state = (
                PipelineState.PLOT_ONLY_READY
                if self.decoder is None
                else PipelineState.STREAMING_WARMUP
            )
        except Exception:
            self.state = PipelineState.FAILED
            self.close(success=False)
            raise

    def _require_processing(self) -> tuple[CausalFilterBank, WindowAssembler]:
        if self.filter_bank is None or self.window_assembler is None:
            raise RuntimeError("pipeline is not prepared")
        if self.state not in {
            PipelineState.PLOT_ONLY_READY,
            PipelineState.STREAMING_WARMUP,
            PipelineState.INFERENCE_READY,
        }:
            raise RuntimeError(f"cannot step pipeline in state {self.state.name}")
        return self.filter_bank, self.window_assembler

    def step(self) -> PipelineStep:
        filter_bank, assembler = self._require_processing()
        total_start = self.clock()
        acquisition_start = total_start
        chunk = self.source.read_new()
        acquisition_s = self.clock() - acquisition_start
        self.metrics.reads += 1
        self.metrics.observe("acquisition", acquisition_s)
        if chunk is None:
            self.metrics.empty_reads += 1
            return PipelineStep(
                had_chunk=False,
                timings_s=MappingProxyType({"acquisition": acquisition_s}),
            )

        self.metrics.samples += int(chunk.eeg.shape[1])
        if self.recorder is not None:
            self.recorder.append(chunk)
        if self.raw_callback is not None:
            self.raw_callback(chunk)
        if chunk.events_before:
            self.metrics.gaps += sum(
                event.code in {"PACKAGE_GAP", "REPLAY_GAP"} for event in chunk.events_before
            )
            self.metrics.resets += 1
            filter_bank.reset()
            assembler.reset()

        filter_start = self.clock()
        filtered = filter_bank.process(chunk)
        filter_s = self.clock() - filter_start
        window_start = self.clock()
        emitted = assembler.push(filtered)
        window_s = self.clock() - window_start
        self.metrics.observe("filter", filter_s)
        self.metrics.observe("window", window_s)
        self.metrics.windows += len(emitted)

        accepted: list[WindowBundle] = []
        predictions: list[PredictionEvent] = []
        feature_predict_s = 0.0
        for window in emitted:
            quality = assess_window_quality(
                window,
                flat_abs_tol=self.config.signal.flat_abs_tol,
                flat_rel_tol=self.config.signal.flat_rel_tol,
            )
            checked = replace(window, quality=quality)
            if not quality.valid:
                self.metrics.invalid_windows += 1
                continue
            accepted.append(checked)
            if self.window_callback is not None:
                self.window_callback(checked)
            if self.decoder is None:
                continue
            decode_start = self.clock()
            batch = WindowBatch.from_windows([checked])
            predicted = self.decoder.predict(batch)
            probabilities = self.decoder.predict_proba(batch)
            elapsed = self.clock() - decode_start
            feature_predict_s += elapsed
            event = PredictionEvent(
                predicted_class=int(predicted[0]),
                probabilities=np.array(probabilities[0], copy=True),
                sequence_range=(checked.start_sequence, checked.end_sequence),
                model_id=self.model_id,
                quality_reasons=checked.quality.reasons,
                stage_timings_s=MappingProxyType({"feature_prediction": elapsed}),
            )
            predictions.append(event)
            self.metrics.predictions += 1
            if self.prediction_callback is not None:
                self.prediction_callback(event)
        if predictions:
            self.state = PipelineState.INFERENCE_READY
        total_s = self.clock() - total_start
        self.metrics.observe("feature_prediction", feature_predict_s)
        self.metrics.observe("total", total_s)
        return PipelineStep(
            had_chunk=True,
            windows=tuple(accepted),
            predictions=tuple(predictions),
            timings_s=MappingProxyType(
                {
                    "acquisition": acquisition_s,
                    "filter": filter_s,
                    "window": window_s,
                    "feature_prediction": feature_predict_s,
                    "total": total_s,
                }
            ),
        )

    def run(
        self,
        *,
        duration_s: float | None = None,
        max_predictions: int | None = None,
        poll_interval_s: float = 0.005,
    ) -> PipelineMetrics:
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if max_predictions is not None and max_predictions <= 0:
            raise ValueError("max_predictions must be positive")
        self.prepare()
        self.start()
        started = self.clock()
        success = False
        try:
            while True:
                if duration_s is not None and self.clock() - started >= duration_s:
                    break
                if max_predictions is not None and self.metrics.predictions >= max_predictions:
                    break
                result = self.step()
                if not result.had_chunk:
                    if bool(getattr(self.source, "exhausted", False)):
                        break
                    if poll_interval_s > 0:
                        time.sleep(poll_interval_s)
            success = True
            return self.metrics
        except KeyboardInterrupt:
            success = True
            return self.metrics
        except Exception:
            self.state = PipelineState.FAILED
            raise
        finally:
            self.close(success=success)

    def close(self, *, success: bool) -> None:
        try:
            self.source.stop()
            if self.state is not PipelineState.RELEASED:
                self.state = PipelineState.STOPPED
            if self.recorder is not None:
                if success:
                    self.recorder.finalize()
                else:
                    self.recorder.abort()
        finally:
            self.source.release()
            self.state = PipelineState.RELEASED
