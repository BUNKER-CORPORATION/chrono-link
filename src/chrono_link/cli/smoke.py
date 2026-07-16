"""Run the locked deterministic two-decoder replay/performance smoke workload."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from chrono_link.cli.common import (
    EXIT_CONFIG,
    EXIT_RUNTIME,
    EXIT_SUCCESS,
    print_error,
)
from chrono_link.config import ChronoConfig
from chrono_link.contracts import SampleChunk
from chrono_link.decoders import BoundDecoder, make_riemann_decoder, make_vector_decoder
from chrono_link.pipeline import RealtimePipeline
from chrono_link.recording import ReplaySource, SessionRecord
from chrono_link.synthetic import (
    DEFAULT_SEED,
    SyntheticMISessionGenerator,
    preprocess_synthetic_session,
)
from chrono_link.visualization import render_headless, write_strict_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chrono-smoke",
        description="Run deterministic replay smoke and latency gates for both decoders.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument("--windows", type=int, default=200)
    return parser


def _record(values: np.ndarray) -> SessionRecord:
    samples = values.shape[1]
    sequence = np.arange(samples, dtype=np.int64)
    return SessionRecord(
        data=np.asarray(values, dtype=np.float64),
        timestamps=1_700_000_000.0 + sequence / 250.0,
        sequence=sequence,
        package_counter=sequence % 256,
        channel_names=("C3", "C4"),
        units=("brainflow_native", "brainflow_native"),
        fs=250,
        board_id=-1,
        created_utc=datetime.now(UTC).isoformat(),
        markers=np.zeros(samples, dtype=np.float64),
        config_json=ChronoConfig.synthetic().canonical_json(),
        events=(),
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(array, 50) * 1000),
        "p95_ms": float(np.percentile(array, 95) * 1000),
        "max_ms": float(np.max(array) * 1000),
    }


def _benchmark(
    decoder_name: str,
    decoder: BoundDecoder,
    record: SessionRecord,
    measured_windows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = ReplaySource(
        record,
        chunk_sizes=(1250, *((64,) * (20 + measured_windows))),
    )
    pipeline = RealtimePipeline(
        source,
        ChronoConfig.synthetic(),
        decoder=decoder,
        model_id=f"smoke-{decoder_name}-{DEFAULT_SEED}",
    )
    pipeline.prepare()
    pipeline.start()
    try:
        prime = pipeline.step()
        if len(prime.predictions) != 1:
            raise RuntimeError("smoke prime step must emit exactly one prediction")
        for _ in range(20):
            warm = pipeline.step()
            if len(warm.predictions) != 1:
                raise RuntimeError("smoke benchmark warm-up must emit exactly one prediction")
        timings: dict[str, list[float]] = {}
        predictions: list[dict[str, Any]] = []
        sequence_ranges: list[tuple[int, int]] = []
        for _ in range(measured_windows):
            result = pipeline.step()
            if len(result.predictions) != 1:
                raise RuntimeError("every measured smoke step must emit exactly one prediction")
            event = result.predictions[0]
            sequence_ranges.append(event.sequence_range)
            predictions.append(
                {
                    "decoder": decoder_name,
                    "predicted_class": event.predicted_class,
                    "probabilities": event.probabilities.tolist(),
                    "sequence_range": event.sequence_range,
                }
            )
            for stage, seconds in result.timings_s.items():
                timings.setdefault(stage, []).append(seconds)
        expected_starts = [2344 + index * 64 for index in range(measured_windows)]
        if [start for start, _ in sequence_ranges] != expected_starts:
            raise RuntimeError("smoke output sequences are duplicated or missing")
        if any(
            not np.isfinite(np.asarray(item["probabilities"], dtype=np.float64)).all()
            for item in predictions
        ):
            raise RuntimeError("smoke predictions contain non-finite probabilities")
        stage_summary = {stage: _summary(values) for stage, values in timings.items()}
        totals = timings["total"]
        deadline_misses = sum(value >= 0.256 for value in totals)
        passed = (
            stage_summary["total"]["p50_ms"] <= 20.0
            and stage_summary["total"]["p95_ms"] <= 50.0
            and deadline_misses == 0
        )
        return (
            {
                "decoder": decoder_name,
                "measured_windows": measured_windows,
                "stage_timings": stage_summary,
                "deadline_misses_256ms": deadline_misses,
                "passed": passed,
            },
            predictions,
        )
    finally:
        pipeline.close(success=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.windows < 200:
        print_error("configuration", ValueError("--windows must be at least 200"))
        return EXIT_CONFIG
    try:
        session = SyntheticMISessionGenerator(DEFAULT_SEED).generate()
        training = preprocess_synthetic_session(session)
        required_samples = 1250 + (20 + args.windows) * 64
        trial_copies = math.ceil(required_samples / session.raw_trials.shape[2])
        values = np.concatenate(session.raw_trials[:trial_copies], axis=1)[:, :required_samples]
        record = _record(values)
        results = []
        predictions: list[dict[str, Any]] = []
        for name, decoder in (
            (
                "riemann",
                make_riemann_decoder().fit(training.windows, training.labels),
            ),
            (
                "vector",
                make_vector_decoder().fit(training.windows, training.labels),
            ),
        ):
            result, decoder_predictions = _benchmark(
                name,
                decoder,
                record,
                args.windows,
            )
            results.append(result)
            predictions.extend(decoder_predictions)
        passed = all(bool(result["passed"]) for result in results)
        metrics = {
            "schema_version": 1,
            "seed": DEFAULT_SEED,
            "workload": {
                "prime_samples": 1250,
                "unmeasured_windows": 20,
                "measured_windows": args.windows,
                "hop_samples": 64,
            },
            "decoders": results,
            "passed": passed,
            "scope": "synthetic replay only; not real EEG or hardware validation",
        }
        sequence = np.arange(record.sample_count, dtype=np.int64)
        render_chunk = SampleChunk(
            eeg=record.data,
            timestamps=record.timestamps,
            sequence=sequence,
            package_counter=record.package_counter,
            channel_names=record.channel_names,
            fs=record.fs,
            markers=record.markers,
        )
        render_headless([render_chunk], args.out_dir, metrics)
        write_strict_json(
            args.out_dir / "predictions.json",
            {"schema_version": 1, "predictions": predictions},
        )
    except FileExistsError as exc:
        print_error("output", exc)
        return EXIT_CONFIG
    except Exception as exc:
        print_error("smoke_runtime", exc)
        return EXIT_RUNTIME
    print(f"smoke passed={str(passed).lower()} output={args.out_dir}")
    return EXIT_SUCCESS if passed else EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
