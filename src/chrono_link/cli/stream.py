"""Stream or replay raw samples through plot-only or model-backed runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chrono_link.acquisition import BrainFlowSource, SampleSource, SourceError
from chrono_link.cli.common import (
    EXIT_ARTIFACT,
    EXIT_CONFIG,
    EXIT_RUNTIME,
    EXIT_SUCCESS,
    csv_tuple,
    print_error,
)
from chrono_link.config import ChronoConfig, ConfigError
from chrono_link.contracts import SampleChunk
from chrono_link.model_store import ModelStore, ModelStoreError, contract_for
from chrono_link.pipeline import PredictionEvent, RealtimePipeline
from chrono_link.recording import (
    RecordingError,
    ReplaySource,
    SessionRecord,
    SessionRecorder,
)
from chrono_link.visualization import LiveVisualizer, write_strict_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chrono-stream",
        description="Stream BrainFlow or replay a local Chrono Link recording.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--board", choices=("synthetic", "cyton"))
    parser.add_argument("--serial-port")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--channels", type=csv_tuple)
    parser.add_argument("--mains-hz", type=int, choices=(50, 60))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--trust-model", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--record", type=Path)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.replay is not None and (
        args.board is not None or args.serial_port is not None or args.record is not None
    ):
        raise ConfigError(
            "--replay is mutually exclusive with --board, --serial-port, and --record"
        )
    if args.seconds is not None and (args.seconds <= 0 or args.seconds > 3600):
        raise ConfigError("--seconds must be positive and no greater than 3600")
    if args.record is not None and args.seconds is None:
        raise ConfigError("--record requires a finite --seconds value")
    if args.trust_model and args.model is None:
        raise ConfigError("--trust-model requires --model")


def _prediction_dict(event: PredictionEvent) -> dict[str, Any]:
    return {
        "predicted_class": event.predicted_class,
        "probabilities": event.probabilities.tolist(),
        "sequence_range": event.sequence_range,
        "model_id": event.model_id,
        "quality_reasons": event.quality_reasons,
        "stage_timings_s": dict(event.stage_timings_s),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source: SampleSource | None = None
    visualizer: LiveVisualizer | None = None
    prepared = False
    try:
        _validate_args(args)
        if args.model is not None and not args.trust_model:
            raise ModelStoreError(
                "model loading requires --trust-model for an explicitly trusted local artifact"
            )
        config = ChronoConfig.from_sources(
            args.config,
            cli={
                "profile": args.board,
                "serial_port": args.serial_port,
                "decode_channels": args.channels,
                "mains_hz": args.mains_hz,
            },
        )
        if args.replay is None:
            source = BrainFlowSource(config.board, config.signal)
        else:
            record = SessionRecord.load(args.replay)
            source = ReplaySource(
                record,
                chunk_sizes=config.signal.hop_samples,
                decode_channels=config.board.decode_channels,
            )
        info = source.prepare()
        prepared = True

        decoder = None
        model_id = None
        if args.model is not None:
            manifest = ModelStore.inspect(args.model)
            expected = contract_for(
                str(manifest["decoder_kind"]),
                config.board.decode_channels,
                info.fs,
                config.signal,
            )
            loaded = ModelStore.load(
                args.model,
                expected,
                trusted_local=args.trust_model,
            )
            decoder = loaded.decoder
            model_id = str(loaded.manifest["model_id"])

        recorder = None
        if args.record is not None:
            if args.seconds is None:
                raise ConfigError("--record requires a finite --seconds value")
            row_names = dict(zip(info.eeg_rows, info.eeg_names, strict=True))
            record_names = tuple(row_names[row] for row in info.record_rows)
            recorder = SessionRecorder(
                args.record,
                info,
                record_names,
                config,
                max_samples=max(1, int(args.seconds * info.fs)),
            )

        if args.headless:
            for filename in ("signal.png", "metrics.json", "predictions.json"):
                target = args.out_dir / filename
                if target.exists():
                    raise FileExistsError(f"output already exists: {target}")
        active_visualizer = LiveVisualizer(
            config.board.decode_channels,
            info.fs,
            headless=args.headless,
        )
        visualizer = active_visualizer

        def update_plot(chunk: SampleChunk) -> None:
            active_visualizer.update(chunk)
            if not args.headless:
                active_visualizer.pause()

        predictions: list[dict[str, Any]] = []
        pipeline = RealtimePipeline(
            source,
            config,
            decoder=decoder,
            recorder=recorder,
            model_id=model_id,
            raw_callback=update_plot,
            prediction_callback=lambda event: predictions.append(_prediction_dict(event)),
        )
        metrics = pipeline.run(duration_s=args.seconds)
        prepared = False
        if args.headless:
            visualizer.save(args.out_dir / "signal.png")
            write_strict_json(
                args.out_dir / "metrics.json",
                {
                    "schema_version": 1,
                    "model_id": model_id,
                    "plot_only": decoder is None,
                    "metrics": metrics.summary(),
                    "scope": "synthetic/replay evidence is not real EEG or Cyton validation",
                },
            )
            write_strict_json(
                args.out_dir / "predictions.json",
                {"schema_version": 1, "predictions": predictions},
            )
    except (ConfigError, ValueError, FileExistsError) as exc:
        print_error("configuration", exc)
        return EXIT_CONFIG
    except (ModelStoreError, RecordingError) as exc:
        print_error("artifact", exc)
        return EXIT_ARTIFACT
    except SourceError as exc:
        print_error("source_runtime", exc)
        return EXIT_RUNTIME
    except Exception as exc:
        print_error("stream_runtime", exc)
        return EXIT_RUNTIME
    finally:
        if prepared and source is not None:
            try:
                source.stop()
            finally:
                source.release()
        if visualizer is not None:
            visualizer.close()
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
