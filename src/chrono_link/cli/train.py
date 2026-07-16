"""Validate, fit, and atomically persist one complete BoundDecoder."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from chrono_link.cli.common import (
    EXIT_ARTIFACT,
    EXIT_CONFIG,
    EXIT_RUNTIME,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    load_validation_report,
    print_error,
)
from chrono_link.model_store import ModelStore, ModelStoreError
from chrono_link.synthetic import (
    DEFAULT_SEED,
    SyntheticMISessionGenerator,
    preprocess_synthetic_session,
)
from chrono_link.validation import (
    ValidationError,
    dataset_fingerprint,
    decoder_factory,
    validate_decoder,
    validate_training_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chrono-train",
        description="Train an immutable Chrono Link model after grouped validation.",
    )
    parser.add_argument("--decoder", choices=("riemann", "vector"), required=True)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--permutations",
        type=int,
        default=25,
        help="Used only when no validation report is supplied.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.permutations < 0:
        print_error("configuration", ValueError("--permutations cannot be negative"))
        return EXIT_CONFIG
    try:
        if args.validation_report is None:
            report = validate_decoder(
                args.decoder,
                permutation_count=args.permutations,
            )
        else:
            report = load_validation_report(args.validation_report, args.decoder)
        if not report.passed:
            print_error("validation_gate", ValidationError(str(report.failures)))
            return EXIT_VALIDATION
        validate_training_report(report)
        if args.seed not in report.seeds:
            raise ModelStoreError(
                f"training seed {args.seed} is absent from the validation report"
            )
        session = SyntheticMISessionGenerator(args.seed).generate()
        fingerprint = dataset_fingerprint(session)
        report_index = report.seeds.index(args.seed)
        if report.dataset_fingerprints[report_index] != fingerprint:
            raise ModelStoreError("validation report dataset fingerprint is stale")
        dataset = preprocess_synthetic_session(session)
        decoder = decoder_factory(args.decoder)().fit(dataset.windows, dataset.labels)
        destination = ModelStore.save(
            args.out_dir,
            decoder,
            dataset.windows,
            report,
            dict(session.metadata),
            seed=args.seed,
        )
    except (ValidationError, ModelStoreError) as exc:
        print_error("artifact_validation", exc)
        return EXIT_ARTIFACT
    except FileExistsError as exc:
        print_error("immutable_model", exc)
        return EXIT_ARTIFACT
    except Exception as exc:
        print_error("training_runtime", exc)
        return EXIT_RUNTIME
    print(destination)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
