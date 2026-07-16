"""Leakage-safe grouped validation for the two baseline decoders."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest  # type: ignore[import-untyped]
from sklearn.metrics import balanced_accuracy_score  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]

from chrono_link import __version__
from chrono_link.config import SignalConfig
from chrono_link.contracts import WindowBatch
from chrono_link.decoders import BoundDecoder, make_riemann_decoder, make_vector_decoder
from chrono_link.features import FeatureExtractor
from chrono_link.filters import design_filters
from chrono_link.synthetic import (
    SyntheticMISession,
    SyntheticMISessionGenerator,
    SyntheticWindowDataset,
    preprocess_synthetic_session,
)

DecoderKind = Literal["riemann", "vector"]
DecoderFactory = Callable[[], BoundDecoder]
VALIDATION_SEEDS = (7, 42, 20260709)


class ValidationError(RuntimeError):
    """Validation cannot be completed under the locked protocol."""


@dataclass(frozen=True)
class FoldResult:
    fold: int
    balanced_accuracy: float
    train_trials: int
    test_trials: int


@dataclass(frozen=True)
class SeedResult:
    seed: int
    folds: tuple[FoldResult, ...]
    mean_balanced_accuracy: float
    minimum_fold_balanced_accuracy: float
    aggregate_balanced_accuracy: float
    correct_trials: int
    trial_count: int
    binomial_pvalue: float
    permutation_p95: float | None
    passed: bool


@dataclass(frozen=True)
class ValidationReport:
    schema_version: int
    decoder_kind: DecoderKind
    seeds: tuple[int, ...]
    results: tuple[SeedResult, ...]
    grand_mean_balanced_accuracy: float
    dataset_fingerprints: tuple[str, ...]
    protocol_fingerprint: str
    passed: bool
    failures: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: object) -> ValidationReport:
        """Parse the exact schema emitted by :meth:`to_json`."""
        if not isinstance(value, dict):
            raise ValidationError("validation report must contain an object")
        required = {
            "schema_version",
            "decoder_kind",
            "seeds",
            "results",
            "grand_mean_balanced_accuracy",
            "dataset_fingerprints",
            "protocol_fingerprint",
            "passed",
            "failures",
        }
        if set(value) != required or value.get("schema_version") != 1:
            raise ValidationError("validation report schema is invalid")
        if value.get("decoder_kind") not in {"riemann", "vector"}:
            raise ValidationError("validation decoder kind is invalid")
        raw_results = value.get("results")
        if not isinstance(raw_results, list):
            raise ValidationError("validation results must be a list")
        parsed_results: list[SeedResult] = []
        try:
            for result in raw_results:
                if not isinstance(result, dict):
                    raise ValidationError("seed result must be an object")
                raw_folds = result["folds"]
                if not isinstance(raw_folds, list):
                    raise ValidationError("fold results must be a list")
                folds = tuple(FoldResult(**fold) for fold in raw_folds)
                parsed_results.append(
                    SeedResult(
                        seed=int(result["seed"]),
                        folds=folds,
                        mean_balanced_accuracy=float(result["mean_balanced_accuracy"]),
                        minimum_fold_balanced_accuracy=float(
                            result["minimum_fold_balanced_accuracy"]
                        ),
                        aggregate_balanced_accuracy=float(
                            result["aggregate_balanced_accuracy"]
                        ),
                        correct_trials=int(result["correct_trials"]),
                        trial_count=int(result["trial_count"]),
                        binomial_pvalue=float(result["binomial_pvalue"]),
                        permutation_p95=(
                            None
                            if result["permutation_p95"] is None
                            else float(result["permutation_p95"])
                        ),
                        passed=bool(result["passed"]),
                    )
                )
            report = cls(
                schema_version=1,
                decoder_kind=value["decoder_kind"],
                seeds=tuple(int(seed) for seed in value["seeds"]),
                results=tuple(parsed_results),
                grand_mean_balanced_accuracy=float(
                    value["grand_mean_balanced_accuracy"]
                ),
                dataset_fingerprints=tuple(
                    str(item) for item in value["dataset_fingerprints"]
                ),
                protocol_fingerprint=str(value["protocol_fingerprint"]),
                passed=bool(value["passed"]),
                failures=tuple(str(item) for item in value["failures"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"validation report values are invalid: {exc}") from exc
        if (
            len(report.results) != len(report.seeds)
            or len(report.dataset_fingerprints) != len(report.seeds)
            or tuple(result.seed for result in report.results) != report.seeds
            or not np.isfinite(report.grand_mean_balanced_accuracy)
        ):
            raise ValidationError("validation report seed/result contracts disagree")
        return report


def decoder_factory(kind: DecoderKind) -> DecoderFactory:
    if kind == "riemann":
        return make_riemann_decoder
    if kind == "vector":
        return make_vector_decoder
    raise ValueError(f"unknown decoder kind {kind!r}")


def subset_batch(batch: WindowBatch, indices: NDArray[np.int64]) -> WindowBatch:
    """Take examples without changing channel, branch, or quality contracts."""
    return WindowBatch(
        cleaned=np.asarray(batch.cleaned[indices], dtype=np.float64),
        bands={
            name: np.asarray(values[indices], dtype=np.float64)
            for name, values in batch.bands.items()
        },
        channel_names=batch.channel_names,
        fs=batch.fs,
        sequence_ranges=np.asarray(batch.sequence_ranges[indices], dtype=np.int64),
        quality=tuple(batch.quality[int(index)] for index in indices),
        trial_ids=(
            None
            if batch.trial_ids is None
            else np.asarray(batch.trial_ids[indices], dtype=np.int64)
        ),
    )


def dataset_fingerprint(session: SyntheticMISession) -> str:
    digest = hashlib.sha256()
    digest.update(session.raw_trials.astype("<f8", copy=False).tobytes())
    digest.update(session.labels.astype("<i8", copy=False).tobytes())
    digest.update(session.trial_ids.astype("<i8", copy=False).tobytes())
    digest.update(
        json.dumps(
            dict(session.metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _protocol_fingerprint(seeds: Sequence[int], permutation_count: int) -> str:
    signal_config = SignalConfig()
    payload = {
        "schema": 1,
        "chrono_link_version": __version__,
        "seeds": list(seeds),
        "splitter": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "aggregation": "mean_trial_probability",
        "permutation_count": permutation_count,
        "signal_contract": {
            "fs": 250,
            "channels": ["C3", "C4"],
            "window_samples": signal_config.window_samples,
            "hop_samples": signal_config.hop_samples,
            "warmup_samples": signal_config.warmup_samples,
            "filter_fingerprint": design_filters(250, signal_config).fingerprint,
            "feature_fingerprint": FeatureExtractor(signal_config).fingerprint(
                ("C3", "C4")
            ),
        },
        "thresholds": {
            "seed_mean": 0.80,
            "minimum_fold": 0.70,
            "grand_mean": 0.85,
            "binomial_pvalue": 1e-6,
            "permutation_p95": 0.65,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_training_report(report: ValidationReport) -> None:
    """Require a complete, internally consistent normative report before training."""
    failures: list[str] = []
    if report.schema_version != 1:
        failures.append("unsupported schema")
    if report.seeds != VALIDATION_SEEDS:
        failures.append(f"seeds must be {VALIDATION_SEEDS}")
    if report.protocol_fingerprint != _protocol_fingerprint(VALIDATION_SEEDS, 25):
        failures.append("protocol/config fingerprint is stale")
    if len(report.results) != len(VALIDATION_SEEDS):
        failures.append("seed result count is invalid")
    if len(report.dataset_fingerprints) != len(VALIDATION_SEEDS) or any(
        re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
        for fingerprint in report.dataset_fingerprints
    ):
        failures.append("dataset fingerprints are invalid")
    for result in report.results:
        scores = [fold.balanced_accuracy for fold in result.folds]
        result_valid = (
            len(scores) == 5
            and all(np.isfinite(scores))
            and min(scores, default=0.0) >= 0.70
            and result.mean_balanced_accuracy >= 0.80
            and result.binomial_pvalue < 1e-6
            and result.permutation_p95 is not None
            and result.permutation_p95 <= 0.65
            and result.passed
        )
        if not result_valid:
            failures.append(f"seed {result.seed} metrics do not satisfy the protocol")
    if report.grand_mean_balanced_accuracy < 0.85 or not report.passed or report.failures:
        failures.append("aggregate report gate is not passing")
    if failures:
        raise ValidationError("invalid training report: " + "; ".join(failures))


def _trial_predictions(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    unique_groups = np.unique(groups)
    truth = np.empty(len(unique_groups), dtype=np.int64)
    prediction = np.empty(len(unique_groups), dtype=np.int64)
    for output_index, group in enumerate(unique_groups):
        selected = groups == group
        group_labels = np.unique(labels[selected])
        if group_labels.size != 1:
            raise ValidationError(f"trial {group} contains mixed labels")
        truth[output_index] = int(group_labels[0])
        prediction[output_index] = int(np.argmax(np.mean(probabilities[selected], axis=0)))
    return unique_groups, truth, prediction


def grouped_cross_validation(
    dataset: SyntheticWindowDataset,
    factory: DecoderFactory,
    *,
    split_seed: int,
    labels: NDArray[np.int64] | None = None,
) -> tuple[tuple[FoldResult, ...], NDArray[np.int64], NDArray[np.int64]]:
    """Fit all transforms inside folds and return trial-level out-of-fold predictions."""
    batch = dataset.windows
    if batch.trial_ids is None:
        raise ValidationError("grouped validation requires trial_ids")
    selected_labels = dataset.labels if labels is None else np.asarray(labels, dtype=np.int64)
    if selected_labels.shape != dataset.labels.shape:
        raise ValidationError("validation labels have the wrong shape")
    groups = batch.trial_ids
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    fold_results: list[FoldResult] = []
    all_truth: list[NDArray[np.int64]] = []
    all_prediction: list[NDArray[np.int64]] = []
    observed_test_groups: set[int] = set()
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(batch.cleaned, selected_labels, groups)
    ):
        train_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.asarray(test_indices, dtype=np.int64)
        train_groups = set(int(value) for value in groups[train_indices])
        test_groups = set(int(value) for value in groups[test_indices])
        if train_groups & test_groups:
            raise ValidationError("trial leakage detected between train and test")
        if observed_test_groups & test_groups:
            raise ValidationError("a trial appeared in more than one test fold")
        observed_test_groups.update(test_groups)

        decoder = factory().fit(
            subset_batch(batch, train_indices),
            selected_labels[train_indices],
        )
        probabilities = decoder.predict_proba(subset_batch(batch, test_indices))
        _, truth, prediction = _trial_predictions(
            probabilities,
            selected_labels[test_indices],
            groups[test_indices],
        )
        score = float(balanced_accuracy_score(truth, prediction))
        fold_results.append(
            FoldResult(
                fold=fold,
                balanced_accuracy=score,
                train_trials=len(train_groups),
                test_trials=len(test_groups),
            )
        )
        all_truth.append(truth)
        all_prediction.append(prediction)
    if observed_test_groups != set(int(value) for value in np.unique(groups)):
        raise ValidationError("grouped folds did not cover every trial exactly once")
    return (
        tuple(fold_results),
        np.concatenate(all_truth),
        np.concatenate(all_prediction),
    )


def _permuted_window_labels(
    dataset: SyntheticWindowDataset,
    seed: int,
) -> NDArray[np.int64]:
    assert dataset.windows.trial_ids is not None
    groups = dataset.windows.trial_ids
    unique_groups = np.unique(groups)
    trial_labels = np.asarray(
        [dataset.labels[np.flatnonzero(groups == group)[0]] for group in unique_groups],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(trial_labels)
    label_by_group = dict(zip(unique_groups, shuffled, strict=True))
    return np.asarray([label_by_group[group] for group in groups], dtype=np.int64)


def validate_decoder(
    kind: DecoderKind,
    *,
    seeds: Sequence[int] = VALIDATION_SEEDS,
    permutation_count: int = 25,
    trial_count: int = 160,
) -> ValidationReport:
    """Run the locked multi-seed grouped protocol and evaluate all pass gates."""
    if not seeds or permutation_count < 0:
        raise ValueError("seeds cannot be empty and permutation_count cannot be negative")
    factory = decoder_factory(kind)
    results: list[SeedResult] = []
    fingerprints: list[str] = []
    failures: list[str] = []
    for seed in seeds:
        session = SyntheticMISessionGenerator(seed, trial_count=trial_count).generate()
        dataset = preprocess_synthetic_session(session)
        fingerprints.append(dataset_fingerprint(session))
        folds, truth, prediction = grouped_cross_validation(
            dataset,
            factory,
            split_seed=seed,
        )
        scores = np.asarray([fold.balanced_accuracy for fold in folds], dtype=np.float64)
        correct = int(np.count_nonzero(truth == prediction))
        pvalue = float(binomtest(correct, len(truth), p=0.5, alternative="greater").pvalue)
        permutation_scores: list[float] = []
        for offset in range(permutation_count):
            permuted = _permuted_window_labels(dataset, seed + 10_000 + offset)
            _, permuted_truth, permuted_prediction = grouped_cross_validation(
                dataset,
                factory,
                split_seed=seed,
                labels=permuted,
            )
            permutation_scores.append(
                float(balanced_accuracy_score(permuted_truth, permuted_prediction))
            )
        permutation_p95 = (
            None
            if not permutation_scores
            else float(np.quantile(permutation_scores, 0.95))
        )
        seed_passed = (
            float(np.mean(scores)) >= 0.80
            and float(np.min(scores)) >= 0.70
            and pvalue < 1e-6
            and (permutation_p95 is None or permutation_p95 <= 0.65)
        )
        if not seed_passed:
            failures.append(f"seed {seed} failed one or more validation gates")
        results.append(
            SeedResult(
                seed=seed,
                folds=folds,
                mean_balanced_accuracy=float(np.mean(scores)),
                minimum_fold_balanced_accuracy=float(np.min(scores)),
                aggregate_balanced_accuracy=float(
                    balanced_accuracy_score(truth, prediction)
                ),
                correct_trials=correct,
                trial_count=len(truth),
                binomial_pvalue=pvalue,
                permutation_p95=permutation_p95,
                passed=seed_passed,
            )
        )
    grand_mean = float(
        np.mean([result.mean_balanced_accuracy for result in results])
    )
    if grand_mean < 0.85:
        failures.append(f"grand mean {grand_mean:.6f} is below 0.85")
    return ValidationReport(
        schema_version=1,
        decoder_kind=kind,
        seeds=tuple(int(seed) for seed in seeds),
        results=tuple(results),
        grand_mean_balanced_accuracy=grand_mean,
        dataset_fingerprints=tuple(fingerprints),
        protocol_fingerprint=_protocol_fingerprint(seeds, permutation_count),
        passed=not failures,
        failures=tuple(failures),
    )
