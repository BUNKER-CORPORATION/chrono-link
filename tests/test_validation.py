from __future__ import annotations

from dataclasses import replace

import pytest

from chrono_link.validation import (
    FoldResult,
    SeedResult,
    ValidationError,
    ValidationReport,
    _protocol_fingerprint,
    validate_decoder,
    validate_training_report,
)


@pytest.mark.parametrize("kind", ["riemann", "vector"])
def test_grouped_validation_is_deterministic_and_passes_small_fixture(kind) -> None:
    first = validate_decoder(
        kind,
        seeds=(7,),
        permutation_count=0,
        trial_count=40,
    )
    second = validate_decoder(
        kind,
        seeds=(7,),
        permutation_count=0,
        trial_count=40,
    )
    assert first.passed
    assert first.to_json() == second.to_json()
    assert first.results[0].mean_balanced_accuracy >= 0.80
    assert first.results[0].minimum_fold_balanced_accuracy >= 0.70
    assert first.results[0].binomial_pvalue < 1e-6


def test_training_report_requires_exact_normative_protocol() -> None:
    folds = tuple(FoldResult(index, 0.9, 128, 32) for index in range(5))
    results = tuple(
        SeedResult(
            seed=seed,
            folds=folds,
            mean_balanced_accuracy=0.9,
            minimum_fold_balanced_accuracy=0.9,
            aggregate_balanced_accuracy=0.9,
            correct_trials=144,
            trial_count=160,
            binomial_pvalue=1e-20,
            permutation_p95=0.6,
            passed=True,
        )
        for seed in (7, 42, 20260709)
    )
    report = ValidationReport(
        schema_version=1,
        decoder_kind="riemann",
        seeds=(7, 42, 20260709),
        results=results,
        grand_mean_balanced_accuracy=0.9,
        dataset_fingerprints=("a" * 64, "b" * 64, "c" * 64),
        protocol_fingerprint=_protocol_fingerprint((7, 42, 20260709), 25),
        passed=True,
        failures=(),
    )
    validate_training_report(report)
    with pytest.raises(ValidationError, match="stale"):
        validate_training_report(replace(report, protocol_fingerprint="bad"))
