from __future__ import annotations

import numpy as np

from chrono_link.synthetic import (
    SyntheticMISessionGenerator,
    preprocess_synthetic_session,
)


def test_default_generator_is_balanced_and_array_deterministic() -> None:
    first = SyntheticMISessionGenerator(20260709).generate()
    second = SyntheticMISessionGenerator(20260709).generate()
    assert first.raw_trials.shape == (160, 2, 1500)
    assert np.bincount(first.labels).tolist() == [80, 80]
    np.testing.assert_array_equal(first.raw_trials, second.raw_trials)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert abs(float(first.metadata["realized_snr_db"]) + 6.0) <= 0.1

    different = SyntheticMISessionGenerator(7).generate()
    assert different.raw_trials.shape == first.raw_trials.shape
    assert not np.array_equal(first.raw_trials, different.raw_trials)


def test_causal_preprocessing_emits_four_locked_windows_per_trial() -> None:
    session = SyntheticMISessionGenerator(42, trial_count=10).generate()
    dataset = preprocess_synthetic_session(session)
    assert dataset.windows.cleaned.shape == (40, 2, 250)
    assert dataset.windows.trial_ids is not None
    assert np.bincount(dataset.windows.trial_ids).tolist() == [4] * 10
    np.testing.assert_array_equal(
        dataset.windows.sequence_ranges[:4, 0],
        [1000, 1064, 1128, 1192],
    )
    assert float(dataset.metadata["post_filter_lateralized_mu_contrast_db"]) >= 4.0
