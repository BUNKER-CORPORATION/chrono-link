from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError  # type: ignore[import-untyped]

from chrono_link.decoders import (
    SPDRegularizer,
    make_riemann_decoder,
    make_vector_decoder,
)
from chrono_link.synthetic import (
    SyntheticMISessionGenerator,
    preprocess_synthetic_session,
)


def small_dataset():
    return preprocess_synthetic_session(
        SyntheticMISessionGenerator(20260709, trial_count=20).generate()
    )


def test_spd_regularizer_hardens_degenerate_matrices() -> None:
    values = np.asarray(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    output = SPDRegularizer().fit(values).transform(values)
    for matrix in output:
        symmetry = np.linalg.norm(matrix - matrix.T) / np.linalg.norm(matrix)
        assert symmetry <= 1e-12
        np.linalg.cholesky(matrix)
        assert np.linalg.cond(matrix) <= 1.01e6


@pytest.mark.parametrize("factory", [make_riemann_decoder, make_vector_decoder])
def test_bound_decoder_fit_probability_and_not_fitted(factory) -> None:
    dataset = small_dataset()
    decoder = factory()
    with pytest.raises(NotFittedError):
        decoder.predict(dataset.windows)
    decoder.fit(dataset.windows, dataset.labels)
    prediction = decoder.predict(dataset.windows)
    probabilities = decoder.predict_proba(dataset.windows)
    assert prediction.shape == dataset.labels.shape
    assert np.mean(prediction == dataset.labels) >= 0.9
    assert probabilities.shape == (len(dataset.labels), 2)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)
