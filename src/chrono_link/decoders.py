"""Typed window adapters and CPU-only baseline decoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Self, cast

import numpy as np
from numpy.typing import NDArray
from pyriemann.estimation import Covariances  # type: ignore[import-untyped]
from pyriemann.tangentspace import TangentSpace  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from sklearn.discriminant_analysis import (  # type: ignore[import-untyped]
    LinearDiscriminantAnalysis,
)
from sklearn.exceptions import NotFittedError  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from chrono_link.config import SignalConfig
from chrono_link.contracts import WindowBatch, WindowBundle
from chrono_link.features import FeatureExtractor

InputKind = Literal["epoch", "vector"]


class DecoderError(RuntimeError):
    """Decoder input or numeric output violates its declared contract."""


class DecoderInputAdapter(Protocol):
    """Convert a WindowBatch to exactly one estimator input kind."""

    @property
    def input_kind(self) -> InputKind: ...

    def transform(self, windows: WindowBatch) -> NDArray[np.float64]: ...


class Classifier(Protocol):
    """Small structural classifier interface used by BoundDecoder."""

    def fit(self, values: NDArray[np.float64], y: NDArray[np.int64]) -> object: ...

    def predict(self, values: NDArray[np.float64]) -> NDArray[np.int64]: ...

    def predict_proba(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...


class SPDRegularizer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Symmetrize covariance matrices and floor eigenvalues before tangent space."""

    def __init__(self, absolute_floor: float = 1e-12, relative_floor: float = 1.01e-6):
        self.absolute_floor = absolute_floor
        self.relative_floor = relative_floor

    def fit(
        self,
        values: NDArray[np.float64],
        y: object = None,
    ) -> Self:
        self._validate_parameters()
        self.n_channels_in_ = self._validate_input(values)
        return self

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        self._validate_parameters()
        self._validate_input(values)
        matrices = np.asarray(values, dtype=np.float64)
        output = np.empty_like(matrices)
        for index, matrix in enumerate(matrices):
            symmetric = (matrix + matrix.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
            maximum = max(float(eigenvalues[-1]), 0.0)
            floor = max(self.absolute_floor, self.relative_floor * maximum)
            regularized = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
            regularized = (regularized + regularized.T) / 2.0
            if not np.isfinite(regularized).all():
                raise DecoderError("regularized covariance is non-finite")
            try:
                np.linalg.cholesky(regularized)
            except np.linalg.LinAlgError as exc:
                raise DecoderError("regularized covariance is not positive definite") from exc
            if np.linalg.cond(regularized) > 1.01e6:
                raise DecoderError("regularized covariance condition exceeds 1.01e6")
            output[index] = regularized
        return output

    def _validate_parameters(self) -> None:
        if self.absolute_floor <= 0 or self.relative_floor <= 0:
            raise ValueError("SPD eigenvalue floors must be positive")

    @staticmethod
    def _validate_input(values: NDArray[np.float64]) -> int:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 3 or array.shape[1] != array.shape[2] or array.shape[1] == 0:
            raise DecoderError("covariances must have shape (examples, channels, channels)")
        if not np.isfinite(array).all():
            raise DecoderError("covariances must be finite")
        return int(array.shape[1])


@dataclass(frozen=True)
class RiemannEpochAdapter:
    """Select the MI-band epoch tensor without estimating covariance."""

    input_kind: InputKind = "epoch"
    band_name: str = "mi"

    def transform(self, windows: WindowBatch) -> NDArray[np.float64]:
        if self.band_name not in windows.bands:
            raise DecoderError(f"window batch is missing {self.band_name!r} epochs")
        values = np.asarray(windows.bands[self.band_name], dtype=np.float64)
        if values.ndim != 3 or not np.isfinite(values).all():
            raise DecoderError("epoch input must be finite and three-dimensional")
        return values


@dataclass
class TemporalVectorAdapter:
    """Build stable FeatureVector rows from each WindowBatch example."""

    config: SignalConfig = field(default_factory=SignalConfig)
    input_kind: InputKind = "vector"

    def __post_init__(self) -> None:
        self.extractor = FeatureExtractor(self.config)
        self.feature_names_: tuple[str, ...] | None = None
        self.feature_fingerprint_: str | None = None

    def transform(self, windows: WindowBatch) -> NDArray[np.float64]:
        rows: list[NDArray[np.float64]] = []
        for index in range(windows.cleaned.shape[0]):
            start, stop = windows.sequence_ranges[index]
            bundle = WindowBundle(
                cleaned=windows.cleaned[index],
                bands={name: values[index] for name, values in windows.bands.items()},
                timestamps=np.arange(
                    int(stop - start),
                    dtype=np.float64,
                )
                / windows.fs,
                start_sequence=int(start),
                end_sequence=int(stop),
                channel_names=windows.channel_names,
                fs=windows.fs,
                quality=windows.quality[index],
            )
            vector = self.extractor.vector(bundle)
            if self.feature_names_ is None:
                self.feature_names_ = vector.names
                self.feature_fingerprint_ = vector.fingerprint
            elif (
                vector.names != self.feature_names_
                or vector.fingerprint != self.feature_fingerprint_
            ):
                raise DecoderError("feature schema changed within a batch")
            rows.append(vector.values)
        if not rows:
            raise DecoderError("decoder requires at least one window")
        return np.stack(rows).astype(np.float64, copy=False)


@dataclass
class BoundDecoder:
    """Bind one explicit WindowBatch adapter to one fitted estimator pipeline."""

    adapter: DecoderInputAdapter
    estimator: Classifier
    decoder_kind: str
    _fitted: bool = field(default=False, init=False, repr=False)

    @property
    def input_kind(self) -> InputKind:
        return self.adapter.input_kind

    def fit(self, windows: WindowBatch, y: NDArray[np.int64]) -> Self:
        labels = np.asarray(y, dtype=np.int64)
        examples = int(windows.cleaned.shape[0])
        if labels.shape != (examples,):
            raise DecoderError(f"labels must have shape ({examples},)")
        if np.unique(labels).size < 2:
            raise DecoderError("decoder fit requires at least two classes")
        values = self.adapter.transform(windows)
        self.estimator.fit(values, labels)
        self._fitted = True
        return self

    def _require_fit(self) -> None:
        if not self._fitted:
            raise NotFittedError("BoundDecoder is not fitted")

    def predict(self, windows: WindowBatch) -> NDArray[np.int64]:
        self._require_fit()
        prediction = np.asarray(
            self.estimator.predict(self.adapter.transform(windows)),
            dtype=np.int64,
        )
        expected = int(windows.cleaned.shape[0])
        if prediction.shape != (expected,):
            raise DecoderError(f"prediction must have shape ({expected},)")
        return prediction

    def predict_proba(self, windows: WindowBatch) -> NDArray[np.float64]:
        self._require_fit()
        probabilities = np.asarray(
            self.estimator.predict_proba(self.adapter.transform(windows)),
            dtype=np.float64,
        )
        examples = int(windows.cleaned.shape[0])
        if probabilities.ndim != 2 or probabilities.shape[0] != examples:
            raise DecoderError("probabilities must have shape (examples, classes)")
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(np.sum(probabilities, axis=1), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise DecoderError("probabilities are not finite normalized values")
        return probabilities


def make_riemann_decoder() -> BoundDecoder:
    """Build Covariances(OAS) -> SPD floor -> Riemann tangent space -> LDA."""
    estimator = Pipeline(
        [
            ("covariances", Covariances("oas")),
            ("spd_regularizer", SPDRegularizer()),
            ("tangent_space", TangentSpace(metric="riemann", tsupdate=False)),
            (
                "lda",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ]
    )
    return BoundDecoder(
        adapter=RiemannEpochAdapter(),
        estimator=cast(Classifier, estimator),
        decoder_kind="riemann_ts_lda",
    )


def make_vector_decoder(config: SignalConfig | None = None) -> BoundDecoder:
    """Build stable temporal features -> standard scaling -> shrinkage LDA."""
    estimator = Pipeline(
        [
            ("standard_scaler", StandardScaler()),
            (
                "lda",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ]
    )
    return BoundDecoder(
        adapter=TemporalVectorAdapter(config or SignalConfig()),
        estimator=cast(Classifier, estimator),
        decoder_kind="temporal_vector_lda",
    )
