"""Immutable, hashed, explicitly trusted local model bundles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from chrono_link import __version__
from chrono_link.config import SignalConfig
from chrono_link.contracts import WindowBatch, WindowQuality
from chrono_link.decoders import BoundDecoder, TemporalVectorAdapter
from chrono_link.features import FeatureExtractor
from chrono_link.filters import design_filters
from chrono_link.validation import ValidationReport

MODEL_SCHEMA_VERSION = 1
SELF_TEST_SCHEMA_VERSION = 1
PAYLOAD_NAME = "bound_decoder.joblib"
SELF_TEST_NAME = "self_test.npz"
MANIFEST_NAME = "manifest.json"
DEPENDENCIES = (
    "brainflow",
    "numpy",
    "scipy",
    "scikit-learn",
    "pyriemann",
    "matplotlib",
    "joblib",
)
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "created_utc",
        "payload_sha256",
        "self_test_sha256",
        "decoder_kind",
        "input_kind",
        "channels",
        "fs",
        "window_samples",
        "window_duration_s",
        "hop_samples",
        "hop_duration_s",
        "preprocessing_mode",
        "filter_fingerprint",
        "filter_contract",
        "feature_fingerprint",
        "feature_names",
        "class_order",
        "chrono_link_version",
        "python_version",
        "dependencies",
        "source_commit",
        "wheel_sha256",
        "generator_recipe",
        "seed",
        "validation",
    }
)


class ModelStoreError(RuntimeError):
    """A model bundle is unsafe, corrupt, stale, or incompatible."""


@dataclass(frozen=True)
class ModelContract:
    """Runtime values that must exactly match a persisted model."""

    decoder_kind: str
    channels: tuple[str, ...]
    fs: int
    window_samples: int
    hop_samples: int
    preprocessing_mode: str
    filter_fingerprint: str
    feature_fingerprint: str


@dataclass(frozen=True)
class LoadedModel:
    """A verified decoder plus its already-validated manifest."""

    decoder: BoundDecoder
    manifest: dict[str, Any]
    path: Path


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in DEPENDENCIES:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError as exc:
            raise ModelStoreError(f"required distribution is missing: {distribution}") from exc
    return versions


def contract_for(
    decoder_kind: str,
    channels: tuple[str, ...],
    fs: int,
    signal_config: SignalConfig | None = None,
) -> ModelContract:
    config = signal_config or SignalConfig()
    return ModelContract(
        decoder_kind=decoder_kind,
        channels=channels,
        fs=fs,
        window_samples=config.window_samples,
        hop_samples=config.hop_samples,
        preprocessing_mode="causal",
        filter_fingerprint=design_filters(fs, config).fingerprint,
        feature_fingerprint=FeatureExtractor(config).fingerprint(channels),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"invalid JSON constant {token}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelStoreError(f"invalid model manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelStoreError("model manifest must contain an object")
    return cast(dict[str, Any], value)


def _write_bytes_fsync(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _estimator_classes(decoder: BoundDecoder) -> tuple[int, ...]:
    classes = getattr(decoder.estimator, "classes_", None)
    if classes is None:
        raise ModelStoreError("fitted decoder does not expose class order")
    array = np.asarray(classes, dtype=np.int64)
    if array.ndim != 1 or array.size < 2:
        raise ModelStoreError("decoder class order is invalid")
    return tuple(int(value) for value in array)


def _feature_contract(
    decoder: BoundDecoder,
    channels: tuple[str, ...],
    config: SignalConfig,
) -> tuple[tuple[str, ...], str]:
    if isinstance(decoder.adapter, TemporalVectorAdapter):
        names = decoder.adapter.feature_names_
        fingerprint = decoder.adapter.feature_fingerprint_
        if names is not None and fingerprint is not None:
            return names, fingerprint
    extractor = FeatureExtractor(config)
    probe_names = extractor.names(channels)
    return probe_names, extractor.fingerprint(channels)


def _write_self_test(
    path: Path,
    decoder: BoundDecoder,
    batch: WindowBatch,
    example_count: int = 4,
) -> None:
    count = min(example_count, int(batch.cleaned.shape[0]))
    if count <= 0:
        raise ModelStoreError("self-test requires at least one window")
    indices = np.arange(count, dtype=np.int64)
    selected = WindowBatch(
        cleaned=np.asarray(batch.cleaned[indices], dtype=np.float64),
        bands={
            name: np.asarray(values[indices], dtype=np.float64)
            for name, values in batch.bands.items()
        },
        channel_names=batch.channel_names,
        fs=batch.fs,
        sequence_ranges=np.asarray(batch.sequence_ranges[indices], dtype=np.int64),
        quality=tuple(batch.quality[int(index)] for index in indices),
        trial_ids=None,
    )
    prediction = decoder.predict(selected)
    probabilities = decoder.predict_proba(selected)
    fields: dict[str, Any] = {
        "schema_version": np.asarray(SELF_TEST_SCHEMA_VERSION, dtype=np.int64),
        "cleaned": selected.cleaned,
        "band_names": np.asarray(tuple(selected.bands), dtype=np.str_),
        "channel_names": np.asarray(selected.channel_names, dtype=np.str_),
        "fs": np.asarray(selected.fs, dtype=np.int64),
        "sequence_ranges": selected.sequence_ranges,
        "expected_prediction": prediction,
        "expected_probabilities": probabilities,
    }
    for name, values in selected.bands.items():
        if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ModelStoreError(f"unsafe band name in self-test: {name!r}")
        fields[f"band__{name}"] = values
    with path.open("xb") as output:
        np.savez_compressed(output, **fields)
        output.flush()
        os.fsync(output.fileno())


def _load_self_test(path: Path) -> tuple[WindowBatch, NDArray[np.int64], NDArray[np.float64]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            base = {
                "schema_version",
                "cleaned",
                "band_names",
                "channel_names",
                "fs",
                "sequence_ranges",
                "expected_prediction",
                "expected_probabilities",
            }
            if not base.issubset(names):
                raise ModelStoreError(f"self-test is missing fields: {sorted(base - names)}")
            arrays = {name: archive[name] for name in archive.files}
    except ModelStoreError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise ModelStoreError(f"invalid self-test payload: {exc}") from exc
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ModelStoreError("self-test object arrays are forbidden")
    if arrays["schema_version"].shape != () or int(arrays["schema_version"]) != 1:
        raise ModelStoreError("unsupported self-test schema")
    if arrays["band_names"].ndim != 1 or arrays["band_names"].dtype.kind != "U":
        raise ModelStoreError("self-test band_names are invalid")
    if arrays["channel_names"].ndim != 1 or arrays["channel_names"].dtype.kind != "U":
        raise ModelStoreError("self-test channel_names are invalid")
    if arrays["fs"].shape != () or arrays["fs"].dtype != np.int64:
        raise ModelStoreError("self-test fs must be an int64 scalar")
    band_names = tuple(str(value) for value in arrays["band_names"])
    expected_names = base | {f"band__{name}" for name in band_names}
    if names != expected_names:
        raise ModelStoreError("self-test fields disagree with band_names")
    cleaned = arrays["cleaned"]
    examples = int(cleaned.shape[0]) if cleaned.ndim == 3 else -1
    batch = WindowBatch(
        cleaned=cleaned,
        bands={name: arrays[f"band__{name}"] for name in band_names},
        channel_names=tuple(str(value) for value in arrays["channel_names"]),
        fs=int(arrays["fs"]),
        sequence_ranges=arrays["sequence_ranges"],
        quality=tuple(WindowQuality() for _ in range(examples)),
        trial_ids=None,
    )
    expected_prediction = arrays["expected_prediction"]
    expected_probabilities = arrays["expected_probabilities"]
    if expected_prediction.dtype != np.int64 or expected_prediction.shape != (examples,):
        raise ModelStoreError("self-test predictions have an invalid contract")
    if expected_probabilities.dtype != np.float64 or (
        expected_probabilities.ndim != 2 or expected_probabilities.shape[0] != examples
    ):
        raise ModelStoreError("self-test probabilities have an invalid contract")
    return batch, expected_prediction, expected_probabilities


class ModelStore:
    """Save and load complete BoundDecoder artifacts with pre-unpickle checks."""

    @staticmethod
    def save(
        root: Path,
        decoder: BoundDecoder,
        self_test_batch: WindowBatch,
        validation: ValidationReport,
        generator_recipe: dict[str, Any],
        *,
        seed: int,
        signal_config: SignalConfig | None = None,
        source_commit: str | None = None,
        wheel_sha256: str | None = None,
        created_utc: str | None = None,
    ) -> Path:
        if not validation.passed:
            raise ModelStoreError("a passing validation report is required")
        expected_validation_kind = (
            "riemann" if decoder.decoder_kind == "riemann_ts_lda" else "vector"
        )
        if validation.decoder_kind != expected_validation_kind:
            raise ModelStoreError("validation report decoder kind does not match decoder")
        config = signal_config or SignalConfig()
        root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".chrono-model-", dir=root))
        try:
            payload_path = temporary / PAYLOAD_NAME
            with payload_path.open("xb") as output:
                joblib.dump(decoder, output, compress=3)
                output.flush()
                os.fsync(output.fileno())
            self_test_path = temporary / SELF_TEST_NAME
            _write_self_test(self_test_path, decoder, self_test_batch)
            payload_hash = _sha256(payload_path)
            self_test_hash = _sha256(self_test_path)
            timestamp = created_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
            compact_time = re.sub(r"[^0-9]", "", timestamp)[:14]
            model_id = f"chrono-{expected_validation_kind}-{compact_time}-{payload_hash[:12]}"
            destination = root / model_id
            if destination.exists():
                raise FileExistsError(f"model already exists: {destination}")
            feature_names, feature_fingerprint = _feature_contract(
                decoder,
                self_test_batch.channel_names,
                config,
            )
            design = design_filters(self_test_batch.fs, config)
            class_order = _estimator_classes(decoder)
            manifest: dict[str, Any] = {
                "schema_version": MODEL_SCHEMA_VERSION,
                "model_id": model_id,
                "created_utc": timestamp,
                "payload_sha256": payload_hash,
                "self_test_sha256": self_test_hash,
                "decoder_kind": decoder.decoder_kind,
                "input_kind": decoder.input_kind,
                "channels": self_test_batch.channel_names,
                "fs": self_test_batch.fs,
                "window_samples": config.window_samples,
                "window_duration_s": config.window_samples / self_test_batch.fs,
                "hop_samples": config.hop_samples,
                "hop_duration_s": config.hop_samples / self_test_batch.fs,
                "preprocessing_mode": "causal",
                "filter_fingerprint": design.fingerprint,
                "filter_contract": {
                    "notch_frequencies": design.notch_frequencies,
                    "dc_highpass_hz": config.dc_highpass_hz,
                    "notch_q": config.notch_q,
                    "bands": [asdict(band) for band in config.filter_bands],
                },
                "feature_fingerprint": feature_fingerprint,
                "feature_names": feature_names,
                "class_order": class_order,
                "chrono_link_version": __version__,
                "python_version": platform.python_version(),
                "dependencies": dependency_versions(),
                "source_commit": source_commit,
                "wheel_sha256": wheel_sha256,
                "generator_recipe": generator_recipe,
                "seed": seed,
                "validation": json.loads(validation.to_json()),
            }
            manifest_bytes = json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            _write_bytes_fsync(temporary / MANIFEST_NAME, manifest_bytes)
            temporary.rename(destination)
            return destination
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    @staticmethod
    def inspect(path: Path) -> dict[str, Any]:
        manifest_path = path / MANIFEST_NAME
        manifest = _strict_json(manifest_path)
        fields = set(manifest)
        if fields != MANIFEST_FIELDS:
            raise ModelStoreError(
                f"manifest fields mismatch: missing={sorted(MANIFEST_FIELDS - fields)}, "
                f"extra={sorted(fields - MANIFEST_FIELDS)}"
            )
        if manifest["schema_version"] != MODEL_SCHEMA_VERSION:
            raise ModelStoreError(f"unsupported model schema {manifest['schema_version']!r}")
        if manifest["model_id"] != path.name:
            raise ModelStoreError("manifest model_id does not match its directory")
        for name in ("payload_sha256", "self_test_sha256"):
            value = manifest[name]
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ModelStoreError(f"manifest {name} is invalid")
        if manifest["input_kind"] not in {"epoch", "vector"}:
            raise ModelStoreError("manifest input_kind is invalid")
        if not isinstance(manifest["dependencies"], dict):
            raise ModelStoreError("manifest dependencies must be an object")
        return manifest

    @staticmethod
    def load(
        path: Path,
        expected: ModelContract,
        *,
        trusted_local: bool = False,
    ) -> LoadedModel:
        manifest = ModelStore.inspect(path)
        payload_path = path / PAYLOAD_NAME
        self_test_path = path / SELF_TEST_NAME
        try:
            if _sha256(payload_path) != manifest["payload_sha256"]:
                raise ModelStoreError("bound decoder SHA-256 mismatch")
            if _sha256(self_test_path) != manifest["self_test_sha256"]:
                raise ModelStoreError("self-test SHA-256 mismatch")
        except OSError as exc:
            raise ModelStoreError(f"model payload is missing or unreadable: {exc}") from exc

        mismatches: list[str] = []
        contract_values = asdict(expected)
        for field_name, expected_value in contract_values.items():
            actual = manifest[field_name]
            if isinstance(expected_value, tuple):
                actual = tuple(actual) if isinstance(actual, list) else actual
            if actual != expected_value:
                mismatches.append(f"{field_name}: expected {expected_value!r}, found {actual!r}")
        if manifest["chrono_link_version"] != __version__:
            mismatches.append(
                f"chrono_link_version: expected {__version__!r}, "
                f"found {manifest['chrono_link_version']!r}"
            )
        current_python = platform.python_version_tuple()[:2]
        recorded_python = tuple(str(manifest["python_version"]).split(".")[:2])
        if recorded_python != current_python:
            mismatches.append(
                f"python major.minor: expected {current_python!r}, found {recorded_python!r}"
            )
        current_dependencies = dependency_versions()
        if manifest["dependencies"] != current_dependencies:
            mismatches.append("dependency versions differ from the current runtime")
        if mismatches:
            raise ModelStoreError("model compatibility mismatch: " + "; ".join(mismatches))
        self_test_batch, expected_prediction, expected_probabilities = _load_self_test(
            self_test_path
        )
        if not trusted_local:
            raise ModelStoreError(
                "joblib models can execute code; pass trusted_local=True only for a "
                "trusted local artifact"
            )
        try:
            loaded = joblib.load(payload_path)
        except Exception as exc:
            raise ModelStoreError(f"cannot load bound decoder: {exc}") from exc
        if not isinstance(loaded, BoundDecoder):
            raise ModelStoreError("payload is not a BoundDecoder")
        if (
            loaded.decoder_kind != manifest["decoder_kind"]
            or loaded.input_kind != manifest["input_kind"]
            or _estimator_classes(loaded) != tuple(manifest["class_order"])
        ):
            raise ModelStoreError("loaded decoder contract disagrees with the manifest")
        prediction = loaded.predict(self_test_batch)
        probabilities = loaded.predict_proba(self_test_batch)
        if not np.array_equal(prediction, expected_prediction) or not np.allclose(
            probabilities,
            expected_probabilities,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ModelStoreError("loaded model failed its self-test")
        return LoadedModel(decoder=loaded, manifest=manifest, path=path)
