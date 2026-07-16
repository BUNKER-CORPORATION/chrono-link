from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import numpy as np
import pytest

import chrono_link.model_store as model_module
from chrono_link.contracts import WindowBatch
from chrono_link.decoders import (
    BoundDecoder,
    RiemannEpochAdapter,
    make_riemann_decoder,
    make_vector_decoder,
)
from chrono_link.model_store import ModelStore, ModelStoreError, contract_for
from chrono_link.synthetic import (
    SyntheticMISessionGenerator,
    preprocess_synthetic_session,
)
from chrono_link.validation import validate_decoder


@pytest.mark.parametrize(
    ("kind", "factory"),
    [("riemann", make_riemann_decoder), ("vector", make_vector_decoder)],
)
def test_model_save_load_parity_and_trust_gate(tmp_path: Path, kind, factory) -> None:
    session = SyntheticMISessionGenerator(7, trial_count=40).generate()
    dataset = preprocess_synthetic_session(session)
    report = validate_decoder(kind, seeds=(7,), permutation_count=0, trial_count=40)
    decoder = factory().fit(dataset.windows, dataset.labels)
    path = ModelStore.save(
        tmp_path,
        decoder,
        dataset.windows,
        report,
        dict(session.metadata),
        seed=7,
        created_utc="2026-07-10T12:00:00Z",
    )
    expected = contract_for(
        decoder.decoder_kind,
        dataset.windows.channel_names,
        dataset.windows.fs,
    )
    with pytest.raises(ModelStoreError, match="trusted_local"):
        ModelStore.load(path, expected)
    loaded = ModelStore.load(path, expected, trusted_local=True)
    assert loaded.decoder.decoder_kind == decoder.decoder_kind
    assert loaded.manifest["model_id"] == path.name
    assert loaded.decoder.predict(dataset.windows).shape == (len(dataset.labels),)


def test_payload_tamper_is_rejected_before_unpickle(tmp_path: Path, monkeypatch) -> None:
    session = SyntheticMISessionGenerator(7, trial_count=40).generate()
    dataset = preprocess_synthetic_session(session)
    report = validate_decoder("riemann", seeds=(7,), permutation_count=0, trial_count=40)
    decoder = make_riemann_decoder().fit(dataset.windows, dataset.labels)
    path = ModelStore.save(
        tmp_path,
        decoder,
        dataset.windows,
        report,
        dict(session.metadata),
        seed=7,
    )
    payload = path / "bound_decoder.joblib"
    with payload.open("ab") as output:
        output.write(b"tamper")
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("joblib.load must not run")

    monkeypatch.setattr("chrono_link.model_store.joblib.load", forbidden_load)
    expected = contract_for(
        decoder.decoder_kind,
        dataset.windows.channel_names,
        dataset.windows.fs,
    )
    with pytest.raises(ModelStoreError, match="SHA-256"):
        ModelStore.load(path, expected, trusted_local=True)
    assert not called


def build_vector_bundle(tmp_path: Path):
    session = SyntheticMISessionGenerator(7, trial_count=40).generate()
    dataset = preprocess_synthetic_session(session)
    report = validate_decoder("vector", seeds=(7,), permutation_count=0, trial_count=40)
    decoder = make_vector_decoder().fit(dataset.windows, dataset.labels)
    path = ModelStore.save(
        tmp_path,
        decoder,
        dataset.windows,
        report,
        dict(session.metadata),
        seed=7,
        created_utc="2026-07-10T12:00:00Z",
    )
    expected = contract_for(
        decoder.decoder_kind,
        dataset.windows.channel_names,
        dataset.windows.fs,
    )
    return path, decoder, dataset, report, expected


def write_manifest(path: Path, manifest: object) -> None:
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_manifest_schema_and_compatibility_fail_closed(tmp_path: Path) -> None:
    path, _, _, _, expected = build_vector_bundle(tmp_path)
    original = ModelStore.inspect(path)
    mutations = [
        ({key: value for key, value in original.items() if key != "seed"}, "fields"),
        ({**original, "schema_version": 2}, "schema"),
        ({**original, "model_id": "wrong"}, "model_id"),
        ({**original, "payload_sha256": "bad"}, "payload_sha256"),
        ({**original, "input_kind": "bad"}, "input_kind"),
        ({**original, "dependencies": []}, "dependencies"),
    ]
    for mutated, match in mutations:
        write_manifest(path, mutated)
        with pytest.raises(ModelStoreError, match=match):
            ModelStore.inspect(path)
    (path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ModelStoreError, match="object"):
        ModelStore.inspect(path)
    (path / "manifest.json").write_text('{"bad":NaN}', encoding="utf-8")
    with pytest.raises(ModelStoreError, match="invalid"):
        ModelStore.inspect(path)
    write_manifest(path, original)

    with pytest.raises(ModelStoreError, match="compatibility"):
        ModelStore.load(path, replace(expected, fs=251), trusted_local=True)
    for field, value in (
        ("chrono_link_version", "99.0.0"),
        ("python_version", "2.7.0"),
        ("dependencies", {"wrong": "1"}),
    ):
        mutated = dict(original)
        mutated[field] = value
        write_manifest(path, mutated)
        with pytest.raises(ModelStoreError, match="compatibility"):
            ModelStore.load(path, expected, trusted_local=True)
    write_manifest(path, original)


def load_npz_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def write_npz(path: Path, fields: dict[str, object]) -> None:
    with path.open("wb") as output:
        np.savez_compressed(output, **fields)


def test_self_test_schema_negatives_and_loaded_parity_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path, _, _, _, expected = build_vector_bundle(tmp_path)
    self_test = path / "self_test.npz"
    fields = load_npz_fields(self_test)
    cases = [
        ({key: value for key, value in fields.items() if key != "cleaned"}, "missing"),
        ({**fields, "schema_version": np.asarray(2, dtype=np.int64)}, "schema"),
        ({**fields, "band_names": np.asarray([1], dtype=np.int64)}, "band_names"),
        ({**fields, "channel_names": np.asarray([1], dtype=np.int64)}, "channel_names"),
        ({**fields, "extra": np.asarray(1)}, "disagree"),
        ({**fields, "expected_prediction": np.ones(4, dtype=np.float64)}, "predictions"),
        (
            {**fields, "expected_probabilities": np.ones((4, 2), dtype=np.float32)},
            "probabilities",
        ),
    ]
    for index, (mutated, match) in enumerate(cases):
        candidate = tmp_path / f"self-{index}.npz"
        write_npz(candidate, mutated)
        with pytest.raises(ModelStoreError, match=match):
            model_module._load_self_test(candidate)
    corrupt = tmp_path / "self-corrupt.npz"
    corrupt.write_bytes(b"bad")
    with pytest.raises(ModelStoreError, match="invalid"):
        model_module._load_self_test(corrupt)

    changed = dict(fields)
    changed["expected_prediction"] = 1 - fields["expected_prediction"]
    write_npz(self_test, changed)
    manifest = ModelStore.inspect(path)
    manifest["self_test_sha256"] = hashlib.sha256(self_test.read_bytes()).hexdigest()
    write_manifest(path, manifest)
    with pytest.raises(ModelStoreError, match="failed its self-test"):
        ModelStore.load(path, expected, trusted_local=True)

    write_npz(self_test, fields)
    manifest["self_test_sha256"] = hashlib.sha256(self_test.read_bytes()).hexdigest()
    write_manifest(path, manifest)
    monkeypatch.setattr(model_module.joblib, "load", lambda target: object())
    with pytest.raises(ModelStoreError, match="not a BoundDecoder"):
        ModelStore.load(path, expected, trusted_local=True)
    monkeypatch.setattr(model_module.joblib, "load", lambda target: 1 / 0)
    with pytest.raises(ModelStoreError, match="cannot load"):
        ModelStore.load(path, expected, trusted_local=True)


def test_model_save_guards_and_internal_contract_errors(tmp_path: Path, monkeypatch) -> None:
    path, decoder, dataset, report, _ = build_vector_bundle(tmp_path)
    with pytest.raises(ModelStoreError, match="passing"):
        ModelStore.save(
            tmp_path,
            decoder,
            dataset.windows,
            replace(report, passed=False),
            {},
            seed=7,
        )
    with pytest.raises(ModelStoreError, match="kind"):
        ModelStore.save(
            tmp_path,
            decoder,
            dataset.windows,
            replace(report, decoder_kind="riemann"),
            {},
            seed=7,
        )
    with pytest.raises(FileExistsError):
        ModelStore.save(
            tmp_path,
            decoder,
            dataset.windows,
            report,
            {},
            seed=7,
            created_utc="2026-07-10T12:00:00Z",
        )
    assert path.exists()

    empty = WindowBatch(
        cleaned=np.empty((0, 2, 250), dtype=np.float64),
        bands={name: values[:0] for name, values in dataset.windows.bands.items()},
        channel_names=dataset.windows.channel_names,
        fs=dataset.windows.fs,
        sequence_ranges=np.empty((0, 2), dtype=np.int64),
        quality=(),
    )
    with pytest.raises(ModelStoreError, match="at least"):
        model_module._write_self_test(tmp_path / "empty.npz", decoder, empty)
    unsafe = replace(
        dataset.windows,
        bands={**dataset.windows.bands, "bad/name": dataset.windows.cleaned},
    )
    with pytest.raises(ModelStoreError, match="unsafe"):
        model_module._write_self_test(tmp_path / "unsafe.npz", decoder, unsafe)

    no_classes = BoundDecoder(RiemannEpochAdapter(), object(), "bad")  # type: ignore[arg-type]
    with pytest.raises(ModelStoreError, match="class"):
        model_module._estimator_classes(no_classes)

    original_version = model_module.version

    def missing(distribution: str) -> str:
        if distribution == "numpy":
            raise PackageNotFoundError(distribution)
        return original_version(distribution)

    monkeypatch.setattr(model_module, "version", missing)
    with pytest.raises(ModelStoreError, match="missing"):
        model_module.dependency_versions()
    monkeypatch.undo()

    failure_root = tmp_path / "failure"
    monkeypatch.setattr(model_module.joblib, "dump", lambda *args, **kwargs: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        ModelStore.save(
            failure_root,
            decoder,
            dataset.windows,
            report,
            {},
            seed=7,
        )
    assert not list(failure_root.glob(".chrono-model-*"))
