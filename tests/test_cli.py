from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import chrono_link.__main__ as module_main
from chrono_link.cli import smoke, stream, train, validate
from chrono_link.cli.common import (
    EXIT_ARTIFACT,
    EXIT_CONFIG,
    EXIT_RUNTIME,
    EXIT_SUCCESS,
    EXIT_VALIDATION,
    csv_tuple,
    integer_csv,
    load_validation_report,
    strict_json_file,
)
from chrono_link.config import ChronoConfig
from chrono_link.recording import SessionRecord
from chrono_link.validation import (
    FoldResult,
    SeedResult,
    ValidationError,
    ValidationReport,
)


def report(kind: str = "riemann", *, passed: bool = True) -> ValidationReport:
    fold = FoldResult(0, 1.0, 32, 8)
    seed = SeedResult(
        seed=20260709,
        folds=(fold,),
        mean_balanced_accuracy=1.0,
        minimum_fold_balanced_accuracy=1.0,
        aggregate_balanced_accuracy=1.0,
        correct_trials=40,
        trial_count=40,
        binomial_pvalue=1e-12,
        permutation_p95=0.5,
        passed=passed,
    )
    return ValidationReport(
        schema_version=1,
        decoder_kind=kind,  # type: ignore[arg-type]
        seeds=(20260709,),
        results=(seed,),
        grand_mean_balanced_accuracy=1.0,
        dataset_fingerprints=("fp",),
        protocol_fingerprint="protocol",
        passed=passed,
        failures=() if passed else ("failed",),
    )


def write_record(path: Path, samples: int = 20) -> None:
    sequence = np.arange(samples, dtype=np.int64)
    time = sequence / 250
    record = SessionRecord(
        data=np.vstack(
            (np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 12 * time + 0.2))
        ).astype(np.float64),
        timestamps=1_700_000_000.0 + time,
        sequence=sequence,
        package_counter=sequence % 256,
        channel_names=("C3", "C4"),
        units=("brainflow_native", "brainflow_native"),
        fs=250,
        board_id=-1,
        created_utc="2026-07-10T12:00:00Z",
        markers=np.zeros(samples, dtype=np.float64),
        config_json=ChronoConfig.synthetic().canonical_json(),
        events=(),
    )
    with path.open("xb") as output:
        np.savez_compressed(output, **record.as_npz_fields())


@pytest.mark.parametrize("entry", [stream.main, validate.main, train.main, smoke.main])
def test_each_cli_help_exits_without_runtime_side_effects(entry) -> None:
    with pytest.raises(SystemExit) as raised:
        entry(["--help"])
    assert raised.value.code == 0


def test_module_dispatches_every_command(monkeypatch) -> None:
    monkeypatch.setattr(stream, "main", lambda argv=None: 10)
    monkeypatch.setattr(validate, "main", lambda argv=None: 11)
    monkeypatch.setattr(train, "main", lambda argv=None: 12)
    monkeypatch.setattr(smoke, "main", lambda argv=None: 13)
    assert module_main.main(["stream"]) == 10
    assert module_main.main(["validate"]) == 11
    assert module_main.main(["train"]) == 12
    assert module_main.main(["smoke"]) == 13


def test_common_parsers_and_strict_json(tmp_path: Path) -> None:
    assert csv_tuple("C3, C4") == ("C3", "C4")
    assert integer_csv("7,42") == (7, 42)
    with pytest.raises(Exception):
        csv_tuple("C3,C3")
    with pytest.raises(Exception):
        integer_csv("nope")
    with pytest.raises(Exception):
        integer_csv("")
    valid = tmp_path / "valid.json"
    valid.write_text('{"ok":true}', encoding="utf-8")
    assert strict_json_file(valid) == {"ok": True}
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"bad":NaN}', encoding="utf-8")
    with pytest.raises(ValidationError):
        strict_json_file(invalid)


def test_validation_report_loader_handles_single_and_combined(tmp_path: Path) -> None:
    single = tmp_path / "single.json"
    single.write_text(report().to_json(), encoding="utf-8")
    assert load_validation_report(single, "riemann").passed
    combined = tmp_path / "combined.json"
    combined.write_text(
        json.dumps({"reports": {"riemann": json.loads(report().to_json())}}),
        encoding="utf-8",
    )
    assert load_validation_report(combined, "riemann").decoder_kind == "riemann"
    with pytest.raises(ValidationError):
        load_validation_report(combined, "vector")


def test_validate_cli_success_gate_failure_and_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(validate, "validate_decoder", lambda kind, **kwargs: report(kind))
    output = tmp_path / "validation.json"
    assert validate.main(["--out", str(output), "--permutations", "0"]) == EXIT_SUCCESS
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"]
    assert set(payload["reports"]) == {"riemann", "vector"}
    assert validate.main(["--out", str(output), "--permutations", "0"]) == EXIT_CONFIG
    assert validate.main(["--permutations", "-1"]) == EXIT_CONFIG

    failed = tmp_path / "failed.json"
    monkeypatch.setattr(
        validate,
        "validate_decoder",
        lambda kind, **kwargs: report(kind, passed=False),
    )
    assert validate.main(["--decoder", "riemann", "--out", str(failed)]) == EXIT_VALIDATION
    monkeypatch.setattr(validate, "validate_decoder", lambda kind, **kwargs: 1 / 0)
    assert validate.main(["--out", str(tmp_path / "error.json")]) == EXIT_RUNTIME


def test_train_cli_success_and_failure_mappings(tmp_path: Path, monkeypatch) -> None:
    passing = report("riemann")
    session = SimpleNamespace(metadata={"seed": 20260709})
    dataset = SimpleNamespace(windows=object(), labels=np.asarray([0, 1], dtype=np.int64))

    class FakeGenerator:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def generate(self):
            return session

    class FakeDecoder:
        def fit(self, windows, labels):
            return self

    monkeypatch.setattr(train, "validate_decoder", lambda *args, **kwargs: passing)
    monkeypatch.setattr(train, "validate_training_report", lambda value: None)
    monkeypatch.setattr(train, "SyntheticMISessionGenerator", FakeGenerator)
    monkeypatch.setattr(train, "dataset_fingerprint", lambda value: "fp")
    monkeypatch.setattr(train, "preprocess_synthetic_session", lambda value: dataset)
    monkeypatch.setattr(train, "decoder_factory", lambda kind: FakeDecoder)
    destination = tmp_path / "models" / "model"
    monkeypatch.setattr(
        train.ModelStore,
        "save",
        lambda *args, **kwargs: destination,
    )
    assert (
        train.main(["--decoder", "riemann", "--out-dir", str(tmp_path / "models")])
        == EXIT_SUCCESS
    )
    assert train.main(["--decoder", "riemann", "--permutations", "-1"]) == EXIT_CONFIG

    monkeypatch.setattr(
        train,
        "validate_decoder",
        lambda *args, **kwargs: replace(passing, passed=False, failures=("bad",)),
    )
    assert train.main(["--decoder", "riemann"]) == EXIT_VALIDATION
    monkeypatch.setattr(train, "validate_decoder", lambda *args, **kwargs: passing)
    monkeypatch.setattr(train, "dataset_fingerprint", lambda value: "stale")
    assert train.main(["--decoder", "riemann"]) == EXIT_ARTIFACT
    monkeypatch.setattr(train, "dataset_fingerprint", lambda value: "fp")
    monkeypatch.setattr(train.ModelStore, "save", lambda *args, **kwargs: 1 / 0)
    assert train.main(["--decoder", "riemann"]) == EXIT_RUNTIME


def test_stream_cli_replay_headless_and_argument_errors(tmp_path: Path) -> None:
    record = tmp_path / "record.npz"
    write_record(record)
    output = tmp_path / "output"
    assert (
        stream.main(["--replay", str(record), "--headless", "--out-dir", str(output)])
        == EXIT_SUCCESS
    )
    assert (output / "signal.png").exists()
    assert stream.main(["--model", str(tmp_path / "model")]) == EXIT_ARTIFACT
    assert stream.main(["--trust-model"]) == EXIT_CONFIG
    assert stream.main(["--record", str(tmp_path / "x.npz")]) == EXIT_CONFIG
    assert stream.main(["--replay", str(record), "--board", "synthetic"]) == EXIT_CONFIG
    assert stream.main(["--seconds", "3601"]) == EXIT_CONFIG
    assert (
        stream.main(["--replay", str(record), "--headless", "--out-dir", str(output)])
        == EXIT_CONFIG
    )
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not npz")
    assert stream.main(["--replay", str(corrupt)]) == EXIT_ARTIFACT


def test_smoke_cli_minimum_and_full_workload(tmp_path: Path) -> None:
    assert smoke.main(["--windows", "199"]) == EXIT_CONFIG
    output = tmp_path / "smoke"
    assert smoke.main(["--windows", "200", "--out-dir", str(output)]) == EXIT_SUCCESS
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["passed"]
    assert len(metrics["decoders"]) == 2
