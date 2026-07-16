# Operations and runbooks

Chrono Link Track A is a local single-process application. It has no server daemon, database, or
remote telemetry. Operators are responsible for the host, source device, storage, and artifact trust.

## Preflight

1. Confirm CPython 3.11.9 and install the committed lock with hashes.
2. Run `python -m pip check`.
3. Confirm the output, recording, and model paths are on approved local storage.
4. For a model, verify its provenance and plan to pass `--trust-model` explicitly.
5. For Cyton configuration, confirm the serial port and physical/safety process outside this software.
6. Do not connect a person or make a medical decision based on Track A synthetic evidence.

## Runbook: synthetic health check

```powershell
chrono-stream --board synthetic --seconds 2 --headless `
  --out-dir artifacts\health-<unique-id>
```

Success returns exit code 0 and writes `signal.png`, `metrics.json`, and `predictions.json`. Inspect
`metrics.json` for lifecycle completion, window counts, reset/gap counts, and stage timings.

## Runbook: bounded recording

```powershell
chrono-stream --board synthetic --seconds 30 `
  --record recordings\session-<unique-id>.npz `
  --headless --out-dir artifacts\record-<unique-id>
```

Recording requires a duration so memory and disk use are bounded before acquisition starts. The
recorder checks available disk, writes a same-filesystem temporary file, flushes it, and refuses to
replace the destination. An aborted run does not publish a valid final recording.

## Runbook: deterministic replay

```powershell
chrono-stream --replay recordings\session-<unique-id>.npz `
  --headless --out-dir artifacts\replay-<unique-id>
```

Replay validates the entire recording before processing and preserves continuity events. Use replay
for incident reproduction and regression tests because chunk sizes can vary without changing causal
results.

## Runbook: decoder evidence and training

```powershell
chrono-validate --decoder both --seeds 7,42,20260709 `
  --out artifacts\validation-<unique-id>.json

chrono-train --decoder riemann `
  --validation-report artifacts\validation-<unique-id>.json `
  --out-dir models
```

Training refuses a report whose decoder, seeds, permutation count, dataset, filter, feature, package,
or protocol fingerprint is stale. Treat the resulting directory as a trusted-code artifact.

## Runbook: trusted model replay

```powershell
chrono-stream --replay recordings\session-<unique-id>.npz `
  --model models\<model-id> --trust-model `
  --headless --out-dir artifacts\inference-<unique-id>
```

Never add `--trust-model` merely to bypass an error. Confirm the artifact came from an approved build
or training run and that the manifest/contract match the intended source.

## Exit codes

| Code | Class | Operator action |
| --- | --- | --- |
| 0 | Success | Preserve only approved/sanitized evidence |
| 2 | Configuration | Correct arguments, JSON, environment, paths, or incompatible choices |
| 3 | Runtime/source | Inspect source lifecycle, timestamps, counters, stalls, and host resources |
| 4 | Validation gate | Do not train/release; inspect fold, permutation, or acceptance failures |
| 5 | Artifact | Treat recording/model/report as invalid, stale, untrusted, or conflicting |

Errors are emitted as strict JSON with `status`, `context`, `error_type`, and `message` and should not
contain raw arrays.

## Troubleshooting

### `No module named 'pkg_resources'`

The environment did not honor the reference dependency boundary. Install the committed lock and
confirm `setuptools==80.9.0`; do not install an unconstrained latest setuptools with BrainFlow 5.22.2.

### Output already exists

Chrono Link refuses overwrite. Choose a new run identifier or move the prior artifact through an
approved retention process. Do not delete evidence automatically inside a production run.

### No samples or source stall

Confirm the source is started, the serial port/profile is correct, and BrainFlow can acquire outside
the pipeline. A stall is fatal after the configured timeout; restarting creates a new continuity
domain.

### Recording rejected

Do not attempt repair in place. Preserve the original if incident analysis is needed, inspect the
reported schema/type/continuity violation, and acquire or generate a new valid record.

### Model rejected

Check trust acknowledgement, provenance, manifest schema, hashes, Python/dependency versions,
channels, sample rate, window/hop, and filter/feature fingerprints. Do not edit the manifest or payload
to force acceptance.

## Retention and cleanup

Recordings, models, plots, and JSON evidence remain local and are gitignored. Retention, encryption,
backup, access, and deletion are deployment responsibilities. Remove data only under the applicable
company, legal, study, and incident-preservation policy.
