# Configuration reference

Chrono Link combines strict JSON, a small environment allowlist, and command-line overrides into an
immutable `ChronoConfig`. Unknown keys fail closed so misspellings cannot silently change a run.

## Precedence

```text
CLI > environment > JSON file > dataclass/profile default
```

Only values explicitly supplied at a higher level replace lower-level values. The final canonical
JSON is stable and SHA-256 fingerprinted for recordings, validation, and model contracts.

## JSON document

Start from [examples/config.synthetic.json](../examples/config.synthetic.json).

### Root fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `board` | object | synthetic profile | Source and channel selection |
| `signal` | object | Track A signal defaults | Window and filter contract |
| `output_root` | path string | `.` | Local output root carried in canonical config |

No additional root fields are accepted.

### `board`

| Field | Type | Default | Rules |
| --- | --- | --- | --- |
| `profile` | string | `synthetic` | Exactly `synthetic` or `cyton` |
| `serial_port` | string or null | null | Required immediately before Cyton acquisition |
| `decode_channels` | array of strings | `["C3", "C4"]` | Non-empty and unique |
| `record_channels` | `"all"` or array | `"all"` | Must contain every decode channel |

Channel names are mapped against runtime board facts. Configuration does not hard-code BrainFlow row
indices.

### `signal`

| Field | Type | Default | Rules |
| --- | --- | --- | --- |
| `window_samples` | integer | 250 | Positive |
| `hop_samples` | integer | 64 | Positive |
| `warmup_samples` | integer | 1000 | Positive |
| `mains_hz` | integer | 50 | Exactly 50 or 60 |
| `notch_q` | number | 30.0 | Positive |
| `dc_highpass_hz` | number | 0.5 | Positive and below Nyquist |
| `flat_abs_tol` | number | `1e-12` | Absolute flat-signal tolerance |
| `flat_rel_tol` | number | `1e-8` | Relative flat-signal tolerance |
| `filter_bands` | array of band objects | broadband/MI/mu/beta | Names unique; high edges below Nyquist |
| `vector_feature_bands` | array of strings | `["mu", "beta"]` | Each name must be filtered |

A band object has `name`, `low_hz`, and `high_hz`. Low/high ordering and source-specific Nyquist
compatibility are validated before processing.

## Environment allowlist

| Variable | Maps to | Valid example |
| --- | --- | --- |
| `CHRONO_BOARD` | `board.profile` | `synthetic` |
| `CHRONO_SERIAL_PORT` | `board.serial_port` | `COM3` or `/dev/ttyUSB0` |
| `CHRONO_MAINS_HZ` | `signal.mains_hz` | `50` |
| `CHRONO_OUTPUT_ROOT` | `output_root` | `artifacts/local` |

Any other variable beginning with `CHRONO_` is an error. This deliberately prevents stale or
misspelled deployment configuration from being ignored.

## CLI overrides

`chrono-stream` exposes source, channel, mains, model, output, recording, and duration overrides.
Run `chrono-stream --help` for the installed-version contract. Important combinations:

- `--replay` is mutually exclusive with live-board, serial-port, and recording inputs.
- `--record` requires a finite positive `--seconds` no greater than one hour.
- `--trust-model` requires `--model`; a model without explicit trust is refused.
- Headless output refuses any pre-existing target file.

Validation, training, and smoke commands intentionally expose fewer runtime configuration fields so
their evidence protocols remain controlled.

## Failure behavior

Configuration errors return CLI exit code 2 and a strict JSON error object containing context, error
type, and message. Raw samples are not included. JSON with duplicate semantic fields, unknown keys,
NaN/Infinity constants, invalid types, or incompatible channel/band selections is rejected.
