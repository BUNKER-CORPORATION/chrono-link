# Security policy

## Supported versions

Chrono Link is pre-release software. Security fixes are applied to `main`; no older commit or model
bundle is supported unless a separate written agreement says otherwise.

| Version | Security support |
| --- | --- |
| `main` | Supported |
| Historic commits and untrusted forks | Not supported |

## Report a vulnerability privately

Do not disclose suspected vulnerabilities, privacy leaks, credentials, recordings, or exploitable
model artifacts in a public issue. Use GitHub's private
[security advisory form](https://github.com/BUNKER-CORPORATION/chrono-link/security/advisories/new).
If that channel is unavailable, contact the repository owner through an approved Bunker Corporation
channel and include only the minimum information needed to establish contact.

Include the affected commit/version, platform, reproduction steps, impact, and whether sensitive data
may have been exposed. Do not attach real EEG, credentials, or proprietary third-party data unless a
secure transfer method has been agreed.

## Security boundaries

- Joblib model payloads can execute code. Chrono Link refuses to load one without explicit local
  trust and verifies its manifest, contract, and hashes before deserialization.
- NPZ recordings are loaded without pickle, validated against strict schemas, and should still be
  treated as sensitive biosignal-adjacent data.
- Raw signal samples must not enter routine logs, metrics, issue reports, or CI artifacts.
- CLI output paths refuse accidental overwrite, and persisted artifacts publish atomically where the
  platform permits.
- Unknown configuration keys and non-finite numeric values fail closed.

See [docs/security-and-privacy.md](docs/security-and-privacy.md) for the threat model and operational
controls.
