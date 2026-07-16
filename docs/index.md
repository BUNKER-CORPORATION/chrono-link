# Chrono Link documentation

This directory is the operational and engineering manual for Chrono Link. The root
[README](../README.md) is the product entry point; the documents below provide the normative detail
needed to develop, review, operate, and release the software.

## Choose your path

| Audience | Start here | Then read |
| --- | --- | --- |
| New authorized user | [README quick start](../README.md#quick-start) | [Configuration](configuration.md), [Operations](operations.md) |
| Developer | [Development](development.md) | [Architecture](architecture.md), [Contributing](../CONTRIBUTING.md) |
| Reviewer or maintainer | [Architecture](architecture.md) | [Scope/evidence](scope-report.md), [Implementation spec](../specs/chrono-link-track-a-software-foundation-implementation-spec.md) |
| Operator | [Operations](operations.md) | [Security/privacy](security-and-privacy.md), [Reproducibility](reproducibility.md) |
| Release owner | [Release process](release-process.md) | [Governance](../GOVERNANCE.md), [Changelog](../CHANGELOG.md) |
| Security reviewer | [Security/privacy](security-and-privacy.md) | [Security policy](../SECURITY.md), [Architecture](architecture.md) |

## Document map

- [Architecture](architecture.md): module ownership, state transitions, data flow, persistence, and
  non-negotiable invariants.
- [Configuration](configuration.md): profiles, JSON schema, environment variables, CLI precedence,
  and validation behavior.
- [Development](development.md): setup, quality gates, test tiers, dependency changes, and review.
- [Operations](operations.md): preflight, command runbooks, exit codes, artifacts, and troubleshooting.
- [Reproducibility](reproducibility.md): deterministic boundaries, reference locks, and evidence
  reproduction.
- [Security and privacy](security-and-privacy.md): assets, threats, controls, model trust, and data
  handling.
- [Release process](release-process.md): versioning, evidence, packaging, tagging, rollback, and
  post-release work.
- [Scope and evidence](scope-report.md): what has actually been observed and what remains explicitly
  unproven.

## Authority and conflict resolution

When documents disagree, use this order:

1. [LICENSE](../LICENSE) for legal permission.
2. Approved specifications under `specs/` for product and acceptance contracts.
3. Executable tests, package metadata, and source code for current implemented behavior.
4. Architecture and security documentation for engineering policy.
5. README and examples for onboarding convenience.

A mismatch is a defect. Update the stale layer in the same pull request; do not silently choose the
more convenient interpretation.

Generated evidence under `artifacts/`, recordings under `recordings/`, and model payloads under
`models/` are intentionally not documentation and are not tracked by Git.
