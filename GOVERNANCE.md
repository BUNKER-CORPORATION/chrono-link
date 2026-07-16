# Governance

Chrono Link is owned and maintained by Bunker Corporation. This document defines the repository's
engineering decision process; commercial, licensing, personnel, and product decisions remain with
the company.

## Roles

- **Repository owner:** accountable for roadmap, licensing, release authorization, and access.
- **CODEOWNER:** accountable for architectural integrity, review, and merge approval in the paths
  assigned by `.github/CODEOWNERS`.
- **Contributor:** an authorized collaborator who prepares changes and evidence for review.

One person may hold multiple roles. GitHub access alone does not confer a role or license.

## Decision policy

Routine fixes and documentation changes are decided in pull-request review. Changes to persisted
schemas, model contracts, scientific validation, security boundaries, supported Python/platform
versions, or the runtime data flow require an updated specification or architecture decision in the
same pull request.

When evidence conflicts with a claim, the claim is reduced or removed. A configured workflow is not
considered demonstrated until an actual run is observed.

## Branch and release policy

- `main` is the integration and release source of truth.
- Direct pushes to `main` are prohibited by policy; changes arrive through reviewed pull requests.
- Required CI must pass on every supported operating system before merge.
- Releases use semantic versioning and an annotated `vMAJOR.MINOR.PATCH` tag from a reviewed commit.
- Release artifacts are built from the tag in a clean locked environment; locally built files are
  never substituted.
- Rollback is performed by reverting the offending change or returning consumers to the last trusted
  model/package contract. Shared history is not force-rewritten during ordinary operations.

## Ownership changes

Changes to CODEOWNERS, licensing, release authority, or security contacts require repository-owner
approval and must be auditable in Git history.
