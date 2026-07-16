# Release process

Chrono Link follows semantic versioning. While the package is below `1.0.0`, minor releases may change
documented pre-release APIs, but persisted recording/model schemas and scientific evidence contracts
still require explicit compatibility decisions.

## Release inputs

- A reviewed commit on `main` with green required CI.
- Clean working tree and no untracked release inputs.
- Version updated consistently in `pyproject.toml` and `src/chrono_link/__init__.py`.
- `CHANGELOG.md` with an accurately dated version section and migration notes.
- Current dependency locks, specs, docs, examples, and scope/evidence report.
- Authorization from the repository owner.

## Candidate verification

1. Create a release branch from current `main`.
2. Update version and changelog; do not include unrelated work.
3. Install the development lock from scratch and install the project with `--no-deps`.
4. Run formatting, lint, strict typing, full tests, branch coverage, repository policy, `pip check`,
   and the 200-window smoke gate.
5. Re-run normative decoder evidence if math, dependencies, defaults, filters, features, validation, or
   model code changed.
6. Build sdist and wheel with `python -m build`.
7. Create a new clean environment, install only the runtime lock and wheel, run all four `--help`
   surfaces, `pip check`, and packaged smoke.
8. Inspect archive contents: the wheel must contain package code, `py.typed`, metadata, and the
   proprietary LICENSE while excluding tests, recordings, models, and local artifacts. Inspect the
   source distribution separately; it may contain tests but must exclude generated/sensitive data.
9. Record SHA-256 for each candidate artifact.
10. Open a release pull request and wait for supported-platform CI.

## Tag and publish

After approval, merge the release pull request and create an annotated `vMAJOR.MINOR.PATCH` tag on the
reviewed merge commit. Release artifacts must come from the clean CI/tag build, never from a developer
worktree. Publish only to the authorized proprietary distribution channel and attach checksums,
changelog, supported environment, known limitations, and upgrade/rollback instructions.

No public PyPI publication is authorized by this repository.

## Post-release verification

- Install the distributed artifact through the real consumer channel.
- Confirm version, metadata, proprietary license, import, CLI help, `pip check`, and smoke.
- Confirm the GitHub release/tag points at the expected commit and artifacts/checksums match CI.
- Update the scope/evidence report only with runs that were actually observed.

## Rollback

Stop distribution of a faulty candidate. Consumers return to the last trusted package and compatible
model/recording contract. Revert the offending source change or create a reviewed forward fix; do not
force-rewrite shared `main` or release tags. If confidentiality, integrity, or privacy may be affected,
follow the private security process before public release notes.
