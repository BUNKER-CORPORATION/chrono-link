## Purpose

<!-- What problem does this change solve, and why now? -->

## Change

<!-- Describe the implementation and the contracts or files it changes. -->

## Risk and rollback

<!-- Cover runtime, data, model, schema, dependency, security, and compatibility risk. -->

## Evidence

<!-- List exact local commands and observed CI runs. Do not attach raw EEG or secrets. -->

## Scope and claims

<!-- State whether scientific, hardware, safety, privacy, or performance claims changed. -->

## Checklist

- [ ] The change is authorized for this proprietary repository.
- [ ] Tests cover new behavior and relevant failure paths.
- [ ] Ruff format/lint, strict mypy, pytest, coverage, and repository policy pass.
- [ ] Runtime/decoder changes pass the 200-window smoke gate.
- [ ] Persisted schemas and model contracts are unchanged or explicitly migrated.
- [ ] Documentation, changelog, and examples are updated.
- [ ] No recordings, secrets, personal data, or generated model payloads are included.
- [ ] The rollback path is clear.
