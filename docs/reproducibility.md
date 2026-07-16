# Reproducibility

## Reference environment

- Python: CPython 3.11.9, 64-bit.
- Compute: CPU-only.
- RNG fixture seed: 20260709.
- Normative validation seeds: 7, 42, 20260709.
- Cross-validation: grouped five-fold, shuffled with the current seed.
- Permutations: 25 trial-label permutations per seed.
- Runtime and development packages: exact versions and hashes under `locks/`.
- Build backend: setuptools 80.9.0 and wheel 0.47.0.

Install a lock with `--require-hashes`, then install the project or wheel with `--no-deps`. This keeps
package metadata useful to downstream tools while preventing a second resolver pass from changing the
evidence environment.

## Compatibility boundary

BrainFlow 5.22.2 declares setuptools without an upper bound and imports `pkg_resources` in its runtime
native-library fallback. Setuptools releases that removed `pkg_resources` break real BrainFlow
synthetic-board tests on both Windows and Linux. Chrono Link therefore declares and locks
`setuptools==80.9.0` as a runtime dependency until a tested BrainFlow version removes that dependency.

This pin must not be relaxed based only on `pip check` or package import; the two real synthetic-board
lifecycle tests and packaged smoke must pass on both supported platforms.

## Determinism boundaries

The synthetic generator accepts an integer seed or a NumPy `Generator`. A fixed seed produces
array-identical trials and labels in the pinned environment. Each trial is independently reset and
processed through the causal live path; trial IDs remain grouped across all four overlapping windows.

Validation uses `StratifiedGroupKFold(5, shuffle=True, random_state=seed)`. Trial probabilities are
averaged before scoring, and each permutation retains all windows from a trial under the same
permuted label. A training report is accepted only when its exact normative seeds, permutation count,
filter, feature, package-version, dataset, and protocol fingerprints match.

Floating-point equality outside deterministic fixtures is not promised across arbitrary CPUs,
BLAS implementations, Python versions, or dependency versions. Acceptance uses explicit tolerances
and aggregate gates where exact identity is not the contract.

## Evidence tiers

| Tier | Meaning |
| --- | --- |
| Configured | A file or workflow describes the behavior; no run is implied |
| Locally observed | A named host/environment produced the result |
| CI observed | GitHub Actions produced the result on a named runner image |
| Hardware observed | A named physical source and protocol produced the result |
| Human-subject observed | Approved study evidence exists; Track A has none |

The scope report uses these distinctions. Never promote evidence between tiers by inference.

## Reproduce the software evidence

```powershell
chrono-validate --decoder both --seeds 7,42,20260709 `
  --out artifacts\full-validation.json
chrono-smoke --windows 200 --out-dir artifacts\smoke
python -m pytest --cov=chrono_link --cov-branch --cov-report=json:coverage.json
python tools\check_coverage.py
python tools\check_repository.py
python -m build
```

Record the commit SHA, Python/platform details, lock hashes, command lines, exit codes, and output
artifact hashes. Generated `recordings/`, `models/`, and `artifacts/` remain gitignored because
recordings may be sensitive and models use trusted-code serialization.

## Updating evidence

Any change to Python/dependency versions, source facts, filters, windows, features, synthetic recipe,
folding/scoring, model persistence, or timing measurement invalidates some prior evidence. Update the
affected fingerprints, rerun the appropriate tier, and revise `docs/scope-report.md` with the date and
observed environment. Preserve honest non-claims when only software evidence changes.
