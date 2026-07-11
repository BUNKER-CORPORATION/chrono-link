# Reproducibility

## Reference environment

- Python: 3.11.9
- RNG fixture seed: 20260709
- Validation seeds: 7, 42, 20260709
- Runtime and development packages: hash-pinned under `locks/`
- Build backend: setuptools 80.9.0 and wheel 0.47.0

Install a lock with `--require-hashes`, then install the project/wheel with `--no-deps`. This keeps
package metadata useful to downstream users while preventing resolver drift in evidence runs.

## Determinism boundaries

The synthetic generator accepts an integer seed or a NumPy `Generator`. A fixed seed produces
array-identical trials and labels in the pinned environment. Each trial is independently reset and
processed through the causal live path; trial IDs remain grouped across all four overlapping windows.

Validation uses `StratifiedGroupKFold(5, shuffle=True, random_state=seed)`. Trial probabilities are
averaged before scoring, and 25 trial-label permutations per seed retain all four windows under the
same permuted label. A training report is accepted only when its exact normative seed, permutation,
filter, feature, package-version, and dataset fingerprints match.

## Reproducing evidence

~~~powershell
chrono-validate --decoder both --out artifacts\full-validation.json
chrono-smoke --windows 200 --out-dir artifacts\smoke
python -m pytest --cov=chrono_link --cov-branch --cov-report=json
python tools\check_coverage.py
python -m build
~~~

Generated `recordings/`, `models/`, and `artifacts/` are root-gitignored because recordings can be
sensitive and models use trusted-code serialization.
