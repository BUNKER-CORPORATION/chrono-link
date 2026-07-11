# Track A Scope and Evidence Report

Evidence date: 2026-07-10. Reference host: Windows, Python 3.11.9, CPU-only.

## Demonstrated locally

- 121 unit/integration/negative-path tests, including a real BrainFlow synthetic-board lifecycle and
  first settled window within 6.5 seconds.
- 91.91% overall branch coverage; each required critical module is above 90%.
- Ruff and strict mypy pass.
- Hash-locked runtime installation, sdist/wheel build, external clean-wheel install, all installed
  help forms, `pip check`, and packaged headless smoke pass.
- A two-second BrainFlow synthetic plot-only run records and replays without gaps or resource leaks.
- Exact three-seed, five-fold, 25-permutation validation passes both decoders: Riemann grand balanced
  accuracy 0.9875 and vector 0.9917; the minimum observed fold is 0.9375 and permutation p95 is at
  most 0.575.
- Both immutable model kinds save/load with prediction parity; tampering is rejected before unpickle.
- Both locked 200-window workloads pass: Riemann total p50/p95 1.06/1.82 ms, vector 2.22/3.96 ms,
  with zero 256 ms deadline misses.
- Headless PNG is 800×600, informative, and accompanied by strict JSON.

Generated local evidence is intentionally gitignored. Reproduce it with the commands in
`docs/reproducibility.md`.

## Not demonstrated

- Cyton hardware streaming or any other person-connected acquisition.
- Electrode contact quality, signal safety, or real EEG validity.
- Human motor-imagery separability, clinical efficacy, medical use, or the neuroscience thesis.
- GPU execution, LSL interoperability, deep learning, or a desktop GUI.
- Linux execution on this local host. Linux is configured in CI and must be observed in an actual CI
  run before cross-platform acceptance is marked complete.

## Explicit deferral

Optional batch Riemannian session recentering (FR-041 / AC-011) is deferred. Fixed Track A does not
require it, and no incremental/adaptive calibration claim is made.
