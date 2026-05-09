# Contributing to AquaCLR

Thanks for your interest in Project LEGION's marine-snow removal
front-end. This document is the canonical onboarding for contributors.

## Code of Conduct

Be excellent to each other. Project LEGION follows the spirit of the
[Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Report unacceptable behaviour via a GitHub issue with the
`code-of-conduct` label or to the maintainers listed in `pyproject.toml`.

## Getting set up

We use [`uv`](https://github.com/astral-sh/uv) for dependency
management and Python toolchain pinning.

```bash
git clone https://github.com/goldr0g3r/aquaclr.git
cd aquaclr
uv sync --extra dev          # installs core + dev tools
uv run pre-commit install    # installs the git hook
```

Verify the toolchain works end-to-end:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

If any of these fail on a clean clone, please open an issue — that's a
real bug for us, not a configuration quirk on your side.

## Branching and commits

- Default branch: **`main`** (always green).
- Active development branch: **`develop`** (CI also runs here).
- Feature branches: `feat/<short-slug>`, e.g. `feat/trt-engine-builder`.
- Bug fixes: `fix/<short-slug>`. Hotfixes for `main`: `hotfix/<slug>`.
- Release branches: `release/v<x.y.z>`.

Please use [Conventional Commits](https://www.conventionalcommits.org/)
in commit messages so that release notes can be generated mechanically:

```
feat(model): add channels-last memory format toggle
fix(loss): clamp transmission denominator before division
docs(readme): document AQUACLR_DATA_ROOT env var
chore(ci): bump uv setup action to v3
```

Keep PRs small, focused, and rebased on `develop`. Squash before merge
unless the history is genuinely useful.

## Style and quality bar

- **Linting / formatting:** [Ruff](https://docs.astral.sh/ruff/),
  configuration in `pyproject.toml`. Line length 100. Quote-style
  double. Run `uv run ruff format` before committing.
- **Type-checking:** [mypy](https://mypy.readthedocs.io/) in `strict`
  mode against `src/aquaclr`. New code must type-check cleanly.
- **Docstrings:** Google convention (`pydocstyle` profile `google`).
  Public APIs must have docstrings; private helpers may use single-line
  doc comments.
- **Tests:** [pytest](https://docs.pytest.org/) with strict markers.
  Use the `gpu`, `trt`, and `slow` markers as appropriate so the CPU
  matrix stays fast.
- **Pre-commit:** runs `ruff` (lint + format), `mypy`, hygiene hooks,
  and `codespell`. Don't bypass with `--no-verify`.

## Tests

```bash
uv run pytest -q                       # full suite
uv run pytest -q -k physics            # filter by name
uv run pytest -q -m "not gpu and not trt"   # CPU-only
uv run pytest -q --cov=aquaclr --cov-report=term-missing
```

CI runs the suite on Python 3.10 / 3.11 / 3.12 (Linux), plus an ONNX
export smoke test that mirrors what TensorRT will see at deployment.
GPU and TensorRT-marked tests are auto-skipped when the runtime isn't
detected.

## Adding new physics or losses

The Jaffe–McGlamery operators in `src/aquaclr/utils/physics.py` are
the **single source of truth** for the image-formation model. If you
need to extend or replace them:

1. Keep `apply_forward_jaffe_mcglamery` and `invert_jaffe_mcglamery`
   numerically consistent — the `phys` term in
   `PhysicsInformedLoss` only makes sense when forward and inverse
   match.
2. Document any new physical assumptions (e.g. wavelength-dependent
   attenuation) in the module docstring with the SI units.
3. Add unit tests covering the boundary conditions you care about
   (`t → 0`, `t → 1`, saturated `B`).

## Adding new datasets

Each dataset lives under `src/aquaclr/data/<name>_dataset.py` and
exposes a Lightning `LightningDataModule`. Please:

- Stream-download with `aquaclr.data.download.fetch_archive`, which
  enforces MD5 verification.
- Apply geometric augmentations to the `(I, J, t_gt)` triple
  atomically (see `aquaclr.data.transforms`); photometric jitter goes
  on `I` only.
- Add a Hydra config under `configs/data/<name>.yaml` and update
  `configs/data/combined.yaml` if it should mix in.

## Filing issues

Please use the issue templates. For bug reports, include:

- The exact `uv run` / `python` command you ran.
- The full traceback.
- `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
- OS, GPU, CUDA driver, and `uv pip list` output for the relevant
  packages.

For feature requests, link to the relevant paper(s), describe the
expected interface, and (ideally) sketch how it slots into the M1
roadmap.

## Pull request checklist

- [ ] Branched off `develop` and rebased before pushing.
- [ ] `uv run ruff check . && uv run ruff format --check .` passes.
- [ ] `uv run mypy` passes.
- [ ] `uv run pytest -q` passes locally (CPU at minimum).
- [ ] New / changed code is covered by tests.
- [ ] Docstrings updated; README touched if user-facing behaviour
      changed.
- [ ] Conventional Commit title and a clear PR description with the
      "why".

By submitting a pull request you agree that your contribution is
licensed under the Apache License, Version 2.0 — the same as the rest
of the project.
