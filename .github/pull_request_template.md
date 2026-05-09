<!-- Thanks for contributing! Please fill in the sections below. -->

## Summary

<!-- 1–3 bullets describing WHAT this change does and WHY it's needed. -->

-
-

## Type of change

- [ ] feat — new user-facing capability
- [ ] fix — bug fix
- [ ] perf — performance improvement
- [ ] refactor — internal restructuring with no behaviour change
- [ ] docs — documentation only
- [ ] test — adds or improves tests
- [ ] chore / ci — build, tooling, dependencies

## Roadmap milestone

- [ ] M1 (model / loss / training)
- [ ] M1.5 (TensorRT / ROS 2)
- [ ] M2 (temporal / INT8 / sea-trial)
- [ ] N/A

## Implementation notes

<!-- Anything reviewers should know: trade-offs, alternatives discarded,
     numerical-stability concerns, ONNX/TRT compatibility caveats. -->

## Test plan

<!-- How did you verify this works? Include commands and expected output. -->

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

- [ ] Added / updated unit tests
- [ ] Verified on CPU
- [ ] Verified on CUDA (RTX 3050 / equivalent)
- [ ] Verified ONNX export still passes the smoke test (if model surface changed)

## Checklist

- [ ] Branched off `develop` and rebased before opening this PR.
- [ ] Conventional Commit title (`feat(...)`, `fix(...)`, ...).
- [ ] Docstrings updated; `README.md` touched if user-facing behaviour changed.
- [ ] No secrets, large binaries, or dataset files committed.
- [ ] By submitting, I agree my contribution is licensed under Apache-2.0.
