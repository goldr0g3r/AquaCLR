# Chapter 5 — Implementation

> **Learning objectives**
> By the end of this chapter you will be able to:
>
> 1. Locate any concept from Chapters 3–4 in the source tree within seconds.
> 2. Explain why we use a `src/`-layout, Hydra configs, uv, ruff, and mypy strict.
> 3. Reproduce the build, lint, type-check, and test cycle on a clean machine.
> 4. Describe the reproducibility and performance-engineering choices made at code level.
>
> **TL;DR.** The codebase follows the modern Python "src layout"
> with strict typing, ruff formatting, Hydra configs, and Lightning
> training. Every file under 250 lines, every public symbol typed,
> every public function documented with a Google-style docstring
> that includes an "Automotive SiL parallel" paragraph.

## 5.1 Repository layout

```
AquaCLR/
├─ DISSERTATION.md          # the document you are reading (master)
├─ README.md                # project quickstart
├─ MODEL_CARD.md            # intended use / limitations / training data
├─ CONTRIBUTING.md          # contributor onboarding
├─ LICENSE                  # Apache-2.0
├─ pyproject.toml           # uv-managed, ruff + mypy + pytest configuration
├─ .pre-commit-config.yaml  # ruff-format, ruff-check, mypy, codespell hooks
├─ .github/
│  ├─ workflows/ci.yml      # lint + tests + ONNX-export smoke test
│  ├─ ISSUE_TEMPLATE/...    # bug + feature templates
│  └─ pull_request_template.md
├─ configs/                 # Hydra configs (composable: model x data x train)
│  ├─ default.yaml
│  ├─ model/legion_desnow_s.yaml
│  ├─ data/{msrb,lsui,combined,combined_a3000}.yaml
│  └─ train/{rtx3050_bf16,rtx_a3000_bf16}.yaml
├─ src/aquaclr/             # package source ("src layout")
│  ├─ __init__.py           # version, top-level docstring
│  ├─ models/               # network definitions
│  │  ├─ model.py           # LEGIONDeSnowNet (top-level)
│  │  ├─ backbones/mobilenet_v3.py
│  │  ├─ decoders/unet_dsc.py
│  │  └─ heads/{transmission,backscatter}.py
│  ├─ losses/               # loss components
│  │  ├─ physics_loss.py
│  │  ├─ ssim.py
│  │  └─ tv.py
│  ├─ data/                 # datasets + transforms
│  │  ├─ msrb_dataset.py
│  │  ├─ lsui_dataset.py
│  │  ├─ combined_datamodule.py
│  │  ├─ snow_synthesis.py
│  │  ├─ transforms.py
│  │  └─ download.py
│  ├─ training/             # Lightning-side machinery
│  │  ├─ lit_module.py
│  │  └─ callbacks.py
│  ├─ inference/            # ONNX, TRT, benchmark
│  │  ├─ onnx_export.py
│  │  ├─ inference_trt.py
│  │  └─ benchmark.py
│  ├─ ros2/ros2_node.py     # ROS2 Humble/Jazzy node skeleton
│  └─ utils/                # cross-cutting helpers
│     ├─ physics.py         # forward + inverse Jaffe-McGlamery
│     └─ seed.py
├─ scripts/                 # thin CLI entry points
│  ├─ train.py              # @hydra.main entrypoint
│  ├─ evaluate.py           # PSNR/SSIM + optional UIQM/UCIQE (--no-ref)
│  ├─ evaluate_slam_features.py  # ORB/SIFT keypoint stability benchmark
│  ├─ infer_camera.py       # real-time webcam / video inference (OpenCV)
│  ├─ export_onnx.py        # ONNX (+ optional TRT) export
│  └─ download_data.py      # dataset layout verifier
├─ tests/                   # pytest suite
│  ├─ test_model.py
│  ├─ test_physics_loss.py
│  ├─ test_ssim_tv.py
│  ├─ test_data.py
│  └─ test_export.py
├─ notebooks/01_explore_msrb.ipynb
├─ docker/Dockerfile.trt    # NVIDIA CUDA + TRT for deployment
├─ docs/
│  ├─ README.md             # docs index
│  ├─ ARCHITECTURE.md       # design rationale
│  ├─ DEPLOYMENT_FEDORA.md  # Fedora-host deployment runbook
│  └─ dissertation/         # this manuscript
└─ data/.gitkeep            # placeholder for runtime dataset roots
```

The repo is intentionally flat — three levels of nesting maximum
under `src/aquaclr/` — so a reviewer can navigate without an IDE.

### 5.1.1 Why "src layout"?

In a "flat layout" the package sits next to scripts at repo root
(`./aquaclr/...`) and is importable just because Python adds CWD
to `sys.path`. This is fragile in three ways:

1. Tests can accidentally import the **un**-installed package.
2. Any script run from a different directory breaks imports.
3. CI may behave differently from dev because of CWD differences.

A "src layout" forces the package to live under `src/` and to be
**installed** (`pip install -e .` or `uv sync`) before it's
importable. This catches packaging bugs immediately and makes dev,
CI, and Docker behave identically.

## 5.2 Module walk

Below, each subsection describes the **purpose**, the **public
surface**, and the **non-obvious decisions** of one module. Read
in order, this section is a guided tour of the codebase.

### 5.2.1 `src/aquaclr/models/model.py` — `LEGIONDeSnowNet`

[`src/aquaclr/models/model.py`](../../src/aquaclr/models/model.py)

| Public symbol     | Type        | Role                |
| ----------------- | ----------- | ------------------- |
| `LEGIONDeSnowNet` | `nn.Module` | The network         |
| `LEGIONOutputs`   | `dataclass` | Carries `(j, t, b)` |

Highlights:

- The constructor enforces `backbone == "mobilenet_v3_small"` (the
  parameter exists for future ablations).
- `use_channels_last=True` switches the model to channels-last
  memory format, a known speedup on Ampere conv kernels.
- `normalize_input=True` does ImageNet normalisation **inside** the
  model so callers can pass raw `[0, 1]` tensors. This is what
  makes the ROS2 path simple — `cv_bridge` returns `[0, 1]` tensors
  directly.
- `forward()` returns the dataclass; `forward_export()` returns a
  flat tuple. Two methods, one job: the dataclass is ergonomic for
  Python; the tuple is required by the ONNX exporter.
- `estimate_size_mb(dtype=...)` reports parameter byte count for
  any dtype — used in the test suite's budget check.

### 5.2.2 `src/aquaclr/models/backbones/mobilenet_v3.py`

[`src/aquaclr/models/backbones/mobilenet_v3.py`](../../src/aquaclr/models/backbones/mobilenet_v3.py)

Encapsulates torchvision's `mobilenet_v3_small` and exposes pyramid
features at the four expected strides. The stage taps are
hard-coded by **block index**, with channel/reduction metadata in
a `dataclass(frozen=True)` so they're easy to read in unit tests.

Two helper methods, `freeze()` and `unfreeze()`, support
backbone-warmup: for the first N epochs we keep the encoder
frozen so the heads can stabilise without back-propagating
through-pretrained weights.

### 5.2.3 `src/aquaclr/models/decoders/unet_dsc.py`

[`src/aquaclr/models/decoders/unet_dsc.py`](../../src/aquaclr/models/decoders/unet_dsc.py)

Implements `DepthwiseSeparableConv` (3×3 depthwise + 1×1 pointwise

- BN + ReLU6, twice) and `UNetDSCDecoder` (4 up-blocks +
  final 2× upsample). The `_UpBlock.forward` defends against
  non-divisible spatial sizes by re-interpolating the upsampled
  tensor to the skip's shape — see [`tests/test_model.py`](../../tests/test_model.py)
  `test_handles_non_divisible_input`.

### 5.2.4 `src/aquaclr/models/heads/transmission.py` & `backscatter.py`

[`transmission.py`](../../src/aquaclr/models/heads/transmission.py),
[`backscatter.py`](../../src/aquaclr/models/heads/backscatter.py)

Discussed in §4.4. The bias initialisations are the only
non-obvious detail: `+2.0` for the t-head's projection bias
(`sigmoid(2) ≈ 0.88`) and `−1.0` for the B-head's last-layer bias
(`sigmoid(−1) ≈ 0.27`). Both are chosen to match prior beliefs
about typical underwater scenes.

### 5.2.5 `src/aquaclr/utils/physics.py`

[`src/aquaclr/utils/physics.py`](../../src/aquaclr/utils/physics.py)

Two free functions: `apply_forward_jaffe_mcglamery` and
`invert_jaffe_mcglamery`. They exist as functions (not methods)
because the loss module needs `apply_forward_…` independently of
the model.

### 5.2.6 `src/aquaclr/losses/`

| File              | Class / function                                      | Lines |
| ----------------- | ----------------------------------------------------- | ----- |
| `ssim.py`         | `SSIM` (module), `ssim` (function), `psnr` (function) | ~150  |
| `tv.py`           | `total_variation`                                     | ~40   |
| `physics_loss.py` | `PhysicsInformedLoss`, `PhysicsLossOutputs`           | ~180  |

The loss module returns a dataclass (`PhysicsLossOutputs`) with
**every per-term value** as a separate scalar tensor in addition
to the total. That makes Lightning logging trivial:
`outputs.to_log_dict("train/loss/")` yields exactly the keys
TensorBoard / W&B expect.

### 5.2.7 `src/aquaclr/data/`

| File                     | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| `msrb_dataset.py`        | MSRB Dataset + DataModule, with synth-snow fallback |
| `lsui_dataset.py`        | LSUI Dataset + DataModule, optional transmission GT |
| `combined_datamodule.py` | Mix MSRB and LSUI per-batch; LSUI is optional       |
| `snow_synthesis.py`      | Procedural marine-snow renderer (NumPy)             |
| `transforms.py`          | Albumentations train/val pipelines                  |
| `download.py`            | MD5-verified streaming downloader                   |

The combined module's `_AlternatingLoader` pulls one batch from one
loader at a time, sampled by `mix_ratio`. This is simpler than
true zip-style multi-source training because each batch carries a
homogeneous `has_t_gt` flag — the loss can apply or skip the `L_t`
term per-batch without per-sample masking inside.

### 5.2.8 `src/aquaclr/training/`

| File            | Class                                                   | Role                   |
| --------------- | ------------------------------------------------------- | ---------------------- |
| `lit_module.py` | `LEGIONDeSnowLitModule`                                 | Lightning wrapper      |
| `callbacks.py`  | `EMAWeightCallback`, `VRAMMonitor`, `SampleImageLogger` | Cross-cutting concerns |

The Lightning module's `_shared_step` covers both train and val
modes; the only difference is which TorchMetrics object is updated
and which keys are logged.

The EMA callback maintains a shadow copy of trainable parameters
and **swaps it in** for `validation_step` so we always validate on
the smoothed weights, then swaps back before the next training
batch.

### 5.2.9 `src/aquaclr/inference/`

| File               | Public surface                               |
| ------------------ | -------------------------------------------- |
| `onnx_export.py`   | `export_to_onnx()`                           |
| `inference_trt.py` | `build_engine_from_onnx()`, `TensorRTRunner` |
| `benchmark.py`     | `benchmark_pytorch()`                        |

`onnx_export` does three things in sequence: `torch.onnx.export`,
`onnxsim.simplify` (if available), then a numerical parity check
via `onnxruntime`. Failure on parity raises and aborts the export
— never silently produce a bad engine.

`inference_trt` uses **CDI-style** device addressing
(`set_tensor_address`) and `execute_async_v3`, the post-TRT-9 idiom
required to support dynamic shapes cleanly.

### 5.2.10 `src/aquaclr/ros2/ros2_node.py`

[`src/aquaclr/ros2/ros2_node.py`](../../src/aquaclr/ros2/ros2_node.py)

Imports `rclpy` lazily so the package remains importable on
Windows / macOS / any environment without ROS2. The
`_TorchBackend` provides a PyTorch fallback if TRT isn't usable.

### 5.2.11 `scripts/evaluate.py` — PSNR / SSIM / UIQM / UCIQE

[`scripts/evaluate.py`](../../scripts/evaluate.py)

Computes **reference-based** (PSNR, SSIM via TorchMetrics) and,
optionally, **no-reference** (UIQM, UCIQE via `pyiqa`) metrics
against an MSRB checkpoint. Key design points:

- `pyiqa` is an _optional_ import; a missing installation emits a
  warning and skips the no-reference block rather than raising.
- No-reference metrics are gated behind `--no-ref` so the default
  PSNR/SSIM pass incurs no overhead.
- Both UIQM and UCIQE scores are reported as `mean ± std` over the
  test set, matching the statistical reporting convention in §8.10.

### 5.2.12 `scripts/evaluate_slam_features.py` — Keypoint Stability

[`scripts/evaluate_slam_features.py`](../../scripts/evaluate_slam_features.py)

Downstream benchmark measuring whether LEGION-DeSnow improves SLAM
feature extraction quality. For each `(I, Ĵ, J_gt)` triple:

| Step               | Code                                 | Details                              |
| ------------------ | ------------------------------------ | ------------------------------------ |
| Tensor → uint8     | `_tensor_to_uint8`                   | CHW float→HWC uint8                  |
| Grayscale          | `cv2.cvtColor(…, COLOR_RGB2GRAY)`    | Standard OpenCV                      |
| Detect + describe  | `_detect(det, gray)`                 | ORB or SIFT                          |
| Repeatability      | `_repeatability(kps_I, kps_pred, δ)` | Brute-force nearest-kp search        |
| Match inlier ratio | `_match_inlier_ratio_and_score(…)`   | kNN + Lowe ratio + RANSAC homography |

All heavy work is on CPU (OpenCV); the GPU is used only for model
inference in the batch loop. Worker count and batch size are
configurable; `--batch-size 1` is the default to keep per-image
statistics clean.

The four output metrics are described in detail in §8.13.

## 5.3 Configuration with Hydra

### 5.3.1 The composition mechanism

The default config (`configs/default.yaml`) lists three children
under `defaults:`:

```yaml
defaults:
  - _self_
  - model: legion_desnow_s
  - data: combined
  - train: rtx3050_bf16
```

Each child is a separate file under `configs/<group>/<name>.yaml`.
At command line you can swap any one:

```bash
python scripts/train.py model=legion_desnow_s data=msrb train=rtx3050_bf16
python scripts/train.py train.optimizer.lr=1e-4 train.max_epochs=30
python scripts/train.py -m train.optimizer.lr=1e-4,3e-4,1e-3   # multirun
```

### 5.3.2 Why Hydra (not argparse, not pydantic-settings)

| Tool              | Multi-source compose | CLI override | Multirun sweep | Adoption in ML                        |
| ----------------- | -------------------- | ------------ | -------------- | ------------------------------------- |
| argparse          | no                   | yes          | no             | declining                             |
| pydantic-settings | yes                  | partial      | no             | growing in web/api                    |
| **Hydra**         | **yes**              | **yes**      | **yes**        | **dominant in 2026 ML research code** |

Hydra also handles run-directory creation, log routing, and
multirun directory templates out of the box, all of which we use.

### 5.3.3 Reading configs from code

`scripts/train.py` uses `hydra.utils.instantiate(cfg.<group>)` to
materialise objects with the `_target_` key in each YAML. Example:

```yaml
# configs/model/legion_desnow_s.yaml
_target_: aquaclr.models.model.LEGIONDeSnowNet
backbone: mobilenet_v3_small
pretrained: true
...
```

— `instantiate(cfg.model)` returns a fully constructed
`LEGIONDeSnowNet`. This is the single source of truth for the
network's hyper-parameters.

## 5.4 Code style and type safety

### 5.4.1 ruff (lint + format)

`pyproject.toml` configures ruff for line-length 100, double
quotes, py310 target, with these rule families enabled:

`E, F, W, I, B, UP, N, SIM, PL, RUF, ANN, D` — pycodestyle,
pyflakes, isort, bugbear, pyupgrade, naming, simplify, pylint,
ruff-specific, annotations, pydocstyle (Google convention).

A handful of rules are disabled with explicit reasons:

| Disabled    | Reason                                       |
| ----------- | -------------------------------------------- |
| D203 / D213 | Conflict with D211 / D212 we keep            |
| PLR0913     | ML configs legitimately have many parameters |
| PLR2004     | We have many physical-constant magic numbers |

### 5.4.2 mypy (strict)

Every public function and method is type-hinted. `mypy --strict`
runs on `src/aquaclr` in CI; new code that doesn't type-check
fails the merge. Specific compromises:

- `cv2`, `albumentations`, `tensorrt`, `pycuda`, `rclpy`, etc. are
  declared `ignore_missing_imports = true` because they have no
  type stubs.
- The Lightning base classes use generic `Any` parameters in some
  callback signatures; we widen our overrides to match.

### 5.4.3 Pre-commit

`.pre-commit-config.yaml` runs before each commit:

1. `ruff check --fix` (lint + auto-fix)
2. `ruff format` (formatter)
3. `mypy` (against `src/aquaclr/`)
4. Standard hygiene hooks (trailing whitespace, EOF, large files,
   merge conflicts, line endings, secret detection)
5. `codespell` (typos)

## 5.5 Reproducibility engineering

### 5.5.1 Seeding

`aquaclr.utils.seed.seed_everything(seed=1337, deterministic=True)`
seeds Python's `random`, NumPy, and PyTorch (CPU + CUDA). With
`deterministic=True` it also:

- Sets `torch.backends.cudnn.deterministic = True`,
- Disables `cudnn.benchmark`,
- Calls `torch.use_deterministic_algorithms(True, warn_only=True)`,
- Exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` (required for
  deterministic cuBLAS on CUDA 11+).

### 5.5.2 Locked dependencies

`uv.lock` (committed) records exact versions for every
transitive dependency. Re-creating the env on any machine:

```bash
uv sync --extra dev --extra trt
```

### 5.5.3 Hydra run directory

Every run produces a self-contained directory under
`outputs/<timestamp>/` containing the resolved config, logs, and
checkpoints. To reproduce a numerical claim from Chapter 10, the
exact run directory is sufficient.

### 5.5.4 MD5-pinned datasets

Every dataset download (`aquaclr.data.download.fetch_archive`)
verifies an MD5 checksum after the file lands on disk. A
mismatched MD5 raises `RuntimeError`; the file is _not_ cached.

### 5.5.5 Tests as documentation

The pytest suite (next section) doubles as executable
documentation: each test asserts a behavioural contract that the
rest of the codebase can assume.

## 5.6 The pytest suite

| File                   | Tests | Asserts                                                                                                              |
| ---------------------- | ----- | -------------------------------------------------------------------------------------------------------------------- |
| `test_model.py`        | 6     | shapes, [0,1] outputs, FP16 size budget, gradient finiteness, non-divisible input handling                           |
| `test_physics_loss.py` | 4     | round-trip identity, gradient finiteness, t-supervision toggles, perfect-signal loss values                          |
| `test_ssim_tv.py`      | 4     | SSIM identity == 1, SSIM decreases with noise, TV zero on constant, TV positive on random                            |
| `test_data.py`         | 4     | snow synthesiser is deterministic, MSRB synth-fallback works, paired files load, missing-clean raises                |
| `test_export.py`       | 4     | ONNX parity (CPU), ONNX dynamic-shape inference, PyTorch latency benchmark (gpu-marked), TRT round-trip (trt-marked) |

Auto-skip rules in [`tests/conftest.py`](../../tests/conftest.py):

- `gpu`-marked tests skip if `torch.cuda.is_available()` is False
  or `AQUACLR_DISABLE_GPU=1` is set.
- `trt`-marked tests skip if `tensorrt` cannot be imported.

## 5.7 Performance engineering at the code level

| Lever                            | Code site                                | Effect                                    |
| -------------------------------- | ---------------------------------------- | ----------------------------------------- |
| Channels-last memory format      | `LEGIONDeSnowNet.__init__`               | ~10–20 % speedup on Ampere convs          |
| BF16 mixed precision             | trainer `precision="bf16-mixed"`         | 2× memory reduction, no GradScaler needed |
| `torch.compile(reduce-overhead)` | `LitModule.setup()`                      | Kernel fusion, ~10 % step speedup         |
| Persistent DataLoader workers    | `DataModule._loader`                     | Avoids worker spin-up cost each epoch     |
| Pinned host memory               | `pin_memory=True`                        | Faster H2D copies on CUDA                 |
| Gradient accumulation            | `accumulate_grad_batches=2` in train cfg | Effective batch ≥ 32 in 4 GB              |
| Gradient clipping                | `gradient_clip_val=1.0`                  | Stabilises near `t → 0`                   |

None of these is novel; the contribution is that they are all
applied **consistently** so the model trains in ≤ 12 hours on a
single RTX 3050 to a ≥ 25 dB PSNR ceiling.

## 5.8 Continuous integration

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)

Three jobs:

1. **lint** — Ubuntu, Python 3.11, runs `ruff check`,
   `ruff format --check`, and `mypy`. Required for merge.
2. **test** — Ubuntu × {Python 3.10, 3.11, 3.12}, runs the CPU
   pytest suite. Coverage uploaded as artifact.
3. **onnx-smoke** — Ubuntu, runs
   `scripts/export_onnx.py --smoke` (tiny model, 64×64 input,
   CPU-only). Verifies the export path even on GPU-less runners.

GPU and TRT tests are auto-skipped on the matrix runners and run
only on tagged self-hosted runners (out of scope for this
dissertation).

## 5.9 Walking a single line of code from CLI to prediction

To make the abstraction concrete, here's what happens when a user
runs `python scripts/train.py data=msrb`:

1. **Hydra resolves** the config tree, picking
   `configs/default.yaml` and overriding `data` to
   `configs/data/msrb.yaml`. The merged `cfg` is a `DictConfig`.
2. **`seed_everything`** is called with `cfg.seed = 1337` and
   `cfg.deterministic = True`.
3. **`hydra.utils.instantiate(cfg.data)`** constructs an
   `MSRBDataModule` from the YAML's `_target_` plus its kwargs.
4. **`hydra.utils.instantiate(cfg.model)`** constructs a
   `LEGIONDeSnowNet`.
5. **`PhysicsInformedLoss`** is instantiated from
   `cfg.train.loss`.
6. The **`LEGIONDeSnowLitModule`** is constructed with
   `(net, loss, optimizer_cfg, scheduler_cfg, compile_cfg)`.
7. **Callbacks** are appended: `EarlyStopping`,
   `ModelCheckpoint`, `EMAWeightCallback`, `VRAMMonitor`,
   `SampleImageLogger`, `LearningRateMonitor`, `ModelSummary`.
8. **Loggers** (TensorBoard always; W&B if enabled) are wired.
9. The `lightning.Trainer` is constructed with `precision`,
   `accumulate_grad_batches`, `gradient_clip_val`, etc.
10. **`trainer.fit(lit, datamodule=datamodule)`** runs.
11. Each batch goes through the sequence in §4.5.1.
12. After every validation epoch, `EMAWeightCallback` swaps in the
    EMA weights, validation runs, then swaps them back.
13. After `cfg.train.max_epochs`, the **best** checkpoint by
    `val/psnr` is exposed at
    `outputs/<run>/ckpts/legion-desnow-XXX-YY.YY.ckpt`.

This is the full execution trace. Everything else in the
documentation either explains _why_ a step is there (Ch. 1–4) or
_how_ to instrument it (Ch. 6–9).

---

## Key takeaways

- The codebase is a strict-typed, ruff-formatted, src-layout
  Python package with Hydra configs, Lightning training, and
  pre-commit hooks.
- Every concept from Chapters 3–4 has a single canonical home in
  the source tree — the §5.2 module walk is the index.
- Reproducibility is built in via seeded RNGs, locked
  dependencies, MD5-pinned datasets, and an exhaustive pytest
  suite.
- Performance is engineered at the code level (channels-last,
  BF16, `torch.compile`, persistent workers, pinned memory) so
  the model trains in ≤ 12 hours on a single RTX 3050.
- The full execution trace from `python scripts/train.py` to a
  saved checkpoint is 13 deterministic steps long.

## Cross-references

- Forward to [Chapter 6 — Datasets](06_datasets.md)
- Code reference appendix: [Appendix C](C_code_reference.md)
- Reproducibility checklist: [Appendix D](D_reproducibility.md)
