# AquaCLR — LEGION Subsea Perception Front-End (M1)

> Real-time **physics-informed** marine-snow removal for underwater video.
> Targets an NVIDIA RTX 3050 (4 GB VRAM) at **< 15 ms / 720 p frame** with **TensorRT FP16**.

[![CI](https://img.shields.io/badge/ci-github_actions-blue)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

---

## Table of contents

- [Why](#why) — what problem this solves and why physics-informed
- [Architecture overview](#architecture-overview)
- [Project layout](#project-layout)
- [Datasets](#datasets)
- [Quickstart](#quickstart)
- [Training](#training)
- [Evaluation](#evaluation)
- [Export to TensorRT](#export-to-tensorrt)
- [ROS2 deployment](#ros2-deployment)
- [Automotive SiL parallels](#automotive-sil-parallels)
- [Cite & references](#cite--references)

---

## Why

Underwater imagery suffers from **marine snow** — bright particulate
streaks created by organic debris drifting between the scene and the
camera. These artefacts (a) wreck downstream SLAM feature-matching
and (b) are extremely hard to remove with vanilla image-to-image
networks because they share statistics with legitimate scene
highlights.

We follow the simplified Jaffe-McGlamery image-formation model:

$$
I(x) \;=\; J(x)\,t(x) \;+\; B \,(1 - t(x))
$$

| Symbol          | Meaning                             |
| --------------- | ----------------------------------- |
| `I(x)`          | Observed (snowy) frame              |
| `J(x)`          | Clean scene radiance (what we want) |
| `t(x) ∈ [0, 1]` | Per-pixel medium transmission       |
| `B ∈ [0, 1]³`   | Global ambient backscatter          |

Instead of regressing `J` directly, the network predicts `(t, B)` and
inverts the equation analytically. This **physics-informed** factoring
both:

- shrinks the hypothesis space (the network can't make up colours
  that aren't compatible with underwater optics), and
- gives downstream SLAM a free **per-pixel confidence map** in the
  form of `t(x)`.

## Architecture overview

```
   I (B,3,H,W)  ────► MobileNetV3-Small encoder  ─┐
                                                   ├──► UNet-DSC decoder ──► t-head ──► t (B,1,H,W)
                                                   │                       └► B-head ──► B (B,3)
                                                   │
                                                   └─► Jaffe-McGlamery inversion ──► J (B,3,H,W)
```

| Block                                           | Role                                     | Params                                  |
| ----------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| MobileNetV3-Small encoder (ImageNet-pretrained) | Multi-scale features at /4, /8, /16, /32 | ~1.5 M                                  |
| UNet decoder w/ depthwise-separable convs       | Aggregate features back to /2            | ~2.5 M                                  |
| Transmission head (1×1 conv → sigmoid)          | `t(x)`                                   | < 0.1 K                                 |
| Backscatter head (GAP → MLP → sigmoid)          | `B` (3-vector)                           | ~3 K                                    |
| **Total**                                       | —                                        | **~4–6 M (≤ 24 MB FP32, ≤ 12 MB FP16)** |

## Project layout

```
AquaCLR/
├─ pyproject.toml           # uv-managed, ruff + mypy + pytest config
├─ .pre-commit-config.yaml  # ruff-format, ruff-check, mypy, codespell
├─ .github/workflows/ci.yml # lint + tests + ONNX-export smoke
├─ configs/                 # Hydra configs (composable model/data/train)
│  ├─ default.yaml
│  ├─ model/legion_desnow_s.yaml
│  ├─ data/{msrb,lsui,combined,combined_a3000}.yaml
│  └─ train/{rtx3050_bf16,rtx_a3000_bf16}.yaml
├─ src/aquaclr/             # package source
│  ├─ models/               # LEGIONDeSnowNet, encoder, decoder, heads
│  ├─ losses/               # physics_loss.py, ssim.py, tv.py
│  ├─ data/                 # MSRB + LSUI datasets, transforms, snow synth
│  ├─ training/             # Lightning module + callbacks
│  ├─ inference/            # ONNX export + TensorRT runner + benchmark
│  ├─ ros2/                 # ROS2 Humble node skeleton
│  └─ utils/                # seed, physics, logging
├─ scripts/                 # thin CLI entry points
│  ├─ train.py              # Hydra entry point
│  ├─ evaluate.py           # PSNR/SSIM + optional UIQM/UCIQE (--no-ref)
│  ├─ evaluate_slam_features.py  # ORB/SIFT downstream keypoint benchmark
│  ├─ infer_camera.py       # real-time webcam / video inference (OpenCV)
│  ├─ export_onnx.py        # ONNX (+ optional TRT) export
│  └─ download_data.py      # dataset layout verifier
├─ tests/                   # pytest suite (CPU + gpu/trt-marked)
└─ docker/Dockerfile.trt    # NVIDIA CUDA + TRT image for deployment
```

## Datasets

We deeply analysed the underwater imaging dataset landscape and
recommend a **two-source** training mix:

| Dataset                                                                                       | Pairs                              | What we use it for                              | Why this one                                                                                                                                                                             |
| --------------------------------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[MSRB](https://github.com/ychtanaka/marine-snow)** (Sato et al., APSIPA 2023)               | 2,300 train / 400 test, 384×384    | **Primary** training signal                     | Purpose-built for marine-snow removal. Two tasks (small particles ≤6 px; mixed sizes ≤32 px). The only public dataset that gives perfect (snowy I, clean J) pairs for our exact problem. |
| **[LSUI](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html)** (Peng et al., 2021) | 4,279 pairs + GT transmission maps | Auxiliary; enables direct supervision on `t(x)` | LSUI is one of the very few public underwater datasets that ships **ground-truth medium transmission maps**. This is gold for a physics-informed model.                                  |
| **[UIEB](https://li-chongyi.github.io/proj_benchmark.html)** (Li et al., 2019)                | 890 pairs + 60 challenge           | **Held-out** real-world evaluation only         | Standard underwater enhancement benchmark; the 60-image _Challenge_ split is unpaired and tests real-world generalisation.                                                               |

> **TL;DR**: train on MSRB + LSUI, evaluate on UIEB-Challenge.

See [`MODEL_CARD.md`](MODEL_CARD.md) for the latest download URLs and
unpacking instructions.

## Quickstart

### 1. Install

We use [`uv`](https://github.com/astral-sh/uv) (fast pip + venv
replacement). If you don't have it: `pip install uv`.

```bash
uv venv .venv
uv sync --extra dev               # core + dev tools
# Optional — only on Linux with NVIDIA stack:
uv sync --extra dev --extra trt   # adds onnx, onnxruntime-gpu, tensorrt, pycuda
```

### 2. Get the data

Place MSRB under the canonical upstream layout from
[`ychtanaka/marine-snow`](https://github.com/ychtanaka/marine-snow):

```
data/msrb/
├─ training/
│  ├─ original/    # clean reference J
│  ├─ MSR_Task1/   # snowy I — small particles
│  └─ MSR_Task2/   # snowy I — mixed sizes
└─ test/
   ├─ original/
   ├─ MSR_Task1/
   └─ MSR_Task2/
```

The `task` knob in `configs/data/msrb.yaml` (`1` or `2`) picks which
snowy variant pairs against `original/`.

Place LSUI under `data/lsui/{input,GT}/` (and optionally
`data/lsui/transmission/` if you have the GT transmission maps —
they're used as a soft supervision signal when present). The
dataloaders raise a clear error pointing here if the layout is wrong.
The legacy flat layout (`data/msrb/{train,test}/{noisy,clean}/`) is
still auto-detected for older user setups.

> **No dataset yet?** The MSRB loader has a built-in synthetic-snow
> fallback (`synthesize_if_missing=True`) so you can smoke-test the
> training loop with only the clean half of the dataset.

### 3. Train

```bash
# Default profile — MSRB only, RTX 3050, 256 px crops:
uv run python scripts/train.py

# RTX A3000 profile — 384 px crops, larger batches, fills 6 GB VRAM:
uv run python scripts/train.py train=rtx_a3000_bf16 data=combined_a3000
```

This composes `configs/default.yaml` (model=`legion_desnow_s`,
data=`combined`, train=`rtx3050_bf16`).

Resume an interrupted run (restores optimizer state + epoch counter):

```bash
uv run python scripts/train.py train=rtx_a3000_bf16 \
  run_name=<timestamp> \
  resume_from=outputs/<timestamp>/ckpts/last.ckpt
```

Override individual hyperparameters:

```bash
uv run python scripts/train.py \
  data=msrb \
  train.max_epochs=30 \
  train.optimizer.lr=1e-4 \
  train.precision=bf16-mixed
```

Multirun sweep:

```bash
uv run python scripts/train.py -m \
  train.optimizer.lr=1e-4,3e-4,1e-3 \
  train.loss.lambda_phys=0.25,0.5,1.0
```

## Training

Two hardware profiles are provided:

| Profile            | Config                                     | GPU  | Image size | MSRB batch | LSUI batch | `torch.compile` |
| ------------------ | ------------------------------------------ | ---- | ---------- | ---------- | ---------- | --------------- |
| RTX 3050 (default) | `train=rtx3050_bf16`                       | 4 GB | 256 px     | 8          | 4          | enabled (Linux) |
| RTX A3000          | `train=rtx_a3000_bf16 data=combined_a3000` | 6 GB | 384 px     | 16         | 8          | disabled¹       |

¹ `torch.compile` requires Triton which is Linux-only; disabled automatically on Windows. Re-enable in a Linux container.

| Knob            | Default                                               | Where                                |
| --------------- | ----------------------------------------------------- | ------------------------------------ |
| Mixed precision | `bf16-mixed`                                          | `configs/train/*.yaml`               |
| Optimiser       | AdamW (`lr=3e-4`, `wd=1e-4`)                          | same                                 |
| Schedule        | OneCycleLR (cosine)                                   | same                                 |
| EMA decay       | 0.9995                                                | same                                 |
| Channels-last   | on                                                    | `configs/model/legion_desnow_s.yaml` |
| Backbone freeze | first 2 epochs                                        | same                                 |
| Loss weights    | `λ_rec=1, λ_phys=0.5, λ_ssim=0.5, λ_tv=1e-2, λ_t=0.5` | same                                 |

Checkpoints are saved to `outputs/<timestamp>/ckpts/`. The three best
`val/psnr` checkpoints and `last.ckpt` are always kept. Logs go to
TensorBoard and W&B (W&B credentials read from `~/_netrc` or
`WANDB_API_KEY`).

## Evaluation

The evaluation script supports three datasets via `--dataset`:

| Dataset | Flag | Data root | Notes |
|---------|------|-----------|-------|
| MSRB | `--dataset msrb` (default) | `data/msrb` | Requires `--split` and `--task` |
| UIEB | `--dataset uieb` | `data/uieb` | Uses raw-890/reference-890 pairs |
| LSUI | `--dataset lsui` | `data/lsui` | Uses input/GT pairs |

### Reference-based (PSNR + SSIM)

```bash
# MSRB (default)
uv run python scripts/evaluate.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --data-root data/msrb \
  --split test --task 1

# UIEB
uv run python scripts/evaluate.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --data-root data/uieb \
  --dataset uieb

# LSUI
uv run python scripts/evaluate.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --data-root data/lsui \
  --dataset lsui
```

### No-reference (NIQE + MUSIQ)

Add `--no-ref` to compute no-reference image quality metrics on the
enhanced output. Requires [`pyiqa`](https://github.com/chaofengc/IQA-PyTorch);
silently skipped if not installed. The script attempts UIQM/UCIQE first
and gracefully falls back to NIQE and MUSIQ if unavailable.

```bash
pip install pyiqa   # one-time
uv run python scripts/evaluate.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --data-root data/msrb \
  --split test --task 1 \
  --no-ref
```

### Downstream SLAM feature stability

Measures how LEGION-DeSnow affects keypoint yield, repeatability, and
descriptor matching quality — the metrics that directly predict SLAM
performance. Uses OpenCV ORB (always available) or SIFT (needs
`opencv-contrib-python`).

```bash
uv run python scripts/evaluate_slam_features.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --data-root data/msrb \
  --split test --task 1

# SIFT + larger feature budget:
uv run python scripts/evaluate_slam_features.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --data-root data/msrb \
  --detector sift --n-features 1500
```

Reports per-image and aggregate statistics for:

| Metric                         | What it signals                                 |
| ------------------------------ | ----------------------------------------------- |
| KP count (raw / enhanced / GT) | Does de-snowing reveal more features?           |
| Repeatability I→Ĵ              | Are scene points re-detected after enhancement? |
| Match inlier ratio I→Ĵ         | RANSAC-verified geometric consistency           |
| Match score                    | Descriptor confidence (lower = sharper matches) |

## Camera / live video inference

Run the model on a webcam or video file in real time. Displays a
side-by-side (Raw | Enhanced) window with per-frame latency overlay.
Supports both **PyTorch** (`--ckpt`) and **TensorRT** (`--engine`) backends.

```bash
# Webcam with PyTorch backend (device 0)
uv run python scripts/infer_camera.py \
  --ckpt outputs/<run>/ckpts/best.ckpt

# Webcam with TensorRT engine (fastest)
uv run python scripts/infer_camera.py \
  --engine outputs/legion_desnow.engine

# Video file
uv run python scripts/infer_camera.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --source path/to/footage.mp4

# Resize to 512 px shorter edge before inference (higher FPS on low-end GPUs)
uv run python scripts/infer_camera.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --resize 512

# Save side-by-side MP4 without opening a window
uv run python scripts/infer_camera.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --source footage.mp4 --save out.mp4 --no-display
```

Expected throughput on the RTX A3000 (BF16, native resolution 1080 p): ~20–30 FPS.
With `--resize 512`: ~60+ FPS. With TensorRT FP16: ~70–100+ FPS.

> For ROS2 integration (Humble / Jazzy) see `src/aquaclr/ros2/ros2_node.py` and
> [`docs/DEPLOYMENT_FEDORA.md`](docs/DEPLOYMENT_FEDORA.md).

## Export & Benchmarking

### ONNX export + TensorRT engine build

```bash
uv run python scripts/export_onnx.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --out outputs/legion_desnow.onnx \
  --height 720 --width 1280 \
  --build-trt
```

What this does:

1. Loads the checkpoint.
2. Exports to ONNX (opset 17, dynamic batch + `H` + `W`).
3. Simplifies the graph with `onnxsim` (if installed: `pip install onnx-simplifier`).
4. Verifies ONNX vs PyTorch parity (skip with `--no-verify` if tolerance is tight).
5. Builds a TensorRT FP16 engine with a 256–720 p dynamic shape profile.

### Latency benchmarking

Three benchmark backends are available:

| Flag | Backend | Requirements |
|------|---------|-------------|
| `--benchmark` | PyTorch FP16 | _(none — always available)_ |
| `--benchmark-onnx` | ONNX Runtime | `pip install onnxruntime-gpu` |
| `--benchmark-trt` | TensorRT engine | `pip install tensorrt pycuda` + `--build-trt` |

Run all three in one command:

```bash
uv run python scripts/export_onnx.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --height 720 --width 1280 \
  --build-trt \
  --benchmark --benchmark-onnx --benchmark-trt \
  --no-verify
```

Each benchmark runs 200 iterations (after 20 warmup) and reports:
- **p50 / p95 / mean** latency (ms)
- **FPS** (frames per second)
- **Peak VRAM** (PyTorch only)

Example results on RTX A3000 at 720×1280:

| Backend | p50 | FPS | Notes |
|---------|-----|-----|-------|
| PyTorch FP16 | ~22 ms | ~45 | Baseline |
| ONNX Runtime (CUDA) | ~18 ms | ~55 | Graph optimizations |
| TensorRT FP16 | ~10 ms | ~100 | Kernel fusion + FP16 |

> **Tip**: TensorRT engines are hardware-specific. Build the `.engine`
> on the deployment machine. On Windows without TRT, use ONNX Runtime
> (`onnxruntime-gpu`) as the deployment backend.

## ROS2 deployment

The skeleton subscribes to `/camera/image_raw` and publishes
`/camera/image_desnowed`. Topics, backend (`trt`/`torch`), engine
path, and checkpoint are all ROS parameters. The node is
API-compatible with both **ROS2 Humble** (Ubuntu 22.04) and
**ROS2 Jazzy** (Ubuntu 24.04, LTS until 2029).

```bash
# Inside a ROS2 container or sourced env (Humble or Jazzy):
ros2 run aquaclr legion_desnow_node \
  --ros-args \
    -p engine_path:=/work/legion_desnow.engine \
    -p input_topic:=/oak/rgb/image_raw \
    -p output_topic:=/legion/image_desnowed
```

> **Running on Fedora 44 with a Ubuntu 24.04 + ROS2 Jazzy
> distrobox / podman container?**
> See the full runbook in [`docs/DEPLOYMENT_FEDORA.md`](docs/DEPLOYMENT_FEDORA.md)
> — host setup, NVIDIA Container Toolkit (CDI), distrobox creation,
> ROS2 Jazzy install, GPU passthrough verification, and end-to-end
> smoke tests.

`M2` will publish the predicted transmission as a side-channel for
SLAM uncertainty weighting.

## Automotive SiL parallels

Every public class/function in this repo carries an "Automotive SiL
parallel" docstring describing the analogous component in an
ADAS / autonomous-driving stack. The headline mappings:

| Marine context                     | Automotive analogue                                         |
| ---------------------------------- | ----------------------------------------------------------- |
| Marine snow                        | Lidar rain clutter / camera rain droplets                   |
| Jaffe-McGlamery `I = J·t + B(1−t)` | Koschmieder atmospheric scattering                          |
| Transmission map `t(x)`            | Per-ray optical depth / lidar return-intensity confidence   |
| Backscatter `B`                    | Airlight constant / radar noise floor                       |
| TensorRT engine                    | DRIVE Orin engine cache                                     |
| ROS2 Humble                        | ROS2 / Apex.AI middleware in autonomy stacks                |
| LEGION-DeSnow                      | ADAS sensor restoration block (de-rain, de-fog) before SLAM |

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md) — intended use, training data, ethical considerations.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — long-form design rationale.
- [`docs/DEPLOYMENT_FEDORA.md`](docs/DEPLOYMENT_FEDORA.md) — Fedora 44 + Ubuntu 24.04 + ROS2 Jazzy runbook.
- [`DISSERTATION.md`](DISSERTATION.md) — **M.Tech dissertation master entry point** with full TOC; chapters live under [`docs/dissertation/`](docs/dissertation/).
- [`docs/README.md`](docs/README.md) — documentation index.

To build the dissertation as a PDF:

```bash
bash scripts/build_dissertation.sh
```

## Cite & references

- Sato, Y. et al. _Marine Snow Removal Benchmarking Dataset_. arXiv:2103.14249. APSIPA 2023.
- Peng, L. et al. _U-shape Transformer for Underwater Image Enhancement_. 2021.
- Li, C. et al. _An Underwater Image Enhancement Benchmark Dataset and Beyond_. IEEE TIP 2019.
- Howard, A. et al. _Searching for MobileNetV3_. ICCV 2019.
- Wang, Z. et al. _Image quality assessment: from error visibility to structural similarity (SSIM)_. IEEE TIP 2004.
- McGlamery, B. _A computer model for underwater camera systems_. SPIE 1980.

The complete bibliography is in [`docs/dissertation/12_references.md`](docs/dissertation/12_references.md).
