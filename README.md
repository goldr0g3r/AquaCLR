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

| Symbol | Meaning |
| --- | --- |
| `I(x)` | Observed (snowy) frame |
| `J(x)` | Clean scene radiance (what we want) |
| `t(x) ∈ [0, 1]` | Per-pixel medium transmission |
| `B ∈ [0, 1]³` | Global ambient backscatter |

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

| Block | Role | Params |
| --- | --- | --- |
| MobileNetV3-Small encoder (ImageNet-pretrained) | Multi-scale features at /4, /8, /16, /32 | ~1.5 M |
| UNet decoder w/ depthwise-separable convs | Aggregate features back to /2 | ~2.5 M |
| Transmission head (1×1 conv → sigmoid) | `t(x)` | < 0.1 K |
| Backscatter head (GAP → MLP → sigmoid) | `B` (3-vector) | ~3 K |
| **Total** | — | **~4–6 M (≤ 24 MB FP32, ≤ 12 MB FP16)** |

## Project layout

```
AquaCLR/
├─ pyproject.toml           # uv-managed, ruff + mypy + pytest config
├─ .pre-commit-config.yaml  # ruff-format, ruff-check, mypy, codespell
├─ .github/workflows/ci.yml # lint + tests + ONNX-export smoke
├─ configs/                 # Hydra configs (composable model/data/train)
├─ src/aquaclr/             # package source
│  ├─ models/               # LEGIONDeSnowNet, encoder, decoder, heads
│  ├─ losses/               # physics_loss.py, ssim.py, tv.py
│  ├─ data/                 # MSRB + LSUI datasets, transforms, snow synth
│  ├─ training/             # Lightning module + callbacks
│  ├─ inference/            # ONNX export + TensorRT runner + benchmark
│  ├─ ros2/                 # ROS2 Humble node skeleton
│  └─ utils/                # seed, physics, logging
├─ scripts/                 # train, evaluate, export_onnx
├─ tests/                   # pytest suite (CPU + gpu/trt-marked)
└─ docker/Dockerfile.trt    # NVIDIA CUDA + TRT image for deployment
```

## Datasets

We deeply analysed the underwater imaging dataset landscape and
recommend a **two-source** training mix:

| Dataset | Pairs | What we use it for | Why this one |
| --- | --- | --- | --- |
| **[MSRB](https://github.com/ychtanaka/marine-snow)** (Sato et al., APSIPA 2023) | 2,300 train / 400 test, 384×384 | **Primary** training signal | Purpose-built for marine-snow removal. Two tasks (small particles ≤6 px; mixed sizes ≤32 px). The only public dataset that gives perfect (snowy I, clean J) pairs for our exact problem. |
| **[LSUI](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html)** (Peng et al., 2021) | 4,279 pairs + GT transmission maps | Auxiliary; enables direct supervision on `t(x)` | LSUI is one of the very few public underwater datasets that ships **ground-truth medium transmission maps**. This is gold for a physics-informed model. |
| **[UIEB](https://li-chongyi.github.io/proj_benchmark.html)** (Li et al., 2019) | 890 pairs + 60 challenge | **Held-out** real-world evaluation only | Standard underwater enhancement benchmark; the 60-image *Challenge* split is unpaired and tests real-world generalisation. |

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

Place MSRB under `data/msrb/{train,test}/{noisy,clean}/` and LSUI
under `data/lsui/{input,GT,transmission}/`. The dataloaders raise a
clear error pointing here if the layout is wrong.

> **No dataset yet?** The MSRB loader has a built-in synthetic-snow
> fallback (`synthesize_if_missing=True`) so you can smoke-test the
> training loop with only the clean half of the dataset.

### 3. Train

```bash
uv run python scripts/train.py
```

This composes `configs/default.yaml` (model=`legion_desnow_s`,
data=`combined`, train=`rtx3050_bf16`).

Override anything from the CLI:

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

| Knob | Default | Where |
| --- | --- | --- |
| Mixed precision | `bf16-mixed` | `configs/train/rtx3050_bf16.yaml` |
| Optimiser | AdamW (`lr=3e-4`, `wd=1e-4`) | same |
| Schedule | OneCycleLR (cosine) | same |
| Batch size | 16 (MSRB) / 8 (LSUI) | `configs/data/*.yaml` |
| EMA decay | 0.9995 | same |
| `torch.compile` | `reduce-overhead` (auto-disabled for export) | same |
| Channels-last | on | `configs/model/legion_desnow_s.yaml` |
| Backbone freeze | first 2 epochs | same |
| Loss weights | `λ_rec=1, λ_phys=0.5, λ_ssim=0.5, λ_tv=1e-2, λ_t=0.5` | same |

Logs go to TensorBoard by default; W&B if `WANDB_API_KEY` is set.

## Evaluation

```bash
uv run python scripts/evaluate.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --data-root data/msrb \
  --split test --task 1
```

Reports PSNR + SSIM. For UIEB no-reference scores (UIQM, UCIQE) we
include hooks but expect the user to install
[`pyiqa`](https://github.com/chaofengc/IQA-PyTorch) separately.

## Export to TensorRT

```bash
uv run python scripts/export_onnx.py \
  --ckpt outputs/<run>/ckpts/best.ckpt \
  --out outputs/legion_desnow.onnx \
  --height 720 --width 1280 \
  --build-trt --benchmark
```

What this does:
1. Loads the checkpoint.
2. Exports to ONNX (opset 17, dynamic batch + `H` + `W`).
3. Verifies ONNX vs PyTorch parity (`atol=1e-3`).
4. Builds a TensorRT FP16 engine with a 256–720 p shape profile.
5. Runs a 200-iter latency benchmark and prints p50/p95/FPS/VRAM.

> **Tip**: TensorRT is Linux-only; on Windows, stop after the ONNX
> export and use `onnxruntime-gpu` for inference.

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

| Marine context | Automotive analogue |
| --- | --- |
| Marine snow | Lidar rain clutter / camera rain droplets |
| Jaffe-McGlamery `I = J·t + B(1−t)` | Koschmieder atmospheric scattering |
| Transmission map `t(x)` | Per-ray optical depth / lidar return-intensity confidence |
| Backscatter `B` | Airlight constant / radar noise floor |
| TensorRT engine | DRIVE Orin engine cache |
| ROS2 Humble | ROS2 / Apex.AI middleware in autonomy stacks |
| LEGION-DeSnow | ADAS sensor restoration block (de-rain, de-fog) before SLAM |

## Cite & references

- Sato, Y. et al. *Marine Snow Removal Benchmarking Dataset*. arXiv:2103.14249. APSIPA 2023.
- Peng, L. et al. *U-shape Transformer for Underwater Image Enhancement*. 2021.
- Li, C. et al. *An Underwater Image Enhancement Benchmark Dataset and Beyond*. IEEE TIP 2019.
- Howard, A. et al. *Searching for MobileNetV3*. ICCV 2019.
- Wang, Z. et al. *Image quality assessment: from error visibility to structural similarity (SSIM)*. IEEE TIP 2004.
- McGlamery, B. *A computer model for underwater camera systems*. SPIE 1980.
