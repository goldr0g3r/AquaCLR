# AquaCLR — Project LEGION M1

> Physics-informed CNN for marine-snow removal in subsea video, targeting RTX 3050 / TensorRT FP16.

[![CI](https://github.com/goldr0g3r/aquaclr/actions/workflows/ci.yml/badge.svg)](https://github.com/goldr0g3r/aquaclr/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typed: mypy --strict](https://img.shields.io/badge/typed-mypy%20strict-2a6db4.svg)](https://mypy.readthedocs.io/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](.pre-commit-config.yaml)

AquaCLR (`aquaclr`) is the Milestone-1 deliverable of **Project LEGION**:
a real-time, physics-informed image-restoration front-end for subsea
ROV / AUV perception. It removes "marine snow" — the floating organic
particles that occlude underwater cameras — while preserving the
geometry that downstream SLAM and detection modules depend on.

The network learns only the **physical parameters** of the
Jaffe–McGlamery scattering model (`t(x)`, `B`) and inverts them
analytically to recover the clean image `J`. This constrains the
network to physically plausible solutions, which is what gives it the
out-of-distribution robustness a vanilla image-to-image translator of
the same size cannot match.

> **Automotive SiL parallel.** AquaCLR plays the role of a sensor
> preprocessing block (think "camera de-rain" / "lidar declutter") in
> an ADAS perception stack. The Jaffe–McGlamery model maps 1:1 onto
> the Koschmieder atmospheric scattering model used to render rain and
> fog in DriveSim-class simulators. Replace MSRB with rain-augmented
> KITTI and the same architecture denoises automotive frames before
> SLAM.

---

## Table of contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Loss design](#loss-design)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Training](#training)
- [Datasets](#datasets)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [Development](#development)
- [License](#license)
- [Citation](#citation)

---

## Highlights

| Property                | Value / target                                                            |
| ----------------------- | ------------------------------------------------------------------------- |
| Task                    | Single-frame marine-snow removal (image restoration)                      |
| Image-formation model   | Simplified Jaffe–McGlamery `I = J·t + B·(1 − t)`                          |
| Backbone                | MobileNetV3-Small (ImageNet pretrained, ~1.5 M params)                    |
| Decoder                 | Lightweight UNet built from depthwise-separable convs (DSC) + ReLU6       |
| Heads                   | Per-pixel transmission `t(x)` + global 3-vector backscatter `B`           |
| Recovery                | **Closed-form** Jaffe–McGlamery inversion (no extra learnable params)     |
| Parameter budget        | ~4–6 M params → ~9–12 MB at FP16 (well under the 50 MB M1 ceiling)        |
| Target hardware         | RTX 3050 4 GB (Ampere) / Jetson Orin-class via TensorRT FP16              |
| Mixed precision         | `bf16-mixed` (Ampere-native; safer than FP16 for squared-term losses)     |
| Reproducibility         | Deterministic seeding, MD5-verified dataset downloads, strict mypy        |
| Deployment              | ONNX export → TensorRT engine; ROS 2 Humble node planned for M1.5         |

## Architecture

```
   I ∈ [0,1]^(B,3,H,W)
         │
         ▼
   ImageNet normalize ──┐
         │              │
   MobileNetV3-Small    │   (4 pyramid taps: /4, /8, /16, /32)
         │              │
         ▼              │
   UNet decoder (DSC) ──┘
   ├─ feat_full  (stride /2 → /1)
   └─ feat_deep  (stride /32, global pool)
         │              │
         ▼              ▼
   Transmission     Backscatter
     head t(x)        head B
   (B,1,H,W)         (B,3)
         │              │
         └──────┬───────┘
                ▼
       J = (I − B·(1 − t)) / clamp(t, ε)
       ──────────── analytic Jaffe–McGlamery inversion ────────────
                ▼
       J ∈ [0,1]^(B,3,H,W)   (LEGIONOutputs.j, .t, .b)
```

See `src/aquaclr/models/model.py` for the canonical
`LEGIONDeSnowNet` definition and `src/aquaclr/utils/physics.py` for
the forward + inverse Jaffe–McGlamery operators.

## Loss design

The training objective combines four signals, plus an optional fifth
when ground-truth transmission is available (LSUI):

```
L = λ_rec · L_rec        # Charbonnier(J_pred, J_gt)
  + λ_phys · L_phys      # Charbonnier(I, J_gt·t + B·(1 − t))      forward consistency
  + λ_ssim · (1 − SSIM)  # protect SLAM-grade structure
  + λ_tv  · TV(t)        # piecewise-smooth transmission
  + λ_t   · ||t − t_gt|| # only when t_gt is provided (LSUI)
```

`L_phys` anchors `(t, B)` to physically meaningful values rather than
arbitrary factorisations that happen to make `J` come out right.
SSIM preserves features (corals, rocks) that downstream SLAM uses as
keypoints. Defaults are tuned in
`configs/train/rtx3050_bf16.yaml`.

## Installation

AquaCLR uses [`uv`](https://github.com/astral-sh/uv) for dependency
management. With `uv` installed:

```bash
git clone https://github.com/goldr0g3r/aquaclr.git
cd aquaclr
uv sync --extra dev          # core + dev tools (ruff, mypy, pytest, pre-commit)
uv run pre-commit install    # enable pre-commit on every git commit
```

Pip-only install (editable, with dev extras):

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
# . .venv/bin/activate                              # Linux / macOS
pip install -e ".[dev]"
```

Optional extras:

| Extra  | What you get                                                              |
| ------ | ------------------------------------------------------------------------- |
| `dev`  | `ruff`, `mypy`, `pytest`, `pytest-cov`, `pre-commit`, type stubs          |
| `trt`  | `onnx`, `onnxsim`, `onnxruntime[-gpu]`; `tensorrt` + `pycuda` on Linux    |
| `ros2` | Python-side helpers for the ROS 2 node (rclpy / cv_bridge come from apt)  |

> Python 3.10–3.12 is supported. PyTorch is pinned to `>=2.5,<2.7`
> (CUDA 12.4 verified). Install a CUDA-matching wheel from the PyTorch
> index if `uv sync` resolves to a CPU-only build.

## Quick start

```python
import torch
from aquaclr.models import LEGIONDeSnowNet

model = LEGIONDeSnowNet(pretrained=True).eval().cuda()

i = torch.rand(1, 3, 384, 384, device="cuda")   # snowy frame in [0, 1]
with torch.no_grad():
    out = model(i)                              # LEGIONOutputs(j, t, b)

print("J:", out.j.shape, "t:", out.t.shape, "B:", out.b.shape)
print(f"params={model.num_parameters/1e6:.2f} M, "
      f"fp16_size={model.estimate_size_mb(dtype=torch.float16):.2f} MB")
```

ONNX-friendly export uses the flat-tuple wrapper:

```python
model.eval()
torch.onnx.export(
    model, (torch.rand(1, 3, 384, 384),),
    "legion_desnow.onnx",
    input_names=["I"], output_names=["J", "t", "B"],
    opset_version=17,
    dynamic_axes={"I": {0: "B", 2: "H", 3: "W"}},
)
# Hint: call model.forward_export(...) inside the export wrapper to
# avoid the dataclass return; see model.py.
```

## Training

Training is wired through Hydra. The default composition is
`model=legion_desnow_s · data=combined · train=rtx3050_bf16`:

```bash
# RTX 3050 4 GB profile (bf16-mixed, 256² crops, OneCycleLR)
uv run python scripts/train.py

# Override anything from the CLI
uv run python scripts/train.py data=msrb train.max_epochs=80 train.optimizer.lr=2e-4

# Multirun sweep
uv run python scripts/train.py -m train.optimizer.lr=1e-4,3e-4,1e-3

# Disable wandb for an offline run
WANDB_DISABLED=1 uv run python scripts/train.py
```

ONNX export and a CPU/GPU benchmark live alongside the trainer:

```bash
uv run python scripts/export_onnx.py --ckpt outputs/<run>/ckpts/last.ckpt --out legion_desnow.onnx
uv run python scripts/evaluate.py --ckpt outputs/<run>/ckpts/last.ckpt
```

Useful environment variables:

| Variable                | Effect                                                          |
| ----------------------- | --------------------------------------------------------------- |
| `AQUACLR_DATA_ROOT`     | Override the data root (defaults to `./data`)                   |
| `AQUACLR_OUTPUT_ROOT`   | Override the output root (defaults to `./outputs`)              |
| `AQUACLR_DISABLE_GPU=1` | Force CPU; used by CI                                           |
| `AQUACLR_WANDB`         | `false` → disable wandb logging (TensorBoard still on)          |

Config knobs you'll touch most often live in
`configs/train/rtx3050_bf16.yaml` (precision, optimiser, loss
weights, EMA, sample logging) and `configs/model/legion_desnow_s.yaml`
(decoder widths, channels-last, transmission `eps`).

## Datasets

| Dataset | Role                                                  | Config                       |
| ------- | ----------------------------------------------------- | ---------------------------- |
| MSRB    | Marine Snow Removal Benchmark (Sato et al., APSIPA'23) — paired (I, J) | `configs/data/msrb.yaml`     |
| LSUI    | Large-Scale Underwater Image (Peng et al., 2021) — adds rare ground-truth transmission `t_gt` for direct supervision | `configs/data/lsui.yaml`     |
| Combined | Mix sampler (default 70 % MSRB, 30 % LSUI)            | `configs/data/combined.yaml` |

Datasets must be placed under `${AQUACLR_DATA_ROOT}` (default `./data`)
following the directory layout each DataModule expects. First-run
download is gated behind `download: true` in the relevant config and
uses MD5-verified streaming (`src/aquaclr/data/download.py`).
A procedural fallback synthesiser (`synthesize_marine_snow`) is
provided in `src/aquaclr/data/snow_synthesis.py` for environments
where the official MSRB pairs are unavailable, but headline numbers
should always be reported on the official MSRB pairs.

## Repository layout

```
.
├── configs/                       # Hydra configs
│   ├── default.yaml               # top-level composition
│   ├── data/                      # dataset configs (msrb, lsui, combined)
│   ├── model/legion_desnow_s.yaml
│   └── train/rtx3050_bf16.yaml
├── scripts/                       # Hydra-driven CLIs (train, export, eval)
│   ├── train.py
│   ├── export_onnx.py
│   └── evaluate.py
├── src/aquaclr/
│   ├── data/                      # MSRB / LSUI / combined DataModules + MD5 download
│   ├── losses/                    # PhysicsInformedLoss, SSIM, TV
│   ├── models/
│   │   ├── backbones/mobilenet_v3.py
│   │   ├── decoders/unet_dsc.py
│   │   ├── heads/{transmission,backscatter}.py
│   │   └── model.py               # LEGIONDeSnowNet (canonical entry point)
│   ├── training/                  # Lightning module + EMA / VRAM / sample-logger callbacks
│   ├── inference/                 # ONNX export, TensorRT builder/runner, benchmark
│   ├── ros2/                      # ROS 2 Humble node skeleton (M1.5)
│   └── utils/
│       ├── physics.py             # Jaffe–McGlamery forward + inverse
│       └── seed.py
├── tests/                         # pytest (auto-skips GPU/TRT-marked tests)
├── data/.gitkeep                  # local dataset root (gitignored)
├── .github/workflows/ci.yml       # lint · type-check · pytest · ONNX smoke
├── .pre-commit-config.yaml        # ruff, mypy, codespell, hygiene
└── pyproject.toml                 # build, deps, ruff, mypy, pytest, coverage
```

## Roadmap

- **M1 (current — Alpha).** LEGION-DeSnow-S reference implementation,
  physics-informed loss, MSRB + LSUI training pipeline, ONNX export
  smoke test in CI. *Status: model + losses + configs complete;
  DataModules + Lightning training loop + entry-point scripts under
  active implementation (see `pyproject.toml` `[project.scripts]`).*
- **M1.5.** TensorRT FP16 engine builder, latency benchmark on RTX 3050
  / Jetson Orin Nano, ROS 2 Humble node (`/camera/raw` → `/camera/desnowed`).
- **M2.** Temporal coherence loss across consecutive ROV frames,
  INT8 quantisation-aware training, sea-trial dataset capture.

Issues and milestones live on the [project board](https://github.com/goldr0g3r/aquaclr/issues).

## Development

```bash
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run mypy                    # strict type-check (src/aquaclr)
uv run pytest -q               # unit tests (auto-skip GPU/TRT if absent)
uv run pre-commit run -a       # all hooks against the whole tree
```

Tests are split with pytest markers — `gpu` and `trt` are auto-skipped
when the relevant hardware/runtime is unavailable, and `slow` carries
tests longer than 5 s. CI runs lint + mypy on every push, then the
pytest matrix on Python 3.10 / 3.11 / 3.12 plus an ONNX export smoke
job.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full development
workflow, branch conventions, and how to run the pre-commit suite
locally.

## License

Apache License 2.0 © Project LEGION. See [`LICENSE`](LICENSE).

## Citation

If you use AquaCLR in academic work, please cite the underlying
benchmarks alongside this repository:

```bibtex
@software{aquaclr2026,
  title  = {AquaCLR: Physics-Informed Marine-Snow Removal for Subsea Video},
  author = {Project LEGION},
  year   = {2026},
  url    = {https://github.com/goldr0g3r/aquaclr},
  note   = {Milestone 1, v0.1.0}
}

@inproceedings{sato2023marine,
  title     = {A Benchmark for Marine Snow Removal in Underwater Images},
  author    = {Sato, et al.},
  booktitle = {APSIPA ASC},
  year      = {2023}
}

@article{peng2023lsui,
  title   = {U-shape Transformer for Underwater Image Enhancement},
  author  = {Peng, et al.},
  journal = {IEEE TIP},
  year    = {2023}
}
```
