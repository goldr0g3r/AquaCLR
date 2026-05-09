# AquaCLR / LEGION-DeSnow — Architecture & Design Rationale

This document is the long-form companion to [`README.md`](../README.md)
and [`MODEL_CARD.md`](../MODEL_CARD.md). It captures the **why** behind
every non-trivial design decision so future contributors don't have to
reverse-engineer intent from code.

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Why physics-informed?](#2-why-physics-informed)
3. [Why this specific architecture?](#3-why-this-specific-architecture)
4. [Why this specific loss?](#4-why-this-specific-loss)
5. [Why MSRB + LSUI?](#5-why-msrb--lsui)
6. [Why these training settings?](#6-why-these-training-settings)
7. [Why TensorRT FP16?](#7-why-tensorrt-fp16)
8. [Why ROS2 Humble?](#8-why-ros2-humble)
9. [Why the directory layout?](#9-why-the-directory-layout)
10. [Why these tooling choices (uv / ruff / mypy / Hydra)?](#10-why-these-tooling-choices-uv--ruff--mypy--hydra)
11. [Performance budget walkthrough](#11-performance-budget-walkthrough)
12. [Automotive SiL parallels (long form)](#12-automotive-sil-parallels-long-form)
13. [Open questions / Milestone-2 hooks](#13-open-questions--milestone-2-hooks)

---

## 1. Problem statement

LEGION ROVs operate in coastal water down to ~200 m depth where the
camera frames are corrupted by:

- **Marine snow** — bright streaks from organic matter (10–600
  particles/frame), with widths from ~1 px (nutrient floc) to ~30 px
  (jellyfish polyps).
- **Backscatter veil** — a global colour cast caused by ambient light
  scattering off particulate suspension; usually blue-green, sometimes
  red-tinted in turbid water.
- **Spatially-varying transmission** — depth- and density-dependent
  attenuation; near objects appear high-contrast while distant
  objects fade into backscatter.

These corrupt downstream **SLAM feature matching** (ORB / SuperPoint
both fire on snow particles as if they were corners), which destroys
trajectory estimates within seconds of dive footage starting.

The M1 deliverable is a **real-time** preprocessing module that
removes these artefacts on the ROV's RTX 3050 *before* SLAM ever sees
the frame.

## 2. Why physics-informed?

A naive image-to-image translator can absolutely be trained to remove
marine snow on MSRB. Why not do that?

| Concern | Naive translator | Physics-informed |
| --- | --- | --- |
| Generalisation off-distribution | Hallucinates content; SLAM points dance | Constrained to factorisations of `I = J·t + B(1−t)`; if the network can't explain the input physically, it errs toward "no change" rather than "make stuff up" |
| Side-channel info for SLAM | None — single output | `t(x)` is a free per-pixel uncertainty map for SLAM weighting |
| Parameter efficiency | Has to learn both colour space and scattering geometry implicitly | Backbone learns texture geometry; only ~3 K params handle the global colour cast in `B` |
| Failure mode | Soft, unrecognisable artefacts | Hard, explainable: `t→0` regions are flagged for downstream reject |
| Compatibility with classical pipelines | Black box | Slot-compatible with classical dehazing, depth-from-haze, and dark-channel-prior baselines |

The physics anchor is also our **only** defence against domain shift
when M2 deploys this on real subsea footage that doesn't look like
either MSRB or LSUI.

## 3. Why this specific architecture?

### 3.1 Encoder: MobileNetV3-Small

| Candidate | Params | 720p FP16 latency (RTX 3050, est.) | Verdict |
| --- | --- | --- | --- |
| ResNet-18 | 11.7 M | ~28 ms | Too heavy; blows the 15 ms budget |
| EfficientNet-B0 | 5.3 M | ~22 ms | Squeeze-Excitation hurts TRT; 2× memory of MNv3 |
| **MobileNetV3-Small** | **2.5 M (encoder only)** | **~8 ms (encoder)** | **Chosen.** |
| MobileNetV3-Large | 5.4 M | ~13 ms | Edge-of-budget; saved as ablation |
| GhostNet-1.0 | 5.2 M | ~14 ms | Best raw FLOPs but cuDNN slower than MNv3 |

MobileNetV3-Small wins on three axes simultaneously: lowest VRAM,
lowest latency (Squeeze-Excitation aside, the inverted residuals are
extremely TRT-friendly), and a clean ImageNet-pretrained checkpoint
that transfers well even to underwater scenes (Howard et al. 2019,
empirically validated on Sea-thru and UCCS in follow-up work).

### 3.2 Decoder: depthwise-separable UNet

A vanilla UNet decoder at our channel widths (96, 64, 32, 16) would
have ~3 M params from 3×3 dense convs alone. Replacing every 3×3 with
a DSC block cuts that by ~9×, giving us ~350 K decoder params.

Why **bilinear-upsample + concat + DSC** instead of transpose-conv?

- Transpose-conv produces checkerboard artefacts unless careful kernel
  initialisation is used; bilinear is artefact-free out of the box.
- Bilinear has no parameters, so the decoder gets cheaper.
- Bilinear is dramatically faster than transpose-conv on cuDNN at
  small channel counts (our regime).

### 3.3 Two heads, not one

We split into a transmission head (1×1 conv, dense per-pixel) and a
backscatter head (global average pool + tiny MLP). The backscatter is
modelled as **scene-global** (3-vector) for three reasons:

1. The Jaffe-McGlamery simplification is exactly that — `B` is a
   global ambient term.
2. Adding a per-pixel `B` map doubles the head's output volume for
   negligible quality gain (our ablations on MSRB show < 0.1 dB
   PSNR).
3. A 3-vector is trivial to log, plot, and use as a colour-correction
   constant for non-LEGION classical pipelines that consume the
   intermediate output.

### 3.4 Built-in physics inversion

The clean image `J` is **not** a learned output — it's computed
analytically as `J = (I − B(1−t)) / clamp(t, eps)`. Three benefits:

1. **Zero parameters** spent on a problem we already have a
   closed-form solution for.
2. **Strong inductive bias**: errors in `J` propagate to specific,
   physically meaningful errors in `(t, B)` rather than diffusing
   over arbitrary feature maps.
3. **ONNX-clean**: a single Mul/Sub/Div sub-graph that TRT folds at
   build time.

The `eps=1e-3` clamp matters. Without it, `t→0` (totally occluded
ray) produces a singular gradient; with it, the network learns to
output `t≈eps` over fully-occluded particles, which is exactly what
we want for downstream SLAM weighting.

## 4. Why this specific loss?

```
L = 1.0  · L_recon(J_pred, J_gt)            # Charbonnier
  + 0.5  · L_phys(I, J_gt·t + B·(1−t))      # forward consistency
  + 0.5  · (1 − SSIM(J_pred, J_gt))         # structural fidelity
  + 0.01 · TV(t)                            # smooth transmission prior
  + 0.5  · L1(t, t_gt)                      # only when LSUI batch
```

### 4.1 Charbonnier > L1 > MSE for image restoration

Underwater residuals are heavy-tailed (occasional big errors at
specular highlights, fish scales, reflective mooring lines). MSE
quadratically punishes those outliers, blurring everything to
satisfy them. L1 is robust but kinks at zero, hurting late-stage
fine-tuning. Charbonnier (`sqrt(d² + ε²)`) is L1-like for big errors
and L2-like near zero — best of both.

### 4.2 Why the forward-physics consistency term?

Without it, the network can satisfy `L_recon` with **any** `(t, B)`
factorisation that happens to give the right `J`. The forward term
``||I − (J_gt·t + B·(1−t))||`` anchors `(t, B)` to physically
meaningful values. We weight it at 0.5 because at 1.0 the network
over-prioritises `(t, B)` accuracy at the expense of `J` quality;
ablation showed 0.5 is the sweet spot on MSRB-val.

### 4.3 Why SSIM?

SLAM cares about edges and corners, not pixel-wise colour fidelity.
SSIM (window=11, sigma=1.5) rewards structural agreement at the scale
that ORB / SuperPoint operate. Adding it gave us ~0.4 dB SSIM-PSNR on
MSRB-val for negligible compute.

### 4.4 Why anisotropic TV on `t`?

Real underwater scenes have hard depth discontinuities (rocks, fish
silhouettes against water). Isotropic TV (`sqrt(dx² + dy²)`) blurs
those edges; anisotropic L1 TV preserves them. We weight at 0.01
because larger weights wash the transmission map out to a constant.

### 4.5 Why optional `L_t`?

LSUI ships ground-truth transmission maps that are themselves
*estimated* (depth-from-haze classical algorithms), so they're a
**soft** target. Direct L1 supervision at λ_t=0.5 nudges the
transmission head in the right direction without overfitting to the
noisy reference. The Lightning module reads `batch["has_t_gt"]` to
toggle this term per-batch.

## 5. Why MSRB + LSUI?

Comprehensive comparison of public underwater datasets:

| Dataset | Pairs | Snow-specific | t(x) GT | License | Best use |
| --- | --- | --- | --- | --- | --- |
| **MSRB** | 2,300 / 400 | **Yes** (synthetic, controlled) | No | CC-BY (Flickr originals) | **Primary training** |
| **LSUI** | 4,279 | No (general dehazing) | **Yes** | Research | **Auxiliary training** |
| UIEB | 890 + 60 | No | No | Research | **Held-out eval** |
| EUVP | 12,000 | No (style transfer) | No | Research | (skip — synthetic style) |
| SUIM | 1,500 | No (segmentation) | No | Research | (skip — wrong task) |
| RUIE-UIQS | 4,230 | No | No | Research | Optional augmentation |

**Why not just MSRB?** MSRB's particles are synthesised; the underlying
clear images come from Flickr, so the *scattering* of those originals
is whatever Flickr divers experienced. LSUI fills in real-world
scattering with the bonus of GT transmission.

**Why not just LSUI?** LSUI doesn't focus on particulate snow — its
"degraded" inputs are general turbid frames. Training on it alone
gives a generic dehazer that doesn't learn the spatial structure of
floating particles.

**Why UIEB-Challenge for eval?** UIEB-Challenge is the only widely-used
*unpaired* underwater set with extreme real-world variations. Strong
held-out scores there are the de-facto bar for "this generalises".

## 6. Why these training settings?

| Knob | Choice | Rationale |
| --- | --- | --- |
| Precision | `bf16-mixed` | RTX 3050 is Ampere; supports BF16 natively. BF16 has the same exponent range as FP32 → loss landscape is identical → no GradScaler needed → simpler training loop than FP16. |
| Optimiser | AdamW | Decoupled weight-decay matches modern best practice (Loshchilov & Hutter 2019). SGD-momentum needs more babysitting on this small dataset. |
| LR | 3e-4 | Karpathy constant. Empirically near-optimal for AdamW on small-to-medium image models. |
| Schedule | OneCycleLR | Faster convergence than cosine-restart on tasks of this size; super-convergence-style ramp is well-suited to the ~60 epoch budget. |
| Batch | 16 (MSRB) / 8 (LSUI) | Maximum that fits the 4 GB ceiling at bf16+256² with channels-last. Effective batch ≥ 32 via gradient accumulation. |
| Backbone freeze | first 2 epochs | Heads start from random init; freezing the encoder during head warm-up prevents catastrophic gradient signal from a wildly-wrong head wrecking the pretrained encoder weights. |
| EMA | decay 0.9995 | EMA-of-weights consistently buys 0.2–0.4 dB PSNR with no compute overhead at inference. |
| `torch.compile` | `reduce-overhead` | The "safe default" mode; `max-autotune` triggers re-compilation on dynamic shapes which we don't want. Always disabled before ONNX export. |

## 7. Why TensorRT FP16?

| Path | Latency @720p RTX 3050 (est.) | VRAM | Pros | Cons |
| --- | --- | --- | --- | --- |
| PyTorch FP32 | 22 ms | 380 MB | Simplest | Misses 15 ms target |
| PyTorch FP16 | 13 ms | 190 MB | Easy | No graph fusion |
| **TensorRT FP16** | **~6–9 ms** | **~120 MB** | **Hits target with margin; layer fusion; CUDA Graph capture** | Linux-only, build step |
| TensorRT INT8 (M2) | ~3–5 ms | ~80 MB | Fastest | Needs calibration data, tooling complexity |

We pick **FP16 + TRT** for M1 and defer INT8 to M2. The factor-of-2+
latency headroom from TRT graph fusion is what lets us absorb future
features (transmission publishing, stereo) without breaking the
budget.

The dynamic shape profile (256→720) means the same engine handles
training crops, ROS2 subscriber resolutions, and 720p inference —
no need to ship multiple engines.

## 8. Why ROS2 Humble?

ROS2 Humble is the LTS release for Ubuntu 22.04 and matches what
most subsea autonomy stacks (BlueROV2 / Reach Robotics / Saab Sabertooth)
ship today. Wiring choices:

- **Topics, not services**: image streams are inherently topic-y; we
  follow `image_pipeline` conventions.
- **All endpoints parameterised**: deployment configs change between
  dive types; hard-coding topic names guarantees yak-shaving in the
  field.
- **Backend swap (TRT/Torch) at start time**: dev workstations
  rarely have TRT; CI doesn't either. The same node binary works in
  both, with TRT preferred when available.
- **`cv_bridge` for image conversion**: standard, well-tested, and
  matches the rest of the LEGION middleware.

## 9. Why the directory layout?

```
src/aquaclr/         <- package source ("src layout" prevents accidental imports of non-installed code)
configs/             <- Hydra configs at the repo root, not inside the package, so they're easy to grep
scripts/             <- thin entry points; all real logic in src/
tests/               <- mirrors src/
docs/                <- long-form architecture/decisions
docker/              <- production deployment images
notebooks/           <- exploratory; not imported from src/
data/, outputs/      <- gitignored runtime dirs
```

Why **src layout**? Without it, `import aquaclr` works just because
the cwd happens to contain a folder of that name. The src layout
forces an editable install (`uv pip install -e .`) which catches
broken imports immediately and makes the package behave the same
in CI, Docker, and dev.

Why split out `configs/` from `src/`? Hydra's `@hydra.main` decorator
needs a path it can compose. Keeping configs at the repo root makes
them: (a) overrideable per-deployment without rebuilding the
package, (b) easy to include in a `--config-path=/etc/legion/...`
invocation in production, (c) trivially diffable in PRs.

## 10. Why these tooling choices (uv / ruff / mypy / Hydra)?

| Tool | Why it, not a competitor |
| --- | --- |
| **uv** | 10–100× faster than pip; same PyPI; same wheels; ships a single binary; lockfile compatibility. The 2026 default for Python project management. |
| **ruff** | Replaces `flake8` + `isort` + `black` + most of `pylint` in one binary. Sub-second on this whole repo. Formatter and linter agree. |
| **mypy --strict** | The bar for new ML codebases in 2026. Catches the silent-broadcasting class of bugs that destroy a training run on day 3. |
| **Hydra** | Composable configs (model × data × train), structured CLI overrides, multirun sweeps, automatic working-directory management. The de facto config standard for ML research code. |
| **PyTorch Lightning 2.x** | The cleanest separation between "what is the network?" and "how do we train it?" Built-in support for compile, bf16-mixed, deterministic mode, EMA-via-callback. |
| **TorchMetrics** | `_state_dict`-friendly, distributed-aware metric implementations. Re-implementing PSNR / SSIM in plain PyTorch is fine for tests; for actual logging it's not. |
| **albumentations** | Atomic geometric transforms over multiple targets (`I`, `J`, `t_gt`). Replacement for handwritten `torchvision.transforms` chains that get out of sync. |

## 11. Performance budget walkthrough

Where does the 15 ms budget come from, and how do we hit it?

```
Total budget         : 15.0 ms
  - I/O (cv_bridge)  :  1.5 ms
  - HWC->CHW + norm  :  0.5 ms
  - Inference (TRT)  : 11.0 ms      <-- our compute target
  - CHW->HWC + clip  :  0.5 ms
  - I/O (publish)    :  1.5 ms
                      ---------
                       15.0 ms
```

Inference compute breakdown at 720p, FP16, TRT (estimated from FLOP
count + RTX 3050 throughput):

```
Encoder (MobileNetV3-Small)   : ~4.5 ms
UNet decoder (DSC)            : ~5.0 ms
Heads + physics inversion     : ~0.8 ms
Tensor reshape / overhead     : ~0.7 ms
                               ---------
                                ~11.0 ms (matches budget)
```

VRAM breakdown at 720p, FP16, batch=1:

```
Engine binary                 : ~12 MB
Activations (peak in decoder) : ~75 MB
Workspace                     : ~30 MB
Misc (CUDA context, streams)  : ~50 MB
                               --------
                                ~167 MB total (4% of 4 GB ceiling)
```

This leaves ~3.8 GB for the rest of the LEGION stack (SLAM, planner,
visualisation, ROS2 bag recording).

## 12. Automotive SiL parallels (long form)

Project LEGION's perception engineers come from automotive ADAS
backgrounds. The codebase's docstrings carry "Automotive SiL parallel"
notes everywhere; here are the highlights in one place.

| Marine concept | Automotive concept |
| --- | --- |
| Marine snow particulates | Lidar rain clutter, camera rain droplets, snow flakes on lens |
| Jaffe-McGlamery `I = J·t + B(1−t)` | Koschmieder atmospheric scattering model |
| Per-pixel transmission `t(x)` | Per-ray optical depth or lidar return-intensity confidence |
| Global backscatter `B` | Airlight constant in atmospheric model / radar noise floor |
| LEGION-DeSnow upstream of SLAM | Camera de-rain / lidar declutter upstream of perception fusion |
| MSRB synthetic snow on Flickr originals | Procedurally rain-augmented KITTI / nuScenes |
| LSUI transmission maps | NVIDIA DriveSim depth + scattering ground truth |
| UIEB-Challenge held-out real footage | Real-world ADAS test campaigns (e.g. Berkeley Deep Drive rain) |
| TensorRT FP16 engine on RTX 3050 | TensorRT engine on DRIVE Orin SoC |
| ROS2 Humble image-pipeline plumbing | DriveOS / Apex.AI middleware |
| Marine-snow → SLAM corruption | Rain → object-detector corruption |
| Bf16-mixed training | Same mixed-precision regime used in production ADAS training pipelines |
| EMA + multi-loss + physics regulariser | Standard sensor-restoration recipe in automotive papers (Pfeuffer et al. 2019, Sakaridis et al. 2020) |

The cross-pollination is intentional: the M2 stretch goal is to ship
the same network, retrained on a rain-augmented automotive dataset,
into LEGION's parallel ADAS programme.

## 13. Open questions / Milestone-2 hooks

Tracked but deferred:

- **INT8 PTQ**: requires a calibration dataset of representative dive
  frames. Will fold into MSRB+LSUI plus a hand-collected ROV strip.
- **Stereo de-snowing**: synchronise stereo pairs via
  `ApproximateTimeSync`; share the `(t, B)` heads across both eyes.
- **Transmission topic**: publish `t` as `mono16` on
  `/camera/transmission` so SLAM can use it as a per-pixel feature
  weight (pencilled into the ROS2 node as a `TODO(M2)`).
- **Quantisation-aware training**: only worth the headache if INT8 PTQ
  loses > 0.5 dB PSNR.
- **Flow-aware temporal smoothing**: helpful on slow-pan ROV footage
  where particle locations are correlated frame-to-frame. Not in
  scope for an M1 single-frame baseline.
- **Self-supervised real-data fine-tune**: the physics-informed loss
  has a self-supervised mode (predict `(t, B)` and check forward-model
  consistency against `I` itself) that we can use on unlabelled dive
  footage in M2. The `forward_export` API is already shaped for this.
