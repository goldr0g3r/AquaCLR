# Model Card — LEGION-DeSnow-S

## Model details

- **Name:** LEGION-DeSnow-S (Small)
- **Version:** 0.1.0 (Milestone 1)
- **Type:** Physics-informed encoder-decoder CNN for marine-snow removal
- **Architecture:** MobileNetV3-Small encoder + Depthwise-Separable UNet decoder + (transmission, backscatter) heads + analytic Jaffe-McGlamery inversion
- **Parameters:** ~4–6 M (≤ 24 MB FP32 / ≤ 12 MB FP16)
- **Input:** 3-channel RGB image, range `[0, 1]`, any spatial size
- **Output:** Recovered clean image `J ∈ [0, 1]^{B×3×H×W}`, transmission `t ∈ (0, 1)^{B×1×H×W}`, backscatter `B ∈ (0, 1)^{B×3}`
- **Target hardware:** NVIDIA RTX 3050 (Ampere, 4 GB VRAM); also runs on Jetson Orin / DRIVE Orin
- **Target latency:** < 15 ms / 720 p frame, FP16 / TensorRT
- **License:** Apache-2.0 (model code and weights)

## Intended use

Real-time particulate (marine-snow) removal upstream of underwater SLAM
and visual odometry pipelines (Project LEGION). Use in:

- Subsea inspection ROVs / AUVs
- Ocean-floor mapping / coral-reef monitoring
- Underwater archaeology video pre-processing
- Research into physics-informed image restoration

## Out-of-scope / known failure modes

- **Extreme turbidity** (silt storms, coastal plumes) where `t(x) → 0`
  almost everywhere; the analytic inversion becomes ill-conditioned
  even with `eps=1e-3` clamping.
- **Severe non-uniform illumination** (single dive light, deep
  shadows): the simplified Jaffe-McGlamery model assumes a global
  ambient `B`, so per-pixel light non-uniformity bleeds into the
  transmission estimate.
- **Caustics on shallow sea-floor** (rippling sun patterns): looks
  superficially like marine snow to the network; expect occasional
  over-smoothing here.
- **Freshwater scenes**: model trained on saltwater + Flickr-CC clean
  imagery; behaviour on rivers / lakes is unverified.

## Training data

- **Primary**: [MSRB](https://github.com/ychtanaka/marine-snow) — 2,300 train / 400 test paired snowy/clean 384×384 images. Marine-snow particles are synthesised on top of clean Flickr-CC originals; particle counts 100–600 per image.
- **Auxiliary**: [LSUI](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html) — 4,279 paired underwater images with reference enhanced ground truth and **medium transmission maps** (used to supervise the `t(x)` head directly when present).
- **Held-out evaluation**: [UIEB-Challenge](https://li-chongyi.github.io/proj_benchmark.html) — 60 unpaired challenging underwater images, used only for qualitative cross-domain evaluation.

### Dataset bias caveats

- MSRB clean originals are sourced from Flickr-CC and skew toward
  recreational dive footage in tropical clear water. Performance on
  cold-water / industrial-inspection footage may be lower.
- LSUI's transmission GT is itself estimated (not measured), so
  direct `t` supervision is a soft target — we weight it at `λ_t=0.5`
  to avoid over-fitting to a noisy reference.

### Download instructions (Milestone 1)

Until automated downloads are wired (Milestone 2), please follow the
official sources directly:

1. **MSRB** — clone or download from <https://github.com/ychtanaka/marine-snow>;
   unpack into `data/msrb/` using the upstream layout, i.e.
   `data/msrb/training/original/`, `data/msrb/training/MSR_Task{1,2}/`,
   and the same trio under `data/msrb/test/`. The `MSRBDataset.task`
   argument (1 or 2) selects which snowy variant pairs against
   `original/`. The legacy flat layout
   (`data/msrb/{train,test}/{noisy,clean}/`) is still auto-detected.
2. **LSUI** — request from the project page above; place under
   `data/lsui/{input,GT}/`. If you also have the GT transmission maps,
   add them under `data/lsui/transmission/` (optional).
3. **UIEB** — download from the project page; place
   `data/uieb/challenging-60/` for evaluation.

## Training procedure

| Setting | Value |
| --- | --- |
| Optimiser | AdamW (`lr=3e-4`, `weight_decay=1e-4`, `betas=(0.9, 0.999)`) |
| Scheduler | OneCycleLR (`pct_start=0.1`, cosine annealing) |
| Precision | `bf16-mixed` (Ampere; native BF16 keeps loss landscape stable) |
| Batch size | 16 (MSRB) / 8 (LSUI), `accumulate_grad_batches=2` |
| Epochs | 60 (typical convergence ~40-50) |
| Backbone freeze | First 2 epochs (heads warm-up) |
| EMA decay | 0.9995 |
| Augmentation | Albumentations: random crop, h-flip, mild colour-jitter on `I` only |

## Loss

```
L = 1.0  · L_recon(J_pred, J_gt)            # Charbonnier
  + 0.5  · L_phys(I, J_gt·t + B·(1−t))      # forward consistency
  + 0.5  · (1 − SSIM(J_pred, J_gt))         # structural fidelity
  + 0.01 · TV(t)                            # smooth transmission prior
  + 0.5  · L1(t, t_gt)                      # only when LSUI batch
```

## Evaluation metrics

- **Reference-based** (MSRB-test, LSUI-val): PSNR, SSIM
- **No-reference** (UIEB-Challenge): UIQM, UCIQE (via `pyiqa`)
- **Latency** (RTX 3050, FP16, TRT): p50 / p95 / mean ms @ 720 p
- **VRAM** (RTX 3050): peak MB

## Ethical considerations

- This model is a **sensor preprocessing** stage; downstream applications
  (e.g. inspection, archaeology, marine biology) may carry their own
  ethical considerations, but the model itself produces colour images
  from colour images and does not perform recognition or classification.
- The model should not be used to reconstruct video evidence in
  contexts where the legal/forensic chain of custody requires
  unmodified raw imagery.

## Automotive SiL parallel

LEGION-DeSnow-S is the **subsea analogue** of an ADAS sensor-restoration
block (camera de-rain or lidar de-clutter). The structure of the
network, the training regime, and the deployment story (TensorRT FP16
on a power-constrained edge SoC, ROS2 middleware) all transfer
directly to automotive perception preprocessing.

## Citation

If you use this code or model, please cite:

```bibtex
@misc{legion-desnow-s-2026,
  title  = {LEGION-DeSnow-S: A physics-informed CNN for real-time underwater marine-snow removal},
  author = {Project LEGION},
  year   = {2026},
  note   = {Milestone 1, AquaCLR repository}
}
```
