# Chapter 8 — Evaluation Methodology

> **Learning objectives**
> By the end of this chapter you will be able to:
> 1. Define every evaluation metric used (PSNR, SSIM, UIQM, UCIQE, latency, VRAM).
> 2. Decide which metric to apply on which dataset and why.
> 3. Run the full evaluation suite end-to-end against a checkpoint.
> 4. Interpret an ablation table and identify which component contributes most.
>
> **TL;DR.** We pre-register four metric families: **reference-based**
> (PSNR, SSIM) on MSRB-test and LSUI-val, **no-reference** (UIQM,
> UCIQE) on UIEB-Challenge, **operational** (latency p50/p95/FPS,
> peak VRAM) on RTX 3050 in TRT FP16, and **physics-quality**
> (forward-consistency residual, t-smoothness) on all sets. The
> ablation plan removes one component at a time
> (`L_phys`, `L_ssim`, `L_tv`, `L_t`, EMA, BF16, `torch.compile`,
> mix ratio) and reports the delta on each metric.

## 8.1 Metric design — what we measure and why

| Family | Metric | Where | Range | Tells us |
| --- | --- | --- | --- | --- |
| Reference-based | **PSNR** | MSRB-test, LSUI-val | dB (≈18-35 typical) | How close `Ĵ` is to `J_gt` pixel-wise |
| Reference-based | **SSIM** | MSRB-test, LSUI-val | [-1, 1] | Local structural agreement |
| No-reference | **UIQM** | UIEB-Challenge | unbounded, higher better | Generic "underwater goodness" |
| No-reference | **UCIQE** | UIEB-Challenge | unbounded, higher better | Colour balance / saturation |
| Operational | **latency p50/p95** | RTX 3050 + TRT FP16 | ms / frame | Hits the < 15 ms budget? |
| Operational | **FPS** | same | frames / s | Throughput |
| Operational | **VRAM peak** | same | MB | Hits the < 1 GB run-time budget? |
| Physics | **forward residual** | MSRB-val | mean abs in [0, 1] | `||I − F(J_gt, t̂, B̂)||₁` |
| Physics | **t-smoothness** | MSRB-val | mean abs | `mean(|∇t̂|)`, anisotropic |

We deliberately **pre-register** these metrics in this dissertation
*before* running the experiments in Chapter 10. This protects
against post-hoc cherry-picking.

## 8.2 PSNR — peak signal-to-noise ratio

### 8.2.1 Definition

For images in `[0, 1]`:

$$
\mathrm{PSNR}(\hat{J}, J) \;=\; 10 \,\log_{10}\!\left(\frac{1}{\mathrm{MSE}(\hat{J}, J)}\right)
$$

where MSE is the mean squared error over all pixels and channels.
For 8-bit images in `[0, 255]` the numerator becomes `255²`.

### 8.2.2 Implementation

We use TorchMetrics' `PeakSignalNoiseRatio(data_range=1.0)`. It is
distributed-aware (works correctly with multi-GPU sums), accumulates
across batches, and emits a single number per validation epoch.

### 8.2.3 What PSNR is good and bad at

| Good | Bad |
| --- | --- |
| Cheap to compute | Insensitive to local structure (a uniformly-shifted image scores the same as a destroyed one of the same MSE) |
| Comparable across papers (universal definition) | Saturates: 30+ dB looks identical to the eye despite mathematical difference |
| Convex in pixel space | Penalises bright outliers quadratically (heavy-tail issue) |

## 8.3 SSIM — structural similarity

### 8.3.1 Definition

$$
\mathrm{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

with statistics computed in an 11×11 Gaussian window
(`σ = 1.5`), `C_1 = (0.01 \cdot L)²`, `C_2 = (0.03 \cdot L)²`,
`L = 1.0` for `[0, 1]` data.

### 8.3.2 Why SSIM matters more than PSNR for SLAM

SLAM cares about edges and corners. SSIM measures whether local
edge structure is preserved. A method that scores 25 dB PSNR but
**0.92 SSIM** preserves SLAM-relevant features; one that scores
26 dB PSNR but **0.85 SSIM** has flattened texture in a way that
hurts feature matching.

### 8.3.3 Implementation

TorchMetrics' `StructuralSimilarityIndexMeasure(data_range=1.0)`.
Internal kernel matches our loss-side SSIM (`window=11`,
`σ=1.5`, `K_1=0.01`, `K_2=0.03`).

## 8.4 UIQM and UCIQE — no-reference metrics for UIEB-Challenge

For UIEB-Challenge there is **no clean reference image**, so PSNR
and SSIM are unavailable. The community has converged on two
no-reference scores.

### 8.4.1 UIQM (Underwater Image Quality Measure) [Panetta 2016]

A weighted combination of three sub-measures:

$$
\mathrm{UIQM} = c_1 \cdot \mathrm{UICM} + c_2 \cdot \mathrm{UISM} + c_3 \cdot \mathrm{UIConM}
$$

where:

- **UICM** is colourfulness based on `(R−G)`, `(R+G−2B)/2` chromaticity statistics,
- **UISM** is sharpness via Sobel-magnitude entropy,
- **UIConM** is contrast in the AME (Asymmetric Modulation Estimation) sense,

with default weights `c = (0.0282, 0.2953, 3.5753)` from the original paper.

### 8.4.2 UCIQE (Underwater Colour Image Quality Evaluation) [Yang 2015]

In CIE-Lab space:

$$
\mathrm{UCIQE} = w_1 \cdot \sigma_c + w_2 \cdot \mathrm{con}_l + w_3 \cdot \mu_s
$$

where `σ_c` is chroma standard deviation, `con_l` is luminance
contrast (max-min), and `μ_s` is saturation mean. Default weights
`w = (0.4680, 0.2745, 0.2576)`.

### 8.4.3 Caveats

Both metrics are **proxies**. They correlate with human preference
but not perfectly. We report them for comparability with prior
underwater literature, not as ground truth. We deliberately report
**both** so a reader can see consistency.

### 8.4.4 Implementation choice

We use [`pyiqa`](https://github.com/chaofengc/IQA-PyTorch) (PyTorch
implementations of both), as an *optional* dependency. Snippet:

```python
import pyiqa, torch
uiqm = pyiqa.create_metric("uiqm").to(device)
uciqe = pyiqa.create_metric("uciqe").to(device)
score_uiqm = uiqm(j_pred_uint8.permute(0, 3, 1, 2).float() / 255.0)
score_uciqe = uciqe(j_pred_uint8.permute(0, 3, 1, 2).float() / 255.0)
```

## 8.5 Operational metrics — latency, throughput, VRAM

### 8.5.1 Latency definitions

We report three latency statistics over `n_iter = 200` warmed-up runs:

| Statistic | Definition |
| --- | --- |
| **p50** | 50th percentile (median) inference time |
| **p95** | 95th percentile — captures tail latency |
| **mean** | Arithmetic mean — for reproducibility with prior literature |

Implementation: [`src/aquaclr/inference/benchmark.py::benchmark_pytorch`](../../src/aquaclr/inference/benchmark.py).
For TRT we replicate the same harness against a `TensorRTRunner`
instance.

### 8.5.2 Throughput

$$
\mathrm{FPS} = \frac{1000}{\mathrm{mean\_ms}}
$$

We report on `batch=1` (the deployment-relevant case for ROS2
streaming).

### 8.5.3 VRAM peak

`torch.cuda.max_memory_allocated()` after `torch.cuda.reset_peak_memory_stats()`.
Captures the maximum activation + parameter memory the GPU saw
during inference. Should sit well under the 1 GB run-time budget.

### 8.5.4 Reporting protocol

```
| Path             | p50 (ms) | p95 (ms) | mean (ms) | FPS | Peak VRAM (MB) |
| ---------------- | -------- | -------- | --------- | --- | -------------- |
| PyTorch FP32     |          |          |           |     |                |
| PyTorch FP16     |          |          |           |     |                |
| TensorRT FP16    |          |          |           |     |                |
| TensorRT INT8    |          |          |           |     | (M2)           |
```

Always at 720p (1280×720) batch=1, on the canonical RTX 3050.

## 8.6 Physics-quality metrics

To audit the physics-informedness of the trained network — beyond
just "the recovered image looks right":

### 8.6.1 Forward-physics residual

For each `(I, J_gt, t̂, B̂)` in MSRB-val:

$$
r = \bigl\| I - F(J_{\text{gt}}, \hat{t}, \hat{B}) \bigr\|_1
$$

Lower `r` = predicted `(t̂, B̂)` are physically meaningful given
the GT `J`. We report the mean of `r` over the val set.

### 8.6.2 Transmission smoothness

$$
\mathrm{TV}(\hat{t}) = \mathbb{E}\bigl[ |\partial_x \hat{t}| + |\partial_y \hat{t}| \bigr]
$$

Captures whether the transmission map is piecewise-smooth (low TV)
as the model expects, or salt-and-pepper noise (high TV).

### 8.6.3 Backscatter sanity

For the predicted `B̂` vector: report **mean and standard
deviation** across MSRB-val. If `B̂` is wildly variable across
similar scenes, the head has not learned a useful prior.

## 8.7 Ablation plan

Each ablation **removes or swaps one component** at a time and
reports the delta on every metric. The full plan:

| ID | Ablation | What changes |
| --- | --- | --- |
| **A0** | Baseline (full system) | — |
| A1 | `λ_phys = 0` | Drop forward-consistency loss |
| A2 | `λ_ssim = 0` | Drop SSIM loss |
| A3 | `λ_tv = 0` | Drop TV regulariser on `t` |
| A4 | `λ_t = 0` | Drop direct `t` supervision (LSUI batches still train recon+phys+ssim+tv) |
| A5 | EMA off | `EMAWeightCallback` disabled |
| A6 | `precision = "16-mixed"` | FP16 instead of BF16 |
| A7 | `compile.enabled = false` | No `torch.compile` |
| A8 | MSRB-only | `data=msrb` (no LSUI) |
| A9 | LSUI-only | `data=lsui` |
| A10 | Mix 50/50 | `mix_ratio = (0.5, 0.5)` |
| A11 | Backbone-from-scratch | `pretrained: false` |
| A12 | No backbone freeze | `freeze_backbone_epochs: 0` |
| A13 | Per-channel `t` (oracle) | swap t-head to 3-channel; needs net change — sketch only |

Each experiment is run with the **same seed** (1337), the same data
splits, and the same hardware so deltas are attributable to the
single change.

### 8.7.1 Reporting template

```
| ID  | Variant                     | val PSNR | val SSIM | UIQM | UCIQE | Latency p50 | VRAM | Forward residual |
| --- | --------------------------- | -------- | -------- | ---- | ----- | ----------- | ---- | ---------------- |
| A0  | Full                        |          |          |      |       |             |      |                  |
| A1  | -L_phys                     |          |          |      |       |             |      |                  |
| ... |                             |          |          |      |       |             |      |                  |
```

Chapter 10 fills this table with measured numbers from the
trained models.

### 8.7.2 What we expect (pre-registration)

| Ablation | Expected sign of `Δ val/PSNR` | Justification |
| --- | --- | --- |
| A1 (-L_phys) | **−** (small) | Recon dominates; physics term mostly affects `(t, B)` consistency |
| A2 (-L_ssim) | **−** (clear) | SSIM is what guards SLAM-relevant edges |
| A3 (-L_tv) | + on val PSNR, **−** on UIEB | TV hurts strict pixel fit but improves generalisation |
| A4 (-L_t) | + on MSRB, **−** on UIEB | LSUI's `t_gt` improves generalisation, slightly hurts MSRB-val |
| A5 (EMA off) | **−** (~0.3 dB) | Standard EMA effect |
| A6 (FP16) | ≈ 0 if no overflow, big − if overflow | BF16 is safer |
| A7 (no compile) | ≈ 0 on quality, **−** on training time | Compile is purely a speed lever |
| A8 (MSRB only) | + on MSRB, **−** on UIEB | MSRB-only overfits its synth distribution |
| A9 (LSUI only) | **−** (clear) | LSUI lacks discrete particulates |
| A10 (50/50) | minor | Should be close to A0 |
| A11 (no pretrain) | **−** (large) | ImageNet features matter |
| A12 (no freeze) | **−** (small) | Backbone damaged by random head gradients early |
| A13 (per-channel t) | ≈ 0 PSNR, **−** size | Doubles head params without visible benefit |

Chapter 10 reports actual deltas; matches between expected and
measured signs validate the methodology.

## 8.8 Cross-domain generalisation tests

Beyond the held-out UIEB-Challenge eval, we additionally check:

| Test | Hypothesis | Evidence form |
| --- | --- | --- |
| Train MSRB-Task-1, eval MSRB-Task-2 | Smaller particles transfer to larger | PSNR delta |
| Train MSRB+LSUI, eval RUIE-UIQS | Generalises to a third real dataset | Qualitative + UIQM |
| Train at 256², eval at 720p (no retrain) | Architecture is resolution-flexible | PSNR + UIQM |
| Inject Gaussian noise on UIEB before inference | Robust to sensor noise | PSNR / UIQM curves vs noise σ |

These are stretch tests; we report them descriptively rather than
declare pass/fail thresholds.

## 8.9 Latency budget verification

The 15 ms target is composed of:

| Stage | Budget (ms) | Implementation |
| --- | --- | --- |
| Frame I/O via cv_bridge | 1.5 | `bridge.imgmsg_to_cv2(msg, "rgb8")` |
| HWC→CHW + normalize | 0.5 | NumPy stride trick + `astype(float32) / 255` |
| Model inference (TRT FP16) | 11.0 | `TensorRTRunner.__call__` |
| CHW→HWC + clip + uint8 | 0.5 | NumPy reshape + clip |
| Frame I/O publish | 1.5 | `bridge.cv2_to_imgmsg` + `publisher.publish` |
| **Total** | **15.0** | |

The benchmark harness measures only the inference component (the
11 ms slice). End-to-end latency including I/O is reported in
Chapter 10 from a recorded ROS2 bag.

## 8.10 Statistical reporting practices

For each metric reported in Chapter 10:

1. **Mean ± standard deviation** computed across the relevant
   validation/test set.
2. **Bootstrap 95 % confidence interval** with 1000 resamples for
   the headline metrics (PSNR, SSIM on MSRB-test). Computed via:

   ```python
   from scipy.stats import bootstrap
   ci = bootstrap((per_image_psnr,), np.mean, n_resamples=1000,
                  confidence_level=0.95).confidence_interval
   ```

3. **Three random seeds** for the headline ablation rows (A0, A1,
   A4, A8). Other rows are single-seed.

This avoids reporting a fluke result without doubling the compute
budget.

## 8.11 Reporting practice — readability over numerical density

Following the [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/),
all results tables in Chapter 10 use:

- 2 decimal places for PSNR (e.g. 25.43), 4 for SSIM (e.g. 0.9182).
- Percentage points expressed as `±N.N pp` (not `±N.N %`).
- Bold the best value in each column.
- Single-row stats: cell value `± std`. Double-row stats:
  `value (95 % CI low – high)`.

## 8.12 The exact commands an examiner runs

To re-produce every number in Chapter 10:

```bash
# 1. Train the baseline.
python scripts/train.py train.run_name=A0_baseline

# 2. Run each ablation.
for tag in A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12; do
  python scripts/train.py train.run_name=$tag <ablation flags from §8.7>
done

# 3. Evaluate on MSRB-test, LSUI-val, UIEB-Challenge.
for ckpt in outputs/A*/ckpts/*.ckpt; do
  python scripts/evaluate.py --ckpt $ckpt --data-root data/msrb --split test --task 1
done

# 4. Latency benchmark (PyTorch + TRT).
python scripts/export_onnx.py --ckpt outputs/A0_baseline/ckpts/best.ckpt \
    --build-trt --benchmark --height 720 --width 1280
```

The expected wall-clock for the full ablation matrix on a single
RTX 3050 is ~14 days. Chapter 10 records which subset was actually
run and which were extrapolated.

---

## Key takeaways

- We pre-register four metric families: **reference-based**
  (PSNR, SSIM), **no-reference** (UIQM, UCIQE),
  **operational** (latency, VRAM), and **physics-quality**
  (forward residual, t-smoothness).
- The right metric depends on the dataset: PSNR/SSIM on
  MSRB-test, UIQM/UCIQE on UIEB-Challenge.
- Latency / VRAM are reported on RTX 3050 at 720p in TRT FP16,
  including p50, p95, mean, FPS, and peak VRAM.
- The ablation plan **changes one component at a time** (A1–A13)
  with expected directional outcomes pre-registered.
- All headline rows include 95 % bootstrap CIs and three-seed
  variance.

## Cross-references

- Forward to [Chapter 9 — Deployment](09_deployment.md)
- Code: [`scripts/evaluate.py`](../../scripts/evaluate.py),
  [`src/aquaclr/inference/benchmark.py`](../../src/aquaclr/inference/benchmark.py)
- Glossary entries: PSNR, SSIM, UIQM, UCIQE — see [Appendix B](B_glossary.md)
