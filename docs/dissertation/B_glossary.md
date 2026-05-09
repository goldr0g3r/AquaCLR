# Appendix B — Glossary

Plain-English definitions of every term, acronym, and symbol used in
this dissertation. Sorted alphabetically.

## Symbols

| Symbol | Definition | First appears |
| --- | --- | --- |
| `B` | Global ambient backscatter vector, shape `(3,)`, range `(0, 1)` | Ch. 3 |
| `B_at_camera(x)` | Backscatter contribution at the camera for ray `x` | Ch. 3 |
| `c(λ)` | Per-wavelength attenuation coefficient (m⁻¹) | Ch. 3 |
| `C₁, C₂` | SSIM stability constants (`10⁻⁴`, `9 · 10⁻⁴` for `[0, 1]` data) | Ch. 3 |
| `d(x)` | Distance from camera along ray `x` (m) | Ch. 3 |
| `ε` | Transmission floor for stable inversion, default `10⁻³` | Ch. 3 |
| `F(J, t, B)` | Forward Jaffe-McGlamery operator | Ch. 3 |
| `H, W` | Spatial dimensions (height, width) of an image tensor | Ch. 3 |
| `I(x)` | Observed (snowy/hazy) image, range `[0, 1]` | Ch. 3 |
| `J(x)` | Clean scene radiance (target output), range `[0, 1]` | Ch. 3 |
| `Ĵ(x)` | Predicted clean image (analytic inversion of predicted `(t, B)`) | Ch. 3 |
| `λ` (lambda) | Wavelength of light (nm) | Ch. 3 |
| `λ_rec, λ_phys, ...` | Loss-term weights | Ch. 3 |
| `μ_x, σ_x²` | Local mean / variance in SSIM Gaussian window | Ch. 3 |
| `t(x)` | Per-pixel transmission, range `(0, 1)` | Ch. 3 |
| `t̂(x)` | Predicted transmission map | Ch. 3 |
| `t_gt(x)` | Ground-truth transmission (only on LSUI) | Ch. 6 |

## Acronyms and abbreviations

| Term | Expansion / definition |
| --- | --- |
| **AdamW** | Adam optimiser with decoupled weight decay [Loshchilov 2019] |
| **ADAS** | Advanced Driver Assistance Systems |
| **API** | Application Programming Interface |
| **AUV** | Autonomous Underwater Vehicle |
| **BF16** | Brain Floating Point 16-bit (8 exponent + 7 mantissa bits, FP32 dynamic range) |
| **C4** | Context, Container, Component, Code — the 4-level architecture-diagram framework |
| **CDI** | Container Device Interface — modern way to expose GPUs to OCI containers |
| **cuDNN** | NVIDIA's CUDA Deep Neural Network library |
| **CV** | Computer Vision |
| **CycleGAN** | Cycle-consistent Generative Adversarial Network |
| **DCP** | Dark Channel Prior [He 2009] |
| **DSC** | Depthwise-Separable Convolution |
| **EMA** | Exponential Moving Average |
| **FP16, FP32** | 16-bit / 32-bit Floating Point |
| **FPS** | Frames Per Second |
| **GAN** | Generative Adversarial Network |
| **GAP** | Global Average Pooling |
| **GFLOP** | Giga (10⁹) Floating-Point Operations |
| **GT** | Ground Truth |
| **Humble** | ROS 2 Humble Hawksbill (LTS, Ubuntu 22.04, → May 2027) |
| **INT8** | 8-bit Integer (used for quantised inference) |
| **ISP** | Image Signal Processor (in cameras) |
| **Jazzy** | ROS 2 Jazzy Jalisco (LTS, Ubuntu 24.04, → May 2029) |
| **L1, L2** | First / second-order norms (sum of absolute / squared values) |
| **LSUI** | Large-Scale Underwater Image dataset [Peng 2021] |
| **M1, M2, M3** | LEGION milestones — Front-end, SLAM hand-off, Sea trial |
| **MAC** | Multiply-Accumulate operation |
| **MNv3** | MobileNetV3 [Howard 2019] |
| **MSE** | Mean Squared Error |
| **MSRB** | Marine Snow Removal Benchmark [Sato 2023] |
| **NaN** | Not a Number — a floating-point representation of an invalid result |
| **NPU** | Neural Processing Unit (mobile/edge ML accelerator) |
| **OCI** | Open Container Initiative |
| **ONNX** | Open Neural Network Exchange — a portable graph format |
| **OOD** | Out-of-Distribution |
| **OOM** | Out-of-Memory |
| **PINN** | Physics-Informed Neural Network [Raissi 2019] |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) |
| **PTQ** | Post-Training Quantisation |
| **QoS** | Quality of Service (in ROS 2) |
| **RANSAC** | Random Sample Consensus (robust estimator used by SLAM) |
| **rclpy** | ROS 2 Client Library for Python |
| **ReLU, ReLU6** | Rectified Linear Unit / clamped to `[0, 6]` |
| **ROS** | Robot Operating System (`ROS 2` in this work — distinct from ROS 1) |
| **ROV** | Remotely Operated Vehicle |
| **RPMFusion** | Third-party Fedora repository for proprietary drivers |
| **SE** | Squeeze-Excitation block (a feature-attention module) |
| **SiL** | Software-in-the-Loop — automotive SW testing pattern |
| **SLAM** | Simultaneous Localisation And Mapping |
| **SSIM** | Structural Similarity Index Measure [Wang 2004] |
| **SWA** | Stochastic Weight Averaging [Izmailov 2018] |
| **TRT** | TensorRT — NVIDIA's high-performance inference SDK |
| **TV** | Total Variation (regulariser) |
| **UCIQE** | Underwater Colour Image Quality Evaluation [Yang 2015] |
| **UDCP** | Underwater Dark Channel Prior [Drews 2013] |
| **UIEB** | Underwater Image Enhancement Benchmark [Li 2019] |
| **UIE** | Underwater Image Enhancement (the broader research field) |
| **UIQM** | Underwater Image Quality Measure [Panetta 2016] |
| **uv** | Astral's fast Python project / package manager |
| **VRAM** | Video Random Access Memory (GPU memory) |
| **W&B** | Weights & Biases (experiment tracking) |

## Concepts

### Anisotropic TV
Total variation with separate L1 contributions from the `x` and `y`
discrete differences. Sharper at axis-aligned edges than the
isotropic (square-root) form.

### Backbone freeze
Holding encoder weights fixed for the first few epochs while the
randomly-initialised heads stabilise. Prevents large head-side
gradients from corrupting the pretrained backbone.

### Backscatter
Photons that are scattered *into* the camera by ambient light
hitting suspended particles, contributing a global "veil" to the
image whose intensity grows with the path length through the
medium.

### Channels-last memory format
PyTorch tensor layout where channels are stored last instead of
second (`NHWC` instead of `NCHW`). Matches Ampere tensor-core
expectations and gives a measurable speedup on convolutions.

### Charbonnier loss
Smooth approximation of L1: `√(d² + β²)`. Robust to outliers like
L1, but with a smooth gradient at zero like L2.

### Dynamic shape profile (TRT)
A `(min, opt, max)` triple of input shapes. The TensorRT engine is
built to handle any shape between `min` and `max` while being
optimised at `opt`.

### Forward-physics consistency
Loss term that penalises violation of `I = J·t + B(1 − t)` evaluated
at the predicted `(t, B)` and the GT `J`. The "soft" physics-informed
constraint.

### Gauge symmetry (in physics-informed inversion)
Multiple `(J, t, B)` triples can produce the same `I`. The
forward-consistency loss + global pooling break the most common
gauges (brightness, near-vs-far ambiguity).

### Identifiability
The property that a parameter (here, the joint `(t, B)`) is
uniquely determined from the observed data. Without `L_phys`, our
parameters are only weakly identifiable.

### Marine snow
Drifting particulate matter (organic detritus, sand, plankton)
that appears as bright streaks in underwater video.

### Mixed precision (mixed-mp / bf16-mixed)
Using lower-precision (FP16/BF16) for activations and gradients
while keeping parameters in FP32. Halves memory and ~doubles
throughput on Ampere+.

### Modus operandi (in ROS 2)
The pattern of "subscribe to a topic, transform, publish to another"
implemented in this work as `LegionDeSnowNode`.

### ONNX dynamic axes
Marking certain tensor dimensions as variable in the exported
graph, so the same .onnx (and resulting .engine) handles a range
of input shapes.

### OneCycleLR
Learning-rate schedule that ramps up to `max_lr` linearly during
the first `pct_start` fraction of training, then anneals (cosine)
to a small final value.

### Physics-informed
A neural network whose architecture or loss explicitly encodes a
known physical equation. Combines structural ("hard") and loss
("soft") constraints in our case.

### Reduce-overhead (`torch.compile` mode)
Compile mode that fuses common operations safely without aggressive
re-autotuning on shape changes. Best default for variable-batch
training.

### Round-trip identity (Jaffe-McGlamery)
The forward operator followed by the inverse with the *same*
`(t, B)` recovers `J` exactly (modulo numerical noise). Used as a
unit-test invariant.

### Soft-pinning (datasets)
Treating a noisy reference (e.g. LSUI's transmission GT) with a
moderate loss weight, rather than as hard supervision, so the
network is guided but not forced to match.

### Total variation (TV)
A smoothness regulariser that penalises high-frequency content.
Encourages piecewise-smooth outputs.

### Transmission map
The Jaffe-McGlamery `t(x) ∈ [0, 1]` representing how much of `J(x)`
survives the path through the medium to reach the camera.

## Cross-references

- Forward to [Appendix C — Code Reference](C_code_reference.md)
- Back to [Chapter 12 — References](12_references.md)
