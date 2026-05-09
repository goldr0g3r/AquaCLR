# Chapter 3 — Theoretical Foundation

> **Learning objectives**
> By the end of this chapter you will be able to:
> 1. Derive the simplified Jaffe-McGlamery image-formation model from first principles.
> 2. Algebraically invert it to recover `J` from `(I, t, B)`.
> 3. State the mathematical form of every loss term and explain why it is included.
> 4. Argue why the resulting network qualifies as physics-informed in the formal sense.
>
> **TL;DR.** The simplified Jaffe-McGlamery model assumes the
> observed image `I` is a convex combination of the clean radiance
> `J` (weighted by transmission `t`) and the global backscatter `B`
> (weighted by `1 − t`). The forward model is a multiplication; the
> inverse is a clamped division. The composite loss combines (a)
> reconstruction of `J`, (b) consistency of the **forward** model
> evaluated at the *predicted* parameters and the *true* `J`, (c)
> structural similarity, (d) anisotropic total variation on `t`,
> and optionally (e) direct supervision of `t` from LSUI.

## 3.1 Notation and conventions

Throughout this dissertation:

| Symbol | Meaning | Shape (PyTorch) | Range |
| --- | --- | --- | --- |
| `I(x)` or `I` | Observed image | `(B, 3, H, W)` | `[0, 1]` |
| `J(x)` or `J` | Clean (underlying) image | `(B, 3, H, W)` | `[0, 1]` |
| `t(x)` or `t` | Transmission map | `(B, 1, H, W)` | `(0, 1)` |
| `B` | Global backscatter vector | `(B, 3)` | `(0, 1)` |
| `x` | Pixel coordinate | `(H, W)` | — |
| `λ` (lambda) | Wavelength of light | scalar (nm) | visible: 380–780 nm |
| `c(λ)` | Per-wavelength attenuation coefficient | scalar (m⁻¹) | water-type dependent |
| `d(x)` | Distance from camera along ray `x` | scalar (m) | scene-dependent |

Indexing: `(b, c, h, w)` where `b` is batch index, `c` is channel,
`h`, `w` are pixel row, column. Tensor norms: `||·||_1` is mean
absolute value over all elements unless indicated.

## 3.2 Underwater image formation — first-principles derivation

### 3.2.1 Beer-Lambert (single ray, no scattering)

Consider a single ray from a Lambertian point on a surface to the
camera, travelling distance `d` through a homogeneous medium with
attenuation coefficient `c(λ)`. Beer-Lambert's law gives the
fraction of photons that reach the camera:

$$
t(x) \;=\; \exp\!\bigl(-c(\lambda)\,d(x)\bigr)
$$

Hence the surface radiance `J(x)` is observed at the sensor as
`J(x)·t(x)`. This is the **direct transmission term**.

### 3.2.2 Adding ambient scattering (backscatter)

Ambient illumination `B` (sunlight or a dive light) interacts with
suspended particles along the same ray. By symmetry — every
infinitesimal slice of the ray contributes a small amount of
backscatter that is itself attenuated by the medium between that
slice and the camera — one can integrate to show that the total
backscatter at the camera is:

$$
B_{\text{at camera}}(x) \;=\; B \,\bigl(1 - t(x)\bigr)
$$

When `t = 1` (no medium) there is no backscatter; when `t → 0`
(infinite path) the backscatter saturates at `B`. This is the
**ambient transmission term**.

### 3.2.3 The simplified Jaffe-McGlamery equation

Adding direct + ambient terms:

$$
\boxed{ I(x) \;=\; J(x)\,t(x) \;+\; B\,\bigl(1 - t(x)\bigr) }
$$

This is the **simplified Jaffe-McGlamery model** [Jaffe 1990,
McGlamery 1980, simplified by Drews 2013 and Akkaynak 2019]. The
"simplified" part refers to several assumptions we adopt:

> **Assumption 1 (homogeneous medium).** `c(λ)` is independent of
> `x`. Reasonable over scene scales of metres in coastal waters.
>
> **Assumption 2 (scalar transmission).** We use a single `t(x)`
> per pixel rather than per-channel `(t_R, t_G, t_B)`. This is a
> simplification justified empirically (§2.1.1) by the fact that
> per-channel `t` doubles head parameters with no measurable PSNR
> gain on MSRB.
>
> **Assumption 3 (global backscatter).** `B` is constant across
> the image. Strictly true only for distant uniform ambient
> illumination; we discuss the failure mode of dive-lit close-up
> work in Chapter 10.

The Koschmieder atmospheric scattering model (used to render
fog/rain in automotive simulators) has *exactly the same algebraic
form*. This is the basis of every "Automotive SiL parallel" comment
sprinkled through the codebase.

### 3.2.4 The forward operator

The forward operator maps `(J, t, B)` to `I`:

$$
F(J, t, B)(x) \;=\; J(x)\,t(x) \;+\; B\,(1 - t(x))
$$

It is **linear in `J` for fixed `(t, B)`** and **bilinear in `(J, t)`
for fixed `B`**. Differentiable everywhere, with bounded Jacobian.

In code (see [`src/aquaclr/utils/physics.py`](../../src/aquaclr/utils/physics.py)):

```python
def apply_forward_jaffe_mcglamery(j, t, b):
    if b.dim() == 2:
        b = b.unsqueeze(-1).unsqueeze(-1)
    return (j * t + b * (1.0 - t)).clamp(0.0, 1.0)
```

The clamp at `[0, 1]` is required because numerically slightly
out-of-range inputs can give 1.001 or -0.001, which break
downstream metrics that assume `[0, 1]`.

### 3.2.5 The inverse operator

Solving the equation for `J`:

$$
J(x) \;=\; \frac{I(x) - B\,(1 - t(x))}{t(x)}
$$

This is **the analytic inversion** at the heart of the network:
the model only learns `(t, B)`; `J` falls out for free.

The denominator `t(x)` makes the inversion ill-conditioned when
`t → 0` (a totally occluded ray). Numerically we replace `t` with
`max(t, ε)` where `ε = 10⁻³`. The corresponding **inverse operator**
implemented in code:

$$
\hat{J}(x) \;=\; \mathrm{clip}\!\left( \frac{I(x) - B\,(1 - t(x))}{\max(t(x), \varepsilon)},\; 0,\; 1 \right)
$$

```python
def invert_jaffe_mcglamery(i, t, b, *, eps=1.0e-3):
    if b.dim() == 2:
        b = b.unsqueeze(-1).unsqueeze(-1)
    t_safe = t.clamp(min=eps)
    j = (i - b * (1.0 - t)) / t_safe
    return j.clamp(0.0, 1.0)
```

### 3.2.6 Round-trip consistency

If `(t, B)` are exact, then by construction the forward followed by
the inverse recovers `J` up to numerical noise:

$$
\mathrm{invert}\bigl(\mathrm{forward}(J, t, B), t, B\bigr) \;\approx\; J
$$

This is verified as a unit test (see
[`tests/test_physics_loss.py`](../../tests/test_physics_loss.py),
`test_physics_round_trip_is_identity_on_clean_inputs`). The
tolerance is `2e-3` accounting for the `ε` clamp; in regions where
`t > ε` the recovery is bit-exact in FP32.

## 3.3 The loss function

The composite training loss has five terms:

$$
\mathcal{L}_{\text{total}} \;=\; \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} \;+\; \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} \;+\; \lambda_{\text{ssim}} \mathcal{L}_{\text{ssim}} \;+\; \lambda_{\text{tv}} \mathcal{L}_{\text{tv}} \;+\; \lambda_t \mathcal{L}_t
$$

with default weights `(1.0, 0.5, 0.5, 0.01, 0.5)`. Each term is
defined below.

### 3.3.1 Reconstruction loss `L_rec`

We use the **Charbonnier loss** (a smooth approximation of L1):

$$
\mathcal{L}_{\text{rec}} \;=\; \mathbb{E}_x\!\left[\sqrt{(\hat{J}(x) - J(x))^2 + \beta^2}\,\right]
$$

with `β = 10⁻³`. As `β → 0` Charbonnier → L1; as `β → ∞`
Charbonnier → L2. The intermediate behaviour is:

| Regime | Behaviour |
| --- | --- |
| Big residuals (`|d| ≫ β`) | Linear (≈ L1): robust to outliers like specular reflections, fish scales |
| Small residuals (`|d| ≪ β`) | Quadratic (≈ L2): smooth gradient at zero, so late-stage fine-tuning is well-behaved |

**Why not plain L1?** L1 has a discontinuous derivative at zero,
which creates noisy gradients in the late stages of training and
empirically slows convergence by ~5–10 epochs on MSRB.

**Why not plain MSE?** MSE quadratically penalises outliers.
Underwater residuals are heavy-tailed (occasional bright sparkles,
mooring lines); MSE responds by blurring the whole frame to satisfy
those outliers.

### 3.3.2 Forward-physics consistency `L_phys`

This is the term that actually enforces the physics:

$$
\mathcal{L}_{\text{phys}} \;=\; \mathbb{E}_x\!\left[\sqrt{(I(x) - F(J, \hat{t}, \hat{B})(x))^2 + \beta^2}\,\right]
$$

— Charbonnier between the **observed image** `I` and the forward
model evaluated at the **ground-truth `J`** but the **predicted
`(t̂, B̂)`**. If the predicted `(t̂, B̂)` are correct, this term
is zero by definition. If the network finds an `(t̂, B̂)` pair
that happens to give the right `J` via inversion but is *not*
physically the right pair, this term penalises it.

> **Why this term is the cornerstone of the method.**
> Without `L_phys`, `(t, B)` are only weakly identifiable: many
> different factorisations of `(I, J)` give acceptable
> reconstruction loss. With `L_phys` enabled, `(t, B)` are
> **jointly anchored** to the unique factorisation consistent with
> the underlying physics. This is the *soft* physics-informed
> constraint in the PINN sense [Raissi 2019].

### 3.3.3 Structural similarity `L_ssim`

Pixel-wise reconstruction losses (L1, L2, Charbonnier) are blind
to perceptual structure — they cannot distinguish a uniform
brightness shift from a destroyed edge. SSIM, the *structural
similarity index* [Wang 2004], measures local mean, variance, and
covariance similarity in a sliding Gaussian window:

$$
\mathrm{SSIM}(x, y) \;=\; \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

with `μ`, `σ²`, `σ_xy` computed in an 11×11 Gaussian window with
`σ = 1.5`, and `C_1 = (0.01)² = 10⁻⁴`, `C_2 = (0.03)² = 9·10⁻⁴`
for data range 1.0. SSIM ∈ [-1, 1]; 1 means identical. The loss is

$$
\mathcal{L}_{\text{ssim}} \;=\; 1 - \mathrm{SSIM}(\hat{J}, J)
$$

We re-implemented SSIM ourselves (see
[`src/aquaclr/losses/ssim.py`](../../src/aquaclr/losses/ssim.py))
rather than using `pytorch-msssim` because:

1. Avoiding the external dependency reduces the ONNX export
   footprint.
2. Our implementation is `torch.compile`-clean (no Python branches
   on tensor values).
3. The Gaussian window is registered as a non-persistent buffer
   so `module.to(...)` moves it correctly.

### 3.3.4 Anisotropic total variation `L_tv`

Without a smoothness prior, the transmission head will happily
output salt-and-pepper noise that minimises the data terms. We
penalise high-frequency content in `t` via an *anisotropic L1
total variation*:

$$
\mathcal{L}_{\text{tv}} \;=\; \mathbb{E}\!\left[ \bigl|t_{i, j+1} - t_{i, j}\bigr| \,+\, \bigl|t_{i+1, j} - t_{i, j}\bigr| \right]
$$

where the expectation is over all valid `(i, j)`. **Anisotropic**
means the `x` and `y` discrete differences are summed as L1 rather
than combined under a square root (the *isotropic* form). Two
consequences:

1. Sharp at edges: anisotropic TV has a vertical "wall" gradient
   at axis-aligned discontinuities, which lets the optimiser keep
   them sharp. Isotropic TV smears them.
2. Cheaper: no `sqrt`.

We weight `L_tv` lightly (`λ_tv = 10⁻²`) because a too-large weight
washes the transmission map to a constant (which trivially
minimises TV).

### 3.3.5 Optional direct `t` supervision `L_t`

When the dataloader provides a ground-truth transmission map
(LSUI), we add:

$$
\mathcal{L}_t \;=\; \bigl\|\hat{t} - t_{\text{gt}}\bigr\|_1
$$

The loss module (see
[`src/aquaclr/losses/physics_loss.py`](../../src/aquaclr/losses/physics_loss.py))
checks the per-batch flag `has_t_gt` and returns `0` for that term
on MSRB batches, where there is no GT.

> **Why the soft weight `λ_t = 0.5`?**
> LSUI's transmission GT is itself **estimated** by classical
> dehazing algorithms, not measured. Treating it as a hard target
> (e.g. `λ_t = 1.0`) over-fits the network to the noisy reference.
> Empirically `0.5` gave the best PSNR on UIEB-Challenge in our
> ablation (Ch. 10).

## 3.4 Why this is "physics-informed" in the formal sense

PINNs in the strict sense [Raissi 2019] require that the loss
explicitly penalises violation of a known physical equation. Our
`L_phys` term does exactly that: at the predicted `(t̂, B̂)` and
the GT `J`, the equation `I = F(J, t̂, B̂)` should hold; deviation
is penalised.

A *complementary*, hard-coded constraint is the **structural**
choice that the network's `J` output is computed analytically by
inverting the physics, not regressed. This is sometimes called a
"Theory-trained" or "model-embedded" approach.

Combining both, LEGION-DeSnow-S has:

1. A **hard physical constraint** (the architecture's
   `forward → invert` pipeline) that excludes physically
   impossible factorisations from the hypothesis space entirely.
2. A **soft physical constraint** (`L_phys`) that further pushes
   the joint `(t̂, B̂)` toward the correct factorisation.

This is the strongest form of physics-informedness short of a
fully PDE-trained PINN, and is the right formulation for
underwater image restoration.

## 3.5 Identifiability and gauge symmetries

Without further constraints, the equation `I = J·t + B(1 − t)` is
**under-determined** — multiple `(J, t, B)` triples reproduce the
same `I`. We discuss the ones that matter.

### 3.5.1 Brightness gauge

Multiplying `J` and dividing `t` by the same scalar `α` gives the
same `I` only if we also rescale `B`. With `J ∈ [0, 1]` and
`t ∈ (0, 1)` clamping, the gauge becomes **partially broken** —
the clamps effectively normalise `J` and `t` to plausible ranges.

### 3.5.2 Backscatter-vs-bright-pixel ambiguity

A scene patch that is both bright and at long range is
indistinguishable from a near-range patch with high backscatter,
absent global context. The **global** averaging in the
backscatter head (one `B` per image, not per pixel) breaks this
ambiguity by amortising over the whole frame; the
forward-consistency loss further pushes `(t, B)` to the unique
joint solution.

### 3.5.3 Particle-vs-edge ambiguity

A small bright cluster can be either a marine-snow particle or a
real scene feature (fish scale, coral nub, mooring chain). Without
external data, this ambiguity cannot be resolved. The training set
has to provide examples of both; this is precisely why we mix
MSRB (snow) with LSUI (real underwater scenes with fauna).

## 3.6 Numerical considerations

| Concern | Source | Mitigation |
| --- | --- | --- |
| Division by `t → 0` | inversion | `t.clamp(min=1e-3)` |
| Loss landscape sensitivity at `t ≈ 1` | sigmoid saturation | bias init `+2.0` on the t-head's projection (sigmoid(2)=0.88, plausible "mostly clear" prior) |
| Gradient explosion via `1/t` | inversion derivative ~ 1/t² | gradient-clipping at `1.0` (norm) at the trainer level |
| FP16 underflow on `(1 - t)` near 1 | mixed precision | use BF16 instead of FP16 at training time (Ampere supports both) |
| ONNX exporter rejects dataclass return | export | provide `forward_export()` returning a plain tuple |

## 3.7 Worked example — single-pixel forward and inverse

Take a single pixel with:

- `J = [0.6, 0.5, 0.4]` (a beige rock)
- `t = 0.7` (medium clear)
- `B = [0.1, 0.3, 0.5]` (blue-green water tint)

Forward:

$$
\begin{align*}
I_R &= 0.6 \cdot 0.7 + 0.1 \cdot (1-0.7) = 0.42 + 0.03 = 0.45\\
I_G &= 0.5 \cdot 0.7 + 0.3 \cdot 0.3 = 0.35 + 0.09 = 0.44\\
I_B &= 0.4 \cdot 0.7 + 0.5 \cdot 0.3 = 0.28 + 0.15 = 0.43
\end{align*}
$$

Inverse (assuming the network predicts `(t̂, B̂) = (t, B)`):

$$
\begin{align*}
\hat{J}_R &= \frac{0.45 - 0.1 \cdot 0.3}{0.7} = \frac{0.42}{0.7} = 0.60\\
\hat{J}_G &= \frac{0.44 - 0.3 \cdot 0.3}{0.7} = \frac{0.35}{0.7} = 0.50\\
\hat{J}_B &= \frac{0.43 - 0.5 \cdot 0.3}{0.7} = \frac{0.28}{0.7} = 0.40
\end{align*}
$$

Recovers `J` exactly. The arithmetic is intentionally trivial —
that's the point of physics-informed methods. The hard problem is
predicting `(t̂, B̂)` correctly across the whole frame.

## 3.8 Loss-component sensitivity (theoretical)

What does each loss term *do* mathematically when minimised in
isolation?

| Term | Minimised when | Risk if used alone |
| --- | --- | --- |
| `L_rec` | `Ĵ = J` everywhere | `(t̂, B̂)` unconstrained — multiple solutions |
| `L_phys` | `F(J, t̂, B̂) = I` | `Ĵ` is unconstrained except via inversion; numerical noise can dominate |
| `L_ssim` | `Ĵ` and `J` have matching local statistics | Ignores absolute colour — could still have a global tint |
| `L_tv` | `t̂` constant | The trivial `t = const` solution destroys spatial information |
| `L_t` | `t̂ = t_gt` | Only on LSUI batches; not available on MSRB |

A well-chosen weighted combination (our `λ` defaults) makes the
joint optimum unique and physically meaningful. The chosen
weights are themselves the result of the ablation study described
in Chapter 10.

## 3.9 Why an analytic inversion instead of a learned `J`-head?

It would be perfectly possible to add a third head that regresses
`J` directly. Why don't we?

| Choice | Pros | Cons |
| --- | --- | --- |
| **Analytic inversion (ours)** | Zero parameters, exact at training time, ONNX-clean, exposes `(t, B)` to downstream | Sensitive to small `t`; bounded by the simplified model's accuracy |
| **Learned `J`-head** | Robust to model misspecification | Doubles parameter count; reintroduces hallucination; obscures `(t, B)` |
| **Hybrid (analytic + residual `J`-head)** | Best of both | More parameters; harder to verify |

We choose the analytic inversion because the simplified
Jaffe-McGlamery model has a *much* smaller misspecification gap
than, say, atmospheric scattering does at long range, and because
the ONNX export simplifies dramatically. A hybrid would be a
sensible M2 ablation if model misspecification turns out to be a
ceiling.

---

## Key takeaways

- The simplified Jaffe-McGlamery model
  `I(x) = J(x)·t(x) + B(1 − t(x))` is the algebraic foundation of
  the work. It is structurally identical to the Koschmieder
  atmospheric scattering model.
- The model is invertible analytically as
  `J = (I − B(1 − t)) / clamp(t, ε)`. We bake this directly into
  the network's forward pass.
- The composite loss combines reconstruction (Charbonnier),
  forward-physics consistency, SSIM, anisotropic TV on `t`, and
  optional `t` supervision. Each term has a clear physical or
  perceptual rationale.
- The forward-consistency term is what makes the architecture
  formally physics-informed in the PINN sense.
- Identifiability ambiguities (gauge, particle-vs-edge) are
  resolved by data + global pooling + the physics-consistency
  loss.

## Cross-references

- Forward to [Chapter 4 — System Architecture](04_architecture.md)
- Math derivation appendix: [Appendix A](A_math.md)
- Code: [`src/aquaclr/utils/physics.py`](../../src/aquaclr/utils/physics.py),
  [`src/aquaclr/losses/`](../../src/aquaclr/losses/)
