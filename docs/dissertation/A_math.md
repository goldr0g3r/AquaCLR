# Appendix A — Mathematical Derivations

This appendix expands every equation in the main text to step-by-step
form so an examiner can verify each manipulation independently. Notation
follows §3.1.

## A.1 Beer-Lambert transmission

For a homogeneous medium with attenuation coefficient `c(λ)` (m⁻¹), the
fraction of photons surviving a path of length `d` is given by the
differential equation:

$$
\frac{d\Phi(z)}{dz} = -c(\lambda) \, \Phi(z)
$$

Integrating from `0` to `d`:

$$
\Phi(d) = \Phi(0) \exp\!\bigl(-c(\lambda)\,d\bigr)
$$

Defining `t(x) = Φ(d(x)) / Φ(0)`:

$$
\boxed{ t(x) = \exp\!\bigl(-c(\lambda)\,d(x)\bigr) \in [0, 1] }
$$

This is the **direct-transmission factor** that multiplies the scene
radiance `J(x)` at the sensor.

## A.2 Backscatter integral

A small slice `dz` along the ray, illuminated by ambient `B`,
contributes `b(λ) B \, dz` of scattered photons (where `b(λ)` is the
volume-scattering coefficient in the camera direction). Each
contribution is itself attenuated by the path between the slice and the
camera, `e^{-c(λ) z}`. Total:

$$
B_{\text{at camera}}(x) = \int_0^{d(x)} b(\lambda)\,B \, e^{-c(\lambda) z}\,dz
                        = \frac{b(\lambda)}{c(\lambda)} B \, \bigl(1 - e^{-c(\lambda) d(x)}\bigr)
$$

The single-scattering-albedo simplification absorbs the prefactor
`b(λ)/c(λ)` into `B`, leaving:

$$
\boxed{ B_{\text{at camera}}(x) = B \, \bigl(1 - t(x)\bigr) }
$$

## A.3 Composing direct + ambient — Jaffe-McGlamery

Adding §A.1 and §A.2:

$$
\boxed{ I(x) = J(x)\,t(x) + B\,(1 - t(x)) }
$$

This is the form used throughout the dissertation (Eq. 3.1).

## A.4 Algebraic inverse

Solving for `J(x)`:

$$
\begin{align*}
I(x) - B(1 - t(x)) &= J(x)\,t(x) \\[2pt]
\frac{I(x) - B(1 - t(x))}{t(x)} &= J(x)
\end{align*}
$$

Stable form with the `ε` clamp:

$$
\boxed{ \hat{J}(x) = \mathrm{clip}\!\left( \frac{I(x) - \hat{B}(1 - \hat{t}(x))}{\max(\hat{t}(x), \varepsilon)},\;0,\;1\right) }
$$

## A.5 Charbonnier loss derivative

For `ψ(d) = √(d² + β²)`:

$$
\frac{\partial \psi}{\partial d} = \frac{d}{\sqrt{d² + β²}}
$$

This is bounded in `[-1, 1]`, vanishes as `d → 0` (smooth at zero,
unlike L1), and saturates linearly for `|d| ≫ β` (robust to outliers,
unlike L2).

## A.6 Anisotropic TV gradient

For `T(t) = mean(|∂_x t| + |∂_y t|)` with discrete differences:

$$
\partial_x t_{i,j} = t_{i, j+1} - t_{i, j}, \qquad
\partial_y t_{i,j} = t_{i+1, j} - t_{i, j}
$$

The subgradient with respect to a single pixel `t_{i, j}` accumulates
contributions from the four discrete differences in which it appears.
PyTorch's autograd handles this automatically.

## A.7 SSIM — full statistic derivation

For a window centred at `(i, j)` weighted by Gaussian `w(u, v)`:

$$
\begin{align*}
\mu_x &= \sum_{u, v} w(u, v) \, x(i+u, j+v) \\
\sigma_x^2 &= \sum_{u, v} w(u, v) \, (x(i+u, j+v) - \mu_x)^2 \\
\sigma_{xy} &= \sum_{u, v} w(u, v) \, (x(i+u, j+v) - \mu_x)(y(i+u, j+v) - \mu_y)
\end{align*}
$$

These are computed efficiently as group-conv2d operations with a
separable Gaussian kernel — see [`src/aquaclr/losses/ssim.py`](../../src/aquaclr/losses/ssim.py).

The SSIM constants:

$$
C_1 = (K_1 \, L)^2 = (0.01 \cdot 1.0)^2 = 10^{-4}, \qquad
C_2 = (K_2 \, L)^2 = (0.03 \cdot 1.0)^2 = 9 \times 10^{-4}
$$

`L = 1.0` for `[0, 1]` data range.

## A.8 Total composite loss expansion

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \mathbb{E}\!\left[\sqrt{(\hat{J}-J)^2 + \beta^2}\right]
+ \lambda_{\text{phys}} \mathbb{E}\!\left[\sqrt{(I - F(J, \hat{t}, \hat{B}))^2 + \beta^2}\right]
+ \lambda_{\text{ssim}}\bigl(1 - \mathrm{SSIM}(\hat{J}, J)\bigr)
+ \lambda_{\text{tv}} \mathbb{E}\!\left[|\partial_x \hat{t}| + |\partial_y \hat{t}|\right]
+ \lambda_t \, \mathbb{E}\!\left[\bigl|\hat{t} - t_{\text{gt}}\bigr|\right]
$$

with default weights `(1.0, 0.5, 0.5, 10⁻², 0.5)`. The last term is
gated by the per-batch `has_t_gt` flag.

## A.9 Inversion gradient analysis

For `Ĵ = (I - B(1 - t)) / t` (pre-clamp):

$$
\frac{\partial \hat{J}}{\partial t} = \frac{1}{t^2} \cdot \bigl(I - B\bigr), \qquad
\frac{\partial \hat{J}}{\partial B} = -\frac{1 - t}{t}
$$

The `1/t²` term is the source of gradient instability near `t → 0`.
With `ε = 10⁻³` clamping and a global gradient-norm clip at `1.0`, the
training loss remains finite throughout the `(0, 1)` range of `t`.

## A.10 Memory budget — encoder activation sizes

For input `(B, 3, H, W)`:

| Stage | Channels | Spatial | Activation tensor count (FP16 bytes) |
| --- | --- | --- | --- |
| stem | 16 | H/2 × W/2 | `B · 16 · H/2 · W/2 · 2` |
| /4 | 16 | H/4 × W/4 | `B · 16 · H/4 · W/4 · 2` |
| /8 | 24 | H/8 × W/8 | `B · 24 · H/8 · W/8 · 2` |
| /16 | 48 | H/16 × W/16 | `B · 48 · H/16 · W/16 · 2` |
| /32 | 96 | H/32 × W/32 | `B · 96 · H/32 · W/32 · 2` |

For `H = 720, W = 1280, B = 1`: total ≈ 2.7 MB encoder activations
(FP16). Decoder peaks ≈ 7 MB.

## A.11 SSIM identity at `x = y`

When `x ≡ y`:

- `μ_x = μ_y` ⇒ `2 μ_x μ_y = μ_x² + μ_y²`,
- `σ_x² = σ_y² = σ_{xy}` ⇒ `2 σ_{xy} = σ_x² + σ_y²`.

Therefore `SSIM = 1` exactly. This is the basis of the unit test in
[`tests/test_ssim_tv.py::test_ssim_identity_is_one`](../../tests/test_ssim_tv.py).

## A.12 PSNR units

Defined for normalised `[0, 1]` images:

$$
\mathrm{PSNR} = 10 \log_{10} \frac{1}{\mathrm{MSE}}
$$

For 8-bit `[0, 255]` images: `1` becomes `255²`, adding
`10 \log_{10}(255²) ≈ 48.13` dB to the value. Throughout this work we
report on `[0, 1]` data — the dataset loaders normalise once, before
any metric.
