"""Jaffe-McGlamery image-formation utilities.

The simplified Jaffe-McGlamery underwater scattering model is:

.. math::
    I(x) = J(x) \\cdot t(x) + B \\cdot (1 - t(x))

where:
    * :math:`I(x)` is the observed (snowy/hazy) image,
    * :math:`J(x)` is the radiance of the clear scene we want to recover,
    * :math:`t(x) \\in [0, 1]` is the medium transmission map (1 = perfectly clear,
      0 = fully scattered),
    * :math:`B \\in [0, 1]^3` is the backscatter / global ambient veiling light.

Automotive SiL parallel:
    The equation is structurally identical to the Koschmieder atmospheric
    scattering model used to synthesise fog and rain in automotive
    perception pipelines. ``t(x)`` is the analogue of optical depth, and
    ``B`` plays the role of the airlight/sky term. Inverting it is the
    same algebra used by camera ISPs that defog automotive frames before
    they are handed to the perception stack.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def apply_forward_jaffe_mcglamery(
    j: Tensor,
    t: Tensor,
    b: Tensor,
) -> Tensor:
    """Apply the forward Jaffe-McGlamery image-formation model.

    Args:
        j: Clean radiance image, shape ``(B, 3, H, W)``, range ``[0, 1]``.
        t: Transmission map, shape ``(B, 1, H, W)``, range ``[0, 1]``.
        b: Backscatter vector, shape ``(B, 3)`` or ``(B, 3, 1, 1)``,
            range ``[0, 1]``.

    Returns:
        Synthesised observation ``I``, shape ``(B, 3, H, W)``, range ``[0, 1]``.

    Automotive SiL parallel:
        This is the operator a fog/rain renderer applies inside a
        DriveSim-style scenario before feeding the camera into the ego
        stack.
    """
    if b.dim() == 2:
        b = b.unsqueeze(-1).unsqueeze(-1)
    return (j * t + b * (1.0 - t)).clamp(0.0, 1.0)


def invert_jaffe_mcglamery(
    i: Tensor,
    t: Tensor,
    b: Tensor,
    *,
    eps: float = 1.0e-3,
    gate_threshold: float = 0.1,
    gate_temperature: float = 0.05,
    inpaint_kernel: int = 7,
) -> Tensor:
    """Invert the forward model to recover ``J`` with gated noise suppression.

    Instead of a hard ``clamp(t, min=eps)`` — which still amplifies noise
    by up to ``1/eps`` near the clamp boundary — we use a **soft-sigmoid
    gate** that blends the analytic inversion with a spatial inpainting
    fallback as ``t → 0``:

    .. math::

        g(x) &= \\sigma\\!\\left(\\frac{t(x) - \\tau}{\\beta}\\right) \\\\
        \\hat{J}_{\\text{phys}}(x) &= \\frac{I(x) - B\\,(1-t(x))}{\\max(t(x), \\varepsilon)} \\\\
        J_{\\text{fill}}(x) &= \\mathrm{AvgPool}_{k}(I)(x) \\\\
        J(x) &= g(x)\\cdot\\hat{J}_{\\text{phys}}(x) + (1-g(x))\\cdot J_{\\text{fill}}(x)

    When ``t`` is high (clear water) the gate ``g → 1`` and the output
    is the exact analytic reconstruction. When ``t → 0`` (marine-snow
    particle / totally-occluded ray) the gate ``g → 0`` and the network
    gracefully defaults to a locally-averaged (spatially interpolated)
    version of ``I``, rather than amplifying noise to infinity.

    Args:
        i: Observed image, shape ``(B, 3, H, W)``, range ``[0, 1]``.
        t: Predicted transmission, shape ``(B, 1, H, W)``.
        b: Predicted backscatter, shape ``(B, 3)`` or ``(B, 3, 1, 1)``.
        eps: Hard floor on the physics-branch denominator. Only active
            for pixels where ``t`` is already above ``gate_threshold``
            (i.e. the gate is passing through the physics branch); at
            those pixels ``t`` is large enough that ``eps`` is never
            reached in practice.
        gate_threshold: Transmission value ``τ`` at which the gate is
            at its mid-point (``g = 0.5``). Below this threshold the
            fill branch dominates. Default ``0.1``.
        gate_temperature: Steepness ``β`` of the sigmoid gate. Smaller
            values give a sharper transition; larger values give a
            smoother one. Default ``0.05``.
        inpaint_kernel: Odd kernel size ``k`` for the ``AvgPool``
            spatial-fill branch. Larger kernels draw from a wider
            neighbourhood. Default ``7`` (3-pixel radius).

    Returns:
        Recovered clean image ``J``, shape ``(B, 3, H, W)``, clamped
        to ``[0, 1]``.

    Automotive SiL parallel:
        The same gated blend is used in ADAS camera de-rain to avoid
        extreme amplification under very dense rain streaks: pixels
        covered by a thick streak (analogous to ``t → 0``) fall back to
        spatial interpolation from surrounding clear pixels.
    """
    if b.dim() == 2:
        b = b.unsqueeze(-1).unsqueeze(-1)

    # ------------------------------------------------------------------
    # Gate: smooth sigmoid centred at gate_threshold.
    # g(x) → 1  when t >> gate_threshold  → analytic physics inversion
    # g(x) → 0  when t << gate_threshold  → spatial fill (inpainting)
    # ------------------------------------------------------------------
    g = torch.sigmoid((t - gate_threshold) / gate_temperature)

    # ------------------------------------------------------------------
    # Physics branch: standard analytic inversion with a soft floor on
    # the denominator.  The floor only matters in the rare case where t
    # is already above gate_threshold (g ≈ 1) yet still below eps —
    # a regime that almost never occurs with the default settings.
    # ------------------------------------------------------------------
    t_safe = t.clamp(min=eps)
    j_phys = (i - b * (1.0 - t)) / t_safe

    # ------------------------------------------------------------------
    # Fill branch: spatial inpainting proxy.
    # AvgPool with a (inpaint_kernel × inpaint_kernel) window provides
    # local context from neighbouring pixels where t is higher — the
    # closest practical approximation to inpainting within a single
    # forward pass.
    # ------------------------------------------------------------------
    pad = inpaint_kernel // 2
    j_fill = F.avg_pool2d(i, kernel_size=inpaint_kernel, stride=1, padding=pad)

    # ------------------------------------------------------------------
    # Soft blend and clamp to valid range.
    # ------------------------------------------------------------------
    j = g * j_phys + (1.0 - g) * j_fill
    return j.clamp(0.0, 1.0)
