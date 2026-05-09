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
) -> Tensor:
    """Algebraically invert the forward model to recover ``J``.

    Args:
        i: Observed image, shape ``(B, 3, H, W)``, range ``[0, 1]``.
        t: Predicted transmission, shape ``(B, 1, H, W)``.
        b: Predicted backscatter, shape ``(B, 3)`` or ``(B, 3, 1, 1)``.
        eps: Floor on the transmission denominator. The Jaffe-McGlamery
            equation becomes singular at ``t -> 0`` (totally occluded
            ray); we clamp it to keep gradients finite.

    Returns:
        Recovered clean image ``J``, shape ``(B, 3, H, W)``, clamped to ``[0, 1]``.
    """
    if b.dim() == 2:
        b = b.unsqueeze(-1).unsqueeze(-1)
    t_safe = t.clamp(min=eps)
    j = (i - b * (1.0 - t)) / t_safe
    return j.clamp(0.0, 1.0)
