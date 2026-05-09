"""Anisotropic total variation regulariser.

Total variation (TV) penalises high-frequency content in a tensor and
is the standard prior for "this map should be piecewise smooth". It is
applied to the predicted transmission map ``t(x)`` so that the network
does not paint random per-pixel values that happen to satisfy the
reconstruction loss but have no plausible physical meaning.

We use the **anisotropic** L1 form because:

* it is sharper at depth/material discontinuities (real underwater
  scenes have hard depth jumps at rocks, fauna, etc.),
* it has a well-conditioned subgradient at the discontinuities, and
* it is cheaper than the isotropic ``sqrt(dx^2 + dy^2)`` form.

Automotive SiL parallel:
    Identical to the smoothness prior used on lidar/depth completion
    networks in ADAS to keep depth maps from looking like swiss
    cheese.
"""

from __future__ import annotations

from torch import Tensor


def total_variation(x: Tensor, *, reduction: str = "mean") -> Tensor:
    """Anisotropic total variation.

    Args:
        x: Tensor of shape ``(B, C, H, W)``. Typically the transmission map.
        reduction: ``"mean"`` (default) or ``"sum"``.

    Returns:
        Scalar TV value.
    """
    if x.dim() != 4:
        msg = f"total_variation expects a 4D tensor, got shape {tuple(x.shape)}"
        raise ValueError(msg)

    dx = (x[..., :, 1:] - x[..., :, :-1]).abs()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs()

    if reduction == "mean":
        return dx.mean() + dy.mean()
    if reduction == "sum":
        return dx.sum() + dy.sum()
    msg = f"Unknown reduction {reduction!r}; expected 'mean' or 'sum'"
    raise ValueError(msg)
