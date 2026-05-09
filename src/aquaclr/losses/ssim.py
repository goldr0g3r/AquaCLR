"""Differentiable SSIM with a fixed Gaussian window.

This is a self-contained re-implementation of the standard SSIM
(Wang et al., 2004) that:

* avoids a runtime dependency on ``pytorch-msssim``;
* is fully autograd-compatible; and
* is friendly to ``torch.compile`` and ONNX tracing (no Python branching
  on tensor values).

Automotive SiL parallel:
    SSIM is the canonical structural-fidelity metric used in automotive
    perception to ensure that a denoising stage doesn't blur away
    safety-critical edges (lane markings, brake lights). For us, the
    safety-critical edges are coral textures and rocks that downstream
    SLAM uses as feature points.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gaussian_kernel_1d(window_size: int, sigma: float) -> Tensor:
    """Return a 1D Gaussian kernel of length ``window_size`` with given ``sigma``."""
    half = (window_size - 1) / 2.0
    coords = torch.arange(window_size, dtype=torch.float32) - half
    kernel = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_window(window_size: int, channels: int, sigma: float) -> Tensor:
    """Build a separable 2D Gaussian kernel as a depthwise conv weight tensor.

    Returned shape: ``(channels, 1, window_size, window_size)``.
    """
    k1 = _gaussian_kernel_1d(window_size, sigma)
    k2 = k1.unsqueeze(0) * k1.unsqueeze(1)
    k2 = k2.unsqueeze(0).unsqueeze(0)
    return k2.repeat(channels, 1, 1, 1)


def ssim(
    x: Tensor,
    y: Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Compute SSIM between two image batches.

    Args:
        x: First image batch, shape ``(B, C, H, W)``.
        y: Second image batch, same shape as ``x``.
        window_size: Size of the Gaussian window. Must be odd.
        sigma: Standard deviation of the Gaussian window.
        data_range: Dynamic range of the input (1.0 for ``[0, 1]`` images).
        reduction: ``"mean"`` returns a scalar, ``"none"`` returns
            ``(B, C, H', W')``.

    Returns:
        SSIM value(s) in ``[-1, 1]``; 1.0 means perfect match.
    """
    if x.shape != y.shape:
        msg = f"SSIM input shapes must match, got {tuple(x.shape)} vs {tuple(y.shape)}"
        raise ValueError(msg)
    if window_size % 2 == 0:
        msg = f"window_size must be odd, got {window_size}"
        raise ValueError(msg)

    channels = x.shape[1]
    weight = _gaussian_window(window_size, channels, sigma).to(device=x.device, dtype=x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, weight, padding=pad, groups=channels)
    mu_y = F.conv2d(y, weight, padding=pad, groups=channels)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, weight, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, weight, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, weight, padding=pad, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numer = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denom = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numer / denom

    if reduction == "mean":
        return ssim_map.mean()
    if reduction == "none":
        return ssim_map
    msg = f"Unknown reduction {reduction!r}; expected 'mean' or 'none'"
    raise ValueError(msg)


class SSIM(nn.Module):
    """Stateful wrapper around :func:`ssim`.

    Pre-computes and registers the Gaussian window as a non-parameter
    buffer so the kernel is moved to the right device / dtype with
    ``module.to(...)`` and survives ``torch.compile``.
    """

    weight: Tensor

    def __init__(
        self,
        *,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 3,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        if window_size % 2 == 0:
            msg = f"window_size must be odd, got {window_size}"
            raise ValueError(msg)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.channels = int(channels)
        self.data_range = float(data_range)
        weight = _gaussian_window(self.window_size, self.channels, self.sigma)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Return scalar SSIM in ``[-1, 1]`` (1.0 == identical)."""
        if x.shape[1] != self.channels:
            return ssim(
                x,
                y,
                window_size=self.window_size,
                sigma=self.sigma,
                data_range=self.data_range,
            )
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        pad = self.window_size // 2
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        mu_x = F.conv2d(x, weight, padding=pad, groups=self.channels)
        mu_y = F.conv2d(y, weight, padding=pad, groups=self.channels)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x_sq = F.conv2d(x * x, weight, padding=pad, groups=self.channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, weight, padding=pad, groups=self.channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, weight, padding=pad, groups=self.channels) - mu_xy

        numer = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denom = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        return (numer / denom).mean()


def psnr(x: Tensor, y: Tensor, *, data_range: float = 1.0, eps: float = 1.0e-10) -> Tensor:
    """Peak signal-to-noise ratio in dB. Useful for monitoring; not for training."""
    mse = F.mse_loss(x, y, reduction="mean").clamp(min=eps)
    return 10.0 * torch.log10(torch.tensor(data_range, device=x.device, dtype=x.dtype) ** 2 / mse)


# Re-export so users can call ``ssim.PSNR_LOG2_10`` etc. without magic numbers.
LOG10_OF_2 = math.log10(2.0)
