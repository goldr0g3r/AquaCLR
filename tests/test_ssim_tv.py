"""SSIM + TV unit tests."""

from __future__ import annotations

import torch

from aquaclr.losses.ssim import SSIM, ssim
from aquaclr.losses.tv import total_variation


def test_ssim_identity_is_one() -> None:
    x = torch.rand(2, 3, 64, 64)
    assert ssim(x, x).item() == 1.0
    module = SSIM()
    assert module(x, x).item() == 1.0


def test_ssim_decreases_with_noise() -> None:
    x = torch.rand(1, 3, 64, 64)
    noise = torch.randn_like(x) * 0.5
    y = (x + noise).clamp(0, 1)
    assert ssim(x, y).item() < ssim(x, x).item()


def test_total_variation_zero_on_constant() -> None:
    x = torch.full((1, 1, 16, 16), 0.5)
    assert total_variation(x).item() == 0.0


def test_total_variation_positive_on_random() -> None:
    x = torch.rand(1, 1, 16, 16)
    assert total_variation(x).item() > 0.0
