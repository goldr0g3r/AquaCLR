"""Loss functions for AquaCLR."""

from __future__ import annotations

from aquaclr.losses.physics_loss import PhysicsInformedLoss, PhysicsLossOutputs
from aquaclr.losses.ssim import SSIM, ssim
from aquaclr.losses.tv import total_variation

__all__ = [
    "SSIM",
    "PhysicsInformedLoss",
    "PhysicsLossOutputs",
    "ssim",
    "total_variation",
]
