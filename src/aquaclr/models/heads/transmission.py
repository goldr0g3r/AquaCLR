"""Transmission map head ``t(x)``.

Predicts the per-pixel medium transmission in ``[0, 1]``. The head is
deliberately tiny (a single 1x1 conv) because all the heavy lifting is
done by the shared decoder; this keeps the parameter budget on a tight
leash.

Automotive SiL parallel:
    The transmission map is the underwater equivalent of the per-ray
    optical-depth field a fog renderer produces. Downstream SLAM /
    perception modules can use it as a confidence prior in exactly the
    same way an ADAS stack uses lidar return-intensity to flag uncertain
    detections in heavy rain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TransmissionHead(nn.Module):
    """Final 1x1 conv -> sigmoid producing ``t(x)`` at full input resolution.

    Args:
        in_channels: Channels coming out of the decoder's highest-res feature.
        target_size_factor: How much to upsample the decoder output to reach
            the input resolution. With our decoder ending at stride /2,
            this is 2.
    """

    def __init__(self, in_channels: int, *, target_size_factor: int = 2) -> None:
        super().__init__()
        self.target_size_factor = target_size_factor
        self.proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        # Bias init: sigmoid(2.0) ~= 0.88, a sensible "mostly clear" prior.
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 2.0)

    def forward(self, feat: Tensor, *, target_size: tuple[int, int] | None = None) -> Tensor:
        """Predict the transmission map.

        Args:
            feat: Decoder feature map.
            target_size: Optional explicit ``(H, W)`` to upsample to. When
                given, this overrides ``target_size_factor``.

        Returns:
            ``t``, shape ``(B, 1, H, W)``, range ``(0, 1)``.
        """
        x = self.proj(feat)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        elif self.target_size_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=float(self.target_size_factor),
                mode="bilinear",
                align_corners=False,
            )
        return torch.sigmoid(x)
