"""Global backscatter / ambient light head ``B``.

The backscatter is modelled as a single 3-vector per image (one
per-channel value), which is consistent with the simplified
Jaffe-McGlamery model in which ``B`` is treated as scene-global ambient
light. We pool the deepest encoder feature and run a tiny MLP.

Automotive SiL parallel:
    ``B`` is analogous to the airlight constant in atmospheric
    scattering, or to the DC offset of a camera under heavy fog. ADAS
    pipelines estimate it once per frame for the same reason: it
    factorises the scene-global colour cast away from the fine-grained
    transmission geometry.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BackscatterHead(nn.Module):
    """Global average pool -> 2-layer MLP -> sigmoid -> 3-vector.

    Args:
        in_channels: Channels of the deepest encoder feature.
        hidden: Hidden width of the MLP.
    """

    def __init__(self, in_channels: int, *, hidden: int = 32) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden, 3),
        )
        # Bias init: sigmoid(-1.0) ~= 0.27, a plausible mid-blue/green water tint.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -1.0)

    def forward(self, deep_feat: Tensor) -> Tensor:
        """Predict the backscatter vector.

        Args:
            deep_feat: Deepest encoder feature map, shape ``(B, C, h, w)``.

        Returns:
            ``B``, shape ``(B, 3)``, range ``(0, 1)``.
        """
        pooled = self.pool(deep_feat).flatten(1)
        b = self.mlp(pooled)
        return torch.sigmoid(b)
