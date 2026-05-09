"""MobileNetV3-Small encoder with multi-scale feature taps.

We tap four pyramid stages from torchvision's ``mobilenet_v3_small`` and
expose them in UNet order (high-res -> low-res) so the decoder can build
skip connections.

Automotive SiL parallel:
    Picking MobileNetV3-Small mirrors the standard practice in
    Tier-1/Tier-2 ADAS perception of using mobile-class backbones (e.g.
    EfficientNet-Lite, MobileNet) when the target SoC is a
    Jetson/Orin-class accelerator. The 4 GB VRAM budget on the RTX 3050
    is comparable to the on-device memory ceiling of automotive edge
    inference modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


@dataclass(frozen=True)
class MobileNetV3SmallStageInfo:
    """Metadata describing a single encoder pyramid stage."""

    block_idx: int
    """Last block index (in ``features``) included in this stage."""

    channels: int
    """Channel count produced by this stage."""

    reduction: int
    """Spatial reduction factor relative to the input image."""


# Stage taps verified against torchvision==0.20 mobilenet_v3_small.
# - block 1  : 16 channels, /4 spatial reduction (after first stride-2 stem + first bneck)
# - block 3  : 24 channels, /8
# - block 8  : 48 channels, /16
# - block 11 : 96 channels, /32
# Each tuple captures the *inclusive* last block index for that stage so we can
# slice ``features[start:end+1]`` cleanly.
_STAGE_TAPS: tuple[MobileNetV3SmallStageInfo, ...] = (
    MobileNetV3SmallStageInfo(block_idx=1, channels=16, reduction=4),
    MobileNetV3SmallStageInfo(block_idx=3, channels=24, reduction=8),
    MobileNetV3SmallStageInfo(block_idx=8, channels=48, reduction=16),
    MobileNetV3SmallStageInfo(block_idx=11, channels=96, reduction=32),
)


class MobileNetV3SmallEncoder(nn.Module):
    """MobileNetV3-Small wrapped to expose pyramid features.

    Args:
        pretrained: If True, loads ImageNet weights. Strongly recommended
            unless training from scratch on a very large underwater corpus.
        out_indices: Which stage taps to return. Defaults to all four.

    Forward output:
        List of feature tensors at strides ``[/4, /8, /16, /32]``
        (after sub-selection by ``out_indices``).
    """

    stage_infos: tuple[MobileNetV3SmallStageInfo, ...] = _STAGE_TAPS

    def __init__(
        self,
        *,
        pretrained: bool = True,
        out_indices: tuple[int, ...] | list[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.out_indices: tuple[int, ...] = tuple(out_indices)

        # Pre-compute the absolute block indices we need to grab outputs at.
        self._tap_block_indices: tuple[int, ...] = tuple(
            self.stage_infos[i].block_idx for i in self.out_indices
        )

    @property
    def out_channels(self) -> tuple[int, ...]:
        """Channel counts of the returned feature maps in order."""
        return tuple(self.stage_infos[i].channels for i in self.out_indices)

    @property
    def out_reductions(self) -> tuple[int, ...]:
        """Spatial reduction factors of the returned feature maps."""
        return tuple(self.stage_infos[i].reduction for i in self.out_indices)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Run the encoder and return tapped pyramid features.

        Args:
            x: Input image batch, shape ``(B, 3, H, W)``, ImageNet-normalised.

        Returns:
            List of feature tensors in increasing-stride order.
        """
        features: list[Tensor] = []
        wanted = set(self._tap_block_indices)
        out: Tensor = x
        for idx, block in enumerate(self.features):
            out = cast(Tensor, block(out))
            if idx in wanted:
                features.append(out)
        return features

    def freeze(self) -> None:
        """Freeze backbone parameters (used during head warmup)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze backbone parameters."""
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()


def imagenet_normalize(x: Tensor) -> Tensor:
    """Normalise a ``[0, 1]`` RGB tensor with ImageNet stats.

    Required because the MobileNetV3 weights expect this normalisation.
    Kept as a free function so it can be applied either inside the model
    (preferred for ONNX) or in the data loader (preferred for training).
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std
