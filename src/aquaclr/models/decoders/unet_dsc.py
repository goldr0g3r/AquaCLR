"""Lightweight UNet decoder built from depthwise-separable convolutions.

Depthwise-separable convolutions (DSC) factor a standard ``KxK`` conv with
``C_in`` -> ``C_out`` channels into:

1. a depthwise ``KxK`` conv (``C_in`` -> ``C_in``, groups=``C_in``), and
2. a pointwise ``1x1`` conv (``C_in`` -> ``C_out``).

The parameter and FLOP count drop by roughly ``C_out / (K*K) + 1/(K*K)``,
which is ~9x for ``K=3``. This is essential for hitting the 50 MB / 15 ms
budget on an RTX 3050.

Automotive SiL parallel:
    DSC blocks are the workhorse of automotive perception backbones
    (MobileNet, EfficientNet-Lite) precisely because they are friendly
    to fixed-function NPUs and quantisation. Trading multiplications for
    memory bandwidth matches the bottleneck profile of Jetson and DRIVE
    Orin SoCs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise + 1x1 pointwise conv with BN + ReLU6.

    ReLU6 is preferred over ReLU because it bounds activations to ``[0, 6]``
    which (a) matches MobileNet's design and (b) plays much nicer with
    INT8 quantisation later.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Depthwise kernel size (default 3).
        stride: Spatial stride (default 1).
        padding: Spatial padding. ``None`` means ``kernel_size // 2``.
        bias: Whether to include conv bias. Disabled by default since BN follows.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class _UpBlock(nn.Module):
    """Upsample-by-2, concat skip, two DSC convs."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels + skip_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor | None) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            # Defensive crop in case of off-by-one from non-divisible input shapes.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDSCDecoder(nn.Module):
    """Lightweight UNet decoder consuming MobileNetV3-Small pyramid features.

    Args:
        encoder_channels: Channels at each encoder stage in stride order
            ``[/4, /8, /16, /32]``.
        decoder_channels: Output channels at each decoder up-stage. Length
            must match ``len(encoder_channels)``.

    Forward output:
        Two tensors:
            * ``feat_full``: highest-resolution feature map at stride ``/2``.
              We do one extra upsample so the transmission head sees
              full-resolution geometry.
            * ``feat_deep``: the deepest encoder feature, useful as input
              to the global backscatter head.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
    ) -> None:
        super().__init__()
        if len(encoder_channels) != len(decoder_channels):
            msg = (
                f"encoder_channels ({len(encoder_channels)}) and decoder_channels "
                f"({len(decoder_channels)}) must have the same length"
            )
            raise ValueError(msg)

        enc = list(encoder_channels)
        dec = list(decoder_channels)

        # Build up-blocks from deepest -> shallowest.
        # The deepest encoder feature has no skip from a deeper layer; we treat
        # it as the input to the first up-block, with skip = enc[-2].
        self.up_blocks = nn.ModuleList()
        in_ch = enc[-1]
        for i in range(len(enc) - 1, 0, -1):
            skip_ch = enc[i - 1]
            out_ch = dec[i - 1]
            self.up_blocks.append(_UpBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        # One final upsample (stride /4 -> /2) without a skip for sharper t(x).
        self.final_up = _UpBlock(in_ch, skip_channels=0, out_channels=dec[0])

        self.out_channels: int = dec[0]
        self.deep_channels: int = enc[-1]

    def forward(self, features: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Decode pyramid features to a high-resolution feature map.

        Args:
            features: List of encoder features in stride order
                ``[/4, /8, /16, /32]``.

        Returns:
            ``(feat_full, feat_deep)`` where ``feat_full`` has stride /2.
        """
        deep = features[-1]
        x = deep
        for i, block in enumerate(self.up_blocks):
            skip = features[-2 - i]
            x = block(x, skip)
        x = self.final_up(x, None)
        return x, deep
