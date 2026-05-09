"""LEGION-DeSnow: physics-informed marine-snow removal network.

This is the **canonical entry point** for the AquaCLR Milestone 1
network. It composes:

1. a MobileNetV3-Small encoder (ImageNet-pretrained, ~1.5 M params),
2. a lightweight UNet decoder built from depthwise-separable convs,
3. a transmission head producing :math:`t(x) \\in [0,1]^{H \\times W}`,
4. a backscatter head producing the global ambient vector
   :math:`B \\in [0, 1]^{3}`,
5. an analytic Jaffe-McGlamery inversion that yields the recovered
   clean image :math:`J` without any extra learnable parameters.

Because all learnable parameters predict only the **physical**
parameters of the scattering model, the network is constrained to
solutions that are consistent with underwater optics. This is the
"physics-informed" property and it is what gives the model its strong
out-of-distribution robustness compared to a vanilla image-to-image
translator of the same size.

Automotive SiL parallel:
    The architectural shape is identical to a sensor-physics
    preprocessing block in an automotive perception stack. Replace
    "marine snow" with "rain droplets on lens", swap MSRB for a
    rain-augmented KITTI, and the same network would denoise camera
    frames before SLAM. The Jaffe-McGlamery model maps 1:1 onto the
    Koschmieder atmospheric model used to render rain in DriveSim.

Memory budget (parameter-only, ignoring activations):
    ~4-6 M params -> ~18-24 MB FP32 -> ~9-12 MB FP16. Well below the
    50 MB ceiling demanded by the M1 spec.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from aquaclr.models.backbones.mobilenet_v3 import (
    MobileNetV3SmallEncoder,
    imagenet_normalize,
)
from aquaclr.models.decoders.unet_dsc import UNetDSCDecoder
from aquaclr.models.heads.backscatter import BackscatterHead
from aquaclr.models.heads.transmission import TransmissionHead
from aquaclr.utils.physics import invert_jaffe_mcglamery


@dataclass
class LEGIONOutputs:
    """Structured output of :class:`LEGIONDeSnowNet`.

    Attributes:
        j: Recovered clean image, shape ``(B, 3, H, W)``, range ``[0, 1]``.
        t: Predicted transmission, shape ``(B, 1, H, W)``, range ``(0, 1)``.
        b: Predicted backscatter, shape ``(B, 3)``, range ``(0, 1)``.
    """

    j: Tensor
    t: Tensor
    b: Tensor

    def as_tuple(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(J, t, B)``. Useful for ONNX export."""
        return self.j, self.t, self.b


class LEGIONDeSnowNet(nn.Module):
    """Physics-informed marine-snow removal network.

    Args:
        backbone: Backbone identifier. Currently only ``"mobilenet_v3_small"``
            is supported; the parameter is kept so future backbones can
            be plugged in without touching call sites.
        pretrained: Whether to load ImageNet pretrained weights for the encoder.
        out_indices: Encoder pyramid stages to tap. Defaults to all four.
        decoder_channels: Channel widths of the decoder up-blocks, shallow-first.
        backscatter_hidden: Hidden width of the backscatter MLP.
        use_depthwise: Currently always True (kept for API symmetry).
        use_channels_last: If True, switch the model to channels-last memory
            format on ``cuda``. Channels-last yields a measurable speedup
            on Ampere and later GPUs.
        eps: Floor on the transmission denominator during the analytic
            inversion. Smaller -> sharper, but more risk of over-amplifying
            noise in dark regions.
        normalize_input: If True, the model applies ImageNet normalisation
            internally so that callers can pass plain ``[0, 1]`` RGB frames
            (this is what we want for ONNX/TRT and ROS2). If False, the
            caller is responsible for normalisation.

    Inputs:
        ``i``: ``(B, 3, H, W)`` float tensor in ``[0, 1]`` (sRGB-linear).

    Outputs:
        :class:`LEGIONOutputs` carrying ``J``, ``t``, ``B``.
    """

    SUPPORTED_BACKBONES: tuple[str, ...] = ("mobilenet_v3_small",)

    def __init__(
        self,
        *,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        decoder_channels: Sequence[int] = (96, 64, 32, 16),
        backscatter_hidden: int = 32,
        use_depthwise: bool = True,
        use_channels_last: bool = True,
        eps: float = 1.0e-3,
        normalize_input: bool = True,
        freeze_backbone_epochs: int = 0,
    ) -> None:
        super().__init__()
        if backbone not in self.SUPPORTED_BACKBONES:
            msg = f"Unsupported backbone {backbone!r}; choose one of {self.SUPPORTED_BACKBONES}"
            raise ValueError(msg)
        if not use_depthwise:
            msg = "use_depthwise=False is reserved for future ablations and not yet supported."
            raise NotImplementedError(msg)

        self.encoder = MobileNetV3SmallEncoder(
            pretrained=pretrained,
            out_indices=tuple(out_indices),
        )
        encoder_channels = self.encoder.out_channels
        self.decoder = UNetDSCDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=tuple(decoder_channels),
        )
        self.transmission_head = TransmissionHead(
            in_channels=self.decoder.out_channels,
            target_size_factor=2,  # decoder ends at /2; head upsamples to /1
        )
        self.backscatter_head = BackscatterHead(
            in_channels=self.decoder.deep_channels,
            hidden=backscatter_hidden,
        )

        self.eps = float(eps)
        self.use_channels_last = bool(use_channels_last)
        self.normalize_input = bool(normalize_input)
        self.freeze_backbone_epochs = int(freeze_backbone_epochs)

        if self.use_channels_last:
            self.to(memory_format=torch.channels_last)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self) -> None:
        """Freeze the encoder (used for the first ``freeze_backbone_epochs`` epochs)."""
        self.encoder.freeze()

    def unfreeze_backbone(self) -> None:
        """Unfreeze the encoder."""
        self.encoder.unfreeze()

    # ---------------------------------------------------------------- forward

    def _encode_decode(self, i: Tensor) -> tuple[Tensor, Tensor]:
        """Run encoder + decoder and return ``(feat_full, feat_deep)``."""
        if self.normalize_input:
            x = imagenet_normalize(i)
        else:
            x = i
        features = self.encoder(x)
        feat_full, feat_deep = self.decoder(features)
        return feat_full, feat_deep

    def forward(self, i: Tensor) -> LEGIONOutputs:
        """Predict ``J``, ``t``, ``B`` from an observed image ``I``.

        Args:
            i: Observed image batch, shape ``(B, 3, H, W)``, range ``[0, 1]``.

        Returns:
            :class:`LEGIONOutputs`.
        """
        h, w = i.shape[-2:]
        feat_full, feat_deep = self._encode_decode(i)
        t = self.transmission_head(feat_full, target_size=(h, w))
        b = self.backscatter_head(feat_deep)
        j = invert_jaffe_mcglamery(i, t, b, eps=self.eps)
        return LEGIONOutputs(j=j, t=t, b=b)

    def forward_export(self, i: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """ONNX/TRT-friendly forward returning a flat tuple ``(J, t, B)``.

        ONNX exporters do not support dataclass returns; this thin wrapper
        is the canonical export entry point.

        Args:
            i: Observed image, shape ``(B, 3, H, W)``, range ``[0, 1]``.

        Returns:
            Tuple ``(J, t, B)``.
        """
        out = self.forward(i)
        return out.as_tuple()

    # ----------------------------------------------------------- helpers

    def estimate_size_mb(self, *, dtype: torch.dtype = torch.float32) -> float:
        """Estimate the on-disk parameter size in megabytes.

        Args:
            dtype: Storage dtype used for the estimate (FP32, FP16, BF16).

        Returns:
            Parameter size in megabytes (1 MB == 1024 KB == 1024**2 bytes).
        """
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        total_bytes = self.num_parameters * bytes_per_elem
        return total_bytes / (1024 * 1024)
