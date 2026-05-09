"""Physics-informed composite loss for LEGION-DeSnow.

The total loss combines four signals:

1. **Reconstruction loss** ``L_rec``: how close the recovered ``J`` is
   to the ground-truth clean image. Charbonnier (a smooth approximation
   of L1) is the default — it tolerates the heavy-tailed residuals of
   underwater imagery much better than MSE while being differentiable
   everywhere.

2. **Physics consistency loss** ``L_phys``: how well the predicted
   ``(t, B)`` reproduce the observed ``I`` from the **ground-truth**
   ``J`` via the forward Jaffe-McGlamery model. This term anchors
   ``(t, B)`` to physically meaningful values rather than arbitrary
   factorisations that happen to make ``J`` come out right.

3. **SSIM loss** ``1 - SSIM(J_pred, J_gt)``: preserves structural
   features that downstream SLAM depends on. Even when L1 is small,
   SSIM can reveal subtle blurring around depth discontinuities.

4. **TV regulariser on ``t``**: enforces piecewise-smooth transmission.
   Without this, the head will happily output salt-and-pepper noise
   that satisfies reconstruction but is physically meaningless.

When the dataloader provides a ground-truth transmission map (LSUI),
an additional **direct ``t`` supervision** term ``L_t`` is enabled.

Total: :math:`L = \\lambda_{rec} L_{rec} + \\lambda_{phys} L_{phys} +
\\lambda_{ssim} L_{ssim} + \\lambda_{tv} L_{tv} + \\lambda_t L_t`.

Automotive SiL parallel:
    Multi-term losses combining pixel fidelity, perceptual structure,
    and physical consistency are the norm in ADAS sensor-restoration
    pipelines (de-rain, de-fog) for the same reason: avoids the network
    learning a "look-good but lie about geometry" shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from aquaclr.losses.ssim import SSIM
from aquaclr.losses.tv import total_variation
from aquaclr.utils.physics import apply_forward_jaffe_mcglamery


@dataclass
class PhysicsLossOutputs:
    """Per-term loss values plus the total. All are scalar tensors.

    Attributes:
        total: Weighted sum used for backprop.
        recon: ``||J_pred - J_gt||``.
        phys: ``||I - (J_gt * t + B * (1 - t))||``.
        ssim: ``1 - SSIM(J_pred, J_gt)``.
        tv: TV regulariser on ``t``.
        t_sup: Direct supervision on ``t`` (zero if ``t_gt`` was missing).
    """

    total: Tensor
    recon: Tensor
    phys: Tensor
    ssim: Tensor
    tv: Tensor
    t_sup: Tensor

    def to_log_dict(self, prefix: str = "loss/") -> dict[str, Tensor]:
        """Flatten to a Lightning-friendly logging dict."""
        return {
            f"{prefix}total": self.total.detach(),
            f"{prefix}recon": self.recon.detach(),
            f"{prefix}phys": self.phys.detach(),
            f"{prefix}ssim": self.ssim.detach(),
            f"{prefix}tv": self.tv.detach(),
            f"{prefix}t_sup": self.t_sup.detach(),
        }


def _charbonnier(x: Tensor, y: Tensor, *, eps: float = 1.0e-3) -> Tensor:
    """Smooth L1 (Charbonnier) loss: ``mean(sqrt((x-y)^2 + eps^2))``."""
    diff = x - y
    return torch.sqrt(diff * diff + eps * eps).mean()


def _l1(x: Tensor, y: Tensor) -> Tensor:
    return (x - y).abs().mean()


class PhysicsInformedLoss(nn.Module):
    """Composite physics-informed loss for LEGION-DeSnow.

    Args:
        lambda_recon: Weight on the reconstruction term.
        lambda_phys: Weight on the forward-physics consistency term.
        lambda_ssim: Weight on the SSIM term.
        lambda_tv: Weight on the TV regulariser on ``t``.
        lambda_t: Weight on direct ``t`` supervision (LSUI batches only).
        charbonnier: If True, use Charbonnier instead of plain L1 for the
            reconstruction and physics-consistency terms.
        ssim_window: Window size for the SSIM Gaussian.
        ssim_sigma: Sigma for the SSIM Gaussian.
    """

    def __init__(
        self,
        *,
        lambda_recon: float = 1.0,
        lambda_phys: float = 0.5,
        lambda_ssim: float = 0.5,
        lambda_tv: float = 1.0e-2,
        lambda_t: float = 0.5,
        charbonnier: bool = True,
        ssim_window: int = 11,
        ssim_sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.lambda_recon = float(lambda_recon)
        self.lambda_phys = float(lambda_phys)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_tv = float(lambda_tv)
        self.lambda_t = float(lambda_t)
        self.use_charbonnier = bool(charbonnier)
        self.ssim_module = SSIM(window_size=ssim_window, sigma=ssim_sigma, channels=3)

    def _pixel_loss(self, x: Tensor, y: Tensor) -> Tensor:
        return _charbonnier(x, y) if self.use_charbonnier else _l1(x, y)

    def forward(
        self,
        i: Tensor,
        j_pred: Tensor,
        j_gt: Tensor,
        t: Tensor,
        b: Tensor,
        *,
        t_gt: Tensor | None = None,
    ) -> PhysicsLossOutputs:
        """Compute the composite loss.

        Args:
            i: Observed image batch ``(B, 3, H, W)`` in ``[0, 1]``.
            j_pred: Recovered clean image ``(B, 3, H, W)`` (from the model).
            j_gt: Ground-truth clean image ``(B, 3, H, W)``.
            t: Predicted transmission ``(B, 1, H, W)``.
            b: Predicted backscatter ``(B, 3)``.
            t_gt: Optional ground-truth transmission ``(B, 1, H, W)``.

        Returns:
            :class:`PhysicsLossOutputs`.
        """
        recon = self._pixel_loss(j_pred, j_gt)
        i_recon = apply_forward_jaffe_mcglamery(j_gt, t, b)
        phys = self._pixel_loss(i_recon, i)
        ssim_val = self.ssim_module(j_pred, j_gt)
        ssim_loss = 1.0 - ssim_val
        tv = total_variation(t)
        if t_gt is not None:
            t_sup = _l1(t, t_gt)
        else:
            t_sup = torch.zeros((), device=i.device, dtype=i.dtype)

        total = (
            self.lambda_recon * recon
            + self.lambda_phys * phys
            + self.lambda_ssim * ssim_loss
            + self.lambda_tv * tv
            + self.lambda_t * t_sup
        )

        return PhysicsLossOutputs(
            total=total,
            recon=recon,
            phys=phys,
            ssim=ssim_loss,
            tv=tv,
            t_sup=t_sup,
        )
