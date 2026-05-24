# Part 4: Adapting the Physics-Informed Loss

The final step is updating `src/aquaclr/losses/physics_loss.py`. The loss must now enforce the physical validity of the Sea-Thru outputs ($z, \beta_D, \beta_B, B_{inf}$) rather than the Jaffe-McGlamery outputs ($t, B$).

## 1. Update the Outputs Dataclass

Update the loss outputs to reflect the supervision of depth instead of transmission.

```python
from dataclasses import dataclass
from torch import Tensor

@dataclass
class PhysicsLossOutputs:
    total: Tensor
    recon: Tensor
    phys: Tensor
    ssim: Tensor
    tv: Tensor
    z_sup: Tensor  # changed from t_sup

    def to_log_dict(self, prefix: str = "loss/") -> dict[str, Tensor]:
        return {
            f"{prefix}total": self.total.detach(),
            f"{prefix}recon": self.recon.detach(),
            f"{prefix}phys": self.phys.detach(),
            f"{prefix}ssim": self.ssim.detach(),
            f"{prefix}tv": self.tv.detach(),
            f"{prefix}z_sup": self.z_sup.detach(),
        }
```

## 2. Update the `PhysicsInformedLoss` Module

You must adapt the initialization parameters (e.g., renaming `lambda_t` to `lambda_z`) and the `forward` function.

### Key Modifications:
1. **Physical consistency loss ($L_{phys}$)**: Uses `apply_forward_seathru` with the predicted parameters.
2. **Total Variation Loss ($L_{TV}$)**: Applied to the depth map $z$ instead of transmission $t$. This forces the network to predict piecewise-smooth 3D structures.
3. **Direct Supervision ($L_z$)**: If you possess ground-truth depth maps, supervise $z$ directly.

```python
from aquaclr.utils.physics import apply_forward_seathru
from aquaclr.losses.tv import total_variation
from aquaclr.losses.ssim import SSIM

class PhysicsInformedLoss(nn.Module):
    def __init__(
        self,
        *,
        lambda_recon: float = 1.0,
        lambda_phys: float = 0.5,
        lambda_ssim: float = 0.5,
        lambda_tv: float = 1.0e-2,
        lambda_z: float = 0.5, # Changed from lambda_t
        charbonnier: bool = True,
        ssim_window: int = 11,
        ssim_sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_phys = lambda_phys
        self.lambda_ssim = lambda_ssim
        self.lambda_tv = lambda_tv
        self.lambda_z = lambda_z
        self.use_charbonnier = charbonnier
        self.ssim_module = SSIM(window_size=ssim_window, sigma=ssim_sigma, channels=3)

    # _pixel_loss logic remains the same...

    def forward(
        self,
        i: Tensor,
        j_pred: Tensor,
        j_gt: Tensor,
        z: Tensor,
        beta_d: Tensor,
        beta_b: Tensor,
        b_inf: Tensor,
        *,
        z_gt: Tensor | None = None,
    ) -> PhysicsLossOutputs:
        
        # 1. Reconstruction Loss
        recon = self._pixel_loss(j_pred, j_gt)
        
        # 2. Physics Consistency Loss
        i_recon = apply_forward_seathru(j_gt, z, beta_d, beta_b, b_inf)
        phys = self._pixel_loss(i_recon, i)
        
        # 3. Structural Loss
        ssim_val = self.ssim_module(j_pred, j_gt)
        ssim_loss = 1.0 - ssim_val
        
        # 4. Total Variation on Depth
        tv = total_variation(z)
        
        # 5. Direct Supervision (if depth GT available)
        if z_gt is not None:
            z_sup = self._pixel_loss(z, z_gt)
        else:
            z_sup = torch.zeros((), device=i.device, dtype=i.dtype)

        # Total Weighted Loss
        total = (
            self.lambda_recon * recon
            + self.lambda_phys * phys
            + self.lambda_ssim * ssim_loss
            + self.lambda_tv * tv
            + self.lambda_z * z_sup
        )

        return PhysicsLossOutputs(
            total=total,
            recon=recon,
            phys=phys,
            ssim=ssim_loss,
            tv=tv,
            z_sup=z_sup,
        )
```

With these 4 parts implemented, your AquaCLR architecture will have successfully transitioned to the physics-informed Sea-Thru representation.
