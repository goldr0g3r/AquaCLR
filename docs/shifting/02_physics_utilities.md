# Part 2: Physics Utilities Implementation

To shift to the Sea-Thru model, you must update `src/aquaclr/utils/physics.py` to handle the new forward and inverse equations.

## 1. Implementing the Forward Equation

The forward equation synthesises a hazy underwater image from a clean one. Add the following function to `physics.py`.

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def apply_forward_seathru(
    j: Tensor,
    z: Tensor,
    beta_d: Tensor,
    beta_b: Tensor,
    b_inf: Tensor,
) -> Tensor:
    """Apply the forward Akkaynak-Treibitz (Sea-Thru) image-formation model.

    Args:
        j: Clean radiance image, shape ``(B, 3, H, W)``, range ``[0, 1]``.
        z: Physical depth map, shape ``(B, 1, H, W)``, range ``[0, inf)``.
        beta_d: Direct attenuation coefficients, shape ``(B, 3)`` or ``(B, 3, 1, 1)``.
        beta_b: Backscatter attenuation coefficients, shape ``(B, 3)`` or ``(B, 3, 1, 1)``.
        b_inf: Veiling light / ambient illumination, shape ``(B, 3)`` or ``(B, 3, 1, 1)``.

    Returns:
        Synthesised observation ``I``, shape ``(B, 3, H, W)``, range ``[0, 1]``.
    """
    # Ensure correct broadcasting shapes
    if beta_d.dim() == 2:
        beta_d = beta_d.unsqueeze(-1).unsqueeze(-1)
    if beta_b.dim() == 2:
        beta_b = beta_b.unsqueeze(-1).unsqueeze(-1)
    if b_inf.dim() == 2:
        b_inf = b_inf.unsqueeze(-1).unsqueeze(-1)

    # Calculate transmissions
    t_d = torch.exp(-beta_d * z)
    t_b = torch.exp(-beta_b * z)

    # Apply Sea-Thru equation
    return (j * t_d + b_inf * (1.0 - t_b)).clamp(0.0, 1.0)
```

## 2. Implementing the Inversion Equation

Just like the Jaffe-McGlamery inversion, you must handle the case where depth is very large (and thus direct transmission $t_D \to 0$). We use a gated fallback mechanism to prevent extreme noise amplification.

```python
def invert_seathru(
    i: Tensor,
    z: Tensor,
    beta_d: Tensor,
    beta_b: Tensor,
    b_inf: Tensor,
    *,
    eps: float = 1.0e-3,
    gate_threshold: float = 0.1,
    gate_temperature: float = 0.05,
    inpaint_kernel: int = 7,
) -> Tensor:
    """Invert the Sea-Thru forward model to recover J with gated noise suppression.

    Args:
        i: Observed image, shape ``(B, 3, H, W)``, range ``[0, 1]``.
        z: Predicted depth map, shape ``(B, 1, H, W)``.
        beta_d: Direct attenuation coefficients, shape ``(B, 3)`` or ``(B, 3, 1, 1)``.
        beta_b: Backscatter attenuation coefficients, shape ``(B, 3)`` or ``(B, 3, 1, 1)``.
        b_inf: Predicted veiling light, shape ``(B, 3)`` or ``(B, 3, 1, 1)``.
        eps: Hard floor on the physics-branch denominator.
        gate_threshold: Transmission value (based on t_d) at which the gate is at mid-point.
        gate_temperature: Steepness of the sigmoid gate.
        inpaint_kernel: Kernel size for the AvgPool spatial-fill branch.

    Returns:
        Recovered clean image J, shape ``(B, 3, H, W)``, clamped to ``[0, 1]``.
    """
    if beta_d.dim() == 2:
        beta_d = beta_d.unsqueeze(-1).unsqueeze(-1)
    if beta_b.dim() == 2:
        beta_b = beta_b.unsqueeze(-1).unsqueeze(-1)
    if b_inf.dim() == 2:
        b_inf = b_inf.unsqueeze(-1).unsqueeze(-1)

    t_d = torch.exp(-beta_d * z)
    t_b = torch.exp(-beta_b * z)

    # 1. Gate calculation based on direct transmission t_d
    # We take the mean across channels to get a single spatial gate
    t_d_mean = t_d.mean(dim=1, keepdim=True)
    g = torch.sigmoid((t_d_mean - gate_threshold) / gate_temperature)

    # 2. Physics Branch
    t_d_safe = t_d.clamp(min=eps)
    j_phys = (i - b_inf * (1.0 - t_b)) / t_d_safe

    # 3. Fill Branch (spatial inpainting proxy)
    pad = inpaint_kernel // 2
    j_fill = F.avg_pool2d(i, kernel_size=inpaint_kernel, stride=1, padding=pad)

    # 4. Soft Blend
    j = g * j_phys + (1.0 - g) * j_fill
    return j.clamp(0.0, 1.0)
```

These functions will completely replace their Jaffe-McGlamery counterparts in both the model inferences and loss calculations.
