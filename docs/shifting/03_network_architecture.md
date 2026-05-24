# Part 3: Modifying Network Architecture

This guide explains how to update `src/aquaclr/models/model.py` to predict the parameters required for the Sea-Thru model.

## 1. Update the Data Structures
Replace `LEGIONOutputs` with an updated dataclass that reflects the new parameters.

```python
from dataclasses import dataclass
from torch import Tensor

@dataclass
class SeaThruOutputs:
    j: Tensor       # (B, 3, H, W)
    z: Tensor       # (B, 1, H, W)
    beta_d: Tensor  # (B, 3)
    beta_b: Tensor  # (B, 3)
    b_inf: Tensor   # (B, 3)

    def as_tuple(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.j, self.z, self.beta_d, self.beta_b, self.b_inf
```

## 2. Implement New Heads

In `src/aquaclr/models/heads/`, you will need to replace `TransmissionHead` with `DepthHead` and `BackscatterHead` with `IlluminationHead`.

### DepthHead
Similar to the transmission head, but outputs values in $[0, \infty)$ rather than $(0, 1)$. Replace the final `Sigmoid` with a `Softplus` or simply scale the output.

```python
import torch.nn as nn
import torch.nn.functional as F

class DepthHead(nn.Module):
    def __init__(self, in_channels, target_size_factor):
        super().__init__()
        # Convolution layers...
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.target_size_factor = target_size_factor

    def forward(self, x, target_size):
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return F.softplus(x) # Ensures z > 0
```

### IlluminationHead
Instead of a single $B \in \mathbb{R}^3$, this MLP must output 9 values (3 for $\beta_D$, 3 for $\beta_B$, 3 for $B_{inf}$).

```python
class IlluminationHead(nn.Module):
    def __init__(self, in_channels, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 9) # 3 for beta_d, 3 for beta_b, 3 for b_inf
        )

    def forward(self, x):
        out = self.mlp(x)
        # Apply activations
        beta_d = F.softplus(out[:, 0:3])    # >= 0
        beta_b = F.softplus(out[:, 3:6])    # >= 0
        b_inf = torch.sigmoid(out[:, 6:9])  # [0, 1]
        return beta_d, beta_b, b_inf
```

## 3. Update the Model Integration

Modify `LEGIONDeSnowNet` to utilize the new heads and the `invert_seathru` function.

```python
# In model.py
from aquaclr.utils.physics import invert_seathru

class LEGIONDeSnowNet(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... encoder and decoder setup ...
        self.depth_head = DepthHead(in_channels=self.decoder.out_channels, target_size_factor=2)
        self.illumination_head = IlluminationHead(in_channels=self.decoder.deep_channels)

    def forward(self, i: Tensor) -> SeaThruOutputs:
        h, w = i.shape[-2:]
        feat_full, feat_deep = self._encode_decode(i)
        
        z = self.depth_head(feat_full, target_size=(h, w))
        beta_d, beta_b, b_inf = self.illumination_head(feat_deep)
        
        j = invert_seathru(
            i, z, beta_d, beta_b, b_inf,
            eps=self.eps,
            gate_threshold=self.gate_threshold,
            gate_temperature=self.gate_temperature,
            inpaint_kernel=self.inpaint_kernel,
        )
        return SeaThruOutputs(j=j, z=z, beta_d=beta_d, beta_b=beta_b, b_inf=b_inf)
```
