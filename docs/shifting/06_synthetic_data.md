# Part 6: Synthetic Data Generation with the Sea-Thru Forward Model

One of the most powerful benefits of a physics-informed model is the ability to **synthesize unlimited training data** from unpaired clean images and plausible physical parameters. This document explains how to use `apply_forward_seathru` to augment the training set and how to sample realistic underwater water-body parameters.

---

## 1. Motivation

The MSRB and LSUI datasets are small (a few thousand paired images). The Sea-Thru forward model allows us to create an unlimited number of new `(I, J)` training pairs by:

1. Starting with any clean, in-focus RGB image $J$ (e.g., COCO, ImageNet, or any clean dive footage).
2. Estimating a plausible monocular depth map $z$ for that scene.
3. Sampling physically realistic water-body parameters $(\beta_D, \beta_B, B_{inf})$.
4. Applying `apply_forward_seathru(j, z, beta_d, beta_b, b_inf)` to synthesize the degraded observation $I$.

This synthetic pair $(I, J)$ is then added to the training dataloader alongside real data.

---

## 2. Depth Estimation for Clean Images

Since clean images lack ground-truth depth, we use a pretrained monocular depth estimator to generate a **pseudo-depth map** $\hat{z}$.

### 2.1 Recommended Model: Depth Anything V2

Depth Anything V2 (ViT-S or ViT-B) provides accurate relative depth for general in-the-wild images and runs efficiently enough for offline preprocessing.

```python
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# Assumes depth_anything_v2 is installed:
#   pip install depth-anything-v2
from depth_anything_v2.dpt import DepthAnythingV2

def load_depth_estimator(model_size: str = "vits") -> DepthAnythingV2:
    """Load a Depth Anything V2 model. model_size: 'vits', 'vitb', or 'vitl'."""
    cfg = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    }[model_size]
    model = DepthAnythingV2(**cfg)
    checkpoint_path = f"checkpoints/depth_anything_v2_{model_size}.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model.eval()


def estimate_depth(model: DepthAnythingV2, image_rgb: np.ndarray) -> np.ndarray:
    """Return a depth map in [0, inf) metres (relative, not absolute).

    The raw output is relative disparity. We convert to metric-like depth
    using a plausible scale: depth = scale / (disparity + shift).
    For synthetic augmentation, exact metric accuracy is not required —
    only the *relative* structure must be correct.
    """
    with torch.no_grad():
        depth = model.infer_image(image_rgb)   # (H, W), values in [0, 1] relative
    # Map relative disparity to plausible metric depth (0–10 m range for near-field)
    depth_metric = 1.0 / (depth.clip(0.01) + 0.01) * 0.5   # Rough scale factor
    return depth_metric.astype(np.float32)
```

### 2.2 Offline Depth Preprocessing

For efficiency, precompute and cache depth maps alongside clean images before training:

```python
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch

def precompute_depths(clean_image_dir: str, output_dir: str, batch_size: int = 8):
    """Precompute and save depth maps for all images in a directory."""
    model = load_depth_estimator("vits").cuda()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    images = sorted(Path(clean_image_dir).glob("*.jpg")) + \
             sorted(Path(clean_image_dir).glob("*.png"))

    for img_path in images:
        depth_path = out_path / (img_path.stem + "_depth.npy")
        if depth_path.exists():
            continue  # Already cached
        img = np.array(Image.open(img_path).convert("RGB"))
        depth = estimate_depth(model, img)
        np.save(depth_path, depth)
        print(f"Saved: {depth_path}")
```

---

## 3. Realistic Parameter Sampling

To generate believable synthetic degradations, the sampled parameters must fall within the physical ranges observed in real ocean surveys.

### 3.1 Empirical Ranges from the Sea-Thru Paper (Akkaynak & Treibitz, 2019)

| Parameter | Channel | Typical Min | Typical Max | Distribution |
|---|---|---|---|---|
| $\beta_{D,R}$ | Red | 0.10 | 0.60 | LogNormal |
| $\beta_{D,G}$ | Green | 0.05 | 0.30 | LogNormal |
| $\beta_{D,B}$ | Blue | 0.03 | 0.20 | LogNormal |
| $\beta_{B,R}$ | Red | 0.15 | 0.80 | LogNormal |
| $\beta_{B,G}$ | Green | 0.07 | 0.40 | LogNormal |
| $\beta_{B,B}$ | Blue | 0.04 | 0.25 | LogNormal |
| $B_{inf,R}$ | Red | 0.02 | 0.20 | Uniform |
| $B_{inf,G}$ | Green | 0.05 | 0.45 | Uniform |
| $B_{inf,B}$ | Blue | 0.10 | 0.70 | Uniform |

**Key physical constraints:**
- $\beta_{D,c} \geq \beta_{B,c}$ is not strictly required but is common in blue-water conditions.
- Red attenuates faster than green, which attenuates faster than blue: $\beta_{D,R} > \beta_{D,G} > \beta_{D,B}$ in most ocean water bodies. This should be enforced in sampling.
- $B_{inf}$ is higher in the blue channel (water is blue because blue light scatters most).

### 3.2 Correlated Sampling Function

```python
import numpy as np
from torch import Tensor
import torch

# Log-normal parameters (mu, sigma) fitted to empirical ranges above
# Values approximate: mu = log(median), sigma chosen for spread
BETA_D_PARAMS = {  # (mu, sigma) for log-normal
    "R": (-1.4, 0.5),   # median ~0.25
    "G": (-2.0, 0.5),   # median ~0.14
    "B": (-2.5, 0.5),   # median ~0.08
}
BETA_B_PARAMS = {
    "R": (-1.1, 0.5),   # median ~0.33
    "G": (-1.8, 0.5),   # median ~0.17
    "B": (-2.3, 0.5),   # median ~0.10
}
B_INF_PARAMS = {        # (low, high) for uniform
    "R": (0.02, 0.20),
    "G": (0.05, 0.45),
    "B": (0.10, 0.70),
}

def sample_water_params(batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a batch of physically plausible Sea-Thru water parameters.

    Returns:
        beta_d: (B, 3) direct attenuation coefficients
        beta_b: (B, 3) backscatter attenuation coefficients
        b_inf:  (B, 3) ambient illumination at infinity
    """
    beta_d_list, beta_b_list, b_inf_list = [], [], []

    for ch in ["R", "G", "B"]:
        mu, sigma = BETA_D_PARAMS[ch]
        beta_d_list.append(np.random.lognormal(mu, sigma, batch_size))

        mu, sigma = BETA_B_PARAMS[ch]
        beta_b_list.append(np.random.lognormal(mu, sigma, batch_size))

        lo, hi = B_INF_PARAMS[ch]
        b_inf_list.append(np.random.uniform(lo, hi, batch_size))

    beta_d = torch.tensor(np.stack(beta_d_list, axis=1), dtype=torch.float32, device=device)
    beta_b = torch.tensor(np.stack(beta_b_list, axis=1), dtype=torch.float32, device=device)
    b_inf  = torch.tensor(np.stack(b_inf_list,  axis=1), dtype=torch.float32, device=device)

    return beta_d, beta_b, b_inf
```

---

## 4. Synthesis Pipeline

```python
from aquaclr.utils.physics import apply_forward_seathru

def synthesize_underwater_batch(
    j_batch: Tensor,     # (B, 3, H, W) clean images in [0, 1]
    z_batch: Tensor,     # (B, 1, H, W) depth maps in metres
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Synthesize a degraded underwater image batch using Sea-Thru.

    Returns:
        i_synth:  Synthesized degraded observation (B, 3, H, W)
        beta_d:   Sampled direct attenuation (B, 3)
        beta_b:   Sampled backscatter attenuation (B, 3)
        b_inf:    Sampled ambient light (B, 3)
        z_batch:  The input depth maps (returned for loss supervision)
    """
    device = j_batch.device
    B = j_batch.shape[0]

    beta_d, beta_b, b_inf = sample_water_params(B, device)

    with torch.no_grad():
        i_synth = apply_forward_seathru(j_batch, z_batch, beta_d, beta_b, b_inf)

    return i_synth, beta_d, beta_b, b_inf, z_batch
```

---

## 5. Integration with the DataModule

### 5.1 `SyntheticUnderwaterDataset`

```python
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

class SyntheticUnderwaterDataset(Dataset):
    """Wraps a directory of clean RGB images + precomputed depth maps.

    At runtime, synthesizes a random underwater degradation using Sea-Thru.
    Returns a batch dict compatible with the standard AquaCLR schema.
    """

    def __init__(self, clean_dir: str, depth_dir: str, image_size: int = 256):
        self.clean_paths = sorted(Path(clean_dir).glob("*.jpg")) + \
                           sorted(Path(clean_dir).glob("*.png"))
        self.depth_dir = Path(depth_dir)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),            # → (3, H, W) in [0, 1]
        ])
        self.depth_transform = T.Resize((image_size, image_size),
                                        interpolation=T.InterpolationMode.BILINEAR)

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.clean_paths[idx]
        j = self.transform(Image.open(path).convert("RGB"))  # (3, H, W)

        depth_path = self.depth_dir / (path.stem + "_depth.npy")
        z_np = np.load(depth_path).astype(np.float32)
        z = torch.from_numpy(z_np).unsqueeze(0)              # (1, H, W)
        z = self.depth_transform(z)

        # Synthesize degradation on-the-fly
        beta_d, beta_b, b_inf = sample_water_params(1, j.device)
        from aquaclr.utils.physics import apply_forward_seathru
        i = apply_forward_seathru(
            j.unsqueeze(0), z.unsqueeze(0),
            beta_d, beta_b, b_inf,
        ).squeeze(0)

        return {
            "i":        i,
            "j":        j,
            "z_gt":     z,
            "has_z_gt": True,
        }
```

### 5.2 Adding to `CombinedDataModule`

In `combined_datamodule.py`, add the synthetic dataset to the training mix with a configurable mixing ratio:

```python
from torch.utils.data import ConcatDataset, WeightedRandomSampler

class CombinedDataModule:
    def __init__(self, ..., synthetic_weight: float = 0.3):
        ...
        self.synthetic_weight = synthetic_weight

    def train_dataloader(self):
        real_ds = ConcatDataset([self.msrb_train, self.lsui_train])
        synth_ds = SyntheticUnderwaterDataset(
            clean_dir=self.cfg.synthetic_clean_dir,
            depth_dir=self.cfg.synthetic_depth_dir,
            image_size=self.cfg.image_size,
        )
        combined = ConcatDataset([real_ds, synth_ds])

        # Weight synthetic samples at synthetic_weight of total
        n_real = len(real_ds)
        n_synth = len(synth_ds)
        weights = [1.0] * n_real + [self.synthetic_weight * n_real / max(n_synth, 1)] * n_synth
        sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)

        return DataLoader(combined, batch_size=self.batch_size,
                          sampler=sampler, collate_fn=seathru_collate_fn,
                          num_workers=self.num_workers, pin_memory=True)
```

---

## 6. Augmentation Curriculum

Training purely on hard synthetic degradations from the start can cause instability. Use a **curriculum** where the maximum depth range of the synthesis increases over epochs:

| Epoch Range | Max Synthetic Depth | Effect |
|---|---|---|
| 0 – 10 | 2 m | Mild degradation; easy pairs for backbone warmup |
| 10 – 30 | 5 m | Moderate degradation; network learns full inversion |
| 30+ | 10 m | Strong degradation; forces robust physics inversion |

```python
def get_max_depth_for_epoch(epoch: int) -> float:
    if epoch < 10:
        return 2.0
    if epoch < 30:
        return 5.0
    return 10.0
```

Inject this into `SyntheticUnderwaterDataset` via a `max_depth` parameter that clips the depth map during `__getitem__`: `z = z.clamp(0, max_depth)`.
