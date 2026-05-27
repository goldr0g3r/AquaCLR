# Part 5: Wiring Sea-Thru into the Training Pipeline

This guide covers all changes required to `src/aquaclr/training/lit_module.py`, the Hydra config YAML files, and the data pipeline to make the Sea-Thru network train end-to-end with PyTorch Lightning.

---

## 1. What Changes and What Does Not

| Component | Status | Summary of Change |
|---|---|---|
| `utils/physics.py` | **Changed** | New `apply_forward_seathru` and `invert_seathru` functions (see Part 2) |
| `models/heads/` | **Changed** | `TransmissionHead` → `DepthHead`; `BackscatterHead` → `IlluminationHead` (see Part 3) |
| `models/model.py` | **Changed** | `LEGIONDeSnowNet` now returns `SeaThruOutputs` (see Part 3) |
| `losses/physics_loss.py` | **Changed** | `lambda_t` → `lambda_z`; TV loss on `z` not `t` (see Part 4) |
| `training/lit_module.py` | **Changed** | `forward()` signature; `_shared_step()` unpacks new output structure |
| `configs/train/*.yaml` | **Changed** | `lambda_t` key renamed to `lambda_z` |
| `data/` modules | **Unchanged** | Batch format gains optional `z_gt` key but is backward-compatible |
| `losses/ssim.py`, `losses/tv.py` | **Unchanged** | Generic; no physics-specific assumptions |
| `inference/onnx_export.py` | **Unchanged** | Exports `self.net` which is still a plain `nn.Module` |

---

## 2. Changes to `lit_module.py`

### 2.1 Update the `forward()` Method

The current `forward()` returns `(j, t, b)`. After the shift it must return `(j, z, beta_d, beta_b, b_inf)`:

```python
# BEFORE
def forward(self, i: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    out = self.net(i)
    return out.j, out.t, out.b

# AFTER
def forward(self, i: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    out = self.net(i)
    return out.j, out.z, out.beta_d, out.beta_b, out.b_inf
```

### 2.2 Update `_shared_step()`

The shared step calls `self.loss(...)` with the current Jaffe-McGlamery signature:

```python
# BEFORE
loss_outputs = self.loss(
    i=i, j_pred=out.j, j_gt=j_gt, t=out.t, b=out.b, t_gt=t_gt
)
```

Replace with the Sea-Thru signature:

```python
# AFTER
out = self.net(i)

# z_gt is optional; only present in LSUI dataset batches
z_gt: Tensor | None = None
if batch.get("has_z_gt") is not None and bool(batch["has_z_gt"].any()):
    z_gt = batch.get("z_gt")

loss_outputs = self.loss(
    i=i,
    j_pred=out.j,
    j_gt=j_gt,
    z=out.z,
    beta_d=out.beta_d,
    beta_b=out.beta_b,
    b_inf=out.b_inf,
    z_gt=z_gt,
)
```

### 2.3 Update Metric Logging

The `to_log_dict()` output now contains `z_sup` instead of `t_sup`. No other metric names change — `val/psnr` and `val/ssim` remain the EarlyStopping monitor targets and do not need updating.

However, add explicit `z_sup` logging for interpretability:

```python
# Inside _shared_step, after calling self.loss(...)
if stage == "val":
    # Log z_sup only if ground-truth depth was available in this batch
    if z_gt is not None:
        self.log("val/loss/z_sup", loss_outputs.z_sup, on_step=False, on_epoch=True)
```

### 2.4 Full Revised `_shared_step()`

```python
def _shared_step(self, batch: dict[str, Any], *, stage: str) -> Tensor:
    i = batch["i"]
    j_gt = batch["j"]
    out = self.net(i)

    z_gt: Tensor | None = None
    if batch.get("has_z_gt") is not None and bool(batch["has_z_gt"].any()):
        z_gt = batch.get("z_gt")

    loss_outputs = self.loss(
        i=i,
        j_pred=out.j,
        j_gt=j_gt,
        z=out.z,
        beta_d=out.beta_d,
        beta_b=out.beta_b,
        b_inf=out.b_inf,
        z_gt=z_gt,
    )

    log_dict = loss_outputs.to_log_dict(prefix=f"{stage}/loss/")
    self.log_dict(
        log_dict,
        prog_bar=False,
        on_step=(stage == "train"),
        on_epoch=True,
        sync_dist=True,
    )

    psnr_metric = self.train_psnr if stage == "train" else self.val_psnr
    ssim_metric = self.train_ssim if stage == "val" else self.val_ssim
    psnr_metric.update(out.j.clamp(0, 1), j_gt.clamp(0, 1))
    ssim_metric.update(out.j.clamp(0, 1), j_gt.clamp(0, 1))
    self.log(f"{stage}/psnr", psnr_metric, prog_bar=True, on_step=False, on_epoch=True)
    self.log(f"{stage}/ssim", ssim_metric, prog_bar=True, on_step=False, on_epoch=True)

    return loss_outputs.total
```

---

## 3. Config YAML Changes

### 3.1 `configs/train/rtx_a3000_bf16.yaml` and `rtx3050_bf16.yaml`

Replace the `lambda_t` key in the `loss` block:

```yaml
# BEFORE
loss:
  _target_: aquaclr.losses.physics_loss.PhysicsInformedLoss
  lambda_t: 0.5

# AFTER
loss:
  _target_: aquaclr.losses.physics_loss.PhysicsInformedLoss
  lambda_z: 0.5   # Supervision weight for ground-truth depth (from LSUI)
```

### 3.2 `configs/model/legion_desnow_s.yaml`

The model config must remove `backscatter_hidden` (used by the old `BackscatterHead`) and add the illumination head hidden size:

```yaml
# BEFORE
backbone: mobilenet_v3_small
pretrained: true
out_indices: [0, 1, 2, 3]
decoder_channels: [96, 64, 32, 16]
backscatter_hidden: 32

# AFTER
backbone: mobilenet_v3_small
pretrained: true
out_indices: [0, 1, 2, 3]
decoder_channels: [96, 64, 32, 16]
illumination_hidden: 32   # Hidden size of IlluminationHead MLP
```

---

## 4. Batch Schema Changes

The data pipeline batch dict must be extended to carry optional Sea-Thru depth ground truth. The key additions are `z_gt` and `has_z_gt`. Existing datasets (`msrb_dataset.py`) that lack depth simply set `has_z_gt = False` and omit `z_gt`.

### 4.1 Dataset Batch Format

```python
# Batch structure after Sea-Thru migration
batch = {
    "i":        Tensor,          # (B, 3, H, W) degraded input
    "j":        Tensor,          # (B, 3, H, W) ground-truth clean image
    "has_z_gt": Tensor,          # (B,) boolean — True only for LSUI samples with depth
    "z_gt":     Tensor | None,   # (B, 1, H, W) ground-truth depth (metres), or absent
}
```

### 4.2 Collation

When combining MSRB (no depth GT) and LSUI (has depth GT) in the same batch via `CombinedDataModule`, the collate function must handle mismatched keys:

```python
def seathru_collate_fn(samples: list[dict]) -> dict:
    """Collate function that handles optional z_gt across dataset sources."""
    import torch
    from torch.utils.data.dataloader import default_collate

    has_z = [s.get("has_z_gt", False) for s in samples]
    collated = default_collate([
        {k: v for k, v in s.items() if k != "z_gt"} for s in samples
    ])
    collated["has_z_gt"] = torch.tensor(has_z, dtype=torch.bool)

    if any(has_z):
        # Pad samples without z_gt with zeros so the batch can be stacked
        z_list = [s.get("z_gt", torch.zeros(1, s["i"].shape[-2], s["i"].shape[-1]))
                  for s in samples]
        collated["z_gt"] = torch.stack(z_list)
    else:
        collated["z_gt"] = None

    return collated
```

Register this collate function in `CombinedDataModule`:

```python
# In combined_datamodule.py
from aquaclr.data.transforms import seathru_collate_fn

DataLoader(..., collate_fn=seathru_collate_fn)
```

---

## 5. Checkpoint Compatibility

Sea-Thru checkpoints are **incompatible** with Jaffe-McGlamery checkpoints because the head parameter names change. Do **not** attempt to resume from an old `.ckpt` across the migration boundary.

If you need to preserve the backbone weights (to avoid re-running the expensive ImageNet pretrain step), extract only the encoder state dict:

```python
import torch

old_ckpt = torch.load("outputs/jm_run/ckpts/last.ckpt", map_location="cpu")
old_state = old_ckpt["state_dict"]

# Extract only backbone (encoder) weights
encoder_weights = {
    k.replace("net.backbone.", ""): v
    for k, v in old_state.items()
    if k.startswith("net.backbone.")
}

# Load into the new Sea-Thru model's backbone
new_model.net.backbone.load_state_dict(encoder_weights, strict=True)
print("Backbone weights restored. Heads will train from scratch.")
```

---

## 6. W&B Metric Schema After Migration

| Metric | Old name | New name | Notes |
|---|---|---|---|
| Reconstruction loss | `train/loss/recon` | `train/loss/recon` | Unchanged |
| Physics consistency | `train/loss/phys` | `train/loss/phys` | Now uses Sea-Thru forward model |
| Transmission supervision | `train/loss/t_sup` | *(removed)* | Replaced by `z_sup` |
| Depth supervision | *(absent)* | `val/loss/z_sup` | Only logged when `z_gt` present |
| Total Variation | `train/loss/tv` | `train/loss/tv` | Now applied to `z` not `t` |
| PSNR | `val/psnr` | `val/psnr` | Unchanged — still the EarlyStopping monitor |
| SSIM | `val/ssim` | `val/ssim` | Unchanged |

Add these to your W&B dashboard panel as a new section titled **"Sea-Thru Run"** to distinguish from previous Jaffe-McGlamery runs. Use W&B run tags: `["sea-thru", "m1", "desnow"]`.
