# Part 7: Sea-Thru Migration Checklist

A sequential, file-by-file migration guide from the Jaffe-McGlamery model to the Sea-Thru (Akkaynak-Treibitz) model. Each step includes a verification command or unit test to confirm correctness before moving on.

**Estimated total time:** ~4–6 hours for an experienced developer.

---

## Pre-Migration Checklist

Before making any changes:

- [ ] Create a git branch: `git checkout -b feat/sea-thru-migration`
- [ ] Run the current test suite and ensure it is fully green: `uv run pytest tests/ -v`
- [ ] Record baseline metrics from the last Jaffe-McGlamery W&B run (PSNR, SSIM) for comparison.
- [ ] Back up the last good checkpoint: copy to `outputs/jm_baseline/ckpts/last.ckpt`

---

## Step 1 — Physics Utilities (`utils/physics.py`)

**Goal:** Add Sea-Thru forward and inverse functions without breaking the existing Jaffe-McGlamery functions.

- [ ] Add `apply_forward_seathru(j, z, beta_d, beta_b, b_inf)` — see [Part 2](02_physics_utilities.md#1-implementing-the-forward-equation)
- [ ] Add `invert_seathru(i, z, beta_d, beta_b, b_inf, ...)` — see [Part 2](02_physics_utilities.md#2-implementing-the-inversion-equation)
- [ ] Keep the old Jaffe-McGlamery functions (`apply_underwater_degradation`, `invert_jaffe`) in place — **do not delete them yet**

**Verify:**

```python
# Run in a Python shell or add as tests/test_seathru_physics.py
import torch
from aquaclr.utils.physics import apply_forward_seathru, invert_seathru

B, H, W = 2, 64, 64
j     = torch.rand(B, 3, H, W)
z     = torch.rand(B, 1, H, W) * 5.0            # 0–5 m depth
beta_d = torch.tensor([[0.3, 0.15, 0.08]] * B)
beta_b = torch.tensor([[0.4, 0.20, 0.10]] * B)
b_inf  = torch.tensor([[0.1, 0.35, 0.60]] * B)

# Forward pass must stay in [0, 1]
i = apply_forward_seathru(j, z, beta_d, beta_b, b_inf)
assert i.shape == (B, 3, H, W)
assert i.min() >= 0.0 and i.max() <= 1.0, "Forward model out of [0,1] range"

# Inversion should approximately recover j (not exact due to the gate)
j_hat = invert_seathru(i, z, beta_d, beta_b, b_inf)
assert j_hat.shape == (B, 3, H, W)

# Shallow pixels (z < 2m) should invert cleanly; check MSE < 0.01
shallow_mask = (z < 2.0).expand_as(j)
mse_shallow = ((j[shallow_mask] - j_hat[shallow_mask]) ** 2).mean()
assert mse_shallow < 0.01, f"Inversion error too high for shallow pixels: {mse_shallow:.5f}"

print("Physics utilities: PASS")
```

---

## Step 2 — Model Heads

**Goal:** Replace `TransmissionHead` with `DepthHead` and `BackscatterHead` with `IlluminationHead`.

- [ ] Create `src/aquaclr/models/heads/depth.py` containing `DepthHead` — see [Part 3](03_network_architecture.md#depthhead)
- [ ] Create `src/aquaclr/models/heads/illumination.py` containing `IlluminationHead` — see [Part 3](03_network_architecture.md#illuminationhead)
- [ ] Update `src/aquaclr/models/heads/__init__.py` to export the new heads
- [ ] Do **not** delete `transmission.py` or `backscatter.py` yet

**Verify:**

```python
import torch
from aquaclr.models.heads.depth import DepthHead
from aquaclr.models.heads.illumination import IlluminationHead

depth_head = DepthHead(in_channels=16)
illum_head = IlluminationHead(in_channels=96, hidden=32)

feat_spatial = torch.rand(2, 16, 64, 64)
feat_deep    = torch.rand(2, 96, 8, 8)

z = depth_head(feat_spatial, target_size=(256, 256))
assert z.shape == (2, 1, 256, 256)
assert z.min() >= 0.0, "Depth must be non-negative"

beta_d, beta_b, b_inf = illum_head(feat_deep)
assert beta_d.shape == (2, 3)
assert b_inf.min() >= 0.0 and b_inf.max() <= 1.0, "b_inf must be in [0, 1]"

print("New heads: PASS")
```

---

## Step 3 — Output Dataclass and Model Forward Pass

**Goal:** Replace `LEGIONOutputs` (which has `.t` and `.b`) with `SeaThruOutputs` (`.z`, `.beta_d`, `.beta_b`, `.b_inf`).

- [ ] In `models/model.py`, add `SeaThruOutputs` dataclass — see [Part 3](03_network_architecture.md#1-update-the-data-structures)
- [ ] Update `LEGIONDeSnowNet.__init__()` to instantiate `DepthHead` and `IlluminationHead`
- [ ] Update `LEGIONDeSnowNet.forward()` to call `invert_seathru` and return `SeaThruOutputs`
- [ ] Keep `LEGIONOutputs` in place until the loss and lit_module are also updated

**Verify:**

```python
import torch
from aquaclr.models.model import LEGIONDeSnowNet

net = LEGIONDeSnowNet(backbone="mobilenet_v3_small", pretrained=False)
i = torch.rand(2, 3, 256, 256)
out = net(i)

assert hasattr(out, "j"),      "Missing output: j"
assert hasattr(out, "z"),      "Missing output: z"
assert hasattr(out, "beta_d"), "Missing output: beta_d"
assert hasattr(out, "beta_b"), "Missing output: beta_b"
assert hasattr(out, "b_inf"),  "Missing output: b_inf"

assert out.j.shape     == (2, 3, 256, 256)
assert out.z.shape     == (2, 1, 256, 256)
assert out.beta_d.shape == (2, 3)

print("Model forward pass: PASS")
```

---

## Step 4 — Physics-Informed Loss

**Goal:** Update `PhysicsInformedLoss` to use the Sea-Thru forward model and supervise $z$ instead of $t$.

- [ ] Replace `apply_underwater_degradation` → `apply_forward_seathru` in the physics consistency term
- [ ] Replace `total_variation(t)` → `total_variation(z)` in the TV term
- [ ] Rename `lambda_t` → `lambda_z` and `t_sup` → `z_sup` throughout
- [ ] Update `PhysicsLossOutputs.to_log_dict()` to use `z_sup` key — see [Part 4](04_physics_loss.md)

**Verify:**

```python
import torch
from aquaclr.losses.physics_loss import PhysicsInformedLoss

loss_fn = PhysicsInformedLoss(lambda_z=0.5, lambda_tv=1e-2)

B, H, W = 2, 64, 64
i      = torch.rand(B, 3, H, W)
j_pred = torch.rand(B, 3, H, W)
j_gt   = torch.rand(B, 3, H, W)
z      = torch.rand(B, 1, H, W).abs() * 5.0
beta_d = torch.rand(B, 3).abs() * 0.5
beta_b = torch.rand(B, 3).abs() * 0.5
b_inf  = torch.rand(B, 3)

out = loss_fn(i=i, j_pred=j_pred, j_gt=j_gt, z=z, beta_d=beta_d, beta_b=beta_b, b_inf=b_inf)

assert out.total.requires_grad, "Loss must be differentiable"
assert not torch.isnan(out.total), "Loss is NaN — check physics constants"
assert not torch.isinf(out.total), "Loss is infinite — check depth range / eps"

log = out.to_log_dict("train/loss/")
assert "train/loss/z_sup" in log, "z_sup must appear in log dict"
assert "train/loss/t_sup" not in log, "t_sup should be removed"

print("Physics loss: PASS")
```

---

## Step 5 — Training Integration (`lit_module.py` and configs)

**Goal:** Wire the updated model and loss into the Lightning training loop.

- [ ] Update `LEGIONDeSnowLitModule.forward()` — see [Part 5](05_training_integration.md#21-update-the-forward-method)
- [ ] Update `LEGIONDeSnowLitModule._shared_step()` — see [Part 5](05_training_integration.md#24-full-revised-_shared_step)
- [ ] Rename `lambda_t` → `lambda_z` in `configs/train/rtx_a3000_bf16.yaml`
- [ ] Rename `lambda_t` → `lambda_z` in `configs/train/rtx3050_bf16.yaml`
- [ ] Rename `backscatter_hidden` → `illumination_hidden` in `configs/model/legion_desnow_s.yaml`

**Verify:**

```bash
# A 1-batch sanity run (fast — no full epoch needed)
uv run python scripts/train.py \
    trainer.fast_dev_run=true \
    trainer.num_sanity_val_steps=0 \
    data=combined
```

Expected output: `Epoch 0: 100%` with no `NaN` or `KeyError` exceptions.

---

## Step 6 — Data Pipeline

**Goal:** Extend the batch schema to support optional `z_gt` and add the synthetic dataset.

- [ ] Add `seathru_collate_fn` to `src/aquaclr/data/transforms.py` — see [Part 5](05_training_integration.md#42-collation)
- [ ] Update `CombinedDataModule.train_dataloader()` to use `seathru_collate_fn`
- [ ] *(Optional)* Create `SyntheticUnderwaterDataset` in `src/aquaclr/data/synthetic_dataset.py` — see [Part 6](06_synthetic_data.md#51-syntheticunderwaterdataset)

**Verify:**

```python
from aquaclr.data.combined_datamodule import CombinedDataModule

dm = CombinedDataModule(...)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))

assert "i" in batch and "j" in batch
assert "has_z_gt" in batch
assert batch["i"].shape[1] == 3         # RGB
print(f"Batch keys: {list(batch.keys())}")
print("Data pipeline: PASS")
```

---

## Step 7 — Unit Test Suite Update

- [ ] Update `tests/test_physics_loss.py`: replace `t_sup` → `z_sup` assertions, add Sea-Thru-specific checks
- [ ] Update `tests/test_model.py`: replace `out.t` / `out.b` assertions with `out.z` / `out.beta_d` etc.
- [ ] Add `tests/test_seathru_physics.py` with the verification tests from Steps 1–4 above
- [ ] Run full suite: `uv run pytest tests/ -v --tb=short`

All tests must pass before proceeding to training.

---

## Step 8 — First Training Run

- [ ] Start a short training run (5 epochs) to confirm training stability:

```bash
uv run python scripts/train.py \
    train=rtx_a3000_bf16 \
    data=combined_a3000 \
    max_epochs=5 \
    project_name=AquaCLR-SeaThru-Debug
```

- [ ] Monitor W&B for `NaN` losses, especially in the `phys` and `z_sup` terms.
- [ ] Confirm `val/psnr` is improving (even 5 epochs should show a positive trend from the pretrained backbone).
- [ ] Confirm `train/loss/tv` is non-zero (sanity check that TV is applied to `z`).

**Common pitfall:** If `val/psnr` is stuck at ~15 dB and not improving, check that the backbone weights were either loaded from the Jaffe-McGlamery checkpoint or that `freeze_backbone_epochs` is set to 2 to allow the heads to warm up first.

---

## Step 9 — Cleanup (After Successful Training Confirmed)

Only after training is verified:

- [ ] Delete `src/aquaclr/models/heads/transmission.py`
- [ ] Delete `src/aquaclr/models/heads/backscatter.py`
- [ ] Remove `LEGIONOutputs` dataclass from `models/model.py`
- [ ] Remove Jaffe-McGlamery functions from `utils/physics.py` if no longer referenced
- [ ] Update `models/heads/__init__.py` to remove old exports
- [ ] Run `uv run pytest tests/ -v` one final time

---

## Common Pitfalls and Debugging Tips

| Symptom | Likely Cause | Fix |
|---|---|---|
| `loss/phys` is `NaN` | `beta_d` or `z` contains zeros causing division by near-zero | Verify `eps` floor in `invert_seathru`; check `Softplus` activation on depth |
| `loss/tv` is 0.0 | TV still computed on `t` (old code not removed) | Confirm `total_variation(z)` in loss forward |
| `val/psnr` < 12 dB after 10 epochs | Heads training from scratch with backbone also unfrozen | Set `freeze_backbone_epochs: 2` in model config |
| `KeyError: 'z_gt'` in loss | `seathru_collate_fn` not registered in DataLoader | Add `collate_fn=seathru_collate_fn` to both train and val dataloaders |
| Depth map all zeros | `DepthHead` using `Sigmoid` instead of `Softplus` | Replace final activation in `DepthHead` |
| `b_inf` values > 1.0 | `IlluminationHead` not applying `sigmoid` to `b_inf` output | Add `torch.sigmoid` to the `b_inf` slice in `IlluminationHead.forward()` |
| OOM after migration | `beta_d`, `beta_b`, `b_inf` being kept on GPU unnecessarily | Detach non-gradient tensors during logging; check `to_log_dict` uses `.detach()` |
