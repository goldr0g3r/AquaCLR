# Appendix C — Code Reference

A guided walk through the most important source files. Each entry has:

- **Path** — markdown link to the file.
- **Public surface** — symbols a downstream user touches.
- **Key invariants** — what the file guarantees so callers can trust it.
- **Annotated extract** — a representative snippet with line numbers
  pointing into the canonical version on disk.

## C.1 The model — `LEGIONDeSnowNet`

**Path:** [`src/aquaclr/models/model.py`](../../src/aquaclr/models/model.py)

**Public surface:**

| Symbol | Type | Use |
| --- | --- | --- |
| `LEGIONDeSnowNet` | `nn.Module` | The trained network |
| `LEGIONOutputs` | `dataclass` | `(j, t, b)` carrier |

**Key invariants:**

1. `forward(I)` returns a `LEGIONOutputs` with all three tensors in
   `[0, 1]`.
2. `forward_export(I)` returns the same triple as a flat tuple,
   ONNX-exportable.
3. `num_parameters` and `estimate_size_mb(dtype)` are stable across
   versions — used by `tests/test_model.py` to assert the size budget.

**Annotated extract:**

```python
class LEGIONDeSnowNet(nn.Module):
    SUPPORTED_BACKBONES = ("mobilenet_v3_small",)

    def __init__(self, *, backbone="mobilenet_v3_small", pretrained=True,
                 out_indices=(0, 1, 2, 3), decoder_channels=(96, 64, 32, 16),
                 backscatter_hidden=32, use_depthwise=True,
                 use_channels_last=True, eps=1e-3, normalize_input=True,
                 freeze_backbone_epochs=0):
        # Encoder, decoder, two heads. All sized to fit the 4 GB ceiling.
        ...

    def forward(self, i):
        feat_full, feat_deep = self._encode_decode(i)
        t = self.transmission_head(feat_full, target_size=i.shape[-2:])
        b = self.backscatter_head(feat_deep)
        j = invert_jaffe_mcglamery(i, t, b, eps=self.eps)
        return LEGIONOutputs(j=j, t=t, b=b)
```

## C.2 Physics ops — `apply_forward_jaffe_mcglamery`, `invert_jaffe_mcglamery`

**Path:** [`src/aquaclr/utils/physics.py`](../../src/aquaclr/utils/physics.py)

**Public surface:** the two free functions above.

**Key invariants:** round-trip identity holds — see
[`tests/test_physics_loss.py::test_physics_round_trip_is_identity_on_clean_inputs`](../../tests/test_physics_loss.py).

**Annotated extract:**

```python
def apply_forward_jaffe_mcglamery(j, t, b):
    if b.dim() == 2:
        b = b.unsqueeze(-1).unsqueeze(-1)
    return (j * t + b * (1.0 - t)).clamp(0.0, 1.0)

def invert_jaffe_mcglamery(i, t, b, *, eps=1e-3):
    if b.dim() == 2:
        b = b.unsqueeze(-1).unsqueeze(-1)
    t_safe = t.clamp(min=eps)
    return ((i - b * (1.0 - t)) / t_safe).clamp(0.0, 1.0)
```

The `eps` clamp is the single line that prevents the network from
diverging in `t → 0` regions.

## C.3 Composite loss — `PhysicsInformedLoss`

**Path:** [`src/aquaclr/losses/physics_loss.py`](../../src/aquaclr/losses/physics_loss.py)

**Public surface:** `PhysicsInformedLoss`, `PhysicsLossOutputs`.

**Key invariants:**

1. Total loss is differentiable everywhere; gradients are finite.
2. When `t_gt` is `None` the `L_t` term is exactly zero (not NaN).

**Annotated extract:**

```python
def forward(self, i, j_pred, j_gt, t, b, *, t_gt=None):
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
    total = (self.lambda_recon * recon
             + self.lambda_phys  * phys
             + self.lambda_ssim  * ssim_loss
             + self.lambda_tv    * tv
             + self.lambda_t     * t_sup)
    return PhysicsLossOutputs(...)
```

## C.4 SSIM — differentiable, ONNX-clean

**Path:** [`src/aquaclr/losses/ssim.py`](../../src/aquaclr/losses/ssim.py)

**Public surface:** `SSIM` (Module), `ssim` (function), `psnr`
(function).

**Key invariants:**

1. `SSIM(x, x) == 1.0` exactly (tested).
2. The Gaussian window is registered as a non-persistent buffer so
   `module.to(device)` moves it correctly.

## C.5 Total variation — anisotropic L1

**Path:** [`src/aquaclr/losses/tv.py`](../../src/aquaclr/losses/tv.py)

**Public surface:** `total_variation(x, reduction="mean")`.

**Key invariants:** `total_variation(constant_tensor) == 0`.

## C.6 Encoder — `MobileNetV3SmallEncoder`

**Path:** [`src/aquaclr/models/backbones/mobilenet_v3.py`](../../src/aquaclr/models/backbones/mobilenet_v3.py)

**Public surface:** `MobileNetV3SmallEncoder`,
`MobileNetV3SmallStageInfo`, `imagenet_normalize`.

**Key invariants:** `forward(x)` returns exactly `len(out_indices)`
tensors in stride order.

## C.7 Decoder — `UNetDSCDecoder`

**Path:** [`src/aquaclr/models/decoders/unet_dsc.py`](../../src/aquaclr/models/decoders/unet_dsc.py)

**Public surface:** `DepthwiseSeparableConv`, `UNetDSCDecoder`.

**Key invariants:** robust to non-divisible spatial sizes (defends
with `F.interpolate(x, size=skip.shape[-2:])`).

## C.8 Datasets — `MSRBDataset`, `LSUIDataset`, `CombinedDataModule`

**Paths:**

- [`src/aquaclr/data/msrb_dataset.py`](../../src/aquaclr/data/msrb_dataset.py)
- [`src/aquaclr/data/lsui_dataset.py`](../../src/aquaclr/data/lsui_dataset.py)
- [`src/aquaclr/data/combined_datamodule.py`](../../src/aquaclr/data/combined_datamodule.py)

**Public surface:** the three classes plus their `LightningDataModule`
counterparts.

**Key invariants:**

1. Per-batch `has_t_gt` flag is homogeneous within a batch.
2. Train/val splits are seeded reproducibly.
3. Every dataset raises `FileNotFoundError` with a descriptive message
   if its expected layout isn't met.

## C.9 Augmentations — `build_train_transform`, `build_val_transform`

**Path:** [`src/aquaclr/data/transforms.py`](../../src/aquaclr/data/transforms.py)

**Public surface:** the two functions above.

**Key invariants:**

1. Geometric transforms are atomic over `(I, J, t_gt)` triples.
2. Photometric jitter is applied **only to `I`**; `J` stays as a
   faithful radiance reference.

## C.10 Snow synthesis — `synthesize_marine_snow`

**Path:** [`src/aquaclr/data/snow_synthesis.py`](../../src/aquaclr/data/snow_synthesis.py)

**Public surface:** the function `synthesize_marine_snow(image, ...)`.

**Key invariants:**

1. Output shape and dtype match input (`HxWx3 uint8`).
2. Output is **deterministic** given a fixed `seed`.
3. Output is **never identical** to input when `n_particles[0] > 0`.

## C.11 Lightning module

**Path:** [`src/aquaclr/training/lit_module.py`](../../src/aquaclr/training/lit_module.py)

**Public surface:** `LEGIONDeSnowLitModule`.

**Key invariants:** wraps an `nn.Module` `net` and a
`PhysicsInformedLoss` `loss` — `net` is the canonical export target
(extracted by `scripts/export_onnx.py` after stripping the
`net._orig_mod.` and `net.` prefixes from the checkpoint state-dict).

## C.12 Callbacks — EMA, VRAM monitor, sample logger

**Path:** [`src/aquaclr/training/callbacks.py`](../../src/aquaclr/training/callbacks.py)

**Public surface:** `EMAWeightCallback`, `VRAMMonitor`,
`SampleImageLogger`.

**Key invariants:** EMA weights are swapped in for validation and
swapped back; never persist into the optimiser's view of the
parameters.

## C.13 ONNX export — `export_to_onnx`

**Path:** [`src/aquaclr/inference/onnx_export.py`](../../src/aquaclr/inference/onnx_export.py)

**Public surface:** `export_to_onnx(model, output_path, ...)`.

**Key invariants:**

1. The exported graph passes ONNXRuntime parity check at `atol=1e-3`,
   else raises.
2. Dynamic axes for `B, H, W` on inputs and matching outputs.

## C.14 TensorRT — `build_engine_from_onnx`, `TensorRTRunner`

**Path:** [`src/aquaclr/inference/inference_trt.py`](../../src/aquaclr/inference/inference_trt.py)

**Public surface:** the two symbols above.

**Key invariants:**

1. Engine build is deterministic for a given `(GPU arch, driver, TRT
   version)` triple.
2. The runner accepts any spatial size in the build's
   `(min, max)` profile and returns `uint8` HWC output.

## C.15 Benchmark — `benchmark_pytorch`

**Path:** [`src/aquaclr/inference/benchmark.py`](../../src/aquaclr/inference/benchmark.py)

**Public surface:** `benchmark_pytorch(model, input_shape, ...)` →
`BenchmarkResult`.

**Key invariants:** measures only model inference; CUDA-syncs before
each timing read.

## C.16 ROS 2 node — `LegionDeSnowNode`

**Path:** [`src/aquaclr/ros2/ros2_node.py`](../../src/aquaclr/ros2/ros2_node.py)

**Public surface:** `LegionDeSnowNode`, `main`.

**Key invariants:**

1. Imports `rclpy` lazily — package remains importable on
   non-ROS systems.
2. Falls back from `trt` → `torch` if the engine fails to load,
   logging clearly.
3. Publishes at the same rate as the input topic, up to the
   inference cap.

## C.17 Test suite — what each file asserts

**Path:** [`tests/`](../../tests/)

| Test file | Asserts |
| --- | --- |
| `test_model.py` | shapes; `[0,1]` outputs; FP32 size ≤ 50 MB; gradient finiteness; non-divisible input handling |
| `test_physics_loss.py` | round-trip identity; gradient finiteness; `t_gt` toggle; perfect-signal loss values |
| `test_ssim_tv.py` | SSIM(x,x)=1; SSIM decreases with noise; TV(constant)=0; TV(random)>0 |
| `test_data.py` | snow synth deterministic; MSRB synth-fallback; paired files load; missing-clean raises |
| `test_export.py` | ONNX parity (CPU); ONNX dynamic shapes; PyTorch benchmark runs (gpu); TRT round-trip (trt) |

## C.18 Hydra configs — composition tree

**Paths:** [`configs/`](../../configs/)

```
configs/default.yaml             # composes the 3 below
├─ model/legion_desnow_s.yaml    # network hyper-parameters
├─ data/msrb.yaml                # MSRB only
├─ data/lsui.yaml                # LSUI only
├─ data/combined.yaml            # MSRB + LSUI per-batch mix (default)
└─ train/rtx3050_bf16.yaml       # optimizer, scheduler, loss, callbacks
```

To **change a hyper-parameter** for a single run, override at the CLI:

```bash
python scripts/train.py train.optimizer.lr=1e-4
```

To **change a hyper-parameter** as the new default, edit the relevant
YAML file. Configs are version-controlled alongside code so any
historical run can be reconstructed.

## C.19 Scripts

**Paths:** [`scripts/`](../../scripts/)

| Script | Purpose |
| --- | --- |
| `train.py` | Hydra entry point for training |
| `evaluate.py` | PSNR/SSIM evaluation on a checkpoint |
| `export_onnx.py` | Checkpoint → ONNX (+ optional TRT engine + benchmark) |
| `download_data.py` | Print dataset fetch instructions and verify on-disk layout |

Each script is intentionally **thin**: the real logic lives in `src/`.
This keeps the dissertation easy to navigate — a reviewer reads `src/`
to understand *what*, and reads `scripts/` to understand *how to
invoke*.

## C.20 Code metrics summary

| Metric | Value (approximate) |
| --- | --- |
| Source files (`.py`) under `src/aquaclr/` | 25 |
| Lines of code under `src/aquaclr/` | ~2,400 |
| Public classes | 12 |
| Public functions | ~30 |
| Test files | 5 |
| Test cases | ~22 |
| Avg. file length | < 200 lines |
| Type-coverage (mypy --strict) | 100 % on `src/aquaclr/` |
| Lint clean (ruff) | yes (CI-enforced) |
| Docstring coverage | 100 % on public surface (Google style) |

This is the artefact a viva-time examiner clones, builds, and runs.
