# Appendix D — Reproducibility Checklist

This appendix is a **submission checklist** for a hypothetical
external reviewer who wants to reproduce every numerical claim in
Chapter 10 from scratch. It follows the spirit of the
[ML Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf).

## D.1 What you need before starting

| Item | Detail |
| --- | --- |
| OS | Fedora 44, Ubuntu 22.04 LTS, or Ubuntu 24.04 LTS |
| Python | 3.10, 3.11, or 3.12 |
| GPU | NVIDIA RTX 3050 (Ampere, 4 GB) or larger Ampere/Ada |
| Driver | NVIDIA ≥ 555 |
| Disk | ≥ 20 GB free (datasets + outputs) |
| Network | Required only for initial dataset download |
| Time | ≤ 12 hours per training run on the reference GPU |

If running on Linux + Ampere, expect everything to work as
described. On non-Linux or non-Ampere systems, the TRT path is
unavailable; the rest works.

## D.2 Step-by-step reproduction recipe

```bash
# 1. Clone and install.
git clone <repository-url> aquaclr
cd aquaclr
git rev-parse HEAD                     # record this for the manifest

uv venv .venv --python 3.11
source .venv/bin/activate
uv sync --extra dev --extra trt        # 'trt' optional on non-Linux

# 2. Run pre-flight tests.
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q -m "not gpu and not trt"

# 3. Get datasets.
uv run python scripts/download_data.py # prints fetch instructions
# ... follow instructions for MSRB and LSUI ...
uv run python scripts/download_data.py # re-run to verify layout

# 4. Train baseline (~10 hours on RTX 3050).
uv run python scripts/train.py train.run_name=A0_baseline

# 5. Run ablations (each ~10 hours).
for tag in A1 A4 A8; do
    uv run python scripts/train.py train.run_name=$tag <ablation-flags>
done

# 6. Evaluate on test sets.
for run in outputs/A0_baseline outputs/A1 outputs/A4 outputs/A8; do
    uv run python scripts/evaluate.py --ckpt $run/ckpts/last.ckpt \
        --data-root data/msrb --split test --task 1
done

# 7. Export + benchmark.
uv run python scripts/export_onnx.py \
    --ckpt outputs/A0_baseline/ckpts/best.ckpt \
    --build-trt --benchmark \
    --height 720 --width 1280

# 8. (Optional) Run inside ROS 2 container.
distrobox-create --name legion-ros2 --image docker.io/library/ubuntu:24.04 --nvidia
distrobox-enter legion-ros2
# (in container) install ROS 2 Jazzy per docs/DEPLOYMENT_FEDORA.md §5
```

## D.3 Pre-experiment checklist

Before clicking "go" on a training run:

- [ ] `git status` clean (`git stash` any local changes first).
- [ ] `git rev-parse HEAD` recorded.
- [ ] Dataset MD5 sums recorded (`md5sum data/msrb/...` and similar).
- [ ] `nvidia-smi` shows expected GPU + driver.
- [ ] `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` matches expected.
- [ ] `uv run ruff check .` and `uv run mypy` clean.
- [ ] `uv run pytest -q -m "not gpu and not trt"` all green.
- [ ] Run name set: `train.run_name=<descriptive_tag>`.
- [ ] Logging sink chosen (TensorBoard always; W&B if `WANDB_API_KEY` set).

## D.4 Hyper-parameter recording

Hydra automatically saves the **resolved** config at run start to:

```
outputs/<run>/.hydra/config.yaml
```

This is the **single source of truth** for "what hyper-parameters
produced this number". If the config can't be reconstructed, the
result is not citable.

## D.5 Random seed records

Every run logs:

- `seed = 1337` (default; override with `seed=`)
- `deterministic = True`

If you change the seed, change the run name to match (e.g.
`A0_baseline_seed42`).

## D.6 Hardware fingerprint

Recorded automatically into the run log on first epoch:

```
GPU: NVIDIA GeForce RTX 3050  | driver 555.42.06 | CUDA 12.4
PyTorch: 2.5.1                | Lightning: 2.4.0
TRT (if --build-trt): 10.0.0.6
OS: Fedora release 44 (Sphinx) | uname: Linux 6.10.x
```

## D.7 Dataset version control

For each dataset, record its **MD5 fingerprint after extraction**:

```bash
find data/msrb -type f -name "*.png" | sort | xargs md5sum > data/msrb/MANIFEST.md5
md5sum data/msrb/MANIFEST.md5         # the "version" of MSRB on this machine
```

Repeat for LSUI and UIEB.

## D.8 Reproducibility of compiled engines

A `.engine` file is portable across machines **only if all of**:

- GPU architecture matches (e.g. SM_86 for RTX 3050).
- TensorRT major and minor versions match.
- Driver version is within the supported range.

Otherwise, regenerate with `scripts/export_onnx.py --build-trt`.
This is the only step in the pipeline that is *not* purely
deterministic from source code.

## D.9 Failure modes you may hit and what they mean

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `pytest` fails on `test_size_budget` | Modified the model and pushed param count over 50 MB | Reduce decoder channels or backbone (revert if unintentional) |
| `pytest` fails on `test_physics_round_trip_is_identity_on_clean_inputs` | Edits to `apply_forward_jaffe_mcglamery` or `invert_jaffe_mcglamery` broke their identity | Verify `eps` clamp; check sign of `1 - t` term |
| Training diverges in epoch 1 (loss → NaN) | Likely FP16 (not BF16); `t → 0` causing inversion blowup | Switch to `precision="bf16-mixed"`; check `gradient_clip_val` |
| `val/psnr` plateaus < 22 dB | Backbone never unfreezing or LR too low | Check `freeze_backbone_epochs`; verify `OneCycleLR` is being used |
| ONNX parity check fails | Operation not yet supported by ONNX exporter at chosen opset | Set `opset=18`; if still failing, file an issue with the specific op |
| TRT engine build fails with "no FP16 support" | GPU lacks tensor cores | Build without FP16 (`fp16=False`); accept ~2× slowdown |
| ROS 2 node sees no images | Topic name mismatch | `ros2 topic list`; verify `input_topic` parameter |

## D.10 Reproducibility report template

For your dissertation submission, include a small table at the
end of Chapter 10 with the following filled in:

| Field | Value |
| --- | --- |
| Git revision (HEAD at evaluation) | `<sha>` |
| Python version | `<x.y.z>` |
| PyTorch version | `<x.y.z>` |
| CUDA version | `<x.y>` |
| Driver version | `<x.y.z>` |
| TRT version | `<x.y.z>` |
| GPU | `<model>` |
| OS | `<distro version>` |
| MSRB MANIFEST.md5 | `<32-char hex>` |
| LSUI MANIFEST.md5 | `<32-char hex>` |
| UIEB-Challenge MANIFEST.md5 | `<32-char hex>` |
| Random seed | `1337` |
| `seed_everything(deterministic=...)` | `True` |
| Total wall-clock (training + eval) | `<HH:MM>` |
| Total disk produced | `<MB>` |

## D.11 Citation of this codebase

```bibtex
@misc{aquaclr2026,
  title       = {AquaCLR: A Physics-Informed CNN for Real-Time Marine Snow
                 Removal on Edge GPUs},
  author      = {Project LEGION},
  year        = {2026},
  howpublished= {Project LEGION Milestone 1 deliverable},
  note        = {git revision available on request}
}
```
