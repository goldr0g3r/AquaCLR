"""Dataset / synthesiser tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from aquaclr.data.msrb_dataset import MSRBDataset
from aquaclr.data.snow_synthesis import synthesize_marine_snow


def _make_fake_msrb(
    root: Path,
    n: int = 4,
    *,
    with_noisy: bool = False,
    task: int = 1,
    layout: str = "upstream",
) -> None:
    """Materialise a tiny MSRB-style dataset on disk.

    Args:
        root: Dataset root.
        n: Number of (clean, noisy) pairs to create.
        with_noisy: If True, also write the snowy variant directory.
        task: 1 or 2; controls the snowy variant folder name.
        layout: ``"upstream"`` (``training/original`` + ``training/MSR_TaskN``)
            or ``"legacy"`` (``train/clean`` + ``train/noisy``).
    """
    if layout == "upstream":
        split_dir = root / "training"
        clean_dir = split_dir / "original"
        noisy_dir = split_dir / f"MSR_Task{task}"
    elif layout == "legacy":
        split_dir = root / "train"
        clean_dir = split_dir / "clean"
        noisy_dir = split_dir / "noisy"
    else:
        raise ValueError(f"unknown layout {layout!r}")

    clean_dir.mkdir(parents=True, exist_ok=True)
    if with_noisy:
        noisy_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        img = (np.random.default_rng(seed=k).random((64, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(clean_dir / f"img_{k:03d}.png")
        if with_noisy:
            Image.fromarray(img).save(noisy_dir / f"img_{k:03d}.png")


def test_synthesize_marine_snow_changes_image() -> None:
    rng = np.random.default_rng(0)
    img = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
    snow = synthesize_marine_snow(img, seed=7)
    assert snow.shape == img.shape
    assert snow.dtype == np.uint8
    assert not np.array_equal(snow, img)


def test_msrb_synth_fallback(tmp_path: Path) -> None:
    _make_fake_msrb(tmp_path, n=2, with_noisy=False)
    ds = MSRBDataset(tmp_path, split="train", transform=None, synthesize_if_missing=True)
    sample = ds[0]
    assert sample["i"].shape == (3, 64, 64)
    assert sample["j"].shape == (3, 64, 64)
    assert sample["has_t_gt"].item() is False


def test_msrb_paired_files_upstream(tmp_path: Path) -> None:
    _make_fake_msrb(tmp_path, n=3, with_noisy=True, task=1, layout="upstream")
    ds = MSRBDataset(tmp_path, split="train", task=1, transform=None)
    assert len(ds) == 3
    sample = ds[1]
    assert isinstance(sample["i"], torch.Tensor)
    assert ds.noisy_dir is not None and ds.noisy_dir.name == "MSR_Task1"
    assert ds.clean_dir.name == "original"


def test_msrb_paired_files_task2(tmp_path: Path) -> None:
    _make_fake_msrb(tmp_path, n=2, with_noisy=True, task=2, layout="upstream")
    ds = MSRBDataset(tmp_path, split="train", task=2, transform=None)
    assert len(ds) == 2
    assert ds.noisy_dir is not None and ds.noisy_dir.name == "MSR_Task2"


def test_msrb_paired_files_legacy(tmp_path: Path) -> None:
    _make_fake_msrb(tmp_path, n=3, with_noisy=True, layout="legacy")
    ds = MSRBDataset(tmp_path, split="train", transform=None)
    assert len(ds) == 3
    assert ds.clean_dir.name == "clean"
    assert ds.noisy_dir is not None and ds.noisy_dir.name == "noisy"


def test_msrb_missing_clean_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MSRBDataset(tmp_path, split="train")


def test_msrb_invalid_task_raises(tmp_path: Path) -> None:
    _make_fake_msrb(tmp_path, n=1, with_noisy=False)
    with pytest.raises(ValueError, match="task must be 1 or 2"):
        MSRBDataset(tmp_path, split="train", task=3)
