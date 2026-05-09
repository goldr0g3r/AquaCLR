"""Dataset / synthesiser tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from aquaclr.data.msrb_dataset import MSRBDataset
from aquaclr.data.snow_synthesis import synthesize_marine_snow


def _make_fake_msrb(root: Path, n: int = 4, *, with_noisy: bool = False) -> None:
    train_clean = root / "train" / "clean"
    train_clean.mkdir(parents=True, exist_ok=True)
    if with_noisy:
        (root / "train" / "noisy").mkdir(parents=True, exist_ok=True)
    for k in range(n):
        img = (np.random.default_rng(seed=k).random((64, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(train_clean / f"img_{k:03d}.png")
        if with_noisy:
            Image.fromarray(img).save(root / "train" / "noisy" / f"img_{k:03d}.png")


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


def test_msrb_paired_files(tmp_path: Path) -> None:
    _make_fake_msrb(tmp_path, n=3, with_noisy=True)
    ds = MSRBDataset(tmp_path, split="train", transform=None)
    assert len(ds) == 3
    sample = ds[1]
    assert isinstance(sample["i"], torch.Tensor)


def test_msrb_missing_clean_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MSRBDataset(tmp_path, split="train")
