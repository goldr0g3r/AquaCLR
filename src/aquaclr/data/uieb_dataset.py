"""UIEB (Underwater Image Enhancement Benchmark) dataset and DataModule.

Reference:
    Li, C. et al. *An Underwater Image Enhancement Benchmark Dataset and
    Beyond*, IEEE TIP 2020.

Layout we expect on disk:

.. code-block::

    root/
    ├─ raw-890/        # degraded underwater images (input I)
    ├─ reference-890/  # corresponding enhanced references (GT J)
    └─ challenging-60/ # extra hard images with NO reference (test-only)

Files in ``raw-890/`` and ``reference-890/`` are paired by **filename stem**.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import lightning as L

    _LIGHTNING_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTNING_AVAILABLE = False
    L = None  # type: ignore[assignment]

from aquaclr.data.msrb_dataset import _collate
from aquaclr.data.transforms import build_train_transform, build_val_transform


class UIEBDataset(Dataset[dict[str, Tensor]]):
    """UIEB paired underwater enhancement dataset.

    Args:
        root: Path to the unpacked UIEB folder.
        transform: Albumentations-style transform accepting ``image=``
            and ``image_clean=``.
    """

    SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def __init__(
        self,
        root: str | Path,
        *,
        transform: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.input_dir = self.root / "raw-890"
        self.gt_dir = self.root / "reference-890"

        if not self.input_dir.exists() or not self.gt_dir.exists():
            msg = (
                f"UIEB 'raw-890' and/or 'reference-890' directories not found under {self.root}. "
                "Download the dataset first."
            )
            raise FileNotFoundError(msg)

        inputs = {
            p.stem: p
            for p in self.input_dir.iterdir()
            if p.suffix.lower() in self.SUFFIXES
        }
        gts = {
            p.stem: p
            for p in self.gt_dir.iterdir()
            if p.suffix.lower() in self.SUFFIXES
        }
        common = sorted(set(inputs) & set(gts))
        if not common:
            msg = f"No paired stems between {self.input_dir} and {self.gt_dir}"
            raise FileNotFoundError(msg)

        self.samples: list[tuple[Path, Path]] = [(inputs[s], gts[s]) for s in common]

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: Path) -> NDArray[np.uint8]:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        in_path, gt_path = self.samples[idx]
        i_np = self._load_rgb(in_path)
        j_np = self._load_rgb(gt_path)

        if self.transform is not None:
            transformed = self.transform(image=i_np, image_clean=j_np)
            i_t: Tensor = transformed["image"]
            j_t: Tensor = transformed["image_clean"]
        else:
            i_t = torch.from_numpy(i_np).permute(2, 0, 1).float() / 255.0
            j_t = torch.from_numpy(j_np).permute(2, 0, 1).float() / 255.0

        return {
            "i": i_t,
            "j": j_t,
            "has_t_gt": torch.tensor(False),
            "source": "uieb",
        }


class UIEBDataModule(L.LightningDataModule if _LIGHTNING_AVAILABLE else object):  # type: ignore[misc]
    """Lightning DataModule wrapping :class:`UIEBDataset`."""

    def __init__(
        self,
        root: str | Path,
        *,
        image_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        val_fraction: float = 0.1,
        augment: bool = True,
        download: bool = False,
        seed: int = 1337,
    ) -> None:
        if _LIGHTNING_AVAILABLE:
            super().__init__()
        self.root = Path(root)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.val_fraction = float(val_fraction)
        self.augment = bool(augment)
        self.download = bool(download)
        self.seed = int(seed)
        self._train: Dataset[dict[str, Tensor]] | None = None
        self._val: Dataset[dict[str, Tensor]] | None = None

    def prepare_data(self) -> None:
        if not self.download:
            return
        if self.root.exists() and any(self.root.iterdir()):
            return
        msg = (
            "Automatic UIEB download is not implemented. "
            "Download raw-890 and reference-890 manually."
        )
        raise NotImplementedError(msg)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        train_tf = (
            build_train_transform(self.image_size)
            if self.augment
            else build_val_transform(self.image_size)
        )
        val_tf = build_val_transform(self.image_size)

        full = UIEBDataset(self.root, transform=train_tf)
        n_val = max(1, int(len(full) * self.val_fraction))
        n_train = len(full) - n_val
        gen = torch.Generator().manual_seed(self.seed)
        train_split, val_split = random_split(full, [n_train, n_val], generator=gen)

        val_dataset = UIEBDataset(self.root, transform=val_tf)
        val_dataset.samples = [val_dataset.samples[i] for i in val_split.indices]

        self._train = train_split
        self._val = val_dataset

    def _loader(
        self, ds: Dataset[dict[str, Tensor]] | None, *, shuffle: bool
    ) -> DataLoader[dict[str, Tensor]]:
        if ds is None:
            msg = "DataModule.setup() must run before requesting a dataloader"
            raise RuntimeError(msg)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=shuffle,
            collate_fn=_collate,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._loader(self._val, shuffle=False)
