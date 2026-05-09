"""LSUI dataset and DataModule.

Reference:
    Peng, L. et al. *U-shape Transformer for Underwater Image
    Enhancement*, 2021. Project page:
    https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html

Layout we expect on disk (after the user has downloaded and unpacked
the archive into ``root``):

.. code-block::

    root/
    ├─ input/         # raw underwater I, jpg/png
    ├─ GT/            # paired enhanced reference J, same filename stem
    └─ transmission/  # optional: paired transmission GT t (single-channel PNG)

Files in the three folders are paired by **filename stem**.
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


class LSUIDataset(Dataset[dict[str, Tensor]]):
    """LSUI underwater dataset with optional transmission ground truth.

    Args:
        root: LSUI root directory.
        transform: Albumentations-style transform accepting ``image=``,
            ``image_clean=``, and (optionally) ``transmission=`` (mask).
        load_transmission_gt: If True and the ``transmission/`` folder is
            present, the dataset yields ``t`` as ground truth.
    """

    SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def __init__(
        self,
        root: str | Path,
        *,
        transform: Callable[..., dict[str, Any]] | None = None,
        load_transmission_gt: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.input_dir = self.root / "input"
        self.gt_dir = self.root / "GT"
        self.t_dir = self.root / "transmission"
        self.load_transmission_gt = bool(load_transmission_gt) and self.t_dir.exists()

        if not self.input_dir.exists() or not self.gt_dir.exists():
            msg = (
                f"LSUI 'input' and/or 'GT' directories not found under {self.root}. "
                "See MODEL_CARD.md for fetch instructions."
            )
            raise FileNotFoundError(msg)

        inputs = {p.stem: p for p in self.input_dir.iterdir() if p.suffix.lower() in self.SUFFIXES}
        gts = {p.stem: p for p in self.gt_dir.iterdir() if p.suffix.lower() in self.SUFFIXES}
        common = sorted(set(inputs) & set(gts))
        if not common:
            msg = f"No paired stems between {self.input_dir} and {self.gt_dir}"
            raise FileNotFoundError(msg)

        ts: dict[str, Path] = {}
        if self.load_transmission_gt:
            ts = {p.stem: p for p in self.t_dir.iterdir() if p.suffix.lower() in self.SUFFIXES}

        self.samples: list[tuple[Path, Path, Path | None]] = [
            (inputs[s], gts[s], ts.get(s)) for s in common
        ]

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: Path) -> NDArray[np.uint8]:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))

    @staticmethod
    def _load_gray(path: Path) -> NDArray[np.float32]:
        with Image.open(path) as img:
            return np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        in_path, gt_path, t_path = self.samples[idx]
        i_np = self._load_rgb(in_path)
        j_np = self._load_rgb(gt_path)
        t_np: NDArray[np.float32] | None = self._load_gray(t_path) if t_path is not None else None

        if self.transform is not None:
            kwargs: dict[str, Any] = {"image": i_np, "image_clean": j_np}
            if t_np is not None:
                kwargs["transmission"] = t_np
            transformed = self.transform(**kwargs)
            i_t: Tensor = transformed["image"]
            j_t: Tensor = transformed["image_clean"]
            t_t: Tensor | None
            if t_np is not None:
                raw = transformed["transmission"]
                if isinstance(raw, torch.Tensor):
                    t_t = raw
                else:
                    t_t = torch.from_numpy(np.asarray(raw, dtype=np.float32))
                if t_t.dim() == 2:
                    t_t = t_t.unsqueeze(0)
            else:
                t_t = None
        else:
            i_t = torch.from_numpy(i_np).permute(2, 0, 1).float() / 255.0
            j_t = torch.from_numpy(j_np).permute(2, 0, 1).float() / 255.0
            t_t = (
                torch.from_numpy(t_np).unsqueeze(0) if t_np is not None else None
            )

        sample: dict[str, Tensor] = {
            "i": i_t,
            "j": j_t,
            "has_t_gt": torch.tensor(t_t is not None),
            "source": "lsui",
        }
        if t_t is not None:
            sample["t_gt"] = t_t.float()
        return sample


class LSUIDataModule(L.LightningDataModule if _LIGHTNING_AVAILABLE else object):  # type: ignore[misc]
    """Lightning DataModule wrapping :class:`LSUIDataset`."""

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
        load_transmission_gt: bool = True,
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
        self.load_transmission_gt = bool(load_transmission_gt)
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
            "Automatic LSUI download is not implemented. Follow MODEL_CARD.md "
            "to fetch and unpack into the configured root."
        )
        raise NotImplementedError(msg)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        train_tf = build_train_transform(self.image_size) if self.augment else build_val_transform(self.image_size)
        val_tf = build_val_transform(self.image_size)

        full = LSUIDataset(
            self.root,
            transform=train_tf,
            load_transmission_gt=self.load_transmission_gt,
        )
        n_val = max(1, int(len(full) * self.val_fraction))
        n_train = len(full) - n_val
        gen = torch.Generator().manual_seed(self.seed)
        train_split, val_split = random_split(full, [n_train, n_val], generator=gen)

        val_dataset = LSUIDataset(
            self.root,
            transform=val_tf,
            load_transmission_gt=self.load_transmission_gt,
        )
        val_dataset.samples = [val_dataset.samples[i] for i in val_split.indices]

        self._train = train_split
        self._val = val_dataset

    def _loader(self, ds: Dataset[dict[str, Tensor]] | None, *, shuffle: bool) -> DataLoader[dict[str, Tensor]]:
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
