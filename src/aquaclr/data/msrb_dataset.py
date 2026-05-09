"""Marine Snow Removal Benchmark (MSRB) dataset and DataModule.

Reference:
    Sato, Y. et al. *Marine Snow Removal Benchmarking Dataset*, APSIPA ASC
    2023. arXiv: 2103.14249. Repository:
    https://github.com/ychtanaka/marine-snow

Layout we expect on disk (after the user has downloaded and unpacked
the archive into ``root``):

.. code-block::

    root/
    ├─ train/
    │  ├─ noisy/   # snowy I, 384x384 PNGs
    │  └─ clean/   # paired clean J, same filename stem
    └─ test/
       ├─ noisy/
       └─ clean/

Files in ``noisy/`` and ``clean/`` are paired by **filename stem**.
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

from aquaclr.data.snow_synthesis import synthesize_marine_snow
from aquaclr.data.transforms import build_train_transform, build_val_transform


class MSRBDataset(Dataset[dict[str, Tensor]]):
    """Paired (snowy, clean) Marine Snow Removal Benchmark dataset.

    Args:
        root: Path to the unpacked MSRB folder.
        split: ``"train"`` or ``"test"``.
        task: ``1`` (small particles) or ``2`` (mixed sizes). Currently the
            class assumes the user has placed the right task's files in
            ``noisy/``; the ``task`` argument is recorded in metadata
            only.
        transform: Albumentations-style transform expected to accept the
            keyword args ``image=`` (snowy I) and ``image_clean=``
            (clean J).
        synthesize_if_missing: If True and ``noisy/`` is empty but
            ``clean/`` is not, the dataset synthesises snow on the fly
            via :func:`synthesize_marine_snow`. Useful for smoke tests
            and CI before the real dataset is downloaded.
    """

    SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        task: int = 1,
        transform: Callable[..., dict[str, Any]] | None = None,
        synthesize_if_missing: bool = True,
    ) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            msg = f"split must be 'train' or 'test', got {split!r}"
            raise ValueError(msg)
        self.root = Path(root)
        self.split = split
        self.task = int(task)
        self.transform = transform
        self.synthesize_if_missing = bool(synthesize_if_missing)

        split_root = self.root / split
        self.clean_dir = split_root / "clean"
        self.noisy_dir = split_root / "noisy"

        if not self.clean_dir.exists():
            msg = (
                f"MSRB clean directory not found at {self.clean_dir}. "
                "Download the dataset (see configs/data/msrb.yaml) "
                "and unpack into the configured root."
            )
            raise FileNotFoundError(msg)

        clean_files = sorted(p for p in self.clean_dir.iterdir() if p.suffix.lower() in self.SUFFIXES)
        if not clean_files:
            msg = f"No images found in {self.clean_dir}"
            raise FileNotFoundError(msg)

        self.synthesize_active = False
        if not self.noisy_dir.exists() or not any(self.noisy_dir.iterdir()):
            if not self.synthesize_if_missing:
                msg = (
                    f"MSRB noisy directory empty at {self.noisy_dir}. "
                    "Either pre-render the snowy pairs or pass synthesize_if_missing=True."
                )
                raise FileNotFoundError(msg)
            self.synthesize_active = True

        self.samples: list[tuple[Path, Path | None]] = []
        for cf in clean_files:
            nf: Path | None = self.noisy_dir / cf.name
            if self.synthesize_active or (nf is not None and not nf.exists()):
                nf = None
            self.samples.append((cf, nf))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: Path) -> NDArray[np.uint8]:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        clean_path, noisy_path = self.samples[idx]
        clean_np = self._load_rgb(clean_path)
        if noisy_path is not None:
            noisy_np = self._load_rgb(noisy_path)
        else:
            noisy_np = synthesize_marine_snow(clean_np, seed=idx)

        if self.transform is not None:
            transformed = self.transform(image=noisy_np, image_clean=clean_np)
            i_t = transformed["image"]
            j_t = transformed["image_clean"]
        else:
            # Bare-bones path used in unit tests when albumentations is unavailable.
            i_t = torch.from_numpy(noisy_np).permute(2, 0, 1).float() / 255.0
            j_t = torch.from_numpy(clean_np).permute(2, 0, 1).float() / 255.0

        return {
            "i": i_t,
            "j": j_t,
            "has_t_gt": torch.tensor(False),
            "source": "msrb",
        }


class MSRBDataModule(L.LightningDataModule if _LIGHTNING_AVAILABLE else object):  # type: ignore[misc]
    """Lightning DataModule wrapping :class:`MSRBDataset`.

    Args:
        root: Path to the MSRB root directory.
        task: 1 or 2 (recorded only).
        image_size: Crop size during training.
        batch_size: Per-step batch size.
        num_workers: DataLoader worker count.
        pin_memory: Pin host memory for faster H2D copies on CUDA.
        persistent_workers: Keep DataLoader workers alive across epochs.
        val_fraction: Fraction of the official train split to hold out
            for validation.
        augment: Whether to apply training augmentations.
        download: If True, attempt to download/verify the dataset
            during ``prepare_data``. Downloads are best-effort; a
            ``FileNotFoundError`` is raised at ``setup`` time if the
            files still aren't on disk afterwards.
        seed: Seed for the train/val random split.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        task: int = 1,
        image_size: int = 256,
        batch_size: int = 16,
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
        self.task = int(task)
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
        self._test: Dataset[dict[str, Tensor]] | None = None

    # --- Lightning hooks -------------------------------------------------

    def prepare_data(self) -> None:
        """Best-effort download placeholder.

        The official MSRB hosting URL has changed across releases;
        rather than baking a URL that will rot, we surface a clear
        error message that points the user at the up-to-date download
        instructions in :file:`MODEL_CARD.md`.
        """
        if not self.download:
            return
        if (self.root / "train" / "clean").exists():
            return
        msg = (
            "Automatic MSRB download is not implemented. Please follow "
            "the instructions in MODEL_CARD.md to fetch the archive "
            f"and unpack it into {self.root}."
        )
        raise NotImplementedError(msg)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Materialise train/val/test datasets."""
        train_tf = build_train_transform(self.image_size) if self.augment else build_val_transform(self.image_size)
        val_tf = build_val_transform(self.image_size)

        full_train = MSRBDataset(self.root, split="train", task=self.task, transform=train_tf)
        n_val = max(1, int(len(full_train) * self.val_fraction))
        n_train = len(full_train) - n_val
        gen = torch.Generator().manual_seed(self.seed)
        train_split, val_split = random_split(full_train, [n_train, n_val], generator=gen)
        # Swap val transform on the val subset (random_split shares the underlying dataset).
        # We rebuild a parallel dataset with val transforms for clean determinism.
        val_dataset = MSRBDataset(self.root, split="train", task=self.task, transform=val_tf)
        val_dataset.samples = [val_dataset.samples[i] for i in val_split.indices]

        self._train = train_split
        self._val = val_dataset
        try:
            self._test = MSRBDataset(self.root, split="test", task=self.task, transform=val_tf)
        except FileNotFoundError:
            self._test = None

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

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._loader(self._test, shuffle=False)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate that handles tensor stacking + string-tag aggregation."""
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        values = [b[k] for b in batch]
        if isinstance(values[0], torch.Tensor):
            out[k] = torch.stack(values, dim=0)
        else:
            out[k] = values
    return out
