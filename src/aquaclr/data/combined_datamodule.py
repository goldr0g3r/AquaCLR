"""Combined MSRB + LSUI DataModule using a weighted-batch sampler.

The trick: each Lightning training step emits **one batch from one
dataset** (sampled according to ``mix_ratio``). This keeps the physics
loss simple — the ``has_t_gt`` flag in the batch tells the loss
whether to apply the direct ``t`` supervision term.

Automotive SiL parallel:
    Equivalent to multiplexing real and synthetic ADAS data sources
    during training, where each batch is sampled stochastically from
    one source so the loss can pivot on a per-batch flag.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

try:
    import lightning as L

    _LIGHTNING_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTNING_AVAILABLE = False
    L = None  # type: ignore[assignment]

from aquaclr.data.lsui_dataset import LSUIDataModule
from aquaclr.data.msrb_dataset import MSRBDataModule


class _AlternatingLoader:
    """Yields batches from a list of dataloaders sampled by ``probs``.

    The combined "epoch length" is the **sum** of the wrapped loader
    lengths so that, in expectation, each loader's underlying dataset
    is fully visited per epoch.
    """

    def __init__(
        self,
        loaders: list[DataLoader[dict[str, Tensor]]],
        probs: list[float],
        seed: int = 1337,
    ) -> None:
        if len(loaders) != len(probs):
            msg = "loaders and probs must have the same length"
            raise ValueError(msg)
        if not loaders:
            msg = "Need at least one loader"
            raise ValueError(msg)
        self.loaders = loaders
        s = float(sum(probs))
        if s <= 0:
            msg = "probs must sum to > 0"
            raise ValueError(msg)
        self.probs = [float(p) / s for p in probs]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        iters = [iter(loader) for loader in self.loaders]
        n = len(self)
        for _ in range(n):
            choice = int(self.rng.choice(len(self.loaders), p=self.probs))
            try:
                yield next(iters[choice])
            except StopIteration:
                # Restart that single iterator and try again. We accept
                # one duplicated batch here in exchange for not aborting
                # the whole epoch.
                iters[choice] = iter(self.loaders[choice])
                yield next(iters[choice])


class CombinedDataModule(L.LightningDataModule if _LIGHTNING_AVAILABLE else object):  # type: ignore[misc]
    """Mix MSRB and LSUI per-batch with configurable probabilities.

    Args:
        msrb: An MSRB DataModule **or** an instance config dict.
        lsui: An LSUI DataModule **or** an instance config dict.
        mix_ratio: ``(p_msrb, p_lsui)``. Normalised internally.
        val_use: Which sub-DataModule's val split is reported as the
            primary metric: ``"msrb"`` or ``"lsui"``.
    """

    def __init__(
        self,
        msrb: MSRBDataModule | dict[str, Any] | str | Path,
        lsui: LSUIDataModule | dict[str, Any] | str | Path,
        *,
        mix_ratio: tuple[float, float] = (0.7, 0.3),
        val_use: str = "msrb",
    ) -> None:
        if _LIGHTNING_AVAILABLE:
            super().__init__()
        self.msrb = self._coerce_msrb(msrb)
        self.lsui = self._coerce_lsui(lsui)
        self.mix_ratio = (float(mix_ratio[0]), float(mix_ratio[1]))
        if val_use not in {"msrb", "lsui"}:
            msg = f"val_use must be 'msrb' or 'lsui', got {val_use!r}"
            raise ValueError(msg)
        self.val_use = val_use

    @staticmethod
    def _coerce_msrb(x: MSRBDataModule | dict[str, Any] | str | Path) -> MSRBDataModule:
        if isinstance(x, MSRBDataModule):
            return x
        if isinstance(x, dict):
            kwargs = {k: v for k, v in x.items() if k != "_target_"}
            return MSRBDataModule(**kwargs)
        return MSRBDataModule(root=x)

    @staticmethod
    def _coerce_lsui(x: LSUIDataModule | dict[str, Any] | str | Path) -> LSUIDataModule:
        if isinstance(x, LSUIDataModule):
            return x
        if isinstance(x, dict):
            kwargs = {k: v for k, v in x.items() if k != "_target_"}
            return LSUIDataModule(**kwargs)
        return LSUIDataModule(root=x)

    def prepare_data(self) -> None:
        self.msrb.prepare_data()
        try:
            self.lsui.prepare_data()
        except (FileNotFoundError, NotImplementedError):
            # LSUI is optional; the combined module still works on MSRB alone.
            pass

    def setup(self, stage: str | None = None) -> None:
        self.msrb.setup(stage)
        try:
            self.lsui.setup(stage)
            self._lsui_ok = True
        except FileNotFoundError:
            self._lsui_ok = False

    def train_dataloader(self) -> Any:
        loaders = [self.msrb.train_dataloader()]
        probs = [self.mix_ratio[0]]
        if self._lsui_ok:
            loaders.append(self.lsui.train_dataloader())
            probs.append(self.mix_ratio[1])
        if len(loaders) == 1:
            return loaders[0]
        return _AlternatingLoader(loaders, probs)

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        if self.val_use == "lsui" and self._lsui_ok:
            return self.lsui.val_dataloader()
        return self.msrb.val_dataloader()

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self.msrb.test_dataloader()
