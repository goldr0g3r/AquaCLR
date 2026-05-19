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
from aquaclr.data.uieb_dataset import UIEBDataModule


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
    """Mix MSRB, LSUI, and optionally UIEB per-batch with configurable probabilities.

    Args:
        msrb: An MSRB DataModule **or** an instance config dict.
        lsui: An LSUI DataModule **or** an instance config dict.
        uieb: An optional UIEB DataModule **or** an instance config dict.
        mix_ratio: ``(p_msrb, p_lsui)`` or ``(p_msrb, p_lsui, p_uieb)``.
            Normalised internally.
        val_use: Which sub-DataModule's val split is reported as the
            primary metric: ``"msrb"``, ``"lsui"``, or ``"uieb"``.
    """

    def __init__(
        self,
        msrb: MSRBDataModule | dict[str, Any] | str | Path,
        lsui: LSUIDataModule | dict[str, Any] | str | Path,
        *,
        uieb: UIEBDataModule | dict[str, Any] | str | Path | None = None,
        mix_ratio: tuple[float, ...] = (0.7, 0.3),
        val_use: str = "msrb",
    ) -> None:
        if _LIGHTNING_AVAILABLE:
            super().__init__()
        self.msrb = self._coerce_msrb(msrb)
        self.lsui = self._coerce_lsui(lsui)
        self.uieb = self._coerce_uieb(uieb) if uieb is not None else None
        self.mix_ratio = tuple(float(x) for x in mix_ratio)
        if val_use not in {"msrb", "lsui", "uieb"}:
            msg = f"val_use must be 'msrb', 'lsui', or 'uieb', got {val_use!r}"
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

    @staticmethod
    def _coerce_uieb(x: UIEBDataModule | dict[str, Any] | str | Path) -> UIEBDataModule:
        if isinstance(x, UIEBDataModule):
            return x
        if isinstance(x, dict):
            kwargs = {k: v for k, v in x.items() if k != "_target_"}
            return UIEBDataModule(**kwargs)
        return UIEBDataModule(root=x)

    def prepare_data(self) -> None:
        self.msrb.prepare_data()
        try:
            self.lsui.prepare_data()
        except (FileNotFoundError, NotImplementedError):
            pass
        if self.uieb is not None:
            try:
                self.uieb.prepare_data()
            except (FileNotFoundError, NotImplementedError):
                pass

    def setup(self, stage: str | None = None) -> None:
        self.msrb.setup(stage)
        try:
            self.lsui.setup(stage)
            self._lsui_ok = True
        except FileNotFoundError:
            self._lsui_ok = False
        self._uieb_ok = False
        if self.uieb is not None:
            try:
                self.uieb.setup(stage)
                self._uieb_ok = True
            except FileNotFoundError:
                self._uieb_ok = False

    def train_dataloader(self) -> Any:
        loaders = [self.msrb.train_dataloader()]
        probs = [self.mix_ratio[0]]
        if self._lsui_ok:
            loaders.append(self.lsui.train_dataloader())
            probs.append(self.mix_ratio[1] if len(self.mix_ratio) > 1 else 0.0)
        if self._uieb_ok and self.uieb is not None:
            loaders.append(self.uieb.train_dataloader())
            probs.append(self.mix_ratio[2] if len(self.mix_ratio) > 2 else 0.0)
        if len(loaders) == 1:
            return loaders[0]
        return _AlternatingLoader(loaders, probs)

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        if self.val_use == "lsui" and self._lsui_ok:
            return self.lsui.val_dataloader()
        if self.val_use == "uieb" and self._uieb_ok and self.uieb is not None:
            return self.uieb.val_dataloader()
        return self.msrb.val_dataloader()

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self.msrb.test_dataloader()
