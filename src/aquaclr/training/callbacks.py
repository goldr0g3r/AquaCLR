"""Training callbacks for AquaCLR.

* :class:`EMAWeightCallback` — exponential moving average over the
  network's parameters. EMA weights typically gain ~0.2-0.4 dB PSNR for
  free.
* :class:`VRAMMonitor` — logs CUDA peak memory at a configurable cadence
  so we can catch creeping memory regressions on the 4 GB target.
* :class:`SampleImageLogger` — periodically renders ``(I, J_pred, J_gt,
  t)`` strips to TensorBoard / W&B for visual sanity checking.

Automotive SiL parallel:
    EMA + memory budgeting + visual replay are exactly the three
    cross-cutting concerns enforced in production ADAS perception
    training pipelines.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

try:
    import lightning as L
    from lightning.pytorch.callbacks import Callback

    _LIGHTNING_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTNING_AVAILABLE = False
    L = None  # type: ignore[assignment]
    Callback = object  # type: ignore[misc,assignment]


class EMAWeightCallback(Callback):  # type: ignore[misc]
    """Maintain an exponential moving average of model parameters.

    Args:
        decay: EMA decay (``0.999`` ... ``0.9999``). Higher = slower.
        every_n_steps: Apply update every ``n`` optimiser steps.
    """

    def __init__(self, *, decay: float = 0.9995, every_n_steps: int = 1) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            msg = f"decay must be in (0, 1), got {decay}"
            raise ValueError(msg)
        self.decay = float(decay)
        self.every_n_steps = int(every_n_steps)
        self._shadow: dict[str, Tensor] = {}
        self._backup: dict[str, Tensor] = {}

    def _params(self, module: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
        return ((n, p) for n, p in module.named_parameters() if p.requires_grad)

    def setup(
        self, trainer: Any, pl_module: nn.Module, stage: str
    ) -> None:  # noqa: ARG002
        # Initialise lazily — do NOT clone parameters here.  setup() is called
        # before Lightning moves the model to the target device, so cloning now
        # would leave shadow tensors on CPU while the live parameters end up on
        # CUDA.  The first on_train_batch_end hit will populate _shadow from the
        # already-placed parameters via the `shadow is None` branch below.
        self._shadow = {}

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: nn.Module,
        outputs: Any,  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        step = getattr(trainer, "global_step", batch_idx)
        if step % self.every_n_steps != 0:
            return
        with torch.no_grad():
            for n, p in self._params(pl_module):
                shadow = self._shadow.get(n)
                if shadow is None or shadow.shape != p.shape:
                    self._shadow[n] = p.detach().clone()
                else:
                    shadow.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def on_validation_start(
        self, trainer: Any, pl_module: nn.Module
    ) -> None:  # noqa: ARG002
        self._backup = {n: p.detach().clone() for n, p in self._params(pl_module)}
        with torch.no_grad():
            for n, p in self._params(pl_module):
                shadow = self._shadow.get(n)
                if shadow is not None and shadow.shape == p.shape:
                    p.data.copy_(shadow)

    def on_validation_end(
        self, trainer: Any, pl_module: nn.Module
    ) -> None:  # noqa: ARG002
        with torch.no_grad():
            for n, p in self._params(pl_module):
                bak = self._backup.get(n)
                if bak is not None and bak.shape == p.shape:
                    p.data.copy_(bak)
        self._backup.clear()

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": deepcopy(self._shadow)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self._shadow = state_dict.get("shadow", {})


class VRAMMonitor(Callback):  # type: ignore[misc]
    """Log CUDA peak memory periodically. CPU-only runs report zeros."""

    def __init__(self, *, every_n_steps: int = 50) -> None:
        super().__init__()
        self.every_n_steps = int(every_n_steps)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: nn.Module,
        outputs: Any,  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        step = getattr(trainer, "global_step", batch_idx)
        if step % self.every_n_steps != 0:
            return
        if not torch.cuda.is_available():
            return
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        pl_module.log(
            "perf/vram_peak_mb", peak_mb, on_step=True, on_epoch=False, prog_bar=False
        )


class SampleImageLogger(Callback):  # type: ignore[misc]
    """Render a ``(I, J_pred, J_gt, t)`` strip every N steps.

    Tries TensorBoard first (always available with Lightning), and
    additionally writes to W&B if the ``wandb`` logger is attached.
    """

    def __init__(self, *, every_n_steps: int = 200, n_samples: int = 4) -> None:
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self.n_samples = int(n_samples)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: nn.Module,
        outputs: Any,  # noqa: ARG002
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        step = getattr(trainer, "global_step", batch_idx)
        if step == 0 or step % self.every_n_steps != 0:
            return
        with torch.no_grad():
            i = batch["i"][: self.n_samples].to(pl_module.device)
            j_gt = batch["j"][: self.n_samples].to(pl_module.device)
            out = pl_module.net(i)
            j_pred = out.j.clamp(0, 1)
            t_3 = out.t.repeat(1, 3, 1, 1)
            grid = torch.cat([i, j_pred, j_gt, t_3], dim=0).clamp(0, 1).cpu()

        try:
            from torchvision.utils import make_grid

            img = make_grid(grid, nrow=self.n_samples, padding=2)
        except ImportError:  # pragma: no cover
            img = grid[0]

        for logger in trainer.loggers:
            try:
                if hasattr(logger, "experiment") and hasattr(
                    logger.experiment, "add_image"
                ):
                    logger.experiment.add_image("samples", img, global_step=step)
                elif hasattr(logger, "log_image"):
                    logger.log_image(
                        key="samples", images=[img.permute(1, 2, 0).numpy()]
                    )
            except Exception:  # pragma: no cover - logging must never break training
                continue
