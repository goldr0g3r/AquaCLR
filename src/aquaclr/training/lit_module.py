"""PyTorch Lightning module for LEGION-DeSnow.

This is a thin wrapper that:

* owns the ``nn.Module`` and the loss,
* runs the train/val steps and logs metrics,
* configures the optimiser + scheduler,
* hands off to callbacks (EMA, sample-image logger, VRAM monitor) for
  cross-cutting concerns.

We intentionally keep the wrapped network as a plain ``nn.Module`` so
that ONNX/TRT export paths can pull it out via ``module.net`` without
any Lightning machinery contaminating the export graph.

Automotive SiL parallel:
    Same pattern as automotive ML training infrastructure: the
    perception network is a plain torch module; an experiment harness
    wraps it for training but never bleeds into the deployed binary.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

try:
    import lightning as L

    _LIGHTNING_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTNING_AVAILABLE = False
    L = None  # type: ignore[assignment]

from aquaclr.losses.physics_loss import PhysicsInformedLoss
from aquaclr.models.model import LEGIONDeSnowNet


class LEGIONDeSnowLitModule(L.LightningModule if _LIGHTNING_AVAILABLE else nn.Module):  # type: ignore[misc]
    """Lightning module wrapping :class:`LEGIONDeSnowNet`.

    Args:
        net: The underlying physics-informed network.
        loss: The composite physics-informed loss.
        optimizer_cfg: Hydra-style instantiable config for the optimiser
            (must accept ``params`` kwarg).
        scheduler_cfg: Hydra-style instantiable config for the scheduler
            (optional). If using OneCycleLR, ``total_steps`` will be
            injected at runtime.
        compile_cfg: Optional torch.compile configuration dict.
        freeze_backbone_epochs: Number of initial epochs in which the
            encoder is frozen for warmup of the heads.
    """

    def __init__(
        self,
        net: LEGIONDeSnowNet,
        loss: PhysicsInformedLoss,
        *,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
        compile_cfg: dict[str, Any] | None = None,
        freeze_backbone_epochs: int = 0,
    ) -> None:
        if _LIGHTNING_AVAILABLE:
            super().__init__()
            self.save_hyperparameters(
                ignore=["net", "loss"],
                logger=False,
            )
        else:
            super().__init__()
        self.net = net
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg or {"lr": 3.0e-4, "weight_decay": 1.0e-4}
        self.scheduler_cfg = scheduler_cfg
        self.compile_cfg = compile_cfg or {"enabled": False}
        self.freeze_backbone_epochs = int(freeze_backbone_epochs)

        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    # ----------------------------------------------------- Lightning hooks

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Compile the inner network if requested. Done in setup so that
        ``trainer.precision`` is already known.
        """
        if self.compile_cfg.get("enabled", False) and hasattr(torch, "compile"):
            self.net = torch.compile(  # type: ignore[assignment]
                self.net,
                mode=self.compile_cfg.get("mode", "reduce-overhead"),
                fullgraph=bool(self.compile_cfg.get("fullgraph", False)),
            )

    def on_train_epoch_start(self) -> None:
        if self.freeze_backbone_epochs > 0 and self.current_epoch < self.freeze_backbone_epochs:
            self.net.freeze_backbone()
        elif self.freeze_backbone_epochs > 0 and self.current_epoch == self.freeze_backbone_epochs:
            self.net.unfreeze_backbone()

    # -------------------------------------------------------------- forward

    def forward(self, i: Tensor) -> tuple[Tensor, Tensor, Tensor]:  # type: ignore[override]
        out = self.net(i)
        return out.j, out.t, out.b

    # --------------------------------------------------------------- steps

    def _shared_step(self, batch: dict[str, Any], *, stage: str) -> Tensor:
        i = batch["i"]
        j_gt = batch["j"]
        out = self.net(i)
        t_gt: Tensor | None = batch.get("t_gt") if bool(batch.get("has_t_gt", False).any()) else None  # type: ignore[union-attr]
        loss_outputs = self.loss(i=i, j_pred=out.j, j_gt=j_gt, t=out.t, b=out.b, t_gt=t_gt)

        log_dict = loss_outputs.to_log_dict(prefix=f"{stage}/loss/")
        self.log_dict(log_dict, prog_bar=False, on_step=stage == "train", on_epoch=True, sync_dist=True)

        if stage == "train":
            self.train_psnr.update(out.j.clamp(0, 1), j_gt.clamp(0, 1))
            self.train_ssim.update(out.j.clamp(0, 1), j_gt.clamp(0, 1))
            self.log("train/psnr", self.train_psnr, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train/ssim", self.train_ssim, prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.val_psnr.update(out.j.clamp(0, 1), j_gt.clamp(0, 1))
            self.val_ssim.update(out.j.clamp(0, 1), j_gt.clamp(0, 1))
            self.log("val/psnr", self.val_psnr, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val/ssim", self.val_ssim, prog_bar=True, on_step=False, on_epoch=True)

        return loss_outputs.total

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:  # noqa: ARG002
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:  # noqa: ARG002
        return self._shared_step(batch, stage="val")

    # -------------------------------------------------------- optimisation

    def configure_optimizers(self) -> dict[str, Any] | Optimizer:
        """Build optimiser + (optional) scheduler.

        The optimiser config is a plain dict whose keys are the kwargs of
        ``torch.optim.AdamW`` (or another optimiser if the config swaps
        ``_target_``). The scheduler config supports either ``OneCycleLR``
        (in which case ``total_steps`` is injected) or anything else
        accepted by Lightning's scheduler dict format.
        """
        opt_cfg = dict(self.optimizer_cfg)
        target = opt_cfg.pop("_target_", "torch.optim.AdamW")
        opt_cls = _resolve(target)
        optimizer: Optimizer = opt_cls(
            (p for p in self.parameters() if p.requires_grad), **opt_cfg
        )

        if not self.scheduler_cfg:
            return optimizer

        sched_cfg = dict(self.scheduler_cfg)
        sched_target = sched_cfg.pop("_target_", "torch.optim.lr_scheduler.OneCycleLR")
        sched_cls = _resolve(sched_target)
        if sched_target.endswith("OneCycleLR") and "total_steps" not in sched_cfg:
            try:
                sched_cfg["total_steps"] = int(self.trainer.estimated_stepping_batches)
            except (RuntimeError, AttributeError):
                sched_cfg.setdefault("epochs", 60)
                sched_cfg.setdefault("steps_per_epoch", 100)
        scheduler: LRScheduler = sched_cls(optimizer, **sched_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def _resolve(dotted: str) -> Any:
    """Resolve a dotted path like ``torch.optim.AdamW`` to the actual class."""
    import importlib

    module_name, _, attr = dotted.rpartition(".")
    if not module_name:
        msg = f"Could not resolve {dotted!r}; expected a dotted module path."
        raise ImportError(msg)
    module = importlib.import_module(module_name)
    return getattr(module, attr)
