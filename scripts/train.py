"""Hydra-driven training entry point.

Examples:
    Train with the default profile (MSRB-only, RTX 3050 BF16, 256 px crops)::

        uv run python scripts/train.py

    Train on the RTX A3000 with the A3000-tuned data config (384 px, batch 16/8)::

        uv run python scripts/train.py train=rtx_a3000_bf16 data=combined_a3000

    Override individual hyperparameters at the CLI::

        uv run python scripts/train.py train=rtx_a3000_bf16 train.max_epochs=40 train.optimizer.lr=1e-4

    Resume an interrupted run (restores optimizer state + epoch counter)::

        uv run python scripts/train.py train=rtx_a3000_bf16 \\
            run_name=20260518-163016 \\
            resume_from=outputs/20260518-163016/ckpts/last.ckpt

    Multirun sweep over learning rates::

        uv run python scripts/train.py -m train.optimizer.lr=1e-4,3e-4,1e-3

    Set HYDRA_FULL_ERROR=1 for a full stack trace on config errors::

        $env:HYDRA_FULL_ERROR=1; uv run python scripts/train.py train=rtx_a3000_bf16
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("aquaclr.train")


@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Train LEGION-DeSnow according to the resolved Hydra config."""
    import lightning as L
    from hydra.utils import instantiate
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        ModelSummary,
    )
    from lightning.pytorch.loggers import (
        TensorBoardLogger,
        WandbLogger,
    )

    from aquaclr.training import (
        EMAWeightCallback,
        LEGIONDeSnowLitModule,
        SampleImageLogger,
        VRAMMonitor,
    )
    from aquaclr.utils import seed_everything

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s"
    )
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed), deterministic=bool(cfg.deterministic))

    Path(cfg.paths.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)

    datamodule = instantiate(cfg.data)

    net = instantiate(cfg.model)
    loss = instantiate(cfg.train.loss)
    lit = LEGIONDeSnowLitModule(
        net=net,
        loss=loss,
        optimizer_cfg=OmegaConf.to_container(cfg.train.optimizer, resolve=True),  # type: ignore[arg-type]
        scheduler_cfg=(
            OmegaConf.to_container(cfg.train.scheduler, resolve=True)  # type: ignore[arg-type]
            if "scheduler" in cfg.train
            else None
        ),
        compile_cfg=OmegaConf.to_container(cfg.train.compile, resolve=True),  # type: ignore[arg-type]
        freeze_backbone_epochs=int(cfg.model.get("freeze_backbone_epochs", 0)),
    )

    callbacks: list[Any] = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=cfg.paths.ckpt_dir,
            filename="legion-desnow-{epoch:03d}-{val/psnr:.2f}",
            **OmegaConf.to_container(cfg.train.callbacks.checkpoint, resolve=True),  # type: ignore[arg-type]
        ),
        EarlyStopping(**OmegaConf.to_container(cfg.train.callbacks.early_stopping, resolve=True)),  # type: ignore[arg-type]
    ]
    if cfg.train.callbacks.ema.enabled:
        callbacks.append(EMAWeightCallback(decay=cfg.train.callbacks.ema.decay))
    if cfg.train.callbacks.vram_monitor.enabled:
        callbacks.append(
            VRAMMonitor(every_n_steps=cfg.train.callbacks.vram_monitor.every_n_steps)
        )
    callbacks.append(
        SampleImageLogger(
            every_n_steps=cfg.train.callbacks.sample_logger.every_n_steps,
            n_samples=cfg.train.callbacks.sample_logger.n_samples,
        )
    )

    loggers: list[Any] = []
    if cfg.logging.tensorboard.enabled:
        loggers.append(TensorBoardLogger(save_dir=cfg.paths.log_dir, name="tb"))
    if cfg.logging.wandb.enabled:
        loggers.append(
            WandbLogger(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.entity,
                tags=list(cfg.logging.wandb.tags),
                save_dir=cfg.paths.log_dir,
            )
        )

    trainer = L.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        precision=cfg.train.precision,
        accumulate_grad_batches=int(cfg.train.accumulate_grad_batches),
        gradient_clip_val=cfg.train.gradient_clip_val,
        gradient_clip_algorithm=cfg.train.gradient_clip_algorithm,
        callbacks=callbacks,
        logger=loggers or False,
        **OmegaConf.to_container(cfg.train.trainer, resolve=True),  # type: ignore[arg-type]
    )

    resume_from: str | None = cfg.get("resume_from") or None
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)
    trainer.fit(lit, datamodule=datamodule, ckpt_path=resume_from)
    trainer.test(lit, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
