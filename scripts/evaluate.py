"""Evaluate a trained checkpoint on UIEB-Challenge or MSRB-test.

Reports PSNR, SSIM, and (best-effort) UIQM/UCIQE no-reference metrics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from aquaclr.data.msrb_dataset import MSRBDataset, _collate
from aquaclr.data.transforms import build_val_transform
from aquaclr.models import LEGIONDeSnowNet

logger = logging.getLogger("aquaclr.eval")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--task", type=int, default=1, choices=[1, 2])
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    """Run evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    args = _parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state_dict = {
            k.removeprefix("net._orig_mod.").removeprefix("net."): v
            for k, v in state["state_dict"].items()
            if k.startswith("net.")
        }
    else:
        state_dict = state

    model = LEGIONDeSnowNet().to(args.device).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected[:5])

    ds = MSRBDataset(
        args.data_root,
        split=args.split,
        task=args.task,
        transform=build_val_transform(args.image_size),
    )
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)

    with torch.no_grad():
        for batch in loader:
            i = batch["i"].to(args.device, non_blocking=True)
            j_gt = batch["j"].to(args.device, non_blocking=True)
            out = model(i)
            psnr.update(out.j.clamp(0, 1), j_gt.clamp(0, 1))
            ssim.update(out.j.clamp(0, 1), j_gt.clamp(0, 1))

    logger.info("PSNR: %.3f dB", psnr.compute().item())
    logger.info("SSIM: %.4f", ssim.compute().item())


if __name__ == "__main__":
    main()
