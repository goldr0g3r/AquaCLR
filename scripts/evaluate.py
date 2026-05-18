"""Evaluate a trained checkpoint on UIEB-Challenge or MSRB-test.

Reports PSNR, SSIM, and (best-effort) UIQM/UCIQE no-reference metrics.

UIQM / UCIQE require the optional ``pyiqa`` package::

    pip install pyiqa

They are silently skipped if ``pyiqa`` is not installed.
"""

from __future__ import annotations

import argparse
import logging
import statistics
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

# ---------------------------------------------------------------------------
# Optional: pyiqa — no-reference IQA metrics (UIQM, UCIQE).
# Install with:  pip install pyiqa
# ---------------------------------------------------------------------------
try:
    import pyiqa  # type: ignore[import-untyped]

    _PYIQA_AVAILABLE = True
except ImportError:
    _PYIQA_AVAILABLE = False

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
    p.add_argument(
        "--no-ref",
        action="store_true",
        default=False,
        help="Compute UIQM / UCIQE no-reference metrics (requires pyiqa). "
        "Applied to the enhanced output Ĵ only — no GT needed.",
    )
    return p.parse_args()


def main() -> None:
    """Run evaluation."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s"
    )
    args = _parse_args()

    # ------------------------------------------------------------------
    # No-reference metric setup
    # ------------------------------------------------------------------
    run_no_ref = args.no_ref
    if run_no_ref and not _PYIQA_AVAILABLE:
        logger.warning(
            "pyiqa is not installed — UIQM/UCIQE will be skipped. "
            "Install with: pip install pyiqa"
        )
        run_no_ref = False

    uiqm_scores: list[float] = []
    uciqe_scores: list[float] = []

    if run_no_ref:
        logger.info("pyiqa found — will compute UIQM / UCIQE on enhanced output Ĵ")
        _uiqm_metric = pyiqa.create_metric("uiqm", device=args.device)
        _uciqe_metric = pyiqa.create_metric("uciqe", device=args.device)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Reference-based metrics
    # ------------------------------------------------------------------
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)

    with torch.no_grad():
        for batch in loader:
            i = batch["i"].to(args.device, non_blocking=True)
            j_gt = batch["j"].to(args.device, non_blocking=True)
            out = model(i)
            j_pred = out.j.clamp(0.0, 1.0)

            psnr.update(j_pred, j_gt.clamp(0.0, 1.0))
            ssim.update(j_pred, j_gt.clamp(0.0, 1.0))

            # ----------------------------------------------------------
            # No-reference metrics (best-effort, no GT required)
            # UIQM  [Panetta 2016]: UICM colourfulness + UISM sharpness
            #                       + UIConM contrast
            # UCIQE [Yang 2015]  : CIE-Lab chroma σ + luminance contrast
            #                       + saturation mean
            # ----------------------------------------------------------
            if run_no_ref:
                # pyiqa expects (B, C, H, W) float32 in [0, 1]
                scores_uiqm: torch.Tensor = _uiqm_metric(j_pred).cpu()
                scores_uciqe: torch.Tensor = _uciqe_metric(j_pred).cpu()
                uiqm_scores.extend(scores_uiqm.tolist())
                uciqe_scores.extend(scores_uciqe.tolist())

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    logger.info("PSNR: %.3f dB", psnr.compute().item())
    logger.info("SSIM: %.4f", ssim.compute().item())

    if run_no_ref and uiqm_scores:
        n = len(uiqm_scores)
        uiqm_mean = statistics.mean(uiqm_scores)
        uciqe_mean = statistics.mean(uciqe_scores)
        uiqm_sd = statistics.stdev(uiqm_scores) if n > 1 else 0.0
        uciqe_sd = statistics.stdev(uciqe_scores) if n > 1 else 0.0
        logger.info(
            "UIQM : %.4f ± %.4f  (n=%d)  [no-reference, higher is better]",
            uiqm_mean,
            uiqm_sd,
            n,
        )
        logger.info(
            "UCIQE: %.4f ± %.4f  (n=%d)  [no-reference, higher is better]",
            uciqe_mean,
            uciqe_sd,
            n,
        )


if __name__ == "__main__":
    main()
