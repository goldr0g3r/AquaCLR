"""Evaluate a trained checkpoint on MSRB-test, UIEB, or LSUI.

Reports PSNR, SSIM, and (best-effort) no-reference metrics.

No-reference metrics require the optional ``pyiqa`` package::

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

from aquaclr.data.lsui_dataset import LSUIDataset
from aquaclr.data.msrb_dataset import MSRBDataset, _collate
from aquaclr.data.transforms import build_val_transform
from aquaclr.data.uieb_dataset import UIEBDataset
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
    p.add_argument(
        "--dataset",
        default="msrb",
        choices=["msrb", "uieb", "lsui"],
        help="Which dataset format to load (default: msrb).",
    )
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
        help="Compute no-reference metrics (requires pyiqa). "
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

    nref_scores: dict[str, list[float]] = {}
    nref_metrics: dict[str, object] = {}

    if run_no_ref:
        # Try UIQM/UCIQE first; fall back to NIQE/MUSIQ if unavailable.
        _metric_candidates = [
            ("uiqm", True),
            ("uciqe", True),
            ("niqe", False),
            ("musiq", True),
        ]
        for mname, higher_better in _metric_candidates:
            try:
                nref_metrics[mname] = pyiqa.create_metric(mname, device=args.device)
                nref_scores[mname] = []
                logger.info("pyiqa: loaded no-reference metric '%s'", mname)
            except (AssertionError, KeyError, ValueError):
                logger.warning("pyiqa: metric '%s' not available — skipping", mname)
        if not nref_metrics:
            logger.warning(
                "No no-reference metrics could be loaded — disabling --no-ref"
            )
            run_no_ref = False

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
    transform = build_val_transform(args.image_size)
    if args.dataset == "msrb":
        ds = MSRBDataset(
            args.data_root,
            split=args.split,
            task=args.task,
            transform=transform,
        )
    elif args.dataset == "uieb":
        ds = UIEBDataset(args.data_root, transform=transform)
    elif args.dataset == "lsui":
        ds = LSUIDataset(args.data_root, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info("Dataset: %s (%d samples)", args.dataset.upper(), len(ds))
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
                for mname, metric_fn in nref_metrics.items():
                    scores: torch.Tensor = metric_fn(j_pred).cpu().flatten()
                    nref_scores[mname].extend(scores.tolist())

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    logger.info("PSNR: %.3f dB", psnr.compute().item())
    logger.info("SSIM: %.4f", ssim.compute().item())

    if run_no_ref and nref_scores:
        for mname, scores_list in nref_scores.items():
            n = len(scores_list)
            if n == 0:
                continue
            m = statistics.mean(scores_list)
            sd = statistics.stdev(scores_list) if n > 1 else 0.0
            logger.info(
                "%s: %.4f ± %.4f  (n=%d)  [no-reference]",
                mname.upper(),
                m,
                sd,
                n,
            )


if __name__ == "__main__":
    main()
