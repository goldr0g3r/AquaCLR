"""Print dataset download instructions and verify on-disk layout.

We deliberately do **not** auto-download MSRB / LSUI / UIEB because
their hosting URLs change over project lifetimes. Instead this script:

* prints the canonical fetch instructions, and
* once the user has placed the archives, verifies the directory
  layout so a typo doesn't show up only at the first training step.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("aquaclr.download")

INSTRUCTIONS = {
    "msrb": (
        "Marine Snow Removal Benchmark (Sato et al., APSIPA 2023)\n"
        "  Repo:   https://github.com/ychtanaka/marine-snow\n"
        "  Paper:  https://arxiv.org/abs/2103.14249\n"
        "  Layout (after unpack):\n"
        "    {root}/train/clean/*.png\n"
        "    {root}/train/noisy/*.png\n"
        "    {root}/test/clean/*.png\n"
        "    {root}/test/noisy/*.png\n"
    ),
    "lsui": (
        "LSUI (Peng et al., 2021 — U-shape Transformer)\n"
        "  Page:   https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html\n"
        "  Layout (after unpack):\n"
        "    {root}/input/*.png|jpg\n"
        "    {root}/GT/*.png|jpg\n"
        "    {root}/transmission/*.png    # optional but recommended\n"
    ),
    "uieb": (
        "Underwater Image Enhancement Benchmark (Li et al., 2019)\n"
        "  Page:   https://li-chongyi.github.io/proj_benchmark.html\n"
        "  Layout (after unpack):\n"
        "    {root}/raw-890/*.png\n"
        "    {root}/reference-890/*.png\n"
        "    {root}/challenging-60/*.png    # used as held-out eval\n"
    ),
}

LAYOUT_CHECKS = {
    "msrb": ("train/clean", "train/noisy"),
    "lsui": ("input", "GT"),
    "uieb": ("challenging-60",),
}


def _verify(root: Path, dataset: str) -> bool:
    expected = LAYOUT_CHECKS[dataset]
    missing = [d for d in expected if not (root / d).exists()]
    if missing:
        logger.warning("[%s] missing under %s: %s", dataset, root, missing)
        return False
    logger.info("[%s] layout OK at %s", dataset, root)
    return True


def main() -> None:
    """CLI entry."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--datasets", nargs="+", default=["msrb", "lsui", "uieb"])
    p.add_argument("--instructions-only", action="store_true")
    args = p.parse_args()

    for ds in args.datasets:
        print("\n" + "=" * 78)
        print(INSTRUCTIONS[ds].format(root=args.data_root / ds))
        if not args.instructions_only:
            _verify(args.data_root / ds, ds)
    print("\nWhen all sections show 'layout OK', you're ready to train.\n")


if __name__ == "__main__":
    main()
