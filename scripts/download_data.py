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
        "  Layout (canonical upstream, after unpack):\n"
        "    {root}/training/original/*.png    # clean reference J\n"
        "    {root}/training/MSR_Task1/*.png   # snowy I, small particles\n"
        "    {root}/training/MSR_Task2/*.png   # snowy I, mixed sizes\n"
        "    {root}/test/original/*.png\n"
        "    {root}/test/MSR_Task1/*.png\n"
        "    {root}/test/MSR_Task2/*.png\n"
        "  Pairing is by filename stem; pick the task via the\n"
        "  MSRBDataset/MSRBDataModule `task` argument (1 or 2).\n"
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

LAYOUT_CHECKS: dict[str, tuple[str, ...]] = {
    "msrb": ("training/original", "test/original"),
    "lsui": ("input", "GT"),
    "uieb": ("challenging-60",),
}

# Per-dataset extras where at least ONE of the listed directories must be
# present (e.g. either snowy variant for MSRB, since some users only fetch
# one task).
LAYOUT_ANY_OF: dict[str, tuple[tuple[str, ...], ...]] = {
    "msrb": (
        ("training/MSR_Task1", "training/MSR_Task2"),
        ("test/MSR_Task1", "test/MSR_Task2"),
    ),
}


def _verify(root: Path, dataset: str) -> bool:
    expected = LAYOUT_CHECKS[dataset]
    missing = [d for d in expected if not (root / d).is_dir()]
    if missing:
        logger.warning("[%s] missing under %s: %s", dataset, root, missing)
        return False
    for group in LAYOUT_ANY_OF.get(dataset, ()):
        if not any((root / d).is_dir() for d in group):
            logger.warning("[%s] none of %s present under %s", dataset, list(group), root)
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
