"""Reproducibility helpers.

Automotive SiL parallel:
    Equivalent to fixing the random seed of a noise injector in a HIL
    (hardware-in-the-loop) rig so that scenarios are byte-for-byte
    repeatable across regression runs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 1337, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Master seed.
        deterministic: If True, enables PyTorch deterministic algorithms and
            disables cuDNN benchmarking. This is REQUIRED for reproducible
            scientific runs but can slow down training by ~5-10%.

    Returns:
        The seed that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass
    else:
        torch.backends.cudnn.benchmark = True

    return seed
