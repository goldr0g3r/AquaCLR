"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest
import torch

from aquaclr.utils import seed_everything


@pytest.fixture(autouse=True)
def _seed_each_test() -> None:
    """Reseed before each test for byte-for-byte reproducibility."""
    seed_everything(1337, deterministic=False)


@pytest.fixture
def cuda_available() -> bool:
    """Whether CUDA is available for this test session."""
    if os.environ.get("AQUACLR_DISABLE_GPU") == "1":
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    """Auto-skip GPU/TRT-marked tests when the deps are missing."""
    cuda = torch.cuda.is_available() and os.environ.get("AQUACLR_DISABLE_GPU") != "1"
    try:
        import tensorrt  # noqa: F401, PLC0415

        trt_ok = True
    except ImportError:
        trt_ok = False

    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_trt = pytest.mark.skip(reason="TensorRT not available")
    for item in items:
        if "gpu" in item.keywords and not cuda:
            item.add_marker(skip_gpu)
        if "trt" in item.keywords and not trt_ok:
            item.add_marker(skip_trt)
