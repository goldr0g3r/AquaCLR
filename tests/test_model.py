"""Shape, parameter-budget, and gradient tests for LEGIONDeSnowNet."""

from __future__ import annotations

import pytest
import torch

from aquaclr.models import LEGIONDeSnowNet


@pytest.fixture(scope="module")
def model() -> LEGIONDeSnowNet:
    return LEGIONDeSnowNet(pretrained=False, use_channels_last=False).eval()


def test_forward_shapes(model: LEGIONDeSnowNet) -> None:
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    assert out.j.shape == (2, 3, 64, 64)
    assert out.t.shape == (2, 1, 64, 64)
    assert out.b.shape == (2, 3)


def test_outputs_in_unit_range(model: LEGIONDeSnowNet) -> None:
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    assert torch.all(out.j >= 0.0) and torch.all(out.j <= 1.0)
    assert torch.all(out.t > 0.0) and torch.all(out.t < 1.0)
    assert torch.all(out.b > 0.0) and torch.all(out.b < 1.0)


def test_forward_export_returns_tuple(model: LEGIONDeSnowNet) -> None:
    x = torch.rand(1, 3, 64, 64)
    j, t, b = model.forward_export(x)
    assert j.shape == (1, 3, 64, 64)
    assert t.shape == (1, 1, 64, 64)
    assert b.shape == (1, 3)


def test_size_budget(model: LEGIONDeSnowNet) -> None:
    """Total FP16 size must comfortably fit the 50 MB cap."""
    fp16_mb = model.estimate_size_mb(dtype=torch.float16)
    fp32_mb = model.estimate_size_mb(dtype=torch.float32)
    assert fp32_mb < 50.0, f"FP32 model is {fp32_mb:.1f} MB, exceeds 50 MB cap"
    assert fp16_mb < 25.0, f"FP16 model is {fp16_mb:.1f} MB"


def test_param_count_reasonable(model: LEGIONDeSnowNet) -> None:
    n = model.num_parameters
    assert 1_000_000 <= n <= 10_000_000, f"unexpected param count: {n}"


def test_backward_finite_gradients(model: LEGIONDeSnowNet) -> None:
    x = torch.rand(2, 3, 64, 64, requires_grad=False)
    out = model(x)
    loss = (out.j.mean() + out.t.mean() + out.b.mean())
    loss.backward()
    bad = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"non-finite gradients in: {bad[:5]}"


def test_handles_non_divisible_input(model: LEGIONDeSnowNet) -> None:
    """The decoder must cope with input shapes that are not multiples of 32."""
    x = torch.rand(1, 3, 73, 99)
    out = model(x)
    assert out.j.shape == (1, 3, 73, 99)
    assert out.t.shape == (1, 1, 73, 99)
