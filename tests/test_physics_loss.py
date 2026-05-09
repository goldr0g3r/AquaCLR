"""Loss correctness tests."""

from __future__ import annotations

import torch

from aquaclr.losses import PhysicsInformedLoss
from aquaclr.utils.physics import (
    apply_forward_jaffe_mcglamery,
    invert_jaffe_mcglamery,
)


def test_physics_round_trip_is_identity_on_clean_inputs() -> None:
    """If we feed the *true* (J, t, B) into the forward model and then invert,
    we should recover J up to small numerical noise.
    """
    j = torch.rand(2, 3, 32, 32)
    t = torch.rand(2, 1, 32, 32) * 0.6 + 0.3  # avoid extreme t -> singular inversion
    b = torch.rand(2, 3) * 0.5

    i = apply_forward_jaffe_mcglamery(j, t, b)
    j_back = invert_jaffe_mcglamery(i, t, b)
    assert torch.allclose(j_back, j, atol=2.0e-3), \
        f"max diff = {(j_back - j).abs().max().item():.4f}"


def test_loss_gradients_finite() -> None:
    loss_fn = PhysicsInformedLoss()
    i = torch.rand(2, 3, 32, 32, requires_grad=True)
    j_pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    j_gt = torch.rand(2, 3, 32, 32)
    t = torch.rand(2, 1, 32, 32, requires_grad=True) * 0.5 + 0.3
    b = torch.rand(2, 3, requires_grad=True) * 0.5

    out = loss_fn(i=i, j_pred=j_pred, j_gt=j_gt, t=t, b=b)
    out.total.backward()

    for tensor in (i, j_pred, t, b):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_loss_with_t_supervision_is_added() -> None:
    """Adding a perfectly-matching ``t_gt`` term shouldn't increase the loss."""
    loss_fn = PhysicsInformedLoss(lambda_t=1.0)
    i = torch.rand(2, 3, 32, 32)
    j_pred = torch.rand(2, 3, 32, 32)
    j_gt = torch.rand(2, 3, 32, 32)
    t = torch.rand(2, 1, 32, 32) * 0.5 + 0.3
    b = torch.rand(2, 3) * 0.5

    out_no_t = loss_fn(i=i, j_pred=j_pred, j_gt=j_gt, t=t, b=b)
    out_with_t = loss_fn(i=i, j_pred=j_pred, j_gt=j_gt, t=t, b=b, t_gt=t.clone())
    assert torch.allclose(out_no_t.total, out_with_t.total, atol=1.0e-6)


def test_loss_components_decrease_when_signals_are_perfect() -> None:
    loss_fn = PhysicsInformedLoss()
    j_gt = torch.rand(2, 3, 16, 16)
    t = torch.rand(2, 1, 16, 16) * 0.5 + 0.3
    b = torch.rand(2, 3) * 0.4
    i = apply_forward_jaffe_mcglamery(j_gt, t, b)
    j_pred = invert_jaffe_mcglamery(i, t, b)
    out = loss_fn(i=i, j_pred=j_pred, j_gt=j_gt, t=t, b=b)
    # Reconstruction and physics terms should be near-zero on perfect signals.
    assert out.recon.item() < 1.0e-2
    assert out.phys.item() < 1.0e-3
