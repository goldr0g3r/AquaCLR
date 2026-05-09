"""PyTorch -> ONNX export for LEGION-DeSnow.

Designed to produce a single ONNX file that:

* takes a 4D ``[0, 1]`` RGB tensor as input,
* emits ``(J, t, B)`` as outputs,
* supports **dynamic batch and spatial dims** so the same engine can
  handle 256x256 training crops, 720p inference, and ROS2 subscriber
  resolutions.

We avoid ``torch.compile`` and channels-last memory format around the
export call because both interact poorly with the ONNX graph
exporter (as of torch 2.5).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger("aquaclr.export")


def export_to_onnx(
    model: nn.Module,
    output_path: Path | str,
    *,
    input_shape: tuple[int, int, int, int] = (1, 3, 720, 1280),
    opset: int = 17,
    dynamic_axes: bool = True,
    simplify: bool = True,
    verify: bool = True,
    rtol: float = 1.0e-2,
    atol: float = 1.0e-3,
) -> Path:
    """Export a LEGION-DeSnow model to ONNX.

    Args:
        model: A :class:`aquaclr.models.LEGIONDeSnowNet` (or any module
            exposing ``forward_export(I) -> (J, t, B)``).
        output_path: Where to write the .onnx file.
        input_shape: ``(B, 3, H, W)`` shape used to trace the graph.
        opset: ONNX opset version. 17+ is recommended for dynamic
            ``Resize`` op support.
        dynamic_axes: If True, mark batch + H + W as dynamic.
        simplify: If True and ``onnxsim`` is installed, run ``onnx-simplifier``
            to fold constants and merge ops.
        verify: Sanity-check the exported graph against PyTorch outputs.
        rtol: Relative tolerance for verification.
        atol: Absolute tolerance for verification.

    Returns:
        Path to the written ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    if hasattr(model, "use_channels_last") and getattr(model, "use_channels_last", False):
        # Channels-last sometimes confuses the ONNX exporter; switch to contiguous.
        model.to(memory_format=torch.contiguous_format)

    dummy = torch.rand(*input_shape, dtype=torch.float32)
    if next(model.parameters()).is_cuda:
        dummy = dummy.cuda()

    forward_export = getattr(model, "forward_export", None)
    if forward_export is None:
        msg = (
            "Model does not expose forward_export(); pass a LEGIONDeSnowNet "
            "or wrap your custom model with a tuple-returning forward."
        )
        raise AttributeError(msg)

    wrapper = _ExportWrapper(model)

    dyn_axes: dict[str, dict[int, str]] | None = None
    if dynamic_axes:
        dyn_axes = {
            "i": {0: "batch", 2: "height", 3: "width"},
            "j": {0: "batch", 2: "height", 3: "width"},
            "t": {0: "batch", 2: "height", 3: "width"},
            "b": {0: "batch"},
        }

    logger.info("Exporting ONNX -> %s (opset=%d, dynamic=%s)", output_path, opset, dynamic_axes)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        input_names=["i"],
        output_names=["j", "t", "b"],
        opset_version=opset,
        dynamic_axes=dyn_axes,
        do_constant_folding=True,
    )

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            graph = onnx.load(str(output_path))
            simplified, ok = onnx_simplify(graph)
            if ok:
                onnx.save(simplified, str(output_path))
                logger.info("Simplified with onnxsim")
            else:
                logger.warning("onnxsim returned ok=False; keeping un-simplified graph")
        except ImportError:
            logger.warning("onnxsim not installed; skipping graph simplification")

    if verify:
        _verify_onnx(model, output_path, dummy, rtol=rtol, atol=atol)

    return output_path


class _ExportWrapper(nn.Module):
    """Tiny shim so ONNX sees a single forward returning a flat tuple."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward_export(i)  # type: ignore[no-any-return]


def _verify_onnx(
    model: nn.Module,
    onnx_path: Path,
    dummy: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Run the exported ONNX with onnxruntime and check parity."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping ONNX verification")
        return

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    with torch.no_grad():
        torch_out = model.forward_export(dummy)  # type: ignore[operator]
        torch_j, torch_t, torch_b = (o.cpu().float().numpy() for o in torch_out)

    onnx_outs = sess.run(None, {"i": dummy.cpu().float().numpy()})
    j, t, b = onnx_outs

    import numpy as np

    for name, ref, got in [("j", torch_j, j), ("t", torch_t, t), ("b", torch_b, b)]:
        max_abs = float(np.abs(ref - got).max())
        if not np.allclose(ref, got, rtol=rtol, atol=atol):
            msg = f"ONNX/PyTorch parity check failed on '{name}': max_abs={max_abs:.5f}"
            raise RuntimeError(msg)
        logger.info("ONNX parity '%s' OK (max_abs=%.5f)", name, max_abs)
