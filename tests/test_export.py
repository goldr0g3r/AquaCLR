"""ONNX export and TRT smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aquaclr.models import LEGIONDeSnowNet


def _have_onnx() -> bool:
    try:
        import onnx  # noqa: F401, PLC0415
        import onnxruntime  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_onnx(), reason="onnx / onnxruntime not installed")
def test_onnx_export_parity(tmp_path: Path) -> None:
    from aquaclr.inference.onnx_export import export_to_onnx

    model = LEGIONDeSnowNet(pretrained=False, use_channels_last=False).eval()
    out = tmp_path / "legion.onnx"
    export_to_onnx(
        model,
        out,
        input_shape=(1, 3, 64, 64),
        opset=17,
        simplify=False,
        verify=True,
        rtol=1e-2,
        atol=1e-3,
    )
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(not _have_onnx(), reason="onnx / onnxruntime not installed")
def test_onnx_dynamic_shapes(tmp_path: Path) -> None:
    """Verify the exported graph can run on a different spatial size."""
    import onnxruntime as ort  # type: ignore[import-not-found]

    from aquaclr.inference.onnx_export import export_to_onnx

    model = LEGIONDeSnowNet(pretrained=False, use_channels_last=False).eval()
    out = tmp_path / "legion_dyn.onnx"
    export_to_onnx(
        model,
        out,
        input_shape=(1, 3, 64, 64),
        opset=17,
        simplify=False,
        verify=False,
    )
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((1, 3, 96, 128), dtype=np.float32)
    j, t, b = sess.run(None, {"i": x})
    assert j.shape == (1, 3, 96, 128)
    assert t.shape == (1, 1, 96, 128)
    assert b.shape == (1, 3)


@pytest.mark.gpu
def test_pytorch_benchmark_runs() -> None:
    from aquaclr.inference.benchmark import benchmark_pytorch

    model = LEGIONDeSnowNet(pretrained=False, use_channels_last=False).eval()
    res = benchmark_pytorch(
        model,
        input_shape=(1, 3, 256, 256),
        n_warmup=2,
        n_iters=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fp16=torch.cuda.is_available(),
    )
    assert res.mean_ms > 0


@pytest.mark.trt
def test_trt_engine_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt")
    pytest.importorskip("pycuda")

    from aquaclr.inference.inference_trt import (
        TensorRTRunner,
        build_engine_from_onnx,
    )
    from aquaclr.inference.onnx_export import export_to_onnx

    model = LEGIONDeSnowNet(pretrained=False, use_channels_last=False).cuda().eval()
    onnx_path = tmp_path / "legion.onnx"
    export_to_onnx(model, onnx_path, input_shape=(1, 3, 256, 256), simplify=False, verify=False)

    eng = tmp_path / "legion.engine"
    build_engine_from_onnx(onnx_path, eng, fp16=True)
    runner = TensorRTRunner(eng)
    img = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
    out = runner(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
