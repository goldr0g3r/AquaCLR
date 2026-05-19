"""TensorRT engine builder + Python runner for LEGION-DeSnow.

This module provides:

1. :func:`build_engine_from_onnx` — construct a TensorRT engine from
   an ONNX file (FP16 by default, INT8 deferred to M2). Accepts dynamic
   shape profiles so one engine can serve 256-720p inputs.
2. :class:`TensorRTRunner` — convenience wrapper around the runtime
   API that hides the host/device buffer choreography and exposes a
   simple ``__call__(I_uint8) -> J_uint8`` interface.

If TensorRT or PyCUDA aren't installed, the module imports cleanly but
all functions raise an actionable error at call time. This way the
package remains import-safe on Windows / macOS dev boxes that lack the
NVIDIA stack.

Automotive SiL parallel:
    The same engine-cache + dynamic-shape-profile pattern is what
    NVIDIA DRIVE Orin SDK uses to ship a single ``.engine`` artefact
    that handles all camera modes the autonomous stack will ever see.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - avoid hard dependency on TRT for type-checking
    import tensorrt as trt

logger = logging.getLogger("aquaclr.trt")


def _require_tensorrt() -> Any:
    try:
        import tensorrt as trt  # noqa: PLC0415

        return trt
    except ImportError as exc:  # pragma: no cover
        msg = (
            "TensorRT is required for build/run but isn't installed. "
            "Install with: pip install 'aquaclr[trt]' (Linux only) "
            "or follow NVIDIA's TensorRT installation guide."
        )
        raise ImportError(msg) from exc


def _require_pycuda() -> Any:
    try:
        import pycuda.autoinit  # noqa: F401, PLC0415
        import pycuda.driver as cuda  # noqa: PLC0415

        return cuda
    except ImportError as exc:  # pragma: no cover
        msg = "pycuda is required for the TensorRT runner. Install with: pip install pycuda"
        raise ImportError(msg) from exc


def build_engine_from_onnx(
    onnx_path: Path | str,
    engine_path: Path | str,
    *,
    fp16: bool = True,
    workspace_gb: float = 1.5,
    min_shape: tuple[int, int, int, int] = (1, 3, 256, 256),
    opt_shape: tuple[int, int, int, int] = (1, 3, 720, 1280),
    max_shape: tuple[int, int, int, int] = (1, 3, 720, 1280),
    use_cuda_graph: bool = True,  # noqa: ARG001 - hint for downstream users
) -> Path:
    """Build a TensorRT engine from an ONNX graph.

    Args:
        onnx_path: Path to the ONNX file.
        engine_path: Where to write the serialised engine (``.engine``).
        fp16: Enable FP16 mode. Strongly recommended on RTX 30xx; halves
            VRAM and ~doubles throughput.
        workspace_gb: Builder workspace memory ceiling. 1.5 GB is plenty
            for our small model on a 4 GB RTX 3050.
        min_shape: Minimum input shape for the dynamic profile.
        opt_shape: The "tuned for this shape" point — pick your most
            common inference resolution (720p by default).
        max_shape: Maximum input shape for the dynamic profile.
        use_cuda_graph: Hint flag (passed through to the runner). The
            engine itself is shape-agnostic.

    Returns:
        Path to the serialised engine.
    """
    trt = _require_tensorrt()
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
            msg = "Failed to parse ONNX:\n" + "\n".join(errors)
            raise RuntimeError(msg)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("i", min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    logger.info(
        "Building TRT engine from %s (FP16=%s, workspace=%.1fGB)",
        onnx_path,
        fp16,
        workspace_gb,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        msg = "TensorRT engine build returned None — check the trt_logger output."
        raise RuntimeError(msg)
    engine_path.write_bytes(bytes(serialized))
    logger.info(
        "Wrote engine to %s (%.1f MB)",
        engine_path,
        engine_path.stat().st_size / (1 << 20),
    )
    return engine_path


class TensorRTRunner:
    """Minimal TensorRT runtime wrapper for LEGION-DeSnow.

    Args:
        engine_path: Path to a serialised ``.engine`` (built via
            :func:`build_engine_from_onnx`).

    Usage::

        runner = TensorRTRunner("legion_desnow.engine")
        clean_uint8 = runner(snowy_uint8)   # HxWx3 -> HxWx3
    """

    def __init__(self, engine_path: Path | str) -> None:
        trt = _require_tensorrt()
        cuda = _require_pycuda()
        self._trt = trt
        self._cuda = cuda

        engine_path = Path(engine_path)
        if not engine_path.exists():
            msg = f"Engine file not found: {engine_path}"
            raise FileNotFoundError(msg)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            msg = f"Failed to deserialise engine at {engine_path}"
            raise RuntimeError(msg)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self._input_name = "i"
        self._output_names: list[str] = ["j", "t", "b"]

    def _set_input_shape(self, shape: tuple[int, int, int, int]) -> None:
        ok = self.context.set_input_shape(self._input_name, shape)
        if not ok:
            msg = f"set_input_shape({shape}) rejected by TRT context"
            raise RuntimeError(msg)

    def _allocate(
        self, shape: tuple[int, ...], dtype: np.dtype[Any]
    ) -> tuple[NDArray[Any], int]:
        host = np.empty(shape, dtype=dtype)
        device_ptr = int(self._cuda.mem_alloc(host.nbytes))
        return host, device_ptr

    def __call__(self, i_hwc_uint8: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Run the engine on a single uint8 HWC image.

        Args:
            i_hwc_uint8: Input image, shape ``(H, W, 3)``, dtype uint8.

        Returns:
            Cleaned image ``J``, shape ``(H, W, 3)``, dtype uint8.
        """
        if i_hwc_uint8.ndim != 3 or i_hwc_uint8.shape[2] != 3:
            msg = f"Expected HWC RGB uint8, got shape {i_hwc_uint8.shape}, dtype {i_hwc_uint8.dtype}"
            raise ValueError(msg)
        h, w = i_hwc_uint8.shape[:2]
        i_chw = np.ascontiguousarray(
            i_hwc_uint8.transpose(2, 0, 1).astype(np.float32) / 255.0
        ).reshape(1, 3, h, w)

        self._set_input_shape((1, 3, h, w))

        # Allocate device buffers — keep DeviceAllocation objects alive
        d_input = self._cuda.mem_alloc(i_chw.nbytes)
        self._cuda.memcpy_htod(d_input, i_chw)

        d_outputs: dict[str, Any] = {}
        h_outputs: dict[str, NDArray[Any]] = {}
        for name in self._output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            host = np.empty(shape, dtype=np.float32)
            d_outputs[name] = self._cuda.mem_alloc(host.nbytes)
            h_outputs[name] = host

        # Set tensor addresses (TRT needs int pointers)
        self.context.set_tensor_address(self._input_name, int(d_input))
        for name, d_buf in d_outputs.items():
            self.context.set_tensor_address(name, int(d_buf))

        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()

        for name, host in h_outputs.items():
            self._cuda.memcpy_dtoh(host, d_outputs[name])

        j_chw = h_outputs["j"][0]
        j_hwc = np.clip(j_chw.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        return j_hwc
