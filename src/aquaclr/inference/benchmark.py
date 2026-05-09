"""Latency / VRAM benchmark for LEGION-DeSnow.

Reports p50 / p95 / mean latency, throughput (FPS), and peak CUDA
memory for both the PyTorch and (optionally) TensorRT inference paths.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger("aquaclr.bench")


@dataclass
class BenchmarkResult:
    """Latency benchmark summary."""

    p50_ms: float
    p95_ms: float
    mean_ms: float
    fps: float
    peak_vram_mb: float
    n_iters: int

    def pretty(self) -> str:
        """Human-readable one-liner."""
        return (
            f"p50={self.p50_ms:.2f}ms  p95={self.p95_ms:.2f}ms  "
            f"mean={self.mean_ms:.2f}ms  FPS={self.fps:.1f}  "
            f"peak_vram={self.peak_vram_mb:.1f}MB  n={self.n_iters}"
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def benchmark_pytorch(
    model: nn.Module,
    *,
    input_shape: tuple[int, int, int, int] = (1, 3, 720, 1280),
    n_warmup: int = 20,
    n_iters: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_fp16: bool = True,
) -> BenchmarkResult:
    """Benchmark PyTorch inference latency.

    Args:
        model: ``nn.Module`` exposing a forward pass on ``[0, 1]`` RGB tensors.
        input_shape: ``(B, 3, H, W)``.
        n_warmup: Warmup iterations (excluded from stats).
        n_iters: Measured iterations.
        device: ``"cuda"`` or ``"cpu"``.
        use_fp16: Cast model + input to FP16 (CUDA only).

    Returns:
        :class:`BenchmarkResult`.
    """
    model = model.to(device).eval()
    dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
    if use_fp16 and device == "cuda":
        model = model.half()

    x = torch.rand(*input_shape, device=device, dtype=dtype)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        timings_ms: list[float] = []
        for _ in range(n_iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
    )
    mean_ms = statistics.fmean(timings_ms)
    return BenchmarkResult(
        p50_ms=_percentile(timings_ms, 50),
        p95_ms=_percentile(timings_ms, 95),
        mean_ms=mean_ms,
        fps=1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        peak_vram_mb=peak_vram_mb,
        n_iters=n_iters,
    )
