"""Export a checkpoint (or fresh init) to ONNX (and optionally to TensorRT).

Smoke mode (``--smoke``) builds a tiny model with random weights and
exports it. Used by CI to catch regressions in the export path even on
GPU-less runners.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from aquaclr.inference.benchmark import benchmark_pytorch
from aquaclr.inference.onnx_export import export_to_onnx
from aquaclr.models import LEGIONDeSnowNet

logger = logging.getLogger("aquaclr.export.cli")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=None, help="Lightning checkpoint (.ckpt)")
    p.add_argument("--out", type=Path, default=Path("outputs/legion_desnow.onnx"))
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-simplify", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--build-trt", action="store_true", help="Also build a TensorRT engine.")
    p.add_argument("--engine-out", type=Path, default=None)
    p.add_argument("--benchmark", action="store_true", help="Run a PyTorch latency benchmark.")
    p.add_argument("--smoke", action="store_true", help="CI smoke mode: tiny init, CPU-only.")
    return p.parse_args()


def main() -> None:
    """Run ONNX export."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    args = _parse_args()

    if args.smoke:
        model = LEGIONDeSnowNet(pretrained=False, use_channels_last=False)
        h, w = 64, 64
        out = args.out.with_name("legion_desnow_smoke.onnx")
    else:
        model = LEGIONDeSnowNet()
        h, w = args.height, args.width
        out = args.out

    if args.ckpt is not None:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            sd = {
                k.removeprefix("net._orig_mod.").removeprefix("net."): v
                for k, v in state["state_dict"].items()
                if k.startswith("net.")
            }
        else:
            sd = state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logger.warning("missing=%s unexpected=%s", missing[:5], unexpected[:5])

    model.eval()
    onnx_path = export_to_onnx(
        model,
        out,
        input_shape=(args.batch, 3, h, w),
        opset=args.opset,
        simplify=not args.no_simplify,
        verify=not args.no_verify,
    )
    logger.info("ONNX export complete: %s", onnx_path)

    if args.build_trt:
        from aquaclr.inference.inference_trt import build_engine_from_onnx

        engine_out = args.engine_out or onnx_path.with_suffix(".engine")
        build_engine_from_onnx(
            onnx_path,
            engine_out,
            min_shape=(1, 3, 256, 256),
            opt_shape=(1, 3, h, w),
            max_shape=(1, 3, h, w),
        )

    if args.benchmark:
        result = benchmark_pytorch(
            model,
            input_shape=(args.batch, 3, h, w),
            use_fp16=torch.cuda.is_available(),
        )
        logger.info("Benchmark | %s", result.pretty())


if __name__ == "__main__":
    main()
