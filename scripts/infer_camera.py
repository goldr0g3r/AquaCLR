"""Real-time camera / video inference with LEGION-DeSnow.

Examples::

    # Webcam (PyTorch)
    uv run python scripts/infer_camera.py --ckpt outputs/20260518-143022/ckpts/best.ckpt

    # Webcam (TensorRT engine)
    uv run python scripts/infer_camera.py --engine outputs/legion_desnow.engine

    # Video file
    uv run python scripts/infer_camera.py --ckpt best.ckpt --source path/to/video.mp4

    # Save output (no display)
    uv run python scripts/infer_camera.py --ckpt best.ckpt --save out.mp4 --no-display
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("aquaclr.infer_camera")


class _InferenceBackend:
    def __init__(self, ckpt_path: Path | None, device: str, half: bool = True) -> None:
        from aquaclr.models.model import LEGIONDeSnowNet

        self.device = torch.device(device)
        self._use_half = half and self.device.type == "cuda"
        self.model = LEGIONDeSnowNet().to(self.device).eval()

        if ckpt_path is not None:
            logger.info("Loading checkpoint: %s", ckpt_path)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = state.get("state_dict", state)
            sd = {
                k.removeprefix("net._orig_mod.").removeprefix("net."): v
                for k, v in sd.items()
                if k.startswith("net.") or not any(s.startswith("net.") for s in sd)
            }
            self.model.load_state_dict(sd, strict=False)
        else:
            logger.warning("No checkpoint — using random weights")

        if self._use_half:
            self.model = self.model.half()

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        t = t.to(self.device)
        if self._use_half:
            t = t.half()
        out = self.model(t)
        j = out.j.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor((j * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


class _TensorRTBackend:
    def __init__(self, engine_path: Path) -> None:
        from aquaclr.inference.inference_trt import TensorRTRunner

        logger.info("Loading TensorRT engine: %s", engine_path)
        self._runner = TensorRTRunner(engine_path)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        clean_rgb = self._runner(rgb)
        return cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--engine", type=Path, default=None, help="TensorRT .engine file (overrides --ckpt)")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Resize shorter edge before inference (e.g. 512)",
    )
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if args.engine is not None:
        backend = _TensorRTBackend(args.engine)
    else:
        backend = _InferenceBackend(args.ckpt, args.device, half=not args.no_half)

    src: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", args.source)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.save:
        writer = cv2.VideoWriter(
            str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w * 2, h)
        )

    frame_count, t_total = 0, 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        infer_frame = frame
        if args.resize:
            scale = args.resize / min(h, w)
            infer_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        t0 = time.perf_counter()
        enhanced = backend(infer_frame)
        t_total += time.perf_counter() - t0
        frame_count += 1

        if args.resize:
            enhanced = cv2.resize(enhanced, (w, h))

        sbs = np.concatenate([frame, enhanced], axis=1)
        avg_ms = t_total / frame_count * 1000
        cv2.putText(
            sbs, "Raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
        )
        cv2.putText(
            sbs,
            f"Enhanced  {avg_ms:.1f}ms",
            (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 100),
            2,
        )

        if writer:
            writer.write(sbs)
        if not args.no_display:
            cv2.imshow("LEGION-DeSnow | Raw (left)  Enhanced (right)", sbs)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    logger.info(
        "%d frames, avg %.1f ms/frame", frame_count, t_total / frame_count * 1000
    )
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
