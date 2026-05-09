"""ROS2 node skeleton for LEGION-DeSnow (Humble + Jazzy compatible).

This is the **M1 skeleton**: it subscribes to a raw camera topic,
runs LEGION-DeSnow on each frame, and publishes the cleaned image.
M2 will extend this with a SLAM-friendly auxiliary topic (transmission
+ confidence) and a synchronised stereo path.

The node only uses the stable ``rclpy.Node`` + ``cv_bridge`` +
``sensor_msgs/Image`` API surface, so it runs unchanged on:

* **ROS2 Humble Hawksbill** (Ubuntu 22.04 LTS, supported until May 2027)
* **ROS2 Jazzy Jalisco** (Ubuntu 24.04 LTS, supported until May 2029)

For a deployment runbook on Fedora 44 + a Ubuntu 24.04/Jazzy distrobox
container with NVIDIA passthrough, see ``docs/DEPLOYMENT_FEDORA.md``.

Topics (all configurable via ROS parameters):

============================ ============================================== ===============
Parameter                    Default                                        Direction
============================ ============================================== ===============
``input_topic``              ``/camera/image_raw``                          subscribe
``output_topic``             ``/camera/image_desnowed``                     publish
``transmission_topic``       ``/camera/transmission`` (M2)                  publish (TODO)
``backend``                  ``trt`` if available, else ``torch``           param
``engine_path``              ``""``                                         param
``checkpoint_path``          ``""``                                         param
``device``                   ``cuda`` if available else ``cpu``             param
``qos_depth``                ``5``                                          param
============================ ============================================== ===============

Automotive SiL parallel:
    Drop-in equivalent of an ADAS de-rain camera preprocessing node:
    same topic-rewiring pattern that DriveOS / Apex.AI use to insert
    sensor restoration in front of perception.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from aquaclr.models import LEGIONDeSnowNet

if TYPE_CHECKING:  # pragma: no cover
    from rclpy.node import Node as RclpyNode
    from sensor_msgs.msg import Image as ImageMsg

logger = logging.getLogger("aquaclr.ros2")


def _import_ros2() -> tuple[Any, Any, Any, Any]:
    """Import ROS2 helpers; raise a helpful error if unavailable."""
    try:
        import rclpy  # noqa: PLC0415
        from cv_bridge import CvBridge  # noqa: PLC0415
        from rclpy.node import Node  # noqa: PLC0415
        from sensor_msgs.msg import Image as ImageMsg  # noqa: PLC0415

        return rclpy, Node, CvBridge, ImageMsg
    except ImportError as exc:  # pragma: no cover
        msg = (
            "ROS2 / cv_bridge are not importable. The ROS2 node requires a "
            "ROS2 Humble or Jazzy environment with cv_bridge installed via apt:\n"
            "  Humble (Ubuntu 22.04): "
            "sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs\n"
            "  Jazzy  (Ubuntu 24.04): "
            "sudo apt install ros-jazzy-cv-bridge ros-jazzy-sensor-msgs\n"
            "See docs/DEPLOYMENT_FEDORA.md for the Fedora-host runbook."
        )
        raise ImportError(msg) from exc


class _TorchBackend:
    """Plain PyTorch fallback inference backend."""

    def __init__(self, ckpt_path: Path | None, device: str) -> None:
        self.device = torch.device(device)
        self.model = LEGIONDeSnowNet().to(self.device).eval()
        if ckpt_path is not None and ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = state.get("state_dict", state)
            sd = {
                k.removeprefix("net._orig_mod.").removeprefix("net."): v
                for k, v in sd.items()
                if k.startswith("net.") or not any(s.startswith("net.") for s in sd)
            }
            self.model.load_state_dict(sd, strict=False)
        if self.device.type == "cuda":
            self.model = self.model.half()

    def __call__(self, image_hwc_uint8: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        with torch.no_grad():
            t = torch.from_numpy(image_hwc_uint8).to(self.device)
            t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            if self.device.type == "cuda":
                t = t.half()
            out = self.model(t)
            j = out.j.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (j * 255.0).clip(0, 255).astype(np.uint8)


class LegionDeSnowNode:
    """ROS2 Humble node: subscribe to a camera topic, publish cleaned frames.

    The class is built so that it can be **imported** even when ROS2 is
    not installed (we only touch ``rclpy`` from inside ``main``). This
    keeps the package CI-friendly on Windows / macOS and lets unit
    tests check the inference plumbing without ROS.
    """

    DEFAULT_INPUT_TOPIC = "/camera/image_raw"
    DEFAULT_OUTPUT_TOPIC = "/camera/image_desnowed"

    def __init__(self, node: RclpyNode | None = None) -> None:
        if node is None:  # pragma: no cover - exercised only inside ROS env
            rclpy, Node, _, _ = _import_ros2()
            rclpy.init()
            node = Node("legion_desnow")  # type: ignore[assignment]
        self.node: Any = node
        self._declare_parameters()
        self._build_backend()
        self._wire_topics()

    # ------------------------------------------------------------ params

    def _declare_parameters(self) -> None:
        self.node.declare_parameter("input_topic", self.DEFAULT_INPUT_TOPIC)
        self.node.declare_parameter("output_topic", self.DEFAULT_OUTPUT_TOPIC)
        self.node.declare_parameter("backend", "auto")
        self.node.declare_parameter("engine_path", "")
        self.node.declare_parameter("checkpoint_path", "")
        self.node.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.node.declare_parameter("qos_depth", 5)
        self.node.declare_parameter("log_every_n_frames", 30)

    def _p(self, name: str) -> Any:
        return self.node.get_parameter(name).get_parameter_value().string_value or self.node.get_parameter(name).value

    # ----------------------------------------------------------- backend

    def _build_backend(self) -> None:
        backend_name = str(self._p("backend") or "auto").lower()
        engine_path_str = str(self._p("engine_path") or "")
        ckpt_path_str = str(self._p("checkpoint_path") or "")
        device = str(self._p("device") or "cpu")

        if backend_name in {"auto", "trt"} and engine_path_str:
            try:
                from aquaclr.inference.inference_trt import TensorRTRunner

                self.backend = TensorRTRunner(engine_path_str)
                self.backend_kind = "trt"
                self.node.get_logger().info(f"Using TensorRT backend: {engine_path_str}")
                return
            except (ImportError, FileNotFoundError, RuntimeError) as exc:
                if backend_name == "trt":
                    raise
                self.node.get_logger().warning(
                    f"TRT backend unavailable ({exc!s}); falling back to PyTorch."
                )

        ckpt = Path(ckpt_path_str) if ckpt_path_str else None
        self.backend = _TorchBackend(ckpt, device=device)
        self.backend_kind = "torch"
        self.node.get_logger().info(f"Using PyTorch backend on {device}")

    # ----------------------------------------------------------- topics

    def _wire_topics(self) -> None:
        _, _, CvBridge, ImageMsg = _import_ros2()
        self.bridge = CvBridge()
        self.frame_count = 0
        depth = int(self._p("qos_depth") or 5)
        in_topic = str(self._p("input_topic") or self.DEFAULT_INPUT_TOPIC)
        out_topic = str(self._p("output_topic") or self.DEFAULT_OUTPUT_TOPIC)

        self.publisher = self.node.create_publisher(ImageMsg, out_topic, depth)
        self.subscription = self.node.create_subscription(
            ImageMsg, in_topic, self._on_image, depth
        )
        self.node.get_logger().info(
            f"LEGION-DeSnow node wired: {in_topic} -> {out_topic} (qos={depth})"
        )

    # ----------------------------------------------------------- on_image

    def _on_image(self, msg: ImageMsg) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:  # pragma: no cover  # noqa: BLE001
            self.node.get_logger().warning(f"cv_bridge conversion failed: {exc!s}")
            return

        t0 = time.perf_counter()
        clean = self.backend(np.ascontiguousarray(cv_image))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        try:
            out_msg = self.bridge.cv2_to_imgmsg(clean, encoding="rgb8")
            out_msg.header = msg.header
            self.publisher.publish(out_msg)
        except Exception as exc:  # pragma: no cover  # noqa: BLE001
            self.node.get_logger().warning(f"Failed to publish cleaned frame: {exc!s}")
            return

        self.frame_count += 1
        log_every = int(self._p("log_every_n_frames") or 30)
        if log_every > 0 and self.frame_count % log_every == 0:
            self.node.get_logger().info(
                f"[{self.backend_kind}] frame {self.frame_count} processed in {elapsed_ms:.1f} ms"
            )

    # --------------------------------------------------------- TODO(M2)
    # M2 will:
    #   - publish the predicted transmission map on /camera/transmission as a
    #     mono16 image (so downstream SLAM can use it as per-pixel confidence),
    #   - synchronise stereo pairs through ApproximateTime,
    #   - expose a service to hot-swap engines without restarting the node.


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - ROS env only
    """ROS2 entry point.

    Args:
        argv: Optional argv override (used by tests).
    """
    if argv is None:
        argv = sys.argv
    rclpy, Node, _, _ = _import_ros2()
    rclpy.init(args=argv)
    node = Node("legion_desnow")
    try:
        LegionDeSnowNode(node)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
