"""ROS2 integration for AquaCLR.

This package is intentionally minimal in M1: it exposes a single
:class:`ros2_node.LegionDeSnowNode` skeleton so the integration story
is in place when M2 wires SLAM downstream.

Run the node with::

    ros2 run aquaclr legion_desnow_node \\
        --ros-args -p engine_path:=/path/to/legion_desnow.engine

(or, during development, ``python -m aquaclr.ros2.ros2_node``).
"""

from __future__ import annotations
