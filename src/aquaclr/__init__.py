"""AquaCLR — Project LEGION subsea perception front-end.

This package provides the physics-informed marine-snow removal model
(:class:`aquaclr.models.LEGIONDeSnowNet`), its training utilities, and the
inference + ROS2 deployment hooks that comprise Milestone 1.

Automotive SiL parallel:
    AquaCLR plays the role of an ADAS sensor preprocessing block (think
    "camera de-rain" or "lidar declutter") sitting upstream of SLAM and
    perception. The Jaffe-McGlamery image formation model is the underwater
    analogue of the Beer-Lambert atmospheric scattering model used to
    synthesise rain/fog in automotive simulators (e.g. NVIDIA DriveSim).
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
