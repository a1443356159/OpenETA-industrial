"""Disabled M3 scene metadata and shared quaternion helpers.

M3 manipulation is deliberately unavailable until a separately approved
native ``DetachableJoint`` implementation exists.  This module retains only
static object declarations consumed by the Oracle-perception and benchmark
metadata contracts; it contains no grasp, contact, attachment, force, motion,
or verification implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .m2 import M2Config


M3_ENV_ID = "openeta/gazebo_rm75_robotiq2f85_pickplace-v0"
M3_MODEL_ID = "rm75_robotiq_2f85_pickplace_sim_v1"
M3_DISPLAY_NAME = "Gazebo 仿真环境（M3 已禁用；DetachableJoint 待批准）"
M3_UNAVAILABLE_REASON = "DETACHABLE_JOINT_UNIMPLEMENTED_OR_UNAPPROVED"


@dataclass(frozen=True, slots=True)
class M3Config(M2Config):
    """Static scene declarations, not an executable manipulation profile."""

    model_id: str = M3_MODEL_ID
    env_id: str = M3_ENV_ID
    display_name: str = M3_DISPLAY_NAME
    target_id: str = "m3_target"
    distractor_id: str = "m3_distractor"
    table_id: str = "m3_table"
    table_size_m: tuple[float, float, float] = (0.70, 0.60, 0.04)
    table_pose_xyz: tuple[float, float, float] = (0.40, 0.0, 0.38)
    table_top_z_m: float = 0.40
    target_size_m: tuple[float, float, float] = (0.04, 0.04, 0.06)
    target_mass_kg: float = 0.10
    target_initial_xyz: tuple[float, float, float] = (0.28, -0.10, 0.43)
    distractor_size_m: tuple[float, float] = (0.05, 0.08)
    distractor_mass_kg: float = 0.12
    distractor_initial_xyz: tuple[float, float, float] = (0.28, 0.12, 0.44)
    destination_center_xy: tuple[float, float] = (0.48, -0.10)
    destination_size_xy_m: tuple[float, float] = (0.12, 0.12)

    def validate_assets(self, *, require_vendor: bool = True) -> None:
        """Fail closed before any M3/M4 manipulation runtime is started."""

        del require_vendor
        raise RuntimeError(M3_UNAVAILABLE_REASON)


def quaternion_rotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    """Rotate vector *v* by an xyzw quaternion without external dependencies."""

    if len(q) != 4 or len(v) != 3:
        raise ValueError("quaternion/vector dimensions are invalid")
    x, y, z, w = (float(value) for value in q)
    vx, vy, vz = (float(value) for value in v)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
