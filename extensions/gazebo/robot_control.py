"""RM75 + Robotiq 2F-85 control contracts.

The module deliberately contains no ROS imports.  ROS action clients and TF/
joint-state subscriptions are injected by the deployment adapter, which keeps
the worker contract testable on machines without Jazzy installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from adapter.protocol import RobotState
from .robotiq_kinematics import (
    GRIPPER_GOAL_TOLERANCE_RAD,
    DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD,
    attached_transport_relief_position,
    common_driver_position,
    minimum_feasible_active_position,
)

GAZEBO_CONTROL_ENV_ID = "openeta/gazebo_rm75_robotiq2f85-v0"
MODEL_ID = "rm75_robotiq_2f85_sim_v1"
ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 8))
GRIPPER_JOINTS = (
    "gripper_left_finger_joint",
    "gripper_right_finger_joint",
    "gripper_left_inner_knuckle_joint",
    "gripper_right_inner_knuckle_joint",
    "gripper_left_finger_tip_joint",
    "gripper_right_finger_tip_joint",
)
JOINT_NAMES = ARM_JOINTS + GRIPPER_JOINTS
ARM_JOINT_BOUNDS = (
    ("joint_1", -3.106, 3.106),
    ("joint_2", -2.2689, 2.2689),
    ("joint_3", -3.106, 3.106),
    ("joint_4", -2.356, 2.356),
    ("joint_5", -3.106, 3.106),
    ("joint_6", -2.234, 2.234),
    ("joint_7", -6.28, 6.28),
)
ARM_HOME_JOINT_POSITIONS = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    math.pi,
)
START_STATE_RECOVERY_SCHEMA_VERSION = "openeta.gazebo.start_state_recovery.v1"
START_STATE_BOUNDS_TOLERANCE_RAD = 1e-6
START_STATE_RECOVERY_INSET_RAD = 1e-3
START_STATE_RECOVERY_TRAJECTORY_S = 1.0
START_STATE_RECOVERY_TIMEOUT_S = 5.0
MOTION_SETTLE_RECHECK_TIMEOUT_S = 1.0
MOTION_SETTLE_RECHECK_INTERVAL_S = 0.1
# Bound the numerical allowance around the nominal post-motion TF tolerance.
# This covers sub-millimetre sampling/rounding noise without changing the
# requested MoveIt goal or accepting a materially incomplete trajectory.
MOTION_VERIFICATION_NUMERIC_MARGIN_M = 0.0001
# FollowJointTrajectory can report CONTROL_FAILED after the arm has already
# reached and settled at the requested pose (for example, a final controller
# path-tolerance sample arriving just outside its time window).  Reconcile
# only that exact MoveIt outcome, and only from a fresh terminal robot state
# whose seven arm joints are effectively stationary.  This does not relax the
# existing Cartesian terminal tolerances.
MOVEIT_CONTROL_FAILED = -4
MOTION_TERMINAL_MAX_ARM_VELOCITY_RAD_S = 0.001
# These targets are relative to the first fresh mount pose after a reset, not
# absolute world coordinates.  They are the small, validated neutral motions
# available in the empty motion-control profile.  Publishing the relation in the existing
# control_spec lets an agent use the runtime contract instead of guessing a
# lateral Cartesian target from an image or a model prior.
NEUTRAL_RELATIVE_MOTION_TARGETS = (
    ("vertical_low", (0.0, 0.0, -0.040)),
    ("vertical_high", (0.0, 0.0, -0.020)),
)

ERROR_CODES = frozenset(
    {
        "MODEL_ASSET_NOT_FOUND",
        "ROS_NOT_READY",
        "JOINT_STATE_TIMEOUT",
        "TF_TIMEOUT",
        "MOVE_GROUP_UNAVAILABLE",
        "PLANNING_SCENE_UNAVAILABLE",
        "PLANNING_SCENE_SYNC_FAILED",
        "START_STATE_INVALID",
        "START_STATE_RECOVERY_FAILED",
        "MOTION_PLAN_FAILED",
        "MOTION_EXECUTION_FAILED",
        "MOTION_EXECUTION_TIMEOUT",
        "MOTION_OUTCOME_UNKNOWN",
        "MOTION_TARGET_NOT_REACHED",
        "GRIPPER_UNAVAILABLE",
        "GRIPPER_FAILED",
        "GRIPPER_TIMEOUT",
        "ATTACHED_TRANSPORT_HOLD_FAILED",
        "INVALID_CONTROL_ACTION",
        "ROBOT_STATE_UNAVAILABLE",
    }
)


def neutral_relative_motion_guidance() -> dict[str, Any]:
    """Return a fresh, serializable motion-control neutral-motion capability.

    This is descriptive runtime guidance, never an executable macro: callers
    still form one normal ``move_to`` target at a time and must honor the
    resulting structured receipt.
    """

    return {
        "schema_version": "openeta.gazebo.relative_motion.v1",
        "reference": "first_fresh_end_effector_pose_after_reset",
        "frame": "world",
        "orientation": "preserve_observed",
        "targets": [
            {"name": name, "xyz_offset_m": list(offset)}
            for name, offset in NEUTRAL_RELATIVE_MOTION_TARGETS
        ],
        "on_rejection": "observe_and_report; do_not_guess_an_unadvertised_target",
    }


def assess_start_state_bounds(
    state: RobotState,
    *,
    tolerance_rad: float = START_STATE_BOUNDS_TOLERANCE_RAD,
    inset_rad: float = START_STATE_RECOVERY_INSET_RAD,
    freshness_s: float = 2.0,
    now_monotonic_s: float | None = None,
) -> dict[str, Any]:
    """Classify the seven RM75 joints without importing ROS or MoveIt."""

    if tolerance_rad < 0 or inset_rad <= 0 or freshness_s <= 0:
        raise ValueError("invalid start-state bounds policy")
    timestamp = state.metadata.get("joint_state_timestamp_s")
    received = state.metadata.get("joint_state_received_monotonic_s")
    base = {
        "tolerance_rad": float(tolerance_rad),
        "inset_rad": float(inset_rad),
        "pre_joint_state_timestamp_s": (
            float(timestamp)
            if isinstance(timestamp, (int, float)) and math.isfinite(float(timestamp))
            else None
        ),
    }
    if not isinstance(received, (int, float)) or not math.isfinite(float(received)):
        return {
            **base,
            "classification": "INVALID",
            "reason_code": "JOINT_STATE_TIMESTAMP_MISSING",
            "joints": [],
            "candidate_positions": None,
        }
    now = time.monotonic() if now_monotonic_s is None else float(now_monotonic_s)
    age = now - float(received)
    if not math.isfinite(age) or age < 0 or age > freshness_s:
        return {
            **base,
            "classification": "INVALID",
            "reason_code": "JOINT_STATE_STALE",
            "joints": [],
            "candidate_positions": None,
        }

    names = state.metadata.get("joint_names")
    if not isinstance(names, (list, tuple)):
        return {
            **base,
            "classification": "INVALID",
            "reason_code": "ARM_JOINT_NAMES_MISSING",
            "joints": [],
            "candidate_positions": None,
        }
    index = {str(name): offset for offset, name in enumerate(names)}
    missing = [name for name in ARM_JOINTS if name not in index]
    if missing or any(index[name] >= len(state.joint_positions) for name in ARM_JOINTS):
        missing.extend(
            name
            for name in ARM_JOINTS
            if name in index and index[name] >= len(state.joint_positions)
        )
        return {
            **base,
            "classification": "INVALID",
            "reason_code": "ARM_JOINT_MISSING",
            "joints": [{"name": name} for name in dict.fromkeys(missing)],
            "candidate_positions": None,
        }

    positions = [float(state.joint_positions[index[name]]) for name in ARM_JOINTS]
    non_finite = [
        {"name": name, "position_rad": None}
        for name, position in zip(ARM_JOINTS, positions)
        if not math.isfinite(position)
    ]
    if non_finite:
        return {
            **base,
            "classification": "INVALID",
            "reason_code": "ARM_JOINT_NONFINITE",
            "joints": non_finite,
            "candidate_positions": None,
        }

    affected: list[dict[str, Any]] = []
    candidate = list(positions)
    outside_tolerance = False
    boundary_inset_required = False
    for offset, ((name, lower, upper), position) in enumerate(
        zip(ARM_JOINT_BOUNDS, positions)
    ):
        # MoveIt can reject a trajectory that starts exactly on a hard limit
        # even though that floating-point value is formally within the URDF
        # interval.  Treat the certified numeric boundary band as recoverable
        # and inset it before planning; values beyond that band still fail
        # closed below.
        near_lower = lower <= position <= lower + tolerance_rad
        near_upper = upper - tolerance_rad <= position <= upper
        if lower <= position <= upper and not (near_lower or near_upper):
            continue
        if position < lower or near_lower:
            boundary = "lower"
            violation = max(0.0, lower - position)
            target = lower + inset_rad
        else:
            boundary = "upper"
            violation = max(0.0, position - upper)
            target = upper - inset_rad
        boundary_inset_required = boundary_inset_required or violation == 0.0
        candidate[offset] = target
        outside_tolerance = outside_tolerance or (
            violation > tolerance_rad
            and not math.isclose(
                violation, tolerance_rad, rel_tol=0.0, abs_tol=1e-15
            )
        )
        affected.append(
            {
                "name": name,
                "position_rad": position,
                "lower_rad": lower,
                "upper_rad": upper,
                "boundary": boundary,
                "violation_rad": violation,
                "recovery_target_rad": target,
            }
        )

    if not affected:
        return {
            **base,
            "classification": "WITHIN_BOUNDS",
            "reason_code": "START_STATE_WITHIN_BOUNDS",
            "joints": [],
            "candidate_positions": positions,
        }
    if outside_tolerance:
        return {
            **base,
            "classification": "INVALID",
            "reason_code": "BOUNDS_VIOLATION_EXCEEDS_TOLERANCE",
            "joints": affected,
            "candidate_positions": None,
        }
    return {
        **base,
        "classification": "RECOVERABLE",
        "reason_code": (
            "START_STATE_BOUNDARY_INSET"
            if boundary_inset_required
            else "NUMERIC_BOUNDS_VIOLATION"
        ),
        "joints": affected,
        "candidate_positions": candidate,
    }


def start_state_recovery_record(
    assessment: Mapping[str, Any],
    *,
    status: str,
    reason_code: str | None = None,
    attempted: bool = False,
    post_joint_state_timestamp_s: float | None = None,
    trajectory_result_code: int | None = None,
) -> dict[str, Any]:
    """Build the stable, JSON-safe motion-control/native-grasp recovery evidence envelope."""

    return {
        "schema_version": START_STATE_RECOVERY_SCHEMA_VERSION,
        "status": status,
        "reason_code": reason_code or str(assessment["reason_code"]),
        "attempted": bool(attempted),
        "tolerance_rad": float(assessment["tolerance_rad"]),
        "inset_rad": float(assessment["inset_rad"]),
        "joints": [dict(item) for item in assessment.get("joints", ())],
        "pre_joint_state_timestamp_s": assessment.get(
            "pre_joint_state_timestamp_s"
        ),
        "post_joint_state_timestamp_s": post_joint_state_timestamp_s,
        "trajectory_result_code": trajectory_result_code,
    }


@dataclass(frozen=True, slots=True)
class GazeboControlConfig:
    model_id: str = MODEL_ID
    base_link: str = "base_link"
    arm_tip: str = "link_7"
    move_group: str = "rm_group"
    mount_parent: str = "link_7"
    mount_child: str = "gripper_mount_link"
    active_joint: str = GRIPPER_JOINTS[0]
    mimic_joint: str = GRIPPER_JOINTS[1]
    closed_position_m: float = 0.0
    active_open_position_m: float = 0.0425
    maximum_aperture_m: float = 0.085
    gripper_controller_boundary_inset_rad: float = (
        DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD
    )
    # The Robotiq mount frame is position-coincident with the RM75 terminal
    # flange.  Its jaws are rotated 90 degrees about tool Z so the native RM75
    # wrist-camera datum lies in the two-finger symmetry plane.  The vendor
    # base frame remains 6 mm forward, as recorded by the grasp calibration;
    # do not reintroduce a second adapter length here.
    mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_quat_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.7071067811865475,
        0.7071067811865476,
    )
    asset_root_override: str | None = None
    calibration: "Robotiq2F85Calibration" = field(
        default_factory=lambda: Robotiq2F85Calibration()
    )

    def __post_init__(self) -> None:
        if self.maximum_aperture_m != 2 * self.active_open_position_m:
            raise ValueError("maximum aperture must equal twice active opening")
        if self.active_joint == self.mimic_joint or self.mount_parent != self.arm_tip:
            raise ValueError("invalid fixed mount or gripper joint mapping")
        if len(self.mount_xyz) != 3 or len(self.mount_quat_xyzw) != 4:
            raise ValueError("mount transform has invalid dimensions")
        if not 0.0 < self.gripper_controller_boundary_inset_rad < 0.4:
            raise ValueError("invalid gripper controller boundary inset")

    def gripper_position(self, position: int) -> float:
        if type(position) is not int or position not in (0, 1):
            raise ValueError("gripper position must be exactly 0 or 1")
        if position == 0:
            return self.calibration.angles_rad[-1]
        # The theoretical zero-angle pose provides the CAD maximum aperture,
        # but its exact four-bar solution crosses the inner-knuckle URDF stop.
        # Command the nearest physically realisable linkage pose instead of
        # making six independent Gazebo controllers fight a clamped loop.
        return minimum_feasible_active_position(
            boundary_inset_rad=self.gripper_controller_boundary_inset_rad
        )

    @property
    def asset_root(self) -> Path:
        """Embedded asset root; overrides are only for explicit test tooling."""
        return (
            Path(self.asset_root_override)
            if self.asset_root_override
            else Path(__file__).parent / "assets" / "rm75_6fb_v_vendor"
        )

    @property
    def ros_workspace(self) -> Path:
        return Path(__file__).parent / "ros2_ws"

    @property
    def gripper_asset_root(self) -> Path:
        return Path(__file__).parent / "assets" / "robotiq_2f85_vendor"

    @property
    def joint_names(self) -> tuple[str, ...]:
        return JOINT_NAMES

    def validate_assets(self, *, require_vendor: bool = True) -> None:
        del require_vendor  # Compatibility with the early motion-control contract.
        try:
            from .asset_preflight import validate_asset_root

            manifest = validate_asset_root(self.asset_root)
        except Exception as exc:
            raise RuntimeError("MODEL_ASSET_NOT_FOUND") from exc
        if manifest.get("description_id") != "RM75-6FB-V":
            raise RuntimeError("MODEL_ASSET_NOT_FOUND")
        try:
            validate_asset_root(self.gripper_asset_root)
        except Exception as exc:
            raise RuntimeError("MODEL_ASSET_NOT_FOUND") from exc
        package = self.ros_workspace / "src" / "openeta_rm75_robotiq2f85_sim"
        if not (package / "package.xml").is_file():
            raise RuntimeError("MODEL_ASSET_NOT_FOUND")


@dataclass(frozen=True, slots=True)
class Robotiq2F85Calibration:
    """Deterministic offline FK calibration (angle radians -> total aperture m).

    The table is sampled from the upstream four-bar URDF fingertip poses.  The
    adapter interpolates only between these fixed points; no vendor runtime or
    CAD dependency is needed by the API process.
    """

    angles_rad: tuple[float, ...] = (0.0, 0.20, 0.40, 0.60, 0.7929)
    apertures_m: tuple[float, ...] = (0.085, 0.0742, 0.0588, 0.0351, 0.0)

    def __post_init__(self) -> None:
        if len(self.angles_rad) != len(self.apertures_m) or any(
            b <= a for a, b in zip(self.angles_rad, self.angles_rad[1:])
        ):
            raise ValueError("invalid Robotiq calibration table")

    def aperture_from_angle(self, angle_rad: float) -> float:
        x = max(self.angles_rad[0], min(self.angles_rad[-1], float(angle_rad)))
        for i, upper in enumerate(self.angles_rad[1:], 1):
            if x <= upper:
                lo, hi = self.angles_rad[i - 1], upper
                t = (x - lo) / (hi - lo)
                return self.apertures_m[i - 1] + t * (self.apertures_m[i] - self.apertures_m[i - 1])
        return self.apertures_m[-1]

    def angle_from_aperture(self, aperture_m: float) -> float:
        y = max(self.apertures_m[-1], min(self.apertures_m[0], float(aperture_m)))
        # Aperture is descending with angle; interpolate on the reversed view.
        for i, upper in enumerate(self.apertures_m[::-1][1:], 1):
            if y <= upper:
                lo_y, hi_y = self.apertures_m[::-1][i - 1], upper
                t = (y - lo_y) / (hi_y - lo_y)
                return self.angles_rad[::-1][i - 1] + t * (self.angles_rad[::-1][i] - self.angles_rad[::-1][i - 1])
        return self.angles_rad[0]


def robotiq_aperture_to_angle(aperture_m: float, *, calibration: Robotiq2F85Calibration | None = None) -> float:
    return (calibration or Robotiq2F85Calibration()).angle_from_aperture(aperture_m)


def robotiq_angle_to_aperture(angle_rad: float, *, calibration: Robotiq2F85Calibration | None = None) -> float:
    return (calibration or Robotiq2F85Calibration()).aperture_from_angle(angle_rad)


def gripper_state(
    active_position_m: float,
    commanded_position_m: float | None = None,
    *,
    reached_goal: bool = True,
    stalled: bool = False,
    config: GazeboControlConfig | None = None,
) -> dict[str, Any]:
    cfg = config or GazeboControlConfig()
    aperture = cfg.calibration.aperture_from_angle(float(active_position_m))
    p = max(cfg.closed_position_m, min(cfg.active_open_position_m, aperture / 2.0))
    openness = p / cfg.active_open_position_m
    return {
        # RobotState defines the compatibility boolean as a 0.5 threshold;
        # do not require an unrealistically exact fully-open joint value.
        "open": openness > 0.5,
        "openness": openness,
        "active_position_m": p,
        "aperture_m": 2 * p,
        "commanded_position_m": p if commanded_position_m is None else float(commanded_position_m),
        "reached_goal": bool(reached_goal),
        "stalled": bool(stalled),
    }


def robot_state_from_sources(
    joint_state: Mapping[str, Sequence[float]],
    tf: Mapping[str, Any],
    *,
    config: GazeboControlConfig | None = None,
) -> RobotState:
    """Build state only when all required joints and the configured TF exist."""
    cfg = config or GazeboControlConfig()
    names = list(joint_state.get("name", ()))
    required_names = tuple(getattr(cfg, "joint_names", JOINT_NAMES))
    positions = list(joint_state.get("position", ()))
    velocities = list(joint_state.get("velocity", ()))
    if any(j not in names for j in required_names) or len(positions) != len(names):
        raise RuntimeError("JOINT_STATE_TIMEOUT")
    index = {n: i for i, n in enumerate(names)}
    jp = [float(positions[index[n]]) for n in required_names]
    jv = [float(velocities[index[n]]) if index[n] < len(velocities) else 0.0 for n in required_names]
    pose = tf.get(f"{cfg.base_link}->{cfg.mount_child}") or tf.get((cfg.base_link, cfg.mount_child))
    if not isinstance(pose, Mapping) or "xyz" not in pose or "quat_xyzw" not in pose:
        raise RuntimeError("TF_TIMEOUT")
    return RobotState(
        joint_positions=jp,
        joint_velocities=jv,
        end_effector_pose={
            "xyz": list(pose["xyz"]),
            "quat_xyzw": list(pose["quat_xyzw"]),
            "frame": cfg.mount_child,
        },
        gripper_state=gripper_state(jp[7], config=cfg),
        metadata={
            "model_id": cfg.model_id,
            "eef_frame": cfg.mount_child,
            "joint_names": list(required_names),
            "camera_frames": ["top_camera_optical_frame", "wrist_camera_optical_frame"],
        },
    )


def make_move_group_goal(
    target_pose: Mapping[str, Any],
    *,
    config: GazeboControlConfig | None = None,
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    cfg = config or GazeboControlConfig()
    xyz, quat = target_pose.get("xyz"), target_pose.get("quat_xyzw")
    if xyz is None or quat is None or len(xyz) != 3 or len(quat) != 4:
        raise ValueError("target pose must contain xyz and quat_xyzw")
    if not all(math.isfinite(float(v)) for v in (*xyz, *quat)):
        raise ValueError("target pose must be finite")
    tool_xyz, tool_q = tuple(map(float, xyz)), _q_normalize(tuple(map(float, quat)))
    mount_q_inv = (
        -cfg.mount_quat_xyzw[0],
        -cfg.mount_quat_xyzw[1],
        -cfg.mount_quat_xyzw[2],
        cfg.mount_quat_xyzw[3],
    )
    link_q = _q_multiply(tool_q, mount_q_inv)
    offset = _q_rotate(link_q, cfg.mount_xyz)
    link_xyz = [tool_xyz[i] - offset[i] for i in range(3)]
    tolerance_values = tolerances or {}
    loaded_motion = str(target_pose.get("placement_stage") or "") in {
        "release",
        "transport",
    } or str(target_pose.get("purpose") or "") == "placement"
    motion_profile = "loaded" if loaded_motion else "unloaded"
    default_velocity_scaling = float(
        getattr(
            cfg,
            f"{motion_profile}_velocity_scaling",
            0.3,
        )
    )
    default_acceleration_scaling = float(
        getattr(
            cfg,
            f"{motion_profile}_acceleration_scaling",
            0.3,
        )
    )
    goal = {
        "group_name": cfg.move_group,
        "base_frame": cfg.base_link,
        "link_name": cfg.arm_tip,
        "requested_tool_pose": {
            "frame_id": cfg.base_link,
            "xyz": list(tool_xyz),
            "quat_xyzw": list(tool_q),
        },
        **{
            key: target_pose[key]
            for key in (
                "purpose",
                "recovery_id",
                "compiled_grasp_id",
                "grasp_stage",
                "placement_candidate_id",
                "compiled_placement_id",
                "placement_stage",
                "scene_revision",
            )
            if key in target_pose
        },
        "target_pose": {"frame_id": cfg.base_link, "xyz": link_xyz, "quat_xyzw": list(link_q)},
        "position_tolerance_m": float(
            tolerance_values.get(
                "position_tolerance_m", tolerance_values.get("tolerance", 0.002)
            )
        ),
        "orientation_tolerance_rad": float(
            tolerance_values.get(
                "orientation_tolerance_rad",
                tolerance_values.get("ori_tolerance", 0.05),
            )
        ),
        # Carrying moves may request gentler trajectory scaling; the default
        # preserves the long-standing motion-control/native-grasp motion contract.
        "motion_profile": motion_profile,
        "max_velocity_scaling_factor": float(
            tolerance_values.get(
                "max_velocity_scaling_factor",
                tolerance_values.get(
                    "velocity_scaling", default_velocity_scaling
                ),
            )
        ),
        "max_acceleration_scaling_factor": float(
            tolerance_values.get(
                "max_acceleration_scaling_factor",
                tolerance_values.get(
                    "acceleration_scaling", default_acceleration_scaling
                ),
            )
        ),
    }
    qualified_joint_goal = _validated_qualified_joint_goal(target_pose)
    if qualified_joint_goal is None:
        qualified_joint_goal = _validated_recovery_collision_policy(target_pose)
    if qualified_joint_goal is not None:
        goal.update(qualified_joint_goal)
    return goal


def _validated_qualified_joint_goal(
    target_pose: Mapping[str, Any],
) -> dict[str, Any] | None:
    state_field = "qualification_goal_joint_state"
    hash_field = "qualification_goal_joint_state_sha256"
    binding_field = "qualification_binding_sha256"
    allowed_field = "qualification_allowed_collisions"
    allowed_hash_field = "qualification_allowed_collisions_sha256"
    if not any(
        field in target_pose for field in (state_field, hash_field, binding_field)
    ):
        return None
    state = target_pose.get(state_field)
    if not isinstance(state, Mapping):
        raise ValueError("qualified joint goal state is missing")
    names = state.get("names")
    positions = state.get("positions")
    if (
        not isinstance(names, Sequence)
        or isinstance(names, (str, bytes))
        or list(names) != list(ARM_JOINTS)
        or not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or len(positions) != len(ARM_JOINTS)
    ):
        raise ValueError(
            "qualified joint goal must match the configured arm joints"
        )
    try:
        normalized_positions = [float(value) for value in positions]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "qualified joint goal positions must be numeric"
        ) from exc
    if any(not math.isfinite(value) for value in normalized_positions):
        raise ValueError("qualified joint goal positions must be finite")
    for (joint_name, lower, upper), position in zip(
        ARM_JOINT_BOUNDS, normalized_positions, strict=True
    ):
        if (
            position < lower - START_STATE_BOUNDS_TOLERANCE_RAD
            or position > upper + START_STATE_BOUNDS_TOLERANCE_RAD
        ):
            raise ValueError(
                f"qualified joint goal exceeds {joint_name} bounds"
            )
    normalized_state = {
        "names": list(ARM_JOINTS),
        "positions": normalized_positions,
    }
    expected_hash = hashlib.sha256(
        json.dumps(
            normalized_state, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    supplied_hash = str(target_pose.get(hash_field) or "")
    if supplied_hash != expected_hash:
        raise ValueError("qualified joint goal hash does not match its state")
    binding = str(target_pose.get(binding_field) or "")
    if len(binding) != 64 or any(
        char not in "0123456789abcdef" for char in binding
    ):
        raise ValueError("qualification binding hash is invalid")
    result = {
        state_field: normalized_state,
        hash_field: supplied_hash,
        binding_field: binding,
    }
    raw_allowed = target_pose.get(allowed_field)
    supplied_allowed_hash = target_pose.get(allowed_hash_field)
    if raw_allowed is None and supplied_allowed_hash is None:
        return result
    if target_pose.get("grasp_stage") != "contact" or not str(
        target_pose.get("compiled_grasp_id") or ""
    ):
        raise ValueError(
            "qualified collision policy is only valid for a compiled grasp contact"
        )
    result.update(_validated_allowed_collision_policy(target_pose))
    return result


def _validated_recovery_collision_policy(
    target_pose: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate the failed contact's target-only touch policy for one restore."""

    allowed_field = "qualification_allowed_collisions"
    allowed_hash_field = "qualification_allowed_collisions_sha256"
    if not any(
        field in target_pose for field in (allowed_field, allowed_hash_field)
    ):
        return None
    if not (
        target_pose.get("purpose") == "grasp_recovery_restore"
        and target_pose.get("grasp_stage") == "recovery_restore"
        and str(target_pose.get("compiled_grasp_id") or "")
        and str(target_pose.get("recovery_id") or "").startswith(
            "grasp-recovery-"
        )
    ):
        raise ValueError(
            "qualified collision policy is only valid for a compiled grasp "
            "contact or its exact recovery restore"
        )
    return _validated_allowed_collision_policy(target_pose)


def _validated_allowed_collision_policy(
    target_pose: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_field = "qualification_allowed_collisions"
    allowed_hash_field = "qualification_allowed_collisions_sha256"
    raw_allowed = target_pose.get(allowed_field)
    if not isinstance(raw_allowed, Mapping) or len(raw_allowed) != 1:
        raise ValueError(
            "qualified collision policy must name exactly one target object"
        )
    normalized_allowed: dict[str, list[str]] = {}
    for raw_object_id, raw_links in raw_allowed.items():
        object_id = str(raw_object_id).strip()
        if (
            not object_id
            or not isinstance(raw_links, Sequence)
            or isinstance(raw_links, (str, bytes))
        ):
            raise ValueError("qualified collision policy is malformed")
        links = sorted({str(link).strip() for link in raw_links if str(link).strip()})
        if not links:
            raise ValueError("qualified collision policy is malformed")
        normalized_allowed[object_id] = links
    expected_hash = hashlib.sha256(
        json.dumps(
            normalized_allowed, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if str(target_pose.get(allowed_hash_field) or "") != expected_hash:
        raise ValueError(
            "qualified collision policy hash does not match its policy"
        )
    return {
        allowed_field: normalized_allowed,
        allowed_hash_field: expected_hash,
    }


def _q_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    n = math.sqrt(sum(v * v for v in q))
    if n <= 1e-12:
        raise ValueError("quaternion must be non-zero")
    return tuple(v / n for v in q)  # type: ignore[return-value]


def _q_multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _pose_goal_errors(
    actual: Mapping[str, Sequence[float]],
    target: Mapping[str, Sequence[float]],
) -> tuple[float, float]:
    """Return translation and sign-invariant quaternion angular error."""

    actual_xyz = tuple(float(value) for value in actual["xyz"])
    target_xyz = tuple(float(value) for value in target["xyz"])
    position_error_m = math.dist(actual_xyz, target_xyz)
    actual_quat = _q_normalize(tuple(float(value) for value in actual["quat_xyzw"]))
    target_quat = _q_normalize(tuple(float(value) for value in target["quat_xyzw"]))
    dot = min(1.0, abs(sum(a * b for a, b in zip(actual_quat, target_quat))))
    orientation_error_rad = 2.0 * math.acos(dot)
    return position_error_m, orientation_error_rad


def _q_rotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


@dataclass(slots=True)
class GazeboControlResult:
    ok: bool
    error_code: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            **({"error_code": self.error_code} if self.error_code else {}),
            **self.payload,
        }


def motion_request_fingerprint(
    *, start: RobotState, goal: Mapping[str, Any], scene_revision: int
) -> str:
    """Hash exactly the state-dependent MoveIt request identity used for recovery."""

    payload = {
        "joint_positions": [round(float(value), 12) for value in start.joint_positions],
        "target_pose": goal.get("requested_tool_pose"),
        "position_tolerance_m": goal.get("position_tolerance_m"),
        "orientation_tolerance_rad": goal.get("orientation_tolerance_rad"),
        "scene_revision": int(scene_revision),
        "qualification_goal_joint_state_sha256": goal.get(
            "qualification_goal_joint_state_sha256"
        ),
        "qualification_binding_sha256": goal.get(
            "qualification_binding_sha256"
        ),
        "qualification_allowed_collisions_sha256": goal.get(
            "qualification_allowed_collisions_sha256"
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class GazeboController:
    """Action facade. ``move_action``/``gripper_action`` return action results."""

    def __init__(
        self,
        *,
        state_provider: Callable[[], RobotState],
        move_action: Callable[[dict, float], Mapping[str, Any]] | None = None,
        gripper_action: Callable[[float, float], Mapping[str, Any]] | None = None,
        start_state_recovery: Callable[[RobotState, float], Mapping[str, Any]] | None = None,
        cancel_pending: Callable[[], None] | None = None,
        close_source: Callable[[], None] | None = None,
        scene_revision_provider: Callable[[], int] | None = None,
        motion_scene_ready: Callable[[], bool] | None = None,
        candidate_qualifier: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        config: GazeboControlConfig | None = None,
    ):
        self.config = config or GazeboControlConfig()
        self.state_provider = state_provider
        self.move_action, self.gripper_action = move_action, gripper_action
        self.start_state_recovery = start_state_recovery
        self.cancel_pending, self.close_source = cancel_pending, close_source
        self.scene_revision_provider = scene_revision_provider or (lambda: 0)
        self.motion_scene_ready = motion_scene_ready or (lambda: True)
        self.candidate_qualifier = candidate_qualifier
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        if self.cancel_pending is not None:
            self.cancel_pending()
        if self.close_source is not None:
            self.close_source()
        self._closed = True

    @staticmethod
    def _gripper_joint_position_map(
        state: RobotState,
        *,
        joint_names: Sequence[str],
    ) -> dict[str, float]:
        positions = list(state.joint_positions)
        names = [str(name) for name in joint_names]
        if len(names) != len(positions) or any(
            name not in names for name in GRIPPER_JOINTS
        ):
            raise RuntimeError("ROBOT_STATE_UNAVAILABLE")
        result = {
            name: float(positions[index])
            for index, name in enumerate(names)
            if name in GRIPPER_JOINTS
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise RuntimeError("ROBOT_STATE_UNAVAILABLE")
        return result

    def establish_attached_transport_hold(
        self,
        *,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        """Relieve fingertip preload only after native fixed-joint attach.

        Gazebo's detachable joint retains the object during this short common-
        actuator opening.  Object-vs-environment collision remains enabled;
        the method removes only the redundant pad squeeze which would
        otherwise over-constrain DART during arm transport.
        """

        if self.gripper_action is None:
            raise RuntimeError("GRIPPER_UNAVAILABLE")
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("attached transport hold timeout must be positive")
        joint_names = tuple(getattr(self.config, "joint_names", JOINT_NAMES))
        before_state = self.state_provider()
        before_positions = self._gripper_joint_position_map(
            before_state,
            joint_names=joint_names,
        )
        measured_before = common_driver_position(
            before_positions,
            closing=False,
        )
        target = attached_transport_relief_position(
            measured_common_active_rad=measured_before,
            minimum_active_rad=float(self.config.gripper_position(1)),
            terminal_tolerance_rad=GRIPPER_GOAL_TOLERANCE_RAD,
        )
        try:
            result = dict(self.gripper_action(target, timeout))
        except TimeoutError as exc:
            raise RuntimeError("ATTACHED_TRANSPORT_HOLD_FAILED") from exc
        if (
            type(result.get("ok")) is not bool
            or result["ok"] is not True
            or bool(result.get("reached_goal", False)) is not True
            or bool(result.get("stalled", False))
        ):
            raise RuntimeError("ATTACHED_TRANSPORT_HOLD_FAILED")

        after_state = self.state_provider()
        after_positions = self._gripper_joint_position_map(
            after_state,
            joint_names=joint_names,
        )
        measured_after = common_driver_position(
            after_positions,
            closing=False,
        )
        maximum_terminal_position = target + GRIPPER_GOAL_TOLERANCE_RAD
        if measured_after > maximum_terminal_position + 1e-9:
            raise RuntimeError("ATTACHED_TRANSPORT_HOLD_FAILED")
        return {
            "schema_version": "openeta.attached_transport_hold.v1",
            "actuator_model": "single_common_driver",
            "object_environment_collision_enabled": True,
            "measured_common_before_rad": measured_before,
            "commanded_common_target_rad": target,
            "measured_common_after_rad": measured_after,
            "commanded_relief_rad": measured_before - target,
            "minimum_proven_relief_rad": (
                measured_before - maximum_terminal_position
            ),
            "reached_goal": True,
            "stalled": False,
            **{
                key: result[key]
                for key in (
                    "action_started_ros_time_s",
                    "action_completed_ros_time_s",
                    "terminal_status",
                    "terminal_status_code",
                    "wall_elapsed_ms",
                )
                if key in result
            },
        }

    def execute(self, action: Mapping[str, Any]) -> GazeboControlResult:
        kind = action.get("action_type")
        try:
            if kind == "qualify_motion_candidates":
                if self.candidate_qualifier is None:
                    return GazeboControlResult(False, "MOVE_GROUP_UNAVAILABLE")
                try:
                    result = dict(self.candidate_qualifier(action))
                except Exception as exc:  # noqa: BLE001 - private qualification boundary.
                    # This RPC is host-only.  Preserve candidate identities so
                    # the caller can validate the evidence, but never collapse
                    # an engine/service defect into INVALID_CONTROL_ACTION (or
                    # worse, an unreachable candidate).
                    binding = str(action.get("qualification_binding_sha256") or "")
                    candidates = action.get("candidates")
                    candidates = candidates if isinstance(candidates, list) else []
                    funnel = action.get("funnel")
                    funnel = funnel if isinstance(funnel, Mapping) else {}
                    detail = f"{type(exc).__name__}: {exc}"[:500]
                    result = {
                        "schema_version": action.get("schema_version"),
                        "planning_scene_revision": action.get(
                            "planning_scene_revision"
                        ),
                        "execution_started": False,
                        "qualification_profile": str(
                            funnel.get("qualification_profile") or "legacy"
                        ),
                        "stop_reason": "infrastructure_error",
                        "infrastructure_error": True,
                        "results": [
                            {
                                "candidate_id": str(
                                    candidate.get("candidate_id") or ""
                                ),
                                "candidate_pose_sha256": str(
                                    candidate.get("candidate_pose_sha256") or ""
                                ),
                                "qualification_binding_sha256": binding,
                                "execution_started": False,
                                "verdict": "UNKNOWN",
                                "reason": "qualification_infrastructure_error",
                                "infrastructure_error": True,
                                "infrastructure_error_detail": detail,
                                "stages": [],
                            }
                            for candidate in candidates
                            if isinstance(candidate, Mapping)
                        ],
                    }
                return GazeboControlResult(True, payload=result)
            if kind == "move_to":
                if self.move_action is None:
                    return GazeboControlResult(False, "MOVE_GROUP_UNAVAILABLE")
                if not self.motion_scene_ready():
                    return GazeboControlResult(
                        False,
                        "PLANNING_SCENE_UNAVAILABLE",
                        {"motion_outcome": "failed", "execution_started": False},
                    )
                start = self.state_provider()
                goal = make_move_group_goal(
                    action["target_pose"], config=self.config, tolerances=action
                )
                scene_revision = int(self.scene_revision_provider())
                goal["model_id"] = self.config.model_id
                goal["planning_scene_revision"] = scene_revision
                goal["live_start_joint_state"] = {
                    "names": list(ARM_JOINTS),
                    "positions": [
                        float(value)
                        for value in start.joint_positions[: len(ARM_JOINTS)]
                    ],
                }
                request_fingerprint = motion_request_fingerprint(
                    start=start,
                    goal=goal,
                    scene_revision=scene_revision,
                )
                timeout_s = float(action.get("timeout_s", 30.0))
                recovery_result: dict[str, Any] | None = None
                recovery_evidence: dict[str, Any] | None = None
                if self.start_state_recovery is not None:
                    try:
                        recovery_result = dict(
                            self.start_state_recovery(
                                start, START_STATE_RECOVERY_TIMEOUT_S
                            )
                        )
                    except TimeoutError:
                        recovery_result = {
                            "ok": False,
                            "error_code": "MOTION_OUTCOME_UNKNOWN",
                            "motion_outcome": "unknown",
                            "reconciliation_required": True,
                        }
                    evidence = recovery_result.get("start_state_recovery")
                    if not isinstance(evidence, Mapping):
                        assessment = assess_start_state_bounds(start)
                        recovery_evidence = start_state_recovery_record(
                            assessment,
                            status="UNKNOWN",
                            reason_code="RECOVERY_RESULT_INVALID",
                            attempted=bool(recovery_result.get("attempted", False)),
                        )
                        recovery_result = {
                            "ok": False,
                            "error_code": "MOTION_OUTCOME_UNKNOWN",
                            "motion_outcome": "unknown",
                            "reconciliation_required": True,
                            "start_state_recovery": recovery_evidence,
                        }
                    else:
                        recovery_evidence = dict(evidence)
                    if type(recovery_result.get("ok")) is not bool:
                        recovery_result.update(
                            {
                                "ok": False,
                                "error_code": "MOTION_OUTCOME_UNKNOWN",
                                "motion_outcome": "unknown",
                                "reconciliation_required": True,
                            }
                        )
                    if not recovery_result["ok"]:
                        unknown = (
                            recovery_result.get("error_code")
                            == "MOTION_OUTCOME_UNKNOWN"
                        )
                        end = start
                        if not unknown:
                            try:
                                end = self.state_provider()
                            except Exception:
                                pass
                        recovery_timing = {
                            key: recovery_result[key]
                            for key in (
                                "action_started_ros_time_s",
                                "action_completed_ros_time_s",
                            )
                            if key in recovery_result
                        }
                        return GazeboControlResult(
                            False,
                            str(
                                recovery_result.get("error_code")
                                or "START_STATE_RECOVERY_FAILED"
                            ),
                            {
                                "target": goal["requested_tool_pose"],
                                "start": start.end_effector_pose,
                                "end": end.end_effector_pose,
                                "start_state": start.to_dict(),
                                "end_state": end.to_dict(),
                                "steps_executed": (
                                    1
                                    if recovery_evidence
                                    and recovery_evidence.get("attempted")
                                    else 0
                                ),
                                "reached_target": False,
                                "terminated": False,
                                "truncated": False,
                                "motion_outcome": recovery_result.get(
                                    "motion_outcome", "unknown" if unknown else "failed"
                                ),
                                "observation": {"robot": end.to_dict()},
                                "start_state_recovery": recovery_evidence,
                                **recovery_timing,
                                **(
                                    {"reconciliation_required": True}
                                    if unknown
                                    else {}
                                ),
                            },
                        )
                if (
                    isinstance(recovery_evidence, Mapping)
                    and recovery_evidence.get("attempted") is True
                ):
                    # Recovery changed the arm after L5 observed its start.
                    # Withhold the live-start proof and use normal replanning.
                    goal.pop("live_start_joint_state", None)
                try:
                    result = dict(self.move_action(goal, timeout_s))
                except TimeoutError:
                    recovery_timing = {
                        key: recovery_result[key]
                        for key in ("action_started_ros_time_s",)
                        if recovery_result is not None and key in recovery_result
                    }
                    return GazeboControlResult(
                        False,
                        "MOTION_OUTCOME_UNKNOWN",
                        {
                            "target": goal["requested_tool_pose"],
                            "start": start.end_effector_pose,
                            "end": start.end_effector_pose,
                            "start_state": start.to_dict(),
                            "end_state": start.to_dict(),
                            "steps_executed": 1,
                            "reached_target": False,
                            "terminated": False,
                            "truncated": False,
                            "motion_outcome": "unknown",
                            "reconciliation_required": True,
                            "observation": {"robot": start.to_dict()},
                            **(
                                {"start_state_recovery": recovery_evidence}
                                if recovery_evidence is not None
                                else {}
                            ),
                            **recovery_timing,
                        },
                    )
                try:
                    end = self.state_provider()
                except Exception:
                    action_timing = {
                        key: result[key]
                        for key in (
                            "action_started_ros_time_s",
                            "action_completed_ros_time_s",
                        )
                        if key in result
                    }
                    if (
                        recovery_result is not None
                        and "action_started_ros_time_s" in recovery_result
                    ):
                        action_timing["action_started_ros_time_s"] = recovery_result[
                            "action_started_ros_time_s"
                        ]
                    return GazeboControlResult(
                        False,
                        "MOTION_OUTCOME_UNKNOWN",
                        {
                            "target": goal["requested_tool_pose"],
                            "start": start.end_effector_pose,
                            "end": start.end_effector_pose,
                            "start_state": start.to_dict(),
                            "end_state": start.to_dict(),
                            "steps_executed": 1,
                            "reached_target": False,
                            "terminated": False,
                            "truncated": False,
                            "motion_outcome": "unknown",
                            "reconciliation_required": True,
                            "observation": {"robot": start.to_dict()},
                            **(
                                {"start_state_recovery": recovery_evidence}
                                if recovery_evidence is not None
                                else {}
                            ),
                            **action_timing,
                        },
                    )
                if type(result.get("ok")) is not bool:
                    return GazeboControlResult(
                        False,
                        "MOTION_OUTCOME_UNKNOWN",
                        {
                            "target": goal["requested_tool_pose"],
                            "start": start.end_effector_pose,
                            "end": end.end_effector_pose,
                            "start_state": start.to_dict(),
                            "end_state": end.to_dict(),
                            "steps_executed": 1,
                            "reached_target": False,
                            "terminated": False,
                            "truncated": False,
                            "motion_outcome": "unknown",
                            "reconciliation_required": True,
                            "observation": {"robot": end.to_dict()},
                            **(
                                {"start_state_recovery": recovery_evidence}
                                if recovery_evidence is not None
                                else {}
                            ),
                        },
                    )
                ok = result["ok"]
                error = result.get("error_code") or (None if ok else "MOTION_EXECUTION_FAILED")
                extra = (
                    {"reconciliation_required": True}
                    if result.get("error_code") == "MOTION_OUTCOME_UNKNOWN"
                    else {}
                )
                action_evidence = {
                    key: result[key]
                    for key in (
                        "action_started_ros_time_s",
                        "action_completed_ros_time_s",
                        "moveit_error_code",
                        "planned_point_count",
                        "execution_started",
                        "motion_profile",
                        "max_velocity_scaling_factor",
                        "max_acceleration_scaling_factor",
                        "l5_trajectory_reused",
                        "l5_trajectory_cache_key",
                        "l5_trajectory_scene_sha256",
                        "l5_trajectory_cache_status",
                        "l5_trajectory_cache_reason",
                        "l5_trajectory_cache_entry_count",
                        "l5_trajectory_cache_requested_scene_revision",
                        "l5_trajectory_cache_current_scene_revision",
                        "l5_trajectory_cache_requested_start_max_delta_rad",
                        "l5_trajectory_cache_measured_start_max_delta_rad",
                    )
                    if key in result
                }
                recovery_started = (
                    recovery_result.get("action_started_ros_time_s")
                    if recovery_result is not None
                    else None
                )
                if recovery_started is not None:
                    action_evidence["action_started_ros_time_s"] = recovery_started
                action_evidence["scene_revision"] = scene_revision
                action_evidence["request_fingerprint"] = request_fingerprint
                # The post-action TF is sampled after MoveIt's result boundary,
                # so allow a small measurement/settling margin while still
                # rejecting a trajectory result that stopped materially away
                # from the requested tool pose.
                position_verification_tolerance_m = max(
                    0.0005, 2.0 * float(goal["position_tolerance_m"])
                )
                effective_position_verification_tolerance_m = (
                    position_verification_tolerance_m
                    + MOTION_VERIFICATION_NUMERIC_MARGIN_M
                )
                orientation_verification_tolerance_rad = max(
                    0.005, 2.0 * float(goal["orientation_tolerance_rad"])
                )
                target_xyz = tuple(
                    float(value) for value in goal["requested_tool_pose"]["xyz"]
                )

                def verification_metrics(
                    state: RobotState,
                ) -> tuple[float, float, float, float, bool]:
                    sample_position_error_m, sample_orientation_error_rad = (
                        _pose_goal_errors(
                            state.end_effector_pose,
                            goal["requested_tool_pose"],
                        )
                    )
                    sample_xyz = tuple(
                        float(value) for value in state.end_effector_pose["xyz"]
                    )
                    sample_horizontal_error_m = math.dist(
                        sample_xyz[:2], target_xyz[:2]
                    )
                    sample_vertical_error_m = sample_xyz[2] - target_xyz[2]
                    sample_verified = (
                        sample_position_error_m
                        <= effective_position_verification_tolerance_m
                        and sample_orientation_error_rad
                        <= orientation_verification_tolerance_rad
                    )
                    return (
                        sample_position_error_m,
                        sample_orientation_error_rad,
                        sample_horizontal_error_m,
                        sample_vertical_error_m,
                        sample_verified,
                    )

                (
                    position_error_m,
                    orientation_error_rad,
                    horizontal_error_m,
                    vertical_error_m,
                    target_verified,
                ) = verification_metrics(end)
                position_verification_policy = (
                    "exact_terminal_euclidean_with_bounded_numeric_margin"
                )
                terminal_reconciliation: dict[str, Any] | None = None
                joint_names = end.metadata.get("joint_names")
                joint_index = (
                    {str(name): index for index, name in enumerate(joint_names)}
                    if isinstance(joint_names, (list, tuple))
                    else {}
                )
                arm_velocities = [
                    float(end.joint_velocities[joint_index[name]])
                    for name in ARM_JOINTS
                    if name in joint_index
                    and joint_index[name] < len(end.joint_velocities)
                    and math.isfinite(
                        float(end.joint_velocities[joint_index[name]])
                    )
                ]
                max_arm_velocity_rad_s = (
                    max(abs(value) for value in arm_velocities)
                    if len(arm_velocities) == len(ARM_JOINTS)
                    else None
                )
                control_failed_at_verified_terminal = (
                    not ok
                    and result.get("execution_started") is True
                    and int(result.get("planned_point_count") or 0) > 0
                    and result.get("moveit_error_code") == MOVEIT_CONTROL_FAILED
                    and target_verified
                    and max_arm_velocity_rad_s is not None
                    and max_arm_velocity_rad_s
                    <= MOTION_TERMINAL_MAX_ARM_VELOCITY_RAD_S
                )
                if control_failed_at_verified_terminal:
                    original_error = str(error or "MOTION_EXECUTION_FAILED")
                    ok = True
                    error = None
                    terminal_reconciliation = {
                        "schema_version": (
                            "openeta.gazebo.motion_terminal_reconciliation.v1"
                        ),
                        "status": "PASS",
                        "reason_code": (
                            "CONTROL_FAILED_AFTER_EXACT_TARGET_REACHED"
                        ),
                        "proof_boundary": (
                            "fresh_terminal_tf_and_stationary_arm_joint_state"
                        ),
                        "original_error_code": original_error,
                        "moveit_error_code": MOVEIT_CONTROL_FAILED,
                        "execution_started": True,
                        "planned_point_count": int(
                            result.get("planned_point_count") or 0
                        ),
                        "target_verified": True,
                        "max_arm_velocity_rad_s": max_arm_velocity_rad_s,
                        "max_arm_velocity_tolerance_rad_s": (
                            MOTION_TERMINAL_MAX_ARM_VELOCITY_RAD_S
                        ),
                        "position_error_m": position_error_m,
                        "orientation_error_rad": orientation_error_rad,
                    }
                settling_recheck: dict[str, Any] | None = None
                if (
                    ok
                    and not bool(result.get("plan_only", False))
                    and not target_verified
                ):
                    initial_position_error_m = position_error_m
                    initial_orientation_error_rad = orientation_error_rad
                    best_position_error_m = position_error_m
                    best_orientation_error_rad = orientation_error_rad
                    sample_count = 0
                    recheck_started = time.monotonic()
                    recheck_status = "timeout"
                    recheck_error_type: str | None = None
                    while True:
                        remaining_s = (
                            MOTION_SETTLE_RECHECK_TIMEOUT_S
                            - (time.monotonic() - recheck_started)
                        )
                        if remaining_s <= 0.0:
                            break
                        time.sleep(
                            min(MOTION_SETTLE_RECHECK_INTERVAL_S, remaining_s)
                        )
                        try:
                            sample = self.state_provider()
                        except Exception as exc:  # noqa: BLE001 - sensor boundary.
                            recheck_status = "state_unavailable"
                            recheck_error_type = type(exc).__name__
                            break
                        sample_count += 1
                        end = sample
                        (
                            position_error_m,
                            orientation_error_rad,
                            horizontal_error_m,
                            vertical_error_m,
                            target_verified,
                        ) = verification_metrics(sample)
                        best_position_error_m = min(
                            best_position_error_m, position_error_m
                        )
                        best_orientation_error_rad = min(
                            best_orientation_error_rad, orientation_error_rad
                        )
                        if target_verified:
                            recheck_status = "target_verified"
                            break
                    settling_recheck = {
                        "attempted": True,
                        "status": recheck_status,
                        "sample_count": sample_count,
                        "timeout_s": MOTION_SETTLE_RECHECK_TIMEOUT_S,
                        "interval_s": MOTION_SETTLE_RECHECK_INTERVAL_S,
                        "elapsed_s": time.monotonic() - recheck_started,
                        "initial_position_error_m": initial_position_error_m,
                        "initial_orientation_error_rad": (
                            initial_orientation_error_rad
                        ),
                        "best_position_error_m": best_position_error_m,
                        "best_orientation_error_rad": best_orientation_error_rad,
                    }
                    if recheck_error_type is not None:
                        settling_recheck["error_type"] = recheck_error_type
                if ok and not bool(result.get("plan_only", False)) and not target_verified:
                    ok = False
                    error = "MOTION_TARGET_NOT_REACHED"
                    extra = {}
                return GazeboControlResult(
                    ok,
                    error,
                    {
                        "target": goal["requested_tool_pose"],
                        "start": start.end_effector_pose,
                        "end": end.end_effector_pose,
                        "start_state": start.to_dict(),
                        "end_state": end.to_dict(),
                        "steps_executed": 1,
                        "reached_target": bool(
                            result.get("reached_goal", ok) and target_verified
                        ),
                        "stalled": False,
                        "terminated": False,
                        "truncated": False,
                        "motion_outcome": (
                            "completed"
                            if terminal_reconciliation is not None
                            else (
                                "failed"
                                if error == "MOTION_TARGET_NOT_REACHED"
                                else result.get(
                                    "motion_outcome",
                                    "completed" if ok else "failed",
                                )
                            )
                        ),
                        "position_error_m": position_error_m,
                        "horizontal_error_m": horizontal_error_m,
                        "vertical_error_m": vertical_error_m,
                        "position_verification_policy": (
                            position_verification_policy
                        ),
                        "orientation_error_rad": orientation_error_rad,
                        "position_verification_tolerance_m": (
                            position_verification_tolerance_m
                        ),
                        "position_verification_numeric_margin_m": (
                            MOTION_VERIFICATION_NUMERIC_MARGIN_M
                        ),
                        "position_verification_effective_tolerance_m": (
                            effective_position_verification_tolerance_m
                        ),
                        "orientation_verification_tolerance_rad": (
                            orientation_verification_tolerance_rad
                        ),
                        **(
                            {"settling_recheck": settling_recheck}
                            if settling_recheck is not None
                            else {}
                        ),
                        **(
                            {"terminal_reconciliation": terminal_reconciliation}
                            if terminal_reconciliation is not None
                            else {}
                        ),
                        "observation": {"robot": end.to_dict()},
                        **(
                            {"start_state_recovery": recovery_evidence}
                            if recovery_evidence is not None
                            else {}
                        ),
                        **action_evidence,
                        **extra,
                    },
                )
            if kind in ("gripper_open", "gripper_close"):
                if self.gripper_action is None:
                    return GazeboControlResult(False, "GRIPPER_UNAVAILABLE")
                p = self.config.gripper_position(1 if kind == "gripper_open" else 0)
                try:
                    # The RGB-D Harmonic stack can run below real time on a
                    # software-rendered host.  A full 2F-85 stroke then takes
                    # longer than the generic 30 s motion timeout even though
                    # the simulated joint is still making progress.
                    result = dict(self.gripper_action(p, float(action.get("timeout_s", 90.0))))
                except TimeoutError:
                    return GazeboControlResult(False, "GRIPPER_TIMEOUT")
                state = self.state_provider()
                if type(result.get("ok")) is not bool:
                    return GazeboControlResult(
                        False,
                        "MOTION_OUTCOME_UNKNOWN",
                        {"motion_outcome": "unknown", "reconciliation_required": True},
                    )
                reached_goal = bool(result.get("reached_goal", result["ok"]))
                stalled = bool(result.get("stalled", False))
                # The native-grasp profile explicitly allows a successful stalled close
                # as an input to its independent native-contact gate.  motion-control's
                # default profile never credits a stalled or unreached action,
                # even if a lower adapter incorrectly labels it ``ok``.
                if (
                    kind == "gripper_close"
                    and bool(getattr(self.config, "allow_stalling", False))
                ):
                    # FollowJointTrajectory reports a load-stalled Robotiq close
                    # as ABORTED even though the terminal state and stalled bit
                    # are known.  In the native-grasp profile this admits only
                    # the subsequent independent Gazebo contact gate; it never
                    # constitutes grasp or attachment evidence by itself.
                    ok = result["ok"] or stalled
                else:
                    ok = result["ok"] and reached_goal and not stalled
                state.gripper_state.update(
                    {
                        "reached_goal": reached_goal,
                        "stalled": stalled,
                    }
                )
                action_timing = {
                    key: result[key]
                    for key in (
                        "action_started_ros_time_s",
                        "action_completed_ros_time_s",
                    )
                    if key in result
                }
                return GazeboControlResult(
                    ok,
                    None if ok else (result.get("error_code") or "GRIPPER_FAILED"),
                    {
                        "gripper_state": state.gripper_state,
                        "reached_goal": reached_goal,
                        "stalled": stalled,
                        "observation": {"robot": state.to_dict()},
                        **action_timing,
                        **{
                            key: result[key]
                            for key in (
                                "terminal_status",
                                "terminal_status_code",
                                "terminal_gripper_joint_state",
                                "stall_accepted_for_command",
                                "wall_elapsed_ms",
                            )
                            if key in result
                        },
                    },
                )
            return GazeboControlResult(False, "INVALID_CONTROL_ACTION")
        except KeyError:
            return GazeboControlResult(False, "INVALID_CONTROL_ACTION")
        except (TypeError, ValueError):
            return GazeboControlResult(False, "INVALID_CONTROL_ACTION")
        except RuntimeError as exc:
            code = str(exc)
            return GazeboControlResult(
                False, code if code in ERROR_CODES else "ROBOT_STATE_UNAVAILABLE"
            )
