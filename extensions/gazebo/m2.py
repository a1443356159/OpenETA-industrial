"""M2 RM75 + Robotiq 2F-85 control contracts.

The module deliberately contains no ROS imports.  ROS action clients and TF/
joint-state subscriptions are injected by the deployment adapter, which keeps
the worker contract testable on machines without Jazzy installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from adapter.protocol import RobotState

M2_ENV_ID = "openeta/gazebo_rm75_robotiq2f85-v0"
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
START_STATE_RECOVERY_SCHEMA_VERSION = "m2_start_state_recovery_v1"
START_STATE_BOUNDS_TOLERANCE_RAD = 1e-6
START_STATE_RECOVERY_INSET_RAD = 1e-3
START_STATE_RECOVERY_TRAJECTORY_S = 1.0
START_STATE_RECOVERY_TIMEOUT_S = 5.0
# These targets are relative to the first fresh mount pose after a reset, not
# absolute world coordinates.  They are the small, validated neutral motions
# available in the empty M2 profile.  Publishing the relation in the existing
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
        "INVALID_CONTROL_ACTION",
        "ROBOT_STATE_UNAVAILABLE",
    }
)


def neutral_relative_motion_guidance() -> dict[str, Any]:
    """Return a fresh, serializable M2 neutral-motion capability.

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
    """Build the stable, JSON-safe M2/M3 recovery evidence envelope."""

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
class M2Config:
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
    mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.025)
    mount_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
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

    def gripper_position(self, position: int) -> float:
        if type(position) is not int or position not in (0, 1):
            raise ValueError("gripper position must be exactly 0 or 1")
        return self.calibration.angles_rad[-1] if position == 0 else self.calibration.angles_rad[0]

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
        del require_vendor  # Compatibility with the early M2 contract.
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
    config: M2Config | None = None,
) -> dict[str, Any]:
    cfg = config or M2Config()
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
    config: M2Config | None = None,
) -> RobotState:
    """Build state only when all required joints and the configured TF exist."""
    cfg = config or M2Config()
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
    target_pose: Mapping[str, Sequence[float]],
    *,
    config: M2Config | None = None,
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    cfg = config or M2Config()
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
    return {
        "group_name": cfg.move_group,
        "base_frame": cfg.base_link,
        "link_name": cfg.arm_tip,
        "requested_tool_pose": {
            "frame_id": cfg.base_link,
            "xyz": list(tool_xyz),
            "quat_xyzw": list(tool_q),
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
        # preserves the long-standing M2/M3 motion contract.
        "max_velocity_scaling_factor": float(
            (tolerances or {}).get("max_velocity_scaling_factor", 0.3)
        ),
        "max_acceleration_scaling_factor": float(
            (tolerances or {}).get("max_acceleration_scaling_factor", 0.3)
        ),
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
class M2ControlResult:
    ok: bool
    error_code: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            **({"error_code": self.error_code} if self.error_code else {}),
            **self.payload,
        }


class M2Controller:
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
        config: M2Config | None = None,
    ):
        self.config = config or M2Config()
        self.state_provider = state_provider
        self.move_action, self.gripper_action = move_action, gripper_action
        self.start_state_recovery = start_state_recovery
        self.cancel_pending, self.close_source = cancel_pending, close_source
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        if self.cancel_pending is not None:
            self.cancel_pending()
        if self.close_source is not None:
            self.close_source()
        self._closed = True

    def execute(self, action: Mapping[str, Any]) -> M2ControlResult:
        kind = action.get("action_type")
        try:
            if kind == "move_to":
                if self.move_action is None:
                    return M2ControlResult(False, "MOVE_GROUP_UNAVAILABLE")
                start = self.state_provider()
                goal = make_move_group_goal(
                    action["target_pose"], config=self.config, tolerances=action
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
                        return M2ControlResult(
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
                try:
                    result = dict(self.move_action(goal, timeout_s))
                except TimeoutError:
                    recovery_timing = {
                        key: recovery_result[key]
                        for key in ("action_started_ros_time_s",)
                        if recovery_result is not None and key in recovery_result
                    }
                    return M2ControlResult(
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
                    return M2ControlResult(
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
                    return M2ControlResult(
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
                position_error_m, orientation_error_rad = _pose_goal_errors(
                    end.end_effector_pose, goal["requested_tool_pose"]
                )
                # The post-action TF is sampled after MoveIt's result boundary,
                # so allow a small measurement/settling margin while still
                # rejecting a trajectory result that stopped materially away
                # from the requested tool pose.
                position_verification_tolerance_m = max(
                    0.0005, 2.0 * float(goal["position_tolerance_m"])
                )
                orientation_verification_tolerance_rad = max(
                    0.005, 2.0 * float(goal["orientation_tolerance_rad"])
                )
                target_verified = (
                    position_error_m <= position_verification_tolerance_m
                    and orientation_error_rad <= orientation_verification_tolerance_rad
                )
                if ok and not bool(result.get("plan_only", False)) and not target_verified:
                    ok = False
                    error = "MOTION_TARGET_NOT_REACHED"
                    extra = {}
                return M2ControlResult(
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
                            "failed"
                            if error == "MOTION_TARGET_NOT_REACHED"
                            else result.get(
                                "motion_outcome", "completed" if ok else "failed"
                            )
                        ),
                        "position_error_m": position_error_m,
                        "orientation_error_rad": orientation_error_rad,
                        "position_verification_tolerance_m": (
                            position_verification_tolerance_m
                        ),
                        "orientation_verification_tolerance_rad": (
                            orientation_verification_tolerance_rad
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
                    return M2ControlResult(False, "GRIPPER_UNAVAILABLE")
                p = self.config.gripper_position(1 if kind == "gripper_open" else 0)
                try:
                    # The RGB-D Harmonic stack can run below real time on a
                    # software-rendered host.  A full 2F-85 stroke then takes
                    # longer than the generic 30 s motion timeout even though
                    # the simulated joint is still making progress.
                    result = dict(self.gripper_action(p, float(action.get("timeout_s", 90.0))))
                except TimeoutError:
                    return M2ControlResult(False, "GRIPPER_TIMEOUT")
                state = self.state_provider()
                if type(result.get("ok")) is not bool:
                    return M2ControlResult(
                        False,
                        "MOTION_OUTCOME_UNKNOWN",
                        {"motion_outcome": "unknown", "reconciliation_required": True},
                    )
                reached_goal = bool(result.get("reached_goal", result["ok"]))
                stalled = bool(result.get("stalled", False))
                # The M3 profile explicitly allows a successful stalled close
                # as an input to its independent native-contact gate.  M2's
                # default profile never credits a stalled or unreached action,
                # even if a lower adapter incorrectly labels it ``ok``.
                if bool(getattr(self.config, "allow_stalling", False)):
                    ok = result["ok"]
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
                return M2ControlResult(
                    ok,
                    result.get("error_code") or (None if ok else "GRIPPER_FAILED"),
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
                                "wall_elapsed_ms",
                            )
                            if key in result
                        },
                    },
                )
            return M2ControlResult(False, "INVALID_CONTROL_ACTION")
        except KeyError:
            return M2ControlResult(False, "INVALID_CONTROL_ACTION")
        except (TypeError, ValueError):
            return M2ControlResult(False, "INVALID_CONTROL_ACTION")
        except RuntimeError as exc:
            code = str(exc)
            return M2ControlResult(
                False, code if code in ERROR_CODES else "ROBOT_STATE_UNAVAILABLE"
            )
