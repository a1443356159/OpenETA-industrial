"""M2 RM75 + parallel gripper control contracts.

The module deliberately contains no ROS imports.  ROS action clients and TF/
joint-state subscriptions are injected by the deployment adapter, which keeps
the worker contract testable on machines without Jazzy installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from adapter.protocol import RobotState

M2_ENV_ID = "openeta/gazebo_rm75_parallel-v0"
MODEL_ID = "rm75_parallel_gripper_sim_v1"
ROBOTIQ2F85_ENV_ID = "openeta/gazebo_rm75_robotiq2f85-v0"
ROBOTIQ2F85_MODEL_ID = "rm75_robotiq_2f85_sim_v1"
ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("gripper_left_finger_joint", "gripper_right_finger_joint")
JOINT_NAMES = ARM_JOINTS + GRIPPER_JOINTS

ERROR_CODES = frozenset(
    {
        "MODEL_ASSET_NOT_FOUND",
        "ROS_NOT_READY",
        "JOINT_STATE_TIMEOUT",
        "TF_TIMEOUT",
        "MOVE_GROUP_UNAVAILABLE",
        "MOTION_PLAN_FAILED",
        "MOTION_EXECUTION_FAILED",
        "MOTION_EXECUTION_TIMEOUT",
        "MOTION_OUTCOME_UNKNOWN",
        "GRIPPER_UNAVAILABLE",
        "GRIPPER_FAILED",
        "GRIPPER_TIMEOUT",
        "INVALID_CONTROL_ACTION",
        "ROBOT_STATE_UNAVAILABLE",
    }
)


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
    active_open_position_m: float = 0.035
    maximum_aperture_m: float = 0.070
    mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.025)
    mount_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    asset_root_override: str | None = None

    def __post_init__(self) -> None:
        if self.maximum_aperture_m != 2 * self.active_open_position_m:
            raise ValueError("maximum aperture must equal twice active opening")
        if self.active_joint == self.mimic_joint or self.mount_parent != self.arm_tip:
            raise ValueError("invalid fixed mount or gripper joint mapping")
        if len(self.mount_xyz) != 3 or len(self.mount_quat_xyzw) != 4:
            raise ValueError("mount transform has invalid dimensions")

    def gripper_position(self, position: int) -> float:
        if type(position) is not int or position not in (0, 1):
            raise ValueError("gripper position must be exactly 0 (closed) or 1 (open)")
        return self.closed_position_m if position == 0 else self.active_open_position_m

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

    def validate_assets(self, *, require_vendor: bool = True) -> None:
        del require_vendor  # Compatibility with the early M2 contract.
        try:
            from .asset_preflight import validate_asset_root

            manifest = validate_asset_root(self.asset_root)
        except Exception as exc:
            raise RuntimeError("MODEL_ASSET_NOT_FOUND") from exc
        if manifest.get("description_id") != "RM75-6FB-V":
            raise RuntimeError("MODEL_ASSET_NOT_FOUND")
        if isinstance(self, Robotiq2F85Config):
            try:
                validate_asset_root(self.gripper_asset_root)
            except Exception as exc:
                raise RuntimeError("MODEL_ASSET_NOT_FOUND") from exc
        package_name = "openeta_rm75_robotiq2f85_sim" if self.model_id == ROBOTIQ2F85_MODEL_ID else "openeta_rm75_parallel_sim"
        package = self.ros_workspace / "src" / package_name
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


@dataclass(frozen=True, slots=True)
class Robotiq2F85Config(M2Config):
    model_id: str = ROBOTIQ2F85_MODEL_ID
    active_joint: str = "gripper_left_finger_joint"
    mimic_joint: str = "gripper_right_finger_joint"
    closed_position_m: float = 0.0
    active_open_position_m: float = 0.0425
    maximum_aperture_m: float = 0.085
    calibration: Robotiq2F85Calibration = field(default_factory=Robotiq2F85Calibration)

    @property
    def gripper_asset_root(self) -> Path:
        return Path(__file__).parent / "assets" / "robotiq_2f85_vendor"

    @property
    def joint_names(self) -> tuple[str, ...]:
        return ARM_JOINTS + (self.active_joint, self.mimic_joint, "gripper_left_inner_knuckle_joint", "gripper_right_inner_knuckle_joint", "gripper_left_finger_tip_joint", "gripper_right_finger_tip_joint")

    def gripper_position(self, position: int) -> float:
        # Public binary command maps to the active knuckle's radians.
        if type(position) is not int or position not in (0, 1):
            raise ValueError("gripper position must be exactly 0 or 1")
        return self.calibration.angles_rad[-1] if position == 0 else self.calibration.angles_rad[0]


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
    if isinstance(cfg, Robotiq2F85Config):
        aperture = cfg.calibration.aperture_from_angle(float(active_position_m))
        p = max(cfg.closed_position_m, min(cfg.active_open_position_m, aperture / 2.0))
    else:
        p = max(cfg.closed_position_m, min(cfg.active_open_position_m, float(active_position_m)))
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
        "position_tolerance_m": float((tolerances or {}).get("position_tolerance_m", 0.002)),
        "orientation_tolerance_rad": float(
            (tolerances or {}).get("orientation_tolerance_rad", 0.05)
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
        cancel_pending: Callable[[], None] | None = None,
        close_source: Callable[[], None] | None = None,
        config: M2Config | None = None,
    ):
        self.config = config or M2Config()
        self.state_provider = state_provider
        self.move_action, self.gripper_action = move_action, gripper_action
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
                try:
                    result = dict(self.move_action(goal, float(action.get("timeout_s", 30.0))))
                except TimeoutError:
                    return M2ControlResult(
                        False,
                        "MOTION_OUTCOME_UNKNOWN",
                        {"motion_outcome": "unknown", "reconciliation_required": True},
                    )
                try:
                    end = self.state_provider()
                except Exception:
                    return M2ControlResult(
                        False,
                        "MOTION_OUTCOME_UNKNOWN",
                        {"motion_outcome": "unknown", "reconciliation_required": True},
                    )
                if type(result.get("ok")) is not bool:
                    return M2ControlResult(
                        False,
                        "MOTION_OUTCOME_UNKNOWN",
                        {"motion_outcome": "unknown", "reconciliation_required": True},
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
                        "reached_target": bool(result.get("reached_goal", ok)),
                        "terminated": False,
                        "truncated": False,
                        "motion_outcome": result.get(
                            "motion_outcome", "completed" if ok else "failed"
                        ),
                        "observation": {"robot": end.to_dict()},
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
                ok = result["ok"]
                state.gripper_state.update(
                    {
                        "reached_goal": bool(result.get("reached_goal", ok)),
                        "stalled": bool(result.get("stalled", False)),
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
                        "reached_goal": bool(result.get("reached_goal", ok)),
                        "stalled": bool(result.get("stalled", False)),
                        "observation": {"robot": state.to_dict()},
                        **action_timing,
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
