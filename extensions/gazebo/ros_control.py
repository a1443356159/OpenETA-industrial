"""Jazzy MoveIt and ros2_control adapter for motion-control.

All ROS imports occur inside :meth:`RosGazeboControllerFactory.create`; importing
OpenETA on a non-ROS test machine remains supported.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree

from .robot_control import (
    ARM_JOINTS,
    ARM_JOINT_BOUNDS,
    ARM_HOME_JOINT_POSITIONS,
    GazeboControlConfig,
    GazeboController,
    MOVEIT_CONTROL_FAILED,
    START_STATE_RECOVERY_TRAJECTORY_S,
    assess_start_state_bounds,
    make_move_group_goal,
    robot_state_from_sources,
    start_state_recovery_record,
)
from .planning_scene import (
    LEFT_FINGERTIP,
    RIGHT_FINGERTIP,
    TARGET_TOUCH_LINKS,
    CollisionBody,
    CollisionBox,
    CollisionGeometry,
    CollisionPrimitive,
    PlanningSceneError,
    PlanningSceneSynchronizer,
)

QUALIFIED_JOINT_GOAL_TOLERANCE_RAD = 0.001
L5_TRAJECTORY_START_TOLERANCE_RAD = 0.001
L5_TRAJECTORY_CACHE_LIMIT = 64
L5_TRAJECTORY_POSE_DECIMALS = 6


def _normalized_arm_joint_state(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    names = value.get("names", value.get("joint_names"))
    positions = value.get("positions")
    if (
        not isinstance(names, Sequence)
        or isinstance(names, (str, bytes, bytearray))
        or not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes, bytearray))
        or len(names) != len(positions)
    ):
        return None
    parsed: dict[str, float] = {}
    try:
        for name, position in zip(names, positions, strict=True):
            joint = str(name)
            numeric = float(position)
            if joint in parsed or not joint or not math.isfinite(numeric):
                return None
            parsed[joint] = numeric
    except (TypeError, ValueError):
        return None
    return {name: parsed[name] for name in ARM_JOINTS} if set(ARM_JOINTS).issubset(parsed) else None


def _joint_states_within_l5_start_tolerance(
    planned: object,
    live: object,
    *,
    tolerance_rad: float = L5_TRAJECTORY_START_TOLERANCE_RAD,
) -> bool:
    """Require the live arm to remain at the L5 trajectory's proven start."""

    if not math.isfinite(tolerance_rad) or tolerance_rad < 0.0:
        return False
    planned_state = _normalized_arm_joint_state(planned)
    live_state = _normalized_arm_joint_state(live)
    return bool(
        planned_state is not None
        and live_state is not None
        and all(abs(planned_state[name] - live_state[name]) <= tolerance_rad for name in ARM_JOINTS)
    )


def _joint_state_max_abs_delta(planned: object, live: object) -> float | None:
    """Return the largest named arm-joint delta, or ``None`` for bad input."""

    planned_state = _normalized_arm_joint_state(planned)
    live_state = _normalized_arm_joint_state(live)
    if planned_state is None or live_state is None:
        return None
    return max(abs(planned_state[name] - live_state[name]) for name in ARM_JOINTS)


def _qualification_joint_state_with_sha256(
    value: object,
) -> tuple[dict[str, list[Any]], str] | None:
    """Canonicalize an internal L5 joint goal with the public proof digest.

    Private qualification owns the newly solved joint state, so it cannot yet
    carry the public tool boundary's caller-supplied digest. The transient L5
    trajectory cache still has to bind both routes to the exact same state.
    """

    parsed = _normalized_arm_joint_state(value)
    if parsed is None:
        return None
    normalized: dict[str, list[Any]] = {
        "names": list(ARM_JOINTS),
        "positions": [float(parsed[name]) for name in ARM_JOINTS],
    }
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return normalized, digest


def _trajectory_end_joint_state_with_sha256(
    trajectory: object,
) -> tuple[dict[str, list[Any]], str] | None:
    """Return the exact executable endpoint used by an L5 trajectory.

    MoveIt may make a small, valid adjustment between the IK seed supplied to
    plan-only and the final point of the time-parameterized trajectory.  The
    public qualification proof exposes that final point, so the transient
    execution cache must bind to the same point rather than the pre-plan IK
    seed.
    """

    points = getattr(trajectory, "points", None)
    names = getattr(trajectory, "joint_names", None)
    if (
        not isinstance(points, Sequence)
        or isinstance(points, (str, bytes, bytearray))
        or not points
        or not isinstance(names, Sequence)
        or isinstance(names, (str, bytes, bytearray))
    ):
        return None
    positions = getattr(points[-1], "positions", None)
    return _qualification_joint_state_with_sha256({"names": list(names), "positions": positions})


def _canonical_l5_pose(value: object) -> object:
    """Canonicalize sub-micrometre/Euler-roundtrip noise in a proven pose.

    The public tool schema transports orientation through roll/pitch/yaw while
    private L5 qualification uses the provider quaternion.  Both routes name
    the same cryptographically bound candidate, but the round trip can differ
    by roughly 1e-7 and must not turn a safe cache hit into a fresh stochastic
    plan.  Six decimal places remain orders of magnitude tighter than the
    execution tolerances while preserving the proof identity fields below.
    """

    if not isinstance(value, Mapping):
        return value
    canonical = dict(value)
    for field in ("xyz", "quat_xyzw"):
        raw = canonical.get(field)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            continue
        try:
            numeric = [float(item) for item in raw]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(item) for item in numeric):
            continue
        if field == "quat_xyzw" and len(numeric) == 4:
            norm = math.sqrt(sum(item * item for item in numeric))
            if norm > 0.0:
                numeric = [item / norm for item in numeric]
                # q and -q encode the same rotation. Keep one representation.
                first_nonzero = next(
                    (item for item in reversed(numeric) if abs(item) > 1e-15),
                    1.0,
                )
                if first_nonzero < 0.0:
                    numeric = [-item for item in numeric]
        canonical[field] = [round(item, L5_TRAJECTORY_POSE_DECIMALS) for item in numeric]
    return canonical


def _l5_trajectory_cache_key(
    goal: Mapping[str, Any],
    *,
    scene_revision: int,
    scene_sha256: str,
) -> str | None:
    """Bind one transient L5 trajectory to geometry, scene, and load state."""

    binding = str(
        goal.get("qualification_cache_binding_sha256")
        or goal.get("qualification_binding_sha256")
        or ""
    )
    if (
        len(binding) != 64
        or any(character not in "0123456789abcdef" for character in binding)
        or not scene_sha256
        or isinstance(scene_revision, bool)
        or scene_revision < 0
    ):
        return None
    payload = {
        "schema_version": "openeta.l5_trajectory_cache_key.v1",
        "binding": binding,
        "qualification_goal_joint_state_sha256": goal.get("qualification_goal_joint_state_sha256"),
        "compiled_grasp_id": goal.get("compiled_grasp_id"),
        "grasp_stage": goal.get("grasp_stage"),
        "compiled_placement_id": goal.get("compiled_placement_id"),
        "placement_candidate_id": goal.get("placement_candidate_id"),
        "placement_stage": goal.get("placement_stage"),
        "model_id": str(goal.get("model_id") or ""),
        "scene_revision": int(scene_revision),
        "scene_sha256": scene_sha256,
        "group_name": goal.get("group_name"),
        "link_name": goal.get("link_name"),
        "requested_tool_pose": _canonical_l5_pose(goal.get("requested_tool_pose")),
        "target_pose": _canonical_l5_pose(goal.get("target_pose")),
        "position_tolerance_m": goal.get("position_tolerance_m"),
        "orientation_tolerance_rad": goal.get("orientation_tolerance_rad"),
        "motion_profile": goal.get("motion_profile"),
        "max_velocity_scaling_factor": goal.get("max_velocity_scaling_factor"),
        "max_acceleration_scaling_factor": goal.get("max_acceleration_scaling_factor"),
        "qualification_allowed_collisions": goal.get("qualification_allowed_collisions"),
        # A virtual attach/detach changes this field and therefore cannot be
        # confused with the physically current PlanningScene.
        "qualification_scene_diff": goal.get("qualification_scene_diff"),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _move_group_failure_result(
    moveit_error_code: int,
    planned_point_count: int,
    *,
    plan_only: bool,
    planning_failure_codes: set[int],
    timed_out_code: int,
    generic_failure_code: int,
) -> dict[str, Any]:
    """Classify a non-success MoveGroup result without inventing execution.

    MoveIt can return a populated candidate trajectory together with generic
    ``FAILURE`` when its response adapter rejects that trajectory.  In a
    plan-only request those points were never sent to a controller, so they
    are planning evidence only and ``execution_started`` must remain false.
    """

    if (
        plan_only
        or moveit_error_code in planning_failure_codes
        or (moveit_error_code == generic_failure_code and planned_point_count == 0)
    ):
        error_code = "MOTION_PLAN_FAILED"
        execution_started = False
    elif moveit_error_code == timed_out_code:
        error_code = "MOTION_EXECUTION_TIMEOUT"
        execution_started = planned_point_count > 0
    else:
        error_code = "MOTION_EXECUTION_FAILED"
        execution_started = planned_point_count > 0
    return {
        "ok": False,
        "error_code": error_code,
        "motion_outcome": "failed",
        "moveit_error_code": int(moveit_error_code),
        "planned_point_count": int(planned_point_count),
        "execution_started": execution_started,
    }


def _qualification_ik_response_timeout_s(seed_timeout_s: float) -> float:
    """Keep a short solver budget without timing out behind concurrent RPCs."""

    from agent.runtime.moveit_qualification import KINEMATIC_IK_TIMEOUT_S

    return max(KINEMATIC_IK_TIMEOUT_S, float(seed_timeout_s) + 0.1)


def _urdf_reach_upper_bound_m(config: GazeboControlConfig) -> float:
    """Return the unique base-to-tip chain bound, plus the fixed tool mount."""

    try:
        chain = _qualification_serial_chain(config)
        mount = math.sqrt(sum(float(value) ** 2 for value in config.mount_xyz))
        total = chain.translation_upper_bound_m + mount
    except (OSError, ValueError):
        return math.inf
    return total if total > 0.0 and math.isfinite(total) else math.inf


def _qualification_model_paths(config: GazeboControlConfig) -> tuple[Path, Path]:
    package = config.ros_workspace / "src" / "openeta_rm75_robotiq2f85_sim"
    pickplace = "pickplace" in config.model_id
    urdf = (
        package
        / "urdf"
        / ("rm75_robotiq2f85_pickplace.urdf.xacro" if pickplace else "rm75_robotiq2f85.urdf.xacro")
    )
    srdf = (
        package
        / "config"
        / ("rm75_robotiq2f85_pickplace.srdf" if pickplace else "rm75_robotiq2f85.srdf")
    )
    return urdf, srdf


@lru_cache(maxsize=8)
def _expanded_qualification_urdf(config: GazeboControlConfig) -> bytes:
    urdf, _ = _qualification_model_paths(config)
    try:
        import xacro

        return xacro.process_file(str(urdf)).toxml().encode("utf-8")
    except Exception as exc:  # noqa: BLE001 - non-ROS unit tests lack xacro.
        if os.environ.get("ROS_DISTRO"):
            raise RuntimeError("qualification URDF expansion failed") from exc
        # Unit-test/non-ROS imports have no xacro package. The live Jazzy
        # runtime expands successfully because the same file already fed the
        # launch robot_description.
        try:
            return urdf.read_bytes()
        except OSError:
            return b"<missing-urdf>"


@lru_cache(maxsize=8)
def _qualification_serial_chain(config: GazeboControlConfig):
    from agent.runtime.urdf_jacobian import UrdfSerialChain

    return UrdfSerialChain.from_urdf(
        _expanded_qualification_urdf(config),
        base_link=config.base_link,
        tip_link=config.arm_tip,
    )


def _qualification_robot_model_sha256(config: GazeboControlConfig) -> str:
    """Hash URDF/SRDF/group/TCP/gripper inputs for capability-map binding."""

    from agent.runtime.capability_map import robot_model_hash

    urdf, srdf = _qualification_model_paths(config)
    # The offline generator hashes the versioned model input supplied on its
    # command line. Hash that same URDF/Xacro source here: expanded XML must
    # not give an identical robot a different capability-map identity merely
    # because the optional xacro module is available. FK/Jacobian evaluation
    # still uses the expanded document above.
    try:
        urdf_bytes = urdf.read_bytes()
    except OSError:
        urdf_bytes = b"<missing-urdf>"
    try:
        srdf_bytes = srdf.read_bytes()
    except OSError:
        srdf_bytes = b"<missing-srdf>"
    return robot_model_hash(
        urdf=urdf_bytes,
        srdf=srdf_bytes,
        planning_group=config.move_group,
        tcp=config.arm_tip,
        gripper="robotiq_2f85",
    )


def _configured_qualification_solver_profile() -> str:
    """Resolve ``auto`` exactly as the ROS launch file does."""

    qualification_profile = os.environ.get("OPENETA_QUALIFICATION_PROFILE", "legacy")
    solver_profile = os.environ.get("OPENETA_QUALIFICATION_SOLVER_PROFILE", "auto")
    if solver_profile == "auto":
        return "kdl_fast" if qualification_profile == "fast_v3" else "kdl_legacy"
    return solver_profile


@lru_cache(maxsize=8)
def _qualification_solver_version(solver_profile: str) -> str:
    """Read the installed ROS package version for artifact provenance."""

    package = (
        "trac_ik_kinematics_plugin"
        if solver_profile.startswith("trac_ik")
        else "pick_ik"
        if solver_profile.startswith("pick_ik")
        else "moveit_kinematics"
        if solver_profile.startswith("kdl")
        else ""
    )
    if not package:
        return "unknown"
    try:
        from ament_index_python.packages import get_package_share_directory

        root = ElementTree.parse(
            Path(get_package_share_directory(package)) / "package.xml"
        ).getroot()
        version = root.findtext("version")
    except Exception:  # noqa: BLE001 - provenance must not block qualification.
        return "unknown"
    return str(version or "unknown")


def _load_qualification_capability_map(
    config: GazeboControlConfig,
    *,
    map_id: str,
    robot_model_sha256: str,
) -> Mapping[str, Any]:
    """Load one content-addressed map without consulting user-home caches."""

    from agent.runtime.capability_map import SparseCapabilityMap

    if len(map_id) != 64 or any(character not in "0123456789abcdef" for character in map_id):
        raise ValueError("capability map ID must be a lowercase SHA-256 digest")
    override = os.environ.get("OPENETA_CAPABILITY_MAP_PATH", "").strip()
    repo_root = config.ros_workspace.parents[2]
    path = (
        Path(override) if override else repo_root / "config" / "capability_maps" / f"{map_id}.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load capability map {path}: {type(exc).__name__}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("capability map root must be an object")
    SparseCapabilityMap.from_dict(
        payload,
        expected_map_id=map_id,
        expected_robot_model_sha256=robot_model_sha256,
    )
    return dict(payload)


def _normalized_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("pose quaternion must be finite and non-zero")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quaternion_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    q = _normalized_quaternion(quaternion)
    conjugate = (-q[0], -q[1], -q[2], q[3])
    rotated = _quaternion_multiply(
        _quaternion_multiply(q, (vector[0], vector[1], vector[2], 0.0)),
        conjugate,
    )
    return rotated[:3]


def _relative_pose(
    *,
    child_xyz: tuple[float, float, float],
    child_quat_xyzw: tuple[float, float, float, float],
    parent_xyz: tuple[float, float, float],
    parent_quat_xyzw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return ``T_parent^-1 * T_child`` without discarding either rotation."""

    parent_q = _normalized_quaternion(parent_quat_xyzw)
    child_q = _normalized_quaternion(child_quat_xyzw)
    parent_inverse = (-parent_q[0], -parent_q[1], -parent_q[2], parent_q[3])
    delta = tuple(child_xyz[index] - parent_xyz[index] for index in range(3))
    return (
        _quaternion_rotate(parent_inverse, delta),
        _normalized_quaternion(_quaternion_multiply(parent_inverse, child_q)),
    )


def _child_world_pose(
    *,
    parent_xyz: tuple[float, float, float],
    parent_quat_xyzw: tuple[float, float, float, float],
    relative_xyz: tuple[float, float, float],
    relative_quat_xyzw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    parent_q = _normalized_quaternion(parent_quat_xyzw)
    offset = _quaternion_rotate(parent_q, relative_xyz)
    return (
        tuple(parent_xyz[index] + offset[index] for index in range(3)),
        _normalized_quaternion(_quaternion_multiply(parent_q, relative_quat_xyzw)),
    )


def _collision_primitive_from_spec(spec: Mapping[str, Any]) -> CollisionPrimitive:
    shape = str(spec.get("shape") or "")
    common = {
        "shape": shape,
        "pose_xyz": tuple(float(value) for value in spec["pose_xyz"]),
        "pose_quat_xyzw": tuple(float(value) for value in spec["pose_quat_xyzw"]),
    }
    if shape == "box":
        return CollisionPrimitive(
            **common,
            size_xyz=tuple(float(value) for value in spec["size_xyz"]),
        )
    if shape == "cylinder":
        return CollisionPrimitive(
            **common,
            radius=float(spec["radius"]),
            length=float(spec["length"]),
        )
    raise PlanningSceneError("collision primitive shape is unsupported")


def _collision_geometry(
    *,
    object_id: str,
    bounding_box_xyz: Sequence[float],
    pose_xyz: Sequence[float],
    pose_quat_xyzw: Sequence[float],
    primitives: Sequence[Mapping[str, Any]] = (),
) -> CollisionGeometry:
    """Build one exact scene body without changing its outer-bound contract."""

    size = tuple(float(value) for value in bounding_box_xyz)
    xyz = tuple(float(value) for value in pose_xyz)
    quat = tuple(float(value) for value in pose_quat_xyzw)
    if primitives:
        return CollisionBody(
            object_id,
            size,  # type: ignore[arg-type]
            xyz,  # type: ignore[arg-type]
            quat,  # type: ignore[arg-type]
            tuple(_collision_primitive_from_spec(spec) for spec in primitives),
        )
    return CollisionBox(
        object_id,
        size,  # type: ignore[arg-type]
        xyz,  # type: ignore[arg-type]
        quat,  # type: ignore[arg-type]
    )


def _target_collision_geometry(
    config: Any,
    *,
    pose_xyz: Sequence[float],
    pose_quat_xyzw: Sequence[float],
) -> CollisionGeometry:
    return _collision_geometry(
        object_id=str(config.target_id),
        bounding_box_xyz=config.target_size_m,
        pose_xyz=pose_xyz,
        pose_quat_xyzw=pose_quat_xyzw,
        primitives=tuple(getattr(config, "target_collision_primitives", ())),
    )


def _stamp_seconds(stamp: Any) -> float | None:
    if stamp is None:
        return None
    return float(int(getattr(stamp, "sec", 0))) + float(int(getattr(stamp, "nanosec", 0))) * 1e-9


def _moveit_scene_frame(frame: object, *, base_link: str) -> str:
    """Map the co-located Gazebo world frame into MoveIt's fixed root frame."""

    value = str(frame or "")
    # This profile spawns the robot at the Gazebo world origin and its SRDF
    # has a fixed base_link root, so MoveIt has no separate `world` TF frame.
    # World collision poses are numerically base_link poses under that asset
    # contract; attached-object link frames must pass through unchanged.
    return base_link if value in {"", "world"} else value


def _canonical_pose_values(
    xyz: Sequence[float], quaternion_xyzw: Sequence[float]
) -> dict[str, list[float]]:
    quaternion = _normalized_quaternion(tuple(float(value) for value in quaternion_xyzw))
    # q and -q are the same rotation. Canonicalize the sign before hashing a
    # MoveIt readback so transport normalization cannot create a false drift.
    for value in reversed(quaternion):
        if abs(value) > 1e-15:
            if value < 0.0:
                quaternion = tuple(-item for item in quaternion)
            break

    def finite(value: float) -> float:
        if not math.isfinite(value):
            raise PlanningSceneError("MoveIt collision readback contains non-finite data")
        return round(value, 12)

    return {
        "xyz": [finite(float(value)) for value in xyz],
        "quat_xyzw": [finite(float(value)) for value in quaternion],
    }


def _ros_pose_values(
    pose: Any,
    *,
    allow_default_identity: bool = False,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    xyz = (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
    quaternion = (
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )
    if not all(math.isfinite(value) for value in (*xyz, *quaternion)):
        raise PlanningSceneError("MoveIt collision readback contains non-finite data")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if allow_default_identity and norm <= 1e-15:
        return xyz, (0.0, 0.0, 0.0, 1.0)
    return xyz, _normalized_quaternion(quaternion)


def _collision_message_geometry_record(collision: Any) -> dict[str, Any]:
    primitives = list(getattr(collision, "primitives", ()))
    poses = list(getattr(collision, "primitive_poses", ()))
    if len(primitives) != len(poses) or not primitives:
        raise PlanningSceneError("MoveIt collision readback primitive count is invalid")
    object_pose = getattr(collision, "pose", None)
    if object_pose is None:
        object_xyz = (0.0, 0.0, 0.0)
        object_quaternion = (0.0, 0.0, 0.0, 1.0)
    else:
        object_xyz, object_quaternion = _ros_pose_values(
            object_pose,
            allow_default_identity=True,
        )
    entries = []
    for primitive, pose in zip(primitives, poses, strict=True):
        dimensions = [round(float(value), 12) for value in primitive.dimensions]
        if not dimensions or any(
            not math.isfinite(value) or value <= 0.0 for value in dimensions
        ):
            raise PlanningSceneError("MoveIt collision readback dimensions are invalid")
        primitive_xyz, primitive_quaternion = _ros_pose_values(pose)
        world_xyz, world_quaternion = _child_world_pose(
            parent_xyz=object_xyz,
            parent_quat_xyzw=object_quaternion,
            relative_xyz=primitive_xyz,
            relative_quat_xyzw=primitive_quaternion,
        )
        entries.append(
            {
                "type": int(primitive.type),
                "dimensions": dimensions,
                "pose": _canonical_pose_values(world_xyz, world_quaternion),
            }
        )
    entries.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return {
        "id": str(collision.id),
        "frame": str(collision.header.frame_id),
        "primitives": entries,
    }


def _collision_message_geometry_map(
    collisions: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for collision in collisions:
        record = _collision_message_geometry_record(collision)
        object_id = record["id"]
        if not object_id or object_id in result:
            raise PlanningSceneError("MoveIt collision readback identity is invalid")
        result[object_id] = record
    return result


def _collision_geometry_manifest_sha256(
    records: Mapping[str, Mapping[str, Any]],
) -> str:
    return hashlib.sha256(
        json.dumps(
            [records[key] for key in sorted(records)],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def gripper_action_success(
    *,
    reached_goal: bool,
    stalled: bool,
    allow_stalling: bool,
    terminal_succeeded: bool = True,
) -> bool:
    """Stall is success only when the ROS action terminal state succeeded."""

    return bool(terminal_succeeded) and (
        bool(reached_goal) or (bool(allow_stalling) and bool(stalled))
    )


def gripper_terminal_succeeded(status: Any) -> bool:
    """Accept only action_msgs GoalStatus.STATUS_SUCCEEDED (numeric 4)."""

    try:
        return int(status) == 4
    except (TypeError, ValueError):
        return False


def _populate_state_validity_request(
    request: Any,
    candidate_positions: list[float],
    *,
    group_name: str,
    joint_names: Sequence[str] = ARM_JOINTS,
) -> None:
    names = [str(name) for name in joint_names]
    if len(names) != len(candidate_positions):
        raise ValueError("state-validity joint names and positions must align")
    request.group_name = group_name
    request.robot_state.is_diff = True
    request.robot_state.joint_state.name = names
    request.robot_state.joint_state.position = list(candidate_positions)


def _populate_recovery_trajectory_goal(
    goal: Any,
    point: Any,
    duration: Any,
    candidate_positions: list[float],
) -> None:
    goal.trajectory.joint_names = list(ARM_JOINTS)
    point.positions = list(candidate_positions)
    point.time_from_start = duration
    goal.trajectory.points = [point]


def _merged_allowed_collision_rows(
    current_names: list[str],
    current_rows: list[list[bool]],
    replacements: Mapping[str, Any],
    *,
    replace_owned: bool = True,
) -> tuple[list[str], list[list[bool]]]:
    """Replace owned object rows without erasing the SRDF self matrix.

    PlanningScene diffs replace an entire ACM, while this adapter starts from
    MoveIt's live SRDF-derived matrix.  Entries named by ``replacements`` are
    OpenETA-owned world-object policies: clear every stale pair involving each
    such object before adding the exact requested links.  This is required when
    upgrading a running workcell that previously allowed target/gripper touch
    before native attachment.
    """

    enabled_pairs: set[tuple[str, str]] = set()
    for row_index, row_name in enumerate(current_names):
        values = current_rows[row_index] if row_index < len(current_rows) else []
        for column_index, column_name in enumerate(current_names):
            if column_index < len(values) and bool(values[column_index]):
                enabled_pairs.add(tuple(sorted((row_name, column_name))))
    owned_objects = {str(key) for key in replacements}
    if replace_owned:
        enabled_pairs = {
            pair for pair in enabled_pairs if not any(name in owned_objects for name in pair)
        }
    for object_id, links in replacements.items():
        for link in links:
            enabled_pairs.add(tuple(sorted((str(object_id), str(link)))))
    names = sorted(
        set(current_names)
        | owned_objects
        | {str(link) for links in replacements.values() for link in links}
    )
    return names, [
        [tuple(sorted((row_name, column))) in enabled_pairs for column in names]
        for row_name in names
    ]


def _state_valid_with_allowed_collision_pairs(
    *,
    response_valid: bool,
    collision_pairs: Sequence[Sequence[str]],
    allowed_collisions: Mapping[str, Any] | None,
) -> tuple[bool, bool]:
    """Accept only collisions explicitly scoped to this contact endpoint."""

    if response_valid:
        return True, False
    if not collision_pairs or not isinstance(allowed_collisions, Mapping):
        return False, False
    allowed_pairs = {
        tuple(sorted((str(object_id), str(link))))
        for object_id, links in allowed_collisions.items()
        if isinstance(links, (list, tuple))
        for link in links
    }
    observed = {
        tuple(sorted((str(pair[0]), str(pair[1])))) for pair in collision_pairs if len(pair) == 2
    }
    accepted = (
        bool(observed) and len(observed) == len(collision_pairs) and observed <= allowed_pairs
    )
    return accepted, accepted


class RosGazeboStateSource:
    def __init__(
        self, node: Any, tf_buffer: Any, *, config: GazeboControlConfig, freshness_s: float = 2.0
    ):
        self.node, self.tf_buffer, self.config = node, tf_buffer, config
        self.freshness_s = float(freshness_s)
        self._lock = threading.Lock()
        self._joint_state: dict[str, list] | None = None
        self._joint_received = 0.0
        self._joint_stamp: float | None = None
        self._minimum_ros_timestamp_s: float | None = None

    def joint_state_callback(self, message: Any) -> None:
        with self._lock:
            self._joint_state = {
                "name": list(message.name),
                "position": list(message.position),
                "velocity": list(message.velocity),
            }
            self._joint_received = time.monotonic()
            stamp = getattr(getattr(message, "header", None), "stamp", None)
            self._joint_stamp = (
                float(int(getattr(stamp, "sec", 0)))
                + float(int(getattr(stamp, "nanosec", 0))) * 1e-9
                if stamp is not None
                else None
            )

    def clear(self, *, min_ros_timestamp_s: float | None = None) -> None:
        with self._lock:
            self._joint_state, self._joint_received, self._joint_stamp = None, 0.0, None
            self._minimum_ros_timestamp_s = (
                float(min_ros_timestamp_s) if min_ros_timestamp_s is not None else None
            )

    def state(self):
        with self._lock:
            joint = dict(self._joint_state) if self._joint_state is not None else None
            received = self._joint_received
            joint_stamp = self._joint_stamp
            minimum_stamp = self._minimum_ros_timestamp_s
        if joint is None or time.monotonic() - received > self.freshness_s:
            raise RuntimeError("JOINT_STATE_TIMEOUT")
        try:
            # A zero ROS time asks tf2 for the latest transform. This avoids
            # wall-clock vs Gazebo `/clock` extrapolation while the node is
            # still receiving simulated time ticks.
            try:
                from rclpy.time import Time

                lookup_time = Time()
            except ImportError:
                lookup_time = self.node.get_clock().now()
            stamped_transform = self.tf_buffer.lookup_transform(
                self.config.base_link, self.config.mount_child, lookup_time
            )
            transform = stamped_transform.transform
        except Exception as exc:
            raise RuntimeError("TF_TIMEOUT") from exc
        tf_stamp = _stamp_seconds(
            getattr(getattr(stamped_transform, "header", None), "stamp", None)
        )
        if minimum_stamp is not None and (
            joint_stamp is None
            or tf_stamp is None
            or joint_stamp + 1e-9 < minimum_stamp
            or tf_stamp + 1e-9 < minimum_stamp
        ):
            raise RuntimeError("POST_ACTION_STATE_NOT_FRESH")
        state = robot_state_from_sources(
            joint,
            {
                f"{self.config.base_link}->{self.config.mount_child}": {
                    "xyz": [
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                    ],
                    "quat_xyzw": [
                        transform.rotation.x,
                        transform.rotation.y,
                        transform.rotation.z,
                        transform.rotation.w,
                    ],
                }
            },
            config=self.config,
        )
        state.metadata.update(
            {
                "joint_state_timestamp_s": joint_stamp,
                "joint_state_received_monotonic_s": received,
                "tf_timestamp_s": tf_stamp,
            }
        )
        return state

    def wait_fresh(self, timeout_s: float = 15.0):
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self.state()
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.05)
        raise last_error or RuntimeError("ROBOT_STATE_UNAVAILABLE")


class RosGazeboController(GazeboController):
    def __init__(self, runtime: "_RosRuntime", *, config: GazeboControlConfig):
        self.runtime = runtime
        super().__init__(
            # Action implementations clear the cached JointState before
            # execution.  Reconciliation must wait for the first post-action
            # sample instead of racing an immediate non-blocking read.
            state_provider=runtime.state_source.wait_fresh,
            move_action=runtime.move,
            gripper_action=runtime.gripper,
            start_state_recovery=runtime.recover_start_state,
            cancel_pending=runtime.cancel_pending,
            close_source=runtime.close,
            scene_revision_provider=lambda: int(runtime.planning_scene.revision),
            motion_scene_ready=lambda: bool(runtime.planning_scene.ready),
            candidate_qualifier=runtime.qualify_motion_candidates,
            config=config,
        )

    def wait_ready(self, timeout_s: float = 30.0) -> None:
        self.runtime.wait_ready(timeout_s)

    def reset_sources(self) -> None:
        self.runtime.cancel_pending()
        self.runtime.state_source.clear()

    @property
    def planning_scene(self) -> PlanningSceneSynchronizer:
        return self.runtime.planning_scene

    def sync_planning_scene_reset(
        self,
        config: Any,
        *,
        target_xyz: Sequence[float] | None = None,
        target_quat_xyzw: Sequence[float] | None = None,
        world_model_poses: Mapping[
            str, tuple[Sequence[float], Sequence[float]]
        ] | None = None,
    ) -> int:
        """Reset MoveIt from the settled native Gazebo target pose.

        RGB-D and native contact observe the dynamic body after Gazebo physics
        starts.  Reusing its pre-physics SDF declaration here can make model
        candidates and collision proof refer to different object poses.
        Callers therefore supply one atomic native Pose_V sample; the optional
        fallback exists only for non-physics/unit-test controller use.
        """

        if (target_xyz is None) != (target_quat_xyzw is None):
            raise ValueError("reset target position and orientation must be paired")
        measured_target_xyz = tuple(
            float(value)
            for value in (config.target_initial_xyz if target_xyz is None else target_xyz)
        )
        measured_target_quat = tuple(
            float(value)
            for value in (
                config.target_initial_quat_xyzw if target_quat_xyzw is None else target_quat_xyzw
            )
        )
        if (
            len(measured_target_xyz) != 3
            or len(measured_target_quat) != 4
            or not all(
                math.isfinite(value) for value in (*measured_target_xyz, *measured_target_quat)
            )
        ):
            raise ValueError("reset target pose must be finite SE(3) data")
        measured_target_quat = _normalized_quaternion(measured_target_quat)
        authoritative_table = getattr(config, "authoritative_table_spec", None)
        if isinstance(authoritative_table, Mapping):
            table = _collision_geometry(
                object_id=str(authoritative_table["id"]),
                bounding_box_xyz=authoritative_table["size_xyz"],
                pose_xyz=authoritative_table["pose_xyz"],
                pose_quat_xyzw=authoritative_table["pose_quat_xyzw"],
                primitives=tuple(authoritative_table.get("primitives") or ()),
            )
        else:
            table = CollisionBox(
                config.table_id,
                tuple(config.table_size_m),
                tuple(config.table_pose_xyz),
            )
        distractor_size = tuple(config.distractor_size_m)
        if len(distractor_size) == 2:
            distractor_size = (distractor_size[0], distractor_size[0], distractor_size[1])
        distractor = (
            None
            if isinstance(authoritative_table, Mapping)
            or getattr(config, "replace_default_distractor", False)
            else CollisionBox(
                config.distractor_id,
                distractor_size,
                tuple(config.distractor_initial_xyz),
            )
        )
        measured_world_poses = dict(world_model_poses or {})
        expected_dynamic_ids = set(
            getattr(config, "authoritative_dynamic_obstacle_ids", ())
        )
        if expected_dynamic_ids and set(measured_world_poses) != expected_dynamic_ids:
            raise ValueError(
                "authoritative Gazebo pose snapshot is incomplete: "
                f"expected={sorted(expected_dynamic_ids)} "
                f"actual={sorted(measured_world_poses)}"
            )

        def settled_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
            result = dict(spec)
            measured = measured_world_poses.get(str(spec["id"]))
            if measured is not None:
                xyz, quaternion = measured
                if len(xyz) != 3 or len(quaternion) != 4 or not all(
                    math.isfinite(float(value)) for value in (*xyz, *quaternion)
                ):
                    raise ValueError("authoritative Gazebo object pose is invalid")
                result["pose_xyz"] = [float(value) for value in xyz]
                result["pose_quat_xyzw"] = list(
                    _normalized_quaternion(tuple(float(value) for value in quaternion))
                )
            return result

        obstacles = tuple(
            _collision_geometry(
                object_id=str(settled["id"]),
                bounding_box_xyz=settled["size_xyz"],
                pose_xyz=settled["pose_xyz"],
                pose_quat_xyzw=settled["pose_quat_xyzw"],
                primitives=tuple(settled.get("primitives") or ()),
            )
            for spec in getattr(config, "static_obstacle_specs", ())
            for settled in (settled_spec(spec),)
        )
        revision = self.planning_scene.reset(
            table=table,
            distractor=distractor,
            target=_target_collision_geometry(
                config,
                pose_xyz=measured_target_xyz,
                pose_quat_xyzw=measured_target_quat,
            ),
            obstacles=obstacles,
            robot_support_link=str(config.base_link),
            authoritative_scene_sha256=str(
                getattr(config, "authoritative_scene_sha256", "")
            ),
        )
        self._require_current_planning_state_valid()
        self.runtime.scene_revision = revision
        self.runtime.planning_scene_ready = self.planning_scene.ready
        return revision

    def sync_planning_scene_empty(self) -> int:
        revision = self.planning_scene.initialize_empty()
        self.runtime.scene_revision = revision
        self.runtime.planning_scene_ready = self.planning_scene.ready
        return revision

    def sync_planning_scene_attach(
        self,
        config: Any,
        *,
        target_xyz: tuple[float, float, float],
        target_quat_xyzw: tuple[float, float, float, float],
        mount_xyz: tuple[float, float, float],
        mount_quat_xyzw: tuple[float, float, float, float],
    ) -> int:
        relative_xyz, relative_quaternion = _relative_pose(
            child_xyz=target_xyz,
            child_quat_xyzw=target_quat_xyzw,
            parent_xyz=mount_xyz,
            parent_quat_xyzw=mount_quat_xyzw,
        )
        revision = self.planning_scene.attach_target(
            target=_target_collision_geometry(
                config,
                pose_xyz=target_xyz,
                pose_quat_xyzw=target_quat_xyzw,
            ),
            link_name=config.parent_link,
            relative_pose_xyz=relative_xyz,
            relative_pose_quat_xyzw=relative_quaternion,
        )
        self._require_current_planning_state_valid()
        self.runtime.scene_revision = revision
        self.runtime.planning_scene_ready = self.planning_scene.ready
        return revision

    def sync_planning_scene_target_pose(
        self,
        config: Any,
        *,
        target_xyz: tuple[float, float, float],
        target_quat_xyzw: tuple[float, float, float, float],
        allow_target_touch: bool = False,
    ) -> int:
        revision = self.planning_scene.update_world_target(
            target=_target_collision_geometry(
                config,
                pose_xyz=target_xyz,
                pose_quat_xyzw=target_quat_xyzw,
            )
        )
        self._require_current_planning_state_valid(
            allowed_collisions=(
                {config.target_id: list(TARGET_TOUCH_LINKS)} if allow_target_touch else None
            )
        )
        self.runtime.scene_revision = revision
        self.runtime.planning_scene_ready = self.planning_scene.ready
        return revision

    def _require_current_planning_state_valid(
        self,
        *,
        allowed_collisions: Mapping[str, Any] | None = None,
    ) -> None:
        validity = (
            self.runtime.current_state_validity(
                timeout_s=3.0,
                allowed_collisions=allowed_collisions,
            )
            if allowed_collisions is not None
            else self.runtime.current_state_validity(timeout_s=3.0)
        )
        self.runtime.planning_scene_validation = validity
        if validity.get("valid") is True:
            return
        self.planning_scene.ready = False
        pairs = validity.get("collision_pairs") or []
        raise PlanningSceneError(
            "planning-scene current state is invalid; collision_pairs=" + repr(pairs)
        )

    def sync_planning_scene_detach(
        self,
        config: Any,
        *,
        target_xyz: tuple[float, float, float],
        target_quat_xyzw: tuple[float, float, float, float],
    ) -> int:
        revision = self.planning_scene.detach_target(
            target=_target_collision_geometry(
                config,
                pose_xyz=target_xyz,
                pose_quat_xyzw=target_quat_xyzw,
            )
        )
        self.runtime.scene_revision = revision
        self.runtime.planning_scene_ready = self.planning_scene.ready
        return revision

    def observation_barrier_s(self) -> float:
        """Current ROS/simulation timestamp for post-action camera ordering."""

        return self.runtime.ros_time_s()

    def return_home(self, timeout_s: float = 15.0):
        """Drive the arm back to the zero (spawn) joint configuration.

        A model-only world reset restores entity poses but leaves the arm at
        whatever configuration the last action ended in, with the trajectory
        controller still holding the stale setpoint.  native-grasp resets once per
        candidate/round and needs every round to start from the same state.
        """

        return dict(self.runtime.return_home(timeout_s))


@dataclass(slots=True)
class RosGazeboControllerFactory:
    readiness_timeout_s: float = 30.0

    def __call__(self, config: GazeboControlConfig | None = None) -> RosGazeboController:
        return self.create(config)

    def create(
        self,
        config: GazeboControlConfig | None = None,
        *,
        context: Any | None = None,
        executor: Any | None = None,
    ) -> RosGazeboController:
        cfg = config or GazeboControlConfig()
        cfg.validate_assets()
        try:
            import rclpy
            from builtin_interfaces.msg import Duration
            from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
            from moveit_msgs.action import MoveGroup
            from controller_manager_msgs.srv import ListControllers
            from moveit_msgs.srv import GetPositionIK, GetStateValidity
            from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
            from moveit_msgs.msg import (
                AllowedCollisionEntry,
                AttachedCollisionObject,
                CollisionObject,
                PlanningScene,
                PlanningSceneComponents,
            )
            from rcl_interfaces.msg import Parameter as InterfaceParameter, ParameterType
            from rcl_interfaces.srv import GetParameters, SetParametersAtomically
            from rclpy.action import ActionClient
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.executors import MultiThreadedExecutor
            from sensor_msgs.msg import JointState
            from tf2_ros import Buffer, TransformListener
            from trajectory_msgs.msg import JointTrajectoryPoint
        except ImportError as exc:
            raise RuntimeError("ROS_NOT_READY") from exc
        owns_context = context is None and not rclpy.ok()
        if owns_context:
            rclpy.init(args=None)
        from rclpy.parameter import Parameter

        node = rclpy.create_node(
            "openeta_gazebo_controller",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
            context=context,
        )
        tf_buffer = Buffer(node=node)
        listener = TransformListener(tf_buffer, node, spin_thread=False)
        source = RosGazeboStateSource(node, tf_buffer, config=cfg)
        subscription = node.create_subscription(
            JointState, "/joint_states", source.joint_state_callback, 10
        )
        move_client = ActionClient(node, MoveGroup, "/move_action")
        gripper_client = ActionClient(
            node, ParallelGripperCommand, "/parallel_gripper_controller/gripper_cmd"
        )
        trajectory_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/rm_group_controller/follow_joint_trajectory",
        )
        controller_list_client = node.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        controller_parameter_client = node.create_client(
            GetParameters, "/controller_manager/get_parameters"
        )
        qualification_callback_group = ReentrantCallbackGroup()
        state_validity_client = node.create_client(
            GetStateValidity,
            "/check_state_validity",
            callback_group=qualification_callback_group,
        )
        compute_ik_client = node.create_client(
            GetPositionIK,
            "/compute_ik",
            callback_group=qualification_callback_group,
        )
        move_group_parameter_client = node.create_client(
            SetParametersAtomically,
            "/move_group/set_parameters_atomically",
            callback_group=qualification_callback_group,
        )
        apply_scene_client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        get_scene_client = node.create_client(GetPlanningScene, "/get_planning_scene")
        shared_executor = executor is not None
        executor = executor or MultiThreadedExecutor(num_threads=12, context=context)
        executor.add_node(node)
        runtime = _RosRuntime(
            rclpy=rclpy,
            node=node,
            executor=executor,
            state_source=source,
            move_client=move_client,
            gripper_client=gripper_client,
            trajectory_client=trajectory_client,
            controller_list_client=controller_list_client,
            controller_parameter_client=controller_parameter_client,
            state_validity_client=state_validity_client,
            compute_ik_client=compute_ik_client,
            move_group_parameter_client=move_group_parameter_client,
            controller_service_type=ListControllers,
            controller_parameter_service_type=GetParameters,
            state_validity_service_type=GetStateValidity,
            compute_ik_service_type=GetPositionIK,
            set_parameters_service_type=SetParametersAtomically,
            interface_parameter_type=InterfaceParameter,
            parameter_type=ParameterType,
            follow_trajectory_action_type=FollowJointTrajectory,
            duration_type=Duration,
            trajectory_point_type=JointTrajectoryPoint,
            listener=listener,
            subscription=subscription,
            owns_context=owns_context,
            config=cfg,
            allow_stalling=bool(getattr(cfg, "allow_stalling", False)),
            shared_executor=shared_executor,
            planning_scene=None,
            scene_revision=0,
            planning_scene_ready=True,
            planning_scene_validation=None,
            apply_scene_client=apply_scene_client,
            get_scene_client=get_scene_client,
            apply_scene_service_type=ApplyPlanningScene,
            get_scene_service_type=GetPlanningScene,
            planning_scene_message_type=PlanningScene,
            planning_scene_components_type=PlanningSceneComponents,
            collision_object_type=CollisionObject,
            attached_collision_object_type=AttachedCollisionObject,
            allowed_collision_entry_type=AllowedCollisionEntry,
            solid_primitive_type=__import__(
                "shape_msgs.msg", fromlist=["SolidPrimitive"]
            ).SolidPrimitive,
            pose_type=__import__("geometry_msgs.msg", fromlist=["Pose"]).Pose,
        )
        runtime.planning_scene = PlanningSceneSynchronizer(runtime.apply_planning_scene)
        runtime.start()
        controller = RosGazeboController(runtime, config=cfg)
        try:
            controller.wait_ready(self.readiness_timeout_s)
        except Exception:
            controller.close()
            raise
        return controller


class _RosRuntime:
    def __init__(self, **values: Any):
        self.__dict__.update(values)
        self._thread: threading.Thread | None = None
        self._pending: list[Any] = []
        self._lock = threading.Lock()
        self._qualification_map_lock = threading.Lock()
        # L5 trajectories are process-local and single-use. They are never
        # persisted under a user-home cache or shared across acceptance runs.
        self._l5_trajectory_cache: dict[str, dict[str, Any]] = {}
        self._closed = False

    def _l5_scene_sha256(self) -> str:
        try:
            return self.qualification_scene_sha256()
        except Exception:  # noqa: BLE001 - cache miss is the safe fallback.
            return ""

    def _store_l5_trajectory(
        self,
        *,
        goal: Mapping[str, Any],
        trajectory: Any,
        point_count: int,
    ) -> str | None:
        start = goal.get("start_joint_state")
        if _normalized_arm_joint_state(start) is None or point_count <= 0:
            return None
        revision = int(
            goal.get(
                "planning_scene_revision",
                getattr(self.planning_scene, "revision", -1),
            )
        )
        scene_sha256 = self._l5_scene_sha256()
        endpoint = _trajectory_end_joint_state_with_sha256(trajectory)
        if endpoint is None:
            return None
        end_joint_state, end_joint_state_sha256 = endpoint
        cache_goal = dict(goal)
        cache_goal["qualification_goal_joint_state"] = end_joint_state
        cache_goal["qualification_goal_joint_state_sha256"] = end_joint_state_sha256
        key = _l5_trajectory_cache_key(
            cache_goal,
            scene_revision=revision,
            scene_sha256=scene_sha256,
        )
        if key is None:
            return None
        self._l5_trajectory_cache[key] = {
            "trajectory": copy.deepcopy(trajectory),
            "start_joint_state": dict(start),
            "scene_revision": revision,
            "scene_sha256": scene_sha256,
            "point_count": int(point_count),
            "end_joint_state": end_joint_state,
            "end_joint_state_sha256": end_joint_state_sha256,
        }
        while len(self._l5_trajectory_cache) > L5_TRAJECTORY_CACHE_LIMIT:
            del self._l5_trajectory_cache[next(iter(self._l5_trajectory_cache))]
        return key

    def _take_matching_l5_trajectory(
        self,
        goal: Mapping[str, Any],
    ) -> tuple[
        tuple[str, dict[str, Any]] | None,
        dict[str, Any],
    ]:
        live_start = goal.get("live_start_joint_state")
        if _normalized_arm_joint_state(live_start) is None:
            return None, {
                "status": "miss",
                "reason": "invalid_live_start_joint_state",
            }
        revision = int(
            goal.get(
                "planning_scene_revision",
                getattr(self.planning_scene, "revision", -1),
            )
        )
        current_revision = int(getattr(self.planning_scene, "revision", -2))
        if revision != current_revision:
            return None, {
                "status": "miss",
                "reason": "planning_scene_revision_changed",
                "requested_scene_revision": revision,
                "current_scene_revision": current_revision,
            }
        scene_sha256 = self._l5_scene_sha256()
        key = _l5_trajectory_cache_key(
            goal,
            scene_revision=revision,
            scene_sha256=scene_sha256,
        )
        if key is None:
            return None, {
                "status": "miss",
                "reason": "proof_cache_key_unavailable",
                "entry_count": len(self._l5_trajectory_cache),
            }
        entry = self._l5_trajectory_cache.get(key or "")
        if not isinstance(entry, dict):
            return None, {
                "status": "miss",
                "reason": "proof_cache_key_not_found",
                "entry_count": len(self._l5_trajectory_cache),
                "lookup_key": key,
            }
        try:
            current = self.state_source.wait_fresh(1.0)
            current_start = {
                "names": list(ARM_JOINTS),
                "positions": [float(value) for value in current.joint_positions[: len(ARM_JOINTS)]],
            }
        except Exception:  # noqa: BLE001 - normal replanning is safe fallback.
            return None, {
                "status": "miss",
                "reason": "fresh_joint_state_unavailable",
            }
        requested_delta = _joint_state_max_abs_delta(entry.get("start_joint_state"), live_start)
        measured_delta = _joint_state_max_abs_delta(entry.get("start_joint_state"), current_start)
        if (
            requested_delta is None
            or measured_delta is None
            or requested_delta > L5_TRAJECTORY_START_TOLERANCE_RAD
            or measured_delta > L5_TRAJECTORY_START_TOLERANCE_RAD
        ):
            return None, {
                "status": "miss",
                "reason": "proven_start_state_changed",
                "requested_start_max_delta_rad": requested_delta,
                "measured_start_max_delta_rad": measured_delta,
            }
        self._l5_trajectory_cache.pop(key, None)
        return (str(key), entry), {
            "status": "hit",
            "reason": "proof_and_start_state_match",
            "entry_count": len(self._l5_trajectory_cache),
            "lookup_key": key,
            "requested_start_max_delta_rad": requested_delta,
            "measured_start_max_delta_rad": measured_delta,
        }

    def current_state_validity(
        self,
        *,
        timeout_s: float,
        allowed_collisions: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Read back MoveIt's verdict and collision pairs for the live arm state."""

        state = self.state_source.wait_fresh(timeout_s)
        request = self.state_validity_service_type.Request()
        _populate_state_validity_request(
            request,
            [float(value) for value in state.joint_positions[: len(ARM_JOINTS)]],
            group_name=self.config.move_group,
        )
        response = self._await(self.state_validity_client.call_async(request), timeout_s)
        pairs = sorted(
            {
                tuple(
                    sorted(
                        (
                            str(getattr(contact, "contact_body_1", "")),
                            str(getattr(contact, "contact_body_2", "")),
                        )
                    )
                )
                for contact in getattr(response, "contacts", ())
                if getattr(contact, "contact_body_1", "") or getattr(contact, "contact_body_2", "")
            }
        )
        valid, contact_override = _state_valid_with_allowed_collision_pairs(
            response_valid=bool(response.valid),
            collision_pairs=pairs,
            allowed_collisions=allowed_collisions,
        )
        return {
            "valid": valid,
            "collision_pairs": [list(pair) for pair in pairs],
            "contact_collision_override": contact_override,
            "joint_state_timestamp_s": state.metadata.get("joint_state_timestamp_s"),
        }

    def qualification_joint_state(self) -> Mapping[str, Any]:
        state = self.state_source.wait_fresh(3.0)
        lower = [float(item[1]) for item in ARM_JOINT_BOUNDS]
        upper = [float(item[2]) for item in ARM_JOINT_BOUNDS]
        positions = [float(value) for value in state.joint_positions[: len(ARM_JOINTS)]]
        jacobian_quality = self.qualification_joint_quality(
            {"names": list(ARM_JOINTS), "positions": positions}
        )
        robot_hash = getattr(self, "_qualification_robot_model_hash", None)
        if robot_hash is None:
            robot_hash = _qualification_robot_model_sha256(self.config)
            self._qualification_robot_model_hash = robot_hash
        scene = self.planning_scene
        return {
            "names": list(ARM_JOINTS),
            "positions": positions,
            "joint_limits": {"lower": lower, "upper": upper},
            "home_joint_state": {
                "names": list(ARM_JOINTS),
                "positions": list(ARM_HOME_JOINT_POSITIONS),
            },
            "robot_model_sha256": robot_hash,
            "planning_group": self.config.move_group,
            "tcp": self.config.arm_tip,
            "gripper": "robotiq_2f85",
            "solver_profile": _configured_qualification_solver_profile(),
            "solver_version": _qualification_solver_version(
                _configured_qualification_solver_profile()
            ),
            "scene_sha256": self.qualification_scene_sha256(),
            "authoritative_scene_sha256": str(
                getattr(scene, "authoritative_scene_sha256", "")
            ),
            "moveit_world_geometry_sha256": str(
                getattr(scene, "world_geometry_sha256", "")
            ),
            "moveit_attached_geometry_sha256": str(
                getattr(scene, "attached_geometry_sha256", "")
            ),
            "moveit_geometry_verified_ids": list(
                getattr(scene, "geometry_verified_ids", ())
            ),
            "jacobian_quality_available": jacobian_quality.get("ok") is True,
            "jacobian_quality_error": jacobian_quality.get("error"),
        }

    def qualification_joint_quality(self, joint_state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Evaluate the concrete IK branch against the expanded runtime URDF."""

        try:
            value = _qualification_serial_chain(self.config).minimum_singular_value(
                [str(name) for name in joint_state.get("names") or ARM_JOINTS],
                [float(position) for position in joint_state.get("positions") or []],
            )
        except Exception as exc:  # noqa: BLE001 - converted to configuration evidence.
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"ok": True, "min_singular_value": value}

    def qualification_scene_sha256(self) -> str:
        snapshot = self.qualification_clone_scene()
        serializable = {
            **snapshot,
            "world_ids": sorted(snapshot.get("world_ids") or []),
            "attached_ids": sorted(snapshot.get("attached_ids") or []),
        }
        return hashlib.sha256(
            json.dumps(
                serializable,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def qualification_services_healthy(self) -> bool:
        clients = [self.compute_ik_client, self.state_validity_client]
        if _configured_qualification_solver_profile() == "pick_ik_local":
            clients.append(self.move_group_parameter_client)
        return all(client.wait_for_service(timeout_sec=0.2) for client in clients)

    def qualification_set_solver_mode(self, mode: str) -> Mapping[str, Any]:
        """Switch pick_ik only at a completed qualification-wave barrier."""

        if _configured_qualification_solver_profile() != "pick_ik_local":
            return {
                "ok": False,
                "infrastructure_error": True,
                "reason": "solver_mode_switch_requires_pick_ik_local",
            }
        if mode not in {"local", "global"}:
            return {
                "ok": False,
                "infrastructure_error": True,
                "reason": "invalid_pick_ik_mode",
            }
        if not self.move_group_parameter_client.wait_for_service(timeout_sec=0.2):
            return {
                "ok": False,
                "infrastructure_error": True,
                "reason": "move_group_parameter_service_unavailable",
            }
        request = self.set_parameters_service_type.Request()
        mode_parameter = self.interface_parameter_type()
        mode_parameter.name = f"robot_description_kinematics.{self.config.move_group}.mode"
        mode_parameter.value.type = self.parameter_type.PARAMETER_STRING
        mode_parameter.value.string_value = mode
        displacement_parameter = self.interface_parameter_type()
        displacement_parameter.name = (
            f"robot_description_kinematics.{self.config.move_group}.minimal_displacement_weight"
        )
        displacement_parameter.value.type = self.parameter_type.PARAMETER_DOUBLE
        displacement_parameter.value.double_value = 0.0 if mode == "global" else 0.001
        request.parameters = [mode_parameter, displacement_parameter]
        try:
            response = self._await(self.move_group_parameter_client.call_async(request), 1.0)
        except Exception as exc:  # noqa: BLE001 - ROS service boundary.
            return {
                "ok": False,
                "infrastructure_error": True,
                "reason": "pick_ik_mode_switch_service_error",
                "error_type": type(exc).__name__,
            }
        result = response.result
        ok = result.successful is True
        if ok:
            self._qualification_pick_ik_mode = mode
        return {
            "ok": ok,
            "infrastructure_error": not ok,
            "reason": "pick_ik_mode_switched" if ok else "pick_ik_mode_switch_rejected",
            "mode": mode,
            "detail": str(result.reason),
        }

    def qualification_capability_map(
        self, *, map_id: str, robot_model_sha256: str
    ) -> tuple[Mapping[str, Any] | None, str]:
        """Load and validate one map once for this ROS runtime."""

        key = (
            map_id,
            robot_model_sha256,
            os.environ.get("OPENETA_CAPABILITY_MAP_PATH", "").strip(),
        )
        with self._qualification_map_lock:
            if getattr(self, "_qualification_capability_map_key", None) == key:
                return (
                    getattr(self, "_qualification_capability_map_payload", None),
                    str(
                        getattr(
                            self,
                            "_qualification_capability_map_error",
                            "",
                        )
                    ),
                )
            try:
                payload = _load_qualification_capability_map(
                    self.config,
                    map_id=map_id,
                    robot_model_sha256=robot_model_sha256,
                )
                error = ""
            except ValueError as exc:
                payload, error = None, str(exc)
            self._qualification_capability_map_key = key
            self._qualification_capability_map_payload = payload
            self._qualification_capability_map_error = error
            return payload, error

    def qualification_workspace_filter(self, target: Mapping[str, Any]) -> bool:
        """Reject only poses beyond a URDF-derived conservative reach envelope."""

        xyz = target.get("xyz")
        if not isinstance(xyz, list) or len(xyz) != 3:
            return False
        try:
            values = [float(value) for value in xyz]
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in values):
            return False
        reach = getattr(self, "_qualification_reach_upper_bound_m", None)
        if reach is None:
            reach = _urdf_reach_upper_bound_m(self.config)
            self._qualification_reach_upper_bound_m = reach
        return (
            not math.isfinite(reach) or math.sqrt(sum(value * value for value in values)) <= reach
        )

    def qualification_compute_ik(
        self,
        target: Mapping[str, Any],
        start: Mapping[str, Any],
        avoid_collisions: bool,
    ) -> Mapping[str, Any]:
        from agent.runtime.moveit_qualification import KINEMATIC_IK_TIMEOUT_S
        from geometry_msgs.msg import PoseStamped

        goal = make_move_group_goal(dict(target), config=self.config, tolerances=target)
        xyz = goal["target_pose"].get("xyz")
        quat = goal["target_pose"].get("quat_xyzw")
        if (
            not isinstance(xyz, list)
            or len(xyz) != 3
            or not isinstance(quat, list)
            or len(quat) != 4
        ):
            return {"ok": False}
        configured_solver = _configured_qualification_solver_profile()
        requested_solver = str(target.get("solver_profile") or "auto")
        dynamic_pick_global = (
            requested_solver == "pick_ik_global"
            and configured_solver == "pick_ik_local"
            and getattr(self, "_qualification_pick_ik_mode", "local") == "global"
        )
        if (
            requested_solver != "auto"
            and requested_solver != configured_solver
            and not dynamic_pick_global
        ):
            return {
                "ok": False,
                "infrastructure_error": True,
                "reason": "qualification_solver_profile_mismatch",
                "requested_solver": requested_solver,
                "configured_solver": configured_solver,
            }
        request = self.compute_ik_service_type.Request()
        ik = request.ik_request
        ik.group_name = self.config.move_group
        ik.ik_link_name = goal["link_name"]
        ik.avoid_collisions = bool(avoid_collisions)
        seed_timeout_s = max(
            0.001,
            min(
                KINEMATIC_IK_TIMEOUT_S,
                float(target.get("ik_seed_timeout_s", KINEMATIC_IK_TIMEOUT_S)),
            ),
        )
        ik.timeout.sec = int(seed_timeout_s)
        ik.timeout.nanosec = int((seed_timeout_s - int(seed_timeout_s)) * 1_000_000_000)
        ik.robot_state.is_diff = False
        ik.robot_state.joint_state.name = list(start.get("names") or ARM_JOINTS)
        ik.robot_state.joint_state.position = [float(v) for v in start.get("positions") or []]
        scene_diff = target.get("qualification_scene_diff")
        if isinstance(scene_diff, Mapping):
            diff_message = self._qualification_scene_diff_message(scene_diff)
            ik.robot_state.is_diff = True
            ik.robot_state.attached_collision_objects = list(
                diff_message.robot_state.attached_collision_objects
            )
        pose = PoseStamped()
        pose.header.frame_id = goal["base_frame"]
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = [float(v) for v in xyz]
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = [float(v) for v in quat]
        ik.pose_stamped = pose
        response = self._await(
            self.compute_ik_client.call_async(request),
            _qualification_ik_response_timeout_s(seed_timeout_s),
        )
        solution = response.solution.joint_state
        names = list(solution.name)
        positions = list(solution.position)
        by_name = dict(zip(names, positions))
        ordered = [float(by_name[name]) for name in ARM_JOINTS if name in by_name]
        ok = int(response.error_code.val) == 1 and len(ordered) == len(ARM_JOINTS)
        quality = (
            self.qualification_joint_quality({"names": list(ARM_JOINTS), "positions": ordered})
            if ok
            else {}
        )
        return {
            "ok": ok,
            "infrastructure_error": ok and quality.get("ok") is not True,
            "reason": (
                "jacobian_quality_unavailable" if ok and quality.get("ok") is not True else None
            ),
            "moveit_error_code": int(response.error_code.val),
            "solver": "pick_ik_global" if dynamic_pick_global else configured_solver,
            "solver_version": _qualification_solver_version(configured_solver),
            "requested_solver": requested_solver,
            "jacobian_quality_available": quality.get("ok") is True if ok else None,
            "jacobian_quality_error": quality.get("error") if ok else None,
            **(
                {
                    "joint_state": {
                        "names": list(ARM_JOINTS),
                        "positions": ordered,
                    },
                    "min_singular_value": quality.get("min_singular_value", 0.0),
                }
                if ok
                else {}
            ),
        }

    def qualification_state_validity(self, joint_state: Mapping[str, Any]) -> Mapping[str, Any]:
        from agent.runtime.moveit_qualification import STATE_VALIDITY_TIMEOUT_S

        names = list(ARM_JOINTS)
        positions = [float(value) for value in joint_state.get("positions") or []]
        requested_gripper_state = joint_state.get("qualification_gripper_state")
        gripper_positions: list[tuple[str, float | None]]
        if requested_gripper_state == "open":
            gripper_positions = [("open", float(self.config.gripper_position(1)))]
        elif requested_gripper_state == "closed":
            gripper_positions = [("closed", float(self.config.gripper_position(0)))]
        elif requested_gripper_state == "closing_sweep":
            closed = float(self.config.gripper_position(0))
            gripper_positions = [
                ("near_open", min(0.05, closed)),
                *[
                    (f"close_{index}", float(value))
                    for index, value in enumerate(self.config.calibration.angles_rad[1:], 1)
                ],
            ]
        elif requested_gripper_state is None:
            gripper_positions = [("current", None)]
        else:
            return {
                "valid": False,
                "collision_pairs": [],
                "contact_collision_override": False,
                "reason": "qualification_gripper_state_unsupported",
            }

        allowed = joint_state.get("qualification_allowed_collisions")
        scene_diff = joint_state.get("qualification_scene_diff")
        checks: list[dict[str, Any]] = []
        all_pairs: set[tuple[str, str]] = set()
        all_valid = True
        any_contact_override = False
        seed_independent_static_collision = False
        target_contact_id = next(
            (
                str(object_id)
                for object_id, links in (allowed.items() if isinstance(allowed, Mapping) else ())
                if isinstance(links, (list, tuple))
                and LEFT_FINGERTIP in {str(link) for link in links}
                and RIGHT_FINGERTIP in {str(link) for link in links}
            ),
            "",
        )
        bilateral_contact_required = bool(
            requested_gripper_state == "closing_sweep" and target_contact_id
        )
        bilateral_contact_predicted = False
        detached_collision_probe_count = 0
        for label, gripper_position in gripper_positions:
            request = self.state_validity_service_type.Request()
            request_names = list(names)
            request_positions = list(positions)
            if gripper_position is not None:
                # MoveIt's mimic-joint model expands the remaining Robotiq
                # linkage from this active joint.  A deterministic close sweep
                # catches static collisions that an open-hand contact state
                # cannot reveal before the physical gripper is actuated.
                request_names.append(self.config.active_joint)
                request_positions.append(gripper_position)
            _populate_state_validity_request(
                request,
                request_positions,
                group_name=self.config.move_group,
                joint_names=request_names,
            )
            if isinstance(scene_diff, Mapping):
                diff_message = self._qualification_scene_diff_message(scene_diff)
                request.robot_state.is_diff = True
                request.robot_state.attached_collision_objects = list(
                    diff_message.robot_state.attached_collision_objects
                )
                # GetStateValidity accepts a RobotState diff but no world
                # PlanningScene diff.  After a virtual detach, checking only
                # the removal silently drops the released object and misses
                # robot/object overlap at the open-gripper endpoint.  Re-add
                # that exact world geometry as a request-local attached-body
                # collision probe with no touch links.  At this single robot
                # state its geometry is identical to the detached world body,
                # while MoveIt can now prove every robot/object collision.
                for spec in scene_diff.get("detached_collision_probe_objects", []):
                    if not isinstance(spec, Mapping):
                        continue
                    probe = self.attached_collision_object_type()
                    probe.link_name = str(spec.get("link_name") or self.config.mount_child)
                    probe.touch_links = []
                    probe.object = self._collision_object_from_spec(spec)
                    # The same RobotState diff already contains a REMOVE for
                    # the real attached object.  Reusing that id for the
                    # request-local ADD leaves the result dependent on how a
                    # MoveIt version coalesces duplicate operations; in
                    # practice the removal can win and the released geometry
                    # silently disappears from collision checking.  A unique
                    # qualification-only id makes both operations unambiguous
                    # and deliberately inherits no target/touch-link ACM
                    # exemptions.
                    probe.object.id = f"{probe.object.id}__openeta_detached_probe"
                    request.robot_state.attached_collision_objects.append(probe)
                    detached_collision_probe_count += 1
            response = self._await(
                self.state_validity_client.call_async(request),
                STATE_VALIDITY_TIMEOUT_S,
            )
            pairs = sorted(
                {
                    tuple(sorted((str(c.contact_body_1), str(c.contact_body_2))))
                    for c in getattr(response, "contacts", ())
                }
            )
            valid, contact_override = _state_valid_with_allowed_collision_pairs(
                response_valid=bool(response.valid),
                collision_pairs=pairs,
                allowed_collisions=(allowed if isinstance(allowed, Mapping) else None),
            )
            allowed_pairs = {
                tuple(sorted((str(object_id), str(link))))
                for object_id, links in (allowed.items() if isinstance(allowed, Mapping) else ())
                if isinstance(links, (list, tuple))
                for link in links
            }
            unallowed_pairs = set(pairs) - allowed_pairs
            planning_scene = getattr(self, "planning_scene", None)
            world_ids = {str(value) for value in getattr(planning_scene, "world_ids", ())}
            gripper_links = set(TARGET_TOUCH_LINKS)
            sample_seed_independent = bool(unallowed_pairs) and all(
                (left in gripper_links and right in world_ids)
                or (right in gripper_links and left in world_ids)
                for left, right in unallowed_pairs
            )
            seed_independent_static_collision = (
                seed_independent_static_collision or sample_seed_independent
            )
            sample_target_contact_links = sorted(
                link
                for link in (LEFT_FINGERTIP, RIGHT_FINGERTIP)
                if tuple(sorted((link, target_contact_id))) in set(pairs)
            )
            sample_bilateral_contact = set(sample_target_contact_links) == {
                LEFT_FINGERTIP,
                RIGHT_FINGERTIP,
            }
            bilateral_contact_predicted = bilateral_contact_predicted or sample_bilateral_contact
            all_valid = all_valid and valid
            any_contact_override = any_contact_override or contact_override
            all_pairs.update(pairs)
            checks.append(
                {
                    "sample": label,
                    "active_joint_position_rad": gripper_position,
                    "valid": valid,
                    "collision_pairs": [list(pair) for pair in pairs],
                    "contact_collision_override": contact_override,
                    "seed_independent_gripper_static_collision": (sample_seed_independent),
                    "target_contact_links": sample_target_contact_links,
                    "bilateral_target_contact": sample_bilateral_contact,
                }
            )
            # One proven static collision is sufficient to reject this grasp
            # endpoint.  Do not spend the remaining sweep RPCs on a candidate
            # that cannot be executed.
            if not valid:
                break
        seed_independent_contact_geometry_failure = bool(
            bilateral_contact_required and not bilateral_contact_predicted
        )
        result: dict[str, Any] = {
            "valid": all_valid and not seed_independent_contact_geometry_failure,
            "collision_pairs": [list(pair) for pair in sorted(all_pairs)],
            "contact_collision_override": any_contact_override,
        }
        if detached_collision_probe_count:
            result["qualification_detached_collision_probe_count"] = (
                detached_collision_probe_count // len(gripper_positions)
            )
        if requested_gripper_state == "closing_sweep":
            result["qualification_gripper_sweep_checks"] = checks
            result["qualification_seed_independent_static_collision"] = (
                seed_independent_static_collision
            )
            result["qualification_bilateral_target_contact_required"] = bilateral_contact_required
            result["qualification_bilateral_target_contact_predicted"] = bilateral_contact_predicted
            result["qualification_seed_independent_contact_geometry_failure"] = (
                seed_independent_contact_geometry_failure
            )
            if seed_independent_contact_geometry_failure:
                result["reason"] = "qualification_bilateral_target_contact_not_predicted"
        return result

    def qualification_plan_only(
        self,
        target: Mapping[str, Any],
        start: Mapping[str, Any],
        planning_time_s: float,
        planning_attempts: int,
    ) -> Mapping[str, Any]:
        # This private L5 call is generating the proof, so its branch cannot
        # yet carry the proof hash required at the public execution boundary.
        pose_target = dict(target)
        cache_binding = str(pose_target.pop("_qualification_cache_binding_sha256", "") or "")
        qualification_goal_joint_state = pose_target.pop("qualification_goal_joint_state", None)
        goal = make_move_group_goal(pose_target, config=self.config, tolerances=target)
        goal.update(
            {
                "plan_only": True,
                "start_joint_state": dict(start),
                "allowed_planning_time_s": planning_time_s,
                "num_planning_attempts": planning_attempts,
                "model_id": self.config.model_id,
                "planning_scene_revision": int(self.planning_scene.revision),
            }
        )
        if cache_binding:
            goal["qualification_cache_binding_sha256"] = cache_binding
        if isinstance(qualification_goal_joint_state, Mapping):
            qualified_state = _qualification_joint_state_with_sha256(qualification_goal_joint_state)
            if qualified_state is None:
                raise ValueError("qualification joint goal state is invalid")
            normalized_state, state_sha256 = qualified_state
            goal["qualification_goal_joint_state"] = normalized_state
            goal["qualification_goal_joint_state_sha256"] = state_sha256
        scene_diff = target.get("qualification_scene_diff")
        if isinstance(scene_diff, Mapping):
            goal["qualification_scene_diff"] = dict(scene_diff)
        allowed_collisions = target.get("qualification_allowed_collisions")
        if isinstance(allowed_collisions, Mapping):
            goal["qualification_allowed_collisions"] = {
                str(key): [str(value) for value in values]
                for key, values in allowed_collisions.items()
                if isinstance(values, (list, tuple))
            }
        return self.move(goal, planning_time_s + 5.0)

    def _qualification_scene_diff_message(self, diff: Mapping[str, Any]) -> Any:
        scene = self.planning_scene_message_type()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        for object_id in diff.get("remove_world_ids", []):
            collision = self.collision_object_type()
            collision.id = str(object_id)
            collision.header.frame_id = self.config.base_link
            collision.operation = self.collision_object_type.REMOVE
            scene.world.collision_objects.append(collision)
        for spec in diff.get("world_objects", []):
            scene.world.collision_objects.append(self._collision_object_from_spec(spec))
        for object_id in diff.get("remove_attached_ids", []):
            attached = self.attached_collision_object_type()
            attached.object.id = str(object_id)
            attached.object.operation = self.collision_object_type.REMOVE
            scene.robot_state.attached_collision_objects.append(attached)
        for spec in diff.get("attached_objects", []):
            attached = self.attached_collision_object_type()
            attached.link_name = str(spec["link_name"])
            attached.touch_links = [str(value) for value in spec.get("touch_links", [])]
            attached.object = self._collision_object_from_spec(spec)
            scene.robot_state.attached_collision_objects.append(attached)
        allowed = diff.get("allowed_collisions")
        if isinstance(allowed, Mapping):
            components = self.planning_scene_components_type
            acm_request = self.get_scene_service_type.Request()
            acm_request.components.components = int(components.ALLOWED_COLLISION_MATRIX)
            acm_readback = self._await(self.get_scene_client.call_async(acm_request), 5.0)
            current_acm = acm_readback.scene.allowed_collision_matrix
            names, merged_rows = _merged_allowed_collision_rows(
                [str(value) for value in current_acm.entry_names],
                [list(row.enabled) for row in current_acm.entry_values],
                allowed,
                replace_owned=False,
            )
            scene.allowed_collision_matrix.entry_names = names
            for enabled in merged_rows:
                row = self.allowed_collision_entry_type()
                row.enabled = enabled
                scene.allowed_collision_matrix.entry_values.append(row)
            scene.allowed_collision_matrix.default_entry_names = list(
                current_acm.default_entry_names
            )
            scene.allowed_collision_matrix.default_entry_values = list(
                current_acm.default_entry_values
            )
        return scene

    def qualification_clone_scene(self) -> dict[str, Any]:
        """Clone only qualification-owned scene identity; never apply a diff."""

        reach = getattr(self, "_qualification_reach_upper_bound_m", None)
        if reach is None:
            reach = _urdf_reach_upper_bound_m(self.config)
            self._qualification_reach_upper_bound_m = reach
        snapshot = {
            "revision": int(self.planning_scene.revision),
            "world_ids": set(self.planning_scene.world_ids),
            "attached_ids": set(self.planning_scene.attached_ids),
            "world_specs": {
                key: dict(value) for key, value in self.planning_scene.world_specs.items()
            },
            "attached_specs": {
                key: dict(value) for key, value in self.planning_scene.attached_specs.items()
            },
            "target_id": self.planning_scene.target_id,
            "authoritative_scene_sha256": str(
                getattr(self.planning_scene, "authoritative_scene_sha256", "")
            ),
            "moveit_world_geometry_sha256": str(
                getattr(self.planning_scene, "world_geometry_sha256", "")
            ),
            "moveit_geometry_verified_ids": list(
                getattr(self.planning_scene, "geometry_verified_ids", ())
            ),
            "target_touch_links": list(TARGET_TOUCH_LINKS),
            "workspace_envelope": {
                "frame": self.config.base_link,
                "base_xyz": [0.0, 0.0, 0.0],
                "outer_radius_m": float(reach) if math.isfinite(reach) else None,
                # This deliberately over-bounds every measured Robotiq/object
                # attachment used by this profile.  It can prove impossible
                # object goals without falsely rejecting a reachable EEF goal.
                "maximum_attachment_offset_m": 0.25,
            },
            "gripper_collision_boxes": [
                {
                    "id": "gripper_mount_link_collision",
                    "frame": self.config.mount_child,
                    "shape": "box",
                    "size_xyz": [0.08, 0.08, 0.012],
                    "pose_xyz": [0.0, 0.0, 0.006],
                    "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "provenance": "exact_urdf_collision_primitive",
                }
            ],
            "transitions": [],
        }
        scene_evidence = getattr(self.config, "acceptance_scene_evidence", None)
        if callable(scene_evidence):
            snapshot["acceptance_scene"] = scene_evidence()
        destination_center = getattr(self.config, "destination_center_xy", None)
        destination_size = getattr(self.config, "destination_size_xy_m", None)
        support_z = getattr(
            self.config,
            "destination_support_z_m",
            getattr(self.config, "table_top_z_m", None),
        )
        if (
            isinstance(destination_center, tuple)
            and len(destination_center) == 2
            and isinstance(destination_size, tuple)
            and len(destination_size) == 2
            and isinstance(support_z, (int, float))
        ):
            snapshot["placement_region"] = {
                "schema_version": "openeta.placement_region_geometry.v1",
                "frame": self.config.base_link,
                "center_xy": [float(value) for value in destination_center],
                "size_xy_m": [float(value) for value in destination_size],
                "support_z_m": float(support_z),
                "acceptance_semantics": str(
                    getattr(
                        self.config,
                        "placement_acceptance_semantics",
                        "complete_footprint",
                    )
                ),
                "support_object_id": str(
                    getattr(
                        self.config,
                        "selected_placement_region_id",
                        getattr(self.config, "table_id", ""),
                    )
                ),
                "support_height_tolerance_m": float(
                    getattr(self.config, "placement_support_height_tolerance_m", 0.01)
                ),
                # AnyPlace consumes RGB-D point geometry.  Treat a sub-5 mm
                # support overlap as the calibrated contact uncertainty band,
                # not a mathematical penetration proof.  Non-support static
                # obstacles retain the tighter exact-box tolerance below.
                "support_penetration_tolerance_m": 0.005,
                "static_penetration_tolerance_m": 0.001,
                "provenance": "acceptance_scene_contract",
            }
        return snapshot

    def qualification_scene_transition(
        self,
        scene: Any,
        transition: str,
        target: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(scene, dict):
            return {"ok": False, "reason": "cloned_scene_missing"}
        if transition not in {"virtual_attach", "virtual_detach"}:
            return {"ok": False, "reason": "unsupported_virtual_transition"}
        target_id = str(scene.get("target_id") or "")
        xyz = target.get("xyz")
        quat = target.get("quat_xyzw")
        if not (
            target_id
            and isinstance(xyz, list)
            and len(xyz) == 3
            and isinstance(quat, list)
            and len(quat) == 4
        ):
            return {"ok": False, "reason": "virtual_transition_pose_missing"}
        if transition == "virtual_attach":
            spec = (scene.get("world_specs") or {}).get(target_id)
            if not isinstance(spec, Mapping):
                return {"ok": False, "reason": "virtual_attach_object_missing"}
            predicted = target.get("attachment_transform")
            if isinstance(predicted, Mapping):
                relative_xyz = tuple(float(value) for value in predicted.get("translation_xyz", []))
                relative_quat = tuple(float(value) for value in predicted.get("quat_xyzw", []))
                if len(relative_xyz) != 3 or len(relative_quat) != 4:
                    return {"ok": False, "reason": "predicted_attachment_invalid"}
            else:
                relative_xyz, relative_quat = _relative_pose(
                    child_xyz=tuple(float(value) for value in spec["pose_xyz"]),
                    child_quat_xyzw=tuple(float(value) for value in spec["pose_quat_xyzw"]),
                    parent_xyz=tuple(float(value) for value in xyz),
                    parent_quat_xyzw=tuple(float(value) for value in quat),
                )
            attached = {
                **dict(spec),
                "frame": self.config.mount_child,
                "pose_xyz": list(relative_xyz),
                "pose_quat_xyzw": list(relative_quat),
                "link_name": self.config.mount_child,
                "touch_links": list(
                    self.planning_scene.attached_specs.get(target_id, {}).get("touch_links") or ()
                ),
            }
            if not attached["touch_links"]:
                attached["touch_links"] = list(TARGET_TOUCH_LINKS)
            scene["attached_specs"] = {target_id: attached}
            scene.get("world_specs", {}).pop(target_id, None)
            planning_diff = {
                "remove_world_ids": [target_id],
                "attached_objects": [attached],
            }
        else:
            attached = (scene.get("attached_specs") or {}).get(target_id)
            if not isinstance(attached, Mapping):
                return {"ok": False, "reason": "virtual_detach_object_missing"}
            world_xyz, world_quat = _child_world_pose(
                parent_xyz=tuple(float(value) for value in xyz),
                parent_quat_xyzw=tuple(float(value) for value in quat),
                relative_xyz=tuple(float(value) for value in attached["pose_xyz"]),
                relative_quat_xyzw=tuple(float(value) for value in attached["pose_quat_xyzw"]),
            )
            world = {
                **dict(attached),
                "frame": self.config.base_link,
                "pose_xyz": list(world_xyz),
                "pose_quat_xyzw": list(world_quat),
            }
            scene["attached_specs"] = {}
            scene.setdefault("world_specs", {})[target_id] = world
            planning_diff = {
                "remove_attached_ids": [target_id],
                "world_objects": [world],
                # See qualification_state_validity: the public
                # GetStateValidity service cannot carry world-object diffs.
                # Preserve the exact released geometry for its request-local
                # post-open collision probe.
                "detached_collision_probe_objects": [world],
            }
        scene.setdefault("transitions", []).append(transition)
        scene_hash = hashlib.sha256(
            json.dumps(
                {
                    "revision": scene.get("revision"),
                    "world_ids": sorted(scene.get("world_ids") or []),
                    "attached_ids": sorted(scene.get("attached_ids") or []),
                    "transitions": list(scene["transitions"]),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "ok": True,
            "transition": transition,
            "virtual": True,
            "scene_hash": scene_hash,
            "real_scene_revision_unchanged": True,
            "planning_scene_diff": planning_diff,
        }

    def qualify_motion_candidates(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        from agent.runtime.moveit_qualification import MoveItQualificationEngine

        engine = MoveItQualificationEngine(
            current_joint_state=self.qualification_joint_state,
            scene_revision=lambda: int(self.planning_scene.revision),
            compute_ik=self.qualification_compute_ik,
            check_state_validity=self.qualification_state_validity,
            plan_only=self.qualification_plan_only,
            workspace_filter=self.qualification_workspace_filter,
            clone_scene=self.qualification_clone_scene,
            apply_scene_transition=self.qualification_scene_transition,
            service_health_check=self.qualification_services_healthy,
            set_solver_mode=self.qualification_set_solver_mode,
        )
        bound_request = dict(request)
        funnel = request.get("funnel")
        funnel = funnel if isinstance(funnel, Mapping) else {}
        source = request.get("source")
        source = dict(source) if isinstance(source, Mapping) else {}
        configured_solver = _configured_qualification_solver_profile()
        source.setdefault("solver_profile", configured_solver)
        source.setdefault("solver_version", _qualification_solver_version(configured_solver))
        source.setdefault(
            "robot_model_sha256",
            _qualification_robot_model_sha256(self.config),
        )
        source.setdefault("scene_sha256", self.qualification_scene_sha256())
        source.setdefault(
            "authoritative_scene_sha256",
            str(getattr(self.planning_scene, "authoritative_scene_sha256", "")),
        )
        source.setdefault(
            "moveit_world_geometry_sha256",
            str(getattr(self.planning_scene, "world_geometry_sha256", "")),
        )
        source.setdefault(
            "moveit_attached_geometry_sha256",
            str(getattr(self.planning_scene, "attached_geometry_sha256", "")),
        )
        source.setdefault(
            "moveit_geometry_verified_ids",
            list(getattr(self.planning_scene, "geometry_verified_ids", ())),
        )
        map_id = str(funnel.get("capability_map_id") or "")
        if map_id:
            payload, error = self.qualification_capability_map(
                map_id=map_id,
                robot_model_sha256=_qualification_robot_model_sha256(self.config),
            )
            if payload is not None:
                source["capability_map"] = payload
            if error:
                source["capability_map_load_error"] = error
        bound_request["source"] = source
        return engine.qualify(bound_request)

    def start(self) -> None:
        if self.shared_executor:
            return
        self._thread = threading.Thread(
            target=self.executor.spin,
            name="openeta-gazebo-ros",
            daemon=True,
        )
        self._thread.start()

    def ros_time_s(self) -> float:
        now = self.node.get_clock().now()
        return float(now.nanoseconds) * 1e-9

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            actions_ready = (
                self.move_client.wait_for_server(timeout_sec=min(0.2, remaining))
                and self.gripper_client.wait_for_server(timeout_sec=min(0.2, remaining))
                and self.trajectory_client.wait_for_server(timeout_sec=min(0.2, remaining))
            )
            if actions_ready:
                services = tuple(
                    client
                    for client in (
                        self.controller_list_client,
                        self.controller_parameter_client,
                        self.state_validity_client,
                        getattr(self, "compute_ik_client", None),
                        getattr(self, "move_group_parameter_client", None),
                    )
                    if client is not None
                )
                if not all(
                    client.wait_for_service(timeout_sec=min(0.2, remaining)) for client in services
                ):
                    continue
                request = self.controller_service_type.Request()
                response = self._await(
                    self.controller_list_client.call_async(request), min(0.5, remaining)
                )
                states = {item.name: item.state for item in response.controller}
                required = {
                    "joint_state_broadcaster",
                    "rm_group_controller",
                    "parallel_gripper_controller",
                }
                if not required.issubset(
                    {name for name, state in states.items() if state == "active"}
                ):
                    continue
                parameter_request = self.controller_parameter_service_type.Request()
                parameter_request.names = ["enforce_command_limits"]
                parameter_response = self._await(
                    self.controller_parameter_client.call_async(parameter_request),
                    min(0.5, remaining),
                )
                if (
                    len(parameter_response.values) != 1
                    or parameter_response.values[0].bool_value is not True
                ):
                    continue
                self.state_source.wait_fresh(min(5.0, remaining))
                return
        if not self.move_client.server_is_ready():
            raise RuntimeError("MOVE_GROUP_UNAVAILABLE")
        if not self.gripper_client.server_is_ready():
            raise RuntimeError("GRIPPER_UNAVAILABLE")
        raise RuntimeError("ROS_NOT_READY")

    def _await(self, future: Any, timeout_s: float) -> Any:
        deadline = time.monotonic() + timeout_s
        with self._lock:
            self._pending.append(future)
        try:
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not future.done():
                raise TimeoutError
            error = future.exception()
            if error is not None:
                raise error
            return future.result()
        finally:
            with self._lock:
                if future in self._pending:
                    self._pending.remove(future)

    def apply_planning_scene(self, diff: dict[str, Any]) -> Mapping[str, Any]:
        """Apply one collision-scene diff and prove exact world/attached ids."""

        if not self.apply_scene_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("PLANNING_SCENE_APPLY_UNAVAILABLE")
        if not self.get_scene_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("PLANNING_SCENE_READBACK_UNAVAILABLE")
        scene = self.planning_scene_message_type()
        scene.is_diff = True
        # AttachedCollisionObject additions/removals live under robot_state.
        # Mark that nested message as a diff too; otherwise MoveIt interprets
        # the intentionally sparse state as a complete RobotState and rejects
        # the ApplyPlanningScene request.
        scene.robot_state.is_diff = True
        for object_id in diff.get("remove_world_ids", []):
            collision = self.collision_object_type()
            collision.id = str(object_id)
            collision.header.frame_id = self.config.base_link
            collision.operation = self.collision_object_type.REMOVE
            scene.world.collision_objects.append(collision)
        for spec in diff.get("world_objects", []):
            collision = self._collision_object_from_spec(spec)
            scene.world.collision_objects.append(collision)
        for object_id in diff.get("remove_attached_ids", []):
            attached = self.attached_collision_object_type()
            attached.object.id = str(object_id)
            attached.object.operation = self.collision_object_type.REMOVE
            scene.robot_state.attached_collision_objects.append(attached)
        for spec in diff.get("attached_objects", []):
            attached = self.attached_collision_object_type()
            attached.link_name = str(spec["link_name"])
            attached.touch_links = [str(value) for value in spec.get("touch_links", [])]
            attached.object = self._collision_object_from_spec(spec)
            scene.robot_state.attached_collision_objects.append(attached)
        allowed = diff.get("allowed_collisions")
        if isinstance(allowed, Mapping):
            # A sparse AllowedCollisionMatrix in a PlanningScene diff replaces
            # MoveIt's SRDF-derived matrix rather than patching it.  Read and
            # merge the live matrix first, otherwise adding target/fingertip
            # exceptions accidentally re-enables every adjacent-link
            # self-collision and all subsequent plans fail at the start state.
            components = self.planning_scene_components_type
            acm_request = self.get_scene_service_type.Request()
            acm_request.components.components = int(components.ALLOWED_COLLISION_MATRIX)
            acm_readback = self._await(self.get_scene_client.call_async(acm_request), 5.0)
            current_acm = acm_readback.scene.allowed_collision_matrix
            names, merged_rows = _merged_allowed_collision_rows(
                [str(value) for value in current_acm.entry_names],
                [list(row.enabled) for row in current_acm.entry_values],
                allowed,
            )
            scene.allowed_collision_matrix.entry_names = names
            for enabled in merged_rows:
                row = self.allowed_collision_entry_type()
                row.enabled = enabled
                scene.allowed_collision_matrix.entry_values.append(row)
            scene.allowed_collision_matrix.default_entry_names = list(
                current_acm.default_entry_names
            )
            scene.allowed_collision_matrix.default_entry_values = list(
                current_acm.default_entry_values
            )
        apply_request = self.apply_scene_service_type.Request()
        apply_request.scene = scene
        expected_world_geometry = _collision_message_geometry_map(
            [
                item
                for item in scene.world.collision_objects
                if list(getattr(item, "primitives", ()))
            ]
        )
        expected_attached_geometry = _collision_message_geometry_map(
            [
                item.object
                for item in scene.robot_state.attached_collision_objects
                if list(getattr(item.object, "primitives", ()))
            ]
        )
        applied = self._await(self.apply_scene_client.call_async(apply_request), 5.0)
        get_request = self.get_scene_service_type.Request()
        components = self.planning_scene_components_type
        get_request.components.components = (
            int(components.WORLD_OBJECT_NAMES)
            | int(components.WORLD_OBJECT_GEOMETRY)
            | int(components.ROBOT_STATE_ATTACHED_OBJECTS)
        )
        readback = self._await(self.get_scene_client.call_async(get_request), 5.0)
        actual_world_geometry = _collision_message_geometry_map(
            list(readback.scene.world.collision_objects)
        )
        actual_attached_geometry = _collision_message_geometry_map(
            [
                item.object
                for item in readback.scene.robot_state.attached_collision_objects
            ]
        )
        for expected, actual, label in (
            (expected_world_geometry, actual_world_geometry, "world"),
            (expected_attached_geometry, actual_attached_geometry, "attached"),
        ):
            mismatched = [
                object_id
                for object_id, record in expected.items()
                if actual.get(object_id) != record
            ]
            if mismatched:
                details = {
                    object_id: {
                        "expected": expected.get(object_id),
                        "actual": actual.get(object_id),
                    }
                    for object_id in mismatched
                }
                raise PlanningSceneError(
                    f"MoveIt {label} collision geometry readback mismatch: "
                    + json.dumps(details, sort_keys=True, separators=(",", ":"))
                )
        return {
            "applied": bool(getattr(applied, "success", False)),
            "world_ids": [item.id for item in readback.scene.world.collision_objects],
            "attached_ids": [
                item.object.id for item in readback.scene.robot_state.attached_collision_objects
            ],
            "world_geometry_sha256": _collision_geometry_manifest_sha256(
                actual_world_geometry
            ),
            "attached_geometry_sha256": _collision_geometry_manifest_sha256(
                actual_attached_geometry
            ),
            "geometry_verified_ids": sorted(
                {*expected_world_geometry, *expected_attached_geometry}
            ),
        }

    def _collision_object_from_spec(self, spec: Mapping[str, Any]) -> Any:
        collision = self.collision_object_type()
        collision.id = str(spec["id"])
        collision.header.frame_id = _moveit_scene_frame(
            spec.get("frame"), base_link=self.config.base_link
        )
        body_xyz = tuple(float(value) for value in spec["pose_xyz"])
        body_quat = _normalized_quaternion(tuple(float(value) for value in spec["pose_quat_xyzw"]))
        raw_primitives = spec.get("primitives")
        primitive_specs = (
            list(raw_primitives)
            if isinstance(raw_primitives, list) and raw_primitives
            else [
                {
                    "shape": "box",
                    "size_xyz": spec["size_xyz"],
                    "pose_xyz": [0.0, 0.0, 0.0],
                    "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ]
        )
        primitives = []
        poses = []
        for raw in primitive_specs:
            if not isinstance(raw, Mapping):
                raise PlanningSceneError("collision primitive is invalid")
            primitive = self.solid_primitive_type()
            shape = str(raw.get("shape") or "")
            if shape == "box":
                primitive.type = self.solid_primitive_type.BOX
                primitive.dimensions = [float(value) for value in raw["size_xyz"]]
            elif shape == "cylinder":
                primitive.type = self.solid_primitive_type.CYLINDER
                primitive.dimensions = [
                    float(raw["length"]),
                    float(raw["radius"]),
                ]
            else:
                raise PlanningSceneError("collision primitive shape is unsupported")
            primitive_xyz, primitive_quat = _child_world_pose(
                parent_xyz=body_xyz,  # type: ignore[arg-type]
                parent_quat_xyzw=body_quat,
                relative_xyz=tuple(float(value) for value in raw["pose_xyz"]),
                relative_quat_xyzw=tuple(float(value) for value in raw["pose_quat_xyzw"]),
            )
            pose = self.pose_type()
            pose.position.x, pose.position.y, pose.position.z = primitive_xyz
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ) = primitive_quat
            primitives.append(primitive)
            poses.append(pose)
        collision.primitives = primitives
        collision.primitive_poses = poses
        collision.operation = self.collision_object_type.ADD
        return collision

    def recover_start_state(self, state: Any, timeout_s: float) -> Mapping[str, Any]:
        assessment = assess_start_state_bounds(state)
        classification = assessment["classification"]
        if classification == "WITHIN_BOUNDS":
            return {
                "ok": True,
                "start_state_recovery": start_state_recovery_record(
                    assessment, status="NOT_REQUIRED"
                ),
            }
        if classification == "INVALID":
            return {
                "ok": False,
                "error_code": "START_STATE_INVALID",
                "motion_outcome": "failed",
                "start_state_recovery": start_state_recovery_record(assessment, status="REJECTED"),
            }

        started = self.ros_time_s()
        deadline = time.monotonic() + float(timeout_s)

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        def failed(reason_code: str, *, result_code: int | None = None):
            return {
                "ok": False,
                "error_code": "START_STATE_RECOVERY_FAILED",
                "motion_outcome": "failed",
                "action_started_ros_time_s": started,
                "action_completed_ros_time_s": self.ros_time_s(),
                "start_state_recovery": start_state_recovery_record(
                    assessment,
                    status="FAILED",
                    reason_code=reason_code,
                    attempted=True,
                    trajectory_result_code=result_code,
                ),
            }

        def unknown(reason_code: str):
            return {
                "ok": False,
                "error_code": "MOTION_OUTCOME_UNKNOWN",
                "motion_outcome": "unknown",
                "reconciliation_required": True,
                "action_started_ros_time_s": started,
                "action_completed_ros_time_s": self.ros_time_s(),
                "start_state_recovery": start_state_recovery_record(
                    assessment,
                    status="UNKNOWN",
                    reason_code=reason_code,
                    attempted=True,
                ),
            }

        candidate_positions = list(assessment["candidate_positions"])
        try:
            validity_request = self.state_validity_service_type.Request()
            _populate_state_validity_request(
                validity_request,
                candidate_positions,
                group_name=self.config.move_group,
            )
            validity_response = self._await(
                self.state_validity_client.call_async(validity_request),
                min(1.0, remaining()),
            )
        except Exception:
            return failed("STATE_VALIDITY_CHECK_FAILED")
        if not bool(validity_response.valid):
            return failed("RECOVERY_STATE_INVALID_OR_IN_COLLISION")

        goal = self.follow_trajectory_action_type.Goal()
        point = self.trajectory_point_type()
        duration_ns = int(START_STATE_RECOVERY_TRAJECTORY_S * 1_000_000_000)
        duration = self.duration_type(
            sec=duration_ns // 1_000_000_000,
            nanosec=duration_ns % 1_000_000_000,
        )
        _populate_recovery_trajectory_goal(goal, point, duration, candidate_positions)
        # Keep only samples produced after this recovery began.  The
        # controller result is itself the completion ACK, so the last
        # still-fresh sample from the trajectory is valid terminal evidence
        # even when the broadcaster does not publish again after the ACK.
        self.state_source.clear(min_ros_timestamp_s=started)
        try:
            handle = self._await(
                self.trajectory_client.send_goal_async(goal),
                min(1.0, remaining()),
            )
        except Exception:
            return failed("RECOVERY_TRAJECTORY_SEND_FAILED")
        if not handle.accepted:
            return failed("RECOVERY_TRAJECTORY_REJECTED")

        result_future = handle.get_result_async()
        try:
            wrapped = self._await(result_future, remaining())
        except TimeoutError:
            try:
                self._await(handle.cancel_goal_async(), min(1.0, remaining()))
            except Exception:
                pass
            self.state_source.clear()
            return unknown("RECOVERY_TRAJECTORY_TIMEOUT_UNCONFIRMED")
        except Exception:
            self.state_source.clear()
            return unknown("RECOVERY_TRAJECTORY_RESULT_UNAVAILABLE")

        result_code = int(wrapped.result.error_code)
        if result_code != int(self.follow_trajectory_action_type.Result.SUCCESSFUL):
            self.state_source.clear()
            return failed("RECOVERY_TRAJECTORY_FAILED", result_code=result_code)

        try:
            post_state = self.state_source.wait_fresh(remaining())
        except Exception:
            return failed("POST_RECOVERY_JOINT_STATE_MISSING", result_code=result_code)
        # Gazebo's joint limiter can report an endpoint one machine epsilon
        # beyond its decimal limit even after the accepted inset trajectory.
        # Apply the same certified numeric tolerance used for the preflight;
        # a zero-tolerance postcheck turns that harmless limiter round-off
        # into a false recovery failure.
        post_assessment = assess_start_state_bounds(post_state)
        post_timestamp = post_assessment.get("pre_joint_state_timestamp_s")
        if post_assessment["classification"] == "INVALID":
            return {
                **failed("POST_RECOVERY_STATE_OUT_OF_BOUNDS", result_code=result_code),
                "start_state_recovery": start_state_recovery_record(
                    assessment,
                    status="FAILED",
                    reason_code="POST_RECOVERY_STATE_OUT_OF_BOUNDS",
                    attempted=True,
                    post_joint_state_timestamp_s=post_timestamp,
                    trajectory_result_code=result_code,
                ),
            }
        return {
            "ok": True,
            "action_started_ros_time_s": started,
            "action_completed_ros_time_s": self.ros_time_s(),
            "start_state_recovery": start_state_recovery_record(
                assessment,
                status="RECOVERED",
                reason_code="NUMERIC_BOUNDS_RECOVERED",
                attempted=True,
                post_joint_state_timestamp_s=post_timestamp,
                trajectory_result_code=result_code,
            ),
        }

    def return_home(self, timeout_s: float) -> Mapping[str, Any]:
        """Command all arm joints to the workcell home configuration."""

        action_started = self.ros_time_s()
        goal = self.follow_trajectory_action_type.Goal()
        point = self.trajectory_point_type()
        duration_ns = int(2.0 * 1_000_000_000)
        duration = self.duration_type(
            sec=duration_ns // 1_000_000_000,
            nanosec=duration_ns % 1_000_000_000,
        )
        _populate_recovery_trajectory_goal(goal, point, duration, list(ARM_HOME_JOINT_POSITIONS))
        self.state_source.clear(min_ros_timestamp_s=action_started)
        handle = self._await(self.trajectory_client.send_goal_async(goal), min(5.0, timeout_s))
        if not handle.accepted:
            raise RuntimeError("HOME_TRAJECTORY_REJECTED")
        wrapped = self._await(handle.get_result_async(), timeout_s)
        if int(wrapped.result.error_code) != int(
            self.follow_trajectory_action_type.Result.SUCCESSFUL
        ):
            raise RuntimeError("HOME_TRAJECTORY_FAILED")
        return {"ok": True, "trajectory_result_code": int(wrapped.result.error_code)}

    def _execute_cached_l5_trajectory(
        self,
        *,
        cache_key: str,
        entry: Mapping[str, Any],
        action_started: float,
        timeout_s: float,
    ) -> Mapping[str, Any] | None:
        """Execute one single-use L5 trajectory, or fall back before motion."""

        trajectory = copy.deepcopy(entry.get("trajectory"))
        if trajectory is None or not getattr(trajectory, "points", None):
            return None
        if (
            int(entry.get("scene_revision", -1))
            != int(getattr(self.planning_scene, "revision", -2))
            or str(entry.get("scene_sha256") or "") != self._l5_scene_sha256()
        ):
            return None
        header = getattr(trajectory, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is not None:
            # A zero header asks the controller to start immediately. Reusing
            # the old plan timestamp would be rejected as an ancient goal.
            stamp.sec = 0
            stamp.nanosec = 0
        follow_goal = self.follow_trajectory_action_type.Goal()
        follow_goal.trajectory = trajectory
        self.state_source.clear(min_ros_timestamp_s=action_started)
        handle = self._await(
            self.trajectory_client.send_goal_async(follow_goal),
            min(5.0, timeout_s),
        )
        if not handle.accepted:
            # No trajectory started. Refresh state and let the ordinary
            # MoveGroup path plan again under the same safety contract.
            self.state_source.wait_fresh(min(2.0, timeout_s))
            return None
        try:
            wrapped = self._await(handle.get_result_async(), timeout_s)
        except TimeoutError:
            try:
                self._await(handle.cancel_goal_async(), min(2.0, timeout_s))
            except Exception:
                pass
            return {
                "ok": False,
                "error_code": "MOTION_OUTCOME_UNKNOWN",
                "motion_outcome": "unknown",
                "reconciliation_required": True,
                "execution_started": True,
                "planned_point_count": int(entry.get("point_count") or 0),
                "l5_trajectory_reused": True,
                "l5_trajectory_cache_key": cache_key,
            }
        result_code = int(wrapped.result.error_code)
        success = result_code == int(self.follow_trajectory_action_type.Result.SUCCESSFUL)
        return {
            "ok": success,
            "reached_goal": success,
            "error_code": None if success else "MOTION_EXECUTION_FAILED",
            "motion_outcome": "completed" if success else "failed",
            "execution_started": True,
            "planned_point_count": int(entry.get("point_count") or 0),
            "trajectory_result_code": result_code,
            **({"moveit_error_code": MOVEIT_CONTROL_FAILED} if not success else {}),
            "l5_trajectory_reused": True,
            "l5_trajectory_cache_key": cache_key,
            "l5_trajectory_scene_sha256": str(entry.get("scene_sha256") or ""),
        }

    def move(self, goal: dict, timeout_s: float) -> Mapping[str, Any]:
        from geometry_msgs.msg import Pose
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import (
            Constraints,
            JointConstraint,
            MoveItErrorCodes,
            OrientationConstraint,
            PositionConstraint,
        )
        from shape_msgs.msg import SolidPrimitive

        action_started = self.ros_time_s()
        cache_lookup: dict[str, Any] = {
            "status": "not_applicable",
            "reason": "plan_only_request",
        }

        def finish(payload: dict[str, Any]) -> dict[str, Any]:
            payload["action_started_ros_time_s"] = action_started
            completed = self.ros_time_s()
            payload["action_completed_ros_time_s"] = completed
            payload["motion_profile"] = str(goal.get("motion_profile") or "unloaded")
            payload["max_velocity_scaling_factor"] = float(
                goal.get("max_velocity_scaling_factor", 0.3)
            )
            payload["max_acceleration_scaling_factor"] = float(
                goal.get("max_acceleration_scaling_factor", 0.3)
            )
            payload["l5_trajectory_cache_status"] = str(cache_lookup.get("status") or "unknown")
            payload["l5_trajectory_cache_reason"] = str(cache_lookup.get("reason") or "unknown")
            for key in (
                "entry_count",
                "lookup_key",
                "requested_scene_revision",
                "current_scene_revision",
                "requested_start_max_delta_rad",
                "measured_start_max_delta_rad",
            ):
                if key in cache_lookup:
                    payload[f"l5_trajectory_cache_{key}"] = cache_lookup[key]
            # Preserve the latest still-fresh state received during execution.
            # Some real controllers stop publishing as soon as their result ACK
            # is emitted.  Requiring an additional sample after that ACK drops
            # truthful terminal evidence and converts a completed motion into a
            # JOINT_STATE_TIMEOUT.  The action-start ROS barrier below rejects
            # queued pre-action samples; GazeboController additionally verifies
            # the measured terminal pose against the requested target.
            return payload

        cached = None
        if goal.get("plan_only") is not True:
            cached, cache_lookup = self._take_matching_l5_trajectory(goal)
        if cached is not None:
            cache_key, entry = cached
            cached_result = self._execute_cached_l5_trajectory(
                cache_key=cache_key,
                entry=entry,
                action_started=action_started,
                timeout_s=timeout_s,
            )
            if cached_result is not None:
                return finish(dict(cached_result))
            cache_lookup = {
                "status": "miss",
                "reason": "cached_trajectory_rejected_before_execution",
            }

        # The start state read in GazeboController happened before this call.  Do
        # not permit it to double as post-action reconciliation state.
        self.state_source.clear(min_ros_timestamp_s=action_started)
        request = MoveGroup.Goal()
        request.request.group_name = goal["group_name"]
        # OMPL is stochastic: a single attempt can return a needlessly long
        # joint-space excursion (winding the redundant wrist onto its limits,
        # or swinging the open gripper into a grasp target mid-path), or fail
        # outright on an unlucky sample.  MoveGroup evaluates several attempts
        # and executes the shortest solution, which keeps both Cartesian hops
        # and physical approaches tidy without changing the goal contract.
        request.request.num_planning_attempts = int(goal.get("num_planning_attempts", 3))
        # Keep the action client's deadline strictly outside MoveIt's own
        # planning deadline. Equal deadlines race cancellation against a
        # terminal result and can trigger an invalid goal-state transition in
        # MoveIt Jazzy. Qualification uses the same planning budget as execution.
        planning_limit_s = float(goal.get("allowed_planning_time_s", 30.0))
        request.request.allowed_planning_time = min(planning_limit_s, max(0.1, timeout_s - 2.0))
        # The caller selects a load-state profile from immutable target
        # semantics. MoveIt still time-parameterizes the complete path and the
        # controller enforces the URDF limits.
        request.request.max_velocity_scaling_factor = float(
            goal.get("max_velocity_scaling_factor", 0.3)
        )
        request.request.max_acceleration_scaling_factor = float(
            goal.get("max_acceleration_scaling_factor", 0.3)
        )
        start_joint_state = goal.get("start_joint_state")
        if isinstance(start_joint_state, Mapping):
            request.request.start_state.is_diff = False
            request.request.start_state.joint_state.name = list(
                start_joint_state.get("names") or ARM_JOINTS
            )
            request.request.start_state.joint_state.position = [
                float(value) for value in start_joint_state.get("positions") or []
            ]
        else:
            request.request.start_state.is_diff = True
        fault_scenario = os.environ.get("OPENETA_ACCEPTANCE_PLACEMENT_FAULT", "")
        placement_id = str(goal.get("placement_candidate_id") or "")
        rejected_ids = getattr(self, "_acceptance_rejected_placement_ids", set())
        inject_rejection = False
        if placement_id and fault_scenario == "reject-first" and not rejected_ids:
            inject_rejection = True
        if inject_rejection:
            rejected_ids.add(placement_id)
            self._acceptance_rejected_placement_ids = rejected_ids
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = goal["target_pose"]["xyz"]
        if inject_rejection:
            # Acceptance-only fault fixture: MoveIt receives an unreachable
            # position constraint and must itself return an empty plan.  No
            # receipt or AnyPlace candidate is fabricated or rewritten.
            pose.position.z = 100.0
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = goal[
            "target_pose"
        ]["quat_xyzw"]
        primitive = SolidPrimitive(
            type=SolidPrimitive.BOX, dimensions=[2 * goal["position_tolerance_m"]] * 3
        )
        pc = PositionConstraint()
        pc.header.frame_id, pc.link_name, pc.weight = goal["base_frame"], goal["link_name"], 1.0
        pc.constraint_region.primitives = [primitive]
        pc.constraint_region.primitive_poses = [pose]
        oc = OrientationConstraint()
        oc.header.frame_id, oc.link_name, oc.orientation, oc.weight = (
            goal["base_frame"],
            goal["link_name"],
            pose.orientation,
            1.0,
        )
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = (
            oc.absolute_z_axis_tolerance
        ) = goal["orientation_tolerance_rad"]
        qualification_goal = goal.get("qualification_goal_joint_state")
        if isinstance(qualification_goal, Mapping) and not inject_rejection:
            names = list(qualification_goal.get("names") or ARM_JOINTS)
            positions = [float(value) for value in qualification_goal.get("positions") or []]
            if len(names) != len(positions) or not positions:
                return finish(
                    {
                        "ok": False,
                        "error_code": "MOTION_PLAN_FAILED",
                        "motion_outcome": "failed",
                        "planned_point_count": 0,
                        "execution_started": False,
                    }
                )
            joint_constraints = []
            for name, position in zip(names, positions, strict=True):
                constraint = JointConstraint()
                constraint.joint_name = str(name)
                constraint.position = position
                constraint.tolerance_above = QUALIFIED_JOINT_GOAL_TOLERANCE_RAD
                constraint.tolerance_below = QUALIFIED_JOINT_GOAL_TOLERANCE_RAD
                constraint.weight = 1.0
                joint_constraints.append(constraint)
            # Preserve the Beam-2 IK branch while still proving the model's
            # exact Cartesian contact/release target.  Joint constraints alone
            # permit MoveIt to stop at their edge, which can accumulate into a
            # several-millimetre TCP residual despite a successful plan.
            request.request.goal_constraints = [
                Constraints(
                    joint_constraints=joint_constraints,
                    position_constraints=[pc],
                    orientation_constraints=[oc],
                )
            ]
        else:
            request.request.goal_constraints = [
                Constraints(position_constraints=[pc], orientation_constraints=[oc])
            ]
        qualification_diff_value = goal.get("qualification_scene_diff")
        qualification_diff = (
            dict(qualification_diff_value) if isinstance(qualification_diff_value, Mapping) else {}
        )
        qualification_allowed = goal.get("qualification_allowed_collisions")
        if isinstance(qualification_allowed, Mapping):
            normalized_allowed = {
                str(key): [str(value) for value in values]
                for key, values in qualification_allowed.items()
                if isinstance(values, (list, tuple))
            }
            normalized_allowed = {
                key: sorted(set(values)) for key, values in normalized_allowed.items()
            }
            expected_allowed = (
                {
                    str(self.planning_scene.target_id): sorted(
                        str(link) for link in TARGET_TOUCH_LINKS
                    )
                }
                if self.planning_scene.target_id
                else {}
            )
            if normalized_allowed != expected_allowed:
                return finish(
                    {
                        "ok": False,
                        "error_code": "MOTION_PLAN_FAILED",
                        "reason": "qualification_collision_policy_mismatch",
                        "motion_outcome": "failed",
                        "planned_point_count": 0,
                        "execution_started": False,
                    }
                )
            qualification_diff["allowed_collisions"] = normalized_allowed
        if qualification_diff:
            request.planning_options.planning_scene_diff = self._qualification_scene_diff_message(
                qualification_diff
            )
        request.planning_options.plan_only = bool(goal.get("plan_only", False))
        send = self.move_client.send_goal_async(request)
        handle = self._await(send, min(5.0, timeout_s))
        if not handle.accepted:
            return finish(
                {
                    "ok": False,
                    "error_code": "MOTION_PLAN_FAILED",
                    "motion_outcome": "failed",
                    "planned_point_count": 0,
                    "execution_started": False,
                }
            )
        result_future = handle.get_result_async()
        try:
            wrapped = self._await(result_future, timeout_s)
        except TimeoutError:
            try:
                self._await(handle.cancel_goal_async(), 2.0)
                self._await(result_future, 2.0)
            except Exception:
                pass
            return finish(
                {
                    "ok": False,
                    "error_code": "MOTION_OUTCOME_UNKNOWN",
                    "motion_outcome": "unknown",
                    "planned_point_count": 0,
                    "execution_started": None,
                }
            )
        code = wrapped.result.error_code.val
        planned_joint_trajectory = getattr(
            wrapped.result.planned_trajectory,
            "joint_trajectory",
            None,
        )
        planned_points = list(getattr(planned_joint_trajectory, "points", ()))
        if code == MoveItErrorCodes.SUCCESS:
            end_joint_state = None
            trajectory_points = []
            trajectory_cache_key = None
            if planned_points:
                trajectory_points = [
                    {"positions": [float(value) for value in point.positions]}
                    for point in planned_points
                ]
                end_joint_state = {
                    "names": list(
                        getattr(
                            wrapped.result.planned_trajectory.joint_trajectory,
                            "joint_names",
                            ARM_JOINTS,
                        )
                    ),
                    "positions": trajectory_points[-1]["positions"],
                }
                if request.planning_options.plan_only:
                    trajectory_cache_key = self._store_l5_trajectory(
                        goal=goal,
                        trajectory=planned_joint_trajectory,
                        point_count=len(planned_points),
                    )
            return finish(
                {
                    "ok": True,
                    "reached_goal": not request.planning_options.plan_only,
                    "plan_only": request.planning_options.plan_only,
                    "motion_outcome": "planned"
                    if request.planning_options.plan_only
                    else "completed",
                    "planned_point_count": len(planned_points),
                    "execution_started": not request.planning_options.plan_only,
                    **(
                        {
                            "trajectory_points": trajectory_points,
                            "end_joint_state": end_joint_state,
                            "l5_trajectory_cache_stored": (trajectory_cache_key is not None),
                            **(
                                {"l5_trajectory_cache_key": trajectory_cache_key}
                                if trajectory_cache_key is not None
                                else {}
                            ),
                        }
                        if request.planning_options.plan_only
                        else {}
                    ),
                }
            )
        planning_failures = {
            MoveItErrorCodes.PLANNING_FAILED,
            MoveItErrorCodes.INVALID_MOTION_PLAN,
            MoveItErrorCodes.START_STATE_IN_COLLISION,
            MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS,
            MoveItErrorCodes.GOAL_IN_COLLISION,
            MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS,
            MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED,
            MoveItErrorCodes.INVALID_GROUP_NAME,
            MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS,
            MoveItErrorCodes.INVALID_ROBOT_STATE,
            MoveItErrorCodes.INVALID_LINK_NAME,
            MoveItErrorCodes.INVALID_OBJECT_NAME,
            MoveItErrorCodes.FRAME_TRANSFORM_FAILURE,
            MoveItErrorCodes.START_STATE_INVALID,
            MoveItErrorCodes.GOAL_STATE_INVALID,
            MoveItErrorCodes.UNRECOGNIZED_GOAL_TYPE,
            MoveItErrorCodes.NO_IK_SOLUTION,
        }
        return finish(
            _move_group_failure_result(
                int(code),
                len(planned_points),
                plan_only=bool(request.planning_options.plan_only),
                planning_failure_codes={int(value) for value in planning_failures},
                timed_out_code=int(MoveItErrorCodes.TIMED_OUT),
                generic_failure_code=int(MoveItErrorCodes.FAILURE),
            )
        )

    def gripper(self, position: float, timeout_s: float) -> Mapping[str, Any]:
        from control_msgs.action import ParallelGripperCommand

        action_started = self.ros_time_s()
        wall_started = time.monotonic()

        def finish(payload: dict[str, Any]) -> dict[str, Any]:
            payload["action_started_ros_time_s"] = action_started
            completed = self.ros_time_s()
            payload["action_completed_ros_time_s"] = completed
            # Diagnostics only: wall-clock duration and terminal status do
            # not affect the strict success predicate below.
            payload["wall_elapsed_ms"] = round((time.monotonic() - wall_started) * 1000, 3)
            return payload

        self.state_source.clear(min_ros_timestamp_s=action_started)
        goal = ParallelGripperCommand.Goal()
        goal.command.name = ["gripper_left_finger_joint"]
        goal.command.position = [float(position)]
        handle = self._await(self.gripper_client.send_goal_async(goal), min(5.0, timeout_s))
        if not handle.accepted:
            return finish(
                {
                    "ok": False,
                    "error_code": "GRIPPER_FAILED",
                    "terminal_status": "rejected",
                }
            )
        try:
            wrapped = self._await(handle.get_result_async(), timeout_s)
        except TimeoutError:
            try:
                self._await(handle.cancel_goal_async(), min(2.0, timeout_s))
            except Exception:
                pass
            return finish(
                {
                    "ok": False,
                    "reached_goal": False,
                    "stalled": False,
                    "error_code": "GRIPPER_TIMEOUT",
                    "terminal_status": "timed_out",
                }
            )
        except Exception:
            return finish(
                {
                    "ok": False,
                    "reached_goal": False,
                    "stalled": False,
                    "error_code": "GRIPPER_FAILED",
                    "terminal_status": "result_error",
                }
            )
        result = wrapped.result
        reached_goal = bool(result.reached_goal)
        stalled = bool(result.stalled)
        terminal_state = getattr(result, "state", None)
        terminal_joint_state = None
        if terminal_state is not None:
            names = [str(value) for value in getattr(terminal_state, "name", ())]
            positions = [float(value) for value in getattr(terminal_state, "position", ())]
            velocities = [float(value) for value in getattr(terminal_state, "velocity", ())]
            if (
                names
                and len(names) == len(positions) == len(velocities)
                and all(math.isfinite(value) for value in (*positions, *velocities))
            ):
                terminal_joint_state = {
                    "names": names,
                    "positions": positions,
                    "velocities": velocities,
                    "maximum_absolute_velocity_rad_s": max(abs(value) for value in velocities),
                }
        terminal_status_code = getattr(wrapped, "status", None)
        terminal_succeeded = gripper_terminal_succeeded(terminal_status_code)
        closed_target = float(self.config.gripper_position(0))
        stall_is_valid_for_command = bool(
            self.allow_stalling
            and math.isclose(
                float(position),
                closed_target,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        ok = gripper_action_success(
            reached_goal=reached_goal,
            stalled=stalled,
            allow_stalling=stall_is_valid_for_command,
            terminal_succeeded=terminal_succeeded,
        )
        return finish(
            {
                "ok": ok,
                "reached_goal": reached_goal,
                "stalled": stalled,
                "stall_accepted_for_command": stall_is_valid_for_command,
                "error_code": None if ok else "GRIPPER_FAILED",
                "terminal_status": "succeeded" if terminal_succeeded else "not_succeeded",
                "terminal_status_code": (
                    int(terminal_status_code)
                    if isinstance(terminal_status_code, int)
                    and not isinstance(terminal_status_code, bool)
                    else None
                ),
                **(
                    {"terminal_gripper_joint_state": terminal_joint_state}
                    if terminal_joint_state is not None
                    else {}
                ),
            }
        )

    def cancel_pending(self) -> None:
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        for future in pending:
            try:
                future.cancel()
            except Exception:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel_pending()
        try:
            if self.shared_executor:
                self.executor.remove_node(self.node)
            else:
                self.executor.shutdown(timeout_sec=2.0)
        finally:
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self.node.destroy_node()
            if self.owns_context and self.rclpy.ok():
                self.rclpy.shutdown()
