"""Jazzy MoveIt and ros2_control adapter for motion-control.

All ROS imports occur inside :meth:`RosGazeboControllerFactory.create`; importing
OpenETA on a non-ROS test machine remains supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping
from xml.etree import ElementTree

from .robot_control import (
    ARM_JOINTS,
    ARM_JOINT_BOUNDS,
    GazeboControlConfig,
    GazeboController,
    START_STATE_RECOVERY_TRAJECTORY_S,
    assess_start_state_bounds,
    make_move_group_goal,
    robot_state_from_sources,
    start_state_recovery_record,
)
from .planning_scene import (
    TARGET_TOUCH_LINKS,
    CollisionBox,
    PlanningSceneError,
    PlanningSceneSynchronizer,
)


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
    urdf = package / "urdf" / (
        "rm75_robotiq2f85_pickplace.urdf.xacro"
        if pickplace
        else "rm75_robotiq2f85.urdf.xacro"
    )
    srdf = package / "config" / (
        "rm75_robotiq2f85_pickplace.srdf"
        if pickplace
        else "rm75_robotiq2f85.srdf"
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

    _, srdf = _qualification_model_paths(config)
    urdf_bytes = _expanded_qualification_urdf(config)
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

    if len(map_id) != 64 or any(
        character not in "0123456789abcdef" for character in map_id
    ):
        raise ValueError("capability map ID must be a lowercase SHA-256 digest")
    override = os.environ.get("OPENETA_CAPABILITY_MAP_PATH", "").strip()
    repo_root = config.ros_workspace.parents[2]
    path = (
        Path(override)
        if override
        else repo_root / "config" / "capability_maps" / f"{map_id}.json"
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


def _stamp_seconds(stamp: Any) -> float | None:
    if stamp is None:
        return None
    return float(int(getattr(stamp, "sec", 0))) + float(
        int(getattr(stamp, "nanosec", 0))
    ) * 1e-9


def _moveit_scene_frame(frame: object, *, base_link: str) -> str:
    """Map the co-located Gazebo world frame into MoveIt's fixed root frame."""

    value = str(frame or "")
    # This profile spawns the robot at the Gazebo world origin and its SRDF
    # has a fixed base_link root, so MoveIt has no separate `world` TF frame.
    # World collision poses are numerically base_link poses under that asset
    # contract; attached-object link frames must pass through unchanged.
    return base_link if value in {"", "world"} else value


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
    request: Any, candidate_positions: list[float], *, group_name: str
) -> None:
    request.group_name = group_name
    request.robot_state.is_diff = True
    request.robot_state.joint_state.name = list(ARM_JOINTS)
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
    additions: Mapping[str, Any],
) -> tuple[list[str], list[list[bool]]]:
    """Merge object/link allowances without erasing the SRDF self matrix."""

    enabled_pairs: set[tuple[str, str]] = set()
    for row_index, row_name in enumerate(current_names):
        values = current_rows[row_index] if row_index < len(current_rows) else []
        for column_index, column_name in enumerate(current_names):
            if column_index < len(values) and bool(values[column_index]):
                enabled_pairs.add(tuple(sorted((row_name, column_name))))
    for object_id, links in additions.items():
        for link in links:
            enabled_pairs.add(tuple(sorted((str(object_id), str(link)))))
    names = sorted(
        set(current_names)
        | {str(key) for key in additions}
        | {str(link) for links in additions.values() for link in links}
    )
    return names, [
        [tuple(sorted((row_name, column))) in enabled_pairs for column in names]
        for row_name in names
    ]


class RosGazeboStateSource:
    def __init__(self, node: Any, tf_buffer: Any, *, config: GazeboControlConfig, freshness_s: float = 2.0):
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
                float(min_ros_timestamp_s)
                if min_ros_timestamp_s is not None
                else None
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
        state = robot_state_from_sources(joint, {
            f"{self.config.base_link}->{self.config.mount_child}": {
                "xyz": [transform.translation.x, transform.translation.y, transform.translation.z],
                "quat_xyzw": [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w],
            }
        }, config=self.config)
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

    def sync_planning_scene_reset(self, config: Any) -> int:
        table = CollisionBox(config.table_id, tuple(config.table_size_m), tuple(config.table_pose_xyz))
        distractor_size = tuple(config.distractor_size_m)
        if len(distractor_size) == 2:
            distractor_size = (distractor_size[0], distractor_size[0], distractor_size[1])
        revision = self.planning_scene.reset(
            table=table,
            distractor=CollisionBox(
                config.distractor_id,
                distractor_size,
                tuple(config.distractor_initial_xyz),
            ),
            target=CollisionBox(
                config.target_id,
                tuple(config.target_size_m),
                tuple(config.target_initial_xyz),
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
            target=CollisionBox(
                config.target_id,
                tuple(config.target_size_m),
                target_xyz,
                target_quat_xyzw,
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
    ) -> int:
        revision = self.planning_scene.update_world_target(
            target=CollisionBox(
                config.target_id,
                tuple(config.target_size_m),
                target_xyz,
                target_quat_xyzw,
            )
        )
        self._require_current_planning_state_valid()
        self.runtime.scene_revision = revision
        self.runtime.planning_scene_ready = self.planning_scene.ready
        return revision

    def _require_current_planning_state_valid(self) -> None:
        validity = self.runtime.current_state_validity(timeout_s=3.0)
        self.runtime.planning_scene_validation = validity
        if validity.get("valid") is True:
            return
        self.planning_scene.ready = False
        pairs = validity.get("collision_pairs") or []
        raise PlanningSceneError(
            "planning-scene current state is invalid; collision_pairs="
            + repr(pairs)
        )

    def sync_planning_scene_detach(
        self,
        config: Any,
        *,
        target_xyz: tuple[float, float, float],
        target_quat_xyzw: tuple[float, float, float, float],
    ) -> int:
        revision = self.planning_scene.detach_target(
            target=CollisionBox(
                config.target_id,
                tuple(config.target_size_m),
                target_xyz,
                target_quat_xyzw,
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

    def create(self, config: GazeboControlConfig | None = None, *, context: Any | None = None,
               executor: Any | None = None) -> RosGazeboController:
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
        subscription = node.create_subscription(JointState, "/joint_states", source.joint_state_callback, 10)
        move_client = ActionClient(node, MoveGroup, "/move_action")
        gripper_client = ActionClient(node, ParallelGripperCommand, "/parallel_gripper_controller/gripper_cmd")
        trajectory_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/rm_group_controller/follow_joint_trajectory",
        )
        controller_list_client = node.create_client(ListControllers, "/controller_manager/list_controllers")
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
            rclpy=rclpy, node=node, executor=executor, state_source=source,
            move_client=move_client, gripper_client=gripper_client,
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
            listener=listener, subscription=subscription,
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
            solid_primitive_type=__import__("shape_msgs.msg", fromlist=["SolidPrimitive"]).SolidPrimitive,
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
        self._closed = False

    def current_state_validity(self, *, timeout_s: float) -> Mapping[str, Any]:
        """Read back MoveIt's verdict and collision pairs for the live arm state."""

        state = self.state_source.wait_fresh(timeout_s)
        request = self.state_validity_service_type.Request()
        _populate_state_validity_request(
            request,
            [float(value) for value in state.joint_positions[: len(ARM_JOINTS)]],
            group_name=self.config.move_group,
        )
        response = self._await(
            self.state_validity_client.call_async(request), timeout_s
        )
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
                if getattr(contact, "contact_body_1", "")
                or getattr(contact, "contact_body_2", "")
            }
        )
        return {
            "valid": bool(response.valid),
            "collision_pairs": [list(pair) for pair in pairs],
            "joint_state_timestamp_s": state.metadata.get("joint_state_timestamp_s"),
        }

    def qualification_joint_state(self) -> Mapping[str, Any]:
        state = self.state_source.wait_fresh(3.0)
        lower = [float(item[1]) for item in ARM_JOINT_BOUNDS]
        upper = [float(item[2]) for item in ARM_JOINT_BOUNDS]
        positions = [
            float(value) for value in state.joint_positions[: len(ARM_JOINTS)]
        ]
        jacobian_quality = self.qualification_joint_quality(
            {"names": list(ARM_JOINTS), "positions": positions}
        )
        robot_hash = getattr(self, "_qualification_robot_model_hash", None)
        if robot_hash is None:
            robot_hash = _qualification_robot_model_sha256(self.config)
            self._qualification_robot_model_hash = robot_hash
        return {
            "names": list(ARM_JOINTS),
            "positions": positions,
            "joint_limits": {"lower": lower, "upper": upper},
            "home_joint_state": {
                "names": list(ARM_JOINTS),
                "positions": [(lo + hi) / 2.0 for lo, hi in zip(lower, upper)],
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
            "jacobian_quality_available": jacobian_quality.get("ok") is True,
            "jacobian_quality_error": jacobian_quality.get("error"),
        }

    def qualification_joint_quality(
        self, joint_state: Mapping[str, Any]
    ) -> Mapping[str, Any]:
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
        mode_parameter.name = (
            f"robot_description_kinematics.{self.config.move_group}.mode"
        )
        mode_parameter.value.type = self.parameter_type.PARAMETER_STRING
        mode_parameter.value.string_value = mode
        displacement_parameter = self.interface_parameter_type()
        displacement_parameter.name = (
            "robot_description_kinematics."
            f"{self.config.move_group}.minimal_displacement_weight"
        )
        displacement_parameter.value.type = self.parameter_type.PARAMETER_DOUBLE
        displacement_parameter.value.double_value = 0.0 if mode == "global" else 0.001
        request.parameters = [mode_parameter, displacement_parameter]
        try:
            response = self._await(
                self.move_group_parameter_client.call_async(request), 1.0
            )
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
        return not math.isfinite(reach) or math.sqrt(sum(value * value for value in values)) <= reach

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
        if not isinstance(xyz, list) or len(xyz) != 3 or not isinstance(quat, list) or len(quat) != 4:
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
        ik.timeout.nanosec = int(
            (seed_timeout_s - int(seed_timeout_s)) * 1_000_000_000
        )
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
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = [float(v) for v in quat]
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
            self.qualification_joint_quality(
                {"names": list(ARM_JOINTS), "positions": ordered}
            )
            if ok
            else {}
        )
        return {
            "ok": ok,
            "infrastructure_error": ok and quality.get("ok") is not True,
            "reason": (
                "jacobian_quality_unavailable"
                if ok and quality.get("ok") is not True
                else None
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

    def qualification_state_validity(
        self, joint_state: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from agent.runtime.moveit_qualification import STATE_VALIDITY_TIMEOUT_S

        request = self.state_validity_service_type.Request()
        _populate_state_validity_request(
            request,
            [float(value) for value in joint_state.get("positions") or []],
            group_name=self.config.move_group,
        )
        scene_diff = joint_state.get("qualification_scene_diff")
        if isinstance(scene_diff, Mapping):
            diff_message = self._qualification_scene_diff_message(scene_diff)
            request.robot_state.is_diff = True
            request.robot_state.attached_collision_objects = list(
                diff_message.robot_state.attached_collision_objects
            )
        response = self._await(
            self.state_validity_client.call_async(request), STATE_VALIDITY_TIMEOUT_S
        )
        pairs = sorted(
            {
                tuple(sorted((str(c.contact_body_1), str(c.contact_body_2))))
                for c in getattr(response, "contacts", ())
            }
        )
        return {"valid": bool(response.valid), "collision_pairs": [list(p) for p in pairs]}

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
        qualification_goal_joint_state = pose_target.pop(
            "qualification_goal_joint_state", None
        )
        goal = make_move_group_goal(
            pose_target, config=self.config, tolerances=target
        )
        goal.update(
            {
                "plan_only": True,
                "start_joint_state": dict(start),
                "allowed_planning_time_s": planning_time_s,
                "num_planning_attempts": planning_attempts,
            }
        )
        if isinstance(qualification_goal_joint_state, Mapping):
            goal["qualification_goal_joint_state"] = dict(
                qualification_goal_joint_state
            )
        scene_diff = target.get("qualification_scene_diff")
        if isinstance(scene_diff, Mapping):
            goal["qualification_scene_diff"] = dict(scene_diff)
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
            scene.world.collision_objects.append(
                self._collision_object_from_spec(spec)
            )
        for object_id in diff.get("remove_attached_ids", []):
            attached = self.attached_collision_object_type()
            attached.object.id = str(object_id)
            attached.object.operation = self.collision_object_type.REMOVE
            scene.robot_state.attached_collision_objects.append(attached)
        for spec in diff.get("attached_objects", []):
            attached = self.attached_collision_object_type()
            attached.link_name = str(spec["link_name"])
            attached.touch_links = [
                str(value) for value in spec.get("touch_links", [])
            ]
            attached.object = self._collision_object_from_spec(spec)
            scene.robot_state.attached_collision_objects.append(attached)
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
                key: dict(value)
                for key, value in self.planning_scene.world_specs.items()
            },
            "attached_specs": {
                key: dict(value)
                for key, value in self.planning_scene.attached_specs.items()
            },
            "target_id": self.planning_scene.target_id,
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
        destination_center = getattr(self.config, "destination_center_xy", None)
        destination_size = getattr(self.config, "destination_size_xy_m", None)
        support_z = getattr(self.config, "table_top_z_m", None)
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
                "support_object_id": str(getattr(self.config, "table_id", "")),
                "support_height_tolerance_m": float(
                    getattr(self.config, "placement_center_height_tolerance_m", 0.01)
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
                relative_xyz = tuple(
                    float(value)
                    for value in predicted.get("translation_xyz", [])
                )
                relative_quat = tuple(
                    float(value) for value in predicted.get("quat_xyzw", [])
                )
                if len(relative_xyz) != 3 or len(relative_quat) != 4:
                    return {"ok": False, "reason": "predicted_attachment_invalid"}
            else:
                relative_xyz, relative_quat = _relative_pose(
                    child_xyz=tuple(float(value) for value in spec["pose_xyz"]),
                    child_quat_xyzw=tuple(
                        float(value) for value in spec["pose_quat_xyzw"]
                    ),
                    parent_xyz=tuple(float(value) for value in xyz),
                    parent_quat_xyzw=tuple(float(value) for value in quat),
                )
            attached = {
                **dict(spec),
                "frame": self.config.mount_child,
                "pose_xyz": list(relative_xyz),
                "pose_quat_xyzw": list(relative_quat),
                "link_name": self.config.mount_child,
                "touch_links": list(self.planning_scene.attached_specs.get(target_id, {}).get("touch_links") or ()),
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
                relative_quat_xyzw=tuple(
                    float(value) for value in attached["pose_quat_xyzw"]
                ),
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
        source.setdefault(
            "solver_version", _qualification_solver_version(configured_solver)
        )
        source.setdefault(
            "robot_model_sha256",
            _qualification_robot_model_sha256(self.config),
        )
        source.setdefault("scene_sha256", self.qualification_scene_sha256())
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
                and self.gripper_client.wait_for_server(
                    timeout_sec=min(0.2, remaining)
                )
                and self.trajectory_client.wait_for_server(
                    timeout_sec=min(0.2, remaining)
                )
            )
            if actions_ready:
                services = tuple(client for client in (
                    self.controller_list_client,
                    self.controller_parameter_client,
                    self.state_validity_client,
                    getattr(self, "compute_ik_client", None),
                    getattr(self, "move_group_parameter_client", None),
                ) if client is not None)
                if not all(
                    client.wait_for_service(timeout_sec=min(0.2, remaining))
                    for client in services
                ):
                    continue
                request = self.controller_service_type.Request()
                response = self._await(self.controller_list_client.call_async(request), min(0.5, remaining))
                states = {item.name: item.state for item in response.controller}
                required = {"joint_state_broadcaster", "rm_group_controller", "parallel_gripper_controller"}
                if not required.issubset({name for name, state in states.items() if state == "active"}):
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
            acm_request.components.components = int(
                components.ALLOWED_COLLISION_MATRIX
            )
            acm_readback = self._await(
                self.get_scene_client.call_async(acm_request), 5.0
            )
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
        applied = self._await(self.apply_scene_client.call_async(apply_request), 5.0)
        get_request = self.get_scene_service_type.Request()
        components = self.planning_scene_components_type
        get_request.components.components = int(components.WORLD_OBJECT_NAMES) | int(
            components.ROBOT_STATE_ATTACHED_OBJECTS
        )
        readback = self._await(self.get_scene_client.call_async(get_request), 5.0)
        return {
            "applied": bool(getattr(applied, "success", False)),
            "world_ids": [item.id for item in readback.scene.world.collision_objects],
            "attached_ids": [
                item.object.id for item in readback.scene.robot_state.attached_collision_objects
            ],
        }

    def _collision_object_from_spec(self, spec: Mapping[str, Any]) -> Any:
        collision = self.collision_object_type()
        collision.id = str(spec["id"])
        collision.header.frame_id = _moveit_scene_frame(
            spec.get("frame"), base_link=self.config.base_link
        )
        primitive = self.solid_primitive_type()
        primitive.type = self.solid_primitive_type.BOX
        primitive.dimensions = [float(value) for value in spec["size_xyz"]]
        pose = self.pose_type()
        pose.position.x, pose.position.y, pose.position.z = [
            float(value) for value in spec["pose_xyz"]
        ]
        quaternion = spec.get("pose_quat_xyzw", (0.0, 0.0, 0.0, 1.0))
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = _normalized_quaternion(tuple(float(value) for value in quaternion))
        collision.primitives = [primitive]
        collision.primitive_poses = [pose]
        collision.operation = self.collision_object_type.ADD
        return collision

    def recover_start_state(
        self, state: Any, timeout_s: float
    ) -> Mapping[str, Any]:
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
                "start_state_recovery": start_state_recovery_record(
                    assessment, status="REJECTED"
                ),
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
        _populate_recovery_trajectory_goal(
            goal, point, duration, candidate_positions
        )
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
                **failed(
                    "POST_RECOVERY_STATE_OUT_OF_BOUNDS", result_code=result_code
                ),
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
        """Command all arm joints to zero through the trajectory controller."""

        action_started = self.ros_time_s()
        goal = self.follow_trajectory_action_type.Goal()
        point = self.trajectory_point_type()
        duration_ns = int(2.0 * 1_000_000_000)
        duration = self.duration_type(
            sec=duration_ns // 1_000_000_000,
            nanosec=duration_ns % 1_000_000_000,
        )
        _populate_recovery_trajectory_goal(
            goal, point, duration, [0.0] * len(ARM_JOINTS)
        )
        self.state_source.clear(min_ros_timestamp_s=action_started)
        handle = self._await(self.trajectory_client.send_goal_async(goal), min(5.0, timeout_s))
        if not handle.accepted:
            raise RuntimeError("HOME_TRAJECTORY_REJECTED")
        wrapped = self._await(handle.get_result_async(), timeout_s)
        if int(wrapped.result.error_code) != int(self.follow_trajectory_action_type.Result.SUCCESSFUL):
            raise RuntimeError("HOME_TRAJECTORY_FAILED")
        return {"ok": True, "trajectory_result_code": int(wrapped.result.error_code)}

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

        def finish(payload: dict[str, Any]) -> dict[str, Any]:
            payload["action_started_ros_time_s"] = action_started
            completed = self.ros_time_s()
            payload["action_completed_ros_time_s"] = completed
            # Preserve the latest still-fresh state received during execution.
            # Some real controllers stop publishing as soon as their result ACK
            # is emitted.  Requiring an additional sample after that ACK drops
            # truthful terminal evidence and converts a completed motion into a
            # JOINT_STATE_TIMEOUT.  The action-start ROS barrier below rejects
            # queued pre-action samples; GazeboController additionally verifies
            # the measured terminal pose against the requested target.
            return payload

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
        request.request.allowed_planning_time = min(
            planning_limit_s, max(0.1, timeout_s - 2.0)
        )
        # The 7-DOF arm is redundant, so OMPL may legitimately choose a
        # several-radian joint-space path even for the small Cartesian moves
        # used by motion-control.  At 10% scaling those collision-checked trajectories can
        # exceed the public action deadline on a software-rendered simulator
        # (which commonly runs below real time).  Thirty percent remains a
        # conservative MoveIt limit while keeping every valid planned path
        # inside the control contract's timeout.  Grasp-critical carry moves
        # may lower the scaling further through the goal to keep a jointed or
        # caged payload quiet.
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
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = goal["target_pose"]["quat_xyzw"]
        primitive = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[2 * goal["position_tolerance_m"]] * 3)
        pc = PositionConstraint()
        pc.header.frame_id, pc.link_name, pc.weight = goal["base_frame"], goal["link_name"], 1.0
        pc.constraint_region.primitives = [primitive]
        pc.constraint_region.primitive_poses = [pose]
        oc = OrientationConstraint()
        oc.header.frame_id, oc.link_name, oc.orientation, oc.weight = goal["base_frame"], goal["link_name"], pose.orientation, 1.0
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = goal["orientation_tolerance_rad"]
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
                constraint.tolerance_above = 0.005
                constraint.tolerance_below = 0.005
                constraint.weight = 1.0
                joint_constraints.append(constraint)
            request.request.goal_constraints = [
                Constraints(joint_constraints=joint_constraints)
            ]
        else:
            request.request.goal_constraints = [
                Constraints(
                    position_constraints=[pc], orientation_constraints=[oc]
                )
            ]
        qualification_diff = goal.get("qualification_scene_diff")
        if isinstance(qualification_diff, Mapping):
            request.planning_options.planning_scene_diff = (
                self._qualification_scene_diff_message(qualification_diff)
            )
        request.planning_options.plan_only = bool(goal.get("plan_only", False))
        send = self.move_client.send_goal_async(request)
        handle = self._await(send, min(5.0, timeout_s))
        if not handle.accepted:
            return finish({
                "ok": False,
                "error_code": "MOTION_PLAN_FAILED",
                "motion_outcome": "failed",
                "planned_point_count": 0,
                "execution_started": False,
            })
        result_future = handle.get_result_async()
        try:
            wrapped = self._await(result_future, timeout_s)
        except TimeoutError:
            try:
                self._await(handle.cancel_goal_async(), 2.0)
                self._await(result_future, 2.0)
            except Exception:
                pass
            return finish({
                "ok": False,
                "error_code": "MOTION_OUTCOME_UNKNOWN",
                "motion_outcome": "unknown",
                "planned_point_count": 0,
                "execution_started": None,
            })
        code = wrapped.result.error_code.val
        planned_points = list(
            getattr(
                getattr(wrapped.result.planned_trajectory, "joint_trajectory", None),
                "points",
                (),
            )
        )
        if code == MoveItErrorCodes.SUCCESS:
            end_joint_state = None
            trajectory_points = []
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
            return finish({
                "ok": True,
                "reached_goal": not request.planning_options.plan_only,
                "plan_only": request.planning_options.plan_only,
                "motion_outcome": "planned" if request.planning_options.plan_only else "completed",
                "planned_point_count": len(planned_points),
                "execution_started": not request.planning_options.plan_only,
                **(
                    {
                        "trajectory_points": trajectory_points,
                        "end_joint_state": end_joint_state,
                    }
                    if request.planning_options.plan_only
                    else {}
                ),
            })
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
        # MoveGroup sometimes collapses a goal-sampling failure to the generic
        # FAILURE code. An empty returned trajectory proves execution never
        # started, so classify it as a planning rejection.
        if code in planning_failures or (
            code == MoveItErrorCodes.FAILURE and not planned_points
        ):
            return finish({
                "ok": False,
                "error_code": "MOTION_PLAN_FAILED",
                "motion_outcome": "failed",
                "moveit_error_code": int(code),
                "planned_point_count": len(planned_points),
                "execution_started": False,
            })
        if code == MoveItErrorCodes.TIMED_OUT:
            return finish({
                "ok": False,
                "error_code": "MOTION_EXECUTION_TIMEOUT",
                "motion_outcome": "failed",
                "moveit_error_code": int(code),
                "planned_point_count": len(planned_points),
                "execution_started": bool(planned_points),
            })
        return finish({
            "ok": False,
            "error_code": "MOTION_EXECUTION_FAILED",
            "motion_outcome": "failed",
            "moveit_error_code": int(code),
            "planned_point_count": len(planned_points),
            "execution_started": bool(planned_points),
        })

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
            return finish({
                "ok": False,
                "reached_goal": False,
                "stalled": False,
                "error_code": "GRIPPER_TIMEOUT",
                "terminal_status": "timed_out",
            })
        except Exception:
            return finish({
                "ok": False,
                "reached_goal": False,
                "stalled": False,
                "error_code": "GRIPPER_FAILED",
                "terminal_status": "result_error",
            })
        result = wrapped.result
        reached_goal = bool(result.reached_goal)
        stalled = bool(result.stalled)
        terminal_status_code = getattr(wrapped, "status", None)
        terminal_succeeded = gripper_terminal_succeeded(terminal_status_code)
        ok = gripper_action_success(
            reached_goal=reached_goal,
            stalled=stalled,
            allow_stalling=bool(self.allow_stalling),
            terminal_succeeded=terminal_succeeded,
        )
        return finish({
            "ok": ok,
            "reached_goal": reached_goal,
            "stalled": stalled,
            "error_code": None if ok else "GRIPPER_FAILED",
            "terminal_status": "succeeded" if terminal_succeeded else "not_succeeded",
            "terminal_status_code": (
                int(terminal_status_code)
                if isinstance(terminal_status_code, int) and not isinstance(terminal_status_code, bool)
                else None
            ),
        })

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
