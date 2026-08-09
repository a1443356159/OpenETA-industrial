"""Dependency-light contracts for M3 contact-only pick/place verification.

This module intentionally imports neither ROS nor Gazebo.  The live adapter
normalizes official ``Contacts`` and ``Odometry`` messages into the immutable
types below; the verifier then decides from physical evidence only.  A MoveIt
attached collision object is represented as a planning command and never as a
Gazebo joint or kinematic attachment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any, Iterable, Mapping, Sequence

from .m2 import Robotiq2F85Config


M3_ENV_ID = "openeta/gazebo_rm75_robotiq2f85_pickplace-v0"
M3_MODEL_ID = "rm75_robotiq_2f85_pickplace_sim_v1"
M3_DISPLAY_NAME = "Gazebo 仿真环境（M3 拾放物理验证）"
M3_SCHEMA_VERSION = "m3_physical_verification_v1"


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class ReasonCode(StrEnum):
    READY = "READY"
    LIFT_REQUIRED = "LIFT_REQUIRED"
    EMPTY_GRASP = "EMPTY_GRASP"
    WRONG_OBJECT = "WRONG_OBJECT"
    TARGET_HELD = "TARGET_HELD"
    TARGET_NOT_LIFTED = "TARGET_NOT_LIFTED"
    CONTACT_LOST = "CONTACT_LOST"
    RELATIVE_POSE_DRIFT = "RELATIVE_POSE_DRIFT"
    OBJECT_DROPPED = "OBJECT_DROPPED"
    TARGET_PLACED = "TARGET_PLACED"
    OUTSIDE_DESTINATION = "OUTSIDE_DESTINATION"
    NOT_SETTLED = "NOT_SETTLED"
    CONTACT_NOT_STABLE = "CONTACT_NOT_STABLE"
    STALL_STATUS_MISSING = "STALL_STATUS_MISSING"
    DATA_MISSING = "DATA_MISSING"
    DATA_STALE = "DATA_STALE"
    IDENTITY_INCOMPLETE = "IDENTITY_INCOMPLETE"


@dataclass(frozen=True, slots=True)
class M3Config(Robotiq2F85Config):
    model_id: str = M3_MODEL_ID
    env_id: str = M3_ENV_ID
    display_name: str = M3_DISPLAY_NAME
    target_id: str = "m3_target"
    distractor_id: str = "m3_distractor"
    table_id: str = "m3_table"
    fingertip_links: tuple[str, str] = (
        "robotiq_85_left_finger_tip_link",
        "robotiq_85_right_finger_tip_link",
    )
    table_size_m: tuple[float, float, float] = (0.70, 0.60, 0.04)
    # Keep the 0.70 m tabletop just clear of the RM75 base collision mesh.
    # x=0.35 places its edge through the base and makes every MoveIt start
    # state invalid; x=0.40 is the closest collision-free documented layout.
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
    destination_margin_m: float = 0.005
    stable_contact_s: float = 0.20
    candidate_aperture_range_m: tuple[float, float] = (0.032, 0.048)
    empty_aperture_m: float = 0.006
    lift_probe_m: float = 0.080
    minimum_lift_m: float = 0.060
    translation_drift_m: float = 0.010
    rotation_drift_rad: float = 0.15
    settle_s: float = 1.0
    settled_linear_speed_m_s: float = 0.02
    settled_angular_speed_rad_s: float = 0.10
    freshness_s: float = 2.0
    allow_stalling: bool = True

    @property
    def ros_package_name(self) -> str:
        return "openeta_rm75_robotiq2f85_sim"

    def validate_assets(self, *, require_vendor: bool = True) -> None:
        del require_vendor
        try:
            from .asset_preflight import validate_asset_root

            validate_asset_root(self.asset_root)
            validate_asset_root(self.gripper_asset_root)
        except Exception as exc:
            raise RuntimeError("MODEL_ASSET_NOT_FOUND") from exc
        package = self.ros_workspace / "src" / self.ros_package_name
        required = (
            package / "package.xml",
            package / "worlds/m3_rm75_robotiq2f85_pickplace.sdf",
            package / "launch/m3_gazebo_pickplace.launch.py",
            package / "urdf/rm75_robotiq2f85_m3.urdf.xacro",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError("MODEL_ASSET_NOT_FOUND")


@dataclass(frozen=True, slots=True)
class Pose:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.position) != 3 or len(self.orientation) != 4:
            raise ValueError("pose dimensions are invalid")
        if not all(math.isfinite(value) for value in (*self.position, *self.orientation)):
            raise ValueError("pose must be finite")
        if math.sqrt(sum(value * value for value in self.orientation)) <= 1e-12:
            raise ValueError("pose quaternion must be non-zero")

    def normalized(self) -> "Pose":
        norm = math.sqrt(sum(value * value for value in self.orientation))
        return Pose(self.position, tuple(value / norm for value in self.orientation))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position),
            "orientation": list(self.orientation),
        }


@dataclass(frozen=True, slots=True)
class ObjectState:
    object_id: str
    name: str
    label: str
    role: str
    pose: Pose
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    support: str | None
    timestamp_s: float
    provenance: str = "gazebo_truth"

    def __post_init__(self) -> None:
        if not self.object_id or not self.name or self.timestamp_s <= 0:
            raise ValueError("object identity and timestamp are required")
        values = (*self.linear_velocity, *self.angular_velocity, self.timestamp_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("object state must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.object_id,
            "name": self.name,
            "label": self.label,
            "role": self.role,
            "position": list(self.pose.position),
            "orientation": list(self.pose.orientation),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
            "support": self.support,
            "gazebo_timestamp_s": self.timestamp_s,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class ContactState:
    left_object_ids: tuple[str, ...]
    right_object_ids: tuple[str, ...]
    target_support_ids: tuple[str, ...]
    left_durations_s: tuple[tuple[str, float], ...]
    right_durations_s: tuple[tuple[str, float], ...]
    timestamps_s: tuple[tuple[str, float], ...]
    identities_complete: bool = True

    def duration(self, side: str, object_id: str) -> float:
        values = self.left_durations_s if side == "left" else self.right_durations_s
        return dict(values).get(object_id, 0.0)

    def bilateral(self, object_id: str, minimum_s: float) -> bool:
        return (
            object_id in self.left_object_ids
            and object_id in self.right_object_ids
            and self.duration("left", object_id) >= minimum_s
            and self.duration("right", object_id) >= minimum_s
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_object_ids": list(self.left_object_ids),
            "right_object_ids": list(self.right_object_ids),
            "target_support_ids": list(self.target_support_ids),
            "left_durations_s": dict(self.left_durations_s),
            "right_durations_s": dict(self.right_durations_s),
            "timestamps_s": dict(self.timestamps_s),
            "identities_complete": self.identities_complete,
        }


@dataclass(frozen=True, slots=True)
class PhysicsSnapshot:
    timestamp_s: float
    received_monotonic_s: float
    eef_pose: Pose
    aperture_m: float
    contacts: ContactState
    objects: tuple[ObjectState, ...]
    stream_timestamps_s: tuple[tuple[str, float], ...]
    gripper_stalled: bool | None = None
    gripper_reached_goal: bool | None = None

    def object(self, object_id: str) -> ObjectState | None:
        return next((item for item in self.objects if item.object_id == object_id), None)

    def stream_timestamps(self) -> dict[str, float]:
        return dict(self.stream_timestamps_s)


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    phase: str
    verdict: Verdict
    reason_code: ReasonCode
    target_id: str
    timestamp_s: float | None
    object_detection: str
    grasp_confirmed: bool
    slip_detected: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = M3_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "verdict": self.verdict.value,
            "reason_code": self.reason_code.value,
            "target_id": self.target_id,
            "timestamp_s": self.timestamp_s,
            "object_detection": self.object_detection,
            "grasp_confirmed": self.grasp_confirmed,
            "slip_detected": self.slip_detected,
            "evidence": dict(self.evidence),
        }


def unknown_record(
    reason: ReasonCode,
    *,
    phase: str = "unknown",
    target_id: str = "m3_target",
    evidence: Mapping[str, Any] | None = None,
) -> VerificationRecord:
    return VerificationRecord(
        phase=phase,
        verdict=Verdict.UNKNOWN,
        reason_code=reason,
        target_id=target_id,
        timestamp_s=None,
        object_detection="unknown",
        grasp_confirmed=False,
        slip_detected=False,
        evidence=evidence or {},
    )


class M3Verifier:
    """Stateful, deterministic verifier for one M3 environment episode."""

    def __init__(self, config: M3Config | None = None) -> None:
        self.config = config or M3Config()
        self.reset()

    def reset(self) -> None:
        self.phase = "ready"
        self.held = False
        self._candidate_target_pose: Pose | None = None
        self._candidate_relative_pose: Pose | None = None
        self._settle_since_s: float | None = None
        self._release_mode: str | None = None
        self._last_target_z: float | None = None
        self._last_record = unknown_record(
            ReasonCode.READY, phase=self.phase, target_id=self.config.target_id
        )

    @property
    def last_record(self) -> VerificationRecord:
        return self._last_record

    def verify(
        self,
        snapshot: PhysicsSnapshot,
        *,
        action_type: str | None = None,
        action_timestamp_s: float | None = None,
    ) -> VerificationRecord:
        invalid = self._validate_snapshot(snapshot, action_timestamp_s)
        if invalid is not None:
            return self._remember(invalid)

        target = snapshot.object(self.config.target_id)
        distractor = snapshot.object(self.config.distractor_id)
        if target is None or distractor is None:
            return self._remember(self._record(snapshot, Verdict.UNKNOWN, ReasonCode.DATA_MISSING))

        if action_type == "gripper_close":
            return self._remember(self._verify_close(snapshot, target))

        if action_type == "gripper_open" and self.held:
            self.held = False
            near_table = (
                target.pose.position[2] - self.config.target_size_m[2] / 2
                <= self.config.table_top_z_m + 0.01
            )
            self._release_mode = (
                "placing"
                if self._inside_destination(target)
                or target.support == self.config.table_id
                or near_table
                else "drop_monitoring"
            )
            self.phase = self._release_mode
            self._settle_since_s = None

        if self._release_mode == "placing":
            return self._remember(self._verify_place(snapshot, target))
        if self._release_mode == "drop_monitoring":
            return self._remember(self._verify_drop(snapshot, target))

        if self._candidate_target_pose is not None and not self.held:
            return self._remember(self._verify_lift(snapshot, target))
        if self.held:
            return self._remember(self._verify_held(snapshot, target))

        self.phase = "ready"
        return self._remember(self._record(snapshot, Verdict.UNKNOWN, ReasonCode.READY))

    def _validate_snapshot(
        self, snapshot: PhysicsSnapshot, action_timestamp_s: float | None
    ) -> VerificationRecord | None:
        required = {
            "joint_state",
            "tf",
            "rgb",
            "depth",
            "contact_left",
            "contact_right",
            "contact_target",
            "odometry_target",
            "odometry_distractor",
        }
        timestamps = snapshot.stream_timestamps()
        missing = sorted(required - set(timestamps))
        if missing:
            return self._record(
                snapshot, Verdict.UNKNOWN, ReasonCode.DATA_MISSING, extra={"missing_streams": missing}
            )
        stale = sorted(
            name
            for name in required
            if timestamps[name] <= 0
            or snapshot.timestamp_s - timestamps[name] > self.config.freshness_s
            or (action_timestamp_s is not None and timestamps[name] <= action_timestamp_s)
        )
        if stale:
            return self._record(
                snapshot, Verdict.UNKNOWN, ReasonCode.DATA_STALE, extra={"stale_streams": stale}
            )
        if not snapshot.contacts.identities_complete:
            return self._record(snapshot, Verdict.UNKNOWN, ReasonCode.IDENTITY_INCOMPLETE)
        return None

    def _verify_close(self, snapshot: PhysicsSnapshot, target: ObjectState) -> VerificationRecord:
        contacts = snapshot.contacts
        cfg = self.config
        if contacts.bilateral(cfg.distractor_id, cfg.stable_contact_s):
            self.phase = "failed"
            self._clear_grasp()
            return self._record(
                snapshot,
                Verdict.FAIL,
                ReasonCode.WRONG_OBJECT,
                object_detection="wrong_object_detected",
            )
        if (
            snapshot.aperture_m <= cfg.empty_aperture_m
            and not set(contacts.left_object_ids).intersection(contacts.right_object_ids)
        ):
            self.phase = "failed"
            self._clear_grasp()
            return self._record(
                snapshot,
                Verdict.FAIL,
                ReasonCode.EMPTY_GRASP,
                object_detection="at_position_no_object",
            )
        if contacts.bilateral(cfg.target_id, cfg.stable_contact_s):
            low, high = cfg.candidate_aperture_range_m
            if not low <= snapshot.aperture_m <= high:
                return self._record(snapshot, Verdict.UNKNOWN, ReasonCode.CONTACT_NOT_STABLE)
            if snapshot.gripper_stalled is not True or snapshot.gripper_reached_goal is not False:
                return self._record(snapshot, Verdict.UNKNOWN, ReasonCode.STALL_STATUS_MISSING)
            self.phase = "lift_required"
            self._candidate_target_pose = target.pose
            self._candidate_relative_pose = relative_pose(snapshot.eef_pose, target.pose)
            self._last_target_z = target.pose.position[2]
            return self._record(
                snapshot,
                Verdict.UNKNOWN,
                ReasonCode.LIFT_REQUIRED,
                object_detection="object_detected_closing",
            )
        self.phase = "closing"
        return self._record(snapshot, Verdict.UNKNOWN, ReasonCode.CONTACT_NOT_STABLE)

    def _verify_lift(self, snapshot: PhysicsSnapshot, target: ObjectState) -> VerificationRecord:
        assert self._candidate_target_pose is not None
        assert self._candidate_relative_pose is not None
        dz = target.pose.position[2] - self._candidate_target_pose.position[2]
        drift_m, drift_rad = pose_distance(
            self._candidate_relative_pose, relative_pose(snapshot.eef_pose, target.pose)
        )
        bilateral = snapshot.contacts.bilateral(
            self.config.target_id, self.config.stable_contact_s
        )
        evidence = {
            "target_lift_m": dz,
            "relative_translation_drift_m": drift_m,
            "relative_rotation_drift_rad": drift_rad,
        }
        if not bilateral:
            self.phase = "failed"
            self._clear_grasp()
            return self._record(
                snapshot, Verdict.FAIL, ReasonCode.CONTACT_LOST, slip=True, extra=evidence
            )
        if drift_m > self.config.translation_drift_m or drift_rad > self.config.rotation_drift_rad:
            self.phase = "failed"
            self._clear_grasp()
            return self._record(
                snapshot,
                Verdict.FAIL,
                ReasonCode.RELATIVE_POSE_DRIFT,
                slip=True,
                extra=evidence,
            )
        if dz < self.config.minimum_lift_m or target.support == self.config.table_id:
            self.phase = "lift_required"
            return self._record(
                snapshot, Verdict.UNKNOWN, ReasonCode.TARGET_NOT_LIFTED, extra=evidence
            )
        self.phase = "held"
        self.held = True
        self._last_target_z = target.pose.position[2]
        return self._record(
            snapshot,
            Verdict.PASS,
            ReasonCode.TARGET_HELD,
            object_detection="object_detected_closing",
            grasp=True,
            extra=evidence,
        )

    def _verify_held(self, snapshot: PhysicsSnapshot, target: ObjectState) -> VerificationRecord:
        assert self._candidate_relative_pose is not None
        current_relative = relative_pose(snapshot.eef_pose, target.pose)
        drift_m, drift_rad = pose_distance(self._candidate_relative_pose, current_relative)
        bilateral = snapshot.contacts.bilateral(
            self.config.target_id, self.config.stable_contact_s
        )
        falling = (
            self._last_target_z is not None
            and target.pose.position[2] < self._last_target_z - 0.005
        ) or target.linear_velocity[2] < -0.02
        self._last_target_z = target.pose.position[2]
        evidence = {
            "relative_translation_drift_m": drift_m,
            "relative_rotation_drift_rad": drift_rad,
            "falling": falling,
        }
        if not bilateral and (falling or target.support is not None):
            self.phase = "failed"
            self._clear_grasp()
            return self._record(
                snapshot, Verdict.FAIL, ReasonCode.OBJECT_DROPPED, slip=True, extra=evidence
            )
        if not bilateral:
            self.phase = "failed"
            self._clear_grasp()
            return self._record(
                snapshot, Verdict.FAIL, ReasonCode.CONTACT_LOST, slip=True, extra=evidence
            )
        if drift_m > self.config.translation_drift_m or drift_rad > self.config.rotation_drift_rad:
            self.phase = "failed"
            self._clear_grasp()
            return self._record(
                snapshot,
                Verdict.FAIL,
                ReasonCode.RELATIVE_POSE_DRIFT,
                slip=True,
                extra=evidence,
            )
        self.phase = "transport"
        return self._record(
            snapshot,
            Verdict.PASS,
            ReasonCode.TARGET_HELD,
            object_detection="object_detected_closing",
            grasp=True,
            extra=evidence,
        )

    def _verify_drop(self, snapshot: PhysicsSnapshot, target: ObjectState) -> VerificationRecord:
        no_contact = not (
            self.config.target_id in snapshot.contacts.left_object_ids
            or self.config.target_id in snapshot.contacts.right_object_ids
        )
        falling = target.linear_velocity[2] < -0.02 or (
            self._last_target_z is not None
            and target.pose.position[2] < self._last_target_z - 0.005
        )
        self._last_target_z = target.pose.position[2]
        if no_contact and (falling or target.support is not None):
            self.phase = "failed"
            self._release_mode = None
            return self._record(
                snapshot,
                Verdict.FAIL,
                ReasonCode.OBJECT_DROPPED,
                slip=True,
                extra={"falling": falling, "support": target.support},
            )
        return self._record(snapshot, Verdict.UNKNOWN, ReasonCode.NOT_SETTLED)

    def _verify_place(self, snapshot: PhysicsSnapshot, target: ObjectState) -> VerificationRecord:
        inside = self._inside_destination(target)
        no_contact = not (
            self.config.target_id in snapshot.contacts.left_object_ids
            or self.config.target_id in snapshot.contacts.right_object_ids
        )
        supported = target.support == self.config.table_id
        linear_speed = vector_norm(target.linear_velocity)
        angular_speed = vector_norm(target.angular_velocity)
        stable = (
            no_contact
            and supported
            and linear_speed <= self.config.settled_linear_speed_m_s
            and angular_speed <= self.config.settled_angular_speed_rad_s
        )
        if stable:
            if self._settle_since_s is None:
                self._settle_since_s = snapshot.timestamp_s
        else:
            self._settle_since_s = None
        stable_duration = (
            0.0
            if self._settle_since_s is None
            else max(0.0, snapshot.timestamp_s - self._settle_since_s)
        )
        evidence = {
            "inside_destination": inside,
            "no_fingertip_contact": no_contact,
            "support": target.support,
            "linear_speed_m_s": linear_speed,
            "angular_speed_rad_s": angular_speed,
            "stable_duration_s": stable_duration,
        }
        if stable_duration < self.config.settle_s:
            return self._record(
                snapshot, Verdict.UNKNOWN, ReasonCode.NOT_SETTLED, extra=evidence
            )
        self._release_mode = None
        if not inside:
            self.phase = "failed"
            return self._record(
                snapshot, Verdict.FAIL, ReasonCode.OUTSIDE_DESTINATION, extra=evidence
            )
        self.phase = "placed"
        return self._record(
            snapshot, Verdict.PASS, ReasonCode.TARGET_PLACED, extra=evidence
        )

    def _inside_destination(self, target: ObjectState) -> bool:
        hx, hy = oriented_box_xy_half_extents(
            self.config.target_size_m, target.pose.orientation
        )
        cx, cy = self.config.destination_center_xy
        sx, sy = self.config.destination_size_xy_m
        margin = self.config.destination_margin_m
        x, y, _ = target.pose.position
        return (
            x - hx >= cx - sx / 2 + margin
            and x + hx <= cx + sx / 2 - margin
            and y - hy >= cy - sy / 2 + margin
            and y + hy <= cy + sy / 2 - margin
        )

    def _record(
        self,
        snapshot: PhysicsSnapshot,
        verdict: Verdict,
        reason: ReasonCode,
        *,
        object_detection: str = "unknown",
        grasp: bool | None = None,
        slip: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> VerificationRecord:
        target = snapshot.object(self.config.target_id)
        evidence: dict[str, Any] = {
            "stream_timestamps_s": snapshot.stream_timestamps(),
            "contacts": snapshot.contacts.to_dict(),
            "aperture_m": snapshot.aperture_m,
            "eef_pose": snapshot.eef_pose.to_dict(),
            "target": target.to_dict() if target else None,
            "relative_pose": (
                relative_pose(snapshot.eef_pose, target.pose).to_dict() if target else None
            ),
            "gripper_stalled": snapshot.gripper_stalled,
            "gripper_reached_goal": snapshot.gripper_reached_goal,
        }
        evidence.update(extra or {})
        return VerificationRecord(
            phase=self.phase,
            verdict=verdict,
            reason_code=reason,
            target_id=self.config.target_id,
            timestamp_s=snapshot.timestamp_s,
            object_detection=object_detection,
            grasp_confirmed=self.held if grasp is None else grasp,
            slip_detected=slip,
            evidence=evidence,
        )

    def _remember(self, record: VerificationRecord) -> VerificationRecord:
        self._last_record = record
        return record

    def _clear_grasp(self) -> None:
        self.held = False
        self._candidate_target_pose = None
        self._candidate_relative_pose = None


@dataclass(frozen=True, slots=True)
class PlanningSceneCommand:
    operation: str
    payload: Mapping[str, Any]


class M3PlanningSceneModel:
    """Pure command model used by the ROS PlanningScene service adapter."""

    def __init__(self, config: M3Config | None = None) -> None:
        self.config = config or M3Config()
        self.initialized = False
        self.attached = False

    def initialize(self, target_pose: Pose, distractor_pose: Pose) -> tuple[PlanningSceneCommand, ...]:
        self.initialized, self.attached = True, False
        cfg = self.config
        return (
            PlanningSceneCommand(
                "replace_world",
                {
                    "objects": [
                        {
                            "id": cfg.table_id,
                            "shape": "box",
                            "dimensions": list(cfg.table_size_m),
                            "pose": Pose(cfg.table_pose_xyz, (0.0, 0.0, 0.0, 1.0)).to_dict(),
                        },
                        {
                            "id": cfg.target_id,
                            "shape": "box",
                            "dimensions": list(cfg.target_size_m),
                            "pose": target_pose.to_dict(),
                        },
                        {
                            "id": cfg.distractor_id,
                            "shape": "cylinder",
                            "dimensions": [cfg.distractor_size_m[1], cfg.distractor_size_m[0] / 2],
                            "pose": distractor_pose.to_dict(),
                        },
                    ]
                },
            ),
            PlanningSceneCommand(
                "allow_target_touch",
                {"object_id": cfg.target_id, "links": list(cfg.fingertip_links)},
            ),
        )

    def attach(self, relative_pose_value: Pose) -> tuple[PlanningSceneCommand, ...]:
        if not self.initialized:
            raise RuntimeError("PLANNING_SCENE_NOT_INITIALIZED")
        self.attached = True
        return (
            PlanningSceneCommand(
                "attach",
                {
                    "object_id": self.config.target_id,
                    "link_name": self.config.mount_child,
                    "touch_links": list(self.config.fingertip_links),
                    "dimensions": list(self.config.target_size_m),
                    "relative_pose": relative_pose_value.to_dict(),
                },
            ),
        )

    def release(self, world_pose: Pose) -> tuple[PlanningSceneCommand, ...]:
        self.attached = False
        return (
            PlanningSceneCommand(
                "release",
                {
                    "object_id": self.config.target_id,
                    "link_name": self.config.mount_child,
                    "dimensions": list(self.config.target_size_m),
                    "world_pose": world_pose.to_dict(),
                },
            ),
        )

    def clear(self) -> tuple[PlanningSceneCommand, ...]:
        self.initialized, self.attached = False, False
        return (
            PlanningSceneCommand(
                "clear",
                {
                    "world_object_ids": [
                        self.config.table_id,
                        self.config.target_id,
                        self.config.distractor_id,
                    ],
                    "attached_object_id": self.config.target_id,
                    "link_name": self.config.mount_child,
                },
            ),
        )


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quaternion_rotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def relative_pose(parent: Pose, child: Pose) -> Pose:
    p, c = parent.normalized(), child.normalized()
    inverse_q = (-p.orientation[0], -p.orientation[1], -p.orientation[2], p.orientation[3])
    delta = tuple(c.position[index] - p.position[index] for index in range(3))
    return Pose(quaternion_rotate(inverse_q, delta), quaternion_multiply(inverse_q, c.orientation)).normalized()


def pose_distance(a: Pose, b: Pose) -> tuple[float, float]:
    translation = vector_norm(tuple(a.position[index] - b.position[index] for index in range(3)))
    qa, qb = a.normalized().orientation, b.normalized().orientation
    dot = min(1.0, max(-1.0, abs(sum(qa[index] * qb[index] for index in range(4)))))
    return translation, 2.0 * math.acos(dot)


def oriented_box_xy_half_extents(
    dimensions: Sequence[float], quaternion: Sequence[float]
) -> tuple[float, float]:
    q = Pose((0.0, 0.0, 0.0), tuple(map(float, quaternion))).normalized().orientation
    x, y, z, w = q
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
    )
    half = tuple(float(value) / 2 for value in dimensions)
    return tuple(sum(abs(row[index]) * half[index] for index in range(3)) for row in rotation)  # type: ignore[return-value]


def namespaced_entity_id(name: str, known_ids: Iterable[str]) -> str | None:
    """Return an exact scoped Gazebo entity id, never a substring guess."""

    components = tuple(part for part in str(name).split("::") if part)
    matches = [item for item in known_ids if item in components]
    return matches[0] if len(matches) == 1 else None
