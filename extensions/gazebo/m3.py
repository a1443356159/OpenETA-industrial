"""Dependency-light contracts for M3 physical pick/place verification.

This module intentionally imports neither ROS nor Gazebo.  The live adapter
normalizes official ``Odometry`` messages and existing robot state into the
immutable types below; the verifier then decides from physical evidence only.  A MoveIt
attached collision object is represented as a planning command and never as a
Gazebo joint or kinematic attachment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .m2 import M2Config


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
    RELATIVE_POSE_DRIFT = "RELATIVE_POSE_DRIFT"
    OBJECT_DROPPED = "OBJECT_DROPPED"
    TARGET_PLACED = "TARGET_PLACED"
    OUTSIDE_DESTINATION = "OUTSIDE_DESTINATION"
    NOT_SETTLED = "NOT_SETTLED"
    STALL_STATUS_MISSING = "STALL_STATUS_MISSING"
    DATA_MISSING = "DATA_MISSING"
    DATA_STALE = "DATA_STALE"
    IDENTITY_INCOMPLETE = "IDENTITY_INCOMPLETE"


@dataclass(frozen=True, slots=True)
class M3Config(M2Config):
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
    # Distal gripper links that may legitimately graze the tabletop while the
    # gripper wraps a 6 cm target standing on it.  Live planning probes showed
    # the contact pose is GOAL_STATE_INVALID unless the planning scene allows
    # these pairs; base/mount links stay collision-checked against the table.
    # The same set contacts a held object: validity queries at the stall-hold
    # state reported the grasped target touching finger and inner-knuckle
    # links, so target/distractor lifts need these exemptions as well.  The
    # reset cycle prunes world-object ACM rows, so every exemption must be
    # re-applied inside initialize().
    table_touch_links: tuple[str, ...] = (
        "robotiq_85_left_finger_tip_link",
        "robotiq_85_right_finger_tip_link",
        "robotiq_85_left_finger_link",
        "robotiq_85_right_finger_link",
        "robotiq_85_left_knuckle_link",
        "robotiq_85_right_knuckle_link",
        "robotiq_85_left_inner_knuckle_link",
        "robotiq_85_right_inner_knuckle_link",
    )
    grasp_touch_links: tuple[str, ...] = table_touch_links
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
    # Detachable-fallback geometry gate: after a verified close stall, only
    # the object whose centre lies within this distance band of the EEF mount
    # origin may be attached (the grasp centre sits ~0.127 m out).
    attach_gate_min_m: float = 0.09
    attach_gate_max_m: float = 0.17

    @property
    def reset_object_poses(self) -> Mapping[str, tuple[float, float, float]]:
        """World objects restored explicitly on reset (model_only does not)."""
        return {
            self.target_id: tuple(self.target_initial_xyz),
            self.distractor_id: tuple(self.distractor_initial_xyz),
        }

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
class PhysicsSnapshot:
    timestamp_s: float
    received_monotonic_s: float
    eef_pose: Pose
    aperture_m: float
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
        self._candidate_distractor_pose: Pose | None = None
        self._candidate_distractor_relative_pose: Pose | None = None
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
            return self._remember(self._verify_close(snapshot, target, distractor))

        if action_type == "gripper_open" and self.held:
            self.held = False
            near_table = (
                target.pose.position[2] - self.config.target_size_m[2] / 2
                <= self.config.table_top_z_m + 0.01
            )
            self._release_mode = (
                "placing"
                if self._inside_destination(target)
                or self._support(target) == self.config.table_id
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
        if (
            not math.isfinite(snapshot.aperture_m)
            or snapshot.aperture_m < 0.0
            or snapshot.aperture_m > self.config.maximum_aperture_m + 1e-6
        ):
            return self._record(
                snapshot,
                Verdict.UNKNOWN,
                ReasonCode.STALL_STATUS_MISSING,
                extra={"invalid_aperture_m": snapshot.aperture_m},
            )
        return None

    def _verify_close(
        self, snapshot: PhysicsSnapshot, target: ObjectState, distractor: ObjectState
    ) -> VerificationRecord:
        cfg = self.config
        if (
            snapshot.aperture_m <= cfg.empty_aperture_m
            and snapshot.gripper_reached_goal is True
            and snapshot.gripper_stalled is False
        ):
            self.phase = "failed"
            self._clear_grasp()
            return self._record(
                snapshot,
                Verdict.FAIL,
                ReasonCode.EMPTY_GRASP,
                object_detection="at_position_no_object",
            )
        if (
            snapshot.aperture_m > cfg.empty_aperture_m
            and snapshot.gripper_stalled is True
            and snapshot.gripper_reached_goal is False
        ):
            self.phase = "lift_required"
            self._candidate_target_pose = target.pose
            self._candidate_relative_pose = relative_pose(snapshot.eef_pose, target.pose)
            self._candidate_distractor_pose = distractor.pose
            self._candidate_distractor_relative_pose = relative_pose(
                snapshot.eef_pose, distractor.pose
            )
            self._last_target_z = target.pose.position[2]
            return self._record(
                snapshot,
                Verdict.UNKNOWN,
                ReasonCode.LIFT_REQUIRED,
                object_detection="object_detected_closing",
            )
        self.phase = "closing"
        return self._record(snapshot, Verdict.UNKNOWN, ReasonCode.STALL_STATUS_MISSING)

    def _verify_lift(self, snapshot: PhysicsSnapshot, target: ObjectState) -> VerificationRecord:
        assert self._candidate_target_pose is not None
        assert self._candidate_relative_pose is not None
        assert self._candidate_distractor_pose is not None
        assert self._candidate_distractor_relative_pose is not None
        distractor = snapshot.object(self.config.distractor_id)
        assert distractor is not None
        target_match = self._comovement(
            snapshot, target, self._candidate_target_pose, self._candidate_relative_pose
        )
        distractor_match = self._comovement(
            snapshot,
            distractor,
            self._candidate_distractor_pose,
            self._candidate_distractor_relative_pose,
        )
        evidence = {
            "target_comovement": target_match,
            "distractor_comovement": distractor_match,
        }
        if target_match["matches"] and distractor_match["matches"]:
            self.phase = "lift_required"
            return self._record(
                snapshot, Verdict.UNKNOWN, ReasonCode.IDENTITY_INCOMPLETE, extra=evidence
            )
        if distractor_match["matches"]:
            self.phase = "failed"
            self._clear_grasp()
            return self._record(
                snapshot,
                Verdict.FAIL,
                ReasonCode.WRONG_OBJECT,
                object_detection="wrong_object_detected",
                extra=evidence,
            )
        if not target_match["matches"]:
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
        supported = self._support(target)
        drifted = (
            drift_m > self.config.translation_drift_m
            or drift_rad > self.config.rotation_drift_rad
        )
        if drifted and (falling or supported is not None):
            self.phase = "failed"
            self._clear_grasp()
            return self._record(
                snapshot, Verdict.FAIL, ReasonCode.OBJECT_DROPPED, slip=True, extra=evidence
            )
        if drifted:
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
        falling = target.linear_velocity[2] < -0.02 or (
            self._last_target_z is not None
            and target.pose.position[2] < self._last_target_z - 0.005
        )
        self._last_target_z = target.pose.position[2]
        support = self._support(target)
        if falling or support is not None:
            self.phase = "failed"
            self._release_mode = None
            return self._record(
                snapshot,
                Verdict.FAIL,
                ReasonCode.OBJECT_DROPPED,
                slip=True,
                extra={"falling": falling, "support": support},
            )
        return self._record(snapshot, Verdict.UNKNOWN, ReasonCode.NOT_SETTLED)

    def _verify_place(self, snapshot: PhysicsSnapshot, target: ObjectState) -> VerificationRecord:
        inside = self._inside_destination(target)
        support = self._support(target)
        supported = support == self.config.table_id
        linear_speed = vector_norm(target.linear_velocity)
        angular_speed = vector_norm(target.angular_velocity)
        stable = (
            supported
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
            "support": support,
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

    def _support(self, item: ObjectState) -> str | None:
        size = (
            self.config.target_size_m
            if item.object_id == self.config.target_id
            else (
                self.config.distractor_size_m[0],
                self.config.distractor_size_m[0],
                self.config.distractor_size_m[1],
            )
        )
        return geometric_table_support(item.pose, size, self.config)

    def _comovement(
        self,
        snapshot: PhysicsSnapshot,
        item: ObjectState,
        baseline_pose: Pose,
        baseline_relative_pose: Pose,
    ) -> dict[str, Any]:
        lift = item.pose.position[2] - baseline_pose.position[2]
        drift_m, drift_rad = pose_distance(
            baseline_relative_pose, relative_pose(snapshot.eef_pose, item.pose)
        )
        support = self._support(item)
        return {
            "matches": (
                lift >= self.config.minimum_lift_m
                and support != self.config.table_id
                and drift_m <= self.config.translation_drift_m
                and drift_rad <= self.config.rotation_drift_rad
            ),
            "lift_m": lift,
            "support": support,
            "relative_translation_drift_m": drift_m,
            "relative_rotation_drift_rad": drift_rad,
        }

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
        self._candidate_distractor_pose = None
        self._candidate_distractor_relative_pose = None


def select_attachment_object(
    *,
    reason_code: str,
    eef_pose: Pose,
    objects: Sequence[ObjectState],
    config: M3Config | None = None,
) -> str | None:
    """Return which manipulated object is geometrically at the gripper pads.

    The user-approved detachable fallback may only attach an object after the
    verifier's own close evidence (``LIFT_REQUIRED``: stall inside the
    aperture hold window) AND with the object centre inside the grasp
    workspace band of the EEF mount.  Empty spots and objects away from the
    pads return ``None``, so the empty-grasp and wrong-object scenarios keep
    their honest negative verdicts.  Verdict computation never reads this.
    """

    cfg = config or M3Config()
    if reason_code != ReasonCode.LIFT_REQUIRED.value:
        return None
    best: tuple[float, str] | None = None
    for item in objects:
        if item.object_id not in (cfg.target_id, cfg.distractor_id):
            continue
        distance = math.dist(item.pose.position, eef_pose.position)
        if cfg.attach_gate_min_m <= distance <= cfg.attach_gate_max_m and (
            best is None or distance < best[0]
        ):
            best = (distance, item.object_id)
    return best[1] if best is not None else None


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
                {"object_id": cfg.target_id, "links": list(cfg.grasp_touch_links)},
            ),
            PlanningSceneCommand(
                "allow_distractor_touch",
                {"object_id": cfg.distractor_id, "links": list(cfg.grasp_touch_links)},
            ),
            PlanningSceneCommand(
                "allow_table_touch",
                {
                    "object_id": cfg.table_id,
                    # Attached bodies are checked against the world through the
                    # ACM under their object id, so the target/distractor names
                    # ride along: lowering a held object onto the table is a
                    # legitimate contact during placement.
                    "links": [
                        *cfg.table_touch_links,
                        cfg.target_id,
                        cfg.distractor_id,
                    ],
                },
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
                    # The held object physically touches the whole distal
                    # linkage (measured contact pairs), not only the pads;
                    # with fingertip-only touch links every sampled goal is
                    # collision-invalid once the object is attached.
                    "touch_links": list(self.config.grasp_touch_links),
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
    return oriented_box_half_extents(dimensions, quaternion)[:2]


def oriented_box_half_extents(
    dimensions: Sequence[float], quaternion: Sequence[float]
) -> tuple[float, float, float]:
    q = Pose((0.0, 0.0, 0.0), tuple(map(float, quaternion))).normalized().orientation
    x, y, z, w = q
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    half = tuple(float(value) / 2 for value in dimensions)
    return tuple(sum(abs(row[index]) * half[index] for index in range(3)) for row in rotation)  # type: ignore[return-value]


def geometric_table_support(
    pose: Pose, dimensions: Sequence[float], config: M3Config
) -> str | None:
    """Infer tabletop support from authoritative pose and known geometry."""

    hx, hy, hz = oriented_box_half_extents(dimensions, pose.orientation)
    x, y, z = pose.position
    table_x, table_y, _ = config.table_pose_xyz
    table_hx, table_hy = config.table_size_m[0] / 2, config.table_size_m[1] / 2
    footprint_on_table = (
        x - hx >= table_x - table_hx
        and x + hx <= table_x + table_hx
        and y - hy >= table_y - table_hy
        and y + hy <= table_y + table_hy
    )
    bottom_distance = z - hz - config.table_top_z_m
    if footprint_on_table and abs(bottom_distance) <= 0.005:
        return config.table_id
    return None


def namespaced_entity_id(name: str, known_ids: Iterable[str]) -> str | None:
    """Return an exact scoped Gazebo entity id, never a substring guess."""

    components = tuple(part for part in str(name).split("::") if part)
    matches = [item for item in known_ids if item in components]
    return matches[0] if len(matches) == 1 else None


def fingertip_collision_center_m(path: Path) -> tuple[float, float, float]:
    """Bounding-box centre of a frozen binary STL (vendor fingertip mesh)."""

    import struct

    data = Path(path).read_bytes()
    triangles = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + triangles * 50:
        raise ValueError(f"invalid frozen STL: {path}")
    vertices = [
        struct.unpack_from("<fff", data, 84 + triangle * 50 + 12 + vertex * 12)
        for triangle in range(triangles)
        for vertex in range(3)
    ]
    return tuple(
        (min(v[index] for v in vertices) + max(v[index] for v in vertices)) / 2
        for index in range(3)
    )
