"""Fail-closed contracts for native-contact grasping in Gazebo.

This path has exactly one grasp mechanism: Gazebo Sim's stock
``gz::sim::systems::DetachableJoint`` fixed joint.  A request is allowed only
after both *native Gazebo* fingertip contact streams identify ``target_object``
after a real close command.  This module deliberately contains neither ROS
nor Gazebo imports so the admission and proof rules are testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any, Iterable, Mapping, Sequence

from .robot_control import GazeboControlConfig


PICKPLACE_ENV_ID = "openeta/gazebo_rm75_robotiq2f85_pickplace-v0"
PICKPLACE_MODEL_ID = "rm75_robotiq_2f85_pickplace_sim_v1"
PICKPLACE_DISPLAY_NAME = "Gazebo 仿真环境（原生接触 DetachableJoint 拾放）"
NATIVE_GRASP_SCHEMA_VERSION = "openeta.gazebo.native_grasp.v1"
PLACEMENT_VERIFICATION_SCHEMA_VERSION = "openeta.gazebo.placement_verification.v1"
# Native Gazebo poses can straddle an otherwise exact closed boundary by less
# than one micrometre.  This epsilon is only for floating-point comparison at
# the configured boundary; it is not added to the configured safety gate.
_POSE_BOUNDARY_ABS_TOL_M = 1e-6


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class ReasonCode(StrEnum):
    READY = "READY"
    CONTACT_WINDOW_NOT_ARMED = "NATIVE_GRASP_CONTACT_WINDOW_NOT_ARMED"
    CONTACT_INSUFFICIENT_SAMPLES = "NATIVE_GRASP_CONTACT_INSUFFICIENT_SAMPLES"
    CONTACT_WINDOW_TOO_SHORT = "NATIVE_GRASP_CONTACT_WINDOW_TOO_SHORT"
    CONTACT_SAMPLE_STALE = "NATIVE_GRASP_CONTACT_SAMPLE_STALE"
    CONTACT_SAMPLE_BEFORE_CLOSE = "NATIVE_GRASP_CONTACT_SAMPLE_BEFORE_CLOSE"
    CONTACT_UNKNOWN = "NATIVE_GRASP_CONTACT_UNKNOWN"
    CONTACT_MIXED = "NATIVE_GRASP_CONTACT_MIXED"
    CONTACT_DISTRACTOR = "NATIVE_GRASP_CONTACT_DISTRACTOR"
    CONTACT_TARGET_CONFIRMED = "NATIVE_GRASP_CONTACT_TARGET_CONFIRMED"
    DETACH_ACK_MISSING = "NATIVE_GRASP_DETACH_ACK_MISSING"
    ATTACH_ACK_MISSING = "NATIVE_GRASP_ATTACH_ACK_MISSING"
    ATTACH_ACKED_UNPROVEN = "NATIVE_GRASP_ATTACH_ACKED_UNPROVEN"
    CHILD_LINK_STATE_UNAVAILABLE = "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
    DART_UNSUPPORTED = "NATIVE_GRASP_DART_UNSUPPORTED"
    TARGET_NOT_LIFTED = "NATIVE_GRASP_TARGET_NOT_LIFTED"
    RELATIVE_POSE_DRIFT = "NATIVE_GRASP_CAPTURE_RELATIVE_TRANSLATION_EXCEEDED"
    TARGET_HELD = "NATIVE_GRASP_TARGET_HELD"
    RELEASE_ACK_MISSING = "NATIVE_GRASP_RELEASE_DETACH_ACK_MISSING"


class PlacementReasonCode(StrEnum):
    PLACED = "PLACEMENT_STABLE_IN_DESTINATION"
    POSE_UNAVAILABLE = "PLACEMENT_NATIVE_POSE_UNAVAILABLE"
    OBSERVATION_TOO_SHORT = "PLACEMENT_OBSERVATION_TOO_SHORT"
    TERMINAL_DRIFT = "PLACEMENT_TERMINAL_DRIFT_EXCEEDED"
    HEIGHT_OUT_OF_RANGE = "PLACEMENT_CENTER_HEIGHT_OUT_OF_RANGE"
    FOOTPRINT_OUTSIDE_DESTINATION = "PLACEMENT_FOOTPRINT_OUTSIDE_DESTINATION"


@dataclass(frozen=True, slots=True)
class PlacementPoseSample:
    """One native Gazebo target pose sampled after the detach ACK."""

    monotonic_s: float
    xyz: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PlacementVerification:
    verdict: Verdict
    reason_code: PlacementReasonCode
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLACEMENT_VERIFICATION_SCHEMA_VERSION,
            "verdict": self.verdict.value,
            "reason_code": self.reason_code.value,
            "placement_confirmed": self.verdict is Verdict.PASS,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class NativePickPlaceConfig(GazeboControlConfig):
    """Static pick/place scene and non-negotiable native-contact thresholds."""

    model_id: str = PICKPLACE_MODEL_ID
    env_id: str = PICKPLACE_ENV_ID
    display_name: str = PICKPLACE_DISPLAY_NAME
    target_id: str = "target_object"
    distractor_id: str = "distractor_object"
    # A real close against the target is allowed to terminate with the
    # ros2_control ``stalled`` result.  That terminal action result proves
    # only that the fingers stopped under load; it never proves a grasp.
    # Attachment is admitted exclusively through the subsequent native
    # bilateral Gazebo-contact window and DetachableJoint attach ACK.
    allow_stalling: bool = True
    table_id: str = "work_table"
    target_link: str = "target_link"
    parent_link: str = "gripper_mount_link"
    left_contact_topic: str = "/openeta/native_grasp/contacts/left_pad"
    right_contact_topic: str = "/openeta/native_grasp/contacts/right_pad"
    attach_topic: str = "/openeta/native_grasp/detachable_joint/target/attach"
    detach_topic: str = "/openeta/native_grasp/detachable_joint/target/detach"
    state_topic: str = "/openeta/native_grasp/detachable_joint/target/state"
    contact_samples_required: int = 3
    contact_span_s: float = 0.100
    contact_freshness_s: float = 2.0
    minimum_lift_m: float = 0.080
    maximum_capture_relative_translation_m: float = 0.010
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
    placement_stability_duration_s: float = 0.50
    placement_sample_interval_s: float = 0.10
    # Allow detached rigid-body dynamics to settle before the unchanged final
    # terminal window is judged. This extends observation only; it does not
    # relax drift, height, footprint, or duration gates.
    placement_settling_observation_s: float = 1.00
    # Native pose queries can each take longer than the nominal 100 ms sample
    # interval. Use the full required stability interval so the terminal drift
    # proof retains at least two real samples under normal ROS/Gazebo latency.
    placement_terminal_window_s: float = 0.50
    maximum_placement_terminal_drift_m: float = 0.005
    placement_center_height_m: float = 0.43
    placement_center_height_tolerance_m: float = 0.01
    # Profile-owned, live-validated gripper-mount poses for this fixed
    # fixture.  They are advertised through control_spec so an agent can use
    # the stable controller contract instead of guessing from image depth.
    approach_mount_xyz: tuple[float, float, float] = (0.1552, -0.1000, 0.5686)
    capture_mount_xyz: tuple[float, float, float] = (0.1552, -0.1000, 0.4976)
    lift_mount_xyz: tuple[float, float, float] = (0.1552, -0.1000, 0.5976)
    mount_euler_xyz_deg: tuple[float, float, float] = (115.0, 0.0, 90.0)

    @property
    def reset_object_poses(self) -> Mapping[str, tuple[float, float, float]]:
        return {
            self.target_id: self.target_initial_xyz,
            self.distractor_id: self.distractor_initial_xyz,
        }

    @property
    def ros_package_name(self) -> str:
        return "openeta_rm75_robotiq2f85_sim"

    def validate_assets(self, *, require_vendor: bool = True) -> None:
        """Validate only the approved pick/place files before a worker starts."""

        # ``dataclass(slots=True)`` creates a replacement class object on
        # Python 3.11+, for which zero-argument ``super()`` can retain the
        # pre-decoration class cell.  Name the inherited contract explicitly.
        GazeboControlConfig.validate_assets(self, require_vendor=require_vendor)
        package = self.ros_workspace / "src" / self.ros_package_name
        required = (
            package / "config/rm75_robotiq2f85_pickplace.srdf",
            package / "launch/gazebo_pickplace.launch.py",
            package / "urdf/rm75_robotiq2f85_pickplace.urdf.xacro",
            package / "worlds/rm75_robotiq2f85_pickplace.sdf",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError("MODEL_ASSET_NOT_FOUND")


def verify_stable_placement(
    samples: Sequence[PlacementPoseSample],
    config: NativePickPlaceConfig | None = None,
) -> PlacementVerification:
    """Prove a settled release using only native poses and conservative geometry."""

    cfg = config or NativePickPlaceConfig()
    if len(samples) < 2:
        return PlacementVerification(Verdict.UNKNOWN, PlacementReasonCode.POSE_UNAVAILABLE)
    if any(
        not math.isfinite(value)
        for sample in samples
        for value in (sample.monotonic_s, *sample.xyz, *sample.quat_xyzw)
    ):
        return PlacementVerification(Verdict.UNKNOWN, PlacementReasonCode.POSE_UNAVAILABLE)

    ordered = sorted(samples, key=lambda sample: sample.monotonic_s)
    final = ordered[-1]
    radius_m = math.hypot(cfg.target_size_m[0], cfg.target_size_m[1]) / 2.0
    half_x = cfg.destination_size_xy_m[0] / 2.0
    half_y = cfg.destination_size_xy_m[1] / 2.0
    x_margin_m = half_x - abs(final.xyz[0] - cfg.destination_center_xy[0]) - radius_m
    y_margin_m = half_y - abs(final.xyz[1] - cfg.destination_center_xy[1]) - radius_m
    height_error_m = abs(final.xyz[2] - cfg.placement_center_height_m)
    terminal = [
        sample
        for sample in ordered
        if final.monotonic_s - sample.monotonic_s <= cfg.placement_terminal_window_s + 1e-9
    ]
    # Pose queries are discrete and can take longer than the requested sample
    # interval. Include the immediately preceding real sample when needed to
    # prove at least the full stability duration. This makes the drift window
    # longer (and therefore stricter), never shorter than the configured gate.
    if (
        len(ordered) >= 3
        and terminal
        and terminal[-1].monotonic_s - terminal[0].monotonic_s
        < cfg.placement_stability_duration_s
    ):
        terminal_start = ordered.index(terminal[0])
        if terminal_start > 0:
            terminal = [ordered[terminal_start - 1], *terminal]
    stable_duration_s = terminal[-1].monotonic_s - terminal[0].monotonic_s
    terminal_drift_m = max(math.dist(sample.xyz, final.xyz) for sample in terminal)
    evidence = {
        "sample_count": len(ordered),
        "stable_duration_s": stable_duration_s,
        "terminal_window_s": cfg.placement_terminal_window_s,
        "terminal_drift_m": terminal_drift_m,
        "maximum_terminal_drift_m": cfg.maximum_placement_terminal_drift_m,
        "final_pose": {
            "frame": "world",
            "xyz": list(final.xyz),
            "quat_xyzw": list(final.quat_xyzw),
        },
        "expected_center_height_m": cfg.placement_center_height_m,
        "center_height_tolerance_m": cfg.placement_center_height_tolerance_m,
        "center_height_error_m": height_error_m,
        "destination_center_xy": list(cfg.destination_center_xy),
        "destination_size_xy_m": list(cfg.destination_size_xy_m),
        "conservative_footprint_radius_m": radius_m,
        "footprint_margin_xy_m": [x_margin_m, y_margin_m],
    }
    if len(terminal) < 2:
        return PlacementVerification(Verdict.UNKNOWN, PlacementReasonCode.POSE_UNAVAILABLE, evidence)
    if stable_duration_s + 1e-9 < cfg.placement_stability_duration_s:
        return PlacementVerification(
            Verdict.UNKNOWN, PlacementReasonCode.OBSERVATION_TOO_SHORT, evidence
        )
    if terminal_drift_m > cfg.maximum_placement_terminal_drift_m:
        return PlacementVerification(Verdict.FAIL, PlacementReasonCode.TERMINAL_DRIFT, evidence)
    if height_error_m > cfg.placement_center_height_tolerance_m and not math.isclose(
        height_error_m,
        cfg.placement_center_height_tolerance_m,
        rel_tol=0.0,
        abs_tol=_POSE_BOUNDARY_ABS_TOL_M,
    ):
        return PlacementVerification(
            Verdict.FAIL, PlacementReasonCode.HEIGHT_OUT_OF_RANGE, evidence
        )
    if min(x_margin_m, y_margin_m) < 0.0:
        return PlacementVerification(
            Verdict.FAIL, PlacementReasonCode.FOOTPRINT_OUTSIDE_DESTINATION, evidence
        )
    return PlacementVerification(Verdict.PASS, PlacementReasonCode.PLACED, evidence)


def validated_pickplace_motion_guidance(
    config: NativePickPlaceConfig | None = None,
) -> dict[str, Any]:
    """Return the stable atomic motion and receipt gates for the fixture."""

    cfg = config or NativePickPlaceConfig()

    def pose(name: str, xyz: tuple[float, float, float]) -> dict[str, Any]:
        return {
            "name": name,
            "target_pose": {
                "frame": "world",
                "xyz": list(xyz),
                "euler_xyz_deg": list(cfg.mount_euler_xyz_deg),
            },
        }

    return {
        "schema_version": "openeta.gazebo.validated_pickplace_motion.v1",
        "pose_semantics": cfg.parent_link,
        "motion_parameters": {
            "velocity_scaling": 0.1,
            "acceleration_scaling": 0.1,
            # The 40 mm fixture can be pushed out of the opposing pad before
            # the second pad makes contact when MoveIt stops at the edge of a
            # 1 mm / 0.01 rad goal window.  Keep the calibrated waypoint and
            # require the controller to settle tightly enough for repeatable
            # bilateral native-contact evidence.
            "tolerance": 0.0002,
            "ori_tolerance": 0.002,
        },
        "poses": [
            pose("approach", cfg.approach_mount_xyz),
            pose("capture", cfg.capture_mount_xyz),
            pose("lift", cfg.lift_mount_xyz),
        ],
        "atomic_order": [
            {"tool": "move_to", "pose": "approach"},
            {"tool": "move_to", "pose": "capture"},
            {
                "tool": "gripper_control",
                "position": 0,
                "requires_receipt": ["native_bilateral_contact", "attached_ack"],
            },
            {
                "tool": "move_to",
                "pose": "lift",
                "requires_receipt": ["native_bilateral_contact", "attached_ack"],
            },
            {
                "tool": "gripper_control",
                "position": 1,
                "requires_receipt": ["child_link_lift_proof", "detached_ack"],
            },
        ],
        "success_evidence": {
            "minimum_child_link_lift_m": cfg.minimum_lift_m,
            "maximum_capture_relative_translation_m": (
                cfg.maximum_capture_relative_translation_m
            ),
            "placement": {
                "minimum_stability_duration_s": cfg.placement_stability_duration_s,
                "maximum_terminal_drift_m": cfg.maximum_placement_terminal_drift_m,
                "center_height_m": cfg.placement_center_height_m,
                "center_height_tolerance_m": cfg.placement_center_height_tolerance_m,
                "destination_center_xy": list(cfg.destination_center_xy),
                "destination_size_xy_m": list(cfg.destination_size_xy_m),
                "footprint_rule": "target_xy_circumscribed_circle_fully_inside",
            },
        },
        "on_rejection": "observe_and_report; do_not_bypass_native_receipt_gates",
    }


@dataclass(frozen=True, slots=True)
class NativeContactSample:
    """One timestamped raw Gazebo ``gz.msgs.Contacts`` observation.

    ``collision_names`` are identities read from Gazebo's contact message;
    they are never inferred from transforms, distances, mesh bounds, or poses.
    """

    side: str
    timestamp_s: float
    received_monotonic_s: float
    collision_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("native contact side must be left or right")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("native contact timestamp must be finite and non-negative")
        if not math.isfinite(self.received_monotonic_s) or self.received_monotonic_s < 0:
            raise ValueError("native contact receive time must be finite and non-negative")
        if not self.collision_names:
            raise ValueError("native contact must include collision identities")


@dataclass(frozen=True, slots=True)
class ContactGateResult:
    accepted: bool
    reason_code: ReasonCode
    left_sample_count: int
    right_sample_count: int
    left_span_s: float | None = None
    right_span_s: float | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason_code": self.reason_code.value,
            "left_sample_count": self.left_sample_count,
            "right_sample_count": self.right_sample_count,
            "left_span_s": self.left_span_s,
            "right_span_s": self.right_span_s,
            "evidence": dict(self.evidence),
        }


def _identity_kind(
    names: Sequence[str], config: NativePickPlaceConfig, side: str
) -> ReasonCode | None:
    """Classify a raw contact message without accepting partial identities."""

    joined = "\n".join(str(name) for name in names)
    has_target = config.target_id in joined
    has_distractor = config.distractor_id in joined
    # The pad stream may only identify its own fingertip and the target.  A
    # message carrying both objects or an unrelated collision is a mixed
    # contact, which cannot be promoted by a later good sample.
    if has_distractor:
        return ReasonCode.CONTACT_DISTRACTOR
    if not has_target:
        return ReasonCode.CONTACT_UNKNOWN
    expected_tip = f"robotiq_85_{side}_finger_tip_link"
    opposite_tip = f"robotiq_85_{'right' if side == 'left' else 'left'}_finger_tip_link"
    if not any(expected_tip in name for name in names) or any(
        opposite_tip in name for name in names
    ):
        return ReasonCode.CONTACT_MIXED
    recognised = (config.target_id, expected_tip)
    if any(not any(token in name for token in recognised) for name in names):
        return ReasonCode.CONTACT_MIXED
    return None


def confirm_native_bilateral_contact(
    samples: Iterable[NativeContactSample],
    *,
    close_completed_sim_time_s: float | None,
    now_monotonic_s: float,
    config: NativePickPlaceConfig | None = None,
) -> ContactGateResult:
    """Accept only stable, post-close, bilateral target contacts.

    Every observed message in the armed window is inspected.  Therefore a
    distractor, unknown, or mixed event rejects the entire request instead of
    silently selecting the later target-looking contacts.
    """

    cfg = config or NativePickPlaceConfig()
    ordered = sorted(samples, key=lambda item: (item.timestamp_s, item.side))
    by_side: dict[str, list[NativeContactSample]] = {"left": [], "right": []}
    if close_completed_sim_time_s is None or not math.isfinite(close_completed_sim_time_s):
        return ContactGateResult(False, ReasonCode.CONTACT_SAMPLE_BEFORE_CLOSE, 0, 0)
    for sample in ordered:
        if now_monotonic_s - sample.received_monotonic_s > cfg.contact_freshness_s:
            return ContactGateResult(
                False, ReasonCode.CONTACT_SAMPLE_STALE,
                len(by_side["left"]), len(by_side["right"]),
            )
        if sample.timestamp_s <= close_completed_sim_time_s:
            return ContactGateResult(
                False, ReasonCode.CONTACT_SAMPLE_BEFORE_CLOSE,
                len(by_side["left"]), len(by_side["right"]),
            )
        rejected = _identity_kind(sample.collision_names, cfg, sample.side)
        if rejected is not None:
            return ContactGateResult(
                False, rejected, len(by_side["left"]), len(by_side["right"]),
                evidence={"collision_names": list(sample.collision_names), "side": sample.side},
            )
        by_side[sample.side].append(sample)

    left, right = by_side["left"], by_side["right"]
    left_span = left[-1].timestamp_s - left[0].timestamp_s if len(left) > 1 else 0.0
    right_span = right[-1].timestamp_s - right[0].timestamp_s if len(right) > 1 else 0.0
    if len(left) < cfg.contact_samples_required or len(right) < cfg.contact_samples_required:
        return ContactGateResult(False, ReasonCode.CONTACT_INSUFFICIENT_SAMPLES, len(left), len(right), left_span, right_span)
    if left_span < cfg.contact_span_s or right_span < cfg.contact_span_s:
        return ContactGateResult(False, ReasonCode.CONTACT_WINDOW_TOO_SHORT, len(left), len(right), left_span, right_span)
    return ContactGateResult(
        True, ReasonCode.CONTACT_TARGET_CONFIRMED, len(left), len(right), left_span, right_span,
        evidence={"target_id": cfg.target_id, "source": "gazebo_native_contacts"},
    )


@dataclass(frozen=True, slots=True)
class ChildLinkProof:
    """Measured solely from Gazebo's parent/child link state."""

    baseline_target_z_m: float
    target_z_m: float
    capture_relative_translation_m: float

    def __post_init__(self) -> None:
        values = (self.baseline_target_z_m, self.target_z_m, self.capture_relative_translation_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("child-link proof must be finite")
        if self.capture_relative_translation_m < 0:
            raise ValueError("relative translation cannot be negative")

    @property
    def lift_m(self) -> float:
        return self.target_z_m - self.baseline_target_z_m


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    phase: str
    verdict: Verdict
    reason_code: ReasonCode
    target_id: str
    grasp_confirmed: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = NATIVE_GRASP_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "verdict": self.verdict.value,
            "reason_code": self.reason_code.value,
            "target_id": self.target_id,
            "grasp_confirmed": self.grasp_confirmed,
            "evidence": dict(self.evidence),
        }


class NativeGraspVerifier:
    """Record native-grasp attach and proof state without any alternate grasp path."""

    def __init__(self, config: NativePickPlaceConfig | None = None) -> None:
        self.config = config or NativePickPlaceConfig()
        self.reset()

    def reset(self) -> VerificationRecord:
        self.phase = "ready"
        self.attached = False
        self._last_record = self._record(Verdict.UNKNOWN, ReasonCode.READY, False)
        return self._last_record

    @property
    def last_record(self) -> VerificationRecord:
        return self._last_record

    def close_result(self, gate: ContactGateResult, *, attach_acked: bool) -> VerificationRecord:
        if not gate.accepted:
            self.phase = "contact_rejected"
            self.attached = False
            return self._remember(self._record(Verdict.FAIL, gate.reason_code, False, gate=gate.to_dict()))
        if not attach_acked:
            self.phase = "attach_unacknowledged"
            self.attached = False
            return self._remember(self._record(Verdict.FAIL, ReasonCode.ATTACH_ACK_MISSING, False, gate=gate.to_dict()))
        self.phase = "attach_acked_unproven"
        self.attached = True
        return self._remember(self._record(Verdict.UNKNOWN, ReasonCode.ATTACH_ACKED_UNPROVEN, False, gate=gate.to_dict()))

    def prove_lift(self, proof: ChildLinkProof | None, *, dart_supported: bool = True) -> VerificationRecord:
        if not self.attached:
            return self._remember(self._record(Verdict.FAIL, ReasonCode.ATTACH_ACK_MISSING, False))
        if not dart_supported:
            self.phase = "dart_unsupported"
            return self._remember(self._record(Verdict.FAIL, ReasonCode.DART_UNSUPPORTED, False))
        if proof is None:
            self.phase = "child_link_unavailable"
            return self._remember(self._record(Verdict.FAIL, ReasonCode.CHILD_LINK_STATE_UNAVAILABLE, False))
        evidence = {
            "source": "gazebo_pose_info_child_link",
            "lift_m": proof.lift_m,
            "capture_relative_translation_m": proof.capture_relative_translation_m,
            "minimum_lift_m": self.config.minimum_lift_m,
            "maximum_capture_relative_translation_m": self.config.maximum_capture_relative_translation_m,
        }
        if proof.lift_m < self.config.minimum_lift_m:
            self.phase = "lift_failed"
            return self._remember(self._record(Verdict.FAIL, ReasonCode.TARGET_NOT_LIFTED, False, **evidence))
        if proof.capture_relative_translation_m > self.config.maximum_capture_relative_translation_m:
            self.phase = "relative_translation_failed"
            return self._remember(self._record(Verdict.FAIL, ReasonCode.RELATIVE_POSE_DRIFT, False, **evidence))
        self.phase = "held_proven"
        return self._remember(self._record(Verdict.PASS, ReasonCode.TARGET_HELD, True, **evidence))

    def prove_retention(
        self, proof: ChildLinkProof | None, *, dart_supported: bool = True
    ) -> VerificationRecord:
        """Revalidate an already-proven attachment during transport or lowering.

        The 80 mm threshold is a one-time lift proof.  Applying it again while
        descending for placement would incorrectly reject a still-attached
        object, so subsequent reads require the native attached state (owned by
        the caller) and the same capture-relative drift bound only.
        """

        if not self.attached:
            return self._remember(
                self._record(Verdict.FAIL, ReasonCode.ATTACH_ACK_MISSING, False)
            )
        if not dart_supported:
            self.phase = "dart_unsupported"
            return self._remember(
                self._record(Verdict.FAIL, ReasonCode.DART_UNSUPPORTED, False)
            )
        if proof is None:
            self.phase = "child_link_unavailable"
            return self._remember(
                self._record(
                    Verdict.FAIL, ReasonCode.CHILD_LINK_STATE_UNAVAILABLE, False
                )
            )
        prior_lift_proven = self.phase in {"held_proven", "retained_proven"}
        evidence = {
            "source": "gazebo_pose_info_child_link",
            "lift_m": proof.lift_m,
            "capture_relative_translation_m": proof.capture_relative_translation_m,
            "minimum_lift_required": False,
            "prior_lift_proven": prior_lift_proven,
            "maximum_capture_relative_translation_m": (
                self.config.maximum_capture_relative_translation_m
            ),
        }
        if not prior_lift_proven:
            self.phase = "lift_failed"
            return self._remember(
                self._record(
                    Verdict.FAIL,
                    ReasonCode.TARGET_NOT_LIFTED,
                    False,
                    **evidence,
                )
            )
        if proof.capture_relative_translation_m > (
            self.config.maximum_capture_relative_translation_m
        ):
            self.phase = "relative_translation_failed"
            return self._remember(
                self._record(
                    Verdict.FAIL,
                    ReasonCode.RELATIVE_POSE_DRIFT,
                    False,
                    **evidence,
                )
            )
        self.phase = "retained_proven"
        return self._remember(
            self._record(Verdict.PASS, ReasonCode.TARGET_HELD, True, **evidence)
        )

    def release_result(self, *, detached_acked: bool) -> VerificationRecord:
        self.attached = False
        self.phase = "released" if detached_acked else "release_unacknowledged"
        return self._remember(self._record(
            Verdict.UNKNOWN if detached_acked else Verdict.FAIL,
            ReasonCode.READY if detached_acked else ReasonCode.RELEASE_ACK_MISSING,
            False,
        ))

    def pregrasp_open_result(self) -> VerificationRecord:
        """Acknowledge an open command when no grasp has ever been attached."""

        self.phase = "ready"
        self.attached = False
        return self._remember(self._record(Verdict.UNKNOWN, ReasonCode.READY, False))

    def _record(self, verdict: Verdict, reason: ReasonCode, grasp: bool, **evidence: Any) -> VerificationRecord:
        return VerificationRecord(self.phase, verdict, reason, self.config.target_id, grasp, evidence)

    def _remember(self, record: VerificationRecord) -> VerificationRecord:
        self._last_record = record
        return record


def quaternion_rotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    """Rotate a vector by an xyzw quaternion (retained for motion-control callers)."""

    if len(q) != 4 or len(v) != 3:
        raise ValueError("quaternion/vector dimensions are invalid")
    x, y, z, w = (float(value) for value in q)
    vx, vy, vz = (float(value) for value in v)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx)
