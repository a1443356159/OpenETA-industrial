"""Fail-closed contracts for native-contact grasping in Gazebo.

This path has exactly one grasp mechanism: Gazebo Sim's stock
``gz::sim::systems::DetachableJoint`` fixed joint.  A request is allowed only
after both *native Gazebo* fingertip contact streams identify ``target_object``
after a real close command.  This module deliberately contains neither ROS
nor Gazebo imports so the admission and proof rules are testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent.runtime.collision_geometry import (
    collision_geometry_volume_centroid,
    compound_axis_aligned_bounds,
    orientation_invariant_radius_m,
    project_collision_geometry,
)

from .robot_control import GazeboControlConfig
from .ros2_ws.src.openeta_rm75_robotiq2f85_sim.launch.acceptance_scene_world import (
    ATTACHED_COLLISION_FILTER_STATE_ACK_TOPIC,
    ATTACHED_COLLISION_FILTER_STATE_REQUEST_TOPIC,
    ATTACHED_COLLISION_FILTER_STATE_TOPIC,
    ATTACHED_TARGET_COLLISION_FILTER_MASK,
    CompiledAuthoritativeScene,
    DETACHED_TARGET_COLLISION_FILTER_MASK,
    ROBOT_COLLISION_FILTER_MASK,
    compile_authoritative_scene,
    scene_target_bindings,
)


PICKPLACE_ENV_ID = "openeta/gazebo_rm75_robotiq2f85_pickplace-v0"
PICKPLACE_MODEL_ID = "rm75_robotiq_2f85_pickplace_sim_v1"
PICKPLACE_DISPLAY_NAME = "Gazebo 仿真环境（原生接触 DetachableJoint 拾放）"
NATIVE_GRASP_SCHEMA_VERSION = "openeta.gazebo.native_grasp.v1"
PLACEMENT_VERIFICATION_SCHEMA_VERSION = "openeta.gazebo.placement_verification.v1"
# Native Gazebo poses can straddle an otherwise exact closed boundary by less
# than one micrometre.  This epsilon is only for floating-point comparison at
# the configured boundary; it is not added to the configured safety gate.
_POSE_BOUNDARY_ABS_TOL_M = 1e-6
ACCEPTANCE_SCENE_ENV = "OPENETA_ACCEPTANCE_SCENE"
ACCEPTANCE_SCENE_SCHEMA_VERSION = "openeta.gazebo_acceptance_scenes.v2"
PLACEMENT_ACCEPTANCE_COMPLETE_FOOTPRINT = "complete_footprint"
PLACEMENT_ACCEPTANCE_STABLE_GEOMETRY_CENTROID = "stable_geometry_centroid_inside"
PLACEMENT_ACCEPTANCE_SEMANTICS = frozenset(
    {
        PLACEMENT_ACCEPTANCE_COMPLETE_FOOTPRINT,
        PLACEMENT_ACCEPTANCE_STABLE_GEOMETRY_CENTROID,
    }
)


def acceptance_scene_contract_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "ros2_ws/src/openeta_rm75_robotiq2f85_sim/config/acceptance_scenes.json"
    )


def load_acceptance_scene_contract(
    scene_id: str,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Load and validate one immutable Gazebo/PlanningScene geometry contract."""

    contract_path = path or acceptance_scene_contract_path()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != ACCEPTANCE_SCENE_SCHEMA_VERSION
        or not isinstance(payload.get("scenes"), Mapping)
    ):
        raise ValueError("acceptance scene catalog is invalid")
    raw = payload["scenes"].get(scene_id)
    if not isinstance(raw, Mapping):
        raise ValueError(f"unsupported acceptance scene: {scene_id}")
    scene = json.loads(json.dumps(raw))
    if not isinstance(scene.get("world_scene"), str) or not scene["world_scene"]:
        raise ValueError("acceptance scene world identity is invalid")
    seed = scene.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("acceptance scene seed is invalid")
    obstacles = scene.get("static_obstacles")
    if not isinstance(obstacles, list):
        raise ValueError("acceptance scene obstacle list is invalid")
    if "planning_scene_obstacles" in scene:
        raise ValueError(
            "legacy planning_scene_obstacles are forbidden; "
            "MoveIt geometry is compiled from the authoritative Gazebo world"
        )
    canonical_world_complete = scene.get("canonical_world_complete")
    if canonical_world_complete is not None and not isinstance(canonical_world_complete, bool):
        raise ValueError("acceptance canonical-world flag is invalid")

    def vector(
        owner: Mapping[str, Any],
        key: str,
        count: int,
        *,
        positive: bool = False,
    ) -> list[float]:
        values = owner.get(key)
        if (
            not isinstance(values, list)
            or len(values) != count
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or (positive and float(value) <= 0.0)
                for value in values
            )
        ):
            raise ValueError(f"acceptance scene {key} is invalid")
        return [float(value) for value in values]

    def validate_primitives(owner: Mapping[str, Any]) -> None:
        primitives = owner.get("primitives")
        if primitives is None:
            return
        if not isinstance(primitives, list) or not primitives:
            raise ValueError("acceptance scene primitives are invalid")
        for primitive in primitives:
            if not isinstance(primitive, Mapping):
                raise ValueError("acceptance scene primitive is invalid")
            shape = str(primitive.get("shape") or "")
            if shape == "box":
                vector(primitive, "size_xyz", 3, positive=True)
            elif shape == "cylinder":
                radius = primitive.get("radius")
                length = primitive.get("length")
                if any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in (radius, length)
                ):
                    raise ValueError("acceptance scene cylinder is invalid")
            else:
                raise ValueError("acceptance scene primitive shape is invalid")
            vector(primitive, "pose_xyz", 3)
            vector(primitive, "pose_rpy", 3)
            vector(primitive, "rgba", 4)

    seen: set[str] = {"work_table", "target_object", "distractor_object"}
    for obstacle in obstacles:
        if not isinstance(obstacle, Mapping):
            raise ValueError("acceptance scene obstacle is invalid")
        obstacle_id = str(obstacle.get("id") or "")
        if not obstacle_id or obstacle_id in seen:
            raise ValueError("acceptance scene obstacle identity is invalid")
        seen.add(obstacle_id)
        for key, count, positive in (
            ("size_xyz", 3, True),
            ("pose_xyz", 3, False),
            ("pose_rpy", 3, False),
            ("rgba", 4, False),
        ):
            vector(obstacle, key, count, positive=positive)
        validate_primitives(obstacle)
    task = scene.get("task")
    if task is not None:
        if not isinstance(task, Mapping) or any(
            not isinstance(task.get(key), str) or not str(task[key]).strip()
            for key in (
                "target_prompt",
                "placement_object_prompt",
                "placement_region_prompt",
            )
        ):
            raise ValueError("acceptance scene task semantics are invalid")
        operator_instruction = task.get("operator_instruction")
        if operator_instruction is not None and (
            not isinstance(operator_instruction, str) or not operator_instruction.strip()
        ):
            raise ValueError("acceptance scene operator instruction is invalid")
    target = scene.get("target_object")
    if target is not None:
        if (
            not isinstance(target, Mapping)
            or not isinstance(target.get("shape_class"), str)
            or not str(target["shape_class"]).strip()
        ):
            raise ValueError("acceptance scene target object is invalid")
        vector(target, "bounding_box_xyz", 3, positive=True)
        vector(target, "pose_xyz", 3)
        vector(target, "pose_rpy", 3)
        mass = target.get("mass_kg")
        if (
            not isinstance(mass, (int, float))
            or isinstance(mass, bool)
            or not math.isfinite(float(mass))
            or float(mass) <= 0.0
        ):
            raise ValueError("acceptance scene target mass is invalid")
        validate_primitives(target)
    placement_regions = scene.get("placement_regions")
    if placement_regions is not None:
        if not isinstance(placement_regions, list) or len(placement_regions) < 2:
            raise ValueError("acceptance scene placement regions are invalid")
        placement_ids: set[str] = set()
        selected: list[Mapping[str, Any]] = []
        for region in placement_regions:
            if not isinstance(region, Mapping):
                raise ValueError("acceptance scene placement region is invalid")
            region_id = str(region.get("id") or "")
            prompt = str(region.get("prompt") or "")
            if not region_id or region_id in placement_ids or not prompt:
                raise ValueError("acceptance scene placement region identity is invalid")
            placement_ids.add(region_id)
            vector(region, "center_xy", 2)
            vector(region, "size_xy_m", 2, positive=True)
            vector(region, "rgba", 4)
            support_z = region.get("support_z_m")
            if support_z is not None and (
                not isinstance(support_z, (int, float))
                or isinstance(support_z, bool)
                or not math.isfinite(float(support_z))
            ):
                raise ValueError("acceptance scene placement support is invalid")
            acceptance_semantics = region.get(
                "acceptance_semantics",
                PLACEMENT_ACCEPTANCE_COMPLETE_FOOTPRINT,
            )
            if acceptance_semantics not in PLACEMENT_ACCEPTANCE_SEMANTICS:
                raise ValueError("acceptance scene placement semantics are invalid")
            if region.get("selected") is True:
                selected.append(region)
            elif region.get("selected") is not False:
                raise ValueError("acceptance scene placement selection is invalid")
        if len(selected) != 1:
            raise ValueError("acceptance scene needs exactly one selected placement region")
        chosen = selected[0]
        if task is None or task["placement_region_prompt"] != chosen["prompt"]:
            raise ValueError("acceptance scene task/bin semantic binding is invalid")
        scene["destination_center_xy"] = [float(value) for value in chosen["center_xy"]]
        scene["destination_size_xy_m"] = [float(value) for value in chosen["size_xy_m"]]
        if chosen.get("support_z_m") is not None:
            scene["destination_support_z_m"] = float(chosen["support_z_m"])
        scene["placement_acceptance_semantics"] = str(
            chosen.get(
                "acceptance_semantics",
                PLACEMENT_ACCEPTANCE_COMPLETE_FOOTPRINT,
            )
        )
        scene["selected_placement_region_id"] = str(chosen["id"])
    sort_assignments = scene.get("sort_assignments")
    if sort_assignments is not None:
        if not isinstance(sort_assignments, list) or len(sort_assignments) < 2:
            raise ValueError("acceptance scene sort assignments are invalid")
        assignment_ids: set[str] = set()
        target_ids: set[str] = set()
        region_by_id = {
            str(region["id"]): region
            for region in (placement_regions or [])
            if isinstance(region, Mapping)
        }
        for assignment in sort_assignments:
            required_fields = (
                "id",
                "target_object_id",
                "target_link",
                "target_prompt",
                "placement_object_prompt",
                "placement_region_id",
                "placement_region_prompt",
            )
            if not isinstance(assignment, Mapping) or any(
                not isinstance(assignment.get(key), str)
                or not str(assignment[key]).strip()
                for key in required_fields
            ):
                raise ValueError("acceptance scene sort assignment is invalid")
            assignment_id = str(assignment["id"])
            target_id = str(assignment["target_object_id"])
            if assignment_id in assignment_ids or target_id in target_ids:
                raise ValueError("acceptance scene sort assignment identity is duplicated")
            assignment_ids.add(assignment_id)
            target_ids.add(target_id)
            region = region_by_id.get(str(assignment["placement_region_id"]))
            if region is None or str(region["prompt"]) != str(
                assignment["placement_region_prompt"]
            ):
                raise ValueError("acceptance scene sort assignment/bin binding is invalid")
        first_assignment = sort_assignments[0]
        if task is None or any(
            str(task[key]) != str(first_assignment[key])
            for key in (
                "target_prompt",
                "placement_object_prompt",
                "placement_region_prompt",
            )
        ):
            raise ValueError("acceptance scene initial sort semantics are invalid")
    destination = scene.get("destination_center_xy")
    if destination is not None and (
        not isinstance(destination, list)
        or len(destination) != 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in destination
        )
    ):
        raise ValueError("acceptance scene destination is invalid")
    scene["scene_id"] = scene_id
    scene["contract_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema_version": ACCEPTANCE_SCENE_SCHEMA_VERSION,
                "scene_id": scene_id,
                "scene": raw,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return scene


def _acceptance_scene_from_environment() -> str:
    return str(os.environ.get(ACCEPTANCE_SCENE_ENV) or "normal").strip()


def _quaternion_from_rpy(values: Sequence[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (float(value) for value in values)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


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
    CONTACT_SAMPLE_BEFORE_WINDOW = "NATIVE_GRASP_CONTACT_SAMPLE_BEFORE_WINDOW"
    CONTACT_SAMPLE_AFTER_WINDOW = "NATIVE_GRASP_CONTACT_SAMPLE_AFTER_WINDOW"
    CONTACT_UNKNOWN = "NATIVE_GRASP_CONTACT_UNKNOWN"
    CONTACT_MIXED = "NATIVE_GRASP_CONTACT_MIXED"
    CONTACT_DISTRACTOR = "NATIVE_GRASP_CONTACT_DISTRACTOR"
    CONTACT_TARGET_CONFIRMED = "NATIVE_GRASP_CONTACT_TARGET_CONFIRMED"
    DETACH_ACK_MISSING = "NATIVE_GRASP_DETACH_ACK_MISSING"
    ATTACH_ACK_MISSING = "NATIVE_GRASP_ATTACH_ACK_MISSING"
    ATTACHMENT_STATE_INVALID = "NATIVE_GRASP_ATTACHMENT_STATE_INVALID"
    ATTACHMENT_CONFIRMED = "NATIVE_GRASP_ATTACHMENT_CONFIRMED"
    CHILD_LINK_STATE_UNAVAILABLE = "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
    DART_UNSUPPORTED = "NATIVE_GRASP_DART_UNSUPPORTED"
    RELATIVE_POSE_DRIFT = "NATIVE_GRASP_CAPTURE_RELATIVE_TRANSLATION_EXCEEDED"
    TARGET_HELD = "NATIVE_GRASP_TARGET_HELD"
    RELEASE_ACK_MISSING = "NATIVE_GRASP_RELEASE_DETACH_ACK_MISSING"


class PlacementReasonCode(StrEnum):
    PLACED = "PLACEMENT_STABLE_IN_DESTINATION"
    POSE_UNAVAILABLE = "PLACEMENT_NATIVE_POSE_UNAVAILABLE"
    OBSERVATION_TOO_SHORT = "PLACEMENT_OBSERVATION_TOO_SHORT"
    TERMINAL_DRIFT = "PLACEMENT_TERMINAL_DRIFT_EXCEEDED"
    HEIGHT_OUT_OF_RANGE = "PLACEMENT_SUPPORT_HEIGHT_OUT_OF_RANGE"
    FOOTPRINT_OUTSIDE_DESTINATION = "PLACEMENT_FOOTPRINT_OUTSIDE_DESTINATION"
    CENTROID_OUTSIDE_DESTINATION = "PLACEMENT_GEOMETRY_CENTROID_OUTSIDE_DESTINATION"


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
    acceptance_scene_id: str = field(default_factory=_acceptance_scene_from_environment)
    active_sort_assignment_index: int = 0
    left_contact_topic: str = "/openeta/native_grasp/contacts/left_pad"
    right_contact_topic: str = "/openeta/native_grasp/contacts/right_pad"
    attach_topic: str = "/openeta/native_grasp/detachable_joint/target/attach"
    detach_topic: str = "/openeta/native_grasp/detachable_joint/target/detach"
    state_topic: str = "/openeta/native_grasp/detachable_joint/target/state"
    attached_collision_filter_state_topic: str = (
        ATTACHED_COLLISION_FILTER_STATE_TOPIC
    )
    attached_collision_filter_state_request_topic: str = (
        ATTACHED_COLLISION_FILTER_STATE_REQUEST_TOPIC
    )
    attached_collision_filter_state_ack_topic: str = (
        ATTACHED_COLLISION_FILTER_STATE_ACK_TOPIC
    )
    robot_collision_filter_mask: int = ROBOT_COLLISION_FILTER_MASK
    detached_target_collision_filter_mask: int = (
        DETACHED_TARGET_COLLISION_FILTER_MASK
    )
    attached_target_collision_filter_mask: int = (
        ATTACHED_TARGET_COLLISION_FILTER_MASK
    )
    contact_samples_required: int = 3
    contact_span_s: float = 0.100
    contact_freshness_s: float = 2.0
    # A contact sensor is not required to publish a new message after the
    # gripper action result.  Admit only a sustained bilateral hold from this
    # short terminal part of the close action; earlier transient collisions
    # remain outside the proof window.
    contact_terminal_lookback_s: float = 0.500
    # Once the gripper action reports its terminal position, keep the same
    # close command active while native contacts advance through one short
    # simulator-time hold.  This strengthens the physical proof for a pad
    # whose first stable contact begins immediately before the action result;
    # the 100 ms bilateral span requirement itself remains unchanged.
    contact_post_close_hold_s: float = 0.120
    maximum_capture_relative_translation_m: float = 0.010
    table_size_m: tuple[float, float, float] = (1.15, 0.95, 0.06)
    table_pose_xyz: tuple[float, float, float] = (0.35, 0.0, -0.03)
    table_top_z_m: float = 0.0
    target_size_m: tuple[float, float, float] = (0.04, 0.04, 0.06)
    target_mass_kg: float = 0.10
    target_initial_xyz: tuple[float, float, float] = (0.28, -0.10, 0.43)
    target_initial_quat_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    distractor_size_m: tuple[float, float, float] = (0.20, 0.06, 0.022)
    distractor_mass_kg: float = 0.23
    distractor_initial_xyz: tuple[float, float, float] = (0.34, 0.375, 0.011)
    destination_center_xy: tuple[float, float] = (0.48, -0.10)
    destination_size_xy_m: tuple[float, float] = (0.12, 0.12)
    destination_support_z_m: float = 0.0
    # AnyPlace describes the settled object goal.  Release the attached object
    # above that goal and let native gravity complete the final drop; this is a
    # single terminal-pose translation, not an artificial hover waypoint.
    placement_release_z_offset_m: float = 0.05
    placement_acceptance_semantics: str | None = None
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
    placement_support_height_tolerance_m: float = 0.01
    # General motion profiles are selected from physical load state, never a
    # scene/object identity.  The unloaded contact move can use more of the
    # RM75 limits; a retained payload uses a gentler profile until release.
    # Keep both profiles inside the measured Gazebo tracking envelope. Under
    # shared GPU/physics load, the previous profiles repeatedly exceeded the
    # former 0.05 rad controller limit on joint 5. A later GUI-on run reached
    # 0.060102 rad on joint 4 at 0.18/0.08, so the unloaded profile also keeps
    # measured margin below the explicit 0.06 rad path envelope. A later
    # GUI-on loaded trajectory reached 0.060016 rad at 0.12/0.06 and was
    # correctly aborted mid-path. Keep payload transport more conservative
    # instead of weakening that controller envelope. The unchanged 0.002 rad
    # terminal goal continues to prove final settling.
    unloaded_velocity_scaling: float = 0.16
    unloaded_acceleration_scaling: float = 0.06
    loaded_velocity_scaling: float = 0.10
    loaded_acceleration_scaling: float = 0.04

    def __post_init__(self) -> None:
        GazeboControlConfig.__post_init__(self)
        contract = self.acceptance_scene_contract
        assignments = self.sort_assignments
        if (
            not isinstance(self.active_sort_assignment_index, int)
            or isinstance(self.active_sort_assignment_index, bool)
            or not 0 <= self.active_sort_assignment_index < len(assignments)
        ):
            raise ValueError("active sort assignment index is invalid")
        assignment = assignments[self.active_sort_assignment_index]
        bindings = scene_target_bindings(contract)
        if len(bindings) != len(assignments):
            raise ValueError("acceptance scene sort binding count is invalid")
        binding = bindings[self.active_sort_assignment_index]
        object.__setattr__(self, "target_id", binding.target_model)
        object.__setattr__(self, "target_link", binding.target_link)
        object.__setattr__(self, "attach_topic", binding.attach_topic)
        object.__setattr__(self, "detach_topic", binding.detach_topic)
        object.__setattr__(self, "state_topic", binding.state_topic)
        object.__setattr__(
            self,
            "attached_collision_filter_state_topic",
            binding.collision_filter_state_topic,
        )
        object.__setattr__(
            self,
            "attached_collision_filter_state_request_topic",
            binding.collision_filter_state_request_topic,
        )
        object.__setattr__(
            self,
            "attached_collision_filter_state_ack_topic",
            binding.collision_filter_state_ack_topic,
        )
        regions = contract.get("placement_regions")
        active_region = next(
            (
                region
                for region in (regions if isinstance(regions, list) else [])
                if isinstance(region, Mapping)
                and str(region.get("id") or "")
                == str(assignment["placement_region_id"])
            ),
            None,
        )
        destination = (
            active_region.get("center_xy")
            if isinstance(active_region, Mapping)
            else contract.get("destination_center_xy")
        )
        if destination is not None:
            object.__setattr__(
                self,
                "destination_center_xy",
                tuple(float(value) for value in destination),
            )
        destination_size = (
            active_region.get("size_xy_m")
            if isinstance(active_region, Mapping)
            else contract.get("destination_size_xy_m")
        )
        if destination_size is not None:
            object.__setattr__(
                self,
                "destination_size_xy_m",
                tuple(float(value) for value in destination_size),
            )
        destination_support = (
            active_region.get("support_z_m")
            if isinstance(active_region, Mapping)
            else contract.get("destination_support_z_m")
        )
        if destination_support is not None:
            object.__setattr__(
                self,
                "destination_support_z_m",
                float(destination_support),
            )
        placement_semantics = str(
            self.placement_acceptance_semantics
            or (
                active_region.get("acceptance_semantics")
                if isinstance(active_region, Mapping)
                else None
            )
            or contract.get("placement_acceptance_semantics")
            or PLACEMENT_ACCEPTANCE_COMPLETE_FOOTPRINT
        )
        if placement_semantics not in PLACEMENT_ACCEPTANCE_SEMANTICS:
            raise ValueError("placement acceptance semantics are invalid")
        object.__setattr__(
            self,
            "placement_acceptance_semantics",
            placement_semantics,
        )
        motion_scalings = (
            self.unloaded_velocity_scaling,
            self.unloaded_acceleration_scaling,
            self.loaded_velocity_scaling,
            self.loaded_acceleration_scaling,
        )
        if any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in motion_scalings
        ):
            raise ValueError("pick/place motion scaling is invalid")
        if (
            self.loaded_velocity_scaling > self.unloaded_velocity_scaling
            or self.loaded_acceleration_scaling > self.unloaded_acceleration_scaling
        ):
            raise ValueError("loaded motion profile cannot exceed unloaded profile")
        if (
            not math.isfinite(self.placement_release_z_offset_m)
            or self.placement_release_z_offset_m < 0.0
        ):
            raise ValueError("placement release Z offset must be finite and non-negative")
        if (
            not self.attached_collision_filter_state_topic
            or not self.attached_collision_filter_state_request_topic
            or not self.attached_collision_filter_state_ack_topic
            or self.robot_collision_filter_mask <= 0
            or self.detached_target_collision_filter_mask <= 0
            or self.attached_target_collision_filter_mask <= 0
            or (
                self.robot_collision_filter_mask
                & self.detached_target_collision_filter_mask
            )
            == 0
            or (
                self.robot_collision_filter_mask
                & self.attached_target_collision_filter_mask
            )
            != 0
        ):
            raise ValueError("attached collision-filter contract is invalid")
        target = contract.get("target_object")
        if isinstance(contract.get("sort_assignments"), list):
            authoritative_target = self.authoritative_scene.object(self.target_id)
            object.__setattr__(
                self,
                "target_size_m",
                tuple(float(value) for value in authoritative_target.bounding_box_xyz),
            )
            object.__setattr__(
                self,
                "target_initial_xyz",
                tuple(float(value) for value in authoritative_target.pose_xyz),
            )
            object.__setattr__(
                self,
                "target_initial_quat_xyzw",
                tuple(float(value) for value in authoritative_target.pose_quat_xyzw),
            )
        elif isinstance(target, Mapping):
            object.__setattr__(
                self,
                "target_size_m",
                tuple(float(value) for value in target["bounding_box_xyz"]),
            )
            object.__setattr__(
                self,
                "target_initial_xyz",
                tuple(float(value) for value in target["pose_xyz"]),
            )
            object.__setattr__(
                self,
                "target_initial_quat_xyzw",
                _quaternion_from_rpy(target["pose_rpy"]),
            )
        if isinstance(target, Mapping) and self.target_id == "target_object":
            object.__setattr__(self, "target_mass_kg", float(target["mass_kg"]))

    @property
    def acceptance_scene_contract(self) -> Mapping[str, Any]:
        return load_acceptance_scene_contract(self.acceptance_scene_id)

    @property
    def sort_assignments(self) -> tuple[Mapping[str, str], ...]:
        contract = self.acceptance_scene_contract
        configured = contract.get("sort_assignments")
        if isinstance(configured, list):
            return tuple(dict(item) for item in configured if isinstance(item, Mapping))
        task = contract.get("task")
        task = task if isinstance(task, Mapping) else {}
        return (
            {
                "id": "default",
                "target_object_id": "target_object",
                "target_link": "target_link",
                "target_prompt": str(task.get("target_prompt") or "target object"),
                "placement_object_prompt": str(
                    task.get("placement_object_prompt") or "target object"
                ),
                "placement_region_id": str(
                    contract.get("selected_placement_region_id") or self.table_id
                ),
                "placement_region_prompt": str(
                    task.get("placement_region_prompt") or "placement region"
                ),
            },
        )

    @property
    def active_sort_assignment(self) -> Mapping[str, str]:
        return self.sort_assignments[self.active_sort_assignment_index]

    @property
    def sort_assignment_configs(self) -> tuple["NativePickPlaceConfig", ...]:
        return tuple(
            replace(self, active_sort_assignment_index=index)
            for index in range(len(self.sort_assignments))
        )

    def for_sort_assignment(self, index: int) -> "NativePickPlaceConfig":
        return replace(self, active_sort_assignment_index=index)

    @property
    def authoritative_scene(self) -> CompiledAuthoritativeScene:
        """Return the one world artifact shared by Gazebo and MoveIt."""

        package = self.ros_workspace / "src" / self.ros_package_name
        return compile_authoritative_scene(
            base_world=package / "worlds/rm75_robotiq2f85_pickplace.sdf",
            catalog_path=package / "config/acceptance_scenes.json",
            scene_id=self.acceptance_scene_id,
        )

    @property
    def authoritative_scene_sha256(self) -> str:
        return self.authoritative_scene.authority_sha256

    @property
    def authoritative_table_spec(self) -> dict[str, Any]:
        return self.authoritative_scene.object(self.table_id).moveit_spec()

    @property
    def authoritative_dynamic_obstacle_ids(self) -> tuple[str, ...]:
        return tuple(
            object_id
            for object_id in self.authoritative_scene.dynamic_object_ids
            if object_id != self.target_id
        )

    @property
    def acceptance_scene_seed(self) -> int:
        return int(self.acceptance_scene_contract["seed"])

    @property
    def selected_placement_region_id(self) -> str:
        return str(self.active_sort_assignment["placement_region_id"])

    @property
    def static_obstacle_specs(self) -> tuple[dict[str, Any], ...]:
        """Exact non-table/non-target bodies extracted from the final SDF.

        ``static`` here means PlanningScene world geometry. Gazebo-dynamic
        clutter is tagged in each spec and its settled pose is overlaid from
        one native Pose_V snapshot during reset.
        """

        return tuple(
            item.moveit_spec()
            for item in self.authoritative_scene.objects
            if item.object_id not in {self.table_id, self.target_id}
        )

    @property
    def target_collision_primitives(self) -> tuple[dict[str, Any], ...]:
        target = self.authoritative_scene.object(self.target_id)
        return tuple(primitive.moveit_spec() for primitive in target.primitives)

    def acceptance_scene_evidence(self) -> dict[str, Any]:
        contract = self.acceptance_scene_contract
        evidence = {
            "schema_version": ACCEPTANCE_SCENE_SCHEMA_VERSION,
            "scene_id": self.acceptance_scene_id,
            "world_scene": str(contract["world_scene"]),
            "seed": int(contract["seed"]),
            "contract_sha256": str(contract["contract_sha256"]),
            "static_obstacle_ids": [
                str(obstacle["id"]) for obstacle in contract["static_obstacles"]
            ],
            "authoritative_world": self.authoritative_scene.evidence(),
            "destination_center_xy": list(self.destination_center_xy),
            "destination_size_xy_m": list(self.destination_size_xy_m),
            "destination_support_z_m": self.destination_support_z_m,
            "placement_release_z_offset_m": self.placement_release_z_offset_m,
            "placement_acceptance_semantics": (self.placement_acceptance_semantics),
            "sort_progress": {
                "schema_version": "openeta.multi_sort_progress.v1",
                "assignment_count": len(self.sort_assignments),
                "active_assignment_index": self.active_sort_assignment_index,
                "active_assignment": dict(self.active_sort_assignment),
            },
            "attached_collision_filter": {
                "schema_version": "openeta.attached_collision_filter.v1",
                "state_topic": self.attached_collision_filter_state_topic,
                "state_request_topic": (
                    self.attached_collision_filter_state_request_topic
                ),
                "state_ack_topic": self.attached_collision_filter_state_ack_topic,
                "robot_mask": self.robot_collision_filter_mask,
                "detached_target_mask": self.detached_target_collision_filter_mask,
                "attached_target_mask": self.attached_target_collision_filter_mask,
                "attached_target_robot_collision_enabled": False,
                "attached_target_environment_collision_enabled": True,
            },
        }
        task = contract.get("task")
        if isinstance(task, Mapping):
            evidence["task"] = dict(task)
        target = contract.get("target_object")
        collision_primitives = self.target_collision_primitives
        evidence["target_object"] = {
            "id": self.target_id,
            "shape_class": str(
                target.get("shape_class")
                if isinstance(target, Mapping) and self.target_id == "target_object"
                else self.active_sort_assignment.get("shape_class")
                or self.target_id
            ),
            "bounding_box_xyz": list(self.target_size_m),
            "collision_model": (
                "compound_primitives" if collision_primitives else "bounding_box"
            ),
            "collision_primitive_count": len(collision_primitives),
        }
        regions = contract.get("placement_regions")
        if isinstance(regions, list):
            evidence["placement_region_ids"] = [str(region["id"]) for region in regions]
            evidence["selected_placement_region_id"] = self.selected_placement_region_id
        return evidence

    @property
    def replace_default_distractor(self) -> bool:
        return self.acceptance_scene_contract.get("replace_default_distractor") is True

    @property
    def reset_object_poses(self) -> Mapping[str, tuple[float, float, float]]:
        poses = {self.target_id: self.target_initial_xyz}
        if not self.replace_default_distractor:
            poses[self.distractor_id] = self.distractor_initial_xyz
        return poses

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
            package / "config/acceptance_scenes.json",
            package / "launch/gazebo_pickplace.launch.py",
            package / "urdf/rm75_robotiq2f85_pickplace.urdf.xacro",
            package / "worlds/rm75_robotiq2f85_pickplace.sdf",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError("MODEL_ASSET_NOT_FOUND")
        # Compile the complete visual/physics/MoveIt world before Gazebo
        # allocates any process. Unsupported or divergent geometry therefore
        # fails at the lifecycle boundary, not during a motion attempt.
        self.acceptance_scene_contract
        self.authoritative_scene


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
    qx, qy, qz, qw = final.quat_xyzw
    quaternion_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if quaternion_norm <= 1e-12:
        return PlacementVerification(Verdict.UNKNOWN, PlacementReasonCode.POSE_UNAVAILABLE)
    qx, qy, qz, qw = (value / quaternion_norm for value in (qx, qy, qz, qw))
    rotation_rows = (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )
    try:
        projected_geometry = project_collision_geometry(
            object_xyz=final.xyz,
            object_rotation=rotation_rows,
            primitives=cfg.target_collision_primitives,
            fallback_size_xyz=cfg.target_size_m,
        )
        bounds = compound_axis_aligned_bounds(projected_geometry)
        geometry_centroid = collision_geometry_volume_centroid(projected_geometry)
        radius_m = orientation_invariant_radius_m(
            projected_geometry,
            object_xyz=final.xyz,
        )
    except ValueError:
        return PlacementVerification(Verdict.UNKNOWN, PlacementReasonCode.POSE_UNAVAILABLE)
    half_x = cfg.destination_size_xy_m[0] / 2.0
    half_y = cfg.destination_size_xy_m[1] / 2.0
    x_margin_m = min(
        bounds.minimum_xyz[0] - (cfg.destination_center_xy[0] - half_x),
        cfg.destination_center_xy[0] + half_x - bounds.maximum_xyz[0],
    )
    y_margin_m = min(
        bounds.minimum_xyz[1] - (cfg.destination_center_xy[1] - half_y),
        cfg.destination_center_xy[1] + half_y - bounds.maximum_xyz[1],
    )
    centroid_x_margin_m = half_x - abs(geometry_centroid[0] - cfg.destination_center_xy[0])
    centroid_y_margin_m = half_y - abs(geometry_centroid[1] - cfg.destination_center_xy[1])
    support_height_error_m = abs(bounds.minimum_xyz[2] - cfg.destination_support_z_m)
    expected_link_origin_height_m = (
        final.xyz[2] + cfg.destination_support_z_m - bounds.minimum_xyz[2]
    )
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
        and terminal[-1].monotonic_s - terminal[0].monotonic_s < cfg.placement_stability_duration_s
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
        "height_rule": "compound_collision_geometry_contacts_destination_plane",
        "geometry_source": (
            "compound_collision_primitives"
            if cfg.target_collision_primitives
            else "centered_bounding_box"
        ),
        "collision_primitive_count": len(projected_geometry),
        "support_plane_height_m": cfg.destination_support_z_m,
        "projected_collision_minimum_z_m": bounds.minimum_xyz[2],
        "projected_collision_maximum_z_m": bounds.maximum_xyz[2],
        "expected_link_origin_height_m": expected_link_origin_height_m,
        "support_height_tolerance_m": cfg.placement_support_height_tolerance_m,
        "support_height_error_m": support_height_error_m,
        "destination_center_xy": list(cfg.destination_center_xy),
        "destination_size_xy_m": list(cfg.destination_size_xy_m),
        "placement_acceptance_semantics": cfg.placement_acceptance_semantics,
        "geometry_volume_centroid_xyz": list(geometry_centroid),
        "centroid_margin_xy_m": [centroid_x_margin_m, centroid_y_margin_m],
        "complete_footprint_inside": min(x_margin_m, y_margin_m) >= 0.0,
        "conservative_footprint_radius_m": radius_m,
        "projected_footprint_half_extent_xy_m": [
            (bounds.maximum_xyz[0] - bounds.minimum_xyz[0]) / 2.0,
            (bounds.maximum_xyz[1] - bounds.minimum_xyz[1]) / 2.0,
        ],
        "projected_footprint_bounds_xy_m": {
            "minimum": list(bounds.minimum_xyz[:2]),
            "maximum": list(bounds.maximum_xyz[:2]),
        },
        "footprint_margin_xy_m": [x_margin_m, y_margin_m],
    }
    if len(terminal) < 2:
        return PlacementVerification(
            Verdict.UNKNOWN, PlacementReasonCode.POSE_UNAVAILABLE, evidence
        )
    if stable_duration_s + 1e-9 < cfg.placement_stability_duration_s:
        return PlacementVerification(
            Verdict.UNKNOWN, PlacementReasonCode.OBSERVATION_TOO_SHORT, evidence
        )
    if terminal_drift_m > cfg.maximum_placement_terminal_drift_m:
        return PlacementVerification(Verdict.FAIL, PlacementReasonCode.TERMINAL_DRIFT, evidence)
    if support_height_error_m > cfg.placement_support_height_tolerance_m and not math.isclose(
        support_height_error_m,
        cfg.placement_support_height_tolerance_m,
        rel_tol=0.0,
        abs_tol=_POSE_BOUNDARY_ABS_TOL_M,
    ):
        return PlacementVerification(
            Verdict.FAIL, PlacementReasonCode.HEIGHT_OUT_OF_RANGE, evidence
        )
    if (
        cfg.placement_acceptance_semantics == PLACEMENT_ACCEPTANCE_STABLE_GEOMETRY_CENTROID
        and min(centroid_x_margin_m, centroid_y_margin_m) < 0.0
    ):
        return PlacementVerification(
            Verdict.FAIL,
            PlacementReasonCode.CENTROID_OUTSIDE_DESTINATION,
            evidence,
        )
    if (
        cfg.placement_acceptance_semantics == PLACEMENT_ACCEPTANCE_COMPLETE_FOOTPRINT
        and min(x_margin_m, y_margin_m) < 0.0
    ):
        return PlacementVerification(
            Verdict.FAIL, PlacementReasonCode.FOOTPRINT_OUTSIDE_DESTINATION, evidence
        )
    return PlacementVerification(Verdict.PASS, PlacementReasonCode.PLACED, evidence)


def validated_pickplace_motion_guidance(
    config: NativePickPlaceConfig | None = None,
) -> dict[str, Any]:
    """Return the stable atomic motion and receipt gates for the fixture."""

    cfg = config or NativePickPlaceConfig()

    return {
        "schema_version": "openeta.gazebo.model_terminal_pickplace.v2",
        "pose_semantics": cfg.parent_link,
        "acceptance_scene": cfg.acceptance_scene_evidence(),
        "motion_parameters": {
            "profile": "unloaded",
            "velocity_scaling": cfg.unloaded_velocity_scaling,
            "acceleration_scaling": cfg.unloaded_acceleration_scaling,
            "tolerance": 0.0002,
            "ori_tolerance": 0.002,
        },
        "motion_profiles": {
            "unloaded": {
                "selection": "no_attached_payload",
                "velocity_scaling": cfg.unloaded_velocity_scaling,
                "acceleration_scaling": cfg.unloaded_acceleration_scaling,
            },
            "loaded": {
                "selection": "verified_attached_payload",
                "velocity_scaling": cfg.loaded_velocity_scaling,
                "acceleration_scaling": cfg.loaded_acceleration_scaling,
            },
            "terminal_time_parameterization": "moveit",
        },
        "terminal_poses": {
            "grasp_contact": "grasp_provider_model_pose_after_calibrated_frame_transform",
            "placement_release": "anyplace_object_goal_times_inverse_measured_attachment",
            "path_owner": "moveit",
            "host_pose_offsets_forbidden": True,
        },
        "atomic_order": [
            {"tool": "move_to", "pose": "grasp_contact", "path": "moveit_full_path"},
            {
                "tool": "gripper_control",
                "position": 0,
                "requires_receipt": ["native_bilateral_contact", "attached_ack"],
            },
            {
                "tool": "move_to",
                "pose": "placement_release",
                "path": "moveit_full_attached_object_path",
                "requires_receipt": ["attached_ack", "retention_drift_bound"],
            },
            {
                "tool": "gripper_control",
                "position": 1,
                "requires_receipt": ["detached_ack", "stable_placement"],
            },
        ],
        "success_evidence": {
            "grasp_admission": "bilateral_native_contact_and_attach_ack",
            "maximum_capture_relative_translation_m": (cfg.maximum_capture_relative_translation_m),
            "placement": {
                "minimum_stability_duration_s": cfg.placement_stability_duration_s,
                "maximum_terminal_drift_m": cfg.maximum_placement_terminal_drift_m,
                "support_plane_height_m": cfg.destination_support_z_m,
                "height_rule": "compound_collision_geometry_contacts_destination_plane",
                "support_height_tolerance_m": cfg.placement_support_height_tolerance_m,
                "destination_center_xy": list(cfg.destination_center_xy),
                "destination_size_xy_m": list(cfg.destination_size_xy_m),
                "footprint_rule": (
                    "stable_geometry_centroid_inside"
                    if cfg.placement_acceptance_semantics
                    == PLACEMENT_ACCEPTANCE_STABLE_GEOMETRY_CENTROID
                    else "compound_collision_projection_fully_inside"
                ),
                "complete_footprint_margin_role": (
                    "ordering_and_evidence_only"
                    if cfg.placement_acceptance_semantics
                    == PLACEMENT_ACCEPTANCE_STABLE_GEOMETRY_CENTROID
                    else "acceptance_gate"
                ),
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
    verification_window_started_sim_time_s: float | None,
    verification_window_ended_sim_time_s: float | None = None,
    now_monotonic_s: float,
    config: NativePickPlaceConfig | None = None,
) -> ContactGateResult:
    """Accept only stable bilateral target contacts in an explicit window.

    Every observed message in the armed window is inspected.  Therefore a
    distractor, unknown, or mixed event rejects the entire request instead of
    silently selecting the later target-looking contacts.  When the caller
    supplies a closed Gazebo-time window, both boundaries share the simulator
    clock and wall-clock receive age is deliberately irrelevant.  This keeps
    GPU rendering or a low real-time factor from making valid terminal contact
    evidence look stale while retaining the wall-clock freshness gate for
    open-ended callers.
    """

    cfg = config or NativePickPlaceConfig()
    ordered = sorted(samples, key=lambda item: (item.timestamp_s, item.side))
    by_side: dict[str, list[NativeContactSample]] = {"left": [], "right": []}
    if verification_window_started_sim_time_s is None or not math.isfinite(
        verification_window_started_sim_time_s
    ):
        return ContactGateResult(False, ReasonCode.CONTACT_SAMPLE_BEFORE_WINDOW, 0, 0)
    closed_window = (
        verification_window_ended_sim_time_s is not None
        and math.isfinite(verification_window_ended_sim_time_s)
        and verification_window_ended_sim_time_s > verification_window_started_sim_time_s
    )
    for sample in ordered:
        if (
            not closed_window
            and now_monotonic_s - sample.received_monotonic_s > cfg.contact_freshness_s
        ):
            return ContactGateResult(
                False,
                ReasonCode.CONTACT_SAMPLE_STALE,
                len(by_side["left"]),
                len(by_side["right"]),
            )
        if sample.timestamp_s <= verification_window_started_sim_time_s:
            return ContactGateResult(
                False,
                ReasonCode.CONTACT_SAMPLE_BEFORE_WINDOW,
                len(by_side["left"]),
                len(by_side["right"]),
            )
        if closed_window and sample.timestamp_s > verification_window_ended_sim_time_s:
            return ContactGateResult(
                False,
                ReasonCode.CONTACT_SAMPLE_AFTER_WINDOW,
                len(by_side["left"]),
                len(by_side["right"]),
            )
        rejected = _identity_kind(sample.collision_names, cfg, sample.side)
        if rejected is not None:
            return ContactGateResult(
                False,
                rejected,
                len(by_side["left"]),
                len(by_side["right"]),
                evidence={"collision_names": list(sample.collision_names), "side": sample.side},
            )
        by_side[sample.side].append(sample)

    left, right = by_side["left"], by_side["right"]
    left_span = left[-1].timestamp_s - left[0].timestamp_s if len(left) > 1 else 0.0
    right_span = right[-1].timestamp_s - right[0].timestamp_s if len(right) > 1 else 0.0
    if len(left) < cfg.contact_samples_required or len(right) < cfg.contact_samples_required:
        return ContactGateResult(
            False,
            ReasonCode.CONTACT_INSUFFICIENT_SAMPLES,
            len(left),
            len(right),
            left_span,
            right_span,
        )
    if left_span < cfg.contact_span_s or right_span < cfg.contact_span_s:
        return ContactGateResult(
            False, ReasonCode.CONTACT_WINDOW_TOO_SHORT, len(left), len(right), left_span, right_span
        )
    return ContactGateResult(
        True,
        ReasonCode.CONTACT_TARGET_CONFIRMED,
        len(left),
        len(right),
        left_span,
        right_span,
        evidence={
            "target_id": cfg.target_id,
            "source": "gazebo_native_contacts",
            "verification_window_started_sim_time_s": (verification_window_started_sim_time_s),
            "verification_window_ended_sim_time_s": (
                verification_window_ended_sim_time_s if closed_window else None
            ),
            "sample_freshness_basis": (
                "closed_gazebo_sim_time_window" if closed_window else "monotonic_receive_time"
            ),
        },
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
    def vertical_displacement_m(self) -> float:
        """Observed displacement telemetry; never an attachment threshold."""

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
            return self._remember(
                self._record(Verdict.FAIL, gate.reason_code, False, gate=gate.to_dict())
            )
        if not attach_acked:
            self.phase = "attach_unacknowledged"
            self.attached = False
            return self._remember(
                self._record(
                    Verdict.FAIL, ReasonCode.ATTACH_ACK_MISSING, False, gate=gate.to_dict()
                )
            )
        self.phase = "attachment_confirmed"
        self.attached = True
        return self._remember(
            self._record(
                Verdict.PASS,
                ReasonCode.ATTACHMENT_CONFIRMED,
                True,
                gate=gate.to_dict(),
                proof_boundary="bilateral_native_contact_and_attach_ack",
            )
        )

    def attachment_state_rejected(
        self,
        gate: ContactGateResult,
        *,
        detail: str,
    ) -> VerificationRecord:
        """Record a measured attached pose rejected by PlanningScene.

        Native bilateral contact and the detachable-joint ACK both occurred,
        but the measured object transform made the current robot/object state
        collide.  The native and PlanningScene bindings have already been
        rolled back before this candidate-level result is emitted.
        """

        self.phase = "attachment_rejected"
        self.attached = False
        return self._remember(
            self._record(
                Verdict.FAIL,
                ReasonCode.ATTACHMENT_STATE_INVALID,
                False,
                gate=gate.to_dict(),
                native_attach_acked_before_rollback=True,
                detail=str(detail),
            )
        )

    def prove_retention(
        self, proof: ChildLinkProof | None, *, dart_supported: bool = True
    ) -> VerificationRecord:
        """Revalidate a contact/attach-proven object during any MoveIt transport."""

        if not self.attached:
            return self._remember(self._record(Verdict.FAIL, ReasonCode.ATTACH_ACK_MISSING, False))
        if not dart_supported:
            self.phase = "dart_unsupported"
            return self._remember(self._record(Verdict.FAIL, ReasonCode.DART_UNSUPPORTED, False))
        if proof is None:
            self.phase = "child_link_unavailable"
            return self._remember(
                self._record(Verdict.FAIL, ReasonCode.CHILD_LINK_STATE_UNAVAILABLE, False)
            )
        evidence = {
            "source": "gazebo_pose_info_child_link",
            "vertical_displacement_m": proof.vertical_displacement_m,
            "capture_relative_translation_m": proof.capture_relative_translation_m,
            "prior_attachment_confirmed": True,
            "maximum_capture_relative_translation_m": (
                self.config.maximum_capture_relative_translation_m
            ),
        }
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
        return self._remember(self._record(Verdict.PASS, ReasonCode.TARGET_HELD, True, **evidence))

    def release_result(self, *, detached_acked: bool) -> VerificationRecord:
        self.attached = False
        self.phase = "released" if detached_acked else "release_unacknowledged"
        return self._remember(
            self._record(
                Verdict.UNKNOWN if detached_acked else Verdict.FAIL,
                ReasonCode.READY if detached_acked else ReasonCode.RELEASE_ACK_MISSING,
                False,
            )
        )

    def detached_open_result(self) -> VerificationRecord:
        """Acknowledge an open command when no grasp is attached."""

        self.phase = "ready"
        self.attached = False
        return self._remember(self._record(Verdict.UNKNOWN, ReasonCode.READY, False))

    def _record(
        self, verdict: Verdict, reason: ReasonCode, grasp: bool, **evidence: Any
    ) -> VerificationRecord:
        return VerificationRecord(
            self.phase, verdict, reason, self.config.target_id, grasp, evidence
        )

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
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )
