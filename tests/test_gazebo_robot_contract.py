from __future__ import annotations

import hashlib
import json
import time

import pytest

from extensions.gazebo.robot_control import (
    ARM_JOINTS,
    JOINT_NAMES,
    START_STATE_BOUNDS_TOLERANCE_RAD,
    START_STATE_RECOVERY_MAX_VIOLATION_RAD,
    GazeboControlConfig,
    GazeboController,
    MODEL_ID,
    assess_start_state_bounds,
    gripper_state,
    make_move_group_goal,
    robot_state_from_sources,
)
from extensions.gazebo.native_grasp import NativePickPlaceConfig
from extensions.gazebo.robotiq_kinematics import (
    DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD,
    minimum_feasible_active_position,
    six_joint_positions,
)
from sim.env_registry import get_env_spec


def _state(active=0.0):
    gripper = [active] + [0.0] * (len(JOINT_NAMES) - 8)
    return robot_state_from_sources(
        {"name": JOINT_NAMES, "position": [0] * 7 + gripper, "velocity": [0] * len(JOINT_NAMES)},
        {"base_link->gripper_mount_link": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]}},
    )


def _bounded_state(arm_positions=None):
    state = _state()
    state.joint_positions[:7] = list(arm_positions or [0.0] * 7)
    state.metadata.update(
        {
            "joint_state_timestamp_s": 42.0,
            "joint_state_received_monotonic_s": time.monotonic(),
        }
    )
    return state


@pytest.mark.parametrize(
    ("joint_3", "classification"),
    [
        (0.0, "WITHIN_BOUNDS"),
        (3.106, "RECOVERABLE"),
        (3.106 + 4.5e-13, "RECOVERABLE"),
        (3.106 + START_STATE_BOUNDS_TOLERANCE_RAD, "RECOVERABLE"),
        (3.106 + 48.3665e-6, "RECOVERABLE"),
        (3.106 + START_STATE_RECOVERY_MAX_VIOLATION_RAD, "RECOVERABLE"),
        (3.106 + START_STATE_RECOVERY_MAX_VIOLATION_RAD + 1e-7, "INVALID"),
    ],
)
def test_start_state_bounds_classifies_numeric_boundary_cases(joint_3, classification) -> None:
    positions = [0.0] * 7
    positions[2] = joint_3
    assessment = assess_start_state_bounds(_bounded_state(positions))

    assert assessment["classification"] == classification
    if classification == "RECOVERABLE":
        assert assessment["candidate_positions"][2] == pytest.approx(3.105)
        assert assessment["joints"][0]["name"] == "joint_3"
        assert assessment["max_recovery_violation_rad"] == pytest.approx(0.001)


def test_controller_exposes_stationary_failed_motion_as_current_state_restart() -> None:
    start = _bounded_state()
    terminal = _bounded_state()
    terminal.joint_positions[5] = 2.2340483665
    terminal.end_effector_pose["xyz"] = [0.00639, 0.0, 0.5]
    controller = GazeboController(
        state_provider=lambda: start,
        failed_motion_terminal_state_provider=lambda _barrier: terminal,
        move_action=lambda _goal, _timeout: {
            "ok": False,
            "error_code": "MOTION_EXECUTION_FAILED",
            "motion_outcome": "failed",
            "moveit_error_code": -4,
            "planned_point_count": 64,
            "execution_started": True,
            "action_completed_ros_time_s": 14.0,
        },
    )

    receipt = controller.execute(
        {
            "action_type": "move_to",
            "target_pose": {
                "xyz": [0.0, 0.0, 0.5],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    ).to_dict()

    assert receipt["ok"] is False
    assert receipt["error_code"] == "MOTION_EXECUTION_FAILED"
    assert receipt["motion_outcome"] == "failed"
    restart = receipt["current_state_restart"]
    assert restart["status"] == "PASS"
    assert restart["reason_code"] == "KNOWN_STATIONARY_TERMINAL_FAILURE"
    assert restart["start_state_bounds"]["status"] == "RECOVERY_REQUIRED"
    assert restart["start_state_bounds"]["reason_code"] == (
        "BOUNDED_CONTROLLER_ENDPOINT_NORMALIZATION"
    )


def test_start_state_bounds_recovers_only_affected_joints() -> None:
    positions = [0.0] * 7
    positions[0] = -3.106 - 4.5e-13
    positions[4] = 3.106 + 4.5e-13

    assessment = assess_start_state_bounds(_bounded_state(positions))

    assert assessment["classification"] == "RECOVERABLE"
    assert assessment["candidate_positions"] == pytest.approx(
        [-3.105, 0.0, 0.0, 0.0, 3.105, 0.0, 0.0]
    )
    assert [item["name"] for item in assessment["joints"]] == [
        "joint_1",
        "joint_5",
    ]


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("missing", "ARM_JOINT_MISSING"),
        ("nan", "ARM_JOINT_NONFINITE"),
        ("stale", "JOINT_STATE_STALE"),
        ("timestamp_missing", "JOINT_STATE_TIMESTAMP_MISSING"),
        ("names_missing", "ARM_JOINT_NAMES_MISSING"),
    ],
)
def test_start_state_bounds_rejects_untrustworthy_state(mutation, reason_code) -> None:
    state = _bounded_state()
    if mutation == "missing":
        state.metadata["joint_names"] = list(JOINT_NAMES[1:])
        state.joint_positions = state.joint_positions[1:]
    elif mutation == "nan":
        state.joint_positions[3] = float("nan")
    elif mutation == "stale":
        state.metadata["joint_state_received_monotonic_s"] -= 3.0
    elif mutation == "names_missing":
        state.metadata.pop("joint_names")
    else:
        state.metadata.pop("joint_state_received_monotonic_s")

    assessment = assess_start_state_bounds(state)

    assert assessment["classification"] == "INVALID"
    assert assessment["reason_code"] == reason_code


def test_gazebo_environment_uses_canonical_display_name() -> None:
    spec = get_env_spec("openeta/gazebo_rm75_robotiq2f85-v0")
    assert spec is not None
    assert spec.display_name == "Gazebo 仿真环境"


def test_robot_names_opening_and_binary_command():
    cfg = GazeboControlConfig()
    feasible_open = minimum_feasible_active_position(
        boundary_inset_rad=DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD
    )
    cfg.validate_assets()
    assert cfg.maximum_aperture_m == 0.085
    assert gripper_state(0.0)["aperture_m"] == 0.085
    assert cfg.gripper_position(0) == pytest.approx(0.7929)
    assert cfg.gripper_position(1) == pytest.approx(feasible_open)
    for invalid in (False, 0.5, 1.0, -1, 2):
        with pytest.raises(ValueError):
            cfg.gripper_position(invalid)
    # Production validation uses the embedded closure even when the legacy
    # require_vendor argument is supplied; no workstation path is necessary.
    cfg.validate_assets(require_vendor=True)


def test_state_is_fail_closed_and_has_model_metadata():
    state = _state()
    assert state.gripper_state["openness"] == 1
    assert state.metadata["joint_names"] == list(JOINT_NAMES)
    with pytest.raises(RuntimeError, match="JOINT_STATE_TIMEOUT"):
        robot_state_from_sources({"name": [], "position": []}, {})


def test_move_goal_applies_inverse_mount_transform():
    goal = make_move_group_goal(
        {"xyz": [1, 0, 0], "quat_xyzw": [0, 0, 0, 1]},
        config=GazeboControlConfig(mount_xyz=(0.1, 0, 0), mount_quat_xyzw=(0, 0, 0, 1)),
    )
    assert goal["group_name"] == "rm_group" and goal["link_name"] == "link_7"
    assert goal["target_pose"]["xyz"] == pytest.approx([0.9, 0, 0])


def test_move_goal_applies_inverse_quarter_turn_gripper_mount() -> None:
    goal = make_move_group_goal({"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]})

    assert goal["target_pose"]["quat_xyzw"] == pytest.approx([0.0, 0.0, -(2**-0.5), 2**-0.5])
    assert goal["requested_tool_pose"]["quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]


def test_move_goal_accepts_public_tool_tolerance_names() -> None:
    goal = make_move_group_goal(
        {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
        tolerances={
            "tolerance": 0.0002,
            "ori_tolerance": 0.002,
            "velocity_scaling": 0.3,
            "acceleration_scaling": 0.2,
        },
    )

    assert goal["position_tolerance_m"] == 0.0002
    assert goal["orientation_tolerance_rad"] == 0.002
    assert goal["max_velocity_scaling_factor"] == 0.3
    assert goal["max_acceleration_scaling_factor"] == 0.2


def test_pickplace_motion_profile_follows_verified_payload_state() -> None:
    config = NativePickPlaceConfig()
    unloaded = make_move_group_goal(
        {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1], "grasp_stage": "contact"},
        config=config,
    )
    loaded = make_move_group_goal(
        {
            "xyz": [0, 0, 0.5],
            "quat_xyzw": [0, 0, 0, 1],
            "placement_stage": "release",
        },
        config=config,
    )

    assert (unloaded["motion_profile"], loaded["motion_profile"]) == (
        "unloaded",
        "loaded",
    )
    assert unloaded["max_velocity_scaling_factor"] == 0.10
    assert unloaded["max_acceleration_scaling_factor"] == 0.04
    assert loaded["max_velocity_scaling_factor"] == 0.10
    assert loaded["max_acceleration_scaling_factor"] == 0.04


def test_move_goal_preserves_a_hash_bound_qualified_joint_branch() -> None:
    state = {
        "names": list(ARM_JOINTS),
        "positions": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
    }
    state_hash = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    goal = make_move_group_goal(
        {
            "xyz": [0, 0, 0.5],
            "quat_xyzw": [0, 0, 0, 1],
            "qualification_goal_joint_state": state,
            "qualification_goal_joint_state_sha256": state_hash,
            "qualification_binding_sha256": "b" * 64,
        }
    )

    assert goal["qualification_goal_joint_state"] == state
    assert goal["qualification_goal_joint_state_sha256"] == state_hash
    assert goal["qualification_binding_sha256"] == "b" * 64


def test_move_goal_preserves_hash_bound_contact_collision_policy() -> None:
    state = {
        "names": list(ARM_JOINTS),
        "positions": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
    }
    state_hash = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    allowed = {"target_object": ["left_tip", "right_tip"]}
    allowed_hash = hashlib.sha256(
        json.dumps(allowed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    goal = make_move_group_goal(
        {
            "xyz": [0, 0, 0.5],
            "quat_xyzw": [0, 0, 0, 1],
            "compiled_grasp_id": "grasp-7",
            "grasp_stage": "contact",
            "qualification_goal_joint_state": state,
            "qualification_goal_joint_state_sha256": state_hash,
            "qualification_binding_sha256": "b" * 64,
            "qualification_allowed_collisions": allowed,
            "qualification_allowed_collisions_sha256": allowed_hash,
        }
    )

    assert goal["qualification_allowed_collisions"] == allowed
    assert goal["qualification_allowed_collisions_sha256"] == allowed_hash


def test_move_goal_preserves_hash_bound_contact_policy_for_exact_restore() -> None:
    allowed = {"target_object": ["left_tip", "right_tip"]}
    allowed_hash = hashlib.sha256(
        json.dumps(allowed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    goal = make_move_group_goal(
        {
            "xyz": [0, 0, 0.5],
            "quat_xyzw": [0, 0, 0, 1],
            "purpose": "grasp_recovery_restore",
            "recovery_id": "grasp-recovery-test",
            "compiled_grasp_id": "grasp-7",
            "grasp_stage": "recovery_restore",
            "qualification_allowed_collisions": allowed,
            "qualification_allowed_collisions_sha256": allowed_hash,
        }
    )

    assert goal["purpose"] == "grasp_recovery_restore"
    assert goal["recovery_id"] == "grasp-recovery-test"
    assert goal["grasp_stage"] == "recovery_restore"
    assert goal["qualification_allowed_collisions"] == allowed
    assert goal["qualification_allowed_collisions_sha256"] == allowed_hash


def test_move_goal_rejects_contact_policy_on_an_unbound_ordinary_move() -> None:
    allowed = {"target_object": ["left_tip", "right_tip"]}
    allowed_hash = hashlib.sha256(
        json.dumps(allowed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="only valid for a compiled grasp"):
        make_move_group_goal(
            {
                "xyz": [0, 0, 0.5],
                "quat_xyzw": [0, 0, 0, 1],
                "qualification_allowed_collisions": allowed,
                "qualification_allowed_collisions_sha256": allowed_hash,
            }
        )


@pytest.mark.parametrize("mutation", ["position", "hash", "binding", "names"])
def test_move_goal_rejects_corrupted_qualified_joint_branch(mutation: str) -> None:
    state = {
        "names": list(ARM_JOINTS),
        "positions": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
    }
    target = {
        "xyz": [0, 0, 0.5],
        "quat_xyzw": [0, 0, 0, 1],
        "qualification_goal_joint_state": state,
        "qualification_goal_joint_state_sha256": hashlib.sha256(
            json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "qualification_binding_sha256": "b" * 64,
    }
    if mutation == "position":
        state["positions"][0] = 3.2
    elif mutation == "hash":
        target["qualification_goal_joint_state_sha256"] = "0" * 64
    elif mutation == "binding":
        target["qualification_binding_sha256"] = "not-a-proof"
    else:
        state["names"] = list(reversed(ARM_JOINTS))

    with pytest.raises(ValueError, match="qualified|qualification"):
        make_move_group_goal(target)


def test_controller_routes_actions_and_unknown_outcome():
    sent = []
    ctl = GazeboController(
        state_provider=_state,
        move_action=lambda goal, timeout: sent.append((goal, timeout)) or {"ok": True},
        gripper_action=lambda position, timeout: {"ok": True},
    )
    result = ctl.execute(
        {"action_type": "move_to", "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]}}
    ).to_dict()
    assert result["ok"] and result["reached_target"] and result["observation"]["robot"]
    assert ctl.execute({"action_type": "gripper_open"}).ok
    assert ctl.execute({"action_type": "other"}).error_code == "INVALID_CONTROL_ACTION"


def test_qualification_exception_is_reported_as_validated_infrastructure_error() -> None:
    def fail(_request):
        raise ValueError("bad real qualification payload")

    receipt = (
        GazeboController(
            state_provider=_state,
            candidate_qualifier=fail,
        )
        .execute(
            {
                "action_type": "qualify_motion_candidates",
                "schema_version": "openeta.moveit_candidate_funnel.v3",
                "planning_scene_revision": 4,
                "qualification_binding_sha256": "binding",
                "funnel": {"qualification_profile": "fast_v3"},
                "candidates": [
                    {
                        "candidate_id": "c0",
                        "candidate_pose_sha256": "pose",
                    }
                ],
            }
        )
        .to_dict()
    )

    assert receipt["ok"] is True
    assert receipt["stop_reason"] == "infrastructure_error"
    assert receipt["infrastructure_error"] is True
    assert receipt["qualification_profile"] == "fast_v3"
    assert receipt["results"] == [
        {
            "candidate_id": "c0",
            "candidate_pose_sha256": "pose",
            "qualification_binding_sha256": "binding",
            "execution_started": False,
            "verdict": "UNKNOWN",
            "reason": "qualification_infrastructure_error",
            "infrastructure_error": True,
            "infrastructure_error_detail": "ValueError: bad real qualification payload",
            "stages": [],
        }
    ]


def test_controller_rejects_moveit_success_when_fresh_pose_misses_goal() -> None:
    start, end = _state(), _state()
    end.end_effector_pose["xyz"] = [0.02, 0.0, 0.5]
    states = iter((start, end))
    controller = GazeboController(
        state_provider=lambda: next(states),
        move_action=lambda _goal, _timeout: {
            "ok": True,
            "reached_goal": True,
            "motion_outcome": "completed",
        },
    )

    receipt = controller.execute(
        {
            "action_type": "move_to",
            "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
            "tolerance": 0.0002,
            "ori_tolerance": 0.002,
        }
    ).to_dict()

    assert receipt["ok"] is False
    assert receipt["reached_target"] is False
    assert receipt["error_code"] == "MOTION_TARGET_NOT_REACHED"
    assert receipt["motion_outcome"] == "failed"
    assert receipt["position_error_m"] == pytest.approx(0.02)


def test_controller_accepts_only_a_fresh_settled_pose_within_same_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, immediate, settled = _state(), _state(), _state()
    immediate.end_effector_pose["xyz"] = [0.0049, 0.0, 0.5]
    settled.end_effector_pose["xyz"] = [0.003, 0.0, 0.5]
    states = iter((start, immediate, settled))
    monkeypatch.setattr("extensions.gazebo.robot_control.time.sleep", lambda _s: None)
    controller = GazeboController(
        state_provider=lambda: next(states),
        move_action=lambda _goal, _timeout: {
            "ok": True,
            "reached_goal": True,
            "motion_outcome": "completed",
        },
    )

    receipt = controller.execute(
        {
            "action_type": "move_to",
            "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
            "tolerance": 0.002,
            "ori_tolerance": 0.05,
        }
    ).to_dict()

    assert receipt["ok"] is True
    assert receipt["reached_target"] is True
    assert receipt["position_error_m"] == pytest.approx(0.003)
    assert receipt["position_verification_tolerance_m"] == pytest.approx(0.004)
    assert receipt["position_verification_numeric_margin_m"] == pytest.approx(0.0001)
    assert receipt["position_verification_effective_tolerance_m"] == pytest.approx(0.0041)
    assert receipt["settling_recheck"]["status"] == "target_verified"
    assert receipt["settling_recheck"]["sample_count"] == 1
    assert receipt["settling_recheck"]["initial_position_error_m"] == pytest.approx(0.0049)


def test_controller_accepts_only_the_bounded_tf_numeric_margin() -> None:
    start, inside = _state(), _state()
    inside.end_effector_pose["xyz"] = [0.004024, 0.0, 0.5]
    states = iter((start, inside))
    controller = GazeboController(
        state_provider=lambda: next(states),
        move_action=lambda _goal, _timeout: {
            "ok": True,
            "reached_goal": True,
            "motion_outcome": "completed",
        },
    )

    receipt = controller.execute(
        {
            "action_type": "move_to",
            "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
            "tolerance": 0.002,
            "ori_tolerance": 0.05,
        }
    ).to_dict()

    assert receipt["ok"] is True
    assert receipt["position_error_m"] == pytest.approx(0.004024)
    assert receipt["position_verification_tolerance_m"] == pytest.approx(0.004)
    assert receipt["position_verification_effective_tolerance_m"] == pytest.approx(0.0041)
    assert receipt["position_verification_policy"] == (
        "exact_terminal_euclidean_with_bounded_numeric_margin"
    )


def test_controller_reconciles_control_failed_only_at_exact_stationary_terminal() -> None:
    start, end = _state(), _state()
    end.end_effector_pose["xyz"] = [0.0007, 0.0, 0.5]
    states = iter((start, end))
    controller = GazeboController(
        state_provider=lambda: next(states),
        move_action=lambda _goal, _timeout: {
            "ok": False,
            "error_code": "MOTION_EXECUTION_FAILED",
            "motion_outcome": "failed",
            "moveit_error_code": -4,
            "planned_point_count": 129,
            "execution_started": True,
        },
    )

    receipt = controller.execute(
        {
            "action_type": "move_to",
            "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
            "tolerance": 0.002,
            "ori_tolerance": 0.05,
        }
    ).to_dict()

    assert receipt["ok"] is True
    assert receipt["reached_target"] is True
    assert receipt["motion_outcome"] == "completed"
    assert receipt["moveit_error_code"] == -4
    proof = receipt["terminal_reconciliation"]
    assert proof["status"] == "PASS"
    assert proof["reason_code"] == "CONTROL_FAILED_AFTER_EXACT_TARGET_REACHED"
    assert proof["original_error_code"] == "MOTION_EXECUTION_FAILED"
    assert proof["max_arm_velocity_rad_s"] == 0.0


@pytest.mark.parametrize("failure_mode", ["wrong_code", "moving", "off_target"])
def test_controller_does_not_reconcile_unproven_execution_failure(
    failure_mode: str,
) -> None:
    start, end = _state(), _state()
    if failure_mode == "moving":
        end.joint_velocities[0] = 0.01
    if failure_mode == "off_target":
        end.end_effector_pose["xyz"] = [0.02, 0.0, 0.5]
    states = iter((start, end))
    controller = GazeboController(
        state_provider=lambda: next(states),
        move_action=lambda _goal, _timeout: {
            "ok": False,
            "error_code": "MOTION_EXECUTION_FAILED",
            "motion_outcome": "failed",
            "moveit_error_code": -5 if failure_mode == "wrong_code" else -4,
            "planned_point_count": 20,
            "execution_started": True,
        },
    )

    receipt = controller.execute(
        {
            "action_type": "move_to",
            "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
            "tolerance": 0.002,
            "ori_tolerance": 0.05,
        }
    ).to_dict()

    assert receipt["ok"] is False
    assert receipt["reached_target"] is False
    assert "terminal_reconciliation" not in receipt


def test_controller_rejects_upward_release_residual_outside_exact_terminal() -> None:
    start, end = _state(), _state()
    end.end_effector_pose["xyz"] = [0.002, -0.001, 0.507]
    states = iter((start, end))
    controller = GazeboController(
        state_provider=lambda: next(states),
        move_action=lambda _goal, _timeout: {
            "ok": True,
            "reached_goal": True,
            "motion_outcome": "completed",
        },
    )

    receipt = controller.execute(
        {
            "action_type": "move_to",
            "target_pose": {
                "xyz": [0, 0, 0.5],
                "quat_xyzw": [0, 0, 0, 1],
                "purpose": "placement",
                "placement_stage": "release",
            },
            "tolerance": 0.002,
            "ori_tolerance": 0.05,
        }
    ).to_dict()

    assert receipt["ok"] is False
    assert receipt["reached_target"] is False
    assert receipt["error_code"] == "MOTION_TARGET_NOT_REACHED"
    assert receipt["position_error_m"] > receipt["position_verification_tolerance_m"]
    assert receipt["horizontal_error_m"] < receipt["position_verification_tolerance_m"]
    assert receipt["vertical_error_m"] == pytest.approx(0.007)
    assert receipt["position_error_m"] > receipt["position_verification_effective_tolerance_m"]
    assert receipt["position_verification_policy"] == (
        "exact_terminal_euclidean_with_bounded_numeric_margin"
    )


@pytest.mark.parametrize("xyz", ([0.007, 0.0, 0.5], [0.0, 0.0, 0.493]))
def test_controller_keeps_all_release_residuals_strict(xyz) -> None:
    start, end = _state(), _state()
    end.end_effector_pose["xyz"] = list(xyz)
    states = iter((start, end))
    controller = GazeboController(
        state_provider=lambda: next(states),
        move_action=lambda _goal, _timeout: {
            "ok": True,
            "reached_goal": True,
            "motion_outcome": "completed",
        },
    )

    receipt = controller.execute(
        {
            "action_type": "move_to",
            "target_pose": {
                "xyz": [0, 0, 0.5],
                "quat_xyzw": [0, 0, 0, 1],
                "purpose": "placement",
                "placement_stage": "release",
            },
            "tolerance": 0.002,
            "ori_tolerance": 0.05,
        }
    ).to_dict()

    assert receipt["ok"] is False
    assert receipt["error_code"] == "MOTION_TARGET_NOT_REACHED"


@pytest.mark.parametrize(
    "result",
    [
        {"ok": True, "reached_goal": False, "stalled": False},
        {"ok": True, "reached_goal": False, "stalled": True},
        {"ok": False, "reached_goal": False, "stalled": False, "error_code": "GRIPPER_TIMEOUT"},
    ],
)
def test_robot_gripper_never_credits_unreached_stalled_or_timed_out_results(result) -> None:
    controller = GazeboController(
        state_provider=_state, gripper_action=lambda _position, _timeout: result
    )

    receipt = controller.execute({"action_type": "gripper_open"}).to_dict()

    assert receipt["ok"] is False
    assert receipt["reached_goal"] is False
    assert receipt["stalled"] is bool(result.get("stalled", False))
    assert receipt["error_code"] in {"GRIPPER_FAILED", "GRIPPER_TIMEOUT"}


def test_native_grasp_profile_admits_known_stalled_close_to_contact_gate() -> None:
    controller = GazeboController(
        config=NativePickPlaceConfig(),
        state_provider=_state,
        gripper_action=lambda _position, _timeout: {
            "ok": False,
            "reached_goal": False,
            "stalled": True,
            "terminal_status": "not_succeeded",
            "terminal_status_code": 6,
        },
    )

    receipt = controller.execute({"action_type": "gripper_close"}).to_dict()

    assert receipt["ok"] is True
    assert receipt["reached_goal"] is False
    assert receipt["stalled"] is True
    assert receipt.get("error_code") is None


def test_native_grasp_profile_never_admits_stalled_open() -> None:
    controller = GazeboController(
        config=NativePickPlaceConfig(),
        state_provider=_state,
        gripper_action=lambda _position, _timeout: {
            "ok": False,
            "reached_goal": False,
            "stalled": True,
            "terminal_status": "not_succeeded",
            "terminal_status_code": 6,
        },
    )

    receipt = controller.execute({"action_type": "gripper_open"}).to_dict()

    assert receipt["ok"] is False
    assert receipt["reached_goal"] is False
    assert receipt["stalled"] is True
    assert receipt["error_code"] == "GRIPPER_FAILED"


def test_robot_gripper_receipt_keeps_terminal_and_wall_clock_diagnostics() -> None:
    controller = GazeboController(
        state_provider=_state,
        gripper_action=lambda _position, _timeout: {
            "ok": True,
            "reached_goal": True,
            "stalled": False,
            "terminal_status": "succeeded",
            "terminal_status_code": 4,
            "wall_elapsed_ms": 123.456,
            "sim_clock_elapsed_ms": 100.0,
            "observed_sim_clock_ratio": 0.81,
        },
    )

    receipt = controller.execute({"action_type": "gripper_open"}).to_dict()

    assert receipt["ok"] is True
    assert receipt["terminal_status"] == "succeeded"
    assert receipt["terminal_status_code"] == 4
    assert receipt["wall_elapsed_ms"] == pytest.approx(123.456)
    assert receipt["sim_clock_elapsed_ms"] == pytest.approx(100.0)
    assert receipt["observed_sim_clock_ratio"] == pytest.approx(0.81)


def test_attached_transport_hold_opens_one_common_driver_after_attach() -> None:
    def linked_state(active: float):
        linkage = six_joint_positions(active)
        return robot_state_from_sources(
            {
                "name": JOINT_NAMES,
                "position": [0.0] * len(ARM_JOINTS)
                + [linkage[name] for name in JOINT_NAMES[len(ARM_JOINTS) :]],
                "velocity": [0.0] * len(JOINT_NAMES),
            },
            {
                "base_link->gripper_mount_link": {
                    "xyz": [0.0, 0.0, 0.5],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            },
        )

    states = iter((linked_state(0.60), linked_state(0.57)))
    commands: list[tuple[float, float]] = []

    def gripper_action(position, timeout):
        commands.append((position, timeout))
        return {
            "ok": True,
            "reached_goal": True,
            "stalled": False,
            "terminal_status": "succeeded",
        }

    controller = GazeboController(
        state_provider=lambda: next(states),
        gripper_action=gripper_action,
    )

    evidence = controller.establish_attached_transport_hold(timeout_s=12.0)

    assert commands == [(pytest.approx(0.56), 12.0)]
    assert evidence["schema_version"] == "openeta.attached_transport_hold.v1"
    assert evidence["actuator_model"] == "single_common_driver"
    assert evidence["object_environment_collision_enabled"] is True
    assert evidence["minimum_proven_relief_rad"] == pytest.approx(0.02)
    assert evidence["measured_common_after_rad"] == pytest.approx(0.57)


def test_attached_transport_hold_rejects_a_stalled_internal_open() -> None:
    linkage = six_joint_positions(0.60)
    state = robot_state_from_sources(
        {
            "name": JOINT_NAMES,
            "position": [0.0] * len(ARM_JOINTS)
            + [linkage[name] for name in JOINT_NAMES[len(ARM_JOINTS) :]],
            "velocity": [0.0] * len(JOINT_NAMES),
        },
        {
            "base_link->gripper_mount_link": {
                "xyz": [0.0, 0.0, 0.5],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
    )
    controller = GazeboController(
        state_provider=lambda: state,
        gripper_action=lambda _position, _timeout: {
            "ok": True,
            "reached_goal": False,
            "stalled": True,
        },
    )

    with pytest.raises(RuntimeError, match="ATTACHED_TRANSPORT_HOLD_FAILED"):
        controller.establish_attached_transport_hold()


def test_controller_runs_one_recovery_then_submits_the_original_target_once() -> None:
    move_calls = []
    recovery_calls = []
    target = {
        "xyz": [0.1, -0.2, 0.5],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    def recover(state, timeout_s):
        recovery_calls.append((state, timeout_s))
        return {
            "ok": True,
            "action_started_ros_time_s": 10.0,
            "action_completed_ros_time_s": 11.0,
            "start_state_recovery": {
                "schema_version": "m2_start_state_recovery_v1",
                "status": "RECOVERED",
                "attempted": True,
            },
        }

    controller = GazeboController(
        state_provider=_bounded_state,
        start_state_recovery=recover,
        move_action=lambda goal, timeout: (
            move_calls.append((goal, timeout))
            or {
                "ok": True,
                "action_started_ros_time_s": 12.0,
                "action_completed_ros_time_s": 20.0,
            }
        ),
    )
    receipt = controller.execute(
        {"action_type": "move_to", "target_pose": target, "timeout_s": 30.0}
    ).to_dict()

    assert len(recovery_calls) == 1
    assert recovery_calls[0][1] == 5.0
    assert len(move_calls) == 1
    assert move_calls[0][0]["requested_tool_pose"]["xyz"] == target["xyz"]
    assert move_calls[0][0]["requested_tool_pose"]["quat_xyzw"] == target["quat_xyzw"]
    assert receipt["target"]["xyz"] == target["xyz"]
    assert receipt["start_state_recovery"]["status"] == "RECOVERED"
    assert receipt["action_started_ros_time_s"] == 10.0
    assert receipt["action_completed_ros_time_s"] == 20.0


@pytest.mark.parametrize(
    ("recovery_result", "expected_code", "expected_status", "unknown"),
    [
        (
            {
                "ok": False,
                "error_code": "START_STATE_INVALID",
                "motion_outcome": "failed",
                "start_state_recovery": {
                    "schema_version": "m2_start_state_recovery_v1",
                    "status": "REJECTED",
                    "attempted": False,
                },
            },
            "START_STATE_INVALID",
            "REJECTED",
            False,
        ),
        (
            {
                "ok": False,
                "error_code": "START_STATE_RECOVERY_FAILED",
                "motion_outcome": "failed",
                "start_state_recovery": {
                    "schema_version": "m2_start_state_recovery_v1",
                    "status": "FAILED",
                    "attempted": True,
                },
            },
            "START_STATE_RECOVERY_FAILED",
            "FAILED",
            False,
        ),
        (
            {
                "ok": False,
                "error_code": "MOTION_OUTCOME_UNKNOWN",
                "motion_outcome": "unknown",
                "reconciliation_required": True,
                "start_state_recovery": {
                    "schema_version": "m2_start_state_recovery_v1",
                    "status": "UNKNOWN",
                    "attempted": True,
                },
            },
            "MOTION_OUTCOME_UNKNOWN",
            "UNKNOWN",
            True,
        ),
    ],
)
def test_controller_recovery_failure_never_submits_user_target(
    recovery_result, expected_code, expected_status, unknown
) -> None:
    move_calls = []
    controller = GazeboController(
        state_provider=_bounded_state,
        start_state_recovery=lambda state, timeout: recovery_result,
        move_action=lambda goal, timeout: move_calls.append(goal) or {"ok": True},
    )

    receipt = controller.execute(
        {
            "action_type": "move_to",
            "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
        }
    ).to_dict()

    assert move_calls == []
    assert receipt["error_code"] == expected_code
    assert receipt["start_state_recovery"]["status"] == expected_status
    assert receipt.get("reconciliation_required", False) is unknown


def test_moveit_start_state_invalid_after_preflight_does_not_loop_recovery() -> None:
    calls = {"recovery": 0, "move": 0}

    def recovery(state, timeout):
        calls["recovery"] += 1
        return {
            "ok": True,
            "start_state_recovery": {
                "schema_version": "m2_start_state_recovery_v1",
                "status": "NOT_REQUIRED",
                "attempted": False,
            },
        }

    def move(goal, timeout):
        calls["move"] += 1
        return {
            "ok": False,
            "error_code": "MOTION_PLAN_FAILED",
            "moveit_error_code": -26,
        }

    receipt = (
        GazeboController(
            state_provider=_bounded_state,
            start_state_recovery=recovery,
            move_action=move,
        )
        .execute(
            {
                "action_type": "move_to",
                "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
            }
        )
        .to_dict()
    )

    assert calls == {"recovery": 1, "move": 1}
    assert receipt["error_code"] == "MOTION_PLAN_FAILED"
    assert receipt["moveit_error_code"] == -26
    assert receipt["start_state_recovery"]["status"] == "NOT_REQUIRED"


def test_robotiq_binary_mapping_and_calibrated_aperture() -> None:
    config = GazeboControlConfig()
    feasible_open = minimum_feasible_active_position(
        boundary_inset_rad=DEFAULT_CONTROLLER_BOUNDARY_INSET_RAD
    )
    assert config.model_id == MODEL_ID
    assert config.gripper_position(1) == pytest.approx(feasible_open)
    assert config.gripper_position(0) == pytest.approx(0.7929)
    open_state = gripper_state(0.0, config=config)
    almost_open_state = gripper_state(0.0001133, config=config)
    closed_state = gripper_state(0.7929, config=config)
    assert open_state["aperture_m"] == pytest.approx(0.085)
    assert open_state["open"] is True
    assert almost_open_state["open"] is True
    assert closed_state["aperture_m"] == pytest.approx(0.0)
    assert closed_state["open"] is False
    for invalid in (False, 0.0, 1.0, -1, 2):
        with pytest.raises(ValueError):
            config.gripper_position(invalid)


@pytest.mark.parametrize(
    ("action_result", "expected_code", "expected_outcome"),
    [
        ({"ok": False, "error_code": "MOTION_PLAN_FAILED"}, "MOTION_PLAN_FAILED", "failed"),
        (
            {"ok": False, "error_code": "MOTION_OUTCOME_UNKNOWN", "motion_outcome": "unknown"},
            "MOTION_OUTCOME_UNKNOWN",
            "unknown",
        ),
        ({"unexpected": True}, "MOTION_OUTCOME_UNKNOWN", "unknown"),
    ],
)
def test_controller_rejection_and_unknown_results_fail_closed(
    action_result, expected_code, expected_outcome
) -> None:
    controller = GazeboController(
        state_provider=_state, move_action=lambda goal, timeout: action_result
    )
    result = controller.execute(
        {
            "action_type": "move_to",
            "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
        }
    ).to_dict()
    assert result["ok"] is False
    assert result["error_code"] == expected_code
    assert result["motion_outcome"] == expected_outcome
    if expected_outcome == "unknown":
        assert result["reconciliation_required"] is True


def test_controller_timeout_and_missing_reconciliation_state_fail_closed() -> None:
    def timeout(_goal, _timeout):
        raise TimeoutError

    timed_out = GazeboController(state_provider=_state, move_action=timeout).execute(
        {
            "action_type": "move_to",
            "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
        }
    )
    assert timed_out.error_code == "MOTION_OUTCOME_UNKNOWN"
    assert timed_out.payload["reconciliation_required"] is True

    calls = 0

    def state_then_fail():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("JOINT_STATE_TIMEOUT")
        return _state()

    unreconciled = GazeboController(
        state_provider=state_then_fail, move_action=lambda goal, timeout: {"ok": True}
    ).execute(
        {
            "action_type": "move_to",
            "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
        }
    )
    assert unreconciled.error_code == "MOTION_OUTCOME_UNKNOWN"
    assert unreconciled.payload["reconciliation_required"] is True


def test_controller_accepts_stationary_action_ordered_terminal_after_success() -> None:
    calls = 0

    def state_then_fail():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("JOINT_STATE_TIMEOUT")
        return _state()

    barriers: list[float] = []

    def action_terminal(barrier: float):
        barriers.append(barrier)
        return _state()

    result = GazeboController(
        state_provider=state_then_fail,
        barrier_ordered_terminal_state_provider=action_terminal,
        move_action=lambda _goal, _timeout: {
            "ok": True,
            "reached_goal": True,
            "execution_started": True,
            "planned_point_count": 2,
            "motion_outcome": "completed",
            "action_started_ros_time_s": 12.5,
            "action_completed_ros_time_s": 14.0,
            "wall_elapsed_ms": 1523.4,
            "sim_clock_elapsed_ms": 1500.0,
            "observed_sim_clock_ratio": 0.984646,
        },
    ).execute(
        {
            "action_type": "move_to",
            "target_pose": {
                "xyz": [0.0, 0.0, 0.5],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    )

    assert result.ok is True
    assert barriers == [14.0]
    assert result.payload["terminal_state_source"] == (
        "barrier_ordered_action_terminal_sample"
    )
    assert result.payload["terminal_state_action_barrier_source"] == (
        "action_completed_ros_time_s"
    )
    assert result.payload["terminal_state_stationary_verified"] is True
    assert result.payload["wall_elapsed_ms"] == pytest.approx(1523.4)
    assert result.payload["sim_clock_elapsed_ms"] == pytest.approx(1500.0)
    assert result.payload["observed_sim_clock_ratio"] == pytest.approx(0.984646)


def test_controller_waits_for_stationary_state_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moving = _state()
    moving.joint_velocities[0] = 0.01
    terminal_states = iter((moving, _state()))
    monkeypatch.setattr("extensions.gazebo.robot_control.time.sleep", lambda _s: None)
    result = GazeboController(
        state_provider=_state,
        barrier_ordered_terminal_state_provider=lambda _barrier: next(terminal_states),
        move_action=lambda _goal, _timeout: {
            "ok": True,
            "reached_goal": True,
            "execution_started": True,
            "planned_point_count": 2,
            "motion_outcome": "completed",
            "action_started_ros_time_s": 12.5,
        },
    ).execute(
        {
            "action_type": "move_to",
            "target_pose": {
                "xyz": [0.0, 0.0, 0.5],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    )

    assert result.ok is True
    assert result.error_code is None
    assert result.payload["settling_recheck"]["status"] == "target_verified"
    assert result.payload["settling_recheck"]["initial_max_arm_velocity_rad_s"] == (
        pytest.approx(0.01)
    )
    assert result.payload["settling_recheck"]["final_max_arm_velocity_rad_s"] == (
        pytest.approx(0.0)
    )
    assert result.payload["terminal_state_stationary_verified"] is True


def test_controller_restarts_from_settled_state_when_success_receipt_misses_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, moving, settled = _bounded_state(), _bounded_state(), _bounded_state()
    moving.end_effector_pose["xyz"] = [0.02, 0.0, 0.5]
    moving.joint_velocities[0] = 0.01
    settled.end_effector_pose["xyz"] = [0.015, 0.0, 0.5]
    terminal_states = iter((moving, settled))
    monkeypatch.setattr("extensions.gazebo.robot_control.time.sleep", lambda _s: None)

    result = GazeboController(
        state_provider=lambda: start,
        barrier_ordered_terminal_state_provider=lambda _barrier: next(terminal_states),
        move_action=lambda _goal, _timeout: {
            "ok": True,
            "reached_goal": True,
            "execution_started": True,
            "planned_point_count": 2,
            "motion_outcome": "completed",
            "action_completed_ros_time_s": 14.0,
        },
    ).execute(
        {
            "action_type": "move_to",
            "target_pose": {
                "xyz": [0.0, 0.0, 0.5],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    )

    assert result.ok is False
    assert result.error_code == "MOTION_TARGET_NOT_REACHED"
    assert result.payload["settling_recheck"]["status"] == ("stationary_target_not_reached")
    assert result.payload["restart_state_stationary_verified"] is True
    assert result.payload["current_state_restart"]["status"] == "PASS"


def test_controller_keeps_unsettled_action_ordered_terminal_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, moving = _state(), _state()
    moving.joint_velocities[0] = 0.01
    calls = 0

    def terminal_then_unavailable(_barrier):
        nonlocal calls
        calls += 1
        if calls == 1:
            return moving
        raise RuntimeError("JOINT_STATE_TIMEOUT")

    monkeypatch.setattr("extensions.gazebo.robot_control.time.sleep", lambda _s: None)
    result = GazeboController(
        state_provider=lambda: start,
        barrier_ordered_terminal_state_provider=terminal_then_unavailable,
        move_action=lambda _goal, _timeout: {
            "ok": True,
            "reached_goal": True,
            "execution_started": True,
            "planned_point_count": 2,
            "motion_outcome": "completed",
            "action_started_ros_time_s": 12.5,
        },
    ).execute(
        {
            "action_type": "move_to",
            "target_pose": {
                "xyz": [0.0, 0.0, 0.5],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    )

    assert result.ok is False
    assert result.error_code == "MOTION_OUTCOME_UNKNOWN"
    assert result.payload["reconciliation_required"] is True
    assert result.payload["terminal_state_stationary_verified"] is False
    assert "current_state_restart" not in result.payload


def test_controller_close_is_idempotent() -> None:
    calls: list[str] = []
    controller = GazeboController(
        state_provider=_state,
        cancel_pending=lambda: calls.append("cancel"),
        close_source=lambda: calls.append("close"),
    )
    controller.close()
    controller.close()
    assert calls == ["cancel", "close"]
