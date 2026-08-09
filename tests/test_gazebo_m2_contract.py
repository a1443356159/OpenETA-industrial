from __future__ import annotations

import pytest

from extensions.gazebo.m2 import (
    JOINT_NAMES,
    M2Config,
    M2Controller,
    ROBOTIQ2F85_MODEL_ID,
    Robotiq2F85Config,
    gripper_state,
    make_move_group_goal,
    robot_state_from_sources,
)
from sim.env_registry import get_env_spec


def _state(active=0.035):
    return robot_state_from_sources(
        {"name": JOINT_NAMES, "position": [0] * 7 + [active, active], "velocity": [0] * 9},
        {"base_link->gripper_mount_link": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]}},
    )


def test_gazebo_environment_uses_canonical_display_name() -> None:
    spec = get_env_spec("openeta/gazebo_rm75_robotiq2f85-v0")
    assert spec is not None
    assert spec.display_name == "Gazebo 仿真环境"


def test_m2_names_opening_and_binary_command():
    cfg = M2Config()
    cfg.validate_assets()
    assert cfg.maximum_aperture_m == 0.070
    assert gripper_state(0.035)["aperture_m"] == 0.070
    assert cfg.gripper_position(0) == 0 and cfg.gripper_position(1) == 0.035
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
        {"xyz": [1, 0, 0], "quat_xyzw": [0, 0, 0, 1]}, config=M2Config(mount_xyz=(0.1, 0, 0))
    )
    assert goal["group_name"] == "rm_group" and goal["link_name"] == "link_7"
    assert goal["target_pose"]["xyz"] == pytest.approx([0.9, 0, 0])


def test_controller_routes_actions_and_unknown_outcome():
    sent = []
    ctl = M2Controller(
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


def test_robotiq_binary_mapping_and_calibrated_aperture() -> None:
    config = Robotiq2F85Config()
    assert config.model_id == ROBOTIQ2F85_MODEL_ID
    assert config.gripper_position(1) == 0.0
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
    controller = M2Controller(state_provider=_state, move_action=lambda goal, timeout: action_result)
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

    timed_out = M2Controller(state_provider=_state, move_action=timeout).execute(
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

    unreconciled = M2Controller(
        state_provider=state_then_fail, move_action=lambda goal, timeout: {"ok": True}
    ).execute(
        {
            "action_type": "move_to",
            "target_pose": {"xyz": [0, 0, 0.5], "quat_xyzw": [0, 0, 0, 1]},
        }
    )
    assert unreconciled.error_code == "MOTION_OUTCOME_UNKNOWN"
    assert unreconciled.payload["reconciliation_required"] is True


def test_controller_close_is_idempotent() -> None:
    calls: list[str] = []
    controller = M2Controller(
        state_provider=_state,
        cancel_pending=lambda: calls.append("cancel"),
        close_source=lambda: calls.append("close"),
    )
    controller.close()
    controller.close()
    assert calls == ["cancel", "close"]
