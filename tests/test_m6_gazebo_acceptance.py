from __future__ import annotations

import json

from scripts import m6_gazebo_acceptance as m6


def test_m6_prepare_registers_real_services_and_constraint_prompt(
    tmp_path, monkeypatch
) -> None:
    allocation = m6.base.Allocation(81, "openeta-tui-m6-test", 18765, "run-id")
    monkeypatch.setattr(m6.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        m6.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )
    services = dict(m6.DEFAULT_SERVICES)

    paths = m6.prepare_case(tmp_path, tmp_path / "run", allocation, services)

    config = json.loads(paths.mcp_config.read_text(encoding="utf-8"))["mcpServers"]
    assert set(config) == {
        "openeta-sim",
        "openeta-sam3",
        "openeta-anyplace",
        "openeta-graspgenx",
    }
    prompt = paths.instructions.read_text(encoding="utf-8")
    assert "GraspGenX" in prompt and "AnyPlace" in prompt
    assert "execution_started=false" in prompt
    assert "稳定 >=0.5 s" in prompt
    assert "禁止 Oracle" in prompt


def test_m6_order_helper_rejects_anyplace_before_lift() -> None:
    valid = ["observe", "graspgenx", "gripper_control", "move_to", "anyplace", "compile_grasp_seed"]
    invalid = ["observe", "graspgenx", "anyplace", "gripper_control", "move_to", "compile_grasp_seed"]

    required = ("observe", "graspgenx", "gripper_control", "move_to", "anyplace", "compile_grasp_seed")
    assert m6._ordered(valid, required)
    assert not m6._ordered(invalid, required)


def test_m6_health_url_preserves_service_root() -> None:
    assert m6._health_url("http://127.0.0.1:8778/sse") == "http://127.0.0.1:8778/"
