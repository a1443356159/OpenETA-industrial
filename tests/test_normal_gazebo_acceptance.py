from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from agent.backends.provider_config import ProviderEndpointConfig
from scripts import normal_gazebo_acceptance as acceptance


NORMAL_RUNNER = Path(__file__).resolve().parents[1] / "scripts/run_normal_gazebo_acceptance.sh"
PICK_PLACE_RUNNER = Path(__file__).resolve().parents[1] / "scripts/run_pick_place_acceptance.sh"


def test_normal_runner_enables_case_owned_operator_gui_by_default() -> None:
    source = NORMAL_RUNNER.read_text(encoding="utf-8")

    assert (
        'export OPENETA_GAZEBO_OPERATOR_GUI="${OPENETA_GAZEBO_OPERATOR_GUI:-1}"'
        in source
    )


def test_normal_prepare_registers_real_services_and_human_task_prompt(tmp_path, monkeypatch) -> None:
    allocation = acceptance.base.Allocation(81, "openeta-normal-test", 18765, "run-id")
    monkeypatch.setattr(acceptance.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        acceptance.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )
    services = dict(acceptance.DEFAULT_SERVICES)

    paths = acceptance.prepare_case(tmp_path, tmp_path / "run", allocation, services)

    assert paths.root == (
        tmp_path / "run" / "pick-place" / acceptance.DEFAULT_OPERATOR_MODE
    )
    config = json.loads(paths.mcp_config.read_text(encoding="utf-8"))["mcpServers"]
    assert set(config) == {
        "openeta-sim",
        "openeta-sam3",
        "openeta-graspgenx",
        "openeta-anyplace",
    }
    assert config["openeta-sim"]["url"] == "http://127.0.0.1:18765/mcp"
    prompt = paths.instructions.read_text(encoding="utf-8")
    assert "黄色活动扳手" in prompt
    assert "绿色零件箱" in prompt
    assert "请先看清工作台" in prompt
    assert "如果视角不清楚，可以换个角度确认" in prompt
    assert "在目标箱里放稳就好" in prompt
    assert "一种办法没成功" not in prompt
    assert "yellow adjustable wrench" not in prompt
    assert "green area inside physical parts bin" not in prompt
    assert "系统保留的候选" not in prompt
    # Backend choice, pose conventions, recovery internals, and acceptance
    # thresholds are host contracts rather than human task instructions.
    for internal_term in (
        "GraspGenX",
        "AnyPlace",
        "exact EEF",
        "T_eef_object",
        "frozen_frontier",
        "Beam-2",
        "4 → 8",
        "python_exec",
        "Oracle",
        "target_object",
        "automation=",
        "planner_mode=",
        "qualification_profile=",
        "environment_id=",
    ):
        assert internal_term not in prompt
    assert len(prompt) < 1_400

    metadata = acceptance._automation_metadata_for_backend("anygrasp")
    assert "planner_mode=agentic_closed_loop" in metadata
    assert f"environment_id={acceptance.ENV_ID}" in metadata
    assert "initial_observe=" not in metadata
    assert "environment_task=" not in metadata


def test_multi_normal_human_tui_uses_human_mode_without_changing_task_prompt(
    tmp_path, monkeypatch
) -> None:
    allocation = acceptance.base.Allocation(83, "human-tui", 18767, "run-id")
    monkeypatch.setattr(acceptance.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        acceptance.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )

    paths = acceptance.prepare_case(
        tmp_path,
        tmp_path / "human",
        allocation,
        dict(acceptance.DEFAULT_SERVICES),
        scenario="multi_normal",
        operator_mode=acceptance.base.HUMAN_TUI,
    )

    assert paths.root == tmp_path / "human" / "pick-place" / "human_tui"
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert receipt["operator_mode"] == "human_tui"
    prompt = paths.instructions.read_text(encoding="utf-8")
    assert "黄色活动扳手" in prompt
    assert "operator=human_tui" not in prompt
    metadata = acceptance._automation_metadata_for_backend(
        "graspgenx",
        scenario="multi_normal",
        operator_mode=acceptance.base.HUMAN_TUI,
    )
    assert metadata.startswith("[operator=human_tui;")
    assert "work_order_source=vlm_conversation" in metadata
    assert "grasp_target=" not in metadata


def test_human_tui_approval_requires_human_gated_provenance() -> None:
    approved = {
        "name": "move_to",
        "result": {
            "details": {
                "supervision": {
                    "allowed": True,
                    "source": "human",
                    "details": {"profile": "human_gated"},
                }
            }
        },
    }
    scripted = {
        "name": "move_to",
        "result": {
            "details": {
                "supervision": {
                    "allowed": True,
                    "source": "scripted_tui",
                    "details": {"profile": "scripted_tui"},
                }
            }
        },
    }

    assert acceptance.base._human_approved(approved) is True
    assert acceptance.base._human_approved(scripted) is False


def test_pick_place_complex_scenes_have_independent_seed_prompt_and_receipt(
    tmp_path, monkeypatch
) -> None:
    allocation = acceptance.base.Allocation(81, "openeta-complex-scenes", 18765, "run-id")
    monkeypatch.setattr(acceptance.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        acceptance.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )

    expectations = {
        "narrow-pick": (17, ["pick_guard_left", "pick_guard_right"]),
        "barrier-transfer": (29, ["transfer_barrier"]),
        "fastener-bin-sort": (
            41,
            [
                "yellow_open_end_wrench",
                "blue_handle_pliers",
                "blue_bin_wall_left",
                "blue_bin_wall_right",
                "blue_bin_wall_near",
                "blue_bin_wall_far",
                "orange_bin_wall_left",
                "orange_bin_wall_right",
                "orange_bin_wall_near",
                "orange_bin_wall_far",
            ],
        ),
        "tool-bin-sort": (
            53,
            [
                "silver_hex_bolt",
                "blue_handle_pliers_tool_scene",
                "red_screwdriver",
                "purple_bin_wall_left",
                "purple_bin_wall_right",
                "purple_bin_wall_near",
                "purple_bin_wall_far",
                "green_bin_wall_left",
                "green_bin_wall_right",
                "green_bin_wall_near",
                "green_bin_wall_far",
            ],
        ),
    }
    for scenario, (seed, obstacle_ids) in expectations.items():
        paths = acceptance.prepare_case(
            tmp_path,
            tmp_path / scenario,
            allocation,
            dict(acceptance.DEFAULT_SERVICES),
            scenario=scenario,
        )
        prompt = paths.instructions.read_text(encoding="utf-8")
        receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
        metadata = acceptance._automation_metadata_for_backend(
            "anygrasp", scenario=scenario
        )

        assert f"environment_seed={seed}" in metadata
        assert f"acceptance_scene={scenario}" in metadata
        assert "验收场景" not in prompt
        assert acceptance.SCENARIO_INSTRUCTIONS[scenario] in prompt
        assert receipt["acceptance_scenario"] == scenario
        assert receipt["acceptance_scene"]["scene_id"] == scenario
        assert receipt["acceptance_scene"]["seed"] == seed
        assert receipt["acceptance_scene"]["static_obstacle_ids"] == obstacle_ids
        expected_destination = {
            "barrier-transfer": [0.48, 0.10],
            "fastener-bin-sort": [0.43, -0.13],
            "tool-bin-sort": [0.43, 0.13],
        }.get(scenario, [0.48, -0.10])
        assert receipt["acceptance_scene"]["destination_center_xy"] == expected_destination
        assert receipt["acceptance_scene"]["schema_version"] == (
            "openeta.gazebo_acceptance_scene_receipt.v2"
        )
        assert len(receipt["acceptance_scene"]["contract_sha256"]) == 64


def test_multi_normal_prepares_one_human_request_with_private_verification_contract(
    tmp_path, monkeypatch
) -> None:
    allocation = acceptance.base.Allocation(81, "openeta-multi-normal", 18765, "run-id")
    monkeypatch.setattr(acceptance.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        acceptance.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )

    paths = acceptance.prepare_case(
        tmp_path,
        tmp_path / "multi-normal",
        allocation,
        dict(acceptance.DEFAULT_SERVICES),
        scenario="multi_normal",
        grasp_backend="graspgenx",
    )

    prompt = paths.instructions.read_text(encoding="utf-8")
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assignments = receipt["acceptance_scene"]["expected_work_order"]

    assert "黄色活动扳手" in prompt and "绿色零件箱" in prompt
    assert "红色六角螺栓" in prompt and "蓝色零件箱" in prompt
    assert "放好第一件后不要关闭环境" in prompt
    assert [item["id"] for item in assignments] == [
        "yellow_wrench_to_green_parts_bin",
        "red_bolt_to_blue_parts_bin",
    ]
    assert [item["target_object_id"] for item in assignments] == [
        "target_object",
        "red_m24_hex_bolt",
    ]
    assert [item["placement_region_id"] for item in assignments] == [
        "green_parts_bin",
        "blue_parts_bin",
    ]
    assert assignments[1]["placement_region_perception_prompt"] == (
        "blue square area inside bin"
    )
    assert acceptance._scenario_environment("multi_normal") == {
        "OPENETA_ACCEPTANCE_SCENE": "multi_normal"
    }
    metadata = acceptance._automation_metadata_for_backend(
        "graspgenx", scenario="multi_normal"
    )
    assert "work_order_source=vlm_conversation" in metadata
    assert "environment_task=" not in metadata
    assert "grasp_target=" not in metadata
    assert "placement_region=" not in metadata


def test_seeded_random_scene_keeps_the_same_vlm_authored_multi_sort_contract(
    tmp_path, monkeypatch
) -> None:
    allocation = acceptance.base.Allocation(84, "random-multi", 18768, "run-id")
    monkeypatch.setattr(acceptance.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        acceptance.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )

    paths = acceptance.prepare_case(
        tmp_path,
        tmp_path / "random-multi",
        allocation,
        dict(acceptance.DEFAULT_SERVICES),
        scenario="multi_normal_random_12345",
        grasp_backend="graspgenx",
    )
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    prompt = paths.instructions.read_text(encoding="utf-8")
    metadata = acceptance._automation_metadata_for_backend(
        "graspgenx", scenario="multi_normal_random_12345"
    )

    assert receipt["acceptance_scene"]["seed"] == 12345
    assert receipt["acceptance_scene"]["scene_id"] == (
        "multi_normal_random_12345"
    )
    assert receipt["acceptance_scene"]["acceptance_request_id"] == (
        "multi_normal_random_12345:wrench-green-bolt-blue"
    )
    assert receipt["task_variant"] == "wrench-green-bolt-blue"
    assert len(receipt["acceptance_scene"]["expected_work_order"]) == 2
    assert "工作台物件的位置和朝向已经变化" in prompt
    assert "environment_seed=12345" in metadata
    assert "work_order_source=vlm_conversation" in metadata
    assert acceptance._scenario_environment("multi_normal_random_12345") == {
        "OPENETA_ACCEPTANCE_SCENE": "multi_normal_random_12345"
    }


def test_multi_normal_counts_legacy_anyplace_model_calls_per_assignment() -> None:
    def call(parameters, outputs):
        return {
            "parameters": parameters,
            "result": {"details": {"outputs": outputs}},
        }

    first_model = call({"scene_revision": 1}, {})
    first_requalification = call(
        {"reuse_frozen_goal_pool": True, "scene_revision": 2},
        {"anyplace_model_inference_invoked": False},
    )
    second_model = call(
        {"scene_revision": 5},
        {"anyplace_model_inference_invoked": True},
    )
    second_requalification = call(
        {"reuse_frozen_goal_pool": True, "scene_revision": 6},
        {"anyplace_model_inference_invoked": False},
    )

    assert acceptance._anyplace_model_inference_calls(
        [
            first_model,
            first_requalification,
            second_model,
            second_requalification,
        ]
    ) == [first_model, second_model]


@pytest.mark.parametrize(
    ("task_variant", "ordered_assignment_ids", "first_destination"),
    [
        (
            "wrench-blue-bolt-green",
            [
                "yellow_wrench_to_blue_parts_bin",
                "red_bolt_to_green_parts_bin",
            ],
            "blue_parts_bin",
        ),
        (
            "bolt-blue-wrench-green",
            [
                "red_bolt_to_blue_parts_bin",
                "yellow_wrench_to_green_parts_bin",
            ],
            "blue_parts_bin",
        ),
        (
            "bolt-green-wrench-blue",
            [
                "red_bolt_to_green_parts_bin",
                "yellow_wrench_to_blue_parts_bin",
            ],
            "green_parts_bin",
        ),
    ],
)
def test_multi_normal_task_variants_change_only_user_words_and_verification_contract(
    tmp_path,
    monkeypatch,
    task_variant: str,
    ordered_assignment_ids: list[str],
    first_destination: str,
) -> None:
    allocation = acceptance.base.Allocation(82, task_variant, 18766, "run-id")
    monkeypatch.setattr(acceptance.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        acceptance.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )

    paths = acceptance.prepare_case(
        tmp_path,
        tmp_path / task_variant,
        allocation,
        dict(acceptance.DEFAULT_SERVICES),
        scenario="multi_normal",
        task_variant=task_variant,
        grasp_backend="graspgenx",
    )
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assignments = receipt["acceptance_scene"]["expected_work_order"]

    assert receipt["acceptance_scenario"] == "multi_normal"
    assert receipt["task_variant"] == task_variant
    assert receipt["acceptance_scene"]["acceptance_request_id"] == (
        f"multi_normal:{task_variant}"
    )
    assert receipt["acceptance_scene"]["scene_id"] == "multi_normal"
    assert [item["id"] for item in assignments] == ordered_assignment_ids
    assert assignments[0]["placement_region_id"] == first_destination
    assert assignments[0]["placement_region_perception_prompt"] == (
        "blue square area inside bin"
        if first_destination == "blue_parts_bin"
        else "green parts bin"
    )
    assert acceptance._scenario_environment("multi_normal") == {
        "OPENETA_ACCEPTANCE_SCENE": "multi_normal"
    }
    assert "放好第一件后不要关闭环境" in paths.instructions.read_text(
        encoding="utf-8"
    )


def test_complex_scene_environment_is_not_a_qualification_fault() -> None:
    assert acceptance._scenario_environment("multi_normal_random_12345") == {
        "OPENETA_ACCEPTANCE_SCENE": "multi_normal_random_12345"
    }
    assert acceptance._scenario_environment("narrow-pick") == {
        "OPENETA_ACCEPTANCE_SCENE": "narrow-pick"
    }
    assert acceptance._scenario_environment("barrier-transfer") == {
        "OPENETA_ACCEPTANCE_SCENE": "barrier-transfer"
    }
    assert acceptance._scenario_environment("fastener-bin-sort") == {
        "OPENETA_ACCEPTANCE_SCENE": "fastener-bin-sort"
    }
    assert acceptance._scenario_environment("tool-bin-sort") == {
        "OPENETA_ACCEPTANCE_SCENE": "tool-bin-sort"
    }


def test_industrial_scene_prompts_bind_one_target_to_one_of_multiple_bins() -> None:
    fastener = acceptance._instructions_for_backend(
        "graspgenx", scenario="fastener-bin-sort"
    )
    tool = acceptance._instructions_for_backend("graspgenx", scenario="tool-bin-sort")

    # Keep the SAM3 query visually grounded; the scenario instruction below
    # still binds that unique red instance to the industrial hex-bolt role.
    fastener_metadata = acceptance._automation_metadata_for_backend(
        "graspgenx", scenario="fastener-bin-sort"
    )
    assert "grasp_target=red_object" in fastener_metadata
    assert "placement_region=blue_square_area_inside_bin" in fastener_metadata
    assert "红色六角螺栓" in fastener and "蓝色零件箱" in fastener
    assert "red object" not in fastener and "blue square area inside bin" not in fastener
    tool_metadata = acceptance._automation_metadata_for_backend(
        "graspgenx", scenario="tool-bin-sort"
    )
    assert "grasp_target=yellow_open_end_tool" in tool_metadata
    assert "placement_region=green_square_area_inside_bin" in tool_metadata
    assert "黄色开口扳手" in tool and "绿色工具箱" in tool
    assert "yellow open end tool" not in tool and "green square area inside bin" not in tool


def test_agentic_profile_does_not_override_deployment_provider_policy() -> None:
    assert acceptance.AGENTIC_PROVIDER_RESILIENCE_ENV == {}


def test_profile_can_tune_request_bounds_without_replacing_provider_identity() -> None:
    config = acceptance.base.PlannerProviderConfig(
        provider="configured-provider",
        model="configured-model",
        api_base="https://configured.invalid/v1",
        api_key="configured-secret",
        timeout_s=30.0,
        max_attempts=2,
        retry_backoff_s=1.0,
        max_tokens=32_768,
        thinking_mode="disabled",
        fallback=ProviderEndpointConfig(
            provider="configured-fallback-provider",
            model="configured-fallback-model",
            api_base="https://configured-fallback.invalid/v1",
            api_key="configured-fallback-secret",
            timeout_s=45.0,
        ),
    )

    environment = acceptance.base._tui_provider_environment(
        config,
        profile_environment={
            **acceptance.AGENTIC_PROVIDER_RESILIENCE_ENV,
            "OPENETA_LLM_PROVIDER": "unexpected-provider",
            "OPENETA_LLM_MODEL": "unexpected-model",
            "OPENETA_LLM_API_KEY": "unexpected-secret",
            "OPENETA_LLM_FALLBACK_API_BASE": "https://unexpected.invalid/v1",
            "OPENETA_LLM_FALLBACK_API_KEY": "unexpected-fallback-secret",
            "OPENETA_LLM_THINKING_MODE": "enabled",
            "OPENETA_UNRELATED": "ignored",
        },
    )

    assert environment["OPENETA_LLM_PROVIDER"] == "configured-provider"
    assert environment["OPENETA_LLM_MODEL"] == "configured-model"
    assert environment["OPENETA_LLM_API_KEY"] == "configured-secret"
    assert environment["OPENETA_LLM_TIMEOUT_S"] == "30.0"
    assert environment["OPENETA_LLM_MAX_ATTEMPTS"] == "2"
    assert environment["OPENETA_LLM_RETRY_BACKOFF_S"] == "1.0"
    assert environment["OPENETA_LLM_MAX_TOKENS"] == "32768"
    assert environment["OPENETA_LLM_THINKING_MODE"] == "disabled"
    assert environment["OPENETA_LLM_FALLBACK_PROVIDER"] == "configured-fallback-provider"
    assert environment["OPENETA_LLM_FALLBACK_MODEL"] == "configured-fallback-model"
    assert (
        environment["OPENETA_LLM_FALLBACK_API_BASE"]
        == "https://configured-fallback.invalid/v1"
    )
    assert environment["OPENETA_LLM_FALLBACK_API_KEY"] == "configured-fallback-secret"
    assert environment["OPENETA_LLM_FALLBACK_TIMEOUT_S"] == "45.0"
    assert "OPENETA_UNRELATED" not in environment


def test_service_preflight_pool_expectations_follow_runtime_environment(
    monkeypatch,
) -> None:
    assert acceptance._expected_service_raw_pool_size("openeta-graspgenx") == 512
    assert acceptance._expected_service_raw_pool_size("openeta-anyplace") == 96

    monkeypatch.setenv("OPENETA_GRASPGENX_RAW_POOL_SIZE", "384")
    assert acceptance._expected_service_raw_pool_size("openeta-graspgenx") == 384

    monkeypatch.setenv("OPENETA_GRASPGENX_RAW_POOL_SIZE", "invalid")
    assert acceptance._expected_service_raw_pool_size("openeta-graspgenx") == -1


def test_pick_place_prepare_can_strictly_select_graspgenx(tmp_path, monkeypatch) -> None:
    allocation = acceptance.base.Allocation(81, "openeta-pick-place-test", 18765, "run-id")
    monkeypatch.setattr(acceptance.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        acceptance.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )
    services = acceptance._services_for_backend(
        "graspgenx",
        sam3_url="http://sam3/sse",
        anygrasp_url="http://anygrasp/sse",
        anyplace_url="http://anyplace/sse",
        graspgenx_url="http://graspgenx/sse",
    )

    paths = acceptance.prepare_case(
        tmp_path,
        tmp_path / "run",
        allocation,
        services,
        grasp_backend="graspgenx",
    )

    config = json.loads(paths.mcp_config.read_text(encoding="utf-8"))["mcpServers"]
    assert set(config) == {
        "openeta-sim",
        "openeta-sam3",
        "openeta-graspgenx",
        "openeta-anyplace",
    }
    assert "openeta-anygrasp" not in config
    assert "GraspGenX" not in paths.instructions.read_text(encoding="utf-8")
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert receipt["grasp_backend_mode"] == "graspgenx"
    assert receipt["qualification_profile"] == "fast_v3"


def test_pick_place_prepare_smoke_normal_is_explicitly_no_vlm(tmp_path, monkeypatch) -> None:
    allocation = acceptance.base.Allocation(81, "openeta-smoke-normal", 18765, "run-id")
    monkeypatch.setattr(acceptance.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        acceptance.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )

    paths = acceptance.prepare_case(
        tmp_path,
        tmp_path / "run",
        allocation,
        dict(acceptance.DEFAULT_SERVICES),
        execution_profile="smoke_normal",
    )

    prompt = paths.instructions.read_text(encoding="utf-8")
    assert "不调用视觉语言模型" in prompt
    assert "你是 OpenETA 闭环 Planner" not in prompt
    assert "再由 Planner" not in prompt
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert receipt["execution_profile"] == "smoke_normal"
    assert receipt["planner_mode"] == "host_macro"
    assert receipt["planner_provider_expected"] is False
    assert receipt["qualification_profile"] == "fast_v3"
    assert receipt["acceptance_scope"] == (
        "control_only_no_vlm_smoke_normal_not_agentic_acceptance"
    )


def test_pick_place_acceptance_can_explicitly_roll_back_to_legacy(tmp_path, monkeypatch) -> None:
    allocation = acceptance.base.Allocation(81, "openeta-legacy-normal", 18765, "run-id")
    monkeypatch.setattr(acceptance.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        acceptance.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )

    paths = acceptance.prepare_case(
        tmp_path,
        tmp_path / "run",
        allocation,
        dict(acceptance.DEFAULT_SERVICES),
        qualification_profile="legacy",
    )

    prompt = paths.instructions.read_text(encoding="utf-8")
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert "qualification_profile=legacy" not in prompt
    assert "qualification_profile=legacy" in acceptance._automation_metadata_for_backend(
        "anygrasp", qualification_profile="legacy"
    )
    assert receipt["qualification_profile"] == "legacy"


def test_pick_place_acceptance_parser_defaults_to_fast_v3() -> None:
    args = acceptance._parser().parse_args([])

    assert args.qualification_profile == "fast_v3"
    assert args.operator_mode == "scripted_tui"
    assert args.task_variant == "wrench-green-bolt-blue"
    assert "multi_normal" in acceptance.SCENARIOS
    assert not {"multi_normal1", "multi_normal2", "multi_normal3"}.intersection(
        acceptance.SCENARIOS
    )


def test_task_variant_cannot_change_a_non_multi_physical_scene() -> None:
    with pytest.raises(ValueError, match="only valid with a multi-sort scenario"):
        acceptance._validated_task_variant("normal", "bolt-green-wrench-blue")


def test_smoke_normal_planner_evidence_requires_host_only_and_zero_tokens() -> None:
    def host_action(name: str) -> dict:
        return {
            "event_type": "action",
            "payload": {
                "command": {
                    "request": {"kind": "tool_call", "name": name, "parameters": {}},
                    "metadata": {
                        "planner_metadata": {
                            "execution_model": "host_obligation_dispatch",
                            "planner_mode": "host_macro",
                            "host_obligation": {"schema_version": "openeta.fixture_obligation.v1"},
                        }
                    },
                }
            },
        }

    events = [host_action(f"tool-{index}") for index in range(10)]
    events.append(
        {
            "event_type": "episode_result",
            "payload": {"metadata": {"usage": {"total_tokens": 0}}},
        }
    )
    evidence = acceptance._planner_evidence(events, expected_planner_mode="host_macro")

    assert (
        acceptance._planner_evidence_errors(
            evidence,
            execution_profile="smoke_normal",
        )
        == []
    )
    assert evidence["host_dispatch_count"] == 10
    assert evidence["closed_loop_action_count"] == 0
    assert evidence["total_tokens"] == 0


def test_normal_order_helper_requires_frozen_anyplace_pool_before_grasp() -> None:
    valid = ["observe", "anyplace", "anygrasp", "move_to", "gripper_control"]
    invalid = ["observe", "anygrasp", "move_to", "gripper_control", "anyplace"]

    required = ("observe", "anyplace", "anygrasp", "move_to", "gripper_control")
    assert acceptance._ordered(valid, required)
    assert not acceptance._ordered(invalid, required)


def test_acceptance_reports_agent_route_findings_without_failing_the_result() -> None:
    assert acceptance._is_non_blocking_flow_finding(
        "exactly one target-object and one placement-region SAM3 call is required"
    )
    assert acceptance._is_non_blocking_flow_finding(
        "AnyPlace model inference count does not match assignments"
    )
    assert not acceptance._is_non_blocking_flow_finding(
        "stable in-zone placement verification is missing per assignment"
    )


def _ordered_call(
    name: str,
    *,
    success: bool = True,
    parameters: dict | None = None,
    assignment_id: str | None = None,
    anyplace_inference: bool | None = None,
) -> dict:
    outputs = {}
    if assignment_id is not None:
        outputs["native_target_binding"] = {"assignment_id": assignment_id}
    if anyplace_inference is not None:
        outputs["anyplace_model_inference_invoked"] = anyplace_inference
    return {
        "name": name,
        "parameters": parameters or {},
        "result": {
            "success": success,
            "details": {"outputs": outputs},
        },
    }


def test_assignment_order_uses_create_and_post_release_observations() -> None:
    first = "red_bolt_to_green_parts_bin"
    second = "yellow_wrench_to_blue_parts_bin"
    calls = [
        _ordered_call("anyplace", anyplace_inference=True),
        _ordered_call("graspgenx"),
        _ordered_call(
            "move_to",
            success=False,
            parameters={"target_pose": {"grasp_stage": "contact"}},
        ),
        _ordered_call(
            "move_to",
            parameters={
                "target_pose": {
                    "purpose": "grasp_recovery_restore",
                    "grasp_stage": "recovery_restore",
                }
            },
        ),
        _ordered_call(
            "move_to",
            parameters={"target_pose": {"grasp_stage": "contact"}},
        ),
        _ordered_call("gripper_control", parameters={"position": 0}, assignment_id=first),
        _ordered_call(
            "anyplace",
            anyplace_inference=False,
        ),
        _ordered_call(
            "move_to",
            parameters={"target_pose": {"purpose": "placement"}},
            assignment_id=first,
        ),
        _ordered_call("gripper_control", parameters={"position": 1}, assignment_id=first),
        _ordered_call("anyplace", anyplace_inference=True),
        _ordered_call("graspgenx"),
        _ordered_call(
            "move_to",
            parameters={"target_pose": {"grasp_stage": "contact"}},
        ),
        _ordered_call("gripper_control", parameters={"position": 0}, assignment_id=second),
        _ordered_call(
            "anyplace",
            anyplace_inference=False,
        ),
        _ordered_call(
            "move_to",
            parameters={"target_pose": {"purpose": "placement"}},
            assignment_id=second,
        ),
        _ordered_call("gripper_control", parameters={"position": 1}, assignment_id=second),
    ]
    assignments = [{"id": first}, {"id": second}]

    assert acceptance._ordered_assignment_execution(
        calls,
        assignments,
        backend="graspgenx",
    )


def test_assignment_order_rejects_cross_assignment_release_evidence() -> None:
    assignment_id = "red_bolt_to_green_parts_bin"
    calls = [
        _ordered_call("anyplace", anyplace_inference=True),
        _ordered_call("graspgenx"),
        _ordered_call(
            "move_to",
            parameters={"target_pose": {"grasp_stage": "contact"}},
        ),
        _ordered_call(
            "gripper_control",
            parameters={"position": 0},
            assignment_id=assignment_id,
        ),
        _ordered_call(
            "anyplace",
            assignment_id=assignment_id,
            anyplace_inference=False,
        ),
        _ordered_call(
            "move_to",
            parameters={"target_pose": {"purpose": "placement"}},
            assignment_id="yellow_wrench_to_blue_parts_bin",
        ),
        _ordered_call(
            "gripper_control",
            parameters={"position": 1},
            assignment_id=assignment_id,
        ),
    ]

    assert not acceptance._ordered_assignment_execution(
        calls,
        [{"id": assignment_id}],
        backend="graspgenx",
    )


def test_normal_canonicalizes_public_grasp_tool_only_with_real_anygrasp_backend() -> None:
    assert (
        acceptance._name(
            {
                "name": "grasp_pose_estimate",
                "result": {"details": {"backend": "anygrasp_mcp"}},
            }
        )
        == "anygrasp"
    )
    assert acceptance._name({"name": "grasp_pose_estimate"}) == "grasp_pose_estimate"


def test_normal_requires_only_executable_public_grasp_tools() -> None:
    assert "grasp_pose_estimate" in acceptance.REQUIRED_REAL_PICK_PLACE_TOOLS
    assert "observe" not in acceptance.REQUIRED_REAL_PICK_PLACE_TOOLS
    assert "anygrasp" not in acceptance.REQUIRED_REAL_PICK_PLACE_TOOLS
    assert "grasp_pose_estimate" not in acceptance._required_tools_for_backend("anygrasp")
    assert "anygrasp" in acceptance._required_tools_for_backend("anygrasp")
    assert "grasp_pose_estimate" not in acceptance._required_tools_for_backend("graspgenx")
    assert "graspgenx" in acceptance._required_tools_for_backend("graspgenx")


def test_normal_health_url_preserves_service_root() -> None:
    assert acceptance._health_url("http://127.0.0.1:8778/sse") == "http://127.0.0.1:8778/"


def test_normal_runtime_preflight_accepts_the_selected_overlay(tmp_path, monkeypatch) -> None:
    overlay = tmp_path / "external-overlay"
    package_prefix = overlay / acceptance.GAZEBO_SIM_PACKAGE
    monkeypatch.setenv("OPENETA_GAZEBO_OVERLAY", str(overlay))
    monkeypatch.setattr(acceptance, "_ros_python_import_error", lambda: "")
    monkeypatch.setattr(acceptance, "_gazebo_package_prefix", lambda _package: package_prefix)
    monkeypatch.setattr(acceptance.shutil, "which", lambda name: f"/usr/bin/{name}")

    result = acceptance.gazebo_runtime_preflight(tmp_path)

    assert result["status"] == "passed"
    assert result["expected_overlay"] == str(overlay.resolve())
    assert result["package_prefix"] == str(package_prefix.resolve())
    assert result["reason_codes"] == []


def test_normal_runtime_preflight_rejects_an_overlay_from_another_checkout(
    tmp_path, monkeypatch
) -> None:
    expected = tmp_path / "extensions/gazebo/ros2_ws/install"
    monkeypatch.delenv("OPENETA_GAZEBO_OVERLAY", raising=False)
    monkeypatch.setattr(acceptance, "_ros_python_import_error", lambda: "")
    monkeypatch.setattr(
        acceptance,
        "_gazebo_package_prefix",
        lambda _package: tmp_path / "stale-checkout/install" / acceptance.GAZEBO_SIM_PACKAGE,
    )
    monkeypatch.setattr(acceptance.shutil, "which", lambda name: f"/usr/bin/{name}")

    result = acceptance.gazebo_runtime_preflight(tmp_path)

    assert result["status"] == "blocked"
    assert result["expected_overlay"] == str(expected.resolve())
    assert result["reason_codes"] == ["OPENETA_GAZEBO_OVERLAY_PACKAGE_MISMATCH"]


def test_normal_canonical_runner_sources_ros_and_executes_normal() -> None:
    source = NORMAL_RUNNER.read_text(encoding="utf-8")

    assert os.access(NORMAL_RUNNER, os.X_OK)
    assert 'source "${SYSTEM_ROS_SETUP}"' in source
    assert 'source "${OVERLAY_SETUP}"' in source
    assert "import rclpy; from rosgraph_msgs.msg import Clock" in source
    assert "ros2 pkg prefix openeta_rm75_robotiq2f85_sim" in source
    assert 'normal_gazebo_acceptance.py" "$@"' in source

    capability_source = PICK_PLACE_RUNNER.read_text(encoding="utf-8")
    assert os.access(PICK_PLACE_RUNNER, os.X_OK)
    assert 'run_normal_gazebo_acceptance.sh" "$@"' in capability_source


def test_normal_verifier_uses_model_raw_count_for_frozen_goal_requalification() -> None:
    call = {
        "result": {
            "details": {
                "outputs": {
                    "model_raw_candidate_count": 96,
                    "raw_candidate_count": 4,
                    "frozen_goal_count": 4,
                }
            }
        }
    }

    assert acceptance._has_minimum_int_value(call, "model_raw_candidate_count", 96)
    assert not acceptance._has_minimum_int_value(call, "raw_candidate_count", 96)


def test_normal_candidate_counts_ignore_nested_wave_cardinality() -> None:
    call = {
        "result": {
            "details": {
                "outputs": {
                    "candidate_count": 1,
                    "full_plan_pass_count": 1,
                    "qualification_evidence": {"waves": [{"candidate_count": 2}]},
                    "qualification_waves": [{"candidate_count": 2}],
                }
            }
        }
    }

    outputs = acceptance._call_outputs(call)

    assert outputs["candidate_count"] == 1
    assert outputs["full_plan_pass_count"] == 1


def test_normal_best_first_frontier_may_keep_l5_backup_private() -> None:
    assert acceptance._candidate_pass_counts_consistent(1, 2)
    assert acceptance._candidate_pass_counts_consistent(1, 1)
    assert not acceptance._candidate_pass_counts_consistent(2, 1)
    assert not acceptance._candidate_pass_counts_consistent(0, 2)
    assert not acceptance._candidate_pass_counts_consistent(True, 2)


def test_process_continuity_allows_an_independent_acceptance_job_to_finish() -> None:
    before = [
        {
            "pid": 101,
            "start_time_ticks": 11,
            "cmdline": "gz sim independent",
            "openeta_tui_run_id": "another-run",
        },
        {
            "pid": 202,
            "start_time_ticks": 22,
            "cmdline": "gz sim operator",
        },
    ]
    after = [dict(before[1])]

    evidence = acceptance.base._preexisting_process_continuity(before, after)

    assert evidence["preexisting_process_snapshot_unchanged"] is False
    assert evidence["preexisting_unmanaged_process_snapshot_unchanged"] is True
    assert evidence["preexisting_missing_managed_processes"] == [before[0]]
    assert evidence["preexisting_missing_unmanaged_processes"] == []


def test_process_continuity_rejects_pid_reuse_or_external_process_loss() -> None:
    before = [{"pid": 202, "start_time_ticks": 22, "cmdline": "gz sim operator"}]
    after = [{"pid": 202, "start_time_ticks": 99, "cmdline": "gz sim replacement"}]

    evidence = acceptance.base._preexisting_process_continuity(before, after)

    assert evidence["preexisting_unmanaged_process_snapshot_unchanged"] is False
    assert evidence["preexisting_missing_unmanaged_processes"] == before


def test_normal_failed_fingerprint_check_ignores_receipt_mirrors() -> None:
    event = {
        "execution_started": False,
        "request_fingerprint": "fingerprint-a",
        "result": {
            "details": {
                "environment_receipt": {
                    "execution_started": False,
                    "request_fingerprint": "fingerprint-a",
                },
                "outputs": {
                    "mcp_calls": [
                        {
                            "environment_receipt": {
                                "execution_started": False,
                                "request_fingerprint": "fingerprint-a",
                            }
                        }
                    ]
                },
            }
        },
    }

    assert not acceptance._repeated_failed_motion_fingerprints([event])
    assert acceptance._repeated_failed_motion_fingerprints([event, event]) == {"fingerprint-a"}


def test_normal_qualification_blocks_resolve_relative_to_case_root(tmp_path) -> None:
    artifact = (
        tmp_path
        / ".openeta_memory"
        / "sessions"
        / "session-a"
        / "artifacts"
        / "moveit_qualification"
        / "qualification.json"
    )
    artifact.parent.mkdir(parents=True)
    proof = {
        "results": [
            {
                "candidate_id": "placement_000",
                "verdict": "FAIL",
                "reason": "plan_only_failed",
                "execution_started": False,
                "full_plan_submitted": True,
            }
        ]
    }
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    call = {
        "qualification_artifact": {
            "kind": "json",
            "path": str(artifact.relative_to(tmp_path)),
        }
    }

    assert acceptance._qualification_blocks(call, artifact_root=tmp_path) == [proof]


def test_normal_qualification_blocks_include_frozen_pair_proof(tmp_path) -> None:
    artifact = (
        tmp_path
        / ".openeta_memory"
        / "sessions"
        / "session-a"
        / "artifacts"
        / "moveit_qualification"
        / "frozen-pair.json"
    )
    artifact.parent.mkdir(parents=True)
    proof = {
        "purpose": "placement",
        "results": [
            {
                "candidate_id": "frozen_pair_grasp_000_placement_000",
                "verdict": "FAIL",
                "reason": "plan_only_failed",
                "execution_started": False,
                "full_plan_submitted": True,
            }
        ],
    }
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    call = {
        "frozen_pair_qualification_artifact": {
            "kind": "json",
            "path": str(artifact.relative_to(tmp_path)),
        }
    }

    assert acceptance._qualification_blocks(call, artifact_root=tmp_path) == [proof]


def test_normal_qualification_blocks_include_internal_frontier_proofs(tmp_path) -> None:
    proofs = [
        {"purpose": purpose, "results": [{"candidate_id": candidate_id}]}
        for purpose, candidate_id in (("grasp", "g2"), ("placement", "pair-g2-p0"))
    ]
    artifacts = []
    for index, proof in enumerate(proofs):
        path = tmp_path / f"frontier-{index}.json"
        path.write_text(json.dumps(proof), encoding="utf-8")
        artifacts.append({"kind": "json", "path": path.name})

    call = {
        "frozen_grasp_frontier_qualification_artifacts": [artifacts[0]],
        "frozen_pair_qualification_artifacts": [artifacts[1]],
    }

    assert acceptance._qualification_blocks(call, artifact_root=tmp_path) == [
        proofs[1],
        proofs[0],
    ]


def test_normal_accepts_complete_v3_grasp_pool_with_two_l5_branches(tmp_path) -> None:
    artifact = tmp_path / "qualification-v3.json"
    results = [
        {
            "candidate_id": f"grasp_{index:03d}",
            "endpoint_pass": index < 2,
            "se3_cluster_id": f"se3_{index:04d}",
            "verdict": "PASS" if index < 2 else "FAIL",
        }
        for index in range(65)
    ]
    proof = {
        "schema_version": "openeta.moveit_candidate_funnel.v3",
        "artifact_schema_version": "openeta.moveit_candidate_qualification.v3",
        "purpose": "grasp",
        "stop_reason": "complete_l5_pass_found",
        "selected_candidate_ids": ["grasp_000", "grasp_001"],
        "metrics": {"generated_count": 65, "l5_pass_count": 2},
        "l5_attempts": [
            {"candidate_id": "grasp_000", "verdict": "PASS"},
            {"candidate_id": "grasp_001", "verdict": "PASS"},
        ],
        "results": results,
    }
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    call = {
        "diversity_selected_count": 65,
        "qualification_artifact": {"kind": "json", "path": artifact.name},
    }

    assert acceptance._has_v3_grasp_search_evidence(call, artifact_root=tmp_path)
    assert acceptance._has_bounded_grasp_l5_evidence(call, artifact_root=tmp_path)

    proof["selected_candidate_ids"] = ["grasp_000"]
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    assert not acceptance._has_v3_grasp_search_evidence(call, artifact_root=tmp_path)
    assert not acceptance._has_bounded_grasp_l5_evidence(call, artifact_root=tmp_path)


def test_normal_accepts_one_grasp_pose_with_two_independent_l5_joint_branches(
    tmp_path,
) -> None:
    artifact = tmp_path / "qualification-v3-joint-branches.json"
    results = [
        {
            "candidate_id": f"grasp_{index:03d}",
            "endpoint_pass": index == 0,
            "se3_cluster_id": f"se3_{index:04d}",
            "verdict": "PASS" if index == 0 else "FAIL",
        }
        for index in range(82)
    ]
    proof = {
        "schema_version": "openeta.moveit_candidate_funnel.v3",
        "artifact_schema_version": "openeta.moveit_candidate_qualification.v3",
        "purpose": "grasp",
        "stop_reason": "complete_l5_pass_found_joint_branch_fallback",
        "selected_candidate_ids": ["grasp_000"],
        "selected_joint_branches": [
            {
                "candidate_id": "grasp_000",
                "joint_branch_index": 0,
                "verdict": "PASS",
                "selected_ik_joint_state_sha256": "state-a",
            },
            {
                "candidate_id": "grasp_000",
                "joint_branch_index": 1,
                "verdict": "PASS",
                "selected_ik_joint_state_sha256": "state-b",
                "normalized_distance_from_primary": 0.2,
            },
        ],
        "metrics": {
            "generated_count": 82,
            "l5_pass_count": 1,
            "l5_joint_branch_pass_count": 2,
        },
        "l5_attempts": [
            {
                "candidate_id": "grasp_000",
                "verdict": "PASS",
                "joint_branch_index": 0,
                "joint_branch_joint_state_sha256": "state-a",
            },
            {
                "candidate_id": "grasp_000",
                "verdict": "PASS",
                "joint_branch_index": 1,
                "joint_branch_joint_state_sha256": "state-b",
                "joint_branch_normalized_distance": 0.2,
            },
        ],
        "results": results,
    }
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    call = {
        "diversity_selected_count": 82,
        "qualification_artifact": {"kind": "json", "path": artifact.name},
    }

    assert acceptance._has_v3_grasp_search_evidence(call, artifact_root=tmp_path)
    assert acceptance._has_bounded_grasp_l5_evidence(call, artifact_root=tmp_path)

    proof["selected_joint_branches"][1][
        "normalized_distance_from_primary"
    ] = 0.049
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    assert not acceptance._has_v3_grasp_search_evidence(call, artifact_root=tmp_path)
    assert not acceptance._has_bounded_grasp_l5_evidence(call, artifact_root=tmp_path)


def test_normal_accepts_single_l5_grasp_only_with_exhaustive_degradation_evidence(
    tmp_path,
) -> None:
    artifact = tmp_path / "qualification-v3-single-exhaustive.json"
    results = [
        {
            "candidate_id": f"grasp_{index:03d}",
            "endpoint_pass": index == 7,
            "se3_cluster_id": f"se3_{index:04d}",
            "verdict": "PASS" if index == 7 else "FAIL",
        }
        for index in range(96)
    ]
    proof = {
        "schema_version": "openeta.moveit_candidate_funnel.v3",
        "artifact_schema_version": "openeta.moveit_candidate_qualification.v3",
        "purpose": "grasp",
        "stop_reason": (
            "complete_l5_pass_found_single_branch_exhaustive_fallback"
        ),
        "selected_candidate_ids": ["grasp_007"],
        "metrics": {
            "generated_count": 96,
            "l5_pass_count": 1,
            "l5_joint_branch_pass_count": 1,
        },
        "l5_attempts": [
            {
                "candidate_id": "grasp_007",
                "verdict": "PASS",
                "joint_branch_index": 0,
            }
        ],
        "search_exhaustion": {
            "fast_wave_count_expected": 4,
            "fast_wave_count_completed": 4,
            "fast_pool_exhausted": True,
            "recovery_wave_count_expected": 4,
            "recovery_wave_count_completed": 4,
            "recovery_pool_exhausted": True,
            "preferred_grasp_branch_target": 2,
            "published_grasp_branch_count": 1,
            "redundancy_degraded": True,
        },
        "infrastructure_error": False,
        "results": results,
    }
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    call = {
        "diversity_selected_count": 96,
        "qualification_artifact": {"kind": "json", "path": artifact.name},
    }

    assert acceptance._has_v3_grasp_search_evidence(call, artifact_root=tmp_path)
    assert acceptance._has_bounded_grasp_l5_evidence(call, artifact_root=tmp_path)

    proof["search_exhaustion"]["recovery_pool_exhausted"] = False
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    assert not acceptance._has_v3_grasp_search_evidence(call, artifact_root=tmp_path)
    assert not acceptance._has_bounded_grasp_l5_evidence(call, artifact_root=tmp_path)


def test_normal_accepts_single_primary_with_a_diverse_model_frozen_recovery_frontier(
    tmp_path,
) -> None:
    artifact = tmp_path / "qualification-v3-deferred-frontier.json"
    results = [
        {
            "candidate_id": f"grasp_{index:03d}",
            "endpoint_pass": index == 3,
            "se3_cluster_id": f"se3_{index:04d}",
            "verdict": "PASS" if index == 3 else "NOT_EVALUATED",
        }
        for index in range(16)
    ]
    proof = {
        "schema_version": "openeta.moveit_candidate_funnel.v3",
        "artifact_schema_version": "openeta.moveit_candidate_qualification.v3",
        "purpose": "grasp",
        "stop_reason": "complete_l5_pass_found",
        "selected_candidate_ids": ["grasp_003"],
        "metrics": {
            "generated_count": 16,
            "l5_pass_count": 1,
            "l5_pass_target": 1,
            "l5_min_pass_target": 1,
        },
        "l5_attempts": [
            {"candidate_id": "grasp_003", "verdict": "PASS"},
        ],
        "results": results,
    }
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    outputs = {
        "candidate_count": 1,
        "grasp_candidates": [{"id": "grasp_003"}],
        "diversity_selected_count": 16,
        "qualification_artifact": {"kind": "json", "path": artifact.name},
        "frozen_pair_execution_target": 1,
        "frozen_pair_qualified_grasp_count": 1,
        "frozen_pair_recovery_policy": (
            "resume_frozen_frontier_after_execution_failure"
        ),
        "frozen_pair_stop_reason": "complete_pair_found",
        "frozen_grasp_frontier_remaining_count": 15,
        "frozen_grasp_frontier_model_inference_invoked": False,
    }
    call = {"result": {"details": {"outputs": outputs}}}

    assert acceptance._has_v3_grasp_search_evidence(call, artifact_root=tmp_path)

    proof["l5_attempts"] = [
        {
            "candidate_id": "grasp_003",
            "verdict": "PASS",
            "joint_branch_index": 0,
            "joint_branch_joint_state_sha256": "state-a",
        },
        {
            "candidate_id": "grasp_003",
            "verdict": "PASS",
            "joint_branch_index": 1,
            "joint_branch_joint_state_sha256": "state-b",
        },
    ]
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    assert acceptance._has_v3_grasp_search_evidence(call, artifact_root=tmp_path)

    for field, invalid in (
        ("frozen_grasp_frontier_remaining_count", 0),
        ("frozen_grasp_frontier_model_inference_invoked", True),
        ("frozen_pair_recovery_policy", "rerun_model"),
    ):
        original = outputs[field]
        outputs[field] = invalid
        assert not acceptance._has_v3_grasp_search_evidence(
            call, artifact_root=tmp_path
        )
        outputs[field] = original


def test_fast_v3_pair_evidence_requires_one_primary_and_frozen_recovery_tail() -> None:
    call = {
        "result": {"details": {"outputs": {
            "candidate_count": 1,
            "grasp_candidates": [{"id": "g0"}],
            "frozen_pair_execution_target": 1,
            "frozen_pair_qualified_grasp_count": 1,
            "frozen_pair_stop_reason": "complete_pair_found",
            "frozen_pair_recovery_policy": (
                "resume_frozen_frontier_after_execution_failure"
            ),
            "frozen_grasp_frontier_remaining_count": 17,
            "frozen_grasp_frontier_model_inference_invoked": False,
        }}}
    }
    outputs = call["result"]["details"]["outputs"]

    assert acceptance._has_resumable_frozen_pair_evidence(call)

    for field, invalid in (
        ("candidate_count", 2),
        ("frozen_grasp_frontier_remaining_count", 0),
        ("frozen_pair_recovery_policy", "rerun_model"),
        ("frozen_grasp_frontier_model_inference_invoked", True),
    ):
        original = outputs[field]
        outputs[field] = invalid
        assert not acceptance._has_resumable_frozen_pair_evidence(call)
        outputs[field] = original

    outputs["grasp_candidates"] = []
    assert not acceptance._has_resumable_frozen_pair_evidence(call)


def test_normal_rejects_obsolete_four_branch_lookahead(tmp_path) -> None:
    artifact = tmp_path / "qualification-v3-reserve.json"
    results = [
        {
            "candidate_id": f"grasp_{index:03d}",
            "endpoint_pass": index < 4,
            "se3_cluster_id": f"se3_{index:04d}",
            "verdict": "PASS" if index < 4 else "FAIL",
        }
        for index in range(65)
    ]
    proof = {
        "schema_version": "openeta.moveit_candidate_funnel.v3",
        "artifact_schema_version": "openeta.moveit_candidate_qualification.v3",
        "purpose": "grasp",
        "stop_reason": "complete_l5_pass_found",
        "selected_candidate_ids": [f"grasp_{index:03d}" for index in range(4)],
        "metrics": {"generated_count": 65, "l5_pass_count": 4},
        "l5_attempts": [
            {"candidate_id": f"grasp_{index:03d}", "verdict": "PASS"} for index in range(4)
        ],
        "results": results,
    }
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    call = {
        "diversity_selected_count": 65,
        "qualification_artifact": {"kind": "json", "path": artifact.name},
    }

    assert not acceptance._has_v3_grasp_search_evidence(call, artifact_root=tmp_path)
    assert not acceptance._has_bounded_grasp_l5_evidence(call, artifact_root=tmp_path)


def test_normal_agentic_planner_evidence_counts_model_and_bounded_host_actions() -> None:
    def action(execution_model, *, schema="", name="observe", tokens=0):
        planner_metadata = {
            "execution_model": execution_model,
            "planner_mode": "agentic_closed_loop",
            "backend_provider": "fixture-provider",
            "backend_model": "fixture-model",
            "backend_usage": {"total_tokens": tokens},
        }
        if schema:
            planner_metadata["host_obligation"] = {
                "schema_version": schema,
            }
        return {
            "event_type": "action",
            "payload": {
                "command": {
                    "request": {
                        "kind": "tool_call",
                        "name": name,
                        "parameters": {},
                    },
                    "metadata": {"planner_metadata": planner_metadata},
                }
            },
        }

    events = [
        action("closed_loop_tool_calling", name="sam3", tokens=120),
        action(
            "host_obligation_dispatch",
            schema="openeta.fresh_observation_obligation.v1",
        ),
        {
            "event_type": "episode_result",
            "payload": {
                "metadata": {
                    "usage": {
                        "total_tokens": 120,
                        "token_usage_sources": {"provider": 1},
                    }
                }
            },
        },
    ]

    evidence = acceptance._agentic_planner_evidence(events)

    assert evidence["closed_loop_tool_call_count"] == 1
    assert evidence["host_dispatches"] == [
        {
            "schema_version": "openeta.fresh_observation_obligation.v1",
            "tool": "observe",
        }
    ]
    assert evidence["total_tokens"] == 120
    assert evidence["providers"] == ["fixture-provider"]


def test_scripted_tui_quit_timeout_returns_through_cleanup_path(tmp_path, monkeypatch) -> None:
    instructions = tmp_path / "instructions.txt"
    instructions.write_text("task\n", encoding="utf-8")
    paths = SimpleNamespace(
        root=tmp_path,
        transcript=tmp_path / "tui.transcript",
        instructions=instructions,
    )

    class Stdin:
        closed = False

        def write(self, value):
            return len(value)

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class Process:
        pid = 12345
        stdin = Stdin()

        def poll(self):
            return None

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("tui", timeout)

    process = Process()
    terminated = []
    monkeypatch.setattr(acceptance.base.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        acceptance.base, "_wait_for_scripted_tui_episode", lambda *args, **kwargs: "completed"
    )
    monkeypatch.setattr(
        acceptance.base, "_terminate_scripted_tui_process", lambda value: terminated.append(value)
    )

    assert acceptance.base._run_scripted_tui("tui", paths, {}) == 1
    assert terminated == [process]
    evidence = json.loads((tmp_path / "scripted-tui-driver.json").read_text())
    assert evidence["reason_code"] == "TUI_DID_NOT_EXIT_AFTER_QUIT"
