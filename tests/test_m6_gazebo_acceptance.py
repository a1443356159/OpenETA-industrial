from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from scripts import m6_gazebo_acceptance as m6


M6_RUNNER = Path(__file__).resolve().parents[1] / "scripts/run_m6_gazebo_acceptance.sh"
PICK_PLACE_RUNNER = (
    Path(__file__).resolve().parents[1] / "scripts/run_pick_place_acceptance.sh"
)


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

    assert paths.root == tmp_path / "run" / "pick-place" / m6.MODE
    config = json.loads(paths.mcp_config.read_text(encoding="utf-8"))["mcpServers"]
    assert set(config) == {
        "openeta-sim",
        "openeta-sam3",
        "openeta-anygrasp",
        "openeta-anyplace",
    }
    prompt = paths.instructions.read_text(encoding="utf-8")
    assert (
        "planner_mode=agentic_closed_loop; "
        f"environment_id={m6.ENV_ID}; environment_task=normal_pick_and_place"
        in prompt
    )
    assert "qualification_profile=fast_v3" in prompt
    assert "grasp_pose_estimate" in prompt and "AnyPlace" in prompt
    assert "精确 EEF contact" in prompt
    assert "一次规划到精确 contact" in prompt
    assert "同一次工具调用内" in prompt
    assert "不增加 TUI/VLM 回合" in prompt
    assert "最终窗口 >=0.5 s" in prompt
    assert "不得" in prompt and "Oracle" in prompt
    assert "initial observation 不计作这次显式 observe" in prompt
    assert "覆盖完整目标轮廓" in prompt
    assert "红色方块对应场景中的 target_object" in prompt
    assert "`red rectangular block`" in prompt
    assert "独立 placement RGB-D" in prompt
    assert "冻结目标池" in prompt
    assert "Best-first 小波次" in prompt and "4 → 8 → 16 → 32 → 剩余" in prompt
    assert "frozen_frontier" in prompt and "不重跑" in prompt
    assert "主 VLM 必须" in prompt and "不能" in prompt
    assert "compile_placement_seed" not in prompt
    assert "实测 T_eef_object" in prompt
    assert "不得固定 detection id" in prompt
    assert "不得调用 python_exec" in prompt


def test_pick_place_prepare_can_strictly_select_graspgenx(
    tmp_path, monkeypatch
) -> None:
    allocation = m6.base.Allocation(81, "openeta-pick-place-test", 18765, "run-id")
    monkeypatch.setattr(m6.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        m6.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )
    services = m6._services_for_backend(
        "graspgenx",
        sam3_url="http://sam3/sse",
        anygrasp_url="http://anygrasp/sse",
        anyplace_url="http://anyplace/sse",
        graspgenx_url="http://graspgenx/sse",
    )

    paths = m6.prepare_case(
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
    assert "GraspGenX" in paths.instructions.read_text(encoding="utf-8")
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert receipt["grasp_backend_mode"] == "graspgenx"
    assert receipt["qualification_profile"] == "fast_v3"


def test_pick_place_prepare_smoke_normal_is_explicitly_no_vlm(
    tmp_path, monkeypatch
) -> None:
    allocation = m6.base.Allocation(81, "openeta-smoke-normal", 18765, "run-id")
    monkeypatch.setattr(m6.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        m6.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )

    paths = m6.prepare_case(
        tmp_path,
        tmp_path / "run",
        allocation,
        dict(m6.DEFAULT_SERVICES),
        execution_profile="smoke_normal",
    )

    prompt = paths.instructions.read_text(encoding="utf-8")
    assert "planner_mode=host_macro; execution_profile=smoke_normal" in prompt
    assert "禁止调用 Planner/VLM" in prompt
    assert "主 VLM 必须" not in prompt
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert receipt["execution_profile"] == "smoke_normal"
    assert receipt["planner_mode"] == "host_macro"
    assert receipt["planner_provider_expected"] is False
    assert receipt["qualification_profile"] == "fast_v3"
    assert receipt["acceptance_scope"] == (
        "control_only_no_vlm_smoke_normal_not_agentic_acceptance"
    )


def test_pick_place_acceptance_can_explicitly_roll_back_to_legacy(
    tmp_path, monkeypatch
) -> None:
    allocation = m6.base.Allocation(81, "openeta-legacy-normal", 18765, "run-id")
    monkeypatch.setattr(m6.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        m6.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {
            "schema_version": "openeta.gazebo_environment_receipt.v1",
            "trusted": True,
        },
    )

    paths = m6.prepare_case(
        tmp_path,
        tmp_path / "run",
        allocation,
        dict(m6.DEFAULT_SERVICES),
        qualification_profile="legacy",
    )

    prompt = paths.instructions.read_text(encoding="utf-8")
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert "qualification_profile=legacy" in prompt
    assert receipt["qualification_profile"] == "legacy"


def test_pick_place_acceptance_parser_defaults_to_fast_v3() -> None:
    args = m6._parser().parse_args([])

    assert args.qualification_profile == "fast_v3"


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
                            "host_obligation": {
                                "schema_version": "openeta.fixture_obligation.v1"
                            },
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
    evidence = m6._planner_evidence(events, expected_planner_mode="host_macro")

    assert m6._planner_evidence_errors(
        evidence,
        execution_profile="smoke_normal",
    ) == []
    assert evidence["host_dispatch_count"] == 10
    assert evidence["closed_loop_action_count"] == 0
    assert evidence["total_tokens"] == 0


def test_m6_order_helper_requires_frozen_anyplace_pool_before_grasp() -> None:
    valid = ["observe", "anyplace", "anygrasp", "move_to", "gripper_control"]
    invalid = ["observe", "anygrasp", "move_to", "gripper_control", "anyplace"]

    required = ("observe", "anyplace", "anygrasp", "move_to", "gripper_control")
    assert m6._ordered(valid, required)
    assert not m6._ordered(invalid, required)


def test_m6_canonicalizes_public_grasp_tool_only_with_real_anygrasp_backend() -> None:
    assert m6._name(
        {
            "name": "grasp_pose_estimate",
            "result": {"details": {"backend": "anygrasp_mcp"}},
        }
    ) == "anygrasp"
    assert m6._name({"name": "grasp_pose_estimate"}) == "grasp_pose_estimate"


def test_m6_requires_only_executable_public_grasp_tools() -> None:
    assert "grasp_pose_estimate" in m6.REQUIRED_REAL_M6_TOOLS
    assert "anygrasp" not in m6.REQUIRED_REAL_M6_TOOLS
    assert "grasp_pose_estimate" not in m6._required_tools_for_backend("anygrasp")
    assert "anygrasp" in m6._required_tools_for_backend("anygrasp")
    assert "grasp_pose_estimate" not in m6._required_tools_for_backend("graspgenx")
    assert "graspgenx" in m6._required_tools_for_backend("graspgenx")


def test_m6_health_url_preserves_service_root() -> None:
    assert m6._health_url("http://127.0.0.1:8778/sse") == "http://127.0.0.1:8778/"


def test_m6_runtime_preflight_accepts_the_selected_overlay(
    tmp_path, monkeypatch
) -> None:
    overlay = tmp_path / "external-overlay"
    package_prefix = overlay / m6.GAZEBO_SIM_PACKAGE
    monkeypatch.setenv("OPENETA_GAZEBO_OVERLAY", str(overlay))
    monkeypatch.setattr(m6, "_ros_python_import_error", lambda: "")
    monkeypatch.setattr(m6, "_gazebo_package_prefix", lambda _package: package_prefix)
    monkeypatch.setattr(m6.shutil, "which", lambda name: f"/usr/bin/{name}")

    result = m6.gazebo_runtime_preflight(tmp_path)

    assert result["status"] == "passed"
    assert result["expected_overlay"] == str(overlay.resolve())
    assert result["package_prefix"] == str(package_prefix.resolve())
    assert result["reason_codes"] == []


def test_m6_runtime_preflight_rejects_an_overlay_from_another_checkout(
    tmp_path, monkeypatch
) -> None:
    expected = tmp_path / "extensions/gazebo/ros2_ws/install"
    monkeypatch.delenv("OPENETA_GAZEBO_OVERLAY", raising=False)
    monkeypatch.setattr(m6, "_ros_python_import_error", lambda: "")
    monkeypatch.setattr(
        m6,
        "_gazebo_package_prefix",
        lambda _package: (
            tmp_path / "stale-checkout/install" / m6.GAZEBO_SIM_PACKAGE
        ),
    )
    monkeypatch.setattr(m6.shutil, "which", lambda name: f"/usr/bin/{name}")

    result = m6.gazebo_runtime_preflight(tmp_path)

    assert result["status"] == "blocked"
    assert result["expected_overlay"] == str(expected.resolve())
    assert result["reason_codes"] == [
        "OPENETA_GAZEBO_OVERLAY_PACKAGE_MISMATCH"
    ]


def test_m6_canonical_runner_sources_ros_and_executes_m6() -> None:
    source = M6_RUNNER.read_text(encoding="utf-8")

    assert os.access(M6_RUNNER, os.X_OK)
    assert 'source "${SYSTEM_ROS_SETUP}"' in source
    assert 'source "${OVERLAY_SETUP}"' in source
    assert "import rclpy; from rosgraph_msgs.msg import Clock" in source
    assert "ros2 pkg prefix openeta_rm75_robotiq2f85_sim" in source
    assert 'm6_gazebo_acceptance.py" "$@"' in source

    capability_source = PICK_PLACE_RUNNER.read_text(encoding="utf-8")
    assert os.access(PICK_PLACE_RUNNER, os.X_OK)
    assert 'run_m6_gazebo_acceptance.sh" "$@"' in capability_source


def test_m6_verifier_uses_model_raw_count_for_frozen_goal_requalification() -> None:
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

    assert m6._has_minimum_int_value(call, "model_raw_candidate_count", 96)
    assert not m6._has_minimum_int_value(call, "raw_candidate_count", 96)


def test_m6_candidate_counts_ignore_nested_wave_cardinality() -> None:
    call = {
        "result": {
            "details": {
                "outputs": {
                    "candidate_count": 1,
                    "full_plan_pass_count": 1,
                    "qualification_evidence": {
                        "waves": [{"candidate_count": 2}]
                    },
                    "qualification_waves": [{"candidate_count": 2}],
                }
            }
        }
    }

    outputs = m6._call_outputs(call)

    assert outputs["candidate_count"] == 1
    assert outputs["full_plan_pass_count"] == 1


def test_m6_rejection_scenario_is_explicit_acceptance_only_fixture(
    tmp_path, monkeypatch
) -> None:
    allocation = m6.base.Allocation(81, "partition", 18765, "run-id")
    monkeypatch.setattr(m6.base, "_process_snapshot", lambda: [])
    monkeypatch.setattr(
        m6.base,
        "environment_receipt",
        lambda *_args, **_kwargs: {"trusted": True},
    )

    scenario = "reject-first"
    paths = m6.prepare_case(
        tmp_path,
        tmp_path / scenario,
        allocation,
        dict(m6.DEFAULT_SERVICES),
        scenario=scenario,
    )
    prompt = paths.instructions.read_text(encoding="utf-8")
    assert scenario in prompt
    assert "execution_started=false" in prompt
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert receipt["acceptance_scenario"] == scenario
    assert receipt["grasp_backend_mode"] == m6.DEFAULT_GRASP_BACKEND
    unhashed = dict(receipt)
    supplied_hash = unhashed.pop("receipt_sha256")
    assert supplied_hash == m6.base.hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m6_failed_fingerprint_check_ignores_receipt_mirrors() -> None:
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

    assert not m6._repeated_failed_motion_fingerprints([event])
    assert m6._repeated_failed_motion_fingerprints([event, event]) == {
        "fingerprint-a"
    }


def test_m6_qualification_blocks_resolve_relative_to_case_root(tmp_path) -> None:
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

    assert m6._qualification_blocks(call, artifact_root=tmp_path) == [proof]


def test_m6_qualification_blocks_include_frozen_pair_proof(tmp_path) -> None:
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

    assert m6._qualification_blocks(call, artifact_root=tmp_path) == [proof]


def test_m6_qualification_blocks_include_internal_frontier_proofs(tmp_path) -> None:
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

    assert m6._qualification_blocks(call, artifact_root=tmp_path) == [
        proofs[1],
        proofs[0],
    ]


def test_m6_accepts_complete_v3_grasp_pool_with_two_l5_branches(tmp_path) -> None:
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

    assert m6._has_v3_grasp_diversity_evidence(call, artifact_root=tmp_path)
    assert m6._has_bounded_grasp_l5_evidence(call, artifact_root=tmp_path)

    proof["selected_candidate_ids"] = ["grasp_000"]
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    assert not m6._has_v3_grasp_diversity_evidence(call, artifact_root=tmp_path)
    assert not m6._has_bounded_grasp_l5_evidence(call, artifact_root=tmp_path)


def test_m6_rejects_obsolete_four_branch_lookahead(tmp_path) -> None:
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
            {"candidate_id": f"grasp_{index:03d}", "verdict": "PASS"}
            for index in range(4)
        ],
        "results": results,
    }
    artifact.write_text(json.dumps(proof), encoding="utf-8")
    call = {
        "diversity_selected_count": 65,
        "qualification_artifact": {"kind": "json", "path": artifact.name},
    }

    assert not m6._has_v3_grasp_diversity_evidence(call, artifact_root=tmp_path)
    assert not m6._has_bounded_grasp_l5_evidence(call, artifact_root=tmp_path)


def test_m6_agentic_planner_evidence_counts_model_and_bounded_host_actions() -> None:
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

    evidence = m6._agentic_planner_evidence(events)

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
    monkeypatch.setattr(m6.base.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(m6.base, "_wait_for_scripted_tui_episode", lambda *args, **kwargs: "completed")
    monkeypatch.setattr(m6.base, "_terminate_scripted_tui_process", lambda value: terminated.append(value))

    assert m6.base._run_scripted_tui("tui", paths, {}) == 1
    assert terminated == [process]
    evidence = json.loads((tmp_path / "scripted-tui-driver.json").read_text())
    assert evidence["reason_code"] == "TUI_DID_NOT_EXIT_AFTER_QUIT"
