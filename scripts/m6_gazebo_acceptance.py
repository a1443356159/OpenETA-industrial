#!/usr/bin/env python3
"""Isolated real-service Gazebo acceptance for constraint-correct placement."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import urllib.request

from scripts import tui_gazebo_acceptance as base
from agent.runtime.calibration_registry import RM75_ROBOTIQ_GRASP_CALIBRATION_PROFILE
from tools.candidate_config import QUALIFICATION_PROFILES


SCHEMA_VERSION = "openeta.gazebo_pick_place_acceptance.v1"
SUITE = "pick-place"
# Compatibility alias for callers that imported the historical module.
MILESTONE = SUITE
MODE = base.SCRIPTED_TUI
ENV_ID = "openeta/gazebo_rm75_robotiq2f85_pickplace-v0"
GRASP_BACKENDS = ("anygrasp", "graspgenx")
DEFAULT_GRASP_BACKEND = "anygrasp"
DEFAULT_SERVICE_URLS = {
    "openeta-sam3": "http://127.0.0.1:8773/sse",
    "openeta-anygrasp": "http://127.0.0.1:8774/sse",
    "openeta-anyplace": "http://127.0.0.1:8775/sse",
    "openeta-graspgenx": "http://127.0.0.1:8778/sse",
}
GRASP_SERVICE_NAMES = {
    "anygrasp": "openeta-anygrasp",
    "graspgenx": "openeta-graspgenx",
}
DEFAULT_SERVICES = {
    name: DEFAULT_SERVICE_URLS[name]
    for name in ("openeta-sam3", "openeta-anygrasp", "openeta-anyplace")
}
REQUIRED_REAL_PICK_PLACE_TOOLS = (
    "create_simulator_env",
    "observe",
    "sam3",
    "grasp_pose_estimate",
    "gripper_control",
    "anyplace",
    "close_simulator_env",
)
# Compatibility alias for tests and integrations importing the historical name.
REQUIRED_REAL_M6_TOOLS = REQUIRED_REAL_PICK_PLACE_TOOLS
SCENARIOS = ("normal", "reject-first")
EXECUTION_PROFILES = ("agentic_normal", "smoke_normal")
DEFAULT_EXECUTION_PROFILE = "agentic_normal"
# Acceptance rolls the already-shadowable v3 funnel forward explicitly while
# the general runtime keeps its conservative `legacy` default and instant
# environment-variable rollback.
DEFAULT_QUALIFICATION_PROFILE = "fast_v3"
GAZEBO_SIM_PACKAGE = "openeta_rm75_robotiq2f85_sim"

EXECUTION_PROFILE_PLANNER_MODES = {
    "agentic_normal": "agentic_closed_loop",
    "smoke_normal": "host_macro",
}
EXECUTION_PROFILE_SCOPES = {
    "agentic_normal": "agentic_normal_vlm_acceptance",
    "smoke_normal": "control_only_no_vlm_smoke_normal_not_agentic_acceptance",
}


AGENTIC_CONTROL = """
你是 OpenETA 闭环 Planner。每回合只选择下一个合法工具或结束；宿主负责精确参数、
坐标变换、候选漏斗和安全证明，不替你做语义决策。SAM3 多候选时检查候选图；
不得固定 detection id、使用 Oracle/假候选或指定具体抓取后端；不得调用 python_exec。
""".strip()

SMOKE_CONTROL = """
本 smoke_normal 只验证无 VLM 控制链：宿主逐步派发确定性 obligation，禁止调用 Planner/VLM。
SAM3 必须返回唯一合法候选；有语义歧义时立即阻断。
""".strip()

TASK_INSTRUCTIONS = """
在隔离 Gazebo RM75/Robotiq 环境完成 normal 拾放。创建后显式 observe 一次（create 的
initial observation 不计）。用 `red rectangular block` 选目标，用
`green placement zone marker` 选放置区；mask 必须完整且不粘连。红色方块是
target_object，物体与放置区使用各自的 RGB-D 证据。

只运行一次 AnyPlace 和一次 unified grasp_pose_estimate，冻结全部模型候选。模型抓取
位姿是 exact EEF contact，放置目标经实测 T_eef_object 转为 exact release；除标定/
TCP 表示变换外不得修改。禁止 pregrasp、hover、lift、approach/descend offset、镜像或
人工位姿变体。MoveIt 自当前状态直接规划到 contact/release。

闭爪后要有双垫 target contact 与 attach ACK；transport 要有 attached ACK 且相对漂移
<=10 mm；到 release 后原地开爪。成功需 detach ACK、稳定落区 PASS 与唯一一次
close_simulator_env。

候选或物理执行失败时，先用已通过备用，再以 `model_inference=false` 从
frozen_frontier 继续；不重跑 SAM3/抓取模型/AnyPlace。只有场景确已改变或冻结池耗尽，
才重新观察或推理。基础设施错误不得记为不可达。
""".strip()


SCENARIO_INSTRUCTIONS = {
    "normal": "执行正常放置路径；不得主动制造规划失败。",
    "reject-first": (
        "验收配置会让首个 placement candidate 的资格规划真实返回无轨迹且 "
        "execution_started=false。保留该回执，宿主不得把该候选存入资格队列；宿主从其余 "
        "PASS 候选的稳定队首开始并完成放置。"
    ),
}


def _validated_grasp_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in GRASP_BACKENDS:
        choices = ", ".join(GRASP_BACKENDS)
        raise ValueError(f"grasp backend must be one of: {choices}")
    return backend


def _validated_execution_profile(value: str) -> str:
    profile = str(value).strip().lower()
    if profile not in EXECUTION_PROFILES:
        choices = ", ".join(EXECUTION_PROFILES)
        raise ValueError(f"execution profile must be one of: {choices}")
    return profile


def _validated_qualification_profile(value: str) -> str:
    profile = str(value).strip().lower()
    if profile not in QUALIFICATION_PROFILES:
        choices = ", ".join(QUALIFICATION_PROFILES)
        raise ValueError(f"qualification profile must be one of: {choices}")
    return profile


def _instructions_for_backend(
    grasp_backend: str,
    *,
    execution_profile: str = DEFAULT_EXECUTION_PROFILE,
    qualification_profile: str = DEFAULT_QUALIFICATION_PROFILE,
) -> str:
    backend = _validated_grasp_backend(grasp_backend)
    profile = _validated_execution_profile(execution_profile)
    funnel_profile = _validated_qualification_profile(qualification_profile)
    planner_mode = EXECUTION_PROFILE_PLANNER_MODES[profile]
    metadata = (
        f"[automation=scripted_tui; planner_mode={planner_mode}; "
        f"execution_profile={profile}; qualification_profile={funnel_profile}; "
        f"environment_id={ENV_ID}; environment_task=normal_pick_and_place; "
        "grasp_target=red_rectangular_block; "
        "placement_object=red_rectangular_block; "
        "placement_region=green_placement_zone_marker]"
    )
    control = AGENTIC_CONTROL if profile == "agentic_normal" else SMOKE_CONTROL
    provider = (
        "宿主已配置 GraspGenX（gripper_name=robotiq_2f_85）。"
        if backend == "graspgenx"
        else "宿主已配置 AnyGrasp。"
    )
    return f"{metadata}\n{control}\n\n{provider}\n{TASK_INSTRUCTIONS}\n"


def _required_tools_for_backend(grasp_backend: str) -> tuple[str, ...]:
    backend = _validated_grasp_backend(grasp_backend)
    return tuple(
        backend if name == "grasp_pose_estimate" else name
        for name in REQUIRED_REAL_PICK_PLACE_TOOLS
    )


def _services_for_backend(
    grasp_backend: str,
    *,
    sam3_url: str,
    anygrasp_url: str,
    anyplace_url: str,
    graspgenx_url: str,
) -> dict[str, str]:
    backend = _validated_grasp_backend(grasp_backend)
    urls = {
        "openeta-sam3": sam3_url,
        "openeta-anygrasp": anygrasp_url,
        "openeta-anyplace": anyplace_url,
        "openeta-graspgenx": graspgenx_url,
    }
    return {
        name: urls[name]
        for name in (
            "openeta-sam3",
            GRASP_SERVICE_NAMES[backend],
            "openeta-anyplace",
        )
    }


def _health_url(sse_url: str) -> str:
    return sse_url.removesuffix("/sse").rstrip("/") + "/"


def service_preflight(services: Mapping[str, str]) -> dict[str, Any]:
    expected = {
        "openeta-sam3": "sam3",
        "openeta-anygrasp": "anygrasp",
        "openeta-anyplace": "anyplace",
        "openeta-graspgenx": "openeta-graspgenx",
    }
    rows: dict[str, Any] = {}
    for name, url in services.items():
        try:
            with urllib.request.urlopen(_health_url(url), timeout=5.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            ok = (
                isinstance(payload, Mapping)
                and payload.get("ok") is True
                and payload.get("server") == expected[name]
                and (
                    name == "openeta-sam3"
                    or (
                        payload.get("raw_pool_size") == (96 if name == "openeta-anyplace" else 200)
                        and isinstance(payload.get("returned_candidate_count"), int)
                        and payload.get("returned_candidate_count") >= 0
                    )
                )
            )
            rows[name] = {
                "status": "passed" if ok else "failed",
                "url": url,
                "server": payload.get("server") if isinstance(payload, Mapping) else None,
                "model_loaded": payload.get("model_loaded")
                if isinstance(payload, Mapping)
                else None,
                "tools": payload.get("tools") if isinstance(payload, Mapping) else None,
                "returned_candidate_count": (
                    payload.get("returned_candidate_count")
                    if isinstance(payload, Mapping)
                    else None
                ),
                "raw_pool_size": (
                    payload.get("raw_pool_size") if isinstance(payload, Mapping) else None
                ),
            }
        except Exception as exc:  # noqa: BLE001 - bounded preflight evidence.
            rows[name] = {
                "status": "blocked",
                "url": url,
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            }
    return {
        "schema_version": "openeta.mcp_service_registration.v1",
        "status": "passed"
        if rows.keys() == services.keys()
        and all(row.get("status") == "passed" for row in rows.values())
        else "blocked",
        "services": rows,
    }


def _ros_python_import_error() -> str:
    try:
        import rclpy  # noqa: F401 - explicit ABI/import preflight.
        from rosgraph_msgs.msg import Clock  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - bounded preflight diagnostic.
        return f"{type(exc).__name__}: {exc}"[:500]
    return ""


def _gazebo_package_prefix(package: str) -> Path:
    from ament_index_python.packages import get_package_prefix

    return Path(get_package_prefix(package)).resolve()


def gazebo_runtime_preflight(repo: Path) -> dict[str, Any]:
    """Fail before allocation when the ROS underlay/overlay was not loaded.

    Shell setup files cannot safely be applied to an already-running Python
    process. The canonical pick-place wrapper sources them before exec; this check
    makes an accidental direct invocation fail immediately and precisely.
    """

    expected_overlay = Path(
        os.environ.get("OPENETA_GAZEBO_OVERLAY") or repo / "extensions/gazebo/ros2_ws/install"
    ).resolve()
    errors: list[str] = []
    import_error = _ros_python_import_error()
    if import_error:
        errors.append("OPENETA_ROS_PYTHON_ABI_UNAVAILABLE")
    actual_prefix: Path | None = None
    package_error = ""
    try:
        actual_prefix = _gazebo_package_prefix(GAZEBO_SIM_PACKAGE)
    except Exception as exc:  # noqa: BLE001 - bounded preflight diagnostic.
        package_error = f"{type(exc).__name__}: {exc}"[:500]
        errors.append("OPENETA_GAZEBO_OVERLAY_PACKAGE_UNAVAILABLE")
    else:
        if actual_prefix != expected_overlay and expected_overlay not in actual_prefix.parents:
            errors.append("OPENETA_GAZEBO_OVERLAY_PACKAGE_MISMATCH")
    missing_commands = [name for name in ("ros2", "gz") if shutil.which(name) is None]
    if missing_commands:
        errors.append("OPENETA_GAZEBO_COMMAND_UNAVAILABLE")
    return {
        "schema_version": "openeta.gazebo_runtime_preflight.v1",
        "status": "passed" if not errors else "blocked",
        "reason_codes": errors,
        "canonical_runner": str((repo / "scripts/run_pick_place_acceptance.sh").resolve()),
        "expected_overlay": str(expected_overlay),
        "declared_overlay": os.environ.get("OPENETA_GAZEBO_OVERLAY", ""),
        "package": GAZEBO_SIM_PACKAGE,
        "package_prefix": str(actual_prefix) if actual_prefix is not None else None,
        "package_error": package_error or None,
        "python_import_error": import_error or None,
        "missing_commands": missing_commands,
    }


def prepare_case(
    repo: Path,
    run_root: Path,
    allocation: base.Allocation,
    services: Mapping[str, str],
    scenario: str = "normal",
    grasp_backend: str = DEFAULT_GRASP_BACKEND,
    execution_profile: str = DEFAULT_EXECUTION_PROFILE,
    qualification_profile: str = DEFAULT_QUALIFICATION_PROFILE,
) -> base.CasePaths:
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported pick-place acceptance scenario: {scenario}")
    backend = _validated_grasp_backend(grasp_backend)
    profile = _validated_execution_profile(execution_profile)
    funnel_profile = _validated_qualification_profile(qualification_profile)
    if profile == "smoke_normal" and scenario != "normal":
        raise ValueError("smoke_normal execution profile requires scenario=normal")
    configured_grasp_services = set(services).intersection(GRASP_SERVICE_NAMES.values())
    required_grasp_service = GRASP_SERVICE_NAMES[backend]
    if configured_grasp_services != {required_grasp_service}:
        raise ValueError(
            f"strict acceptance requires exactly one grasp service: {required_grasp_service}"
        )
    paths = base.case_paths(run_root, MILESTONE, MODE)
    paths.root.mkdir(parents=True, exist_ok=False)
    base._json_dump(
        paths.mcp_config,
        {
            "mcpServers": {
                "openeta-sim": {"url": f"http://127.0.0.1:{allocation.port}/sse"},
                **{name: {"url": url} for name, url in services.items()},
            }
        },
    )
    paths.instructions.write_text(
        _instructions_for_backend(
            backend,
            execution_profile=profile,
            qualification_profile=funnel_profile,
        )
        + "\n验收场景："
        + scenario
        + "。"
        + SCENARIO_INSTRUCTIONS[scenario]
        + "\n",
        encoding="utf-8",
    )
    receipt = base.environment_receipt(
        repo,
        allocation,
        case_name=f"{MILESTONE}-{MODE}",
        before=base._process_snapshot(),
    )
    receipt["acceptance_scenario"] = scenario
    receipt["grasp_backend_mode"] = backend
    receipt["execution_profile"] = profile
    receipt["qualification_profile"] = funnel_profile
    receipt["planner_mode"] = EXECUTION_PROFILE_PLANNER_MODES[profile]
    receipt["acceptance_scope"] = EXECUTION_PROFILE_SCOPES[profile]
    receipt["planner_provider_expected"] = profile == "agentic_normal"
    base._json_dump(paths.receipt, base.seal_environment_receipt(receipt))
    return paths


def _name(call: Mapping[str, Any]) -> str:
    name = str(call.get("name") or call.get("tool_name") or "")
    if name == "grasp_pose_estimate":
        if (
            base._contains(call, "backend", "anygrasp_mcp")
            or base._contains(call, "source_backend", "anygrasp")
            or base._contains(call, "selected_backend", "anygrasp")
        ):
            return "anygrasp"
        if (
            base._contains(call, "backend", "graspgenx_mcp")
            or base._contains(call, "source_backend", "graspgenx")
            or base._contains(call, "selected_backend", "graspgenx")
        ):
            return "graspgenx"
    return name


def _has_minimum_int_value(payload: object, key: str, minimum: int) -> bool:
    return any(
        value >= minimum
        for value in base._values(payload, key)
        if isinstance(value, int) and not isinstance(value, bool)
    )


def _call_outputs(call: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the public tool-output object without descending into evidence."""

    result = call.get("result")
    result = result if isinstance(result, Mapping) else call
    details = result.get("details")
    if not isinstance(details, Mapping):
        return {}
    outputs = details.get("outputs")
    return outputs if isinstance(outputs, Mapping) else {}


def _parameters(call: Mapping[str, Any]) -> Mapping[str, Any]:
    value = call.get("parameters")
    return value if isinstance(value, Mapping) else {}


def _ordered(names: Sequence[str], required: Sequence[str]) -> bool:
    cursor = iter(names)
    return all(any(name == wanted for name in cursor) for wanted in required)


def _repeated_failed_motion_fingerprints(
    events: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Find fingerprints repeated across failed actions, ignoring receipt mirrors."""

    failed: list[str] = []
    for event in events:
        if not base._contains(event, "execution_started", False):
            continue
        failed.extend(
            sorted({str(value) for value in base._values(event, "request_fingerprint") if value})
        )
    counts = {fingerprint: failed.count(fingerprint) for fingerprint in set(failed)}
    return {fingerprint for fingerprint, count in counts.items() if count > 1}


def _planner_evidence(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_planner_mode: str,
) -> dict[str, Any]:
    """Summarize decision provenance from durable action commands only."""

    action_count = 0
    closed_loop_actions = 0
    closed_loop_tool_calls = 0
    isolated_selection_actions = 0
    no_vlm_block_actions = 0
    wrong_planner_mode_actions = 0
    host_dispatches: list[dict[str, str]] = []
    missing_or_unknown_actions = 0
    providers: set[str] = set()
    models: set[str] = set()
    observed_planner_modes: set[str] = set()
    for event in events:
        if event.get("event_type") != "action":
            continue
        action_count += 1
        payload = event.get("payload")
        command = payload.get("command") if isinstance(payload, Mapping) else None
        if not isinstance(command, Mapping):
            continue
        metadata = command.get("metadata")
        planner_metadata = (
            metadata.get("planner_metadata") if isinstance(metadata, Mapping) else None
        )
        if not isinstance(planner_metadata, Mapping):
            missing_or_unknown_actions += 1
            continue
        execution_model = str(planner_metadata.get("execution_model") or "")
        planner_mode = str(planner_metadata.get("planner_mode") or "")
        if planner_mode:
            observed_planner_modes.add(planner_mode)
        if planner_mode != expected_planner_mode:
            wrong_planner_mode_actions += 1
        request = command.get("request")
        request = request if isinstance(request, Mapping) else {}
        if execution_model == "closed_loop_tool_calling":
            closed_loop_actions += 1
            if request.get("kind") == "tool_call":
                closed_loop_tool_calls += 1
            provider = str(planner_metadata.get("backend_provider") or "")
            model = str(planner_metadata.get("backend_model") or "")
            if provider:
                providers.add(provider)
            if model:
                models.add(model)
        elif execution_model == "isolated_semantic_selection":
            isolated_selection_actions += 1
            provider = str(planner_metadata.get("backend_provider") or "")
            model = str(planner_metadata.get("backend_model") or "")
            if provider:
                providers.add(provider)
            if model:
                models.add(model)
        elif execution_model == "host_obligation_dispatch":
            obligation = planner_metadata.get("host_obligation")
            obligation = obligation if isinstance(obligation, Mapping) else {}
            host_dispatches.append(
                {
                    "schema_version": str(obligation.get("schema_version") or ""),
                    "tool": str(request.get("name") or ""),
                }
            )
        elif execution_model == "host_macro_no_vlm_block":
            no_vlm_block_actions += 1
        else:
            missing_or_unknown_actions += 1

    episode_total_tokens = 0
    token_usage_sources: dict[str, int] = {}
    for event in events:
        if event.get("event_type") != "episode_result":
            continue
        payload = event.get("payload")
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        usage = metadata.get("usage") if isinstance(metadata, Mapping) else None
        if not isinstance(usage, Mapping):
            continue
        total = usage.get("total_tokens")
        if isinstance(total, int) and not isinstance(total, bool):
            episode_total_tokens = max(episode_total_tokens, total)
        sources = usage.get("token_usage_sources")
        if isinstance(sources, Mapping):
            token_usage_sources = {
                str(key): int(value)
                for key, value in sources.items()
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            }
    return {
        "planner_mode": expected_planner_mode,
        "observed_planner_modes": sorted(observed_planner_modes),
        "action_count": action_count,
        "closed_loop_action_count": closed_loop_actions,
        "closed_loop_tool_call_count": closed_loop_tool_calls,
        "isolated_selection_action_count": isolated_selection_actions,
        "no_vlm_block_action_count": no_vlm_block_actions,
        "host_dispatches": host_dispatches,
        "host_dispatch_count": len(host_dispatches),
        "missing_or_unknown_action_count": missing_or_unknown_actions,
        "wrong_planner_mode_action_count": wrong_planner_mode_actions,
        "providers": sorted(providers),
        "models": sorted(models),
        "total_tokens": episode_total_tokens,
        "token_usage_sources": token_usage_sources,
    }


def _agentic_planner_evidence(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compatibility wrapper for the formal agentic-normal verifier."""

    return _planner_evidence(
        events,
        expected_planner_mode=EXECUTION_PROFILE_PLANNER_MODES["agentic_normal"],
    )


def _planner_evidence_errors(
    evidence: Mapping[str, Any],
    *,
    execution_profile: str,
) -> list[str]:
    """Validate that decision provenance matches the requested run profile."""

    profile = _validated_execution_profile(execution_profile)
    errors: list[str] = []
    if evidence.get("missing_or_unknown_action_count"):
        errors.append("one or more actions lack planner decision provenance")
    if evidence.get("wrong_planner_mode_action_count"):
        errors.append(
            "one or more actions were not bound to "
            + EXECUTION_PROFILE_PLANNER_MODES[profile]
            + " mode"
        )
    if profile == "smoke_normal":
        if evidence.get("closed_loop_action_count"):
            errors.append("smoke_normal unexpectedly invoked the main VLM planner")
        if evidence.get("isolated_selection_action_count"):
            errors.append("smoke_normal unexpectedly invoked VLM semantic selection")
        if evidence.get("no_vlm_block_action_count"):
            errors.append("smoke_normal stopped at an obligation not covered by the host macro")
        if evidence.get("total_tokens"):
            errors.append("smoke_normal recorded non-zero VLM token usage")
        if evidence.get("providers") or evidence.get("models"):
            errors.append("smoke_normal recorded planner provider/model invocation")
        if int(evidence.get("host_dispatch_count") or 0) < 10:
            errors.append("smoke_normal has too few host-dispatched control decisions")
        return errors

    allowed_host_dispatch_schemas = {
        "openeta.fresh_observation_obligation.v1",
        "openeta.motion_reconciliation.v1",
    }
    disallowed_host_dispatches = [
        dispatch
        for dispatch in evidence.get("host_dispatches", [])
        if isinstance(dispatch, Mapping)
        and dispatch.get("schema_version") not in allowed_host_dispatch_schemas
    ]
    if disallowed_host_dispatches:
        errors.append("agentic normal used host dispatch for planner-owned task progression")
    if int(evidence.get("closed_loop_tool_call_count") or 0) < 10:
        errors.append("too few normal tool decisions reached the main VLM planner")
    if int(evidence.get("total_tokens") or 0) <= 0:
        errors.append("agentic normal recorded zero main-VLM token usage")
    if not evidence.get("providers") or not evidence.get("models"):
        errors.append("agentic normal lacks concrete planner provider/model evidence")
    return errors


def _qualification_blocks(
    call: Mapping[str, Any], *, artifact_root: Path
) -> list[Mapping[str, Any]]:
    """Read exact host proof artifacts without placing raw proof in VLM output."""

    blocks = [
        value
        for value in base._values(call, "qualification_evidence")
        if isinstance(value, Mapping) and isinstance(value.get("results"), list)
    ]
    seen_paths: set[str] = set()
    for field in (
        "qualification_artifact",
        "frozen_pair_qualification_artifact",
        "frozen_pair_qualification_artifacts",
        "frozen_grasp_frontier_qualification_artifacts",
    ):
        for value in base._values(call, field):
            artifacts = value if isinstance(value, list) else [value]
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    continue
                path_value = artifact.get("path")
                if not isinstance(path_value, str) or not path_value.endswith(".json"):
                    continue
                if path_value in seen_paths:
                    continue
                seen_paths.add(path_value)
                try:
                    path = Path(path_value)
                    if not path.is_absolute():
                        path = artifact_root / path
                    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                if isinstance(payload, Mapping) and isinstance(payload.get("results"), list):
                    blocks.append(payload)
    return blocks


def _has_v3_grasp_diversity_evidence(call: Mapping[str, Any], *, artifact_root: Path) -> bool:
    """Accept the deterministic two-branch best-first grasp result."""

    reported_counts = {
        int(value)
        for value in base._values(call, "diversity_selected_count")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    }
    for block in _qualification_blocks(call, artifact_root=artifact_root):
        if (
            block.get("schema_version") != "openeta.moveit_candidate_funnel.v3"
            or block.get("artifact_schema_version") != "openeta.moveit_candidate_qualification.v3"
            or block.get("purpose") != "grasp"
            or block.get("stop_reason")
            not in {
                "complete_l5_pass_found",
                "complete_l5_pass_found_minimum_lookahead",
                "complete_l5_pass_found_joint_space_fallback",
                "complete_l5_pass_found_partial_lookahead",
            }
        ):
            continue
        results = block.get("results")
        metrics = block.get("metrics")
        selected_ids = block.get("selected_candidate_ids")
        l5_attempts = block.get("l5_attempts")
        if not (
            isinstance(results, list)
            and isinstance(metrics, Mapping)
            and isinstance(selected_ids, list)
            and isinstance(l5_attempts, list)
        ):
            continue
        generated_count = metrics.get("generated_count")
        l5_pass_count = metrics.get("l5_pass_count")
        if not (
            isinstance(generated_count, int)
            and not isinstance(generated_count, bool)
            and generated_count >= 10
            and generated_count == len(results)
            and generated_count in reported_counts
            and isinstance(l5_pass_count, int)
            and not isinstance(l5_pass_count, bool)
            and l5_pass_count >= 2
            and len(selected_ids) == 2
            and len(set(selected_ids)) == len(selected_ids)
        ):
            continue
        result_by_id = {
            row.get("candidate_id"): row
            for row in results
            if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str)
        }
        selected_rows = [result_by_id.get(candidate_id) for candidate_id in selected_ids]
        if not all(
            isinstance(row, Mapping)
            and row.get("verdict") == "PASS"
            and row.get("endpoint_pass") is True
            for row in selected_rows
        ):
            continue
        l5_pass_ids = {
            row.get("candidate_id")
            for row in l5_attempts
            if isinstance(row, Mapping) and row.get("verdict") == "PASS"
        }
        if not set(selected_ids).issubset(l5_pass_ids):
            continue
        qualified_clusters = {
            row.get("se3_cluster_id")
            for row in results
            if isinstance(row, Mapping)
            and row.get("endpoint_pass") is True
            and isinstance(row.get("se3_cluster_id"), str)
            and row.get("se3_cluster_id")
        }
        selected_clusters = {
            row.get("se3_cluster_id")
            for row in selected_rows
            if isinstance(row, Mapping)
            and isinstance(row.get("se3_cluster_id"), str)
            and row.get("se3_cluster_id")
        }
        if not qualified_clusters or len(selected_clusters) < min(2, len(qualified_clusters)):
            continue
        return True
    return False


def _has_bounded_grasp_l5_evidence(call: Mapping[str, Any], *, artifact_root: Path) -> bool:
    """Accept the legacy submit cap or the v3 two-branch proof."""

    legacy_bound = any(
        1 <= value <= 2
        for value in base._values(call, "full_plan_submitted_count")
        if isinstance(value, int) and not isinstance(value, bool)
    )
    return legacy_bound or _has_v3_grasp_diversity_evidence(call, artifact_root=artifact_root)


def verify_case(
    paths: base.CasePaths,
    *,
    scenario: str = "normal",
    grasp_backend: str = DEFAULT_GRASP_BACKEND,
    execution_profile: str = DEFAULT_EXECUTION_PROFILE,
    qualification_profile: str = DEFAULT_QUALIFICATION_PROFILE,
) -> dict[str, Any]:
    backend = _validated_grasp_backend(grasp_backend)
    profile = _validated_execution_profile(execution_profile)
    funnel_profile = _validated_qualification_profile(qualification_profile)
    backend_label = "GraspGenX" if backend == "graspgenx" else "AnyGrasp"
    errors: list[str] = []
    try:
        events, trace_paths = base._load_trace_events(paths.trace_root)
        calls = base._tool_calls(events)
        errors.extend(base._base_errors(paths, events))
        planner_evidence = _planner_evidence(
            events,
            expected_planner_mode=EXECUTION_PROFILE_PLANNER_MODES[profile],
        )
        errors.extend(
            _planner_evidence_errors(
                planner_evidence,
                execution_profile=profile,
            )
        )
        payloads, mcp_errors = base._mcp_response_payloads(calls, paths)
        errors.extend(mcp_errors)
        names = [_name(call) for call in calls]
        for required in _required_tools_for_backend(backend):
            if required not in names:
                errors.append(f"required real pick-place tool call missing: {required}")
        if names.count("sam3") < 2:
            errors.append("target-object and placement-region SAM3 calls are required")
        grasp_indices = [index for index, name in enumerate(names) if name == backend]
        sam_indices = [index for index, name in enumerate(names) if name == "sam3"]
        if grasp_indices and sam_indices and max(sam_indices) > min(grasp_indices):
            errors.append("SAM3 was rerun after the cached grasp/placement funnel started")
        if not _ordered(
            names,
            (
                "observe",
                "anyplace",
                backend,
                "move_to",
                "gripper_control",
                "anyplace",
                "move_to",
                "gripper_control",
            ),
        ):
            errors.append("model-contact, attach, frozen-goal release order is invalid")
        if any(base._contains(event, "perception_source", "gazebo_oracle") for event in events):
            errors.append("Oracle perception is forbidden")
        if any(base._contains(event, "fake_grasp_candidate") for event in events):
            errors.append("fake candidate evidence is forbidden")
        if any(
            base._contains(event, "plan_only", True)
            and not any(
                base._contains(event, "schema_version", schema)
                for schema in (
                    "openeta.moveit_candidate_funnel.v2",
                    "openeta.moveit_candidate_funnel.v3",
                )
            )
            for event in events
        ):
            errors.append("agent-visible motion preview evidence is forbidden")
        create = next((call for call in calls if _name(call) == "create_simulator_env"), {})
        if not base._contains(create, "env_id", ENV_ID):
            errors.append("pick-place Gazebo environment identity missing")
        if not any(
            base._contains(event, "schema_version", "openeta.host_candidate_compilation.v1")
            and base._contains(event, "purpose", "placement")
            and base._contains(event, "execution_started", False)
            for event in events
        ):
            errors.append("host placement candidate compilation evidence missing")
        if not any(
            base._contains(event, "schema_version", "openeta.host_candidate_compilation.v1")
            and base._contains(event, "purpose", "grasp")
            and base._contains(event, "execution_started", False)
            for event in events
        ):
            errors.append("host grasp candidate compilation evidence missing")
        grasp_calls = [call for call in calls if _name(call) == backend]
        provider_inference_calls = [
            call for call in grasp_calls if _parameters(call).get("mode") != "frozen_frontier"
        ]
        if len(provider_inference_calls) != 1:
            errors.append(f"normal requires exactly one {backend_label} model inference")
        for call in grasp_calls:
            if _parameters(call).get("mode") != "frozen_frontier":
                continue
            if _parameters(call).get("model_inference") is not False or not base._contains(
                call, "model_inference_invoked", False
            ):
                errors.append(
                    "frozen grasp frontier expansion did not prove model-inference bypass"
                )
        raw_grasp_counts = [
            int(value)
            for grasp_call in grasp_calls
            for value in base._values(grasp_call, "raw_candidate_count")
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        if not raw_grasp_counts or max(raw_grasp_counts) < 10:
            errors.append(f"{backend_label} raw candidate count evidence is missing")
        has_legacy_diversity_pool = any(
            1 <= value <= 64
            for grasp_call in grasp_calls
            for value in base._values(grasp_call, "diversity_selected_count")
            if isinstance(value, int) and not isinstance(value, bool)
        )
        has_v3_diversity = any(
            _has_v3_grasp_diversity_evidence(grasp_call, artifact_root=paths.root)
            for grasp_call in grasp_calls
        )
        if not has_legacy_diversity_pool and not has_v3_diversity:
            errors.append(f"{backend_label} diversity pool evidence is missing")
        if not any(
            _has_bounded_grasp_l5_evidence(grasp_call, artifact_root=paths.root)
            for grasp_call in grasp_calls
        ):
            errors.append(f"{backend_label} L5 diversity evidence is missing")
        anyplace_calls = [call for call in calls if _name(call) == "anyplace"]
        anyplace = anyplace_calls[-1] if anyplace_calls else {}
        first_anyplace = anyplace_calls[0] if anyplace_calls else {}
        anyplace_outputs = _call_outputs(anyplace)
        if len(anyplace_calls) < 2:
            errors.append(
                "normal flow requires one model AnyPlace call and at least one "
                "frozen-pool requalification"
            )
        qualification_outputs = [
            _call_outputs(call) for call in [*grasp_calls, *anyplace_calls[1:]]
        ]
        observed_profiles = {
            str(outputs.get("qualification_profile") or "")
            for outputs in qualification_outputs
            if str(outputs.get("qualification_profile") or "")
        }
        if observed_profiles != {funnel_profile}:
            errors.append(
                "qualification profile evidence mismatch: expected "
                f"{funnel_profile}, observed {sorted(observed_profiles)}"
            )
        for requalification_index, requalification in enumerate(anyplace_calls[1:], start=1):
            if not base._contains(requalification, "anyplace_model_inference_invoked", False):
                errors.append(
                    "post-attach AnyPlace requalification "
                    f"{requalification_index} did not prove frozen-pool inference bypass"
                )
        candidate_ids = {
            str(value)
            for value in base._values(anyplace, "id")
            if str(value).startswith("placement_")
        }
        if not _has_minimum_int_value(anyplace, "model_raw_candidate_count", 96):
            errors.append("AnyPlace model raw pool evidence is missing")
        candidate_count = anyplace_outputs.get("candidate_count")
        full_plan_pass_count = anyplace_outputs.get("full_plan_pass_count")
        candidate_count_valid = isinstance(candidate_count, int) and not isinstance(
            candidate_count, bool
        )
        full_plan_pass_count_valid = isinstance(full_plan_pass_count, int) and not isinstance(
            full_plan_pass_count, bool
        )
        if not candidate_count_valid or candidate_count < 1:
            errors.append("final AnyPlace result stored no MoveIt PASS candidate")
        if (
            not full_plan_pass_count_valid
            or not candidate_count_valid
            or full_plan_pass_count != candidate_count
        ):
            errors.append("AnyPlace candidate_count/full_plan_pass_count mismatch")
        if not candidate_ids:
            errors.append("AnyPlace stored no MoveIt PASS placement candidate")
        for legacy_key in ("selected_grasp", "source_grasp_id", "place_grasp_pose"):
            if base._contains(anyplace, legacy_key):
                errors.append(f"AnyPlace leaked forbidden grasp-coupled field: {legacy_key}")
        if not any(
            base._contains(anyplace, "schema_version", schema)
            for schema in (
                "openeta.moveit_candidate_funnel.v2",
                "openeta.moveit_candidate_funnel.v3",
            )
        ):
            errors.append("AnyPlace qualification evidence is missing")
        if not base._contains(anyplace, "type", "placement_candidate_image"):
            errors.append("AnyPlace candidate image attachment is missing")
        rotations = [
            value
            for value in base._values(anyplace, "rotation_matrix")
            if isinstance(value, list)
            and len(value) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in value)
        ]
        if len(rotations) < len(candidate_ids):
            errors.append("AnyPlace PASS candidates do not retain full rotations")
        # Measured-attachment requalification uses the host-normalized frozen
        # packet. Validate independent public observations on the one model
        # inference call before that non-inference replay boundary.
        anyplace_parameters = _parameters(first_anyplace)
        object_observation = anyplace_parameters.get("object_observation")
        placement_observation = anyplace_parameters.get("placement_observation")
        if not isinstance(object_observation, Mapping) or not isinstance(
            placement_observation, Mapping
        ):
            errors.append("AnyPlace independent object/placement observations are missing")
        elif not base._contains(object_observation, "object_mask") or not base._contains(
            placement_observation, "placement_region_mask"
        ):
            errors.append("AnyPlace independent masks are missing")
        if not any(base._contains(payload, "state", "attached") for payload in payloads):
            errors.append("native attach ACK evidence missing")
        if not any(
            base._contains(payload, "schema_version", "openeta.attachment_transform.v1")
            for payload in payloads
        ):
            errors.append("measured T_eef_object_attached evidence missing")
        if not any(
            base._contains(
                payload,
                "reason_code",
                "NATIVE_GRASP_ATTACHMENT_CONFIRMED",
            )
            and base._contains(payload, "grasp_confirmed", True)
            for payload in payloads
        ):
            errors.append("native bilateral-contact plus attach-ACK proof missing")
        if not any(
            base._contains(payload, "reason_code", "NATIVE_GRASP_TARGET_HELD")
            and any(
                float(value) <= 0.01
                for value in base._values(payload, "capture_relative_translation_m")
                if isinstance(value, (int, float))
            )
            for payload in payloads
        ):
            errors.append("attached transport drift proof missing")
        forbidden_pose_stages = {
            "hover",
            "align",
            "align_move",
            "precontact",
            "descend",
            "lift",
            "retreat",
            "carry_raise",
            "carry_hover",
        }
        if any(
            str(value) in forbidden_pose_stages for value in base._values(calls, "grasp_stage")
        ) or any(
            str(value) in forbidden_pose_stages for value in base._values(calls, "placement_stage")
        ):
            errors.append("artificial grasp/place waypoint stage was executed")
        if not any(base._contains(payload, "state", "detached") for payload in payloads):
            errors.append("native detach ACK evidence missing")
        placements = [
            value
            for payload in payloads
            for value in base._values(payload, "placement_verification")
            if isinstance(value, Mapping)
        ]
        if not any(
            value.get("placement_confirmed") is True
            and value.get("verdict") == "PASS"
            and float((value.get("evidence") or {}).get("stable_duration_s", 0.0)) >= 0.5
            and float((value.get("evidence") or {}).get("terminal_drift_m", 1.0)) <= 0.005
            for value in placements
        ):
            errors.append("stable in-zone placement verification missing")
        fingerprints = [
            str(value) for value in base._values(events, "request_fingerprint") if value
        ]
        if _repeated_failed_motion_fingerprints(calls):
            errors.append("a failed motion request fingerprint was repeated")
        placement_failures = [
            call
            for call in calls
            if _name(call) == "move_to"
            and base._contains(call, "purpose", "placement")
            and base._contains(call, "error_code", "MOTION_PLAN_FAILED")
            and base._contains(call, "execution_started", False)
        ]
        if scenario == "normal" and placement_failures:
            errors.append("normal scenario unexpectedly injected a placement rejection")
        frozen_pair_qualification_blocks = [
            block
            for grasp_call in grasp_calls
            for block in _qualification_blocks(grasp_call, artifact_root=paths.root)
            if block.get("purpose") == "placement"
        ]
        qualification_results = [
            result
            for block in frozen_pair_qualification_blocks
            for result in block.get("results", [])
        ]
        first_full_plan_rejections = [
            value
            for value in qualification_results
            if isinstance(value, Mapping)
            and value.get("full_plan_submitted") is True
            and value.get("verdict") == "FAIL"
            and value.get("reason") == "plan_only_failed"
            and value.get("execution_started") is False
        ]
        if scenario == "reject-first":
            if len(first_full_plan_rejections) != 1:
                errors.append("reject-first did not retain exactly one qualification rejection")
            if placement_failures:
                errors.append("reject-first reached execution planning with a rejected candidate")
        if not fingerprints:
            errors.append("motion request fingerprint evidence missing")
        scene_revisions = [
            int(value)
            for value in base._values(payloads, "planning_scene_revision")
            if isinstance(value, int)
        ]
        if len(set(scene_revisions)) < 3:
            errors.append("reset/attach/detach planning-scene revision chain missing")
        for call in calls:
            if _name(call) in base.MUTATING_TOOLS and not base._scripted_approved(call):
                errors.append(f"{_name(call)} lacks scripted_tui approval")
        status = "failed" if errors else "passed"
        return {
            "status": status,
            "errors": list(dict.fromkeys(errors)),
            "trace_paths": [str(path.resolve()) for path in trace_paths],
            "tool_call_count": len(calls),
            "planner_evidence": planner_evidence,
            "execution_profile": profile,
            "qualification_profile": funnel_profile,
            "acceptance_scope": EXECUTION_PROFILE_SCOPES[profile],
            "planner_provider_invoked": bool(
                planner_evidence["closed_loop_action_count"]
                or planner_evidence["isolated_selection_action_count"]
                or planner_evidence["total_tokens"]
            ),
            "scenario": scenario,
            "grasp_backend": backend,
        }
    except Exception as exc:  # noqa: BLE001 - verifier must return a report.
        return {
            "status": "blocked",
            "errors": [f"evidence unreadable: {type(exc).__name__}: {exc}"],
            "trace_paths": [],
            "tool_call_count": 0,
            "planner_evidence": {},
            "execution_profile": profile,
            "qualification_profile": funnel_profile,
            "acceptance_scope": EXECUTION_PROFILE_SCOPES[profile],
            "planner_provider_invoked": None,
            "scenario": scenario,
            "grasp_backend": backend,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-provider-preflight", action="store_true")
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal")
    parser.add_argument(
        "--execution-profile",
        choices=EXECUTION_PROFILES,
        default=DEFAULT_EXECUTION_PROFILE,
        help=(
            "agentic_normal requires planner/VLM decisions; smoke_normal exercises "
            "the same normal model/control chain through deterministic host obligations "
            "and forbids planner/VLM invocation."
        ),
    )
    parser.add_argument(
        "--qualification-profile",
        choices=QUALIFICATION_PROFILES,
        default=DEFAULT_QUALIFICATION_PROFILE,
        help=(
            "MoveIt candidate funnel used by this acceptance. The repository-wide "
            "runtime default remains legacy for immediate rollback."
        ),
    )
    parser.add_argument(
        "--grasp-backend",
        choices=GRASP_BACKENDS,
        default=DEFAULT_GRASP_BACKEND,
        help="Strict grasp backend for this acceptance run; no cross-backend fallback.",
    )
    for name, url in DEFAULT_SERVICE_URLS.items():
        parser.add_argument("--" + name.removeprefix("openeta-") + "-url", default=url)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execution_profile == "smoke_normal" and args.scenario != "normal":
        raise base.AcceptanceError("--execution-profile smoke_normal requires --scenario normal")
    repo = Path(__file__).resolve().parents[1]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = (
        Path(args.run_root).resolve()
        if args.run_root
        else repo / ".cache/reports" / f"pick-place-gazebo-{stamp}"
    )
    paths = base.case_paths(run_root, MILESTONE, MODE)
    if args.verify_only:
        report = verify_case(
            paths,
            scenario=args.scenario,
            grasp_backend=args.grasp_backend,
            execution_profile=args.execution_profile,
            qualification_profile=args.qualification_profile,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    runtime = gazebo_runtime_preflight(repo)
    if runtime["status"] != "passed":
        print(json.dumps(runtime, ensure_ascii=False, indent=2))
        return 2
    services = _services_for_backend(
        args.grasp_backend,
        sam3_url=args.sam3_url,
        anygrasp_url=args.anygrasp_url,
        anyplace_url=args.anyplace_url,
        graspgenx_url=args.graspgenx_url,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    preflight = service_preflight(services)
    base._json_dump(run_root / "service-preflight.json", preflight)
    if preflight["status"] != "passed":
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2
    if args.execution_profile == "smoke_normal":
        base._json_dump(
            run_root / base.PROVIDER_PREFLIGHT_FILENAME,
            {
                "schema_version": "openeta.planner_provider_preflight.v1",
                "status": "not_run",
                "reason": "smoke_normal_forbids_planner_vlm",
                "planner_provider_invoked": False,
            },
        )
    elif not args.skip_provider_preflight:
        provider = base._provider_preflight_result(repo)
        base._json_dump(run_root / base.PROVIDER_PREFLIGHT_FILENAME, provider)
        if provider["status"] != "passed":
            print(json.dumps(provider, ensure_ascii=False, indent=2))
            return 2
    allocation = base.allocate(f"{MILESTONE}-{MODE}", preflight=not args.prepare_only)
    paths = prepare_case(
        repo,
        run_root,
        allocation,
        services,
        scenario=args.scenario,
        grasp_backend=args.grasp_backend,
        execution_profile=args.execution_profile,
        qualification_profile=args.qualification_profile,
    )
    if args.prepare_only:
        print(run_root)
        return 0
    code = base.run_case(
        repo,
        paths,
        allocation,
        calibration_profile=RM75_ROBOTIQ_GRASP_CALIBRATION_PROFILE,
        extra_environment={
            "OPENETA_EPISODE_MAX_TOTAL_TOKENS": (
                "1" if args.execution_profile == "smoke_normal" else "10000000"
            ),
            "OPENETA_GRASP_BACKEND": args.grasp_backend,
            "OPENETA_QUALIFICATION_PROFILE": args.qualification_profile,
            # Cold software-rendered Gazebo launches occasionally need more
            # than the deployment default before the documented world-control
            # service is discoverable.  This affects startup only; motion,
            # planning, execution, and verification deadlines stay unchanged.
            "OPENETA_GAZEBO_STARTUP_TIMEOUT_S": "90",
            # Model inference is read-only and normally completes well below
            # this bound.  A broken legacy SSE return channel is retried once
            # inside the host, without adding a TUI/model-planner turn.
            "OPENETA_PERCEPTION_RPC_TIMEOUT_S": "90",
            **(
                {"OPENETA_ACCEPTANCE_PLACEMENT_FAULT": args.scenario}
                if args.scenario != "normal"
                else {}
            ),
        },
    )
    report = verify_case(
        paths,
        scenario=args.scenario,
        grasp_backend=args.grasp_backend,
        execution_profile=args.execution_profile,
        qualification_profile=args.qualification_profile,
    )
    report.update({"schema_version": SCHEMA_VERSION, "run_root": str(run_root.resolve())})
    base._json_dump(run_root / "acceptance-report.json", report, exclusive=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if code == 0 and report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
