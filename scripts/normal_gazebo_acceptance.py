#!/usr/bin/env python3
"""Isolated real-service Gazebo acceptance for constraint-correct placement."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import urllib.request

from scripts import gazebo_acceptance_runtime as base
from agent.runtime.calibration_registry import RM75_ROBOTIQ_GRASP_CALIBRATION_PROFILE
from agent.runtime.qualification_v3 import JOINT_SOLUTION_DEDUP_DISTANCE
from agent.runtime.release_evidence import ordered_native_release_proof
from agent.runtime.runtime_assembly import DEFAULT_PERCEPTION_RPC_TIMEOUT_S
from extensions.gazebo.native_grasp import load_acceptance_scene_contract
from tools.candidate_config import (
    DEFAULT_GRASPGENX_RAW_POOL_SIZE,
    QUALIFICATION_PROFILES,
)


SCHEMA_VERSION = "openeta.gazebo_pick_place_acceptance.v1"
SUITE = "pick-place"
OPERATOR_MODES = (base.SCRIPTED_TUI, base.HUMAN_TUI)
DEFAULT_OPERATOR_MODE = base.SCRIPTED_TUI
ENV_ID = "openeta/gazebo_rm75_robotiq2f85_pickplace-v0"
GRASP_BACKENDS = ("anygrasp", "graspgenx")
DEFAULT_GRASP_BACKEND = "graspgenx"
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
    for name in ("openeta-sam3", "openeta-graspgenx", "openeta-anyplace")
}
REQUIRED_REAL_PICK_PLACE_TOOLS = (
    "create_simulator_env",
    "sam3",
    "grasp_pose_estimate",
    "gripper_control",
    "anyplace",
    "close_simulator_env",
)
DEFAULT_MULTI_NORMAL_TASK_VARIANT = "wrench-green-bolt-blue"
# Acceptance-only user requests. These are written to the TUI as ordinary
# operator utterances and retained privately by the verifier. They are never
# placed in the physical scene contract or planner context.
MULTI_NORMAL_TASK_VARIANTS = {
    DEFAULT_MULTI_NORMAL_TASK_VARIANT: {
        "operator_instruction": (
            "请先把黄色活动扳手放进绿色零件箱，再把红色六角螺栓放进蓝色零件箱。"
            "其他物件不要动，每放好一件后继续下一件，全部完成后再结束。"
        ),
        "items": [
            ("target_object", "target_link", "yellow wrench", "green_parts_bin", "green parts bin"),
            ("red_m24_hex_bolt", "red_m24_hex_bolt_link", "red hex bolt", "blue_parts_bin", "blue parts bin"),
        ],
    },
    "wrench-blue-bolt-green": {
        "operator_instruction": (
            "请先把黄色活动扳手放进蓝色零件箱，再把红色六角螺栓放进绿色零件箱。"
            "其他物件不要动，每放好一件后继续下一件，全部完成后再结束。"
        ),
        "items": [
            ("target_object", "target_link", "yellow wrench", "blue_parts_bin", "blue parts bin"),
            ("red_m24_hex_bolt", "red_m24_hex_bolt_link", "red hex bolt", "green_parts_bin", "green parts bin"),
        ],
    },
    "bolt-blue-wrench-green": {
        "operator_instruction": (
            "请先把红色六角螺栓放进蓝色零件箱，再把黄色活动扳手放进绿色零件箱。"
            "其他物件不要动，每放好一件后继续下一件，全部完成后再结束。"
        ),
        "items": [
            ("red_m24_hex_bolt", "red_m24_hex_bolt_link", "red hex bolt", "blue_parts_bin", "blue parts bin"),
            ("target_object", "target_link", "yellow wrench", "green_parts_bin", "green parts bin"),
        ],
    },
    "bolt-green-wrench-blue": {
        "operator_instruction": (
            "请先把红色六角螺栓放进绿色零件箱，再把黄色活动扳手放进蓝色零件箱。"
            "其他物件不要动，每放好一件后继续下一件，全部完成后再结束。"
        ),
        "items": [
            ("red_m24_hex_bolt", "red_m24_hex_bolt_link", "red hex bolt", "green_parts_bin", "green parts bin"),
            ("target_object", "target_link", "yellow wrench", "blue_parts_bin", "blue parts bin"),
        ],
    },
}
LEGACY_MULTI_NORMAL_REQUEST_IDS = {
    DEFAULT_MULTI_NORMAL_TASK_VARIANT: "multi_normal",
    "wrench-blue-bolt-green": "multi_normal1",
    "bolt-blue-wrench-green": "multi_normal2",
    "bolt-green-wrench-blue": "multi_normal3",
}
MULTI_SORT_SCENARIOS = (
    "multi_normal",
    "multi_normal_random_12345",
)
SCENARIOS = (
    "normal",
    *MULTI_SORT_SCENARIOS,
)
EXECUTION_PROFILES = ("agentic_normal", "smoke_normal")
DEFAULT_EXECUTION_PROFILE = "agentic_normal"
# Acceptance rolls the already-shadowable v3 funnel forward explicitly while
# the general runtime keeps its conservative `legacy` default and instant
# environment-variable rollback.
DEFAULT_QUALIFICATION_PROFILE = "fast_v3"
DEFAULT_GAZEBO_ACCEPTANCE_STARTUP_TIMEOUT_S = 180.0
GAZEBO_SIM_PACKAGE = "openeta_rm75_robotiq2f85_sim"

EXECUTION_PROFILE_PLANNER_MODES = {
    "agentic_normal": "agentic_closed_loop",
    "smoke_normal": "host_macro",
}
EXECUTION_PROFILE_SCOPES = {
    "agentic_normal": "agentic_normal_vlm_acceptance",
    "smoke_normal": "control_only_no_vlm_smoke_normal_not_agentic_acceptance",
}

# These findings describe how an episode reached its result.  They are useful
# for latency and regression analysis, but they must never prescribe the
# agent's number of observations, model retries, candidate waves, or recovery
# order.  Formal PASS is decided by provider provenance, scene integrity,
# approved mutations, physical safety receipts, and terminal task completion.
_NON_BLOCKING_FLOW_FINDING_PREFIXES = (
    "required real pick-place tool call missing:",
    "VLM must configure exactly one session work order",
    "exactly one target-object and one placement-region SAM3 call",
    "SAM3 semantic prompts do not match",
    "SAM3 was rerun after",
    "model-contact, attach, frozen-goal release order",
    "agent-visible motion preview evidence",
    "host placement candidate compilation evidence",
    "host grasp candidate compilation evidence",
    "scenario requires exactly one",
    "frozen grasp frontier expansion",
    "GraspGenX raw candidate count evidence",
    "AnyGrasp raw candidate count evidence",
    "GraspGenX validated grasp search evidence",
    "AnyGrasp validated grasp search evidence",
    "GraspGenX L5 diversity evidence",
    "AnyGrasp L5 diversity evidence",
    "GraspGenX fast-v3 resumable grasp/place evidence",
    "AnyGrasp fast-v3 resumable grasp/place evidence",
    "each sort assignment requires one model AnyPlace call",
    "AnyPlace model inference count",
    "AnyPlace frozen-pool requalification",
    "post-attach AnyPlace requalification",
    "AnyPlace model raw pool evidence",
    "final AnyPlace result stored no MoveIt PASS candidate",
    "AnyPlace exposed candidate count",
    "AnyPlace stored no MoveIt PASS placement candidate",
    "AnyPlace leaked forbidden grasp-coupled field",
    "AnyPlace qualification evidence",
    "AnyPlace candidate image attachment",
    "AnyPlace PASS candidates do not retain full rotations",
    "AnyPlace independent object/placement observations",
    "AnyPlace independent masks",
    "artificial grasp/place waypoint stage",
    "a failed motion request fingerprint was repeated",
    "qualification evidence did not evaluate scene obstacle:",
    "motion request fingerprint evidence missing",
)


def _is_non_blocking_flow_finding(message: str) -> bool:
    return any(
        message.startswith(prefix)
        for prefix in _NON_BLOCKING_FLOW_FINDING_PREFIXES
    )

# Formal acceptance must exercise the deployment's real provider policy.  It
# may record request latency, retries, and token use, but it must not replace
# the configured context, timeout, fallback, or retry policy.
AGENTIC_PROVIDER_RESILIENCE_ENV: dict[str, str] = {}


AGENTIC_CONTROL = """
请先看清工作台，然后帮我完成下面这件事。
""".strip()

SMOKE_CONTROL = """
这是一次不调用视觉语言模型的控制链检查，系统会按当前任务逐步执行；遇到语义歧义时停止。
""".strip()

TASK_INSTRUCTIONS = """
如果视角不清楚，可以换个角度确认。完成后确认物品已经在目标箱里放稳就好。
""".strip()


SCENARIO_INSTRUCTIONS = {
    "normal": "桌上还有几件外观相近的工具，拿之前请确认没有拿错。",
    "multi_normal": (
        "这是同一工作单元内的连续分拣；放好第一件后不要关闭环境，"
        "重新看清当前场景并继续第二件。"
    ),
    "multi_normal_random_12345": (
        "工作台物件的位置和朝向已经变化。这是同一工作单元内的连续分拣；"
        "放好第一件后不要关闭环境，重新看清当前场景并继续第二件。"
    ),
}


def _scene_contract(scenario: str) -> dict[str, Any]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported pick-place acceptance scenario: {scenario}")
    return load_acceptance_scene_contract(scenario)


def _validated_task_variant(scenario: str, value: str) -> str:
    variant = str(value).strip().lower()
    if scenario in MULTI_SORT_SCENARIOS:
        if variant not in MULTI_NORMAL_TASK_VARIANTS:
            choices = ", ".join(MULTI_NORMAL_TASK_VARIANTS)
            raise ValueError(f"multi-sort task variant must be one of: {choices}")
        return variant
    if variant != DEFAULT_MULTI_NORMAL_TASK_VARIANT:
        raise ValueError("--task-variant is only valid with a multi-sort scenario")
    return variant


def _scene_task(
    scene: Mapping[str, Any],
    *,
    scenario: str,
    task_variant: str = DEFAULT_MULTI_NORMAL_TASK_VARIANT,
) -> dict[str, str]:
    variant = (
        MULTI_NORMAL_TASK_VARIANTS[_validated_task_variant(scenario, task_variant)]
        if scenario in MULTI_SORT_SCENARIOS
        else None
    )
    if variant is not None:
        first = variant["items"][0]
        return {
            "target_prompt": str(first[2]),
            "placement_object_prompt": str(first[2]),
            "placement_region_prompt": str(first[4]),
            "operator_instruction": str(variant["operator_instruction"]),
        }
    raw = scene.get("task")
    if isinstance(raw, Mapping):
        return {
            "target_prompt": str(raw["target_prompt"]),
            "placement_object_prompt": str(raw["placement_object_prompt"]),
            "placement_region_prompt": str(raw["placement_region_prompt"]),
            "operator_instruction": str(
                raw.get("operator_instruction")
                or "请把工作台上的目标物体放进指定的放置区，其他物件不要动。"
            ),
        }
    return {
        "target_prompt": "red rectangular block",
        "placement_object_prompt": "red rectangular block",
        "placement_region_prompt": "green placement zone marker",
        "operator_instruction": "请把工作台上的红色长方体放进绿色放置区，其他物件不要动。",
    }


def _expected_work_order(
    scene: Mapping[str, Any],
    *,
    scenario: str,
    task_variant: str = DEFAULT_MULTI_NORMAL_TASK_VARIANT,
) -> list[dict[str, str]]:
    variant = (
        MULTI_NORMAL_TASK_VARIANTS[_validated_task_variant(scenario, task_variant)]
        if scenario in MULTI_SORT_SCENARIOS
        else None
    )
    if variant is not None:
        target_catalog = {
            str(item["target_object_id"]): item
            for item in scene.get("manipulation_targets", [])
            if isinstance(item, Mapping)
        }
        region_catalog = {
            str(item["id"]): item
            for item in scene.get("placement_regions", [])
            if isinstance(item, Mapping)
        }
        expected: list[dict[str, str]] = []
        for target_id, target_link, target_prompt, region_id, region_prompt in variant[
            "items"
        ]:
            target = target_catalog[str(target_id)]
            region = region_catalog[str(region_id)]
            expected.append(
                {
                    "id": (
                        f"{('yellow_wrench' if target_id == 'target_object' else 'red_bolt')}"
                        f"_to_{region_id}"
                    ),
                    "target_object_id": str(target_id),
                    "target_link": str(target_link),
                    "target_prompt": str(target_prompt),
                    "target_perception_prompt": str(
                        target.get("perception_prompt") or target_prompt
                    ),
                    "placement_object_prompt": str(target_prompt),
                    "source_support_object_id": "work_table",
                    "placement_region_id": str(region_id),
                    "placement_region_prompt": str(region_prompt),
                    "placement_region_perception_prompt": str(
                        region.get("perception_prompt") or region_prompt
                    ),
                }
            )
        return expected
    task = _scene_task(scene, scenario=scenario, task_variant=task_variant)
    return [
        {
            "id": "default",
            "target_object_id": "target_object",
            "target_link": "target_link",
            "target_prompt": task["target_prompt"],
            "placement_object_prompt": task["placement_object_prompt"],
            "source_support_object_id": "work_table",
            "placement_region_id": str(
                scene.get("selected_placement_region_id") or "placement_zone_marker"
            ),
            "placement_region_prompt": task["placement_region_prompt"],
        }
    ]


def _metadata_semantic(value: str) -> str:
    return "_".join(str(value).strip().split())


def _scene_receipt(
    scene: Mapping[str, Any],
    *,
    scenario: str,
    task_variant: str = DEFAULT_MULTI_NORMAL_TASK_VARIANT,
) -> dict[str, Any]:
    task = _scene_task(scene, scenario=scenario, task_variant=task_variant)
    expected_work_order = _expected_work_order(
        scene,
        scenario=scenario,
        task_variant=task_variant,
    )
    target = scene.get("target_object")
    target_evidence = {
        "id": "target_object",
        "shape_class": (
            str(target["shape_class"])
            if isinstance(target, Mapping)
            else "rectangular_block"
        ),
        "bounding_box_xyz": [
            float(value)
            for value in (
                target["bounding_box_xyz"]
                if isinstance(target, Mapping)
                else [0.04, 0.04, 0.06]
            )
        ],
    }
    regions = scene.get("placement_regions")
    region_ids = (
        [str(region["id"]) for region in regions]
        if isinstance(regions, list)
        else ["placement_zone_marker"]
    )
    first_region_id = expected_work_order[0]["placement_region_id"]
    first_region = next(
        (
            region
            for region in (regions if isinstance(regions, list) else [])
            if isinstance(region, Mapping) and str(region.get("id") or "") == first_region_id
        ),
        None,
    )
    return {
        "schema_version": "openeta.gazebo_acceptance_scene_receipt.v2",
        "acceptance_request_id": (
            f"{scenario}:{task_variant}"
            if scenario in MULTI_SORT_SCENARIOS
            else scenario
        ),
        "scene_id": str(scene["scene_id"]),
        "seed": int(scene["seed"]),
        "contract_sha256": str(scene["contract_sha256"]),
        "static_obstacle_ids": [
            str(obstacle["id"]) for obstacle in scene["static_obstacles"]
        ],
        "destination_center_xy": [
            float(value)
            for value in (
                first_region.get("center_xy")
                if isinstance(first_region, Mapping)
                else scene.get("destination_center_xy", [0.48, -0.10])
            )
        ],
        "destination_size_xy_m": [
            float(value)
            for value in scene.get("destination_size_xy_m", [0.12, 0.12])
        ],
        "task": task,
        "target_object": target_evidence,
        "placement_region_ids": region_ids,
        "selected_placement_region_id": str(
            first_region_id
        ),
        "expected_work_order": expected_work_order,
    }


def _scenario_environment(scenario: str) -> dict[str, str]:
    """Bind one physical scene for the complete worker lifetime."""

    scene = _scene_contract(scenario)
    return {"OPENETA_ACCEPTANCE_SCENE": str(scene["scene_id"])}


def _validated_grasp_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in GRASP_BACKENDS:
        choices = ", ".join(GRASP_BACKENDS)
        raise ValueError(f"grasp backend must be one of: {choices}")
    return backend


def _validated_operator_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in OPERATOR_MODES:
        choices = ", ".join(OPERATOR_MODES)
        raise ValueError(f"operator mode must be one of: {choices}")
    return mode


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


def _automation_metadata_for_backend(
    grasp_backend: str,
    *,
    execution_profile: str = DEFAULT_EXECUTION_PROFILE,
    qualification_profile: str = DEFAULT_QUALIFICATION_PROFILE,
    scenario: str = "normal",
    task_variant: str = DEFAULT_MULTI_NORMAL_TASK_VARIANT,
    operator_mode: str = DEFAULT_OPERATOR_MODE,
) -> str:
    _validated_grasp_backend(grasp_backend)
    mode = _validated_operator_mode(operator_mode)
    profile = _validated_execution_profile(execution_profile)
    funnel_profile = _validated_qualification_profile(qualification_profile)
    scene = _scene_contract(scenario)
    task = _scene_task(scene, scenario=scenario, task_variant=task_variant)
    planner_mode = EXECUTION_PROFILE_PLANNER_MODES[profile]
    semantic_metadata = (
        "work_order_source=vlm_conversation"
        if scenario in MULTI_SORT_SCENARIOS
        else (
            f"grasp_target={_metadata_semantic(task['target_prompt'])}; "
            f"placement_object={_metadata_semantic(task['placement_object_prompt'])}; "
            f"placement_region={_metadata_semantic(task['placement_region_prompt'])}"
        )
    )
    return (
        f"[{'automation=scripted_tui' if mode == base.SCRIPTED_TUI else 'operator=human_tui'}; "
        f"planner_mode={planner_mode}; "
        f"execution_profile={profile}; qualification_profile={funnel_profile}; "
        f"environment_id={ENV_ID}; "
        f"environment_seed={int(scene['seed'])}; "
        f"acceptance_scene={str(scene['scene_id'])}; "
        f"{semantic_metadata}]"
    )


def _instructions_for_backend(
    grasp_backend: str,
    *,
    execution_profile: str = DEFAULT_EXECUTION_PROFILE,
    qualification_profile: str = DEFAULT_QUALIFICATION_PROFILE,
    scenario: str = "normal",
    task_variant: str = DEFAULT_MULTI_NORMAL_TASK_VARIANT,
) -> str:
    _validated_grasp_backend(grasp_backend)
    profile = _validated_execution_profile(execution_profile)
    _validated_qualification_profile(qualification_profile)
    scene = _scene_contract(scenario)
    task = _scene_task(scene, scenario=scenario, task_variant=task_variant)
    control = AGENTIC_CONTROL if profile == "agentic_normal" else SMOKE_CONTROL
    recovery = f"\n{TASK_INSTRUCTIONS}" if profile == "agentic_normal" else ""
    return (
        f"{control}\n"
        f"{task['operator_instruction']}\n"
        f"{SCENARIO_INSTRUCTIONS[scenario]}"
        f"{recovery}\n"
    )


def _required_tools_for_backend(
    grasp_backend: str,
    *,
    scenario: str = "normal",
) -> tuple[str, ...]:
    backend = _validated_grasp_backend(grasp_backend)
    required = tuple(
        backend if name == "grasp_pose_estimate" else name
        for name in REQUIRED_REAL_PICK_PLACE_TOOLS
    )
    if scenario in MULTI_SORT_SCENARIOS:
        return (*required[:1], "configure_work_order", *required[1:])
    return required


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


def _expected_service_raw_pool_size(name: str) -> int | None:
    """Mirror the runtime's configurable provider-pool registration values."""

    configuration = {
        "openeta-anygrasp": ("OPENETA_ANYGRASP_RAW_POOL_SIZE", 200),
        "openeta-graspgenx": (
            "OPENETA_GRASPGENX_RAW_POOL_SIZE",
            DEFAULT_GRASPGENX_RAW_POOL_SIZE,
        ),
        "openeta-anyplace": ("OPENETA_ANYPLACE_RAW_POOL_SIZE", 96),
    }
    if name not in configuration:
        return None
    environment_name, default = configuration[name]
    try:
        value = int(os.environ.get(environment_name, default))
    except (TypeError, ValueError):
        return -1
    return value if value > 0 else -1


def service_preflight(services: Mapping[str, str]) -> dict[str, Any]:
    expected = {
        "openeta-sam3": "sam3",
        "openeta-anygrasp": "anygrasp",
        "openeta-anyplace": "anyplace",
        "openeta-graspgenx": "openeta-graspgenx",
    }
    rows: dict[str, Any] = {}
    for name, url in services.items():
        expected_raw_pool_size = _expected_service_raw_pool_size(name)
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
                        payload.get("raw_pool_size") == expected_raw_pool_size
                        and isinstance(payload.get("returned_candidate_count"), int)
                        and payload.get("returned_candidate_count") >= 0
                        and payload.get("returned_candidate_count")
                        <= expected_raw_pool_size
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
        from rosgraph_msgs.msg import Clock

        # Importing a generated message class is lazy and does not prove that
        # its native ROS libraries share the same ABI.  Force the exact load
        # rclpy performs when it creates a /clock subscription so a mixed ROS
        # underlay fails before provider preflight or a TUI episode begins.
        Clock.__class__.__import_type_support__()
        if Clock.__class__._TYPE_SUPPORT is None:
            raise RuntimeError("rosgraph_msgs Clock type support is unavailable")
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
    task_variant: str = DEFAULT_MULTI_NORMAL_TASK_VARIANT,
    operator_mode: str = DEFAULT_OPERATOR_MODE,
) -> base.CasePaths:
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported pick-place acceptance scenario: {scenario}")
    backend = _validated_grasp_backend(grasp_backend)
    mode = _validated_operator_mode(operator_mode)
    profile = _validated_execution_profile(execution_profile)
    funnel_profile = _validated_qualification_profile(qualification_profile)
    variant = _validated_task_variant(scenario, task_variant)
    scene = _scene_contract(scenario)
    if profile == "smoke_normal" and scenario != "normal":
        raise ValueError("smoke_normal execution profile requires scenario=normal")
    configured_grasp_services = set(services).intersection(GRASP_SERVICE_NAMES.values())
    required_grasp_service = GRASP_SERVICE_NAMES[backend]
    if configured_grasp_services != {required_grasp_service}:
        raise ValueError(
            f"strict acceptance requires exactly one grasp service: {required_grasp_service}"
        )
    paths = base.case_paths(run_root, SUITE, mode)
    paths.root.mkdir(parents=True, exist_ok=False)
    base._json_dump(
        paths.mcp_config,
        {
            "mcpServers": {
                "openeta-sim": {"url": f"http://127.0.0.1:{allocation.port}/mcp"},
                **{name: {"url": url} for name, url in services.items()},
            }
        },
    )
    paths.instructions.write_text(
        _instructions_for_backend(
            backend,
            execution_profile=profile,
            qualification_profile=funnel_profile,
            scenario=scenario,
            task_variant=variant,
        ),
        encoding="utf-8",
    )
    receipt = base.environment_receipt(
        repo,
        allocation,
        case_name=f"{SUITE}-{mode}",
        before=base._process_snapshot(),
    )
    receipt["acceptance_scenario"] = scenario
    receipt["acceptance_scene"] = _scene_receipt(
        scene,
        scenario=scenario,
        task_variant=variant,
    )
    receipt["operator_mode"] = mode
    receipt["task_variant"] = variant if scenario in MULTI_SORT_SCENARIOS else None
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


def _candidate_pass_counts_consistent(
    candidate_count: object, full_plan_pass_count: object
) -> bool:
    """Prove every exposed queue entry has a corresponding L5 PASS."""

    return bool(
        isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and isinstance(full_plan_pass_count, int)
        and not isinstance(full_plan_pass_count, bool)
        and 1 <= candidate_count <= full_plan_pass_count
    )


def _parameters(call: Mapping[str, Any]) -> Mapping[str, Any]:
    value = call.get("parameters")
    return value if isinstance(value, Mapping) else {}


def _anyplace_model_inference_calls(
    calls: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Classify model calls without confusing frozen requalification.

    Current producers emit an explicit Boolean. Older traces omitted only the
    positive value, so every legacy call without ``reuse_frozen_goal_pool`` is
    still one model invocation; this matters for same-session multi-sort.
    """

    inference_calls: list[Mapping[str, Any]] = []
    for call in calls:
        flag = _call_outputs(call).get("anyplace_model_inference_invoked")
        if flag is True or (
            flag is None and _parameters(call).get("reuse_frozen_goal_pool") is not True
        ):
            inference_calls.append(call)
    return inference_calls


def _ordered(names: Sequence[str], required: Sequence[str]) -> bool:
    cursor = iter(names)
    return all(any(name == wanted for name in cursor) for wanted in required)


def _call_succeeded(call: Mapping[str, Any]) -> bool:
    result = call.get("result")
    return isinstance(result, Mapping) and result.get("success") is True


def _call_assignment_ids(call: Mapping[str, Any]) -> set[str]:
    return {
        str(binding.get("assignment_id") or "")
        for binding in base._values(call, "native_target_binding")
        if isinstance(binding, Mapping) and str(binding.get("assignment_id") or "")
    }


def _assignment_execution_token(
    call: Mapping[str, Any], *, backend: str
) -> str:
    """Reduce one successful call to its assignment-order evidence token."""

    if not _call_succeeded(call):
        return ""
    name = _name(call)
    parameters = _parameters(call)
    pose = parameters.get("target_pose")
    pose = pose if isinstance(pose, Mapping) else {}
    assignment_ids = _call_assignment_ids(call)
    assignment_id = next(iter(assignment_ids)) if len(assignment_ids) == 1 else ""
    if name == "observe":
        return "observe"
    if name == "anyplace":
        if call in _anyplace_model_inference_calls([call]):
            return "anyplace_model"
        if _call_outputs(call).get("anyplace_model_inference_invoked") is False:
            # Frozen-goal requalification is performed against the currently
            # attached PlanningScene state.  Older traces did not repeat the
            # work-order binding on this call, so correlate it by requiring it
            # between assignment-bound attach and placement transitions.
            return "placement_qualification"
    if name == backend and parameters.get("mode") != "frozen_frontier":
        return "grasp_model"
    if name == "move_to" and pose.get("grasp_stage") == "contact":
        return "grasp_contact"
    if name == "move_to" and pose.get("purpose") == "placement" and assignment_id:
        return f"placement_move:{assignment_id}"
    if name == "gripper_control" and assignment_id:
        if parameters.get("position") == 0:
            return f"attach:{assignment_id}"
        if parameters.get("position") == 1:
            return f"release:{assignment_id}"
    return ""


def _ordered_assignment_execution(
    calls: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    *,
    backend: str,
) -> bool:
    """Correlate each successful pick/release chain with its work-order item.

    Candidate motion failures and their exact-anchor recovery may appear between
    model inference and the eventual successful contact.  A name-only
    subsequence can accidentally treat that failed contact as the accepted
    grasp, and it also incorrectly requires a standalone ``observe`` before the
    first assignment even though environment creation already returns calibrated
    RGB-D.  Match only successful physical transitions and bind close,
    requalification, placement, and open evidence to the same assignment.
    A standalone ``observe`` is deliberately not part of the sequence: the
    initial calibrated frame comes from environment creation, and a successful
    release can carry the next assignment's causal post-action RGB-D frame.
    """

    observed = [
        token
        for call in calls
        if (token := _assignment_execution_token(call, backend=backend))
    ]
    required: list[str] = []
    for assignment in assignments:
        assignment_id = str(assignment.get("id") or "")
        if not assignment_id:
            return False
        required.extend(
            (
                "anyplace_model",
                "grasp_model",
                "grasp_contact",
                f"attach:{assignment_id}",
                "placement_qualification",
                f"placement_move:{assignment_id}",
                f"release:{assignment_id}",
            )
        )
    return _ordered(observed, required)


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

    if int(evidence.get("closed_loop_action_count") or 0) < 1:
        errors.append("agentic normal did not invoke the main VLM planner")
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


def _has_v3_grasp_search_evidence(call: Mapping[str, Any], *, artifact_root: Path) -> bool:
    """Validate a complete primary grasp and its bounded recovery search."""

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
                "complete_l5_pass_found_joint_branch_fallback",
                "complete_l5_pass_found_joint_space_fallback",
                "complete_l5_pass_found_partial_lookahead",
                "complete_l5_pass_found_single_branch_exhaustive_fallback",
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
        resumable_primary = _has_resumable_frozen_pair_evidence(call)
        if not (
            isinstance(generated_count, int)
            and not isinstance(generated_count, bool)
            and generated_count >= 10
            and generated_count == len(results)
            and (
                generated_count in reported_counts
                or (
                    resumable_primary
                    and reported_counts
                    and generated_count <= max(reported_counts)
                )
            )
            and isinstance(l5_pass_count, int)
            and not isinstance(l5_pass_count, bool)
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
        if (
            block.get("stop_reason") == "complete_l5_pass_found"
            and l5_pass_count == 1
            and len(selected_ids) == 1
            and metrics.get("l5_pass_target") == 1
            and metrics.get("l5_min_pass_target") == 1
        ):
            # Best-first fast_v3 executes the first complete grasp/place pair
            # and leaves the unvisited, model-frozen tail for physical-failure
            # recovery. Beam-2 may prove two joint branches for that one
            # candidate; those branches are not two candidate-level primaries.
            selected_cluster = selected_rows[0].get("se3_cluster_id")
            deferred_rows = [
                row
                for row in results
                if isinstance(row, Mapping)
                and row.get("verdict") == "NOT_EVALUATED"
                and isinstance(row.get("candidate_id"), str)
                and row.get("candidate_id")
                and isinstance(row.get("se3_cluster_id"), str)
                and row.get("se3_cluster_id")
                and row.get("se3_cluster_id") != selected_cluster
            ]
            pass_attempts = [
                attempt
                for attempt in l5_attempts
                if isinstance(attempt, Mapping)
                and attempt.get("candidate_id") == selected_ids[0]
                and attempt.get("verdict") == "PASS"
            ]
            published_grasps = _call_outputs(call).get("grasp_candidates")
            published_ids = [
                str(candidate.get("id") or "")
                for candidate in published_grasps
                if isinstance(candidate, Mapping) and str(candidate.get("id") or "")
            ] if isinstance(published_grasps, list) else []
            pass_branch_indices = {
                attempt.get("joint_branch_index")
                for attempt in pass_attempts
                if isinstance(attempt.get("joint_branch_index"), int)
                and not isinstance(attempt.get("joint_branch_index"), bool)
            }
            pass_branch_hashes = {
                str(attempt.get("joint_branch_joint_state_sha256") or "")
                for attempt in pass_attempts
                if str(attempt.get("joint_branch_joint_state_sha256") or "")
            }
            bounded_beam_proof = 1 <= len(pass_attempts) <= 2
            if len(pass_attempts) == 2:
                bounded_beam_proof = (
                    pass_branch_indices == {0, 1}
                    and len(pass_branch_hashes) == 2
                )
            if (
                isinstance(selected_cluster, str)
                and selected_cluster
                and deferred_rows
                and bounded_beam_proof
                and published_ids == selected_ids
                and resumable_primary
            ):
                return True
        if block.get("stop_reason") == "complete_l5_pass_found_joint_branch_fallback":
            branch_count = metrics.get("l5_joint_branch_pass_count")
            branch_proofs = block.get("selected_joint_branches")
            if not (
                l5_pass_count == 1
                and branch_count == 2
                and len(selected_ids) == 1
                and isinstance(selected_ids[0], str)
                and selected_ids[0]
                and isinstance(branch_proofs, list)
                and len(branch_proofs) == 2
            ):
                continue
            indexed_proofs = {
                proof.get("joint_branch_index"): proof
                for proof in branch_proofs
                if isinstance(proof, Mapping)
                and isinstance(proof.get("joint_branch_index"), int)
                and not isinstance(proof.get("joint_branch_index"), bool)
            }
            if set(indexed_proofs) != {0, 1}:
                continue
            selected_id = selected_ids[0]
            if any(
                proof.get("candidate_id") != selected_id
                or proof.get("verdict") != "PASS"
                for proof in indexed_proofs.values()
            ):
                continue
            branch_hashes = {
                index: proof.get("selected_ik_joint_state_sha256")
                for index, proof in indexed_proofs.items()
            }
            distance = indexed_proofs[1].get("normalized_distance_from_primary")
            if not (
                all(isinstance(value, str) and value for value in branch_hashes.values())
                and len(set(branch_hashes.values())) == 2
                and isinstance(distance, (int, float))
                and not isinstance(distance, bool)
                and math.isfinite(float(distance))
                and float(distance) >= JOINT_SOLUTION_DEDUP_DISTANCE
            ):
                continue
            pass_attempts = [
                attempt
                for attempt in l5_attempts
                if isinstance(attempt, Mapping)
                and attempt.get("candidate_id") == selected_id
                and attempt.get("verdict") == "PASS"
            ]
            indexed_attempts = {
                attempt.get("joint_branch_index"): attempt
                for attempt in pass_attempts
                if isinstance(attempt.get("joint_branch_index"), int)
                and not isinstance(attempt.get("joint_branch_index"), bool)
            }
            alternate_attempt_distance = (
                indexed_attempts.get(1, {}).get("joint_branch_normalized_distance")
                if isinstance(indexed_attempts.get(1), Mapping)
                else None
            )
            if not (
                set(indexed_attempts) == {0, 1}
                and all(
                    indexed_attempts[index].get(
                        "joint_branch_joint_state_sha256"
                    )
                    == branch_hashes[index]
                    for index in (0, 1)
                )
                and isinstance(alternate_attempt_distance, (int, float))
                and not isinstance(alternate_attempt_distance, bool)
                and math.isfinite(float(alternate_attempt_distance))
                and float(alternate_attempt_distance)
                >= JOINT_SOLUTION_DEDUP_DISTANCE
            ):
                continue
            return True
        if (
            block.get("stop_reason")
            == "complete_l5_pass_found_single_branch_exhaustive_fallback"
        ):
            branch_count = metrics.get("l5_joint_branch_pass_count")
            exhaustion = block.get("search_exhaustion")
            selected_id = selected_ids[0] if len(selected_ids) == 1 else None
            pass_attempts = [
                attempt
                for attempt in l5_attempts
                if isinstance(attempt, Mapping)
                and attempt.get("candidate_id") == selected_id
                and attempt.get("verdict") == "PASS"
            ]
            if not (
                l5_pass_count == 1
                and branch_count == 1
                and isinstance(selected_id, str)
                and selected_id
                and len(pass_attempts) == 1
                and isinstance(exhaustion, Mapping)
                and exhaustion.get("fast_pool_exhausted") is True
                and exhaustion.get("recovery_pool_exhausted") is True
                and exhaustion.get("redundancy_degraded") is True
                and exhaustion.get("published_grasp_branch_count") == 1
                and exhaustion.get("fast_wave_count_completed")
                == exhaustion.get("fast_wave_count_expected")
                and exhaustion.get("recovery_wave_count_completed")
                == exhaustion.get("recovery_wave_count_expected")
                and block.get("infrastructure_error") is not True
            ):
                continue
            return True
        if not (
            l5_pass_count >= 2
            and len(selected_ids) == 2
            and len(set(selected_ids)) == len(selected_ids)
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
    """Accept the legacy cap or a strictly validated v3 grasp proof."""

    legacy_bound = any(
        1 <= value <= 2
        for value in base._values(call, "full_plan_submitted_count")
        if isinstance(value, int) and not isinstance(value, bool)
    )
    return legacy_bound or _has_v3_grasp_search_evidence(call, artifact_root=artifact_root)


def _has_resumable_frozen_pair_evidence(call: Mapping[str, Any]) -> bool:
    """Require one complete pair plus a model-free physical-failure frontier."""

    outputs = _call_outputs(call)
    grasps = outputs.get("grasp_candidates")
    grasp_ids = [
        str(candidate.get("id") or "")
        for candidate in grasps
        if isinstance(candidate, Mapping) and str(candidate.get("id") or "")
    ] if isinstance(grasps, list) else []
    return bool(
        outputs.get("frozen_pair_execution_target") == 1
        and outputs.get("frozen_pair_stop_reason") == "complete_pair_found"
        and outputs.get("frozen_pair_recovery_policy")
        == "resume_frozen_frontier_after_execution_failure"
        and outputs.get("frozen_pair_qualified_grasp_count") == 1
        and outputs.get("candidate_count") == 1
        and len(grasp_ids) == 1
        and isinstance(outputs.get("frozen_grasp_frontier_remaining_count"), int)
        and not isinstance(outputs.get("frozen_grasp_frontier_remaining_count"), bool)
        and outputs.get("frozen_grasp_frontier_remaining_count") > 0
        and outputs.get("frozen_grasp_frontier_model_inference_invoked") is False
    )


def verify_case(
    paths: base.CasePaths,
    *,
    scenario: str = "normal",
    grasp_backend: str = DEFAULT_GRASP_BACKEND,
    execution_profile: str = DEFAULT_EXECUTION_PROFILE,
    qualification_profile: str = DEFAULT_QUALIFICATION_PROFILE,
    task_variant: str = DEFAULT_MULTI_NORMAL_TASK_VARIANT,
    operator_mode: str = DEFAULT_OPERATOR_MODE,
) -> dict[str, Any]:
    backend = _validated_grasp_backend(grasp_backend)
    mode = _validated_operator_mode(operator_mode)
    profile = _validated_execution_profile(execution_profile)
    funnel_profile = _validated_qualification_profile(qualification_profile)
    variant = _validated_task_variant(scenario, task_variant)
    scene = _scene_contract(scenario)
    assignments = _expected_work_order(
        scene,
        scenario=scenario,
        task_variant=variant,
    )
    assignment_count = len(assignments)
    backend_label = "GraspGenX" if backend == "graspgenx" else "AnyGrasp"
    errors: list[str] = []
    try:
        events, trace_paths = base._load_trace_events(paths.trace_root)
        calls = base._tool_calls(events)
        errors.extend(base._base_errors(paths, events))
        receipt = base._json_load(paths.receipt)
        receipt_scene = (
            receipt.get("acceptance_scene") if isinstance(receipt, Mapping) else None
        )
        receipt_operator_mode = (
            receipt.get("operator_mode") if isinstance(receipt, Mapping) else None
        )
        if receipt_operator_mode is not None and receipt_operator_mode != mode:
            errors.append("operator mode receipt does not match the requested TUI mode")
        expected_scene_receipt = _scene_receipt(
            scene,
            scenario=scenario,
            task_variant=variant,
        )
        compatible_scene_receipts = [expected_scene_receipt]
        if scenario == "multi_normal":
            legacy_scene_receipt = dict(expected_scene_receipt)
            legacy_scene_receipt["acceptance_request_id"] = (
                LEGACY_MULTI_NORMAL_REQUEST_IDS[variant]
            )
            compatible_scene_receipts.append(legacy_scene_receipt)
        if (
            not isinstance(receipt_scene, Mapping)
            or receipt_scene not in compatible_scene_receipts
        ):
            errors.append("acceptance scene receipt does not match the versioned contract")
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
        names = [_name(call) for call in calls]
        observed_simulator_tools = frozenset(
            name for name in names if name in base.SIX_SIMULATOR_TOOLS
        )
        payloads, mcp_errors = base._mcp_response_payloads(
            calls,
            paths,
            required_tools=observed_simulator_tools,
        )
        errors.extend(mcp_errors)
        for required in _required_tools_for_backend(backend, scenario=scenario):
            if required not in names:
                errors.append(f"required real pick-place tool call missing: {required}")
        if scenario in MULTI_SORT_SCENARIOS:
            work_order_calls = [
                call for call in calls if _name(call) == "configure_work_order"
            ]
            if len(work_order_calls) != 1:
                errors.append("VLM must configure exactly one session work order")
            normalized_orders = [
                value
                for call in work_order_calls
                for value in base._values(call, "work_order")
                if isinstance(value, Mapping)
                and value.get("schema_version") == "openeta.work_order.v1"
                and value.get("source") == "vlm_tool_call"
            ]
            expected_items = [
                {**assignment, "source": "vlm_work_order"}
                for assignment in assignments
            ]
            if not any(order.get("items") == expected_items for order in normalized_orders):
                errors.append(
                    "VLM-authored work order does not match the user-requested task"
                )
            if base._contains(create := next(
                (call for call in calls if _name(call) == "create_simulator_env"),
                {},
            ), "sort_assignments"):
                errors.append("physical scene injected a static sort assignment")
        if names.count("sam3") != 2 * assignment_count:
            errors.append(
                "exactly one target-object and one placement-region SAM3 call "
                "are required per sort assignment"
            )
        sam3_prompts = [
            str(_parameters(call).get("prompt") or "")
            for call in calls
            if _name(call) == "sam3"
        ]
        expected_prompts = [
            prompt
            for assignment in assignments
            for prompt in (
                assignment.get("target_perception_prompt")
                or assignment["target_prompt"],
                assignment.get("placement_region_perception_prompt")
                or assignment["placement_region_prompt"],
            )
        ]
        if sam3_prompts != expected_prompts:
            errors.append("SAM3 semantic prompts do not match the selected scene task")
        grasp_indices = [index for index, name in enumerate(names) if name == backend]
        sam_indices = [index for index, name in enumerate(names) if name == "sam3"]
        if (
            assignment_count == 1
            and grasp_indices
            and sam_indices
            and max(sam_indices) > min(grasp_indices)
        ):
            errors.append("SAM3 was rerun after the cached grasp/placement funnel started")
        if not _ordered_assignment_execution(
            calls,
            assignments,
            backend=backend,
        ):
            errors.append("model-contact, attach, frozen-goal release order is invalid")
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
        if not base._contains(create, "scene_id", str(scene["world_scene"])):
            errors.append("created Gazebo environment lacks the selected scene identity")
        if not base._contains(create, "contract_sha256", str(scene["contract_sha256"])):
            errors.append("created Gazebo environment lacks the selected scene hash")
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
        if len(provider_inference_calls) != assignment_count:
            errors.append(
                f"scenario requires exactly one {backend_label} model inference "
                "per sort assignment"
            )
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
        has_v3_search = any(
            _has_v3_grasp_search_evidence(grasp_call, artifact_root=paths.root)
            for grasp_call in grasp_calls
        )
        if not has_legacy_diversity_pool and not has_v3_search:
            errors.append(f"{backend_label} validated grasp search evidence is missing")
        if not all(
            _has_bounded_grasp_l5_evidence(grasp_call, artifact_root=paths.root)
            for grasp_call in provider_inference_calls
        ):
            errors.append(f"{backend_label} L5 diversity evidence is missing per assignment")
        if funnel_profile == "fast_v3" and not all(
            _has_resumable_frozen_pair_evidence(grasp_call)
            for grasp_call in provider_inference_calls
        ):
            errors.append(
                f"{backend_label} fast-v3 resumable grasp/place evidence is missing"
            )
        anyplace_calls = [call for call in calls if _name(call) == "anyplace"]
        anyplace = anyplace_calls[-1] if anyplace_calls else {}
        anyplace_inference_calls = _anyplace_model_inference_calls(anyplace_calls)
        anyplace_requalification_calls = [
            call
            for call in anyplace_calls
            if base._contains(call, "anyplace_model_inference_invoked", False)
        ]
        first_anyplace = anyplace_inference_calls[0] if anyplace_inference_calls else {}
        anyplace_outputs = _call_outputs(anyplace)
        if len(anyplace_calls) < 2 * assignment_count:
            errors.append(
                "each sort assignment requires one model AnyPlace call and at "
                "least one frozen-pool requalification"
            )
        if len(anyplace_inference_calls) != assignment_count:
            errors.append("AnyPlace model inference count does not match assignments")
        if len(anyplace_requalification_calls) < assignment_count:
            errors.append("AnyPlace frozen-pool requalification is missing per assignment")
        qualification_outputs = [
            _call_outputs(call)
            for call in [*grasp_calls, *anyplace_requalification_calls]
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
        for requalification_index, requalification in enumerate(
            anyplace_requalification_calls,
            start=1,
        ):
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
        if not candidate_count_valid or candidate_count < 1:
            errors.append("final AnyPlace result stored no MoveIt PASS candidate")
        # The best-first funnel may prove more than one L5 backup inside a
        # wave while exposing only its deterministic queue head to the agent.
        # Requiring equality was the old breadth-funnel contract and falsely
        # rejects a successful resumed frozen frontier.  Every exposed
        # candidate must still come from the proven set, so the public count
        # may be smaller, never larger.
        if not _candidate_pass_counts_consistent(
            candidate_count, full_plan_pass_count
        ):
            errors.append(
                "AnyPlace exposed candidate count is inconsistent with full-plan PASS count"
            )
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
        release_sequences = [
            value
            for payload in payloads
            for value in base._values(payload, "release_sequence")
            if isinstance(value, list)
        ]
        if not any(
            isinstance(proof := ordered_native_release_proof(sequence), dict)
            and isinstance(proof.get("planning_scene_detach_ack"), dict)
            and isinstance(proof.get("gripper_open_completed"), dict)
            for sequence in release_sequences
        ):
            errors.append("detach-before-open ordered release evidence missing")
        release_evidence = [
            value
            for payload in payloads
            for value in base._values(payload, "release_evidence")
            if isinstance(value, Mapping)
        ]
        visually_reviewable_releases = [
            value
            for value in release_evidence
            if value.get("schema_version")
            == "openeta.native_release_evidence.v1"
            and value.get("detached_confirmed") is True
            and value.get("gripper_open_confirmed") is True
            and isinstance(value.get("post_release_visual_observation"), Mapping)
            and value["post_release_visual_observation"].get("available") is True
            and value["post_release_visual_observation"].get("review_authority")
            == "vlm"
        ]
        if len(visually_reviewable_releases) < assignment_count:
            errors.append(
                "native release plus causal VLM observation is missing per assignment"
            )
        if assignment_count > 1:
            final_progress = next(
                (
                    value
                    for payload in reversed(payloads)
                    for value in base._values(payload, "multi_sort_progress")
                    if isinstance(value, Mapping)
                    and value.get("schema_version") == "openeta.multi_sort_progress.v1"
                    and value.get("all_completed") is True
                ),
                None,
            )
            expected_assignment_ids = [assignment["id"] for assignment in assignments]
            if not (
                isinstance(final_progress, Mapping)
                and final_progress.get("assignment_count") == assignment_count
                and final_progress.get("remaining_count") == 0
                and final_progress.get("completed_assignment_ids")
                == expected_assignment_ids
                and final_progress.get("same_environment_session") is True
            ):
                errors.append("same-session multi-sort completion evidence is missing")
            observed_target_ids = {
                str(binding.get("target_id") or "")
                for payload in payloads
                for binding in base._values(payload, "native_target_binding")
                if isinstance(binding, Mapping)
            }
            if not {
                assignment["target_object_id"] for assignment in assignments
            } <= observed_target_ids:
                errors.append("native target binding evidence is missing per assignment")
        if not visually_reviewable_releases and assignment_count == 1:
            errors.append("native release plus causal VLM observation is missing")
        fingerprints = [
            str(value) for value in base._values(events, "request_fingerprint") if value
        ]
        if _repeated_failed_motion_fingerprints(calls):
            errors.append("a failed motion request fingerprint was repeated")
        if not fingerprints:
            errors.append("motion request fingerprint evidence missing")
        scene_revisions = [
            int(value)
            for value in base._values(payloads, "planning_scene_revision")
            if isinstance(value, int)
        ]
        if len(set(scene_revisions)) < 2 * assignment_count + 1:
            errors.append("reset/attach/detach planning-scene revision chain missing")
        for call in calls:
            if _name(call) not in base.MUTATING_TOOLS:
                continue
            approved = (
                base._scripted_approved(call)
                if mode == base.SCRIPTED_TUI
                else base._human_approved(call)
            )
            if not approved:
                errors.append(f"{_name(call)} lacks {mode} approval")
        findings = list(dict.fromkeys(errors))
        flow_diagnostics = [
            finding
            for finding in findings
            if _is_non_blocking_flow_finding(finding)
        ]
        blocking_errors = [
            finding
            for finding in findings
            if not _is_non_blocking_flow_finding(finding)
        ]
        status = "failed" if blocking_errors else "passed"
        return {
            "status": status,
            "errors": blocking_errors,
            "flow_diagnostics": flow_diagnostics,
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
            "task_variant": variant if scenario in MULTI_SORT_SCENARIOS else None,
            "operator_mode": mode,
            "acceptance_scene": _scene_receipt(
                scene,
                scenario=scenario,
                task_variant=variant,
            ),
            "grasp_backend": backend,
        }
    except Exception as exc:  # noqa: BLE001 - verifier must return a report.
        return {
            "status": "blocked",
            "errors": [f"evidence unreadable: {type(exc).__name__}: {exc}"],
            "flow_diagnostics": [],
            "trace_paths": [],
            "tool_call_count": 0,
            "planner_evidence": {},
            "execution_profile": profile,
            "qualification_profile": funnel_profile,
            "acceptance_scope": EXECUTION_PROFILE_SCOPES[profile],
            "planner_provider_invoked": None,
            "scenario": scenario,
            "task_variant": variant if scenario in MULTI_SORT_SCENARIOS else None,
            "operator_mode": mode,
            "acceptance_scene": _scene_receipt(
                scene,
                scenario=scenario,
                task_variant=variant,
            ),
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
        "--task-variant",
        choices=tuple(MULTI_NORMAL_TASK_VARIANTS),
        default=DEFAULT_MULTI_NORMAL_TASK_VARIANT,
        help=(
            "Private verification fixture for varied human prompts in a "
            "task-neutral multi-sort physical scene."
        ),
    )
    parser.add_argument(
        "--operator-mode",
        choices=OPERATOR_MODES,
        default=DEFAULT_OPERATOR_MODE,
        help=(
            "scripted_tui drives the real PTY for repeatable acceptance; human_tui "
            "waits for an operator prompt and explicit mutation approvals."
        ),
    )
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
    try:
        task_variant = _validated_task_variant(args.scenario, args.task_variant)
    except ValueError as exc:
        raise base.AcceptanceError(str(exc)) from exc
    if args.execution_profile == "smoke_normal" and args.scenario != "normal":
        raise base.AcceptanceError("--execution-profile smoke_normal requires --scenario normal")
    if args.execution_profile == "smoke_normal" and args.operator_mode != base.SCRIPTED_TUI:
        raise base.AcceptanceError("--execution-profile smoke_normal requires scripted_tui")
    repo = Path(__file__).resolve().parents[1]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = (
        Path(args.run_root).resolve()
        if args.run_root
        else repo / ".cache/reports" / f"pick-place-gazebo-{stamp}"
    )
    paths = base.case_paths(run_root, SUITE, args.operator_mode)
    if args.verify_only:
        report = verify_case(
            paths,
            scenario=args.scenario,
            grasp_backend=args.grasp_backend,
            execution_profile=args.execution_profile,
            qualification_profile=args.qualification_profile,
            task_variant=task_variant,
            operator_mode=args.operator_mode,
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
    allocation = base.allocate(
        f"{SUITE}-{args.operator_mode}",
        preflight=not args.prepare_only,
    )
    paths = prepare_case(
        repo,
        run_root,
        allocation,
        services,
        scenario=args.scenario,
        grasp_backend=args.grasp_backend,
        execution_profile=args.execution_profile,
        qualification_profile=args.qualification_profile,
        task_variant=task_variant,
        operator_mode=args.operator_mode,
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
            "OPENETA_GRASP_BACKEND": args.grasp_backend,
            "OPENETA_QUALIFICATION_PROFILE": args.qualification_profile,
            (
                "OPENETA_SCRIPTED_TASK_METADATA"
                if args.operator_mode == base.SCRIPTED_TUI
                else "OPENETA_OPERATOR_TASK_METADATA"
            ): _automation_metadata_for_backend(
                args.grasp_backend,
                execution_profile=args.execution_profile,
                qualification_profile=args.qualification_profile,
                scenario=args.scenario,
                task_variant=task_variant,
                operator_mode=args.operator_mode,
            ),
            # A cold or shared-GPU launch must still leave time to prove the
            # stock DetachableJoint endpoints after world-control discovery.
            # This affects startup only; motion, planning, execution, and
            # verification deadlines stay unchanged.  Preserve an explicit
            # operator/deployment override when one is supplied.
            "OPENETA_GAZEBO_STARTUP_TIMEOUT_S": os.environ.get(
                "OPENETA_GAZEBO_STARTUP_TIMEOUT_S",
                f"{DEFAULT_GAZEBO_ACCEPTANCE_STARTUP_TIMEOUT_S:g}",
            ),
            # Share the deployment runtime's bounded perception budget.  GPU
            # contention may make a healthy cold inference much slower than
            # its usual latency; qualification, IK, planning, and execution
            # retain their own independent deadlines.
            "OPENETA_PERCEPTION_RPC_TIMEOUT_S": f"{DEFAULT_PERCEPTION_RPC_TIMEOUT_S:g}",
            **_scenario_environment(args.scenario),
        },
    )
    report = verify_case(
        paths,
        scenario=args.scenario,
        grasp_backend=args.grasp_backend,
        execution_profile=args.execution_profile,
        qualification_profile=args.qualification_profile,
        task_variant=task_variant,
        operator_mode=args.operator_mode,
    )
    report.update({"schema_version": SCHEMA_VERSION, "run_root": str(run_root.resolve())})
    base._json_dump(run_root / "acceptance-report.json", report, exclusive=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if code == 0 and report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
