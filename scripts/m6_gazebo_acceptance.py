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


SCHEMA_VERSION = "openeta.gazebo_m6_acceptance.v1"
MILESTONE = "m6"
MODE = base.SCRIPTED_TUI
ENV_ID = "openeta/gazebo_rm75_robotiq2f85_pickplace-v0"
DEFAULT_SERVICES = {
    "openeta-sam3": "http://127.0.0.1:8773/sse",
    "openeta-anygrasp": "http://127.0.0.1:8774/sse",
    "openeta-anyplace": "http://127.0.0.1:8775/sse",
}
REQUIRED_REAL_M6_TOOLS = (
    "create_simulator_env",
    "observe",
    "sam3",
    "anygrasp",
    "gripper_control",
    "anyplace",
    "close_simulator_env",
)
SCENARIOS = ("normal", "reject-first")
GAZEBO_SIM_PACKAGE = "openeta_rm75_robotiq2f85_sim"


INSTRUCTIONS = """
[automation=scripted_tui] 在隔离 Gazebo pick-place 环境中完成一次真实约束放置验收。
创建 openeta/gazebo_rm75_robotiq2f85_pickplace-v0 后，先 observe 并用该抓取观察供 SAM3
分割红色方块 target_object，并让宿主通过统一 grasp_pose_estimate 调用真实 AnyGrasp；create 返回的
initial observation 不计作这次显式 observe。禁止 Oracle、
fake candidate、直接调用后端工具、固定抓法、固定腕姿、IK preview 或新增运动工具。不得调用 python_exec
读取或处理感知 artifact。SAM3 点提示若返回嵌套候选，主 VLM 必须根据候选图选择覆盖完整目标轮廓的 mask，
拒绝只覆盖单个表面或包含宽泛背景的 mask；不得固定 detection id。必须由主 VLM 选择
目标 mask；运动前再分割绿色 placement_zone_marker 并按宿主 obligation 调用一次 AnyPlace，形成
不向 VLM 暴露的 object-goal 池。宿主先完成普通 grasp 漏斗，再对最多 2 个 grasp PASS × 当前轮
完整 96 个 object goal 做分层有限联查，全局最多 2 对进入 plan-only；仅保留至少有一个 place PASS
的 grasp。宿主按稳定资格队列激活首个等价 PASS，并完成不可执行的内部
host_candidate_compilation grasp event，
随后主 VLM 执行真实接近、close，
仅在双垫 native contact 与 attached ACK 后 lift；lift 必须 >=80 mm 且抓持相对漂移 <=10 mm。
之后重新 observe 获取独立 placement RGB-D，分别用 SAM3 分割被抓物体和绿色
placement_zone_marker，再把两个独立观察交给 AnyPlace。AnyPlace 只输出物体目标位姿，禁止接收
selected_grasp/source_grasp_id 或输出 place_grasp_pose。模型 reserve pool 只进入宿主分层 MoveIt 漏斗；
宿主先取抓前随实际执行 grasp 通过 plan-only 的冻结绝对 object goals，使用 attach ACK 实测的
T_eef_object_attached 重新编译 EEF hover/release 并重跑完整资格漏斗；抓前轨迹不得作为执行证明。
若这些冻结目标零 PASS，才用新 seed 的当前轮 AnyPlace pool，且不与失败目标合并。主 VLM 只能看到 PASS
结果，不负责选择候选；宿主存储全部 PASS，按稳定队列激活首个等价 PASS，并生成内部
host_candidate_compilation placement event。
资格轨迹必须丢弃，真实
动作重新规划。规划失败仅当 execution_started=false 时拒绝当前候选，
不得重复失败 fingerprint；若 execution_started=true 或结果 unknown，立即停止并请求人工。
成功释放必须有 detach ACK、planning-scene revision；允许自然落稳观测时域完成后，仍只按最终
0.5 s 判断稳定且末段漂移 <=5 mm、中心高度
0.43±0.01 m，且目标 XY 外接圆完全位于标记区域。完成后唯一一次 close_simulator_env。
""".strip() + "\n"


SCENARIO_INSTRUCTIONS = {
    "normal": "执行正常放置路径；不得主动制造规划失败。",
    "reject-first": (
        "验收配置会让首个 placement candidate 的资格规划真实返回无轨迹且 "
        "execution_started=false。保留该回执，宿主不得把该候选存入资格队列；宿主从其余 "
        "PASS 候选的稳定队首开始并完成放置。"
    ),
}


def _health_url(sse_url: str) -> str:
    return sse_url.removesuffix("/sse").rstrip("/") + "/"


def service_preflight(services: Mapping[str, str]) -> dict[str, Any]:
    expected = {
        "openeta-sam3": "sam3",
        "openeta-anygrasp": "anygrasp",
        "openeta-anyplace": "anyplace",
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
                        payload.get("raw_pool_size")
                        == (96 if name == "openeta-anyplace" else 200)
                        and isinstance(payload.get("returned_candidate_count"), int)
                        and payload.get("returned_candidate_count") >= 0
                    )
                )
            )
            rows[name] = {
                "status": "passed" if ok else "failed",
                "url": url,
                "server": payload.get("server") if isinstance(payload, Mapping) else None,
                "model_loaded": payload.get("model_loaded") if isinstance(payload, Mapping) else None,
                "tools": payload.get("tools") if isinstance(payload, Mapping) else None,
                "returned_candidate_count": (
                    payload.get("returned_candidate_count")
                    if isinstance(payload, Mapping)
                    else None
                ),
                "raw_pool_size": (
                    payload.get("raw_pool_size")
                    if isinstance(payload, Mapping)
                    else None
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
    process. The canonical M6 wrapper sources them before exec; this check
    makes an accidental direct invocation fail immediately and precisely.
    """

    expected_overlay = Path(
        os.environ.get("OPENETA_GAZEBO_OVERLAY")
        or repo / "extensions/gazebo/ros2_ws/install"
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
        if (
            actual_prefix != expected_overlay
            and expected_overlay not in actual_prefix.parents
        ):
            errors.append("OPENETA_GAZEBO_OVERLAY_PACKAGE_MISMATCH")
    missing_commands = [
        name for name in ("ros2", "gz") if shutil.which(name) is None
    ]
    if missing_commands:
        errors.append("OPENETA_GAZEBO_COMMAND_UNAVAILABLE")
    return {
        "schema_version": "openeta.gazebo_runtime_preflight.v1",
        "status": "passed" if not errors else "blocked",
        "reason_codes": errors,
        "canonical_runner": str(
            (repo / "scripts/run_m6_gazebo_acceptance.sh").resolve()
        ),
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
) -> base.CasePaths:
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported M6 acceptance scenario: {scenario}")
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
        INSTRUCTIONS
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
    receipt["m6_scenario"] = scenario
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
            sorted(
                {
                    str(value)
                    for value in base._values(event, "request_fingerprint")
                    if value
                }
            )
        )
    counts = {fingerprint: failed.count(fingerprint) for fingerprint in set(failed)}
    return {fingerprint for fingerprint, count in counts.items() if count > 1}


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
        "pregrasp_joint_qualification_artifact",
    ):
        for artifact in base._values(call, field):
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


def verify_case(paths: base.CasePaths, *, scenario: str = "normal") -> dict[str, Any]:
    errors: list[str] = []
    try:
        events, trace_paths = base._load_trace_events(paths.trace_root)
        calls = base._tool_calls(events)
        errors.extend(base._base_errors(paths, events))
        payloads, mcp_errors = base._mcp_response_payloads(calls, paths)
        errors.extend(mcp_errors)
        names = [_name(call) for call in calls]
        for required in REQUIRED_REAL_M6_TOOLS:
            if required not in names:
                errors.append(f"required real M6 tool call missing: {required}")
        if names.count("sam3") < 4:
            errors.append("pregrasp and post-attachment object/region SAM3 calls are required")
        if not _ordered(
            names,
            (
                "observe",
                "anyplace",
                "anygrasp",
                "gripper_control",
                "move_to",
                "anyplace",
            ),
        ):
            errors.append("pregrasp look-ahead, grasp/lift, and executable placement order is invalid")
        if any(base._contains(event, "perception_source", "gazebo_oracle") for event in events):
            errors.append("Oracle perception is forbidden")
        if any(base._contains(event, "fake_grasp_candidate") for event in events):
            errors.append("fake candidate evidence is forbidden")
        if any(
            base._contains(event, "plan_only", True)
            and not base._contains(
                event, "schema_version", "openeta.moveit_candidate_funnel.v2"
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
        grasp_calls = [call for call in calls if _name(call) == "anygrasp"]
        final_grasp = grasp_calls[-1] if grasp_calls else {}
        raw_grasp_counts = [
            int(value)
            for value in base._values(final_grasp, "raw_candidate_count")
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        if not raw_grasp_counts or raw_grasp_counts[-1] < 10:
            errors.append("AnyGrasp raw candidate count evidence is missing")
        if not any(
            1 <= value <= 64
            for value in base._values(final_grasp, "diversity_selected_count")
            if isinstance(value, int) and not isinstance(value, bool)
        ):
            errors.append("AnyGrasp diversity pool evidence is missing")
        if not any(1 <= value <= 2 for value in base._values(final_grasp, "full_plan_submitted_count") if isinstance(value, int)):
            errors.append("AnyGrasp full-plan submission bound is missing")
        anyplace_calls = [call for call in calls if _name(call) == "anyplace"]
        anyplace = anyplace_calls[-1] if anyplace_calls else {}
        first_anyplace = anyplace_calls[0] if anyplace_calls else {}
        candidate_ids = {
            str(value)
            for value in base._values(anyplace, "id")
            if str(value).startswith("placement_")
        }
        if not _has_minimum_int_value(anyplace, "model_raw_candidate_count", 96):
            errors.append("AnyPlace model raw pool evidence is missing")
        if not any(1 <= value <= 2 for value in base._values(anyplace, "full_plan_submitted_count") if isinstance(value, int)):
            errors.append("AnyPlace full-plan submission bound is missing")
        candidate_counts = [
            int(value)
            for value in base._values(anyplace, "candidate_count")
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        full_plan_pass_counts = [
            int(value)
            for value in base._values(anyplace, "full_plan_pass_count")
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        if not candidate_counts or candidate_counts[-1] < 1:
            errors.append("final AnyPlace result stored no MoveIt PASS candidate")
        if (
            not full_plan_pass_counts
            or not candidate_counts
            or full_plan_pass_counts[-1] != candidate_counts[-1]
        ):
            errors.append("AnyPlace candidate_count/full_plan_pass_count mismatch")
        if not candidate_ids:
            errors.append("AnyPlace stored no MoveIt PASS placement candidate")
        for legacy_key in ("selected_grasp", "source_grasp_id", "place_grasp_pose"):
            if base._contains(anyplace, legacy_key):
                errors.append(f"AnyPlace leaked forbidden grasp-coupled field: {legacy_key}")
        if not base._contains(
            anyplace, "schema_version", "openeta.moveit_candidate_funnel.v2"
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
        # The bounded retry may use the host-normalized frozen packet
        # (mask/mask_artifact). Validate the independent public observations on
        # the initial inference call, before that internal replay boundary.
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
            any(float(value) >= 0.08 for value in base._values(payload, "lift_m") if isinstance(value, (int, float)))
            and any(
                float(value) <= 0.01
                for value in base._values(payload, "capture_relative_translation_m")
                if isinstance(value, (int, float))
            )
            for payload in payloads
        ):
            errors.append("80 mm lift / 10 mm grasp-drift proof missing")
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
        fingerprints = [str(value) for value in base._values(events, "request_fingerprint") if value]
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
        pregrasp_qualification_blocks = [
            block
            for block in _qualification_blocks(final_grasp, artifact_root=paths.root)
            if block.get("purpose") == "placement"
        ]
        qualification_results = [
            result
            for block in pregrasp_qualification_blocks
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
            "scenario": scenario,
        }
    except Exception as exc:  # noqa: BLE001 - verifier must return a report.
        return {
            "status": "blocked",
            "errors": [f"evidence unreadable: {type(exc).__name__}: {exc}"],
            "trace_paths": [],
            "tool_call_count": 0,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-provider-preflight", action="store_true")
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal")
    for name, url in DEFAULT_SERVICES.items():
        parser.add_argument("--" + name.removeprefix("openeta-") + "-url", default=url)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(args.run_root).resolve() if args.run_root else repo / ".cache/reports" / f"m6-gazebo-{stamp}"
    paths = base.case_paths(run_root, MILESTONE, MODE)
    if args.verify_only:
        report = verify_case(paths, scenario=args.scenario)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    runtime = gazebo_runtime_preflight(repo)
    if runtime["status"] != "passed":
        print(json.dumps(runtime, ensure_ascii=False, indent=2))
        return 2
    services = {
        "openeta-sam3": args.sam3_url,
        "openeta-anygrasp": args.anygrasp_url,
        "openeta-anyplace": args.anyplace_url,
    }
    run_root.mkdir(parents=True, exist_ok=False)
    preflight = service_preflight(services)
    base._json_dump(run_root / "service-preflight.json", preflight)
    if preflight["status"] != "passed":
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2
    if not args.skip_provider_preflight:
        provider = base._provider_preflight_result(repo)
        base._json_dump(run_root / base.PROVIDER_PREFLIGHT_FILENAME, provider)
        if provider["status"] != "passed":
            print(json.dumps(provider, ensure_ascii=False, indent=2))
            return 2
    allocation = base.allocate(f"{MILESTONE}-{MODE}", preflight=not args.prepare_only)
    paths = prepare_case(repo, run_root, allocation, services, scenario=args.scenario)
    if args.prepare_only:
        print(run_root)
        return 0
    code = base.run_case(
        repo,
        paths,
        allocation,
        calibration_profile=RM75_ROBOTIQ_GRASP_CALIBRATION_PROFILE,
        extra_environment={
            "OPENETA_EPISODE_MAX_TOTAL_TOKENS": "10000000",
            # Cold software-rendered Gazebo launches occasionally need more
            # than the deployment default before the documented world-control
            # service is discoverable.  This affects startup only; motion,
            # planning, execution, and verification deadlines stay unchanged.
            "OPENETA_GAZEBO_STARTUP_TIMEOUT_S": "90",
            **(
                {"OPENETA_ACCEPTANCE_PLACEMENT_FAULT": args.scenario}
                if args.scenario != "normal"
                else {}
            ),
        },
    )
    report = verify_case(paths, scenario=args.scenario)
    report.update({"schema_version": SCHEMA_VERSION, "run_root": str(run_root.resolve())})
    base._json_dump(run_root / "acceptance-report.json", report, exclusive=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if code == 0 and report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
