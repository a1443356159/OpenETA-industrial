#!/usr/bin/env python3
"""Launch a task-neutral, VLM-authored multi-object Gazebo sorting session.

This is intentionally separate from ``normal_gazebo_acceptance.py``.  The
latter records a fixed private acceptance fixture; this launcher creates the
same isolated physical workcell but leaves object selection, ordering, grouping
and destination choice to the operator's natural-language TUI request.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.runtime.calibration_registry import RM75_ROBOTIQ_GRASP_CALIBRATION_PROFILE
from agent.runtime.runtime_assembly import DEFAULT_PERCEPTION_RPC_TIMEOUT_S
from extensions.gazebo.native_grasp import load_acceptance_scene_contract
from scripts import gazebo_acceptance_runtime as runtime
from scripts import normal_gazebo_acceptance as acceptance


SCHEMA_VERSION = "openeta.operator_sort_session.v1"
SUITE = "open-sort"


def _scene(scenario: str) -> dict[str, Any]:
    if scenario not in acceptance.MULTI_SORT_SCENARIOS:
        raise ValueError("operator sorting requires a multi-object Gazebo scene")
    return load_acceptance_scene_contract(scenario)


def operator_instructions(*, scenario: str) -> str:
    """Return human-facing guidance without a hidden object-to-bin assignment."""

    _scene(scenario)
    return (
        "这是一个任务中立的连续分拣会话。Gazebo 场景已经准备好；请在 TUI 提示符处"
        "用自然语言说明要处理哪些物件，或要求系统按自己的有序原则整理工作台。\n"
        "\n"
        "系统会先观察 RGB-D 画面，再由视觉语言模型建立工单。对于开放式整理，模型会"
        "自行说明分类原则、覆盖授权目录中的所有待操作物件，并在同一环境中逐件完成。"
        "不要输入候选编号、坐标、关节值或人为偏移；世界状态变更时只确认与当前画面"
        "一致的动作。输入 /quit 才会关闭本次会话。"
    )


def operator_metadata(*, qualification_profile: str) -> str:
    """Bind provenance and safety profile without injecting task semantics."""

    profile = str(qualification_profile).strip()
    if profile not in acceptance.QUALIFICATION_PROFILES:
        raise ValueError(f"unsupported qualification profile: {profile}")
    return (
        "[operator=human_tui; planner_mode=agentic_closed_loop; "
        f"execution_profile=agentic_normal; qualification_profile={profile}; "
        "session_kind=task_neutral_open_sort; work_order_source=vlm_conversation]"
    )


def prepare_operator_session(
    repo: Path,
    run_root: Path,
    allocation: runtime.Allocation,
    services: Mapping[str, str],
    *,
    scenario: str,
    grasp_backend: str,
    qualification_profile: str,
) -> runtime.CasePaths:
    """Materialize only launch provenance, never a fixture work order."""

    scene = _scene(scenario)
    backend = acceptance._validated_grasp_backend(grasp_backend)
    profile = acceptance._validated_qualification_profile(qualification_profile)
    paths = runtime.case_paths(run_root, SUITE, runtime.HUMAN_TUI)
    paths.root.mkdir(parents=True, exist_ok=False)
    runtime._json_dump(
        paths.mcp_config,
        {
            "mcpServers": {
                "openeta-sim": {"url": f"http://127.0.0.1:{allocation.port}/mcp"},
                **{name: {"url": url} for name, url in services.items()},
            }
        },
    )
    paths.instructions.write_text(
        operator_instructions(scenario=scenario), encoding="utf-8"
    )
    receipt = runtime.environment_receipt(
        repo,
        allocation,
        case_name=f"{SUITE}-{runtime.HUMAN_TUI}",
        before=runtime._process_snapshot(),
    )
    # Scene identity is needed to reproduce the physical cell.  The catalog is
    # intentionally not copied as an expected assignment: the live MCP reset
    # publishes it to the VLM, which authors the actual work order.
    receipt.update(
        {
            "operator_session": True,
            "operator_task_source": "human_tui_vlm_authored",
            "scene": {
                "id": str(scene["scene_id"]),
                "seed": int(scene["seed"]),
                "contract_sha256": str(scene["contract_sha256"]),
            },
            "grasp_backend_mode": backend,
            "qualification_profile": profile,
            "planner_mode": "agentic_closed_loop",
            "static_work_order_injected": False,
        }
    )
    runtime._json_dump(paths.receipt, runtime.seal_environment_receipt(receipt))
    return paths


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = runtime._json_load(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _walk(value: Any):
    """Yield nested trace values without assuming one trace event schema."""

    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _work_order_progress_summary(trace_root: Path) -> dict[str, Any] | None:
    """Read the latest host-owned work-order progress from this session only."""

    latest: Mapping[str, Any] | None = None
    for trace_path in sorted(
        trace_root.glob("sessions/*/trace.jsonl"),
        key=lambda path: path.stat().st_mtime_ns,
    ):
        try:
            lines = trace_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            for value in _walk(event):
                if not isinstance(value, Mapping):
                    continue
                if value.get("schema_version") != "openeta.multi_sort_progress.v1":
                    continue
                latest = value
    if latest is None:
        return None
    try:
        assignment_count = int(latest["assignment_count"])
        completed_count = int(latest["completed_count"])
        remaining_count = int(latest["remaining_count"])
    except (KeyError, TypeError, ValueError):
        return None
    all_completed = latest.get("all_completed")
    if (
        isinstance(all_completed, bool)
        and assignment_count >= 1
        and completed_count >= 0
        and remaining_count >= 0
        and completed_count + remaining_count == assignment_count
        and all_completed == (remaining_count == 0)
    ):
        work_order = latest.get("work_order")
        return {
            "configured": True,
            "all_completed": all_completed,
            "assignment_count": assignment_count,
            "completed_count": completed_count,
            "remaining_count": remaining_count,
            "selection_scope": (
                str(work_order.get("selection_scope") or "")
                if isinstance(work_order, Mapping)
                else ""
            ),
        }
    return None


def session_report(
    paths: runtime.CasePaths,
    *,
    run_root: Path,
    scenario: str,
    tui_exit_code: int,
) -> dict[str, Any]:
    """Describe closure honestly; task outcome remains runtime/VLM evidence."""

    lifecycle = _read_json(paths.root / "host-simulator-lifecycle.json")
    cleanup = _read_json(paths.root / "cleanup.json")
    progress = _work_order_progress_summary(paths.trace_root)
    work_order_outcome = (
        "completed"
        if progress is not None and progress["all_completed"] is True
        else "incomplete"
        if progress is not None
        else "not_configured_or_unavailable"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "closed" if tui_exit_code == 0 else "tui_exited_nonzero",
        "run_root": str(run_root.resolve()),
        "scenario": scenario,
        "operator_task_source": "human_tui_vlm_authored",
        "static_work_order_injected": False,
        "formal_acceptance_verifier": "not_run",
        "tui_exit_code": int(tui_exit_code),
        "work_order_outcome": work_order_outcome,
        "multi_sort_progress": progress,
        "host_environment_closed": lifecycle.get("status") == "closed",
        "host_environment_close_proven": lifecycle.get("closed") is True,
        "cleanup_port_free": cleanup.get("port_free") is True,
        "note": (
            "A closed session is not a fixed acceptance PASS. work_order_outcome "
            "summarizes the host-owned multi_sort_progress for the VLM-authored task."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="")
    parser.add_argument(
        "--scenario",
        choices=acceptance.MULTI_SORT_SCENARIOS,
        default="multi_normal",
        help="Task-neutral multi-object physical scene to launch.",
    )
    parser.add_argument(
        "--grasp-backend",
        choices=acceptance.GRASP_BACKENDS,
        default=acceptance.DEFAULT_GRASP_BACKEND,
        help="Grasp provider registered for this operator session.",
    )
    parser.add_argument(
        "--qualification-profile",
        choices=acceptance.QUALIFICATION_PROFILES,
        default=acceptance.DEFAULT_QUALIFICATION_PROFILE,
        help="MoveIt qualification profile; fast_v3 is the release default.",
    )
    parser.add_argument("--skip-provider-preflight", action="store_true")
    for name, url in acceptance.DEFAULT_SERVICE_URLS.items():
        parser.add_argument("--" + name.removeprefix("openeta-") + "-url", default=url)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    scene = _scene(args.scenario)
    backend = acceptance._validated_grasp_backend(args.grasp_backend)
    profile = acceptance._validated_qualification_profile(args.qualification_profile)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = (
        Path(args.run_root).resolve()
        if args.run_root
        else repo / ".cache/reports" / f"open-sort-human-{stamp}"
    )
    if run_root.exists():
        raise runtime.AcceptanceError(f"operator session run root already exists: {run_root}")
    gazebo_preflight = acceptance.gazebo_runtime_preflight(repo)
    if gazebo_preflight["status"] != "passed":
        print(json.dumps(gazebo_preflight, ensure_ascii=False, indent=2))
        return 2
    services = acceptance._services_for_backend(
        backend,
        sam3_url=args.sam3_url,
        anygrasp_url=args.anygrasp_url,
        anyplace_url=args.anyplace_url,
        graspgenx_url=args.graspgenx_url,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    service_preflight = acceptance.service_preflight(services)
    runtime._json_dump(run_root / "service-preflight.json", service_preflight)
    if service_preflight["status"] != "passed":
        print(json.dumps(service_preflight, ensure_ascii=False, indent=2))
        return 2
    if not args.skip_provider_preflight:
        provider_preflight = runtime._provider_preflight_result(repo)
        runtime._json_dump(
            run_root / runtime.PROVIDER_PREFLIGHT_FILENAME, provider_preflight
        )
        if provider_preflight["status"] != "passed":
            print(json.dumps(provider_preflight, ensure_ascii=False, indent=2))
            return 2

    allocation = runtime.allocate(f"{SUITE}-{runtime.HUMAN_TUI}", preflight=True)
    paths: runtime.CasePaths | None = None
    try:
        paths = prepare_operator_session(
            repo,
            run_root,
            allocation,
            services,
            scenario=args.scenario,
            grasp_backend=backend,
            qualification_profile=profile,
        )
        tui_exit_code = runtime.run_case(
            repo,
            paths,
            allocation,
            environment_config={
                "env_id": acceptance.ENV_ID,
                "seed": int(scene["seed"]),
                "render_mode": "rgb_array",
                "image_width": 512,
                "image_height": 512,
            },
            calibration_profile=RM75_ROBOTIQ_GRASP_CALIBRATION_PROFILE,
            extra_environment={
                "OPENETA_GRASP_BACKEND": backend,
                "OPENETA_QUALIFICATION_PROFILE": profile,
                "OPENETA_OPERATOR_TASK_METADATA": operator_metadata(
                    qualification_profile=profile
                ),
                "OPENETA_GAZEBO_STARTUP_TIMEOUT_S": os.environ.get(
                    "OPENETA_GAZEBO_STARTUP_TIMEOUT_S",
                    f"{acceptance.DEFAULT_GAZEBO_ACCEPTANCE_STARTUP_TIMEOUT_S:g}",
                ),
                "OPENETA_PERCEPTION_RPC_TIMEOUT_S": f"{DEFAULT_PERCEPTION_RPC_TIMEOUT_S:g}",
                "OPENETA_ACCEPTANCE_SCENE": str(scene["scene_id"]),
            },
        )
        report = session_report(
            paths,
            run_root=run_root,
            scenario=args.scenario,
            tui_exit_code=tui_exit_code,
        )
        runtime._json_dump(run_root / "operator-session-report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return tui_exit_code
    finally:
        runtime._release_mcp_listener(allocation)


if __name__ == "__main__":
    raise SystemExit(main())
