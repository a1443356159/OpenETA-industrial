from __future__ import annotations

import base64
import io
import json
import os
import signal
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agent.backends.planner import ProviderHttpError
from agent.backends.provider_config import PlannerProviderConfig, ProviderEndpointConfig
import scripts.tui_gazebo_acceptance as tui_acceptance
from agent.tools.registry import ToolResult, build_default_tool_registry
from extensions.gazebo.ros2_ws.acceptance_isolation import _normalise_graph_rows
from scripts.tui_gazebo_acceptance import (
    AUTONOMY,
    CONTROL_ONLY,
    CONTROL_REPORT_FILENAME,
    DETERMINISTIC,
    ENV_IDS,
    SCHEMA_VERSION,
    SCRIPTED_TUI,
    SIX_SIMULATOR_TOOLS,
    allocate,
    assemble_control_report,
    assemble_report,
    case_paths,
    environment_receipt,
    prepare_case,
    report_exit_code,
    run_case,
    scripted_tui_input,
    main,
    verify_case,
    verify_receipt,
)


ROOT = Path(__file__).parents[1]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _provider_config(*, vision: bool = False) -> PlannerProviderConfig:
    return PlannerProviderConfig(
        provider="unit-openai-compatible",
        model="unit-model",
        api_base="https://provider.example.test/v1?never=persisted",
        api_key="unit-provider-secret",
        timeout_s=5.0,
        max_attempts=4,
        retry_backoff_s=0.25,
        context_window_tokens=32000,
        max_tokens=256,
        fallback=ProviderEndpointConfig(
            provider="fallback-provider",
            model="fallback-model",
            api_base="https://fallback.example.test/v1",
            api_key="fallback-secret",
        ),
        metadata={"enable_vision": vision},
    )


def _tool(name: str, *, parameters=None, outputs=None, receipt=None, profile="human_gated"):
    return {
        "event_type": "action",
        "payload": {
            "command": {
                "tool_calls": [
                    {
                        "name": name,
                        "parameters": parameters or {},
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "supervision": {
                                    "allowed": True,
                                    "profile": profile,
                                    "details": {"profile": profile},
                                },
                                "outputs": outputs or {},
                                **({"environment_receipt": receipt} if receipt else {}),
                            },
                        },
                    }
                ]
            }
        },
    }


def _mcp_outputs(
    paths: Path,
    agent_tool: str,
    responses: list[tuple[str, dict]],
    *,
    handle: str = "test-handle",
    session_id: str = "test-session",
) -> tuple[dict, dict]:
    """Build the same per-RPC evidence contract formal TUI cases require."""

    entries = []
    for index, (remote_tool, payload) in enumerate(responses, 1):
        request_id = f"{agent_tool}-{index}-request"
        arguments = {"session_id": session_id}
        if remote_tool != "create_env":
            arguments["handle"] = handle
        response_path = paths / "responses" / f"{agent_tool}-{index}-{remote_tool}.json"
        _write_json(response_path, payload)
        response = {
            "response_path": str(response_path),
            "response_omitted": True,
            "request_id": request_id,
            "tool": remote_tool,
            "handle": handle,
            "session_id": session_id,
        }
        receipt = {
            "remote_tool": remote_tool,
            "mcp_request_id": request_id,
            "handle": handle,
            "simulator_session_id": session_id,
            "receipt_id": f"receipt-{request_id}",
        }
        entries.append(
            {
                "request": {
                    "request_id": request_id,
                    "tool": remote_tool,
                    "arguments": arguments,
                },
                "response": response,
                "environment_receipt": receipt,
            }
        )
    assert entries
    return (
        {
            "mcp": {
                "tool": entries[0]["request"]["tool"],
                "handle": handle,
                "session_id": session_id,
                "request": entries[0]["request"],
                "response": entries[0]["response"],
            },
            "response": entries[0]["response"],
            "mcp_calls": entries,
        },
        entries[-1]["environment_receipt"],
    )


def _prepare_evidence(tmp_path: Path, milestone: str, mode: str):
    allocation = allocate(f"{milestone}-{mode}")
    paths = prepare_case(ROOT, tmp_path, milestone, mode, allocation)
    _write_json(
        paths.root / "cleanup.json",
        {
            "mcp_group_exited": True,
            "port_free": True,
            "owned_worker_groups": [],
            "owned_worker_groups_exited": True,
            "owned_process_residuals": [],
            "preexisting_process_snapshot_unchanged": True,
            "protected_ros_graphs_unchanged": True,
            "ros_graph": {"state": "PASSED"},
            "gz_partition": {"state": "PASSED"},
        },
    )
    return paths


def _m1_materialized_observe(root: Path, *, timestamp_s: float) -> tuple[dict, Path]:
    """Build one compact observe result plus its durable RGB-D response."""

    image_root = root / "images"
    rgb_path = image_root / "top-rgb.png"
    depth_path = image_root / "top-depth.png"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_path.write_bytes(b"rgb")
    depth_path.write_bytes(b"depth")
    camera = {
        "frame_id": "top_camera",
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "scale": 1000.0},
        "extrinsics": {"frame_transform": "camera_to_world"},
        "timestamp_s": timestamp_s,
    }
    response_path = root / "responses" / "observe.json"
    _write_json(
        response_path,
        {"cameras": [camera], "metadata": {"observation_provenance": "gazebo_ros_live"}},
    )
    return (
        {
            "success": True,
            "details": {
                "outputs": {
                    "response": {
                        "response_path": str(response_path),
                        "cameras": [
                            {
                                "frame_id": "top_camera",
                                "rgb_ref": "top-camera.rgb",
                                "depth_ref": "top-camera.depth",
                                "intrinsics": dict(camera["intrinsics"]),
                                "extrinsics": dict(camera["extrinsics"]),
                                "timestamp_s": timestamp_s,
                            }
                        ],
                        "image_artifacts": [
                            {"index": "top-camera.rgb", "kind": "rgb", "path": str(rgb_path)},
                            {"index": "top-camera.depth", "kind": "depth", "path": str(depth_path)},
                        ],
                        "observation_provenance": "gazebo_ros_live",
                    }
                }
            },
        },
        response_path,
    )


def test_allocation_and_receipt_exclude_protected_domains() -> None:
    allocation = allocate("unit", occupied_domains={80, 81})
    assert allocation.ros_domain_id not in {42, 80, 81, 100}
    receipt = environment_receipt(
        ROOT, allocation, case_name="unit", before=[], capture_protected=False
    )
    assert verify_receipt(receipt) == []
    assert receipt["python_executable"] == str(Path(sys.executable).absolute())
    tampered = dict(receipt, ros_domain_id=42)
    assert verify_receipt(tampered)


def test_root_provider_resolution_preserves_precedence_and_never_copies_credentials(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text(
        "\n".join(
            (
                "OPENETA_LLM_PROVIDER=dotenv-provider",
                "OPENETA_LLM_MODEL=dotenv-model",
                "OPENETA_LLM_API_BASE=https://dotenv.example.test/v1",
                "OPENETA_LLM_API_KEY=dotenv-provider-secret",
                "OPENETA_LLM_ENABLE_VISION=false",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (repo / "apikey.md").write_text(
        '{"_type":"newapi_channel_conn","url":"https://apikey.example.test",'
        '"key":"apikey-provider-secret"}\n',
        encoding="utf-8",
    )

    config = tui_acceptance._root_provider_config(  # noqa: SLF001 - boundary contract.
        repo,
        environ={
            "OPENETA_LLM_MODEL": "environment-model",
            "OPENETA_LLM_API_KEY": "environment-provider-secret",
        },
    )
    child = tui_acceptance._resolved_provider_environment(config)  # noqa: SLF001

    assert child["OPENETA_LLM_PROVIDER"] == "dotenv-provider"
    assert child["OPENETA_LLM_MODEL"] == "environment-model"
    assert child["OPENETA_LLM_API_BASE"] == "https://dotenv.example.test/v1"
    assert child["OPENETA_LLM_API_KEY"] == "environment-provider-secret"
    assert child["OPENETA_LLM_ENABLE_VISION"] == "false"
    assert not any(key.startswith("OPENETA_LLM_FALLBACK_") for key in child)

    case = tmp_path / "case"
    case.mkdir()
    assert not (case / ".env").exists()
    assert not (case / "apikey.md").exists()


def test_provider_preflight_passes_only_a_primary_structured_response(monkeypatch) -> None:
    config = _provider_config(vision=False)
    seen = {}

    monkeypatch.setattr(tui_acceptance, "_root_provider_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(tui_acceptance, "list_openai_compatible_models", lambda _config: ["unit-model"])

    class FakeBackend:
        def __init__(self, backend_config) -> None:
            seen["backend_config"] = backend_config

        def decide(self, request):
            seen["request"] = request
            return SimpleNamespace(
                status=tui_acceptance.PipelineStatus.PLANNED,
                payload=(
                    '{"kind":"response","name":"ask_human",'
                    '"parameters":{"message":"ok"},"reasoning":"ok"}'
                ),
            )

    monkeypatch.setattr(tui_acceptance, "OpenAICompatiblePlannerBackend", FakeBackend)

    result = tui_acceptance._provider_preflight_result(Path("/unused"))  # noqa: SLF001

    assert result["status"] == "passed"
    assert result["endpoint_id"] == "https://provider.example.test"
    assert result["max_tokens"] == 256
    assert result["vision_enabled"] is False
    assert result["fallback_used"] is False
    assert result["planner_smoke"]["response_schema"] == "response/ask_human"
    assert seen["backend_config"].model == "unit-model"
    assert seen["backend_config"].fallback is None
    assert seen["backend_config"].max_attempts == 1
    assert seen["request"].tool_context["available_tools"] == []
    assert "unit-provider-secret" not in json.dumps(result)
    assert "never=persisted" not in json.dumps(result)


@pytest.mark.parametrize(
    ("models", "error", "expected_status", "reason_code"),
    [
        (None, None, "blocked", "PROVIDER_CONFIG_MISSING"),
        ([], None, "failed", "PROVIDER_MODEL_NOT_FOUND"),
        (
            None,
            ProviderHttpError(401, "unit-provider-secret"),
            "blocked",
            "PROVIDER_AUTH_FAILED",
        ),
        (
            None,
            TimeoutError("unit-provider-secret"),
            "blocked",
            "PROVIDER_NETWORK_OR_TIMEOUT",
        ),
    ],
)
def test_provider_preflight_fails_closed_without_secret_evidence(
    monkeypatch,
    models,
    error,
    expected_status: str,
    reason_code: str,
) -> None:
    config = _provider_config()
    if models is None and error is None:
        config = PlannerProviderConfig()
    monkeypatch.setattr(tui_acceptance, "_root_provider_config", lambda *_args, **_kwargs: config)

    def list_models(_config):
        if error is not None:
            raise error
        return models

    monkeypatch.setattr(tui_acceptance, "list_openai_compatible_models", list_models)
    result = tui_acceptance._provider_preflight_result(Path("/unused"))  # noqa: SLF001

    assert result["status"] == expected_status
    assert result["reason_code"] == reason_code
    assert "unit-provider-secret" not in json.dumps(result)


def test_provider_preflight_rejects_non_structured_planner_response(monkeypatch) -> None:
    monkeypatch.setattr(
        tui_acceptance,
        "_root_provider_config",
        lambda *_args, **_kwargs: _provider_config(),
    )
    monkeypatch.setattr(tui_acceptance, "list_openai_compatible_models", lambda _config: ["unit-model"])

    class FakeBackend:
        def __init__(self, _config) -> None:
            pass

        def decide(self, _request):
            return SimpleNamespace(
                status=tui_acceptance.PipelineStatus.PLANNED,
                payload="this is not JSON",
            )

    monkeypatch.setattr(tui_acceptance, "OpenAICompatiblePlannerBackend", FakeBackend)
    result = tui_acceptance._provider_preflight_result(Path("/unused"))  # noqa: SLF001

    assert result["status"] == "failed"
    assert result["reason_code"] == "PROVIDER_STRUCTURED_RESPONSE_INCOMPATIBLE"


def test_provider_preflight_classifies_backend_wrapped_http_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        tui_acceptance,
        "_root_provider_config",
        lambda *_args, **_kwargs: _provider_config(),
    )
    monkeypatch.setattr(tui_acceptance, "list_openai_compatible_models", lambda _config: ["unit-model"])

    class FakeBackend:
        def __init__(self, _config) -> None:
            pass

        def decide(self, _request):
            return SimpleNamespace(
                status=tui_acceptance.PipelineStatus.FAILED,
                payload={
                    "kind": "response",
                    "name": "ask_human",
                    "parameters": {"message": "failed"},
                    "reasoning": "failed",
                },
                details={
                    "error_type": "ProviderHttpError",
                    "error": "HTTP 402 unit-provider-secret",
                },
            )

    monkeypatch.setattr(tui_acceptance, "OpenAICompatiblePlannerBackend", FakeBackend)
    result = tui_acceptance._provider_preflight_result(Path("/unused"))  # noqa: SLF001

    assert result["status"] == "blocked"
    assert result["reason_code"] == "PROVIDER_HTTP_402"
    assert result["error_type"] == "ProviderHttpError"
    assert "unit-provider-secret" not in json.dumps(result)


def test_provider_preflight_blocks_before_any_gazebo_or_mcp_case(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "scripted-provider-blocked"
    preflight = {
        "schema_version": "openeta.provider_preflight.v1",
        "status": "blocked",
        "provider": "unit-provider",
        "model": "unit-model",
        "endpoint_id": "https://provider.example.test",
        "reason_code": "PROVIDER_AUTH_FAILED",
        "model_list": {"status": "failed"},
        "planner_smoke": {"status": "not_run"},
    }
    monkeypatch.setattr(tui_acceptance, "_provider_preflight_result", lambda _repo: preflight)
    monkeypatch.setattr(
        tui_acceptance,
        "allocate",
        lambda *_args, **_kwargs: pytest.fail("a provider-blocked run must not allocate Gazebo"),
    )

    assert (
        tui_acceptance.main(
            [
                "--scripted-tui",
                "--provider-preflight",
                "--run-root",
                str(run_root),
            ]
        )
        == 2
    )
    assert not (run_root / "m0").exists()
    durable = json.loads((run_root / tui_acceptance.PROVIDER_PREFLIGHT_FILENAME).read_text())
    report = json.loads((run_root / "acceptance-report.json").read_text())
    assert durable["reason_code"] == "PROVIDER_AUTH_FAILED"
    assert report["overall_status"] == "inconclusive"
    assert report["acceptance_scope"] == "local_automated_scripted_tui_m0_m4"
    assert report["human_approval"] == "not_claimed"


def test_m5_cli_is_control_only_and_requires_an_external_sam3_url() -> None:
    with pytest.raises(tui_acceptance.AcceptanceError, match="valid only with --control-only"):
        tui_acceptance.main(["--include-m5", "--sam3-url", "http://sam3.example/sse"])
    with pytest.raises(tui_acceptance.AcceptanceError, match="requires --sam3-url"):
        tui_acceptance.main(["--control-only", "--include-m5", "--prepare-only"])
    assert (
        tui_acceptance._m5_endpoint_id(  # noqa: SLF001 - redaction is an acceptance contract.
            "https://token@example.invalid:9443/sse?credential=never-record"
        )
        == "https://example.invalid:9443"
    )


@pytest.mark.parametrize(
    ("outputs", "code"),
    [
        ({"detections": []}, "M5_ZERO_SAM3_CANDIDATES"),
        (
            {"detections": [{"id": "detection_000"}, {"id": "detection_001"}]},
            "M5_MULTIPLE_SAM3_CANDIDATES",
        ),
        (
            {
                "detections": [{"id": "detection_000"}],
                "perception_source": "gazebo_oracle",
            },
            "M5_ORACLE_OR_FAKE_CANDIDATE",
        ),
        (
            {
                "detections": [{"id": "detection_000"}],
                "fake_grasp_candidate": {"kind": "contractual_fake_grasp_candidate"},
            },
            "M5_ORACLE_OR_FAKE_CANDIDATE",
        ),
    ],
)
def test_m5_rejects_ambiguous_or_nonreal_candidates_before_m3_motion(outputs, code) -> None:
    with pytest.raises(tui_acceptance.M5FailedError) as error:
        tui_acceptance._m5_require_single_real_candidate(outputs)  # noqa: SLF001
    assert error.value.code == code


def test_m5_single_real_candidate_uses_existing_selection_then_gates_m3(
    tmp_path: Path, monkeypatch
) -> None:
    """The no-provider M5 path still uses real handler/selection contracts."""

    paths = _prepare_evidence(tmp_path, "m5", CONTROL_ONLY)
    rgb_path = paths.root / "mcp-images" / "rgb.png"
    depth_path = paths.root / "mcp-images" / "depth.png"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(rgb_path)
    Image.new("I;16", (4, 4), 2000).save(depth_path)
    response_path = paths.root / "mcp-responses" / "observe.json"
    _write_json(
        response_path,
        {
            "cameras": [
                {
                    "frame_id": "top_camera_optical_frame",
                    "role": "scene_primary",
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.5, "cy": 1.5, "scale": 1000.0},
                    "extrinsics": {
                        "frame_transform": "camera_to_world",
                        "camera_frame": "opencv",
                        "pos": [0.0, 0.0, 0.0],
                        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            ]
        },
    )
    observe = ToolResult(
        True,
        details={
            "outputs": {
                "response": {"response_path": str(response_path), "request_id": "observe-1", "tool": "render_env"},
                "mcp_calls": [],
            }
        },
    )
    mask = io.BytesIO()
    Image.new("L", (4, 4), 255).save(mask, format="PNG")
    encoded_mask = base64.b64encode(mask.getvalue()).decode("ascii")

    sam3_arguments: list[dict] = []

    class FakeSam3Transport:
        def __init__(self, _url: str) -> None:
            pass

        def list_tools(self, *, timeout_s=None):
            return {"tools": [{"name": "segment"}], "tool_count": 1}

        def call_tool(self, name, arguments, *, timeout_s=None):
            assert name == "segment"
            sam3_arguments.append(arguments)
            return {
                "success": True,
                "details": {
                    "detection_count": 1,
                    "metadata": {"api_key": "must-not-be-persisted"},
                    "detections": [
                        {
                            "label": "red rectangular block",
                            "score": 0.95,
                            "mask": {"format": "png", "base64": encoded_mask},
                        }
                    ],
                },
            }

    import agent.tools.handlers as handlers_module
    import agent.tools.sim_mcp as sim_mcp_module

    monkeypatch.setattr(handlers_module, "SseSimulatorMcpTransport", FakeSam3Transport)
    monkeypatch.setattr(sim_mcp_module, "SseSimulatorMcpTransport", FakeSam3Transport)

    class FakeRunner:
        def __init__(self) -> None:
            self.registry = build_default_tool_registry(perception_profile="sam3")
            self.metadata = {
                "execution_id": "control-m5",
                "session_id": "control-m5",
                "agent_session_id": "control-m5",
                "execution_profile": CONTROL_ONLY,
                "planner_invoked": False,
                "provider_invoked": False,
            }
            self.calls = []

        def require_success(self, name, parameters=None, *, observation=None, metadata=None):
            self.calls.append(name)
            if name == "observe":
                return observe
            return ToolResult(True, details={})

        def invoke(self, name, parameters=None, *, observation=None, metadata=None):
            self.calls.append(name)
            return self.registry.call(
                name,
                parameters or {},
                observation=observation,
                metadata={**self.metadata, **(metadata or {})},
            )

    runner = FakeRunner()
    monkeypatch.setattr(tui_acceptance, "_m3_motion", lambda _runner: [observe])

    tui_acceptance._run_m5_control(  # noqa: SLF001 - focused control-path contract.
        runner, paths=paths, sam3_url="http://sam3.example/sse?secret=not-recorded"
    )

    evidence = json.loads((paths.root / "m5-perception.json").read_text())
    assert evidence["status"] == "passed"
    assert evidence["sam3"]["endpoint_id"] == "http://sam3.example"
    assert evidence["selection"]["selection_source"] == "scripted_single_candidate"
    assert len(sam3_arguments) == 1
    assert sam3_arguments[0]["prompt"] == "red rectangular block"
    assert "move_to" not in runner.calls  # monkeypatched M3 invokes only after the gate.
    assert (paths.root / "m5-object-summary.json").is_file()
    assert "not-recorded" not in (paths.root / "m5-perception.json").read_text()
    sam3_response = evidence["sam3"]["request_response_artifacts"]["response"]["path"]
    assert "must-not-be-persisted" not in (paths.root / sam3_response).read_text()


def test_live_allocation_preflight_skips_a_busy_ros_domain(monkeypatch) -> None:
    """A stale external daemon must reserve its domain instead of failing cleanup."""

    seen = []

    def preflight(domain: int):
        seen.append(domain)
        return {"state": "FAILED", "reason_code": "ROS2CLI_DAEMON_PRESENT"} if domain == 80 else {
            "state": "PASSED", "reason_code": "ROS_DOMAIN_EMPTY"
        }

    monkeypatch.setattr(tui_acceptance, "_candidate_domain_preflight", preflight)
    allocation = tui_acceptance.allocate("live-unit", preflight=True)

    assert seen == [80, 81]
    assert allocation.ros_domain_id == 81
    assert allocation.candidate_domain_preflight == {
        "state": "PASSED", "reason_code": "ROS_DOMAIN_EMPTY"
    }
    receipt = environment_receipt(
        ROOT, allocation, case_name="live-unit", before=[], capture_protected=False
    )
    assert verify_receipt(receipt) == []


def test_live_allocation_preflight_fails_closed_when_no_domain_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(tui_acceptance, "DOMAIN_CANDIDATES", (80, 81))
    monkeypatch.setattr(
        tui_acceptance,
        "_candidate_domain_preflight",
        lambda _domain: {"state": "FAILED", "reason_code": "ROS_DOMAIN_NOT_EMPTY"},
    )

    with pytest.raises(tui_acceptance.AcceptanceError, match="after preflight"):
        tui_acceptance.allocate("live-unit", preflight=True)


def test_protected_graph_rows_are_json_native_before_baseline_comparison() -> None:
    """An unchanged rclpy tuple row must equal its JSON-loaded list form."""

    live = _normalise_graph_rows(
        [("/parameter_events", ["rcl_interfaces/msg/ParameterEvent"])]
    )
    persisted = [["/parameter_events", ["rcl_interfaces/msg/ParameterEvent"]]]

    assert live == persisted


def test_m1_verifier_accepts_paired_materialized_rgbd_refs(tmp_path: Path) -> None:
    first, _ = _m1_materialized_observe(tmp_path / "first", timestamp_s=1.0)
    second, _ = _m1_materialized_observe(tmp_path / "second", timestamp_s=2.0)
    calls = [
        {"name": "observe", "result": first},
        {"name": "observe", "result": second},
    ]

    assert tui_acceptance._verify_m1(calls, SimpleNamespace(root=tmp_path)) == []


def test_m1_verifier_rejects_missing_mismatched_or_nonexistent_rgbd_artifacts(
    tmp_path: Path,
) -> None:
    missing, _ = _m1_materialized_observe(tmp_path / "missing", timestamp_s=1.0)
    missing_response = missing["details"]["outputs"]["response"]
    missing_response["image_artifacts"] = missing_response["image_artifacts"][:1]
    frames, errors = tui_acceptance._m1_camera_frames(missing, root=tmp_path)
    assert not frames and any("missing" in error for error in errors)

    mismatched, response_path = _m1_materialized_observe(tmp_path / "mismatched", timestamp_s=1.0)
    durable = json.loads(response_path.read_text(encoding="utf-8"))
    other_depth = tmp_path / "mismatched" / "images" / "other-depth.png"
    other_depth.write_bytes(b"other-depth")
    durable["cameras"][0]["depth_path"] = str(other_depth)
    _write_json(response_path, durable)
    frames, errors = tui_acceptance._m1_camera_frames(mismatched, root=tmp_path)
    assert not frames and any("does not match" in error for error in errors)

    nonexistent, _ = _m1_materialized_observe(tmp_path / "nonexistent", timestamp_s=1.0)
    nonexistent["details"]["outputs"]["response"]["image_artifacts"][0]["path"] = str(
        tmp_path / "nonexistent" / "images" / "absent.png"
    )
    frames, errors = tui_acceptance._m1_camera_frames(nonexistent, root=tmp_path)
    assert not frames and any("nonlocal" in error for error in errors)


def test_cleanup_waits_briefly_for_mcp_listener_release(monkeypatch) -> None:
    """A just-reaped process must not make cleanup race its socket teardown."""

    checks = iter((False, False, True))
    monkeypatch.setattr(tui_acceptance, "_port_is_free", lambda _port: next(checks))
    monkeypatch.setattr(tui_acceptance.time, "sleep", lambda _seconds: None)

    assert tui_acceptance._wait_for_free_port(45678, timeout_s=1.0) is True


def test_cleanup_port_wait_remains_fail_closed_for_a_bound_listener(monkeypatch) -> None:
    clock = iter((0.0, 0.2))
    monkeypatch.setattr(tui_acceptance, "_port_is_free", lambda _port: False)
    monkeypatch.setattr(tui_acceptance.time, "monotonic", lambda: next(clock))

    assert tui_acceptance._wait_for_free_port(45678, timeout_s=0.1) is False


def test_cleanup_port_probe_rejects_an_active_loopback_listener() -> None:
    """SO_REUSEADDR accepts transient TCP state, never a real listener."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    try:
        assert tui_acceptance._port_is_free(port) is False
    finally:
        listener.close()
    assert tui_acceptance._port_is_free(port) is True


def test_tui_runner_sets_a_case_local_worker_log_directory(tmp_path: Path, monkeypatch) -> None:
    """Gazebo launch diagnostics must survive in the formal case directory."""

    allocation = allocate("m1-worker-log")
    paths = case_paths(tmp_path, "m1", SCRIPTED_TUI)
    paths.root.mkdir(parents=True)
    paths.instructions.write_text("scripted M1 task", encoding="utf-8")
    ros_python_path = "/opt/ros/jazzy/lib/python3.12/site-packages"
    monkeypatch.setenv("PYTHONPATH", ros_python_path)
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_STATIC_PEERS", "127.0.0.1")
    _write_json(
        paths.receipt,
        {"preexisting_processes": [], "protected_ros_graphs": {}},
    )
    seen = {}

    class Process:
        pid = 12345
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def popen(*_args, **kwargs):
        seen["mcp_environment"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(tui_acceptance.subprocess, "Popen", popen)

    def scripted_run(_command, _paths, env):
        seen["tui_environment"] = dict(env)
        return 0

    monkeypatch.setattr(tui_acceptance, "_run_scripted_tui", scripted_run)
    monkeypatch.setattr(tui_acceptance, "_wait_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tui_acceptance, "_process_snapshot", lambda: [])
    monkeypatch.setattr(tui_acceptance, "_terminate_owned_worker_groups", lambda **_kwargs: [])
    monkeypatch.setattr(tui_acceptance, "_wait_for_free_port", lambda _port: True)
    monkeypatch.setattr(tui_acceptance, "_partition_cleanup", lambda _partition: {"state": "PASSED"})
    monkeypatch.setattr(tui_acceptance.shutil, "which", lambda _name: "/usr/bin/script")
    monkeypatch.setattr(tui_acceptance.os, "getpgid", lambda _pid: 12345)
    from extensions.gazebo.ros2_ws import acceptance_isolation

    monkeypatch.setattr(
        acceptance_isolation, "candidate_domain_evidence", lambda _domain: {"state": "PASSED"}
    )
    monkeypatch.setattr(
        acceptance_isolation,
        "probe_ros_graph",
        lambda _domain: {"availability": "AVAILABLE", "nodes": [], "topics": []},
    )

    assert run_case(ROOT, paths, allocation) == 0
    mcp_environment = seen["mcp_environment"]
    tui_environment = seen["tui_environment"]
    assert mcp_environment["PYTHONPATH"] == os.pathsep.join((str(ROOT), ros_python_path))
    assert mcp_environment["OPENETA_WORKER_LOG_DIR"] == str(paths.root / "worker-logs")
    assert "ROS_LOCALHOST_ONLY" not in mcp_environment
    assert "ROS_STATIC_PEERS" not in mcp_environment
    assert mcp_environment["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"
    assert not any(key.startswith("OPENETA_LLM_") for key in mcp_environment)
    assert "OPENETA_LLM_MODEL" in tui_environment
    assert not any(key.startswith("OPENETA_LLM_FALLBACK_") for key in tui_environment)


def test_human_gated_tui_receives_resolved_root_provider_config(
    tmp_path: Path, monkeypatch
) -> None:
    """A case-local human TUI must not lose the repository's provider config."""

    allocation = allocate("m0-human-provider")
    paths = case_paths(tmp_path, "m0", DETERMINISTIC)
    paths.root.mkdir(parents=True)
    paths.instructions.write_text("human M0 task", encoding="utf-8")
    _write_json(
        paths.receipt,
        {"preexisting_processes": [], "protected_ros_graphs": {}},
    )
    config = _provider_config(vision=False)
    seen = {}

    class Process:
        pid = 12345
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def popen(*_args, **kwargs):
        seen["mcp_environment"] = kwargs["env"]
        return Process()

    def human_run(*_args, **kwargs):
        seen["tui_environment"] = dict(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tui_acceptance.subprocess, "Popen", popen)
    monkeypatch.setattr(tui_acceptance.subprocess, "run", human_run)
    monkeypatch.setattr(tui_acceptance, "_root_provider_config", lambda _repo: config)
    monkeypatch.setattr(tui_acceptance, "_wait_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tui_acceptance, "_process_snapshot", lambda: [])
    monkeypatch.setattr(tui_acceptance, "_terminate_owned_worker_groups", lambda **_kwargs: [])
    monkeypatch.setattr(tui_acceptance, "_wait_for_free_port", lambda _port: True)
    monkeypatch.setattr(tui_acceptance, "_partition_cleanup", lambda _partition: {"state": "PASSED"})
    monkeypatch.setattr(tui_acceptance.shutil, "which", lambda _name: "/usr/bin/script")
    monkeypatch.setattr(tui_acceptance.os, "getpgid", lambda _pid: 12345)
    from extensions.gazebo.ros2_ws import acceptance_isolation

    monkeypatch.setattr(
        acceptance_isolation, "candidate_domain_evidence", lambda _domain: {"state": "PASSED"}
    )
    monkeypatch.setattr(
        acceptance_isolation,
        "probe_ros_graph",
        lambda _domain: {"availability": "AVAILABLE", "nodes": [], "topics": []},
    )

    assert run_case(ROOT, paths, allocation) == 0
    assert not any(
        key.startswith("OPENETA_LLM_") for key in seen["mcp_environment"]
    )
    assert seen["tui_environment"]["OPENETA_LLM_MODEL"] == "unit-model"
    assert seen["tui_environment"]["OPENETA_LLM_MAX_TOKENS"] == "256"
    assert seen["tui_environment"]["OPENETA_LLM_ENABLE_VISION"] == "false"
    assert not any(
        key.startswith("OPENETA_LLM_FALLBACK_") for key in seen["tui_environment"]
    )


def test_owned_worker_cleanup_uses_only_matching_run_process_group(monkeypatch) -> None:
    """A runner must not leave (or signal) a worker outside its own case."""

    rows = [
        {
            "pid": 31001,
            "cmdline": "python sim/bench_worker.py --bench dummy --port 0",
            "openeta_tui_run_id": "this-case",
        },
        {
            "pid": 31002,
            "cmdline": "python sim/bench_worker.py --bench dummy --port 0",
            "openeta_tui_run_id": "other-case",
        },
        {
            "pid": 31003,
            "cmdline": "python sim/bench_worker.py --bench dummy --port 0",
            "openeta_tui_run_id": "this-case",
        },
    ]
    terminated: set[int] = set()

    # The server may stop the worker before the runner acts.  Cleanup must use
    # the pre-shutdown ownership snapshot rather than rediscovering nothing.
    monkeypatch.setattr(tui_acceptance, "_process_snapshot", lambda: [])
    monkeypatch.setattr(tui_acceptance.os, "getpgid", lambda pid: pid)

    def fake_killpg(pgid: int, action: int) -> None:
        if action == signal.SIGTERM:
            terminated.add(pgid)

    monkeypatch.setattr(tui_acceptance.os, "killpg", fake_killpg)
    monkeypatch.setattr(
        tui_acceptance,
        "_process_group_exited",
        lambda pgid: pgid in terminated,
    )

    evidence = tui_acceptance._terminate_owned_worker_groups(
        run_id="this-case",
        before=[{"pid": 31003}],
        candidates=rows,
        timeout_s=0.01,
    )

    assert terminated == {31001}
    assert evidence == [
        {
            "pid": 31001,
            "cmdline": "python sim/bench_worker.py --bench dummy --port 0",
            "run_id": "this-case",
            "owned": True,
            "pgid": 31001,
            "termination_signal": "SIGTERM",
            "group_exited": True,
            "state": "exited",
        }
    ]


def test_owned_residuals_ignore_unmarked_diagnostic_commands() -> None:
    """A command line mentioning Gazebo is not ownership evidence by itself."""

    rows = [
        {"pid": 41001, "cmdline": "bash -c ros2 launch ... gz sim ..."},
        {
            "pid": 41002,
            "cmdline": "python sim/bench_worker.py --bench gazebo",
            "openeta_tui_run_id": "this-case",
        },
        {
            "pid": 41003,
            "cmdline": "python sim/bench_worker.py --bench gazebo",
            "openeta_tui_run_id": "other-case",
        },
    ]

    assert tui_acceptance._owned_process_residuals(rows, run_id="this-case") == [rows[1]]


def test_process_snapshot_ignores_shell_prose_but_keeps_real_workloads() -> None:
    """Only actual runtime argv may become a pre-existing-process gate."""

    assert not tui_acceptance._is_snapshot_candidate_argv(
        ["/bin/bash", "-c", "ros2 launch pkg file.py; gz sim world.sdf"]
    )
    assert tui_acceptance._is_snapshot_candidate_argv(
        ["/usr/bin/python3", "/opt/ros/jazzy/bin/ros2", "launch", "pkg", "file.py"]
    )
    assert tui_acceptance._is_snapshot_candidate_argv(
        ["/usr/bin/python3", "-u", "/workspace/sim/bench_worker.py"]
    )
    assert tui_acceptance._is_snapshot_candidate_argv(["/usr/bin/gz", "sim", "world.sdf"])


def test_control_m2_reads_precise_failure_code_from_durable_response(tmp_path: Path) -> None:
    """A compact proxy diagnostic cannot erase MoveIt's persisted error code."""

    response = tmp_path / "move-to-response.json"
    _write_json(response, {"ok": False, "error_code": "MOTION_PLAN_FAILED"})
    result = SimpleNamespace(
        details={"outputs": {"response": {"response_path": str(response)}}}
    )

    assert tui_acceptance._control_response_has(result, "error_code", "MOTION_PLAN_FAILED")
    assert not tui_acceptance._control_response_has(result, "error_code", "GRIPPER_FAILED")


def test_control_m2_uses_the_inset_bidirectional_moveit_probe(monkeypatch, tmp_path: Path) -> None:
    """Four real A<->B moves must avoid the initial hard-limit IK branch."""

    observe_path = tmp_path / "observe.json"
    _write_json(
        observe_path,
        {"observation": {"robot": {"end_effector_pose": {"xyz": [0.1, 0.2, 0.9]}}}},
    )
    observed = SimpleNamespace(
        success=True,
        details={"outputs": {"response": {"response_path": str(observe_path)}}},
    )
    calls = []

    class Runner:
        def require_success(self, name, parameters=None):
            calls.append((name, dict(parameters or {})))
            return observed if name == "observe" else SimpleNamespace(success=True)

        def invoke(self, name, parameters=None):
            calls.append((name, dict(parameters or {})))
            return SimpleNamespace(success=False)

    monkeypatch.setattr(tui_acceptance, "_control_response_has", lambda *_args: True)
    tui_acceptance._m2_control_motion(Runner())

    assert [name for name, _ in calls] == [
        "observe",
        "move_to",
        "move_to",
        "move_to",
        "move_to",
        *(["gripper_control"] * 6),
        "move_to",
        "observe",
    ]
    assert [
        parameters["position"]
        for name, parameters in calls
        if name == "gripper_control"
    ] == list(tui_acceptance.M2_GRIPPER_SEQUENCE)
    moves = [parameters for name, parameters in calls if name == "move_to"]
    assert [move["z"] for move in moves[:4]] == pytest.approx([0.86, 0.88, 0.86, 0.88])
    assert moves[4]["x"] == 99.0 and moves[4]["y"] == 99.0 and moves[4]["z"] == 99.0


def test_m2_verifier_uses_only_correlated_durable_motion_receipts() -> None:
    """M2 credits only the ordered strict receipts, never parameter echoes."""

    def action(
        name: str,
        request_id: str,
        *,
        success: bool,
        parameters: dict | None = None,
    ) -> dict:
        return {
            "name": name,
            "parameters": parameters or {},
            "status": "executed" if success else "failed",
            "result": {
                "success": success,
                "details": {
                    "outputs": {
                        "mcp_calls": [
                            {"request": {"request_id": request_id, "tool": name}}
                        ]
                    },
                    "environment_receipt": {"observation_fresh": True},
                    "observation": {
                        "robot": {"joint_positions": [0.0] * 7},
                        "cameras": [
                            {"rgb_path": "one.png", "depth_path": "one-depth.png"},
                            {"rgb_path": "two.png", "depth_path": "two-depth.png"},
                        ],
                    },
                },
            },
        }

    calls = [
        action(
            "create_simulator_env",
            "create",
            success=True,
            parameters={"env_id": ENV_IDS["m2"]},
        )
    ]
    payloads = []
    targets = (
        {"xyz": [0.1, 0.2, 0.8], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"xyz": [0.1, 0.2, 0.82], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
    )
    for index, target in enumerate((targets[0], targets[1], targets[0], targets[1]), 1):
        request_id = f"move-{index}"
        calls.append(action("move_to", request_id, success=True, parameters={"target_pose": target}))
        payloads.append(
            {
                tui_acceptance._MCP_EVIDENCE_REQUEST_ID: request_id,
                "ok": True,
                "reached_target": True,
                "stalled": False,
                "target": target,
                "action_completed_ros_time_s": float(index),
                "start_state_recovery": {
                    "schema_version": "m2_start_state_recovery_v1"
                },
            }
        )
    for index, position in enumerate(tui_acceptance.M2_GRIPPER_SEQUENCE, 1):
        request_id = f"gripper-{index}"
        calls.append(
            action(
                "gripper_control",
                request_id,
                success=True,
                parameters={"position": position},
            )
        )
        payloads.append(
            {
                tui_acceptance._MCP_EVIDENCE_REQUEST_ID: request_id,
                "ok": True,
                "reached_goal": True,
                "stalled": False,
                "action_completed_ros_time_s": 10.0 + index,
            }
        )
    calls.append(action("move_to", "unreachable", success=False))
    payloads.append(
        {
            tui_acceptance._MCP_EVIDENCE_REQUEST_ID: "unreachable",
            "ok": False,
            "error_code": "MOTION_PLAN_FAILED",
            "reached_target": False,
            "stalled": False,
        }
    )
    calls.extend(
        [
            action("observe", "observe-final", success=True),
            action("close_simulator_env", "close", success=True),
        ]
    )
    payloads.append(
        {
            tui_acceptance._MCP_EVIDENCE_REQUEST_ID: "observe-final",
            "observation_fresh": True,
        }
    )

    assert tui_acceptance._verify_m2(calls, payloads) == []

    payloads[-2][tui_acceptance._MCP_EVIDENCE_REQUEST_ID] = "wrong-request"
    errors = tui_acceptance._verify_m2(calls, payloads)
    assert any("lacks one correlated durable MCP response" in error for error in errors)
    assert any("MOTION_PLAN_FAILED" in error for error in errors)


def test_m2_verifier_allows_one_observed_same_position_retry_but_rejects_stalls() -> None:
    """A gripper retry is a narrow recovery branch, not a success loophole."""

    # Reuse the independently verified baseline and replace its first gripper
    # action with failed → fresh observe → strict same-position success.
    calls: list[dict] = []
    payloads: list[dict] = []

    def add(name: str, request_id: str, *, success: bool, parameters: dict | None = None, **receipt) -> None:
        calls.append(
            {
                "name": name,
                "parameters": parameters or {},
                "status": "executed" if success else "failed",
                "result": {
                    "success": success,
                    "details": {
                        "outputs": {"mcp_calls": [{"request": {"request_id": request_id}}]},
                        "environment_receipt": {"observation_fresh": True},
                        "observation": {
                            "robot": {"joint_positions": [0.0] * 7},
                            "cameras": [
                                {"rgb_path": "one.png", "depth_path": "one-depth.png"},
                                {"rgb_path": "two.png", "depth_path": "two-depth.png"},
                            ],
                        },
                    },
                },
            }
        )
        payloads.append(
            {
                tui_acceptance._MCP_EVIDENCE_REQUEST_ID: request_id,
                "action_completed_ros_time_s": float(len(payloads) + 1),
                **receipt,
            }
        )

    add("create_simulator_env", "create", success=True, parameters={"env_id": ENV_IDS["m2"]})
    target_a = {"xyz": [0.1, 0.2, 0.8], "quat_xyzw": [0, 0, 0, 1]}
    target_b = {"xyz": [0.1, 0.2, 0.82], "quat_xyzw": [0, 0, 0, 1]}
    for index, target in enumerate((target_a, target_b, target_a, target_b), 1):
        add(
            "move_to",
            f"move-{index}",
            success=True,
            parameters={"target_pose": target},
            ok=True,
            reached_target=True,
            stalled=False,
            target=target,
            start_state_recovery={"schema_version": "m2_start_state_recovery_v1"},
        )
    add(
        "gripper_control",
        "gripper-failed",
        success=False,
        parameters={"position": 1},
        ok=False,
        reached_goal=False,
        stalled=False,
    )
    add("observe", "retry-observe", success=True)
    add(
        "gripper_control",
        "gripper-retry",
        success=True,
        parameters={"position": 1},
        ok=True,
        reached_goal=True,
        stalled=False,
    )
    for index, position in enumerate(tui_acceptance.M2_GRIPPER_SEQUENCE[1:], 2):
        add(
            "gripper_control",
            f"gripper-{index}",
            success=True,
            parameters={"position": position},
            ok=True,
            reached_goal=True,
            stalled=False,
        )
    add(
        "move_to",
        "unreachable",
        success=False,
        ok=False,
        reached_target=False,
        stalled=False,
        error_code="MOTION_PLAN_FAILED",
    )
    add("observe", "observe-final", success=True)
    add("close_simulator_env", "close", success=True)

    assert tui_acceptance._verify_m2(calls, payloads) == []

    # A misleading action-level success cannot turn a stalled terminal
    # receipt into a successful sixth gripper transition.
    payloads[7]["stalled"] = True
    payloads[7]["reached_goal"] = False
    errors = tui_acceptance._verify_m2(calls, payloads)
    assert any("fresh observe" in error or "retry" in error for error in errors)


def test_m2_ab_targets_use_the_submitted_pose_not_controller_orientation_normalisation() -> None:
    call = {"parameters": {"target_pose": {"xyz": [0.25, 0.0, 0.65]}}}
    nodes = [
        {
            "target": {
                "xyz": [0.25, 0.0, 0.65],
                "quat_xyzw": [0.001, 0.0, 0.0, 0.999999],
            }
        }
    ]

    assert tui_acceptance._m2_move_target_key(call, nodes) == '{"xyz":[0.25,0.0,0.65]}'
    assert tui_acceptance._m2_move_target_key(
        {"parameters": {"x": 0.25, "y": 0.0, "z": 0.65}}, []
    ) == '{"xyz":[0.25,0.0,0.65]}'


def test_m3_control_uses_fixed_world_fixture_poses_not_a_geometry_gate() -> None:
    """The preflight path may command a pose, but never infer contact from it."""

    calls: list[tuple[str, dict]] = []

    class Runner:
        def require_success(self, name: str, parameters: dict):
            calls.append((name, parameters))

    tui_acceptance._m3_motion(Runner())

    assert [name for name, _ in calls] == [
        "move_to",
        "move_to",
        "gripper_control",
        "move_to",
        "gripper_control",
    ]
    approach, capture, _, lift, _ = [parameters for _, parameters in calls]
    assert approach["target_pose"] == {
        "frame": "world",
        "euler_xyz_deg": [115.0, 0.0, 90.0],
        "xyz": [0.1552, -0.1, 0.5686],
    }
    assert capture["target_pose"]["xyz"] == [0.1552, -0.1, 0.4976]
    assert lift["target_pose"] == {
        "frame": "world",
        "euler_xyz_deg": [115.0, 0.0, 90.0],
        "xyz": [0.1552, -0.1, 0.5976],
    }
    assert lift["target_pose"]["xyz"][2] - capture["target_pose"]["xyz"][2] >= 0.1 - 1e-9
    for parameters in (approach, capture, lift):
        assert parameters["tolerance"] == 0.0002
        assert parameters["ori_tolerance"] == 0.002
    for _, parameters in calls:
        assert not {"distance", "contact_gate", "tf_lookup", "geometry"} & set(parameters)


def test_formal_verifier_rejects_missing_worker_cleanup_evidence(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m0", DETERMINISTIC)
    cleanup_path = paths.root / "cleanup.json"
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    cleanup.pop("owned_worker_groups_exited")
    _write_json(cleanup_path, cleanup)

    result = verify_case(paths, "m0", DETERMINISTIC)

    assert result["status"] == "failed"
    assert "bench-worker process groups" in " ".join(result["errors"])


def test_tool_call_reader_prefers_raw_command_over_compact_action_summary() -> None:
    """Compact episode summaries must never hide the correlated MCP result."""

    raw = {
        "kind": "tool_call",
        "name": "create_simulator_env",
        "parameters": {"env_id": ENV_IDS["m0"]},
        "status": "executed",
        "result": {
            "success": True,
            "details": {"outputs": {"mcp_calls": [{"request": {"request_id": "raw"}}]}},
        },
    }
    compact = {
        "name": "create_simulator_env",
        "status": "executed",
        "result": {"success": True, "details": {"outputs": {"mcp": {}}}},
    }
    events = [
        {
            "event_type": "action",
            "payload": {
                # The episode-step form is intentionally compact and lacks
                # mcp_calls. It must not be mistaken for evidence.
                "action": {"tool_calls": [compact]},
                "command": {"tool_calls": [raw]},
            },
        },
        {
            "event_type": "tool_execution",
            "payload": {"tool_calls": [raw]},
        },
    ]

    assert tui_acceptance._tool_calls(events) == [raw]


def test_m0_verifier_uses_trace_and_artifacts_not_planner_summary(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m0", DETERMINISTIC)
    paths.mcp_log.write_text("OpenETA MCP server started\n", encoding="utf-8")
    create_outputs, create_receipt = _mcp_outputs(
        paths.root,
        "create_simulator_env",
        [("create_env", {"handle": "test-handle"}), ("reset_env", {"success": True})],
    )
    create_outputs.update(
        {
            "environment": {"env_id": ENV_IDS["m0"]},
            "initial_observation": {"cameras": [{"rgb_path": "rgb.png"}]},
        }
    )
    observe_outputs, observe_receipt = _mcp_outputs(
        paths.root, "observe", [("render_env", {"success": True})]
    )
    close_outputs, close_receipt = _mcp_outputs(
        paths.root, "close_simulator_env", [("close_env", {"success": True})]
    )
    events = [
        _tool(
            "create_simulator_env",
            parameters={"env_id": ENV_IDS["m0"]},
            outputs=create_outputs,
            receipt=create_receipt,
        ),
        _tool("observe", outputs=observe_outputs, receipt=observe_receipt),
        _tool("close_simulator_env", outputs=close_outputs, receipt=close_receipt),
    ]
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    artifacts = paths.trace_root / "sessions/unit/working/artifacts.json"
    _write_json(artifacts, {"rgb": "rgb.png"})
    paths.transcript.write_text(
        "\n".join(
            ["/config", "/tools", "/session", "/memory all --json", *sorted(SIX_SIMULATOR_TOOLS)]
        ),
        encoding="utf-8",
    )

    result = verify_case(paths, "m0", DETERMINISTIC)

    assert result["status"] == "passed", result["errors"]
    assert result["tool_call_count"] == 3


def test_control_only_m0_uses_the_same_mcp_evidence_without_tui_claims(tmp_path: Path) -> None:
    """Control preflight remains strict on MCP evidence but never needs a PTY."""

    paths = _prepare_evidence(tmp_path, "m0", CONTROL_ONLY)
    paths.mcp_log.write_text("OpenETA MCP server started\n", encoding="utf-8")
    create_outputs, create_receipt = _mcp_outputs(
        paths.root,
        "create_simulator_env",
        [("create_env", {"handle": "control-handle"}), ("reset_env", {"success": True})],
        handle="control-handle",
    )
    create_outputs.update(
        {
            "environment": {"env_id": ENV_IDS["m0"]},
            "initial_observation": {"cameras": [{"rgb_path": "rgb.png"}]},
        }
    )
    observe_outputs, observe_receipt = _mcp_outputs(
        paths.root, "observe", [("render_env", {"success": True})], handle="control-handle"
    )
    close_outputs, close_receipt = _mcp_outputs(
        paths.root, "close_simulator_env", [("close_env", {"success": True})], handle="control-handle"
    )
    events = [
        _tool(
            "create_simulator_env",
            parameters={"env_id": ENV_IDS["m0"]},
            outputs=create_outputs,
            receipt=create_receipt,
        ),
        _tool("observe", outputs=observe_outputs, receipt=observe_receipt),
        _tool("close_simulator_env", outputs=close_outputs, receipt=close_receipt),
    ]
    for event in events:
        event["payload"].update(
            {
                "execution_profile": CONTROL_ONLY,
                "planner_invoked": False,
                "provider_invoked": False,
            }
        )
    trace = paths.trace_root / "sessions/control/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")

    result = verify_case(paths, "m0", CONTROL_ONLY)

    assert result["status"] == "passed", result["errors"]
    assert not paths.transcript.exists()


def test_control_report_is_not_a_formal_tui_report_and_stops_after_failure(
    tmp_path: Path,
) -> None:
    report = assemble_control_report(tmp_path)

    assert report["schema_version"] == "openeta.gazebo_control_acceptance.v1"
    assert report["acceptance_scope"] == "control_only_no_provider_not_formal_tui"
    assert report["formal_tui_acceptance"] == "not_run"
    assert report["overall_status"] == "inconclusive"
    assert report["milestones"]["m0"]["control_layer_status"]["status"] == "blocked"
    assert report["milestones"]["m1"]["control_layer_status"]["status"] == "not_run"
    assert CONTROL_REPORT_FILENAME != "acceptance-report.json"


def test_report_keeps_planner_status_separate_and_stops_formal_chain(tmp_path: Path) -> None:
    # Missing M0 evidence is a blocked infrastructure result. Later milestones
    # must be not_run rather than accidentally inferred from planner prose.
    report = assemble_report(tmp_path)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["overall_status"] == "inconclusive"
    assert report["milestones"]["m0"]["backend_chain_status"]["status"] == "blocked"
    assert report["milestones"]["m1"]["backend_chain_status"]["status"] == "not_run"
    assert report_exit_code(report) == 2


def test_scripted_tui_report_never_verifies_unrun_planner_autonomy(
    tmp_path: Path, monkeypatch
) -> None:
    """Scripted PTY runs are backend-only; absent autonomy cases are not failures."""

    seen: list[tuple[str, str]] = []

    def passed_backend(_paths, milestone: str, mode: str):
        seen.append((milestone, mode))
        return {"status": "passed", "errors": []}

    monkeypatch.setattr(tui_acceptance, "verify_case", passed_backend)

    report = assemble_report(tmp_path, formal_mode=SCRIPTED_TUI)

    assert report["overall_status"] == "passed"
    assert seen == [(milestone, SCRIPTED_TUI) for milestone in tui_acceptance.MILESTONES]
    for milestone in tui_acceptance.MILESTONES:
        autonomy = report["milestones"][milestone]["planner_autonomy_status"]
        assert autonomy == {
            "status": "not_applicable",
            "errors": [],
            "reason_code": "SCRIPTED_TUI_AUTONOMY_NOT_REQUIRED",
        }


def test_exact_pre_tool_provider_billing_exhaustion_is_blocked_not_m2_failure(
    tmp_path: Path,
) -> None:
    paths = _prepare_evidence(tmp_path, "m2", SCRIPTED_TUI)
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "event_type": "pipeline_plan",
                "payload": {
                    "metadata": {
                        "planner_metadata": {
                            "backend_status": "failed",
                            "backend_details": {
                                "error_type": "ProviderHttpError",
                                "error": "HTTP 402: Insufficient Balance",
                                "provider_attempts": 1,
                            },
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = verify_case(paths, "m2", SCRIPTED_TUI)

    assert result["status"] == "blocked"
    assert result["infrastructure_codes"] == ["PROVIDER_BILLING_EXHAUSTED"]
    assert "billing exhausted" in " ".join(result["errors"])
    assert not any("M2 requires" in error for error in result["errors"])
    assert not any("formal case has no simulator" in error for error in result["errors"])


def test_non_billing_planner_failure_remains_a_strict_m2_failure(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m2", SCRIPTED_TUI)
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "event_type": "pipeline_plan",
                "payload": {
                    "metadata": {
                        "planner_metadata": {
                            "backend_status": "failed",
                            "backend_details": {
                                "error_type": "ProviderHttpError",
                                "error": "HTTP 400: invalid planner request",
                            },
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = verify_case(paths, "m2", SCRIPTED_TUI)

    assert result["status"] == "failed"
    assert tui_acceptance.PROVIDER_BILLING_EXHAUSTED not in result["infrastructure_codes"]
    assert any("M2 requires" in error for error in result["errors"])


def test_autonomy_failure_cannot_change_backend_result_shape(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m1", AUTONOMY)
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({"event_type": "assistant_message", "payload": {}}) + "\n")
    result = verify_case(paths, "m1", AUTONOMY)
    assert result["status"] == "failed"
    assert "Planner did not create" in " ".join(result["errors"])


def test_scripted_tui_cli_prepares_only_scripted_cases(tmp_path: Path) -> None:
    run_root = tmp_path / "scripted"
    assert main(["--prepare-only", "--scripted-tui", "--run-root", str(run_root)]) == 0
    for milestone in ("m0", "m1", "m2", "m3", "m4"):
        instructions = (run_root / milestone / SCRIPTED_TUI / "operator-instructions.txt").read_text(
            encoding="utf-8"
        )
        assert "automation=scripted_tui" in instructions
        assert "human approval" in instructions
    m0_paths = case_paths(run_root, "m0", SCRIPTED_TUI)
    keys = scripted_tui_input(m0_paths)
    for command in ("/config", "/tools", "/session", "/memory all --json", "/quit"):
        assert command in keys
    submissions = keys.splitlines()
    assert submissions[:4] == ["/config", "/tools", "/session", "/memory all --json"]
    # The first planner prompt after console setup is one complete task, never
    # a standalone scripted_tui prefix that can trigger generic planning.
    assert len(submissions) == 6
    assert submissions[4].startswith("[automation=scripted_tui;")
    assert "openeta/dummy_sim-v0" in submissions[4]
    assert "\n" not in submissions[4]
    assert submissions[5] == "/quit"


def test_scripted_tui_selected_milestone_scope_is_explicit_and_m2_prompt_is_exact(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "scripted-m2-only"
    assert (
        main(
            [
                "--prepare-only",
                "--scripted-tui",
                "--milestones",
                "m2",
                "--run-root",
                str(run_root),
            ]
        )
        == 0
    )
    assert sorted(path.name for path in run_root.iterdir()) == ["m2"]
    instructions = (
        run_root / "m2" / SCRIPTED_TUI / "operator-instructions.txt"
    ).read_text(encoding="utf-8")
    for required in (
        "A、B、A、B",
        "[1, 0, 1, 1, 0, 1]",
        "ok=true、reached_goal/reached_target=true 且 stalled=false",
        "fresh observe",
        "仅重试一次",
        "MOTION_PLAN_FAILED",
        "唯一一次 close_simulator_env",
    ):
        assert required in instructions

    report = assemble_report(
        run_root,
        formal_mode=SCRIPTED_TUI,
        milestones=("m2",),
    )
    assert report["acceptance_scope"] == "local_automated_scripted_tui_selected_milestones"
    assert report["selected_milestones"] == ["m2"]
    assert report["full_m0_m4_acceptance"] is False
    assert set(report["milestones"]) == {"m2"}

    with pytest.raises(tui_acceptance.AcceptanceError, match="only with --scripted-tui"):
        main(["--prepare-only", "--milestones", "m2", "--run-root", str(tmp_path / "bad")])


def test_scripted_tui_driver_sends_quit_only_after_the_terminal_episode_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = prepare_case(ROOT, tmp_path, "m0", SCRIPTED_TUI, allocate("scripted-driver"))
    seen = {}

    class FakeStdin:
        def __init__(self) -> None:
            self.parts: list[str] = []
            self.closed = False

        def write(self, value: str) -> int:
            self.parts.append(value)
            return len(value)

        @staticmethod
        def flush() -> None:
            return None

        def close(self) -> None:
            self.closed = True

        def getvalue(self) -> str:
            return "".join(self.parts)

    class FakeProcess:
        pid = 43210

        def __init__(self) -> None:
            self.stdin = FakeStdin()

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(*, timeout):
            assert timeout == 30
            return 0

    process = FakeProcess()

    def popen(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return process

    def episode_finished(_paths, _process, *, timeout_s):
        seen["input_before_finish"] = process.stdin.getvalue()
        assert timeout_s == 3600.0
        return "completed"

    monkeypatch.setattr(tui_acceptance.subprocess, "Popen", popen)
    monkeypatch.setattr(tui_acceptance, "_wait_for_scripted_tui_episode", episode_finished)

    assert (
        tui_acceptance._run_scripted_tui("python -m agent.cli.openeta_cli", paths, {})
        == 0
    )  # noqa: SLF001
    assert "\n/quit\n" not in seen["input_before_finish"]
    assert process.stdin.getvalue().endswith("/quit\n")
    assert process.stdin.closed
    assert seen["kwargs"]["start_new_session"] is True


def test_scripted_tui_driver_blocks_instead_of_answering_ask_human(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = prepare_case(ROOT, tmp_path, "m0", SCRIPTED_TUI, allocate("scripted-human"))
    trace = paths.trace_root / "sessions" / "session" / "trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "event_type": "episode_step",
                "payload": {"step_result": {"info": {"pause_reason": "ask_human"}}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeStdin:
        closed = False

        def __init__(self) -> None:
            self.parts: list[str] = []

        def write(self, value: str) -> int:
            self.parts.append(value)
            return len(value)

        @staticmethod
        def flush() -> None:
            return None

        def close(self) -> None:
            self.closed = True

        def getvalue(self) -> str:
            return "".join(self.parts)

    class FakeProcess:
        pid = 43211

        def __init__(self) -> None:
            self.stdin = FakeStdin()
            self.terminated = False

        def poll(self):
            return None if not self.terminated else 1

        def wait(self, *, timeout):
            self.terminated = True
            return 1

    process = FakeProcess()
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(tui_acceptance.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        tui_acceptance.os,
        "killpg",
        lambda pgid, sig: kill_calls.append((pgid, sig)),
    )

    assert tui_acceptance._run_scripted_tui("tui", paths, {}) == 1  # noqa: SLF001
    assert "/quit\n" not in process.stdin.getvalue()
    assert kill_calls == [(process.pid, tui_acceptance.signal.SIGTERM)]
    assert json.loads((paths.root / "scripted-tui-driver.json").read_text()) == {
        "schema_version": "openeta.scripted_tui_driver.v1",
        "status": "blocked",
        "reason_code": "TUI_HUMAN_INPUT_REQUIRED",
    }


def test_scripted_tui_m2_to_m4_require_one_exact_agenttool_environment(tmp_path: Path) -> None:
    """Formal prompts must not leave the model an alternate environment path."""

    for milestone in ("m2", "m3", "m4"):
        allocation = allocate(f"{milestone}-instruction-contract")
        paths = prepare_case(ROOT, tmp_path, milestone, SCRIPTED_TUI, allocation)
        instructions = paths.instructions.read_text(encoding="utf-8")

        assert "第一步且唯一的环境创建必须是 AgentTool create_simulator_env" in instructions
        assert f"`{ENV_IDS[milestone]}`" in instructions
        assert "禁止调用 python_exec" in instructions
        assert "任何其他 env_id（包括 libero）" in instructions
        assert "最后唯一一次 close_simulator_env" in instructions


def test_m3_verifier_correlates_tui_mcp_responses_ack_and_numeric_proof(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m3", DETERMINISTIC)
    paths.mcp_log.write_text("OpenETA MCP server started\n", encoding="utf-8")

    close = {
        "native_contact_gate": {
            "accepted": True,
            "left_sample_count": 3,
            "right_sample_count": 3,
            "left_span_s": 0.100,
            "right_span_s": 0.101,
            "evidence": {"target_id": "m3_target"},
        },
        "detachable_joint": {"state": "attached"},
        "physical_verification": {
            "schema_version": "openeta.m3.detachable_joint.v1",
            "reason_code": "M3_ATTACH_ACKED_UNPROVEN",
            "grasp_confirmed": False,
        },
    }
    lift = {
        "child_link_proof": {"lift_m": 0.080, "capture_relative_translation_m": 0.010},
        "physical_verification": {
            "schema_version": "openeta.m3.detachable_joint.v1",
            "reason_code": "M3_TARGET_HELD",
            "grasp_confirmed": True,
            "evidence": {"lift_m": 0.080, "capture_relative_translation_m": 0.010},
        },
    }
    opened = {
        "detachable_joint": {"state": "detached"},
        "physical_verification": {
            "schema_version": "openeta.m3.detachable_joint.v1",
            "reason_code": "READY",
            "grasp_confirmed": False,
        },
    }
    create_outputs, create_receipt = _mcp_outputs(
        paths.root,
        "create_simulator_env",
        [("create_env", {"handle": "m3-handle"}), ("reset_env", {"success": True})],
        handle="m3-handle",
        session_id="m3-session",
    )
    create_outputs["initial_observation"] = {"success": True}
    close_outputs, close_receipt = _mcp_outputs(
        paths.root,
        "gripper_control",
        [("gripper_close", close)],
        handle="m3-handle",
        session_id="m3-session",
    )
    lift_outputs, lift_receipt = _mcp_outputs(
        paths.root,
        "move_to",
        [("move_to", lift)],
        handle="m3-handle",
        session_id="m3-session",
    )
    open_outputs, open_receipt = _mcp_outputs(
        paths.root,
        "gripper_control",
        [("gripper_open", opened)],
        handle="m3-handle",
        session_id="m3-session",
    )
    environment_close_outputs, environment_close_receipt = _mcp_outputs(
        paths.root,
        "close_simulator_env",
        [("close_env", {"success": True})],
        handle="m3-handle",
        session_id="m3-session",
    )
    events = [
        _tool(
            "create_simulator_env",
            parameters={"env_id": ENV_IDS["m3"]},
            outputs=create_outputs,
            receipt=create_receipt,
        ),
        _tool(
            "gripper_control",
            outputs=close_outputs,
            receipt=close_receipt,
        ),
        _tool(
            "move_to",
            outputs=lift_outputs,
            receipt=lift_receipt,
        ),
        _tool(
            "gripper_control",
            outputs=open_outputs,
            receipt=open_receipt,
        ),
        _tool(
            "close_simulator_env",
            outputs=environment_close_outputs,
            receipt=environment_close_receipt,
        ),
    ]
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")

    assert verify_case(paths, "m3", DETERMINISTIC)["status"] == "passed"

    lift["physical_verification"]["evidence"]["lift_m"] = 0.079
    lift_path = Path(lift_outputs["response"]["response_path"])
    _write_json(lift_path, lift)
    failed = verify_case(paths, "m3", DETERMINISTIC)
    assert failed["status"] == "failed"
    assert "numeric child-link" in " ".join(failed["errors"])


def test_formal_tui_rejects_missing_or_mismatched_mcp_chain(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m0", SCRIPTED_TUI)
    paths.mcp_log.write_text("OpenETA MCP server started\n", encoding="utf-8")
    outputs, receipt = _mcp_outputs(
        paths.root,
        "create_simulator_env",
        [("create_env", {"handle": "test-handle"}), ("reset_env", {"success": True})],
    )
    outputs["initial_observation"] = {"success": True}
    observe_outputs, observe_receipt = _mcp_outputs(
        paths.root, "observe", [("render_env", {"success": True})]
    )
    close_outputs, close_receipt = _mcp_outputs(
        paths.root, "close_simulator_env", [("close_env", {"success": True})]
    )
    close_outputs["mcp_calls"][0]["environment_receipt"]["mcp_request_id"] = "wrong-request"
    events = [
        _tool("create_simulator_env", parameters={"env_id": ENV_IDS["m0"]}, outputs=outputs, receipt=receipt, profile=SCRIPTED_TUI),
        _tool("observe", outputs=observe_outputs, receipt=observe_receipt, profile=SCRIPTED_TUI),
        _tool("close_simulator_env", outputs=close_outputs, receipt=close_receipt, profile=SCRIPTED_TUI),
    ]
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    _write_json(paths.trace_root / "sessions/unit/working/artifacts.json", {"m": 1})
    paths.transcript.write_text(
        "\n".join(["/config", "/tools", "/session", "/memory all --json", *sorted(SIX_SIMULATOR_TOOLS)]),
        encoding="utf-8",
    )
    result = verify_case(paths, "m0", SCRIPTED_TUI)
    assert result["status"] == "failed"
    assert "not correlated" in " ".join(result["errors"])


def test_m4_requires_actual_oracle_output_and_truthful_fake_candidate(tmp_path: Path) -> None:
    paths = _prepare_evidence(tmp_path, "m4", SCRIPTED_TUI)
    paths.mcp_log.write_text("OpenETA MCP server started\n", encoding="utf-8")
    attached = {
        "native_contact_gate": {
            "accepted": True, "left_sample_count": 3, "right_sample_count": 3,
            "left_span_s": 0.101, "right_span_s": 0.101,
            "evidence": {"target_id": "m3_target"},
        },
        "detachable_joint": {"state": "attached"},
    }
    held = {
        "physical_verification": {
            "schema_version": "openeta.m3.detachable_joint.v1",
            "reason_code": "M3_TARGET_HELD", "grasp_confirmed": True,
            "evidence": {"lift_m": 0.080, "capture_relative_translation_m": 0.010},
        }
    }
    detached = {"detachable_joint": {"state": "detached"}}
    create_outputs, create_receipt = _mcp_outputs(
        paths.root, "create_simulator_env",
        [("create_env", {"handle": "m4-handle"}), ("reset_env", {"success": True})],
        handle="m4-handle", session_id="m4-session",
    )
    gripper_close, close_receipt = _mcp_outputs(
        paths.root, "gripper_control", [("gripper_close", attached)],
        handle="m4-handle", session_id="m4-session",
    )
    move, move_receipt = _mcp_outputs(
        paths.root, "move_to", [("move_to", held)], handle="m4-handle", session_id="m4-session"
    )
    gripper_open, open_receipt = _mcp_outputs(
        paths.root, "gripper_control", [("gripper_open", detached)],
        handle="m4-handle", session_id="m4-session",
    )
    env_close, env_close_receipt = _mcp_outputs(
        paths.root, "close_simulator_env", [("close_env", {"success": True})],
        handle="m4-handle", session_id="m4-session",
    )
    candidate = {
        "schema_version": "openeta.m4.contractual_fake_grasp_candidate.v1",
        "kind": "contractual_fake_grasp_candidate",
        "candidate_id": "m4-contractual-test",
        "perception_source": "gazebo_oracle",
        "is_model_prediction": False,
        "provenance": "oracle_contract_fixture",
    }
    oracle_outputs, oracle_receipt = _mcp_outputs(
        paths.root,
        "oracle_perceive",
        [("oracle_perceive", {"success": True})],
        handle="m4-handle",
        session_id="m4-session",
    )
    oracle_outputs.update(
        {"perception_source": "gazebo_oracle", "fake_grasp_candidate": candidate}
    )
    events = [
        _tool("create_simulator_env", parameters={"env_id": ENV_IDS["m4"]}, outputs=create_outputs, receipt=create_receipt, profile=SCRIPTED_TUI),
        _tool("gripper_control", outputs=gripper_close, receipt=close_receipt, profile=SCRIPTED_TUI),
        _tool("move_to", outputs=move, receipt=move_receipt, profile=SCRIPTED_TUI),
        _tool("oracle_perceive", outputs=oracle_outputs, receipt=oracle_receipt, profile=SCRIPTED_TUI),
        _tool("gripper_control", outputs=gripper_open, receipt=open_receipt, profile=SCRIPTED_TUI),
        _tool("close_simulator_env", outputs=env_close, receipt=env_close_receipt, profile=SCRIPTED_TUI),
    ]
    trace = paths.trace_root / "sessions/unit/trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    verified = verify_case(paths, "m4", SCRIPTED_TUI)
    assert verified["status"] == "passed", verified["errors"]

    candidate["is_model_prediction"] = True
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    failed = verify_case(paths, "m4", SCRIPTED_TUI)
    assert failed["status"] == "failed"
    assert "fake candidate" in " ".join(failed["errors"])

    candidate["is_model_prediction"] = False
    oracle_outputs["mcp_calls"][0]["environment_receipt"]["mcp_request_id"] = "wrong-oracle-request"
    trace.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    uncorrelated = verify_case(paths, "m4", SCRIPTED_TUI)
    assert uncorrelated["status"] == "failed"
    assert "not correlated" in " ".join(uncorrelated["errors"])
