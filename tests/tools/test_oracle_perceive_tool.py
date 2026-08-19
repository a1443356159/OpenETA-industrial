from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("gymnasium")

from adapter.protocol import EnvAction
from agent.runtime.memory import (
    PENDING_SAM3_SELECTION_KEY,
    AgentMemory,
)
from agent.runtime.runtime_assembly import (
    RuntimeMcpEndpoints,
    _OracleMcpEvidence,
    _with_contractual_fake_candidate,
    bind_runtime_perception_tools,
)
from agent.tools.handlers import (
    build_oracle_perceive_segmenter,
    build_sam3_handler,
)
from agent.tools.registry import (
    ToolEffect,
    ToolExecutionContext,
    build_default_tool_registry,
    perception_segmenter_tool_name,
    resolve_perception_profile,
)
from agent.tools.sim_mcp import SimulatorMcpToolProxyConfig

FIXTURE_IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "sam3" / "sam_test.png"


class FakeSimulatorTransport:
    def __init__(self, response: dict | None = None) -> None:
        self.calls: list[tuple[str, dict, float | None]] = []
        self.response = response if response is not None else {"success": True}

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append((name, arguments, timeout_s))
        return self.response


def _mask_png(size: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[str, int]:
    image = Image.new("L", size, 0)
    left, top, right, bottom = box
    image.paste(255, box)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), (right - left) * (
        bottom - top
    )


def _oracle_response(*, image_size: tuple[int, int], prompt: str) -> dict:
    boxes = [(20, 30, 80, 100), (100, 40, 150, 110)]
    detections = []
    for box in boxes:
        encoded, area = _mask_png(image_size, box)
        detections.append(
            {
                "label": prompt,
                "score": 1.0,
                "bbox_xyxy": list(box),
                "area_px": area,
                "mask": {"format": "png", "base64": encoded},
            }
        )
    return {
        "success": True,
        "content": "Oracle perception completed.",
        "details": {
            "detection_count": len(detections),
            "detections": detections,
            "metadata": {"perception_source": "gazebo_oracle"},
        },
    }


def _oracle_context(parameters: dict) -> ToolExecutionContext:
    spec = build_default_tool_registry(perception_profile="oracle").get("oracle_perceive")
    return ToolExecutionContext(
        name="oracle_perceive",
        spec=spec,
        parameters=parameters,
        observation=None,
        metadata={},
    )


def _run_oracle_handler(tmp_path: Path):
    calls: list[dict] = []
    image_size = Image.open(FIXTURE_IMAGE).size

    def segment(request: dict) -> dict:
        calls.append(request)
        return _oracle_response(
            image_size=image_size,
            prompt=str(request.get("prompt") or ""),
        )

    handler = build_sam3_handler(
        segment,
        tool_name="oracle_perceive",
        output_root=tmp_path / "images",
        result_output_root=tmp_path / "results",
    )
    result = handler(
        _oracle_context({"image": str(FIXTURE_IMAGE), "prompt": "red cube"})
    )
    return calls, result


def test_default_profile_registers_sam3_without_oracle() -> None:
    tools = build_default_tool_registry()

    assert "sam3" in {spec.name for spec in tools.list()}
    assert "oracle_perceive" not in {spec.name for spec in tools.list()}


def test_oracle_profile_registers_oracle_without_sam3() -> None:
    tools = build_default_tool_registry(perception_profile="oracle")
    names = {spec.name for spec in tools.list()}

    assert "oracle_perceive" in names
    assert "sam3" not in names
    spec = tools.get("oracle_perceive")
    assert spec.category == "perception"
    assert spec.effect == ToolEffect.READ_ONLY
    assert {"image", "prompt"}.issubset(spec.parameters)
    assert "oracle" in spec.description
    assert "ground truth" in spec.description


def test_perception_profile_env_var_switch(monkeypatch) -> None:
    monkeypatch.delenv("OPENETA_PERCEPTION_PROFILE", raising=False)
    assert resolve_perception_profile() == "sam3"
    assert perception_segmenter_tool_name(resolve_perception_profile()) == "sam3"

    monkeypatch.setenv("OPENETA_PERCEPTION_PROFILE", "oracle")
    assert resolve_perception_profile() == "oracle"
    tools = build_default_tool_registry()
    names = {spec.name for spec in tools.list()}
    assert "oracle_perceive" in names
    assert "sam3" not in names

    monkeypatch.setenv("OPENETA_PERCEPTION_PROFILE", "unknown-value")
    assert resolve_perception_profile() == "sam3"
    assert "sam3" in {spec.name for spec in build_default_tool_registry().list()}


def test_oracle_segmenter_forwards_image_base64_and_prompt() -> None:
    transport = FakeSimulatorTransport(response={"success": True})
    segment = build_oracle_perceive_segmenter(transport)
    request = {
        "image_base64": "QUJD",
        "image_format": "png",
        "prompt": "red cube",
    }

    assert segment(request) == {"success": True}
    assert transport.calls == [
        ("oracle_perceive", {"image_base64": "QUJD", "prompt": "red cube"}, 600.0)
    ]


def test_oracle_response_flows_through_sam3_handler_pipeline(tmp_path: Path) -> None:
    calls, result = _run_oracle_handler(tmp_path)

    assert result.success is True
    assert calls[0]["prompt"] == "red cube"
    assert calls[0]["image_base64"]
    details = result.details
    assert details["tool"] == "oracle_perceive"
    assert details["backend"] == "oracle_perceive_mcp"
    assert details["model"] == "oracle_perceive"
    assert details["metadata"]["perception_source"] == "gazebo_oracle"
    assert details["detection_count"] == 2
    assert details["selection_required"] is True
    detections = details["detections"]
    assert [d["id"] for d in detections] == ["detection_000", "detection_001"]
    for detection in detections:
        mask_ref = Path(detection["mask_ref"])
        assert mask_ref.is_file()
        assert detection["bbox_xyxy"]
        assert detection["area_px"] > 0
    assert details["selection_bundle"]["candidate_count"] == 2
    artifact_tools = {
        artifact.get("tool") for artifact in details["artifacts"] if isinstance(artifact, dict)
    }
    assert artifact_tools == {"oracle_perceive"}


def test_m4_oracle_wrapper_marks_candidate_as_contractual_not_prediction(tmp_path: Path) -> None:
    image_size = Image.open(FIXTURE_IMAGE).size
    transport = FakeSimulatorTransport(
        _oracle_response(image_size=image_size, prompt="red cube")
    )
    proxy_config = SimulatorMcpToolProxyConfig(
        handle="oracle-handle",
        session_id="oracle-session",
        response_output_root=tmp_path / "responses",
    )
    mcp_evidence = _OracleMcpEvidence(
        proxy_config=proxy_config,
        response_output_root=tmp_path / "responses",
    )

    handler = _with_contractual_fake_candidate(
        build_sam3_handler(
            build_oracle_perceive_segmenter(
                transport,
                handle_provider=lambda: proxy_config.handle,
                session_id_provider=lambda: proxy_config.session_id,
                response_callback=mcp_evidence.record,
            ),
            tool_name="oracle_perceive",
            output_root=tmp_path / "images",
            result_output_root=tmp_path / "results",
        ),
        mcp_evidence=mcp_evidence,
    )
    tools = build_default_tool_registry(perception_profile="oracle")
    tools.bind_handler("oracle_perceive", handler, replace=True)
    result = tools.call(
        "oracle_perceive", {"image": str(FIXTURE_IMAGE), "prompt": "red cube"}
    )

    assert result.success is True
    outputs = result.details["outputs"]
    assert outputs["perception_source"] == "gazebo_oracle"
    candidate = outputs["fake_grasp_candidate"]
    assert candidate["kind"] == "contractual_fake_grasp_candidate"
    assert candidate["is_model_prediction"] is False
    assert candidate["perception_source"] == "gazebo_oracle"
    assert transport.calls[0][0] == "oracle_perceive"
    assert transport.calls[0][1]["handle"] == "oracle-handle"
    evidence = outputs["mcp_calls"][0]
    assert evidence["request"]["tool"] == "oracle_perceive"
    assert evidence["response"]["request_id"] == evidence["request"]["request_id"]
    assert evidence["environment_receipt"]["mcp_request_id"] == evidence["request"]["request_id"]
    assert Path(evidence["response"]["response_path"]).is_file()


def test_oracle_result_captured_into_pending_selection_and_gate(tmp_path: Path) -> None:
    _calls, result = _run_oracle_handler(tmp_path)
    details = result.details
    memory = AgentMemory()
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "oracle_perceive",
                        "result": {
                            "success": True,
                            "details": {
                                "parameters": {
                                    "image": str(FIXTURE_IMAGE),
                                    "prompt": "red cube",
                                },
                                "outputs": details,
                            },
                        },
                    }
                ]
            },
        )
    )

    pending = memory.pending_sam3_selection()
    assert pending is not None
    assert pending["result_id"] == details["result_id"]
    assert pending["candidate_count"] == 2
    assert memory.facts[PENDING_SAM3_SELECTION_KEY]["source"] == "oracle_perceive"

    blocked = memory.detection_selection_gate_error(
        tool_name="move_to",
        parameters={},
        world_mutating=True,
    )
    assert blocked is not None

    selected = memory.resolve_sam3_selection(
        result_id=details["result_id"],
        detection_id="detection_000",
        selection_source="main_agent_vlm",
        reason="leftmost cube matches the task target",
    )
    assert selected["mask_ref"] == details["detections"][0]["mask_ref"]
    assert memory.pending_sam3_selection() is None

    assert (
        memory.detection_selection_gate_error(
            tool_name="anygrasp",
            parameters={"mode": "targeted", "target_mask": selected["mask_ref"]},
        )
        is None
    )
    mismatch = memory.detection_selection_gate_error(
        tool_name="anygrasp",
        parameters={"mode": "targeted", "target_mask": "stale-mask.png"},
    )
    assert mismatch is not None


def test_sam3_result_still_captured_with_sam3_source() -> None:
    memory = AgentMemory()
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "sam3",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": "sam-regression",
                                    "detections": [
                                        {"id": "detection_000", "rank": 0, "score": 0.9}
                                    ],
                                }
                            },
                        },
                    }
                ]
            },
        )
    )

    pending = memory.pending_sam3_selection()
    assert pending is not None
    assert pending["result_id"] == "sam-regression"
    assert memory.facts[PENDING_SAM3_SELECTION_KEY]["source"] == "sam3"


def test_oracle_profile_binds_oracle_handler_over_sim_transport(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )
    tools = build_default_tool_registry(perception_profile="oracle")
    transport = FakeSimulatorTransport()

    bind_runtime_perception_tools(
        tools,
        endpoints=RuntimeMcpEndpoints(sam3_url="http://sam3.example/sse"),
        backend_factory=lambda **_kwargs: object(),
        artifact_root=tmp_path / "artifacts",
        perception_profile="oracle",
        simulator_transport=transport,
    )

    assert tools.can_execute("oracle_perceive") is True
    assert "sam3" not in {spec.name for spec in tools.list()}


def test_oracle_profile_without_sim_transport_leaves_oracle_unbound(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )
    tools = build_default_tool_registry(perception_profile="oracle")

    bind_runtime_perception_tools(
        tools,
        endpoints=RuntimeMcpEndpoints(sam3_url="http://sam3.example/sse"),
        backend_factory=lambda **_kwargs: object(),
        artifact_root=tmp_path / "artifacts",
        perception_profile="oracle",
    )

    assert tools.can_execute("oracle_perceive") is False


def test_sam3_profile_never_binds_oracle(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )
    tools = build_default_tool_registry(perception_profile="sam3")

    bind_runtime_perception_tools(
        tools,
        endpoints=RuntimeMcpEndpoints(sam3_url="http://sam3.example/sse"),
        backend_factory=lambda **_kwargs: object(),
        artifact_root=tmp_path / "artifacts",
        perception_profile="sam3",
        simulator_transport=FakeSimulatorTransport(),
    )

    names = {spec.name for spec in tools.list()}
    assert "sam3" in names
    assert "oracle_perceive" not in names
