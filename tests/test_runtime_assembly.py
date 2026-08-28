from __future__ import annotations

import json
from types import SimpleNamespace

import agent.cli.batch_eval as batch_eval
import agent.cli.openeta_cli as cli_module
import pytest
from adapter.protocol import EnvObservation, RobotState
from agent.backends.planner import StaticPlannerBackend
from agent.backends.provider_config import PlannerProviderConfig
from agent.cli.batch_eval import build_mcp_episode_worker_factory
from agent.cli.openeta_cli import OpenEtaCli
from agent.runtime.parallel import ParallelEpisodeSpec
from agent.runtime.runtime_assembly import (
    ENVIRONMENT_PLACEHOLDER_TOOLS,
    GRASP_BACKEND_ENV_VAR,
    PERCEPTION_RPC_TIMEOUT_ENV_VAR,
    REMOTE_PLACEHOLDER_TOOLS,
    RuntimeAssemblyConfig,
    RuntimeMcpEndpoints,
    _FrozenGoalPairCoordinator,
    _build_sam3_selection_reviewer,
    _qualifying_handler,
    assemble_runtime,
    resolve_runtime_mcp_endpoints,
    runtime_grasp_backend_order_from_env,
    runtime_perception_rpc_timeout_s_from_env,
)
from agent.runtime.session_workspace import SessionWorkspace
from agent.runtime.supervision import SupervisionPolicy
from agent.tools.sim_mcp import SimulatorMcpToolProxyConfig
from agent.tools.web_access import WebAccessConfig
from agent.tools.registry import (
    ToolExecutionContext,
    ToolResult,
    build_default_tool_registry,
)


class FakeSimulatorTransport:
    def __init__(self, url: str = "") -> None:
        self.url = url

    def call_tool(self, name, arguments, *, timeout_s=None):
        del name, arguments, timeout_s
        return {"success": True}


def _backend_factory(**_kwargs):
    return StaticPlannerBackend(
        {
            "kind": "response",
            "name": "talk",
            "parameters": {"message": "fixture"},
        }
    )


def _contract_snapshot(assembly):
    tools = assembly.runtime.tools
    return {
        "specs": [
            (
                spec.name,
                spec.description,
                spec.category,
                spec.effect.value,
                spec.parameters,
            )
            for spec in tools.list()
        ],
        "executable": sorted(
            spec.name for spec in tools.list() if tools.can_execute(spec.name)
        ),
        "max_validation_retries": assembly.runtime.planner.max_validation_retries,
    }


def test_sam3_selection_reviewer_has_one_shared_bounded_provider_budget() -> None:
    calls = []

    def factory(**kwargs):
        calls.append(dict(kwargs))
        return _backend_factory()

    reviewer = _build_sam3_selection_reviewer(factory)

    assert callable(reviewer)
    assert calls == [
        {
            "max_tokens": 256,
            "max_vision_images": 2,
            "timeout_s": 30.0,
            "max_attempts": 1,
        }
    ]


def test_frozen_grasp_frontier_retains_only_not_evaluated_provider_tail(
    tmp_path,
) -> None:
    artifact = tmp_path / "qualification.json"
    artifact.write_text(
        json.dumps(
            {
                "results": [
                    {"candidate_id": "g0", "verdict": "PASS"},
                    {"candidate_id": "g1", "verdict": "FAIL"},
                    {"candidate_id": "g2", "verdict": "NOT_EVALUATED"},
                    {"candidate_id": "g3", "verdict": "NOT_EVALUATED"},
                ]
            }
        ),
        encoding="utf-8",
    )
    coordinator = _FrozenGoalPairCoordinator(qualifier=SimpleNamespace())
    provider_result = ToolResult(
        True,
        "provider output",
        {
            "backend": "graspgenx_mcp",
            "grasp_candidates": [{"id": f"g{index}"} for index in range(4)],
        },
    )
    qualified_result = ToolResult(
        True,
        "qualified",
        {
            "qualification_profile": "fast_v3",
            "qualification_stop_reason": "complete_l5_pass_found",
            "qualification_artifact": {"path": str(artifact)},
            "grasp_candidates": [{"id": "g0"}],
        },
    )

    coordinator.update_grasp_frontier(
        provider_result,
        qualified_result,
        scene_epoch=1,
        planning_scene_revision=4,
    )
    expanded = coordinator.prepare_grasp_frontier_expansion(
        scene_epoch=2,
        planning_scene_revision=4,
    )

    assert [
        candidate["id"] for candidate in expanded.details["grasp_candidates"]
    ] == ["g2", "g3"]
    assert expanded.details["model_inference_invoked"] is False
    assert qualified_result.details["frozen_grasp_frontier_remaining_count"] == 2
    assert coordinator.scene_epoch == 2


def test_authoritative_fully_evaluated_artifact_does_not_requeue_failures(
    tmp_path,
) -> None:
    artifact = tmp_path / "qualification.json"
    artifact.write_text(
        json.dumps(
            {
                "results": [
                    {"candidate_id": "g0", "verdict": "PASS"},
                    {"candidate_id": "g1", "verdict": "FAIL"},
                    {"candidate_id": "g2", "verdict": "FAIL"},
                ]
            }
        ),
        encoding="utf-8",
    )
    coordinator = _FrozenGoalPairCoordinator(qualifier=SimpleNamespace())
    provider_result = ToolResult(
        True,
        "provider output",
        {"grasp_candidates": [{"id": f"g{index}"} for index in range(3)]},
    )
    qualified_result = ToolResult(
        True,
        "qualified",
        {
            "qualification_profile": "fast_v3",
            "qualification_stop_reason": "complete_l5_pass_found",
            "qualification_artifact": {"path": str(artifact)},
            "grasp_candidates": [{"id": "g0"}],
        },
    )

    coordinator.update_grasp_frontier(
        provider_result,
        qualified_result,
        scene_epoch=1,
        planning_scene_revision=4,
    )

    assert coordinator.grasp_frontier_candidates == []
    assert qualified_result.details["frozen_grasp_frontier_remaining_count"] == 0


def test_frozen_frontier_prioritizes_centered_sibling_of_failed_parent() -> None:
    coordinator = _FrozenGoalPairCoordinator(qualifier=SimpleNamespace())
    coordinator.grasp_candidate_catalog = {
        "failed": {"id": "failed", "backend_index": 118}
    }
    coordinator.grasp_frontier_candidates = [
        {"id": "other", "backend_index": 7},
        {
            "id": "sibling",
            "backend_index": 1062,
            "target_closing_alignment": {
                "compatible_parent_backend_indices": [118, 2290]
            },
        },
    ]

    count = coordinator.prioritize_grasp_frontier_for_parent("failed")

    assert count == 1
    assert [item["id"] for item in coordinator.grasp_frontier_candidates] == [
        "sibling",
        "other",
    ]
    assert coordinator.grasp_frontier_candidates[0][
        "frozen_frontier_parent_priority"
    ] is True
    assert coordinator.grasp_frontier_candidates[0][
        "frozen_frontier_parent_priority_basis"
    ] == "direct_backend_parent"


def test_frozen_frontier_prioritizes_sibling_with_shared_model_parent() -> None:
    coordinator = _FrozenGoalPairCoordinator(qualifier=SimpleNamespace())
    coordinator.grasp_candidate_catalog = {
        "primary": {
            "id": "primary",
            "backend_index": 499,
            "target_closing_alignment": {
                "compatible_parent_backend_indices": [707, 287]
            },
        }
    }
    coordinator.grasp_frontier_candidates = [
        {
            "id": "different_family",
            "backend_index": 445,
            "target_closing_alignment": {
                "compatible_parent_backend_indices": [545, 765]
            },
        },
        {
            "id": "same_model_family",
            "backend_index": 715,
            "target_closing_alignment": {
                "compatible_parent_backend_indices": [707, 924]
            },
        },
    ]

    count = coordinator.prioritize_grasp_frontier_for_parent("primary")

    assert count == 1
    assert [item["id"] for item in coordinator.grasp_frontier_candidates] == [
        "same_model_family",
        "different_family",
    ]
    assert coordinator.grasp_frontier_candidates[0][
        "frozen_frontier_parent_priority_basis"
    ] == "shared_model_centering_parent"


def test_frozen_frontier_rebase_requalifies_catalog_and_excludes_physical_failure() -> None:
    def candidate(candidate_id: str, x: float) -> dict:
        return {
            "id": candidate_id,
            "frame": "camera",
            "camera_frame": "opencv",
            "width": 0.06,
            "translation_xyz": [x, 0.2, 0.3],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "transform_matrix": [
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }

    failed = candidate("failed", 0.10)
    qualified_backup = candidate("qualified_backup", 0.12)
    provider_tail = candidate("provider_tail", 0.14)
    coordinator = _FrozenGoalPairCoordinator(qualifier=SimpleNamespace())
    coordinator.grasp_candidate_catalog = {
        item["id"]: item
        for item in (failed, qualified_backup, provider_tail)
    }
    # Before the physical attempt only the not-yet-qualified provider tail was
    # in the expansion queue.  Once the object moves, the qualified backup's
    # old IK/L5 proof is stale and it must rejoin the requalification frontier.
    coordinator.grasp_frontier_candidates = [provider_tail]
    coordinator.grasp_frontier_template = {
        "source": {
            "camera_extrinsics": {
                "camera_frame": "opencv",
                "pos": [0.0, 0.0, 0.0],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        }
    }
    coordinator.grasp_frontier_planning_scene_revision = 7
    receipt = {
        "detachable_joint": {"state": "detached"},
        "planning_scene_target_pose_sync": {
            "schema_version": "openeta.planning_scene_target_pose_sync.v1",
            "operation": "update_world_target",
            "topology_unchanged": True,
            "static_world_unchanged": True,
            "world_ids_before": ["target", "table", "bin"],
            "world_ids_after": ["target", "table", "bin"],
            "attached_ids_before": [],
            "attached_ids_after": [],
            "source_revision": 7,
            "revision": 8,
            "source_target_pose": {
                "frame": "world",
                "translation_xyz": [0.25, -0.1, 0.43],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "target_pose": {
                "frame": "world",
                "translation_xyz": [0.27, -0.1, 0.43],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "translation_delta_m": 0.02,
            "rotation_delta_rad": 0.0,
            "static_world_sha256_after": "static-scene",
        },
    }

    rebased = coordinator.rebase_grasp_frontier_from_target_pose_sync(
        receipt,
        scene_epoch=1,
        planning_scene_revision=8,
        failed_candidate_id="failed",
    )

    assert rebased.success is True
    assert [
        item["id"] for item in coordinator.grasp_frontier_candidates
    ] == ["qualified_backup", "provider_tail"]
    assert coordinator.physically_rejected_grasp_ids == {"failed"}
    assert coordinator.grasp_frontier_candidates[0][
        "translation_xyz"
    ] == pytest.approx([0.14, 0.2, 0.3])
    assert coordinator.grasp_frontier_candidates[0][
        "frozen_object_motion_rebase"
    ]["physically_rejected_candidate_ids"] == ["failed"]
    assert coordinator.grasp_candidate_catalog["qualified_backup"][
        "translation_xyz"
    ] == pytest.approx([0.14, 0.2, 0.3])


def test_frozen_grasp_frontier_expansion_bypasses_provider_inference() -> None:
    provider_calls: list[dict] = []
    qualifier_calls: list[dict] = []
    coordinator_calls: list[tuple[str, int, int]] = []

    class Qualifier:
        def qualify_result(self, result, **kwargs):
            qualifier_calls.append(dict(kwargs))
            result.details["qualification_profile"] = "fast_v3"
            result.details["qualification_stop_reason"] = (
                "complete_l5_pass_found"
            )
            return result

    class Coordinator:
        object_goals = [{"id": "p0"}]
        grasp_branch_limit = 2

        def prepare_grasp_frontier_expansion(
            self, *, scene_epoch, planning_scene_revision
        ):
            coordinator_calls.append(
                ("prepare", scene_epoch, planning_scene_revision)
            )
            return ToolResult(
                True,
                "frozen",
                {
                    "grasp_candidates": [{"id": "g2"}],
                    "model_inference_invoked": False,
                    "scene_revision": planning_scene_revision,
                },
            )

        def update_grasp_frontier(
            self,
            _provider_result,
            _qualified_result,
            *,
            scene_epoch,
            planning_scene_revision,
        ):
            coordinator_calls.append(
                ("update", scene_epoch, planning_scene_revision)
            )

        def filter_grasps(
            self,
            result,
            *,
            scene_epoch,
            planning_scene_revision,
            source,
        ):
            del source
            coordinator_calls.append(
                ("filter", scene_epoch, planning_scene_revision)
            )
            return result

    def provider(_context):
        provider_calls.append({"called": True})
        raise AssertionError("frozen frontier must not invoke the grasp provider")

    wrapped = _qualifying_handler(
        provider,
        Qualifier(),
        purpose="grasp",
        frozen_pair_coordinator=Coordinator(),
    )
    context = ToolExecutionContext(
        name="grasp_pose_estimate",
        spec=build_default_tool_registry().get("grasp_pose_estimate"),
        parameters={
            "mode": "frozen_frontier",
            "model_inference": False,
            "scene_revision": 7,
        },
        observation=EnvObservation(
            task="pick and place",
            cameras=[],
            robot=RobotState(),
            metadata={"planning_scene_revision": 7},
        ),
        metadata={
            "supervision_context": {"memory": {"scene_epoch": 3}}
        },
    )

    result = wrapped(context)

    assert result.success
    assert result.details["model_inference_invoked"] is False
    assert provider_calls == []
    assert coordinator_calls == [
        ("prepare", 3, 7),
        ("update", 3, 7),
        ("filter", 3, 7),
    ]
    assert qualifier_calls[0]["l5_pass_target"] == 1
    assert qualifier_calls[0]["l5_min_pass_target"] == 1


def test_qualification_infrastructure_error_is_not_reported_as_unreachable() -> None:
    class Qualifier:
        qualification_profile = "fast_v3"

        def qualify_result(self, result, **_kwargs):
            result.details.update(
                {
                    "grasp_candidates": [],
                    "qualification_stop_reason": "infrastructure_error",
                    "qualification_evidence": {
                        "infrastructure_error": True,
                        "stop_reason": "infrastructure_error",
                    },
                    "rejection_reason_counts": {"qualification_rpc_error": 2},
                }
            )
            return result

    wrapped = _qualifying_handler(
        lambda _context: ToolResult(
            True,
            "provider output",
            {"grasp_candidates": [{"id": "g0"}]},
        ),
        Qualifier(),
        purpose="grasp",
    )
    context = ToolExecutionContext(
        name="grasp_pose_estimate",
        spec=build_default_tool_registry().get("grasp_pose_estimate"),
        parameters={},
        observation=EnvObservation(
            task="pick and place",
            cameras=[],
            robot=RobotState(),
            metadata={"planning_scene_revision": 7},
        ),
        metadata={"supervision_context": {"memory": {"scene_epoch": 3}}},
    )

    result = wrapped(context)

    assert result.success is False
    assert result.details["reason"] == "qualification_infrastructure_error"
    assert result.details["infrastructure_error"] is True
    assert result.details["qualification_infrastructure_reason"] == (
        "qualification_rpc_error"
    )


def test_tui_and_batch_profiles_share_runtime_contracts(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )
    provider = PlannerProviderConfig(
        model="fixture",
        api_base="http://provider.example/v1",
        api_key="test",
    )
    endpoints = RuntimeMcpEndpoints(
        sam3_url="http://sam3.example/sse",
        depth_prior_url="http://depth.example/sse",
        anygrasp_url="http://anygrasp.example/sse",
        anyplace_url="http://anyplace.example/sse",
        graspgenx_url="http://graspgenx.example/sse",
        contact_graspnet_url="http://contact.example/sse",
        molmopoint_url="http://molmo.example/sse",
    )
    transport = FakeSimulatorTransport()
    policy = SupervisionPolicy.for_profile("standard")

    tui_workspace = SessionWorkspace.create("tui", root=tmp_path / "tui")
    batch_workspace = SessionWorkspace.create("batch", root=tmp_path / "batch")
    tui = assemble_runtime(
        RuntimeAssemblyConfig(
            workspace=tui_workspace,
            provider=provider,
            backend_factory=_backend_factory,
            supervision_policy=policy,
            endpoints=endpoints,
            simulator_transport=transport,
            simulator_proxy_config=SimulatorMcpToolProxyConfig(),
            web_access_config=WebAccessConfig(),
            allow_outside_sandbox=True,
            max_validation_retries=2,
        )
    )
    batch = assemble_runtime(
        RuntimeAssemblyConfig(
            workspace=batch_workspace,
            provider=provider,
            backend_factory=_backend_factory,
            supervision_policy=policy,
            endpoints=endpoints,
            simulator_transport=transport,
            simulator_proxy_config=SimulatorMcpToolProxyConfig(),
            web_access_config=WebAccessConfig(),
            allow_outside_sandbox=False,
            max_validation_retries=2,
        )
    )

    assert _contract_snapshot(tui) == _contract_snapshot(batch)
    assert tui.depth_prefetch is not None
    assert batch.depth_prefetch is not None
    assert tui_workspace.grasp_profile_id == batch_workspace.grasp_profile_id
    assert tui.runtime.memory.store.root == tui_workspace.memory_root
    assert batch.runtime.memory.store.root == batch_workspace.memory_root
    assert tui.runtime.memory.store.session_dir("tui") == tui_workspace.root
    assert batch.runtime.memory.store.session_dir("batch") == batch_workspace.root
    assert tui.runtime.planner.sam3_selection_parent_context is not None
    assert batch.runtime.planner.sam3_selection_parent_context is not None
    assert (
        tui.runtime.planner.sam3_selection_reviewer.__self__.parent_context
        is tui.runtime.planner.sam3_selection_parent_context
    )
    assert (
        batch.runtime.planner.sam3_selection_reviewer.__self__.parent_context
        is batch.runtime.planner.sam3_selection_parent_context
    )
    tui.runtime.start_session(task="tui task")
    batch.runtime.start_session(task="batch task")
    assert tui.runtime.memory.session_id == tui_workspace.session_id
    assert batch.runtime.memory.session_id == batch_workspace.session_id
    assert tui.runtime.memory.store.session_path("tui") == tui_workspace.root / "trace.jsonl"
    assert batch.runtime.memory.store.session_path("batch") == batch_workspace.root / "trace.jsonl"


def test_industrial_tool_profile_assembles_with_prebound_hidden_handlers(
    monkeypatch,
    tmp_path,
) -> None:
    """Profile visibility must not make Runtime rebind production handlers."""

    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )
    assembly = assemble_runtime(
        RuntimeAssemblyConfig(
            workspace=SessionWorkspace.create("industrial", root=tmp_path),
            provider=PlannerProviderConfig(
                model="fixture",
                api_base="http://provider.example/v1",
                api_key="test",
            ),
            backend_factory=_backend_factory,
            supervision_policy=SupervisionPolicy.for_profile("standard"),
            simulator_transport=FakeSimulatorTransport(),
            web_access_config=WebAccessConfig(),
            tool_profile="gazebo_industrial",
        )
    )

    tools = assembly.runtime.tools
    assert tools.can_execute("active_observe") is True
    assert tools.can_execute("python_exec") is False
    assert tools.bound_handler("python_exec") is not None
    assert {spec.name for spec in tools.list()} == {
        "active_observe",
        "activate_final_grasp_candidate",
        "anyplace",
        "camera_pose_to_world",
        "close_simulator_env",
        "create_simulator_env",
        "grasp_pose_estimate",
        "gripper_control",
        "move_to",
        "observe",
        "reject_sam3_detections",
        "sam3",
        "select_sam3_detection",
    }


def test_shared_runtime_fails_closed_without_remote_backends(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )
    workspace = SessionWorkspace.create("closed", root=tmp_path)
    assembly = assemble_runtime(
        RuntimeAssemblyConfig(
            workspace=workspace,
            provider=PlannerProviderConfig(
                model="fixture",
                api_base="http://provider.example/v1",
                api_key="test",
            ),
            backend_factory=_backend_factory,
            supervision_policy=SupervisionPolicy.for_profile("standard"),
            web_access_config=WebAccessConfig(),
        )
    )

    for name in (*REMOTE_PLACEHOLDER_TOOLS, *ENVIRONMENT_PLACEHOLDER_TOOLS):
        assert assembly.runtime.tools.can_execute(name) is False

    assert assembly.runtime.tools.can_execute("prepare_attachment_probe") is True
    assert assembly.runtime.tools.can_execute("assess_attachment_probe") is True


def test_shared_endpoint_resolution_owns_names_aliases_and_overrides() -> None:
    calls = []

    def loader(name, *, aliases=()):
        calls.append((name, aliases))
        return f"http://{name}.example/sse"

    endpoints = resolve_runtime_mcp_endpoints(
        RuntimeMcpEndpoints(anygrasp_url="http://override.example/sse"),
        loader=loader,
    )

    assert endpoints.anygrasp_url == "http://override.example/sse"
    assert endpoints.depth_prior_url == "http://openeta-depth-prior.example/sse"
    assert (
        "openeta-depth-prior",
        ("depth-prior", "depth_prior", "unidepth"),
    ) in calls
    assert not any(name == "openeta-anygrasp" for name, _aliases in calls)


def test_runtime_grasp_backend_policy_can_select_either_backend(monkeypatch) -> None:
    monkeypatch.delenv(GRASP_BACKEND_ENV_VAR, raising=False)
    assert runtime_grasp_backend_order_from_env() == ("graspgenx",)

    monkeypatch.setenv(GRASP_BACKEND_ENV_VAR, "auto")
    assert runtime_grasp_backend_order_from_env() == (
        "anygrasp",
        "contact_graspnet",
        "graspgenx",
    )

    monkeypatch.setenv(GRASP_BACKEND_ENV_VAR, "anygrasp")
    assert runtime_grasp_backend_order_from_env() == ("anygrasp",)

    monkeypatch.setenv(GRASP_BACKEND_ENV_VAR, "graspgenx")
    assert runtime_grasp_backend_order_from_env() == ("graspgenx",)


def test_runtime_perception_rpc_timeout_is_explicit_and_validated(monkeypatch) -> None:
    monkeypatch.delenv(PERCEPTION_RPC_TIMEOUT_ENV_VAR, raising=False)
    assert runtime_perception_rpc_timeout_s_from_env() == 600.0

    monkeypatch.setenv(PERCEPTION_RPC_TIMEOUT_ENV_VAR, "90")
    assert runtime_perception_rpc_timeout_s_from_env() == 90.0

    for invalid in ("0", "-1", "nan", "invalid"):
        monkeypatch.setenv(PERCEPTION_RPC_TIMEOUT_ENV_VAR, invalid)
        with pytest.raises(ValueError, match="finite positive number"):
            runtime_perception_rpc_timeout_s_from_env()


def test_runtime_perception_rpc_timeout_reaches_remote_builder(
    monkeypatch,
    tmp_path,
) -> None:
    captured = []
    monkeypatch.setenv(PERCEPTION_RPC_TIMEOUT_ENV_VAR, "90")
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.build_sse_anyplace_mcp_placer",
        lambda *, url, timeout_seconds: (
            captured.append((url, timeout_seconds)) or (lambda _request: {})
        ),
    )

    assemble_runtime(
        RuntimeAssemblyConfig(
            workspace=SessionWorkspace.create("rpc-timeout", root=tmp_path),
            provider=PlannerProviderConfig(
                model="fixture",
                api_base="http://provider.example/v1",
                api_key="test",
            ),
            backend_factory=_backend_factory,
            supervision_policy=SupervisionPolicy.for_profile("standard"),
            endpoints=RuntimeMcpEndpoints(
                anyplace_url="http://anyplace.example/sse"
            ),
            web_access_config=WebAccessConfig(),
        )
    )

    assert captured == [("http://anyplace.example/sse", 90.0)]


def test_contact_graspnet_is_disabled_from_executable_runtime(tmp_path) -> None:
    workspace = SessionWorkspace.create("contact-disabled", root=tmp_path)
    assembly = assemble_runtime(
        RuntimeAssemblyConfig(
            workspace=workspace,
            provider=PlannerProviderConfig(
                model="fixture",
                api_base="http://provider.example/v1",
                api_key="test",
            ),
            backend_factory=_backend_factory,
            supervision_policy=SupervisionPolicy.for_profile("standard"),
            endpoints=RuntimeMcpEndpoints(
                anygrasp_url="http://anygrasp.example/sse",
                graspgenx_url="http://graspgenx.example/sse",
                contact_graspnet_url="http://contact.example/sse",
            ),
            web_access_config=WebAccessConfig(),
        )
    )

    assert assembly.runtime.tools.can_execute("contact_graspnet") is False
    assert assembly.runtime.tools.can_execute("grasp_pose_estimate") is True


def test_real_tui_and_batch_entries_have_runtime_parity(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    provider = PlannerProviderConfig(
        model="fixture",
        api_base="http://provider.example/v1",
        api_key="test",
    )
    urls = {
        "openeta-sim": "http://sim.example/sse",
        "openeta-sam3": "http://sam3.example/sse",
        "openeta-depth-prior": "http://depth.example/sse",
        "openeta-anygrasp": "http://anygrasp.example/sse",
        "openeta-anyplace": "http://anyplace.example/sse",
        "openeta-graspgenx": "http://graspgenx.example/sse",
        "openeta-contact-graspnet": "http://contact.example/sse",
        "openeta-molmopoint": "http://molmo.example/sse",
    }

    monkeypatch.setattr(
        batch_eval,
        "load_planner_provider_config",
        lambda: provider,
    )
    monkeypatch.setattr(
        batch_eval,
        "load_mcp_server_url",
        lambda name, **_kwargs: urls.get(name, ""),
    )
    monkeypatch.setattr(
        batch_eval,
        "load_configured_web_access",
        lambda **_kwargs: WebAccessConfig(),
    )
    monkeypatch.setattr(batch_eval, "SseSimulatorMcpTransport", FakeSimulatorTransport)
    monkeypatch.setattr(
        cli_module,
        "_load_mcp_url",
        lambda name, **_kwargs: urls.get(name, ""),
    )
    monkeypatch.setattr(
        cli_module,
        "_ensure_simulator_mcp_transport",
        lambda _cli: FakeSimulatorTransport(),
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )

    tui = OpenEtaCli()
    tui.state.config = provider
    tui._build_runtime()
    batch = build_mcp_episode_worker_factory()(
        ParallelEpisodeSpec(
            episode_id="parity",
            task="inspect the scene",
            env_id="openeta/test-v0",
            metadata={"workspace_parent": str(tmp_path / "batch")},
        ),
        "parity-batch",
    )
    tui_runtime = tui._require_runtime()
    batch_runtime = batch.runner.runtime

    tui_executable = {
        spec.name for spec in tui_runtime.tools.list() if tui_runtime.tools.can_execute(spec.name)
    }
    batch_executable = {
        spec.name
        for spec in batch_runtime.tools.list()
        if batch_runtime.tools.can_execute(spec.name)
    }
    assert tui_executable == batch_executable
    assert tui_runtime.planner.max_validation_retries == 2
    assert batch_runtime.planner.max_validation_retries == 2
    assert tui.state.workspace is not None
    assert (
        tui.state.workspace.grasp_profile_id
        == batch.run_metadata["calibration_profile_id"]
    )
