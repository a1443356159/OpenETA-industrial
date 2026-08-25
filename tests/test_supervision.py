from __future__ import annotations

import math
from pathlib import Path

import pytest

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.backends.planner import (
    CallablePlannerBackend,
    PlannerBackendResult,
    StaticPlannerBackend,
)
from agent.runtime.episode import OpenEtaEpisodeRunner, ToolFeedbackEpisodeEnvironment
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.session_workspace import SessionWorkspace
from agent.runtime.supervision import (
    ACTION_REVIEW_SYSTEM_PROMPT,
    GUIDANCE_SYSTEM_PROMPT,
    BackendActionReviewer,
    BackendGuidanceResolver,
    SupervisionDecision,
    SupervisionGate,
    SupervisionPolicy,
    SupervisionProfile,
    _current_observation_rgb_paths,
)
from agent.tools.registry import ToolExecutionContext, ToolResult, build_default_tool_registry


def test_human_gated_world_action_fails_closed_before_handler() -> None:
    tools = build_default_tool_registry()
    called = []
    tools.bind_handler(
        "move_to",
        lambda _context: called.append(True) or ToolResult(True, "moved"),
    )
    gate = SupervisionGate(
        SupervisionPolicy.for_profile(SupervisionProfile.HUMAN_GATED),
        human_approval=lambda _context: False,
    )
    tools.set_execution_gate(gate.authorize)

    result = tools.call("move_to", {"x": 0.1, "y": 0.2, "z": 0.3})

    assert result.success is False
    assert result.details["diagnostics"][0]["code"] == "supervision_denied"
    assert called == []


def test_scripted_tui_world_action_is_explicitly_automated_not_human() -> None:
    tools = build_default_tool_registry()
    gate = SupervisionGate(
        SupervisionPolicy.for_profile(SupervisionProfile.SCRIPTED_TUI)
    )
    decision = gate.authorize(
        ToolExecutionContext(
            name="move_to",
            spec=tools.get("move_to"),
            parameters={"x": 0.1, "y": 0.2, "z": 0.3},
        )
    )

    assert decision.allowed is True
    assert decision.source == "scripted_tui"
    assert decision.details == {"profile": "scripted_tui", "automation": True}


def test_supervision_prompts_define_compact_machine_readable_outputs() -> None:
    assert '"decision":"approve|reject|abstain"' in ACTION_REVIEW_SYSTEM_PROMPT
    for label in ("answer:", "abstain:"):
        assert label in GUIDANCE_SYSTEM_PROMPT
    assert "empty observation.objects list alone is not evidence" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "overlay colors are synthetic" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "position=0 closes" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "exact terminal EEF contact pose" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "must not invent intermediate waypoints" in ACTION_REVIEW_SYSTEM_PROMPT
    assert '"grasp_outcome":"pass|fail|unknown|not_assessed"' in (
        ACTION_REVIEW_SYSTEM_PROMPT
    )
    assert "Articulated handles retain their separate bounded attachment probe" in (
        ACTION_REVIEW_SYSTEM_PROMPT
    )


def test_supervision_prefers_primary_scene_and_wrist_roles() -> None:
    observation = {
        "metadata": {
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "zed_head",
                    "role": "scene_primary",
                    "path": "zed.png",
                },
                {
                    "kind": "rgb",
                    "frame_id": "wrist_left",
                    "role": "wrist_secondary",
                    "path": "left.png",
                },
                {
                    "kind": "rgb",
                    "frame_id": "wrist_right",
                    "role": "wrist_primary",
                    "path": "right.png",
                },
            ]
        }
    }

    assert _current_observation_rgb_paths(
        observation,
        limit=2,
        prefer_grasp_views=True,
    ) == ["zed.png", "right.png"]
    assert _current_observation_rgb_paths(
        observation,
        limit=2,
        preferred_frame_id="wrist",
        preferred_role="wrist_primary",
        prefer_grasp_views=True,
    ) == ["right.png", "zed.png"]


def test_action_reviewer_allows_exact_host_articulated_probe() -> None:
    required_parameters = {
        "trajectory": [
            {
                "frame": "world",
                "xyz": [0.01, 0.0, 0.0],
                "probe_type": "articulated_attachment",
                "source_grasp_id": "handle-1",
                "probe_path_sha256": "a" * 64,
            }
        ]
    }
    backend = StaticPlannerBackend(
        {
            "decision": "reject",
            "reason": "motion evidence is not available yet",
            "grasp_outcome": "unknown",
            "candidate_id": "handle-1",
        }
    )
    tools = build_default_tool_registry()
    reviewer = BackendActionReviewer(backend)
    context = ToolExecutionContext(
        name="follow_eef_trajectory",
        spec=tools.get("follow_eef_trajectory"),
        parameters=required_parameters,
            observation=EnvObservation(
                task="open the microwave",
                cameras=[],
                robot=RobotState(),
            metadata={
                "image_artifacts": [
                    {"kind": "rgb", "frame_id": "agentview", "path": "agent.png"},
                    {"kind": "rgb", "frame_id": "wrist", "path": "wrist.png"},
                ]
            },
        ),
        metadata={
            "task": "open the microwave",
            "supervision_context": {
                "memory": {
                    "grasp_execution": {
                        "status": "required",
                        "stage": "probe",
                        "candidate_id": "handle-1",
                    },
                    "articulated_attachment_probe": {
                        "status": "required",
                        "candidate_id": "handle-1",
                        "distance_m": 0.05,
                        "path_sha256": "a" * 64,
                        "required_action": {
                            "name": "follow_eef_trajectory",
                            "parameters": required_parameters,
                        },
                    },
                }
            },
        },
    )

    decision = reviewer.review(context)

    assert decision.allowed is True
    assert decision.details["review_contract_override"]["schema_version"] == (
        "openeta.fixed_articulated_probe_contract.v1"
    )


def test_action_reviewer_receives_grasp_candidate_policy_with_empty_object_list() -> None:
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(
            payload={"decision": "approve", "reason": "provenance is consistent"}
        )

    tools = build_default_tool_registry()
    context = ToolExecutionContext(
        name="move_to",
        spec=tools.get("move_to"),
        parameters={"target_pose": {"id": "grasp_001", "frame": "world"}},
        observation=EnvObservation(
            task="pick cube",
            cameras=[CameraFrame(frame_id="agentview", rgb=[])],
            robot=RobotState(),
            objects=[],
            metadata={
                "image_artifacts": [
                    {
                        "kind": "rgb",
                        "frame_id": "agentview",
                        "path": "current-agentview.png",
                    }
                ]
            },
        ),
        metadata={
            "supervision_context": {
                "memory": {
                    "grasp_candidate_policy": {
                        "status": "active",
                        "active_rank": 1,
                        "active_candidate": {"id": "grasp_001"},
                        "target_detection": {
                            "id": "detection_000",
                            "overlay_ref": "target-overlay.png",
                            "source_image": "target-source.png",
                        },
                    }
                },
                "vision_image_paths": ["current-agentview.png"],
            }
        },
    )

    decision = BackendActionReviewer(CallablePlannerBackend(decide)).review(context)

    assert decision.allowed is True
    tool_context = requests[0].tool_context
    assert tool_context["observation"]["objects"] == []
    assert tool_context["grasp_candidate_policy"]["active_rank"] == 1
    assert tool_context["grasp_candidate_policy"]["target_detection"]["id"] == ("detection_000")
    assert tool_context["vision_image_paths"] == [
        "current-agentview.png",
        "target-source.png",
    ]
    assert tool_context["vision_evidence"] == [
        {"role": "current_scene", "path": "current-agentview.png"},
        {"role": "target_source_before_grasp", "path": "target-source.png"},
    ]


def test_action_reviewer_keeps_two_current_views_only_for_articulated_probe() -> None:
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(
            payload={
                "decision": "abstain",
                "reason": "both current views are required",
                "grasp_outcome": "unknown",
                "candidate_id": "g",
            }
        )

    tools = build_default_tool_registry()
    context = ToolExecutionContext(
        name="move_to",
        spec=tools.get("move_to"),
            parameters={
                "target_pose": {
                    "frame": "world",
                    "probe_type": "articulated_attachment",
                    "xyz": [0, 0, 1],
                }
            },
        observation=EnvObservation(
            task="pick bowl",
            cameras=[CameraFrame(frame_id="agentview", rgb=[]), CameraFrame(frame_id="wrist", rgb=[])],
            robot=RobotState(),
            objects=[],
            metadata={
                "image_artifacts": [
                    {"kind": "rgb", "frame_id": "agentview", "path": "current-agentview.png"},
                    {"kind": "rgb", "frame_id": "wrist", "path": "current-wrist.png"},
                ]
            },
        ),
        metadata={
            "supervision_context": {
                "memory": {
                        "grasp_execution": {
                            "status": "required",
                            "stage": "probe",
                            "candidate_id": "g",
                            "attachment_mode": "articulated_handle",
                        },
                        "articulated_attachment_probe": {
                            "status": "required",
                            "candidate_id": "g",
                        },
                }
            }
        },
    )

    decision = BackendActionReviewer(CallablePlannerBackend(decide)).review(context)

    assert decision.allowed is False
    assert requests[0].tool_context["vision_image_paths"] == [
        "current-wrist.png",
        "current-agentview.png",
    ]
    assert requests[0].tool_context["vision_evidence"] == [
        {"role": "current_scene", "path": "current-wrist.png"},
        {"role": "current_scene", "path": "current-agentview.png"},
    ]


def test_action_reviewer_receives_alternate_view_reestimate_obligation() -> None:
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(
            payload={
                "decision": "approve",
                "reason": "The exact fresh observation enables alternate-view re-estimation.",
            }
        )

    tools = build_default_tool_registry()
    parameters = {}
    recovery = {
        "schema_version": "openeta.grasp_recovery.v1",
        "status": "required",
        "candidate_id": "grasp_000",
        "reestimate_strategy": "alternate_camera_view",
        "previous_view": "agentview",
        "required_action": {"name": "observe", "parameters": parameters},
    }
    context = ToolExecutionContext(
        name="observe",
        spec=tools.get("observe"),
        parameters=parameters,
        observation=EnvObservation(
            task="pick cube",
            cameras=[],
            robot=RobotState(end_effector_pose={"xyz": [0.10, 0.20, 0.04]}),
        ),
        metadata={"supervision_context": {"memory": {"grasp_recovery": recovery}}},
    )

    decision = BackendActionReviewer(CallablePlannerBackend(decide)).review(context)

    assert decision.allowed is True
    assert requests[0].tool_context["grasp_recovery"] == recovery
    assert "new SAM/model inference" in requests[0].system_prompt


def test_action_reviewer_prioritizes_exact_asset_reference_for_close_identity() -> None:
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(
            payload={
                "decision": "reject",
                "reason": "The blue can appears separated from the gripper.",
                "grasp_outcome": "not_assessed",
                "candidate_id": "",
            }
        )

    tools = build_default_tool_registry()
    context = ToolExecutionContext(
        name="gripper_control",
        spec=tools.get("gripper_control"),
        parameters={"position": 0},
        observation=EnvObservation(
            task="pick alphabet soup",
            cameras=[CameraFrame(frame_id="agentview", rgb=[])],
            robot=RobotState(),
            metadata={
                "image_artifacts": [
                    {
                        "kind": "rgb",
                        "frame_id": "agentview",
                        "path": "current-agentview.png",
                    }
                ]
            },
        ),
        metadata={
            "supervision_context": {
                "memory": {
                    "target_asset_reference": {
                        "target_object": "alphabet_soup",
                        "scene_image": "verified-wrist.png",
                        "reference_images": [
                            "reference_front.png",
                            "reference_side.png",
                            "reference_top.png",
                        ],
                        "exact_instance_verification": {
                            "decision": "match",
                            "confidence": 0.93,
                            "reason": "The current lid matches the asset reference.",
                        },
                    },
                    "selected_sam3_detection": {
                        "id": "detection_000",
                        "mask_ref": "verified-mask.png",
                        "source_image": "verified-wrist.png",
                        "selection_source": "reference_verified",
                    },
                    "grasp_candidate_policy": {
                        "target_detection": {
                            "source_image": "target-source.png",
                            "overlay_ref": "target-overlay.png",
                        }
                    },
                    "grasp_execution": {
                        "status": "required",
                        "stage": "close",
                        "candidate_id": "grasp_002",
                        "required_action": {
                            "name": "gripper_control",
                            "parameters": {"position": 0},
                        },
                    },
                }
            }
        },
    )

    decision = BackendActionReviewer(CallablePlannerBackend(decide)).review(context)

    assert decision.allowed is True
    tool_context = requests[0].tool_context
    assert tool_context["vision_image_paths"] == [
        "current-agentview.png",
        "reference_side.png",
    ]
    assert tool_context["vision_evidence"] == [
        {"role": "current_scene", "path": "current-agentview.png"},
        {"role": "exact_asset_reference", "path": "reference_side.png"},
    ]
    assert tool_context["target_asset_reference"]["exact_instance_verification"][
        "decision"
    ] == "match"
    assert tool_context["host_action_stage"]["required_action_matches"] is True
    assert decision.details["decision"] == "approve"
    override = decision.details["review_contract_override"]
    assert override["candidate_id"] == "grasp_002"
    assert override["original_review"]["decision"] == "reject"
    assert override["reason"] == (
        "native_contact_and_attach_ack_own_portable_attachment_proof"
    )






def test_action_reviewer_prioritizes_current_observation_rgb() -> None:
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(payload={"decision": "approve", "reason": "consistent"})

    tools = build_default_tool_registry()
    observation = EnvObservation(
        task="pick soup can",
        cameras=[CameraFrame(frame_id="agentview", rgb=[])],
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "agentview",
                    "path": "current-agentview.png",
                }
            ]
        },
    )
    context = ToolExecutionContext(
        name="move_to",
        spec=tools.get("move_to"),
        parameters={"target_pose": {"id": "grasp_000", "frame": "world"}},
        observation=observation,
        metadata={
            "supervision_context": {
                "memory": {"overlay_ref": "synthetic-overlay.png"},
                "vision_image_paths": ["stale-scene.png"],
            }
        },
    )

    decision = BackendActionReviewer(CallablePlannerBackend(decide)).review(context)

    assert decision.allowed is True
    assert requests[0].tool_context["vision_image_paths"] == [
        "current-agentview.png",
        "synthetic-overlay.png",
    ]
    assert requests[0].tool_context["tool_contract"] == {
        "description": "Move the end effector to one world-frame target pose through the controller.",
        "effect": "world_mutating",
        "parameters": tools.get("move_to").parameters,
    }




def test_reviewed_action_gate_uses_independent_reviewer() -> None:
    class Reviewer:
        def review(self, context):
            return SupervisionDecision(
                context.name == "move_to",
                "independent_reviewer",
                "bounded action is task-consistent",
                {"isolated_context": True},
            )

    gate = SupervisionGate(
        SupervisionPolicy.for_profile(SupervisionProfile.REVIEWED_AUTONOMY),
        action_reviewer=Reviewer(),
    )

    tools = build_default_tool_registry()
    tools.bind_handler("move_to", lambda _context: ToolResult(True, "moved"))
    tools.set_execution_gate(gate.authorize)

    result = tools.call("move_to", {"x": 0.1, "y": 0.2, "z": 0.3})

    assert result.success is True
    assert result.details["supervision"]["source"] == "independent_reviewer"
    assert result.details["supervision"]["details"]["isolated_context"] is True


def test_action_reviewer_structures_candidate_linked_failed_grasp() -> None:
    backend = CallablePlannerBackend(
        lambda _request: PlannerBackendResult(
            payload={
                "decision": "approve",
                "reason": "The target stayed on the table; reopen for recovery.",
                "grasp_outcome": "failed",
                "candidate_id": "grasp_000",
            }
        )
    )
    tools = build_default_tool_registry()
    context = ToolExecutionContext(
        name="gripper_control",
        spec=tools.get("gripper_control"),
        parameters={"position": 1},
        observation=EnvObservation(task="pick cube", cameras=[], robot=RobotState()),
        metadata={
            "supervision_context": {
                "memory": {
                    "grasp_candidate_policy": {
                        "status": "accepted",
                        "active_candidate": {"id": "grasp_000"},
                    }
                }
            }
        },
    )

    decision = BackendActionReviewer(backend).review(context)

    assert decision.allowed is True
    assert decision.details["grasp_outcome"] == "fail"
    assert decision.details["candidate_id"] == "grasp_000"


def test_action_reviewer_marks_exact_fallback_candidate_restart_open() -> None:
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(
            payload={
                "decision": "approve",
                "reason": "Exact pre-contact open edge for the fresh fallback candidate.",
                "grasp_outcome": "not_assessed",
                "candidate_id": "",
            }
        )

    tools = build_default_tool_registry()
    context = ToolExecutionContext(
        name="gripper_control",
        spec=tools.get("gripper_control"),
        parameters={"position": 1},
        observation=EnvObservation(task="pick cube", cameras=[], robot=RobotState()),
        metadata={
            "supervision_context": {
                "memory": {
                    "grasp_execution": {
                        "candidate_id": "grasp_001",
                        "stage": "open",
                        "required_action": {
                            "name": "gripper_control",
                            "parameters": {"position": 1},
                        },
                    },
                    "grasp_candidate_policy": {
                        "active_candidate": {"id": "grasp_001", "rank": 1},
                        "last_rejection": {
                            "candidate_id": "grasp_000",
                            "reason": "target_not_reached",
                        },
                    },
                }
            }
        },
    )

    decision = BackendActionReviewer(CallablePlannerBackend(decide)).review(context)

    assert decision.allowed is True
    assert requests[0].tool_context["host_action_stage"] == {
        "schema_version": "openeta.host_action_stage.v1",
        "candidate_id": "grasp_001",
        "stage": "open",
        "required_action": {
            "name": "gripper_control",
            "parameters": {"position": 1},
        },
        "required_action_matches": True,
        "phase": "candidate_restart_after_structured_rejection",
        "previous_candidate_id": "grasp_000",
        "previous_rejection_reason": "target_not_reached",
    }




def test_action_reviewer_receives_only_exact_model_release_evidence() -> None:
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(
            payload={
                "decision": "approve",
                "reason": "The exact host-qualified action matches.",
                "grasp_outcome": "not_assessed",
                "candidate_id": "",
            }
        )

    tools = build_default_tool_registry()
    release_pose = {
        "frame": "world",
        "xyz": [0.08, 0.29, 0.19],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "placement_stage": "release",
    }
    memory = {
        "grasp_execution": {
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_003",
        },
        "attachment_gate": {"status": "resolved", "verdict": "PASS"},
        "placement_candidate_policy": {
            "status": "active",
            "active_candidate_id": "place_007",
            "compiled_placement": {"release_pose": release_pose},
        },
    }
    context = ToolExecutionContext(
        name="move_to",
        spec=tools.get("move_to"),
        parameters={"target_pose": dict(release_pose)},
        observation=EnvObservation(task="pick and place", cameras=[], robot=RobotState()),
        metadata={"supervision_context": {"memory": memory}},
    )

    decision = BackendActionReviewer(CallablePlannerBackend(decide)).review(context)

    assert decision.allowed is True
    assert requests[-1].tool_context["placement_action_stage"] == {
        "schema_version": "openeta.placement_action_stage.v2",
        "stage": "attached_transport_to_exact_release",
        "candidate_id": "grasp_003",
        "placement_pose_id": "place_007",
        "exact_release_pose": release_pose,
        "required_action_matches": True,
        "path_owner": "moveit",
        "host_pose_offsets_allowed": False,
    }

    memory["placement_release"] = {
        "status": "ready",
        "placement_pose_id": "place_007",
        "release_pose": release_pose,
    }
    context.name = "gripper_control"
    context.spec = tools.get("gripper_control")
    context.parameters = {"position": 1}
    BackendActionReviewer(CallablePlannerBackend(decide)).review(context)
    release_stage = requests[-1].tool_context["placement_action_stage"]
    assert release_stage["stage"] == "release"
    assert release_stage["exact_release_pose"] == release_pose
    assert release_stage["post_release_retreat_required"] is False
    assert "safe_hover_xyz" not in release_stage
    assert "release_clearance_m" not in release_stage


def test_guidance_agent_resolves_ask_human_inline_without_human_provenance() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "response",
                    "name": "ask_human",
                    "parameters": {"question": "Which object should I pick?"},
                },
                {
                    "kind": "response",
                    "name": "task_complete",
                    "parameters": {"success": True},
                },
            ]
        )
    )
    resolver = BackendGuidanceResolver(
        StaticPlannerBackend(
            {
                "decision": "answer",
                "answer": "Pick the red cube.",
                "reason": "The task metadata identifies it.",
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner)
    runner = OpenEtaEpisodeRunner(
        runtime=runtime,
        environment=ToolFeedbackEpisodeEnvironment(),
        interaction_resolver=resolver,
    )

    result = runner.run(task="pick the designated object", max_turns=3)

    assert result.terminated is True
    assert result.metadata["waiting_for_human"] is False
    assert result.metadata["assistance"]["guidance_intervention_count"] == 1
    assert runtime.memory.latest_human_interaction() is None
    assert runtime.memory.latest_guidance_interaction()["answer"] == "Pick the red cube."


def test_session_workspace_snapshots_skills_and_separates_owned_roots(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_skills"
    source.mkdir()
    (source / "pick.md").write_text(
        "---\nname: pick\ndescription: Pick objects.\n---\n\nObserve first.\n",
        encoding="utf-8",
    )

    first = SessionWorkspace.create("session-a", root=tmp_path / "workspaces", source_skills=source)
    second = SessionWorkspace.create(
        "session-b", root=tmp_path / "workspaces", source_skills=source
    )
    (first.skills_dir / "pick.md").write_text("session-a change\n", encoding="utf-8")

    assert first.root != second.root
    assert (second.skills_dir / "pick.md").read_text(encoding="utf-8").startswith("---")
    assert first.root == tmp_path / "workspaces" / "sessions" / "session-a"
    assert first.memory_root == tmp_path / "workspaces"
    assert first.working_dir == first.root / "working"
    assert first.artifacts_dir.parent == first.root
    assert first.sandbox_dir.parent == first.root
    assert first.grasp_strategy_root.parent == first.strategies_dir
    assert first.task_playbooks_dir.parent == first.root
    first_playbook = (
        first.task_playbooks_dir
        / "candidate"
        / "libero-object-task0-alphabet-soup.json"
    )
    second_playbook = (
        second.task_playbooks_dir
        / "candidate"
        / "libero-object-task0-alphabet-soup.json"
    )
    assert first_playbook.is_file()
    first_playbook.write_text(
        first_playbook.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert first_playbook.read_text(encoding="utf-8") != second_playbook.read_text(
        encoding="utf-8"
    )
    strategy = first.grasp_strategy_root / "candidate" / "top-down-vertical-panda-p8.json"
    assert strategy.is_file()
    strategy.write_text(strategy.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    second_strategy = (
        second.grasp_strategy_root / "candidate" / "top-down-vertical-panda-p8.json"
    )
    assert strategy.read_text(encoding="utf-8") != second_strategy.read_text(encoding="utf-8")


def test_session_workspace_migrates_legacy_paused_roots(tmp_path: Path) -> None:
    source_skills = tmp_path / "source_skills"
    source_skills.mkdir()
    legacy_memory = tmp_path / "legacy_memory"
    legacy_artifacts = tmp_path / "legacy_artifacts"
    legacy_memory.mkdir()
    legacy_artifacts.mkdir()
    (legacy_memory / "trace.jsonl").write_text("{}\n", encoding="utf-8")
    (legacy_artifacts / "frame.png").write_bytes(b"png")

    workspace = SessionWorkspace.create(
        "legacy-session",
        root=tmp_path / "workspaces",
        source_skills=source_skills,
    )
    workspace.import_legacy_roots(
        memory_root=legacy_memory,
        artifact_root=legacy_artifacts,
    )

    assert (workspace.root / "trace.jsonl").exists()
    assert (workspace.artifacts_dir / "frame.png").read_bytes() == b"png"
    (BackendActionReviewer,)


def test_session_workspace_imports_nested_legacy_memory_store(tmp_path: Path) -> None:
    source_skills = tmp_path / "source_skills"
    source_skills.mkdir()
    legacy_memory = tmp_path / "workspaces" / "legacy-session" / "memory"
    legacy_store = JsonMemoryStore(root=legacy_memory)
    legacy_store.start_session(
        session_id="legacy-session",
        task="legacy task",
        metadata={"layout": "workspace"},
    )
    legacy_store.session_path("legacy-session").write_text(
        '{"event_type":"legacy","timestamp_s":1.0,"payload":{}}\n',
        encoding="utf-8",
    )

    workspace = SessionWorkspace.create(
        "legacy-session",
        root=tmp_path / "canonical",
        source_skills=source_skills,
    )
    workspace.import_legacy_roots(memory_root=legacy_memory)

    target_store = JsonMemoryStore(root=workspace.memory_root)
    assert target_store.session_dir("legacy-session") == workspace.root
    assert target_store.session_path("legacy-session").read_text(encoding="utf-8").startswith(
        '{"event_type":"legacy"'
    )
    assert target_store.load_session_metadata("legacy-session")["task"] == "legacy task"
