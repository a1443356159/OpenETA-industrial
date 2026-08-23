"""Shared runtime assembly for interactive and batch OpenETA entry points."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable, Mapping
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.runtime.artifact_paths import artifact_session_id
from agent.runtime.response_artifacts import materialize_json_response
from agent.backends.planner import PlannerBackend
from agent.backends.provider_config import PlannerProviderConfig
from agent.runtime.calibration import (
    BackendCalibrationReviewer,
    CalibrationLifecycleConfig,
    CalibrationLifecycleManager,
)
from agent.runtime.checkers import CheckerSubagentConfig
from agent.runtime.grasp_strategy_lifecycle import (
    BackendGraspStrategyReviewer,
    GraspStrategyLifecycleConfig,
    GraspStrategyLifecycleManager,
)
from agent.runtime.memory import AgentMemory
from agent.runtime.moveit_qualification import (
    MoveItCandidateQualifier,
    QualificationCache,
    PRIVATE_RPC_NAME,
    private_qualification_rpc,
)
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerContextConfig, ToolCallingPlanner
from agent.runtime.reference_localization import (
    REFERENCE_POINT_LOCALIZATION_MAX_OUTPUT_TOKENS,
    BackendReferencePointLocalizer,
)
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.self_improvement import (
    BackendReviewedSkillAutoApplier,
    SelfImprovementConfig,
    SelfImprovementReviewer,
    SkillReviewProposalStore,
)
from agent.runtime.session_workspace import SessionWorkspace
from agent.runtime.skill_authoring import (
    SKILL_AUTHORING_MAX_OUTPUT_TOKENS,
    BackendSkillAuthoringSubagent,
    BackendSkillChangeReviewer,
    SkillAuthoringRequest,
)
from agent.runtime.skills import SkillSpec
from agent.runtime.supervision import (
    BackendActionReviewer,
    SupervisionGate,
    SupervisionPolicy,
    SupervisionProfile,
)
from agent.tools.asset_references import (
    build_asset_reference_handler,
    build_object_memory_configuration_warning_handler,
    build_object_memory_reference_handler,
    load_configured_asset_reference_catalog,
)
from agent.tools.attachment_probe import (
    build_assess_attachment_probe_handler,
    build_prepare_attachment_probe_handler,
)
from agent.tools.coding import PythonExecConfig, PythonExecRuntime
from agent.tools.depth_prefetch import DepthPriorPrefetchCoordinator
from agent.tools.handlers import (
    DEFAULT_GRASP_POSE_BACKEND_ORDER,
    bind_dummy_tool_handlers,
    build_anygrasp_handler,
    build_anyplace_handler,
    build_depth_prior_handler,
    build_grasp_pose_estimate_handler,
    build_graspgenx_handler,
    build_molmopoint_handler,
    build_oracle_perceive_segmenter,
    build_sam3_handler,
    build_sse_anygrasp_mcp_grasper,
    build_sse_anyplace_mcp_placer,
    build_sse_contact_graspnet_mcp_predictor,
    build_sse_depth_prior_mcp_estimator,
    build_sse_graspgenx_mcp_gripper_lister,
    build_sse_graspgenx_mcp_predictor,
    build_sse_molmopoint_mcp_pointer,
    build_sse_sam3_mcp_segmenter,
)
from agent.tools.grasp_geometry import (
    build_compile_grasp_seed_handler,
    build_compile_placement_seed_handler,
    build_wrist_alignment_handler,
    compile_placement_seed,
    compile_grasp_seed,
    materialize_world_object_goal,
    materialize_world_object_goal_from_current_pose,
    pregrasp_eef_goal_from_object_motion,
    qualification_grasp_pose_chain,
)
from agent.tools.grasp_strategies import load_grasp_strategies
from agent.tools.mcp_registry import load_mcp_server_url
from agent.tools.object_memory import (
    ObjectMemoryBankClient,
    ObjectMemoryBankConfigurationError,
    load_configured_object_memory_bank,
)
from agent.tools.registry import (
    ToolEventListener,
    ToolExecutionContext,
    ToolHandler,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    build_default_tool_registry,
    perception_segmenter_tool_name,
    resolve_perception_profile,
)
from agent.tools.sim_mcp import (
    SimulatorMcpResponseCallback,
    SimulatorMcpToolProxyConfig,
    SimulatorMcpTransport,
    bind_simulator_mcp_tool_handlers,
)
from agent.tools.web_access import WebAccessConfig, bind_configured_web_tool_handlers
from tools.candidate_config import CandidateFunnelConfig


BackendFactory = Callable[..., PlannerBackend]
McpUrlLoader = Callable[..., str]
ApprovalCallback = Callable[[ToolExecutionContext], bool]
PublicationApproval = Callable[[JsonDict], bool]
SkillApproval = Callable[[str], bool]

CONTRACTUAL_FAKE_CANDIDATE_ENV_VAR = "OPENETA_CONTRACTUAL_FAKE_CANDIDATE"
CONTRACTUAL_FAKE_CANDIDATE_SCHEMA = "openeta.contractual_fake_grasp_candidate.v1"


@dataclass(slots=True)
class _OracleMcpEvidence:
    """One-use correlated evidence record for the Oracle simulator MCP RPC."""

    proxy_config: SimulatorMcpToolProxyConfig
    response_output_root: Path
    pending: tuple[str, JsonDict, JsonDict] | None = None

    def record(self, remote_tool: str, arguments: JsonDict, response: JsonDict) -> None:
        descriptor_arguments = dict(arguments)
        image_base64 = descriptor_arguments.pop("image_base64", None)
        if isinstance(image_base64, str):
            descriptor_arguments["image_base64_sha256"] = hashlib.sha256(
                image_base64.encode("ascii", errors="ignore")
            ).hexdigest()
            descriptor_arguments["image_base64_chars"] = len(image_base64)
        self.pending = (remote_tool, descriptor_arguments, dict(response))

    def attach(self, result: ToolResult, context: ToolExecutionContext) -> ToolResult:
        pending = self.pending
        self.pending = None
        if pending is None:
            return result
        remote_tool, arguments, response_payload = pending
        request_id = uuid4().hex
        session_id = str(self.proxy_config.session_id or "")
        handle = str(self.proxy_config.handle or "")
        artifact = materialize_json_response(
            response_payload,
            output_root=self.response_output_root,
            bundle_id=f"oracle-{request_id}",
            name=f"{remote_tool}-response",
            session_id=artifact_session_id(context.metadata),
        )
        response: JsonDict = {
            "response_path": artifact.path,
            "response_chars": artifact.chars,
            "response_omitted": True,
            "grep_hint": artifact.grep_hint,
            "request_id": request_id,
            "tool": remote_tool,
            "session_id": session_id,
            "handle": handle,
        }
        request: JsonDict = {
            "request_id": request_id,
            "tool": remote_tool,
            "arguments": arguments,
        }
        receipt: JsonDict = {
            "schema_version": "openeta.environment_receipt.v1",
            "receipt_id": uuid4().hex,
            "backend": "simulator_mcp",
            "agent_tool": "oracle_perceive",
            "remote_tool": remote_tool,
            "mcp_request_id": request_id,
            "execution_id": str(context.metadata.get("execution_id") or ""),
            "agent_session_id": str(context.metadata.get("session_id") or ""),
            "simulator_session_id": session_id,
            "handle": handle,
            "timestamp_s": time.time(),
            "reward_present": "reward" in response_payload,
            "observation_fresh": False,
        }
        details = dict(result.details)
        artifacts = details.get("artifacts")
        details["artifacts"] = [
            *(artifacts if isinstance(artifacts, list) else []),
            artifact.to_dict(),
        ]
        details["mcp"] = {
            "tool": remote_tool,
            "agent_tool": "oracle_perceive",
            "session_id": session_id,
            "handle": handle,
            "request": request,
            "response": response,
        }
        details["response"] = response
        details["mcp_calls"] = [
            {
                "request": request,
                "response": response,
                "environment_receipt": receipt,
            }
        ]
        result.details = details
        return result


def _with_contractual_fake_candidate(
    handler: Callable[[ToolExecutionContext], ToolResult],
    *,
    mcp_evidence: _OracleMcpEvidence | None = None,
) -> Callable[[ToolExecutionContext], ToolResult]:
    """Add the oracle-fixture fixture marker to a successful *Oracle* tool result only.

    This is deliberately data-only: no pose is inferred or acted upon and the
    marker is never read by the native-grasp attachment path.  It records that the oracle-fixture
    candidate is a contractual fake rather than a visual-model prediction.
    """

    def wrapped(context: ToolExecutionContext) -> ToolResult:
        result = handler(context)
        if not result.success:
            return result
        if mcp_evidence is not None:
            result = mcp_evidence.attach(result, context)
        details = dict(result.details)
        result_id = str(details.get("result_id") or "oracle-result")
        details["perception_source"] = "gazebo_oracle"
        details["fake_grasp_candidate"] = {
            "schema_version": CONTRACTUAL_FAKE_CANDIDATE_SCHEMA,
            "kind": "contractual_fake_grasp_candidate",
            "candidate_id": f"contractual-{result_id}",
            "perception_source": "gazebo_oracle",
            "is_model_prediction": False,
            "provenance": "oracle_contract_fixture",
            "oracle_result_id": result_id,
        }
        result.details = details
        return result

    return wrapped


REMOTE_PLACEHOLDER_TOOLS = (
    "scene_detector",
    "sam3",
    "anygrasp",
    "grasp_pose_estimate",
    "contact_graspnet",
    "graspgenx",
    "list_graspgenx_grippers",
    "hand_pose_database",
    "obstacle_avoidance",
    "lower_body_control_policy",
    "estimate_depth_prior",
)

ENVIRONMENT_PLACEHOLDER_TOOLS = (
    "observe",
    "move_to",
    "follow_eef_trajectory",
    "gripper_control",
)


@dataclass(frozen=True, slots=True)
class RuntimeMcpEndpoints:
    """Configured remote perception backends for one runtime."""

    sam3_url: str = ""
    depth_prior_url: str = ""
    anygrasp_url: str = ""
    anyplace_url: str = ""
    graspgenx_url: str = ""
    contact_graspnet_url: str = ""
    molmopoint_url: str = ""


@dataclass(frozen=True, slots=True)
class RuntimeCandidateCounts:
    """Host registration values that must match remote service metadata."""

    graspgenx_raw_pool_size: int = 200
    anygrasp_raw_pool_size: int = 200
    anyplace_raw_pool_size: int = 96
    grasp_diversity_pool_size: int = 64
    anyplace_diversity_pool_size: int = 96
    grasp_full_plan_limit: int = 2
    anyplace_full_plan_limit: int = 2
    pregrasp_joint_grasp_branch_limit: int = 4
    pregrasp_joint_full_plan_limit: int = 2
    moveit_ik_seed_count: int = 8
    anyplace_max_qualification_rounds: int = 2

    def __post_init__(self) -> None:
        validated = CandidateFunnelConfig(
            graspgenx_raw_pool_size=self.graspgenx_raw_pool_size,
            anygrasp_raw_pool_size=self.anygrasp_raw_pool_size,
            anyplace_raw_pool_size=self.anyplace_raw_pool_size,
            grasp_diversity_pool_size=self.grasp_diversity_pool_size,
            anyplace_diversity_pool_size=self.anyplace_diversity_pool_size,
            grasp_full_plan_limit=self.grasp_full_plan_limit,
            anyplace_full_plan_limit=self.anyplace_full_plan_limit,
            pregrasp_joint_grasp_branch_limit=self.pregrasp_joint_grasp_branch_limit,
            pregrasp_joint_full_plan_limit=self.pregrasp_joint_full_plan_limit,
            moveit_ik_seed_count=self.moveit_ik_seed_count,
            anyplace_max_qualification_rounds=self.anyplace_max_qualification_rounds,
        )
        for name in (
            "graspgenx_raw_pool_size", "anygrasp_raw_pool_size", "anyplace_raw_pool_size",
            "grasp_diversity_pool_size", "anyplace_diversity_pool_size",
            "grasp_full_plan_limit", "anyplace_full_plan_limit", "moveit_ik_seed_count",
            "pregrasp_joint_grasp_branch_limit", "pregrasp_joint_full_plan_limit",
            "anyplace_max_qualification_rounds",
        ):
            object.__setattr__(self, name, getattr(validated, name))


def runtime_candidate_counts_from_env() -> RuntimeCandidateCounts:
    return RuntimeCandidateCounts(
        graspgenx_raw_pool_size=os.environ.get("OPENETA_GRASPGENX_RAW_POOL_SIZE", 200),
        anygrasp_raw_pool_size=os.environ.get("OPENETA_ANYGRASP_RAW_POOL_SIZE", 200),
        anyplace_raw_pool_size=os.environ.get("OPENETA_ANYPLACE_RAW_POOL_SIZE", 96),
        grasp_diversity_pool_size=os.environ.get("OPENETA_GRASP_DIVERSITY_POOL_SIZE", 64),
        anyplace_diversity_pool_size=os.environ.get("OPENETA_ANYPLACE_DIVERSITY_POOL_SIZE", 96),
        grasp_full_plan_limit=os.environ.get("OPENETA_GRASP_FULL_PLAN_LIMIT", 2),
        anyplace_full_plan_limit=os.environ.get("OPENETA_ANYPLACE_FULL_PLAN_LIMIT", 2),
        pregrasp_joint_grasp_branch_limit=os.environ.get(
            "OPENETA_PREGRASP_JOINT_GRASP_BRANCH_LIMIT", 4
        ),
        pregrasp_joint_full_plan_limit=os.environ.get(
            "OPENETA_PREGRASP_JOINT_FULL_PLAN_LIMIT", 2
        ),
        moveit_ik_seed_count=os.environ.get("OPENETA_MOVEIT_IK_SEED_COUNT", 8),
        anyplace_max_qualification_rounds=os.environ.get("OPENETA_ANYPLACE_MAX_QUALIFICATION_ROUNDS", 2),
    )


@dataclass(slots=True)
class RuntimeAssemblyConfig:
    """Host-owned inputs shared by TUI and batch runtime construction."""

    workspace: SessionWorkspace
    provider: PlannerProviderConfig
    backend_factory: BackendFactory
    supervision_policy: SupervisionPolicy
    endpoints: RuntimeMcpEndpoints = field(default_factory=RuntimeMcpEndpoints)
    candidate_counts: RuntimeCandidateCounts = field(
        default_factory=runtime_candidate_counts_from_env
    )
    simulator_transport: SimulatorMcpTransport | None = None
    simulator_proxy_config: SimulatorMcpToolProxyConfig | None = None
    web_access_config: WebAccessConfig | None = None
    mcp_response_callback: SimulatorMcpResponseCallback | None = None
    allow_outside_sandbox: bool = False
    approve_outside_sandbox: Callable[[ToolExecutionContext, str], bool] | None = None
    human_action_approval: ApprovalCallback | None = None
    calibration_approval: PublicationApproval | None = None
    strategy_approval: PublicationApproval | None = None
    skill_approval: SkillApproval | None = None
    supervision_policy_provider: Callable[[], SupervisionPolicy] | None = None
    pre_safety_checks: dict[str, str] = field(default_factory=dict)
    tool_listeners: tuple[ToolEventListener, ...] = ()
    max_validation_retries: int = 2


@dataclass(frozen=True, slots=True)
class RuntimeAssembly:
    """Assembled runtime plus host controls retained by its entry point."""

    runtime: OpenEtaAgentRuntime
    supervision_gate: SupervisionGate
    depth_prefetch: DepthPriorPrefetchCoordinator | None


def resolve_runtime_mcp_endpoints(
    overrides: RuntimeMcpEndpoints | None = None,
    *,
    loader: McpUrlLoader = load_mcp_server_url,
) -> RuntimeMcpEndpoints:
    """Resolve shared perception endpoint names and aliases."""

    configured = overrides or RuntimeMcpEndpoints()
    return RuntimeMcpEndpoints(
        sam3_url=configured.sam3_url
        or loader("openeta-sam3", aliases=("sam3",)),
        depth_prior_url=configured.depth_prior_url
        or loader(
            "openeta-depth-prior",
            aliases=("depth-prior", "depth_prior", "unidepth"),
        ),
        anygrasp_url=configured.anygrasp_url
        or loader("openeta-anygrasp", aliases=("anygrasp",)),
        anyplace_url=configured.anyplace_url
        or loader("openeta-anyplace", aliases=("anyplace",)),
        graspgenx_url=configured.graspgenx_url
        or loader("openeta-graspgenx", aliases=("graspgenx",)),
        contact_graspnet_url=configured.contact_graspnet_url
        or loader(
            "openeta-contact-graspnet",
            aliases=("contact-graspnet", "contact_graspnet"),
        ),
        molmopoint_url=configured.molmopoint_url
        or loader(
            "openeta-molmopoint",
            aliases=("molmopoint", "molmo-point"),
        ),
    )


def assemble_runtime(config: RuntimeAssemblyConfig) -> RuntimeAssembly:
    """Build one fail-closed runtime from shared host configuration."""

    workspace = config.workspace
    simulator_proxy_config = config.simulator_proxy_config or SimulatorMcpToolProxyConfig()
    tools = bind_dummy_tool_handlers(
        build_default_tool_registry(),
        include_dummy_safety=False,
    )
    registered_tool_names = {spec.name for spec in tools.list()}
    for name in REMOTE_PLACEHOLDER_TOOLS:
        if name in registered_tool_names:
            tools.unbind_handler(name)
    if config.simulator_transport is None:
        for name in ENVIRONMENT_PLACEHOLDER_TOOLS:
            tools.unbind_handler(name)

    qualification_cache = QualificationCache()
    qualifier = _runtime_candidate_qualifier(
        config.simulator_transport,
        simulator_proxy_config,
        cache=qualification_cache,
        artifact_root=config.workspace.artifacts_dir / "moveit_qualification",
        compile_candidate=_candidate_qualification_compiler(config.workspace),
        candidate_counts=config.candidate_counts,
    )
    internal_candidate_compilers: dict[str, ToolHandler] = {}
    if qualifier is not None:
        internal_candidate_compilers["grasp"] = build_compile_grasp_seed_handler(
            workspace.grasp_profile_path,
            strategy_root=workspace.grasp_strategy_root,
            qualification_cache=qualification_cache,
        )
        internal_candidate_compilers["placement"] = build_compile_placement_seed_handler(
            workspace.grasp_profile_path,
            qualification_cache=qualification_cache,
        )
    wrist_handler = build_wrist_alignment_handler()
    if qualifier is not None:
        wrist_handler = _qualifying_wrist_alignment_handler(wrist_handler, qualifier)
    tools.bind_handler(
        "compute_wrist_alignment",
        wrist_handler,
        replace=True,
    )
    tools.bind_handler(
        "prepare_attachment_probe",
        build_prepare_attachment_probe_handler(),
        replace=True,
    )
    tools.bind_handler(
        "assess_attachment_probe",
        build_assess_attachment_probe_handler(
            config.backend_factory(max_tokens=256, max_vision_images=4)
        ),
        replace=True,
    )
    if config.simulator_transport is not None:
        bind_simulator_mcp_tool_handlers(
            tools,
            transport=config.simulator_transport,
            config=simulator_proxy_config,
            response_callback=config.mcp_response_callback,
            replace=True,
        )

    artifact_root = workspace.artifacts_dir
    tools.bind_handler(
        "python_exec",
        PythonExecRuntime(
            PythonExecConfig(
                mcp_transport=config.simulator_transport,
                image_output_root=str(artifact_root / "images"),
                text_output_root=str(artifact_root / "text"),
                response_output_root=str(artifact_root / "responses"),
                allow_outside_sandbox=config.allow_outside_sandbox,
                approve_outside_sandbox=config.approve_outside_sandbox,
                mcp_response_callback=config.mcp_response_callback,
                workspace_root=str(workspace.sandbox_dir),
            )
        ).handler,
        replace=True,
    )

    policy_provider = config.supervision_policy_provider or (
        lambda: config.supervision_policy
    )
    calibration_manager = CalibrationLifecycleManager(
        config=CalibrationLifecycleConfig(
            root=workspace.calibrations_dir,
            evidence_roots=(Path(".openeta_memory"), workspace.root),
            publication_mode=lambda: policy_provider().skill_change_mode,
            human_approval=config.calibration_approval,
        ),
        reviewer=BackendCalibrationReviewer(config.backend_factory()),
    )
    tools.bind_handler(
        "propose_calibration_profile",
        calibration_manager.propose_handler,
        replace=True,
    )
    tools.bind_handler(
        "promote_calibration_profile",
        calibration_manager.promote_handler,
        replace=True,
    )

    staged_profile = workspace.grasp_profile()
    strategy_lifecycle_root = workspace.working_dir / "grasp_strategy_lifecycle"
    strategy_manager = GraspStrategyLifecycleManager(
        config=GraspStrategyLifecycleConfig(
            root=strategy_lifecycle_root,
            session_strategy_root=strategy_lifecycle_root / "staged",
            calibration_profile=staged_profile,
            evidence_roots=(Path(".openeta_memory"), workspace.root),
            publication_mode=lambda: policy_provider().skill_change_mode,
            human_approval=config.strategy_approval,
        ),
        reviewer=BackendGraspStrategyReviewer(config.backend_factory()),
    )
    tools.bind_handler(
        "propose_grasp_strategy",
        strategy_manager.propose_handler,
        replace=True,
    )
    tools.bind_handler(
        "promote_grasp_strategy",
        strategy_manager.promote_handler,
        replace=True,
    )

    bind_configured_web_tool_handlers(
        tools,
        config=config.web_access_config,
        provider_config=config.provider,
    )
    depth_prefetch = bind_runtime_perception_tools(
        tools,
        endpoints=config.endpoints,
        backend_factory=config.backend_factory,
        artifact_root=artifact_root,
        simulator_transport=config.simulator_transport,
        simulator_proxy_config=simulator_proxy_config,
        candidate_qualifier=qualifier,
        candidate_counts=config.candidate_counts,
        internal_candidate_compilers=internal_candidate_compilers,
    )

    planner = ToolCallingPlanner(
        config.backend_factory(),
        max_validation_retries=config.max_validation_retries,
        context_config=PlannerContextConfig(
            context_window_tokens=config.provider.context_window_tokens,
            token_estimator_model=config.provider.model,
        ),
    )
    skill_review_config = SelfImprovementConfig(
        proposal_root=workspace.working_dir / "skill_reviews" / "pending",
        auto_apply_reviewed=(
            config.supervision_policy.profile == SupervisionProfile.REVIEWED_AUTONOMY
        ),
        skill_dir=workspace.skills_dir,
        task_playbook_candidate_root=workspace.working_dir / "task_playbook_reviews",
        rollout_root=workspace.memory_root,
    )
    skill_reviewer = SelfImprovementReviewer(
        config=skill_review_config,
        store=SkillReviewProposalStore(skill_review_config.proposal_root),
    )
    checker_config = _checker_config(
        tools,
        pre_safety_checks=config.pre_safety_checks,
    )
    runtime = OpenEtaAgentRuntime(
        planner=planner,
        tools=tools,
        memory=AgentMemory(store=JsonMemoryStore(root=workspace.memory_root)),
        skills=workspace.skill_registry(),
        pipeline=ActionPipeline(checker_subagents=checker_config),
        self_improvement_reviewer=skill_reviewer,
        default_session_id=workspace.session_id,
    )
    configure_runtime_self_improvement(
        runtime,
        policy=config.supervision_policy,
        backend_factory=config.backend_factory,
    )
    _bind_skill_change_tools(
        runtime,
        backend_factory=config.backend_factory,
        policy_provider=policy_provider,
        human_approval=config.skill_approval,
    )

    gate = SupervisionGate(
        config.supervision_policy,
        human_approval=config.human_action_approval,
        action_reviewer=BackendActionReviewer(config.backend_factory(max_tokens=512)),
    )
    tools.set_execution_gate(gate.authorize)
    for listener in config.tool_listeners:
        tools.add_listener(listener)
    return RuntimeAssembly(
        runtime=runtime,
        supervision_gate=gate,
        depth_prefetch=depth_prefetch,
    )


def configure_runtime_self_improvement(
    runtime: OpenEtaAgentRuntime,
    *,
    policy: SupervisionPolicy,
    backend_factory: BackendFactory,
) -> None:
    """Apply the active supervision profile to reviewed skill auto-apply."""

    reviewer = runtime.self_improvement_reviewer
    reviewed = policy.profile == SupervisionProfile.REVIEWED_AUTONOMY
    reviewer.config = replace(
        reviewer.config,
        auto_apply_reviewed=reviewed,
    )
    if not reviewed:
        reviewer.auto_applier = None
        return
    reviewer.auto_applier = BackendReviewedSkillAutoApplier(
        author=BackendSkillAuthoringSubagent(
            backend_factory(max_tokens=SKILL_AUTHORING_MAX_OUTPUT_TOKENS)
        ),
        reviewer=BackendSkillChangeReviewer(backend_factory()),
        executable_tools=_skill_authoring_tools(runtime.tools),
    )


def bind_runtime_perception_tools(
    tools: ToolRegistry,
    *,
    endpoints: RuntimeMcpEndpoints,
    backend_factory: BackendFactory,
    artifact_root: Path,
    perception_profile: str | None = None,
    simulator_transport: SimulatorMcpTransport | None = None,
    simulator_proxy_config: SimulatorMcpToolProxyConfig | None = None,
    candidate_qualifier: MoveItCandidateQualifier | None = None,
    candidate_counts: RuntimeCandidateCounts | None = None,
    internal_candidate_compilers: Mapping[str, ToolHandler] | None = None,
) -> DepthPriorPrefetchCoordinator | None:
    counts = candidate_counts or runtime_candidate_counts_from_env()
    pregrasp_coordinator = (
        _PregraspGraspPlaceCoordinator(
            candidate_qualifier,
            grasp_branch_limit=counts.pregrasp_joint_grasp_branch_limit,
        )
        if candidate_qualifier is not None
        else None
    )
    segmenter_tool = perception_segmenter_tool_name(
        resolve_perception_profile()
        if perception_profile is None
        else perception_profile
    )
    object_memory_configuration_error = ""
    try:
        object_memory_config = load_configured_object_memory_bank()
    except ObjectMemoryBankConfigurationError as exc:
        object_memory_config = None
        object_memory_configuration_error = str(exc)
    asset_catalog = load_configured_asset_reference_catalog()
    if object_memory_config is not None:
        tools.bind_handler(
            "retrieve_asset_reference",
            build_object_memory_reference_handler(
                ObjectMemoryBankClient(object_memory_config),
                BackendReferencePointLocalizer(
                    backend_factory(
                        max_tokens=REFERENCE_POINT_LOCALIZATION_MAX_OUTPUT_TOKENS,
                        max_vision_images=4,
                    )
                ),
                output_root=artifact_root / "asset_references",
            ),
            replace=True,
        )
    elif asset_catalog is not None and not object_memory_configuration_error:
        tools.bind_handler(
            "retrieve_asset_reference",
            build_asset_reference_handler(
                asset_catalog,
                output_root=artifact_root / "asset_references",
            ),
            replace=True,
        )
    else:
        tools.bind_handler(
            "retrieve_asset_reference",
            build_object_memory_configuration_warning_handler(
                configuration_error=object_memory_configuration_error,
            ),
            replace=True,
        )

    depth_prefetch: DepthPriorPrefetchCoordinator | None = None
    if endpoints.depth_prior_url:
        depth_handler = build_depth_prior_handler(
            build_sse_depth_prior_mcp_estimator(url=endpoints.depth_prior_url),
            output_root=artifact_root / "depth_prior_results",
        )
        depth_prefetch = DepthPriorPrefetchCoordinator(
            depth_handler,
            spec=tools.get("estimate_depth_prior"),
        )
        tools.bind_handler(
            "estimate_depth_prior",
            depth_prefetch.handler,
            replace=True,
        )
    if segmenter_tool == "oracle_perceive":
        # Simulator-only oracle (Gazebo ground truth) reuses the SAM3 handler
        # pipeline over the existing simulator MCP transport; it is exposed
        # instead of sam3, never alongside it.
        if simulator_transport is not None:
            proxy_config = simulator_proxy_config or SimulatorMcpToolProxyConfig()
            oracle_mcp_evidence = _OracleMcpEvidence(
                proxy_config=proxy_config,
                response_output_root=Path(proxy_config.response_output_root),
            )
            oracle_handler = build_sam3_handler(
                build_oracle_perceive_segmenter(
                    simulator_transport,
                    handle_provider=lambda: proxy_config.handle,
                    session_id_provider=lambda: proxy_config.session_id,
                    response_callback=oracle_mcp_evidence.record,
                ),
                tool_name="oracle_perceive",
                output_root=artifact_root / "oracle_perceive_images",
                result_output_root=artifact_root / "oracle_perceive_results",
            )
            if os.environ.get(CONTRACTUAL_FAKE_CANDIDATE_ENV_VAR) == "1":
                oracle_handler = _with_contractual_fake_candidate(
                    oracle_handler,
                    mcp_evidence=oracle_mcp_evidence,
                )
            tools.bind_handler(
                "oracle_perceive",
                oracle_handler,
                replace=True,
            )
    elif endpoints.sam3_url:
        tools.bind_handler(
            "sam3",
            build_sam3_handler(
                build_sse_sam3_mcp_segmenter(url=endpoints.sam3_url),
                segment_points=build_sse_sam3_mcp_segmenter(
                    url=endpoints.sam3_url,
                    tool_name="segment_points",
                ),
                depth_prior_prefetch=(
                    depth_prefetch.prefetch_for_sam3
                    if depth_prefetch is not None
                    else None
                ),
                output_root=artifact_root / "sam3_images",
                result_output_root=artifact_root / "sam3_results",
            ),
            replace=True,
        )
    if endpoints.molmopoint_url:
        tools.bind_handler(
            "molmopoint",
            build_molmopoint_handler(
                build_sse_molmopoint_mcp_pointer(url=endpoints.molmopoint_url),
                output_root=artifact_root / "molmopoint_results",
            ),
            replace=True,
        )
    if endpoints.anyplace_url:
        anyplace_handler = build_anyplace_handler(
            build_sse_anyplace_mcp_placer(url=endpoints.anyplace_url),
            output_root=artifact_root / "anyplace_results",
            expected_raw_pool_size=counts.anyplace_raw_pool_size,
            pre_inference=(
                lambda context, request: _prepare_postattachment_frozen_goals(
                    context,
                    request,
                    coordinator=pregrasp_coordinator,
                )
                if pregrasp_coordinator is not None
                else None
            ),
        )
        if candidate_qualifier is not None:
            anyplace_handler = _qualifying_handler(
                anyplace_handler,
                candidate_qualifier,
                purpose="placement",
                pregrasp_coordinator=pregrasp_coordinator,
                candidate_compiler=(internal_candidate_compilers or {}).get(
                    "placement"
                ),
            )
        tools.bind_handler(
            "anyplace",
            anyplace_handler,
            replace=True,
        )

    grasp_backends = {}
    if endpoints.anygrasp_url:
        grasp_backends["anygrasp"] = build_anygrasp_handler(
            build_sse_anygrasp_mcp_grasper(url=endpoints.anygrasp_url),
            output_root=artifact_root / "anygrasp_results",
            expected_raw_pool_size=counts.anygrasp_raw_pool_size,
        )
    if endpoints.graspgenx_url:
        list_grippers = build_sse_graspgenx_mcp_gripper_lister(
            url=endpoints.graspgenx_url
        )
        grasp_backends["graspgenx"] = build_graspgenx_handler(
            build_sse_graspgenx_mcp_predictor(url=endpoints.graspgenx_url),
            list_grippers,
            output_root=artifact_root / "graspgenx_results",
            expected_raw_pool_size=counts.graspgenx_raw_pool_size,
        )
    # Resolve the configured Contact-GraspNet endpoint for startup/discovery,
    # but keep the backend disabled until its planner-facing contract is
    # explicitly re-enabled for the target deployment.
    if endpoints.contact_graspnet_url:
        build_sse_contact_graspnet_mcp_predictor(url=endpoints.contact_graspnet_url)
    # Contact-GraspNet is temporarily disabled for the simulator drawer track.
    # Keep its endpoint/configuration and implementation available for a later
    # re-enable, but do not expose it as an executable grasp backend here.
    if grasp_backends:
        grasp_handler = build_grasp_pose_estimate_handler(
            grasp_backends,
            backend_order=DEFAULT_GRASP_POSE_BACKEND_ORDER,
            graspgenx_gripper_name="robotiq_2f_85",
        )
        if candidate_qualifier is not None:
            grasp_handler = _qualifying_handler(
                grasp_handler,
                candidate_qualifier,
                purpose="grasp",
                pregrasp_coordinator=pregrasp_coordinator,
                candidate_compiler=(internal_candidate_compilers or {}).get("grasp"),
            )
        tools.bind_handler(
            "grasp_pose_estimate",
            grasp_handler,
            replace=True,
        )
    return depth_prefetch


def _runtime_candidate_qualifier(
    transport: SimulatorMcpTransport | None,
    proxy_config: SimulatorMcpToolProxyConfig,
    *,
    cache: QualificationCache,
    artifact_root: Path,
    compile_candidate: Callable,
    candidate_counts: RuntimeCandidateCounts | None = None,
) -> MoveItCandidateQualifier | None:
    """Discover the private RPC without adding it to the planner tool registry."""

    if transport is None:
        return None
    try:
        listing = transport.list_tools(timeout_s=5.0)
    except Exception:  # noqa: BLE001 - optional private capability discovery.
        return None
    tools_value = listing.get("tools") if isinstance(listing, dict) else None
    names = {
        str(item.get("name") or "")
        for item in tools_value or []
        if isinstance(item, dict)
    }
    if PRIVATE_RPC_NAME not in names:
        return None
    counts = candidate_counts or runtime_candidate_counts_from_env()
    return MoveItCandidateQualifier(
        private_qualification_rpc(
            transport,
            handle_provider=lambda: proxy_config.handle,
            session_id_provider=lambda: proxy_config.session_id,
        ),
        cache=cache,
        artifact_root=artifact_root,
        compile_candidate=compile_candidate,
        grasp_diversity_limit=counts.grasp_diversity_pool_size,
        placement_diversity_limit=counts.anyplace_diversity_pool_size,
        grasp_full_plan_limit=counts.grasp_full_plan_limit,
        placement_full_plan_limit=counts.anyplace_full_plan_limit,
        pregrasp_joint_full_plan_limit=counts.pregrasp_joint_full_plan_limit,
        ik_seed_count=counts.moveit_ik_seed_count,
        placement_max_rounds=counts.anyplace_max_qualification_rounds,
    )


def _candidate_qualification_compiler(
    workspace: SessionWorkspace,
) -> Callable:
    """Compile immutable candidates to the exact world poses sent to MoveIt."""

    profile_path = Path(workspace.grasp_profile_path)

    def compile_with_snapshot(
        candidate: Mapping[str, object],
        purpose: str,
        source: Mapping[str, object],
        scene_epoch: int,
        planning_scene_revision: int,
        *,
        profile: Mapping[str, object],
        profile_sha256: str,
        strategies: list[JsonDict],
    ) -> JsonDict:
        if purpose == "placement":
            pregrasp_contact = candidate.get("pregrasp_contact_pose")
            if isinstance(pregrasp_contact, Mapping):
                eef_goal = pregrasp_eef_goal_from_object_motion(
                    contact_pose=pregrasp_contact,
                    placement_candidate=candidate,
                )
                compiled_candidate = dict(candidate)
                compiled_candidate["object_goal_pose"] = eef_goal
                attachment_transform: object = {
                    "schema_version": "openeta.pregrasp_eef_identity.v1",
                    "parent_frame": "eef",
                    "child_frame": "object",
                    "translation_xyz": [0.0, 0.0, 0.0],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            else:
                compiled_candidate = dict(candidate)
                attachment_transform = source.get("attachment_transform")
            if not isinstance(attachment_transform, dict):
                raise ValueError("measured attachment transform unavailable")
            start_joint_state = candidate.get(
                "qualification_start_joint_state",
                source.get("start_joint_state"),
            )
            if not isinstance(start_joint_state, dict):
                raise ValueError("placement qualification start state unavailable")
            parameters: JsonDict = {
                "placement_candidate_id": str(candidate.get("id") or ""),
                "placement_candidate": compiled_candidate,
                "attachment_transform": dict(attachment_transform),
                "scene_epoch": scene_epoch,
                "scene_revision": planning_scene_revision,
                "qualified_attachment_transform_sha256": hashlib.sha256(
                    json.dumps(
                        attachment_transform, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "qualified_start_state_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "joint_positions": start_joint_state.get("joint_positions", []),
                            "gripper_state": start_joint_state.get("gripper_state", {}),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            compiled = compile_placement_seed(
                parameters,
                profile=profile,
                profile_sha256=profile_sha256,
            )
            compiled_pose_chain = [
                dict(compiled["hover_pose"]),
                dict(compiled["release_pose"]),
            ]
            retreat_pose = dict(compiled["release_pose"])
            retreat_xyz = retreat_pose.get("xyz")
            if not isinstance(retreat_xyz, list) or len(retreat_xyz) != 3:
                raise ValueError("compiled placement release pose is invalid")
            retreat_pose["xyz"] = [
                float(retreat_xyz[0]),
                float(retreat_xyz[1]),
                float(retreat_xyz[2]) + 0.1,
            ]
            compiled_pose_chain.append(retreat_pose)
            stages = [
                _qualification_pose("hover", compiled["hover_pose"]),
                {
                    **_qualification_pose("release", compiled["release_pose"]),
                    "scene_transition": "virtual_detach",
                },
                _qualification_pose("retreat", retreat_pose),
            ]
        else:
            extrinsics = source.get("camera_extrinsics")
            if not isinstance(extrinsics, dict):
                raise ValueError("camera extrinsics unavailable for qualification")
            parameters = {
                "purpose": "grasp",
                "camera_pose": dict(candidate),
                "camera_extrinsics": dict(extrinsics),
                "camera_frame_id": str(source.get("camera_frame_id") or ""),
                "scene_epoch": scene_epoch,
            }
            compiled = compile_grasp_seed(
                parameters,
                profile=profile,
                profile_sha256=profile_sha256,
                strategies=strategies,
            )
            compiled_pose_chain = qualification_grasp_pose_chain(compiled)
            stage_names = (
                ("hover", "precontact", "contact", "lift")
                if isinstance(compiled.get("precontact_pose"), Mapping)
                else ("hover", "contact", "lift")
            )
            stages = [
                _qualification_pose(name, pose)
                for name, pose in zip(stage_names, compiled_pose_chain, strict=True)
            ]
            for stage in stages:
                if stage["name"] == "contact":
                    stage["scene_transition"] = "virtual_attach"
        return {
            "qualification_stages": stages,
            "compile_parameters": {
                **parameters,
                "qualification_profile_sha256": profile_sha256,
                "qualified_compiled_pose_sha256": hashlib.sha256(
                    json.dumps(
                        (
                            compiled_pose_chain[:2]
                            if purpose == "placement"
                            else compiled_pose_chain
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "qualified_stage_pose_sha256": hashlib.sha256(
                    json.dumps(
                        compiled_pose_chain,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            },
        }

    def prepare_batch(*, purpose: str) -> Callable:
        """Freeze calibration and strategies once for one qualification batch."""

        if purpose not in {"grasp", "placement"}:
            raise ValueError("candidate qualification purpose is invalid")
        profile_bytes = profile_path.read_bytes()
        profile = json.loads(profile_bytes.decode("utf-8"))
        profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
        strategies = (
            load_grasp_strategies(Path(workspace.grasp_strategy_root))
            if purpose == "grasp"
            else []
        )

        def prepared(
            candidate: Mapping[str, object],
            candidate_purpose: str,
            source: Mapping[str, object],
            scene_epoch: int,
            planning_scene_revision: int,
        ) -> JsonDict:
            if candidate_purpose != purpose:
                raise ValueError("prepared candidate compiler purpose changed")
            return compile_with_snapshot(
                candidate,
                candidate_purpose,
                source,
                scene_epoch,
                planning_scene_revision,
                profile=profile,
                profile_sha256=profile_sha256,
                strategies=strategies,
            )

        return prepared

    def compile_one(
        candidate: Mapping[str, object],
        purpose: str,
        source: Mapping[str, object],
        scene_epoch: int,
        planning_scene_revision: int,
    ) -> JsonDict:
        return prepare_batch(purpose=purpose)(
            candidate,
            purpose,
            source,
            scene_epoch,
            planning_scene_revision,
        )

    # MoveItCandidateQualifier detects this host-private hook and prepares one
    # immutable snapshot before compiling the batch.  Plain compiler callables
    # remain supported for tests and other integrations.
    setattr(compile_one, "prepare_batch", prepare_batch)
    return compile_one


def _qualification_pose(name: str, pose: object) -> JsonDict:
    if not isinstance(pose, Mapping):
        raise ValueError("compiled qualification pose is invalid")
    result = {"name": name, **dict(pose)}
    rotation = result.get("rotation_matrix")
    if not isinstance(rotation, list) or len(rotation) != 3:
        raise ValueError("compiled qualification rotation is missing")
    m = [[float(value) for value in row] for row in rotation]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        quat = [
            (m[2][1] - m[1][2]) / scale,
            (m[0][2] - m[2][0]) / scale,
            (m[1][0] - m[0][1]) / scale,
            0.25 * scale,
        ]
    else:
        index = max(range(3), key=lambda item: m[item][item])
        if index == 0:
            scale = (1.0 + m[0][0] - m[1][1] - m[2][2]) ** 0.5 * 2.0
            quat = [0.25 * scale, (m[0][1] + m[1][0]) / scale, (m[0][2] + m[2][0]) / scale, (m[2][1] - m[1][2]) / scale]
        elif index == 1:
            scale = (1.0 + m[1][1] - m[0][0] - m[2][2]) ** 0.5 * 2.0
            quat = [(m[0][1] + m[1][0]) / scale, 0.25 * scale, (m[1][2] + m[2][1]) / scale, (m[0][2] - m[2][0]) / scale]
        else:
            scale = (1.0 + m[2][2] - m[0][0] - m[1][1]) ** 0.5 * 2.0
            quat = [(m[0][2] + m[2][0]) / scale, (m[1][2] + m[2][1]) / scale, 0.25 * scale, (m[1][0] - m[0][1]) / scale]
    result["quat_xyzw"] = quat
    return result


@dataclass(slots=True)
class _PregraspGraspPlaceCoordinator:
    """Host-private bounded look-ahead used only to rank executable grasps."""

    qualifier: MoveItCandidateQualifier
    grasp_branch_limit: int = 4
    object_goals: list[JsonDict] = field(default_factory=list)
    object_current_pose: JsonDict | None = None
    scene_epoch: int = -1
    planning_scene_revision: int = -1
    qualified_goals_by_grasp: dict[str, list[JsonDict]] = field(default_factory=dict)
    consumed_attachment_bindings: set[str] = field(default_factory=set)
    consumed_model_retry_bindings: set[str] = field(default_factory=set)
    source_model_raw_candidate_count: int = 0
    source_candidate_image_ref: str = ""
    source_candidate_artifacts: list[JsonDict] = field(default_factory=list)

    def retain_goal_pool(
        self,
        result: ToolResult,
        *,
        source: Mapping[str, object],
        scene_epoch: int,
        planning_scene_revision: int,
    ) -> ToolResult:
        candidates = result.details.get("placement_candidates")
        current_pose = result.details.get("object_current_pose")
        extrinsics = source.get("placement_camera_extrinsics")
        if not (
            isinstance(candidates, list)
            and candidates
            and isinstance(current_pose, Mapping)
            and isinstance(extrinsics, Mapping)
        ):
            return ToolResult(
                False,
                "pregrasp placement goal binding evidence is incomplete",
                {"reason": "pregrasp_goal_binding_missing", "execution_started": False},
            )
        try:
            goals = [
                materialize_world_object_goal_from_current_pose(
                    candidate,
                    placement_camera_extrinsics=extrinsics,
                    object_current_pose=current_pose,
                )
                for candidate in candidates
                if isinstance(candidate, Mapping)
            ]
        except Exception as exc:  # noqa: BLE001 - fail closed at frame boundary.
            return ToolResult(
                False,
                f"pregrasp placement goal binding failed: {exc}",
                {"reason": "pregrasp_goal_binding_invalid", "execution_started": False},
            )
        self.object_goals = goals
        self.object_current_pose = dict(current_pose)
        self.scene_epoch = scene_epoch
        self.planning_scene_revision = planning_scene_revision
        self.qualified_goals_by_grasp.clear()
        self.consumed_attachment_bindings.clear()
        self.consumed_model_retry_bindings.clear()
        model_raw_count = result.details.get("model_raw_candidate_count")
        self.source_model_raw_candidate_count = (
            model_raw_count
            if isinstance(model_raw_count, int) and not isinstance(model_raw_count, bool)
            else len(candidates)
        )
        candidate_image_ref = result.details.get("candidate_image_ref")
        self.source_candidate_image_ref = (
            candidate_image_ref if isinstance(candidate_image_ref, str) else ""
        )
        artifacts = result.details.get("artifacts")
        self.source_candidate_artifacts = [
            json.loads(json.dumps(artifact))
            for artifact in artifacts
            if isinstance(artifact, Mapping)
        ] if isinstance(artifacts, list) else []
        # Raw goals remain host-private.  They are not executable placement
        # candidates and are never inserted into the placement qualification cache.
        result.details["placement_candidates"] = []
        result.details["candidate_count"] = 0
        result.details["selection_required"] = False
        result.details["pregrasp_goal_pool_ready"] = True
        result.details["pregrasp_goal_pool_count"] = len(goals)
        result.details["execution_started"] = False
        result.content = (
            f"Retained {len(goals)} host-private object goals for bounded "
            "pregrasp grasp-place qualification."
        )
        return result

    def filter_grasps(
        self,
        result: ToolResult,
        *,
        scene_epoch: int,
        planning_scene_revision: int,
        source: Mapping[str, object],
    ) -> ToolResult:
        grasps = result.details.get("grasp_candidates")
        if not isinstance(grasps, list) or not grasps:
            return result
        if not (
            self.object_goals
            and isinstance(self.object_current_pose, Mapping)
            and self.scene_epoch == scene_epoch
            and self.planning_scene_revision == planning_scene_revision
        ):
            # A stale or absent pool cannot influence grasp qualification.
            return result

        retained_entries: dict[str, JsonDict] = {}
        per_grasp_pairs: list[list[JsonDict]] = []
        # Preserve the complete current AnyPlace pool through compilation and
        # conservative structural screening.  Preselecting a handful by
        # farthest-first SE(3) overrepresents extreme object rotations and can
        # discard every attachment-aware reachable goal before MoveIt sees it.
        # L3/L4 then traverse the round-robin pair order progressively until
        # the two-slot plan-only capacity is filled or the batch is exhausted.
        current_goals = [dict(goal) for goal in self.object_goals]
        for grasp in grasps[: self.grasp_branch_limit]:
            if not isinstance(grasp, Mapping):
                continue
            grasp_id = str(grasp.get("id") or "")
            entry = self.qualifier.cache.resolve(
                purpose="grasp",
                candidate_id=grasp_id,
                scene_epoch=scene_epoch,
                planning_scene_revision=planning_scene_revision,
            )
            if not isinstance(entry, Mapping):
                continue
            proof = entry.get("proof")
            stages = proof.get("stages") if isinstance(proof, Mapping) else None
            if not isinstance(stages, list) or not stages:
                continue
            contact = next(
                (
                    stage.get("target_pose")
                    for stage in stages
                    if isinstance(stage, Mapping)
                    and str(stage.get("name") or "") == "contact"
                    and isinstance(stage.get("target_pose"), Mapping)
                ),
                None,
            )
            lift_state = stages[-1].get("end_joint_state") if isinstance(stages[-1], Mapping) else None
            if not isinstance(contact, Mapping) or not isinstance(lift_state, Mapping):
                continue
            retained_entries[grasp_id] = dict(entry)
            pairs: list[JsonDict] = []
            for goal in current_goals:
                pair = dict(goal)
                goal_id = str(goal.get("id") or "goal")
                pair["id"] = f"pregrasp_pair_{grasp_id}_{goal_id}"
                pair["source_grasp_id"] = grasp_id
                pair["source_object_goal_id"] = goal_id
                pair["pregrasp_contact_pose"] = dict(contact)
                pair["qualification_start_joint_state"] = dict(lift_state)
                pair["initial_scene_transition"] = "virtual_attach"
                pair["initial_scene_transition_pose"] = dict(contact)
                pairs.append(pair)
            per_grasp_pairs.append(pairs)

        # Round-robin ordering ensures the global top-two plan-only tail does
        # not get consumed by several goals from the first grasp alone.
        pair_depth = max((len(group) for group in per_grasp_pairs), default=0)
        pairs = [
            group[index]
            for index in range(pair_depth)
            for group in per_grasp_pairs
            if index < len(group)
        ]
        if not pairs:
            return self._replace_grasps(result, [], {}, scene_epoch, planning_scene_revision)
        joint = self.qualifier.qualify_result(
            ToolResult(
                True,
                "bounded pregrasp grasp-place qualification",
                {
                    "placement_candidates": pairs,
                    "model_raw_candidate_count": len(pairs),
                    "raw_candidate_count": len(pairs),
                },
            ),
            purpose="placement",
            scene_epoch=scene_epoch,
            planning_scene_revision=planning_scene_revision,
            source=source,
            cache_result=False,
            qualification_mode="pregrasp_joint",
        )
        passed_pairs = joint.details.get("placement_candidates")
        passed_pairs = passed_pairs if isinstance(passed_pairs, list) else []
        pass_count: dict[str, int] = {}
        goal_ids: dict[str, list[str]] = {}
        for pair in passed_pairs:
            if not isinstance(pair, Mapping):
                continue
            grasp_id = str(pair.get("source_grasp_id") or "")
            if not grasp_id:
                continue
            pass_count[grasp_id] = pass_count.get(grasp_id, 0) + 1
            goal_ids.setdefault(grasp_id, []).append(
                str(pair.get("source_object_goal_id") or "")
            )
        goal_lookup = {
            str(goal.get("id") or ""): goal
            for goal in self.object_goals
            if isinstance(goal, Mapping) and str(goal.get("id") or "")
        }
        self.qualified_goals_by_grasp = {
            grasp_id: [
                json.loads(json.dumps(goal_lookup[goal_id]))
                for goal_id in ids
                if goal_id in goal_lookup
            ]
            for grasp_id, ids in goal_ids.items()
        }
        retained: list[JsonDict] = []
        cache_grasps: list[JsonDict] = []
        proofs: dict[str, Mapping[str, object]] = {}
        for grasp in grasps:
            grasp_id = str(grasp.get("id") or "") if isinstance(grasp, Mapping) else ""
            entry = retained_entries.get(grasp_id)
            if not grasp_id or not entry or pass_count.get(grasp_id, 0) <= 0:
                continue
            annotated = dict(grasp)
            annotated["grasp_place_joint_qualified"] = True
            annotated["grasp_place_pass_count"] = pass_count[grasp_id]
            annotated["grasp_place_goal_ids"] = goal_ids.get(grasp_id, [])
            retained.append(annotated)
            cached_candidate = entry.get("candidate")
            if isinstance(cached_candidate, Mapping):
                cache_grasps.append(dict(cached_candidate))
            if isinstance(entry.get("proof"), Mapping):
                proofs[grasp_id] = entry["proof"]
        result.details["pregrasp_joint_pair_count"] = len(pairs)
        result.details["pregrasp_joint_workspace_pass_count"] = joint.details.get(
            "workspace_pass_count", 0
        )
        result.details["pregrasp_joint_endpoint_evaluated_count"] = joint.details.get(
            "endpoint_evaluated_count", 0
        )
        result.details["pregrasp_joint_endpoint_not_evaluated_count"] = joint.details.get(
            "endpoint_not_evaluated_count", 0
        )
        result.details["pregrasp_joint_endpoint_pass_count"] = joint.details.get(
            "endpoint_pass_count", 0
        )
        result.details["pregrasp_joint_full_plan_submitted_count"] = joint.details.get(
            "full_plan_submitted_count", 0
        )
        result.details["pregrasp_joint_full_plan_pass_count"] = joint.details.get(
            "full_plan_pass_count", 0
        )
        if joint.details.get("qualification_artifact"):
            result.details["pregrasp_joint_qualification_artifact"] = joint.details[
                "qualification_artifact"
            ]
        return self._replace_grasps(
            result,
            retained,
            proofs,
            scene_epoch,
            planning_scene_revision,
            cache_grasps=cache_grasps,
        )

    def prepare_frozen_goal_requalification(
        self,
        *,
        source_grasp_id: str,
        attachment_transform: Mapping[str, object],
        source: Mapping[str, object],
        scene_revision: int,
    ) -> ToolResult | None:
        """Prepare one measured-attachment qualification without model inference."""

        goals = self.qualified_goals_by_grasp.get(source_grasp_id)
        if not goals:
            return None
        binding = self.attachment_binding(
            source_grasp_id=source_grasp_id,
            attachment_transform=attachment_transform,
        )
        if binding in self.consumed_attachment_bindings:
            return None
        self.consumed_attachment_bindings.add(binding)
        frozen = [json.loads(json.dumps(goal)) for goal in goals]
        artifacts: list[JsonDict] = []
        for source_artifact in self.source_candidate_artifacts:
            artifact = json.loads(json.dumps(source_artifact))
            artifact["provenance"] = "frozen_pregrasp_anyplace_pool"
            artifact["reused_for_measured_attachment_requalification"] = True
            artifact["anyplace_model_inference_invoked"] = False
            artifacts.append(artifact)
        metadata = {
            "candidate_source": "frozen_pregrasp_pass_goals",
            "source_model_raw_candidate_count": self.source_model_raw_candidate_count,
            "anyplace_model_inference_invoked": False,
        }
        return ToolResult(
            True,
            (
                f"Prepared {len(frozen)} frozen pregrasp PASS object goals for "
                "measured-attachment qualification without AnyPlace inference."
            ),
            {
                "tool": "anyplace",
                "backend": "openeta_frozen_goal_requalifier",
                "frame": "world",
                "source": json.loads(json.dumps(source)),
                "scene_revision": scene_revision,
                "candidate_count": len(frozen),
                "model_raw_candidate_count": self.source_model_raw_candidate_count,
                "raw_candidate_count": len(frozen),
                "generated_candidate_count": len(frozen),
                "placement_candidates": frozen,
                "candidate_image_ref": self.source_candidate_image_ref,
                "artifacts": artifacts,
                "metadata": metadata,
                "frozen_pregrasp_goal_requalification": True,
                "frozen_pregrasp_goal_count": len(frozen),
                "discarded_postattach_model_candidate_count": 0,
                "anyplace_model_inference_invoked": False,
                "execution_started": False,
            },
        )

    def attachment_binding(
        self,
        *,
        source_grasp_id: str,
        attachment_transform: Mapping[str, object],
    ) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "source_grasp_id": source_grasp_id,
                    "attachment_transform": attachment_transform,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def has_frozen_goals(self, source_grasp_id: str) -> bool:
        return bool(self.qualified_goals_by_grasp.get(source_grasp_id))

    def attachment_binding_consumed(
        self,
        *,
        source_grasp_id: str,
        attachment_transform: Mapping[str, object],
    ) -> bool:
        return self.attachment_binding(
            source_grasp_id=source_grasp_id,
            attachment_transform=attachment_transform,
        ) in self.consumed_attachment_bindings

    def consume_model_retry_binding(
        self,
        *,
        source_grasp_id: str,
        attachment_transform: Mapping[str, object],
    ) -> bool:
        binding = self.attachment_binding(
            source_grasp_id=source_grasp_id,
            attachment_transform=attachment_transform,
        )
        if binding in self.consumed_model_retry_bindings:
            return False
        self.consumed_model_retry_bindings.add(binding)
        return True

    def _replace_grasps(
        self,
        result: ToolResult,
        grasps: list[JsonDict],
        proofs: Mapping[str, Mapping[str, object]],
        scene_epoch: int,
        planning_scene_revision: int,
        *,
        cache_grasps: list[JsonDict] | None = None,
    ) -> ToolResult:
        self.qualifier.cache.replace(
            purpose="grasp",
            candidates=grasps if cache_grasps is None else cache_grasps,
            proofs=proofs,
            scene_epoch=scene_epoch,
            planning_scene_revision=planning_scene_revision,
        )
        result.details["grasp_candidates"] = grasps
        result.details["candidate_count"] = len(grasps)
        result.details["full_plan_pass_count"] = len(grasps)
        result.details["selection_required"] = bool(grasps)
        result.details["pregrasp_joint_qualified_grasp_count"] = len(grasps)
        return result


def _prepare_postattachment_frozen_goals(
    context: ToolExecutionContext,
    request: JsonDict,
    *,
    coordinator: _PregraspGraspPlaceCoordinator,
) -> ToolResult | None:
    """Short-circuit model inference for the selected grasp's frozen PASS goals."""

    supervision = context.metadata.get("supervision_context")
    memory = supervision.get("memory") if isinstance(supervision, Mapping) else None
    if not isinstance(memory, Mapping):
        return None
    execution = memory.get("grasp_execution")
    attachment_gate = memory.get("attachment_gate")
    if not (
        isinstance(execution, Mapping)
        and execution.get("status") == "completed"
        and execution.get("stage") == "attached"
        and execution.get("attachment_mode") != "articulated_handle"
        and isinstance(attachment_gate, Mapping)
        and attachment_gate.get("status") == "resolved"
        and str(attachment_gate.get("verdict") or "").upper() == "PASS"
    ):
        return None
    full_proof = attachment_gate.get("full_lift_proof")
    attachment_transform = (
        full_proof.get("attachment_transform")
        if isinstance(full_proof, Mapping)
        else None
    )
    if not isinstance(attachment_transform, Mapping):
        return None
    compiled_source_grasp = execution.get("compiled_grasp")
    source_grasp_id = _active_source_grasp_id(
        memory,
        compiled_source_grasp=compiled_source_grasp,
    )
    if not source_grasp_id:
        return None
    scene_revision = request.get("scene_revision")
    if not isinstance(scene_revision, int) or isinstance(scene_revision, bool):
        return None
    attachment_revision = attachment_gate.get("planning_scene_revision")
    if (
        isinstance(attachment_revision, int)
        and not isinstance(attachment_revision, bool)
        and scene_revision != attachment_revision
    ):
        return ToolResult(
            False,
            "frozen placement qualification scene revision does not match attachment proof",
            {
                "reason": "frozen_placement_scene_revision_mismatch",
                "execution_started": False,
            },
        )
    if coordinator.attachment_binding_consumed(
        source_grasp_id=source_grasp_id,
        attachment_transform=attachment_transform,
    ):
        if _placement_model_retry_authorized(memory, request):
            if coordinator.consume_model_retry_binding(
                source_grasp_id=source_grasp_id,
                attachment_transform=attachment_transform,
            ):
                return None
            return ToolResult(
                False,
                "The bounded new-seed AnyPlace inference was already consumed.",
                {
                    "reason": "placement_model_retry_already_consumed",
                    "execution_started": False,
                },
            )
        return ToolResult(
            False,
            (
                "AnyPlace inference is blocked until a zero-PASS frozen-goal "
                "qualification authorizes the bounded new-seed retry."
            ),
            {
                "reason": "placement_model_retry_not_authorized",
                "execution_started": False,
            },
        )
    placement_observation = request.get("placement_observation")
    if not isinstance(placement_observation, Mapping):
        return None
    source = {
        "object_observation": json.loads(
            json.dumps(request.get("object_observation"))
        ),
        "placement_observation": json.loads(json.dumps(placement_observation)),
        "object_camera_to_placement_camera": json.loads(
            json.dumps(request.get("object_camera_to_placement_camera"))
        ),
        "placement_camera_to_world": json.loads(
            json.dumps(request.get("placement_camera_to_world"))
        ),
        "placement_camera_extrinsics": json.loads(
            json.dumps(placement_observation.get("camera_extrinsics"))
        ),
    }
    return coordinator.prepare_frozen_goal_requalification(
        source_grasp_id=source_grasp_id,
        attachment_transform=attachment_transform,
        source=source,
        scene_revision=scene_revision,
    )


def _placement_model_retry_authorized(
    memory: Mapping[str, object],
    request: Mapping[str, object],
) -> bool:
    policy = memory.get("placement_candidate_policy")
    if not (
        isinstance(policy, Mapping)
        and policy.get("status") == "qualification_retry_required"
    ):
        return False
    policy_revision = policy.get("planning_scene_revision", policy.get("scene_revision"))
    request_revision = request.get("scene_revision")
    if (
        not isinstance(policy_revision, int)
        or isinstance(policy_revision, bool)
        or policy_revision != request_revision
    ):
        return False
    recovery = policy.get("recovery")
    required_action = (
        recovery.get("required_action") if isinstance(recovery, Mapping) else None
    )
    required_parameters = (
        required_action.get("parameters")
        if isinstance(required_action, Mapping)
        and required_action.get("name") == "anyplace"
        else None
    )
    return isinstance(required_parameters, Mapping) and dict(required_parameters) == dict(
        request
    )


def _qualifying_handler(
    handler: ToolHandler,
    qualifier: MoveItCandidateQualifier,
    *,
    purpose: str,
    pregrasp_coordinator: _PregraspGraspPlaceCoordinator | None = None,
    candidate_compiler: ToolHandler | None = None,
) -> ToolHandler:
    """Apply private MoveIt qualification before a result reaches memory/VLM."""

    def qualified(context: ToolExecutionContext) -> ToolResult:
        supervision = context.metadata.get("supervision_context")
        memory = supervision.get("memory") if isinstance(supervision, dict) else None
        placement_policy = (
            memory.get("placement_candidate_policy")
            if isinstance(memory, dict)
            else None
        )
        if (
            purpose == "placement"
            and isinstance(placement_policy, dict)
            and placement_policy.get("status") == "stopped_requires_human"
        ):
            # Qualification is bounded.  Once the frozen pool is exhausted
            # and the return-to-source proof is unavailable, fresh model
            # samples cannot safely alter the current attached state.
            return ToolResult(
                False,
                "CURRENT_GRASP_PLACE_INFEASIBLE: placement recovery is blocked pending human intervention.",
                {
                    "reason": "CURRENT_GRASP_PLACE_INFEASIBLE",
                    "execution_started": False,
                    "qualification_round": placement_policy.get("qualification_round"),
                    "max_qualification_rounds": placement_policy.get("max_qualification_rounds"),
                },
            )
        result = handler(context)
        if not result.success:
            return result
        observation_metadata = (
            context.observation.metadata if context.observation is not None else {}
        )
        # Use the runtime invalidation epoch consistently for qualification,
        # cache storage, and later compilation.  A simulator observation can
        # report its reset epoch after the runtime has advanced its mutation
        # counter, and mixing the two makes a valid PASS proof unexecutable.
        scene_epoch_value = (
            memory.get("scene_epoch")
            if isinstance(memory, Mapping)
            and isinstance(memory.get("scene_epoch"), int)
            and not isinstance(memory.get("scene_epoch"), bool)
            else observation_metadata.get(
                "scene_epoch",
                result.details.get("scene_epoch", context.parameters.get("scene_epoch", 0)),
            )
        )
        scene_epoch = (
            scene_epoch_value
            if isinstance(scene_epoch_value, int) and not isinstance(scene_epoch_value, bool)
            else 0
        )
        revision_value = result.details.get(
            "scene_revision", context.parameters.get("scene_revision")
        )
        if revision_value is None:
            revision_value = observation_metadata.get("planning_scene_revision")
        if not isinstance(revision_value, int) or isinstance(revision_value, bool):
            revision_value = scene_epoch
        source = result.details.get("source")
        source = dict(source) if isinstance(source, dict) else {}
        if context.observation is not None:
            source["start_joint_state"] = context.observation.robot.to_dict()
            frame_id = str(source.get("camera_frame_id") or result.details.get("camera_frame_id") or "")
            camera = next(
                (
                    camera
                    for camera in context.observation.cameras
                    if not frame_id or camera.frame_id == frame_id
                ),
                None,
            )
            if camera is not None:
                source["camera_frame_id"] = camera.frame_id
                source["camera_extrinsics"] = dict(camera.extrinsics)
        if purpose == "placement":
            attachment_gate = memory.get("attachment_gate") if isinstance(memory, dict) else None
            full_proof = (
                attachment_gate.get("full_lift_proof")
                if isinstance(attachment_gate, dict)
                else None
            )
            attachment_transform = (
                full_proof.get("attachment_transform")
                if isinstance(full_proof, dict)
                else None
            )
            if (
                not isinstance(attachment_transform, dict)
                and pregrasp_coordinator is not None
            ):
                return pregrasp_coordinator.retain_goal_pool(
                    result,
                    source=source,
                    scene_epoch=scene_epoch,
                    planning_scene_revision=revision_value,
                )
            grasp_execution = (
                memory.get("grasp_execution") if isinstance(memory, dict) else None
            )
            compiled_source_grasp = (
                grasp_execution.get("compiled_grasp")
                if isinstance(grasp_execution, dict)
                else None
            )
            frozen_goal_requalification = (
                result.details.get("frozen_pregrasp_goal_requalification") is True
            )
            current_eef_pose = (
                context.observation.robot.end_effector_pose
                if context.observation is not None
                else None
            )
            placement_extrinsics = source.get("placement_camera_extrinsics")
            candidates = result.details.get("placement_candidates")
            already_world_goals = isinstance(candidates, list) and all(
                isinstance(candidate, Mapping)
                and isinstance(candidate.get("object_goal_pose"), Mapping)
                for candidate in candidates
            )
            if not (
                isinstance(attachment_transform, dict)
                and isinstance(current_eef_pose, dict)
                and isinstance(candidates, list)
                and (
                    already_world_goals
                    or isinstance(placement_extrinsics, dict)
                )
            ):
                return ToolResult(
                    False,
                    "placement qualification lacks measured attachment or observation pose",
                    {"reason": "placement_binding_evidence_missing"},
                )
            try:
                materialized = (
                    [dict(candidate) for candidate in candidates]
                    if already_world_goals
                    else [
                        materialize_world_object_goal(
                            candidate,
                            placement_camera_extrinsics=placement_extrinsics,
                            current_eef_pose=current_eef_pose,
                            attachment_transform=attachment_transform,
                        )
                        for candidate in candidates
                        if isinstance(candidate, Mapping)
                    ]
                )
            except Exception as exc:  # noqa: BLE001 - fail closed at evidence boundary.
                return ToolResult(
                    False,
                    f"placement object-goal binding failed: {exc}",
                    {"reason": "placement_binding_invalid"},
                )
            result.details["placement_candidates"] = materialized
            result.details["candidate_count"] = len(materialized)
            source["attachment_transform"] = dict(attachment_transform)
            source["current_eef_pose"] = dict(current_eef_pose)
            if frozen_goal_requalification:
                source["frozen_pregrasp_goal_requalification"] = True
            if isinstance(compiled_source_grasp, dict):
                source["source_grasp_compiled"] = dict(compiled_source_grasp)
        qualified_result = qualifier.qualify_result(
            result,
            purpose=purpose,
            scene_epoch=scene_epoch,
            planning_scene_revision=revision_value,
            source=source,
        )
        if purpose == "grasp" and pregrasp_coordinator is not None:
            qualified_result = pregrasp_coordinator.filter_grasps(
                qualified_result,
                scene_epoch=scene_epoch,
                planning_scene_revision=revision_value,
                source=source,
            )
        return _compile_qualified_queue(
            qualified_result,
            purpose=purpose,
            context=context,
            scene_epoch=scene_epoch,
            planning_scene_revision=revision_value,
            compiler=candidate_compiler,
        )

    return qualified


_HOST_CANDIDATE_COMPILER_SPEC = ToolSpec(
    name="host_candidate_compiler",
    category="host_workflow",
    description="Host-private qualified-candidate compilation transition.",
    parameters={},
    effect="read_only",
    batchable=False,
)


def _compile_qualified_queue(
    result: ToolResult,
    *,
    purpose: str,
    context: ToolExecutionContext,
    scene_epoch: int,
    planning_scene_revision: int,
    compiler: ToolHandler | None,
) -> ToolResult:
    """Compile an equal-status PASS queue and activate its stable head."""

    if not result.success or compiler is None:
        return result
    key = "placement_candidates" if purpose == "placement" else "grasp_candidates"
    candidates = result.details.get(key)
    if not isinstance(candidates, list) or not candidates:
        return result
    expected_schema = (
        "openeta.compiled_placement_seed.v2"
        if purpose == "placement"
        else "openeta.compiled_grasp_seed.v1"
    )
    events: list[JsonDict] = []
    for queue_position, selected in enumerate(candidates):
        candidate_id = (
            str(selected.get("id") or "")
            if isinstance(selected, Mapping)
            else ""
        )
        if not candidate_id:
            return ToolResult(
                False,
                "qualified candidate queue contains an invalid entry",
                {
                    **result.details,
                    "reason": "host_candidate_compilation_failed",
                    "queue_position": queue_position,
                    "execution_started": False,
                },
            )
        parameters = (
            {"placement_candidate_id": candidate_id}
            if purpose == "placement"
            else {"purpose": "grasp", "grasp_candidate_id": candidate_id}
        )
        metadata = dict(context.metadata)
        metadata["_openeta_host_candidate_compilation_binding"] = {
            "purpose": purpose,
            "candidate_id": candidate_id,
            "scene_epoch": scene_epoch,
            "planning_scene_revision": planning_scene_revision,
            "selection_source": "host_qualified_queue",
        }
        compiled = compiler(
            ToolExecutionContext(
                name="host_candidate_compiler",
                spec=_HOST_CANDIDATE_COMPILER_SPEC,
                parameters=parameters,
                observation=context.observation,
                metadata=metadata,
            )
        )
        compiled_outputs = (
            compiled.details.get("outputs")
            if isinstance(compiled, ToolResult) and compiled.success
            else None
        )
        if (
            not isinstance(compiled_outputs, Mapping)
            or compiled_outputs.get("schema_version") != expected_schema
        ):
            failure_details = (
                compiled.details if isinstance(compiled, ToolResult) else {}
            )
            return ToolResult(
                False,
                "host failed to compile a qualified candidate",
                {
                    **result.details,
                    "reason": "host_candidate_compilation_failed",
                    "candidate_id": candidate_id,
                    "queue_position": queue_position,
                    "compilation_diagnostics": failure_details.get(
                        "diagnostics", []
                    ),
                    "execution_started": False,
                },
            )
        events.append(
            {
                "schema_version": "openeta.host_candidate_compilation.v1",
                "event_type": "candidate_compiled",
                "purpose": purpose,
                "candidate_id": candidate_id,
                "queue_position": queue_position,
                "queue_count": len(candidates),
                "selection_policy": "stable_qualified_queue_head",
                "scene_epoch": scene_epoch,
                "planning_scene_revision": planning_scene_revision,
                "execution_started": False,
                "compiled_seed": dict(compiled_outputs),
            }
        )
    candidate_id = str(events[0]["candidate_id"])
    result.details["selection_required"] = False
    result.details["host_selected_candidate_id"] = candidate_id
    result.details["host_candidate_compilation"] = dict(events[0])
    result.details["host_candidate_compilation_queue"] = events
    return result


def _active_source_grasp_id(
    memory: Mapping[str, object] | None,
    *,
    compiled_source_grasp: object,
) -> str:
    """Resolve the executed grasp identity from host-owned runtime state."""

    if isinstance(compiled_source_grasp, Mapping):
        for value in (
            compiled_source_grasp.get("source_grasp_id"),
            compiled_source_grasp.get("grasp_candidate_id"),
        ):
            if isinstance(value, str) and value:
                return value
        for pose_key in ("contact_pose", "hover_pose", "lift_pose"):
            pose = compiled_source_grasp.get(pose_key)
            if not isinstance(pose, Mapping):
                continue
            value = pose.get("source_grasp_id") or pose.get("grasp_candidate_id")
            if isinstance(value, str) and value:
                return value
    policy = memory.get("grasp_candidate_policy") if isinstance(memory, Mapping) else None
    if isinstance(policy, Mapping):
        active = policy.get("active_candidate")
        if isinstance(active, Mapping):
            value = active.get("id") or active.get("source_grasp_id")
            if isinstance(value, str) and value:
                return value
        for key in ("selected_grasp_id", "candidate_id"):
            value = policy.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _qualifying_wrist_alignment_handler(
    handler: ToolHandler,
    qualifier: MoveItCandidateQualifier,
) -> ToolHandler:
    """Re-qualify the final hover/contact chain after geometry refinement."""

    def qualified(context: ToolExecutionContext) -> ToolResult:
        result = handler(context)
        if not result.success:
            return result
        outputs = result.details.get("outputs")
        outputs = outputs if isinstance(outputs, dict) else result.details
        aligned = outputs.get("aligned_hover_pose")
        compiled = context.parameters.get("compiled_grasp")
        contact = compiled.get("contact_pose") if isinstance(compiled, dict) else None
        if not isinstance(aligned, dict) or not isinstance(contact, dict):
            return ToolResult(
                False,
                "wrist alignment lacks a complete final pose chain",
                {"reason": "final_pose_qualification_missing"},
            )
        scene_epoch = int(aligned.get("scene_epoch", 0))
        revision = scene_epoch
        source: JsonDict = {}
        if context.observation is not None:
            revision_value = context.observation.metadata.get("planning_scene_revision")
            if isinstance(revision_value, int) and not isinstance(revision_value, bool):
                revision = revision_value
            source["start_joint_state"] = context.observation.robot.to_dict()
        candidate = {
            "id": str(aligned.get("alignment_id") or "final-wrist-alignment"),
            "qualification_stages": [
                _qualification_pose("aligned_hover", aligned),
                _qualification_pose("contact", contact),
            ],
        }
        proof_result = qualifier.qualify_result(
            ToolResult(True, "final pose qualification", {
                "candidate_count": 1,
                "grasp_candidates": [candidate],
            }),
            purpose="grasp",
            scene_epoch=scene_epoch,
            planning_scene_revision=revision,
            source=source,
            cache_result=False,
        )
        if proof_result.details.get("candidate_count") != 1:
            return ToolResult(
                False,
                "wrist-aligned final pose failed MoveIt qualification",
                {
                    "reason": "final_pose_qualification_failed",
                    "qualification_evidence": proof_result.details.get(
                        "qualification_evidence"
                    ),
                },
            )
        outputs["qualification_evidence"] = proof_result.details[
            "qualification_evidence"
        ]
        outputs["final_pose_qualified"] = True
        return result

    return qualified


def _bind_skill_change_tools(
    runtime: OpenEtaAgentRuntime,
    *,
    backend_factory: BackendFactory,
    policy_provider: Callable[[], SupervisionPolicy],
    human_approval: SkillApproval | None,
) -> None:
    runtime.tools.bind_handler(
        "register_skill",
        _skill_change_handler(
            runtime,
            operation="register",
            backend_factory=backend_factory,
            policy_provider=policy_provider,
            human_approval=human_approval,
        ),
        replace=True,
    )
    runtime.tools.bind_handler(
        "update_skill",
        _skill_change_handler(
            runtime,
            operation="update",
            backend_factory=backend_factory,
            policy_provider=policy_provider,
            human_approval=human_approval,
        ),
        replace=True,
    )


def _skill_change_handler(
    runtime: OpenEtaAgentRuntime,
    *,
    operation: str,
    backend_factory: BackendFactory,
    policy_provider: Callable[[], SupervisionPolicy],
    human_approval: SkillApproval | None,
) -> Callable[[ToolExecutionContext], ToolResult]:
    def handler(context: ToolExecutionContext) -> ToolResult:
        try:
            requested_name = str(context.parameters.get("name") or "").strip()
            if not requested_name:
                raise ValueError("skill name is required")
            if not _has_skill_authoring_instruction(
                context.parameters,
                operation=operation,
            ):
                raise ValueError(f"{operation}_skill requires authoring instructions")
            current = None
            try:
                current = runtime.skills.get(requested_name)
            except KeyError:
                if operation == "update":
                    raise ValueError(f"Unknown skill: {requested_name}") from None
            else:
                if operation == "register":
                    raise ValueError(f"Skill already registered: {requested_name}")
                if not current.editable:
                    raise ValueError(f"Skill is not editable: {requested_name}")
            request = SkillAuthoringRequest(
                operation=operation,
                parameters=context.parameters,
                current_skill=current,
                executable_tools=_skill_authoring_tools(runtime.tools),
            )
            authored = BackendSkillAuthoringSubagent(
                backend_factory(max_tokens=SKILL_AUTHORING_MAX_OUTPUT_TOKENS)
            ).author(request)
            review = _authorize_skill_change(
                request=request,
                skill=authored.skill,
                policy=policy_provider(),
                backend_factory=backend_factory,
                human_approval=human_approval,
            )
            if not review["approved"]:
                raise PermissionError(str(review.get("reason") or "skill change denied"))
            if operation == "register":
                runtime.skills.register(authored.skill)
            else:
                runtime.skills.update(authored.skill)
        except Exception as exc:  # noqa: BLE001 - provider failures stay structured.
            return ToolResult(
                False,
                content=f"skill {operation} failed: {exc}",
                details={
                    "tool": context.name,
                    "reason": "skill_authoring_failed",
                    "error_type": type(exc).__name__,
                },
            )
        action = "registered" if operation == "register" else "updated"
        return ToolResult(
            True,
            content=f"skill {action}: {authored.skill.name}",
            details={
                "tool": context.name,
                "skill": authored.skill.name,
                "source": authored.skill.source,
                "authoring": authored.details,
                "provider": authored.provider,
                "model": authored.model,
                "review": review,
            },
        )

    return handler


def _authorize_skill_change(
    *,
    request: SkillAuthoringRequest,
    skill: SkillSpec,
    policy: SupervisionPolicy,
    backend_factory: BackendFactory,
    human_approval: SkillApproval | None,
) -> JsonDict:
    skill_name = str(getattr(skill, "name", "") or "")
    if policy.profile == SupervisionProfile.HUMAN_GATED:
        approved = bool(human_approval and human_approval(skill_name))
        return {
            "approved": approved,
            "source": "human",
            "reason": "Approved by human operator." if approved else "Human approval denied.",
        }
    if policy.profile in {SupervisionProfile.STANDARD, SupervisionProfile.SCRIPTED_TUI}:
        return {
            "approved": True,
            "source": "scripted_tui" if policy.profile == SupervisionProfile.SCRIPTED_TUI else "runtime_policy",
            "reason": "Scripted TUI permits session-local registry changes." if policy.profile == SupervisionProfile.SCRIPTED_TUI else "Standard profile permits session-local registry changes.",
        }
    reviewed = BackendSkillChangeReviewer(backend_factory()).review(
        request=request,
        skill=skill,
    )
    return {
        "approved": reviewed.approved,
        "source": "independent_reviewer",
        "decision": reviewed.decision,
        "reason": reviewed.reason,
        "details": reviewed.details,
    }


def _checker_config(
    tools: ToolRegistry,
    *,
    pre_safety_checks: dict[str, str],
) -> CheckerSubagentConfig:
    validated: dict[str, str] = {}
    for target, checker in pre_safety_checks.items():
        tools.get(target)
        checker_spec = tools.get(checker)
        if checker_spec.category != "safety":
            raise ValueError(f"Configured checker is not a safety tool: {checker}")
        if not tools.can_execute(checker):
            raise ValueError(f"Configured checker has no executable handler: {checker}")
        validated[target] = checker
    return CheckerSubagentConfig(
        pre_safety_checks=validated,
        post_failure_checks=tuple(spec.name for spec in tools.list()),
    )


def _skill_authoring_tools(tools: ToolRegistry) -> tuple[ToolSpec, ...]:
    return tuple(
        spec
        for spec in tools.list()
        if tools.can_execute(spec.name) and spec.category != "skill_management"
    )


def _has_skill_authoring_instruction(
    parameters: JsonDict,
    *,
    operation: str,
) -> bool:
    fields = (
        ("goal", "requirements", "description", "content")
        if operation == "register"
        else ("requested_changes", "requirements", "content")
    )
    return any(str(parameters.get(field) or "").strip() for field in fields)
