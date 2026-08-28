"""Shared runtime assembly for interactive and batch OpenETA entry points."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from tools.candidate_config import (
    DEFAULT_GRASPGENX_RAW_POOL_SIZE,
    DEFAULT_GRASP_WAVES,
)

from adapter.protocol import JsonDict
from agent.backends.planner import PlannerBackend
from agent.backends.provider_config import PlannerProviderConfig
from agent.runtime.calibration import (
    BackendCalibrationReviewer,
    CalibrationLifecycleConfig,
    CalibrationLifecycleManager,
)
from agent.runtime.checkers import CheckerSubagentConfig
from agent.runtime.memory import AgentMemory
from agent.runtime.moveit_qualification import (
    MoveItCandidateQualifier,
    QualificationCache,
    PRIVATE_RPC_NAME,
    SAME_RUN_QUALIFICATION_SEED_FIELD,
    private_qualification_rpc,
)
from agent.runtime.qualification_v3 import (
    candidate_physical_quality_key,
    grasp_symmetry_family_id,
    parallel_gripper_centering_quality,
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
from agent.runtime.sam3_selection import (
    BackendSam3SelectionReviewer,
    SAM3_SELECTION_REVIEW_MAX_ATTEMPTS,
    SAM3_SELECTION_REVIEW_MAX_OUTPUT_TOKENS,
    SAM3_SELECTION_REVIEW_TIMEOUT_S,
    Sam3SelectionParentContext,
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
from agent.tools.active_vision import build_active_observe_handler
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
    compile_placement_seed,
    compile_grasp_seed,
    materialize_world_object_goal,
    materialize_world_object_goal_from_current_pose,
    predicted_attachment_from_grasp,
    qualification_grasp_pose_chain,
    rebase_camera_grasp_candidate_for_object_motion,
)
from agent.tools.mcp_registry import load_mcp_server_url
from agent.tools.object_memory import (
    ObjectMemoryBankClient,
    ObjectMemoryBankConfigurationError,
    load_configured_object_memory_bank,
)
from agent.tools.registry import (
    ENVIRONMENT_AUTHORITY,
    ToolEventListener,
    ToolExecutionContext,
    ToolHandler,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    apply_tool_profile,
    build_default_tool_registry,
    resolve_tool_profile,
)
from agent.tools.sim_mcp import (
    SimulatorMcpResponseCallback,
    SimulatorMcpToolProxy,
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


GRASP_BACKEND_ENV_VAR = "OPENETA_GRASP_BACKEND"
GRASP_BACKEND_MODES = ("auto", "anygrasp", "graspgenx")
PERCEPTION_RPC_TIMEOUT_ENV_VAR = "OPENETA_PERCEPTION_RPC_TIMEOUT_S"
DEFAULT_PERCEPTION_RPC_TIMEOUT_S = 600.0


def runtime_grasp_backend_order_from_env() -> tuple[str, ...]:
    """Resolve the configured facade order; concrete modes never cross-fallback."""

    # GraspGenX is the conservative deployment default.  ``auto`` remains an
    # explicit operator choice, but must not silently probe a licensed
    # AnyGrasp installation before the selected backend is known healthy.
    mode = os.environ.get(GRASP_BACKEND_ENV_VAR, "graspgenx").strip().lower() or "graspgenx"
    if mode == "auto":
        return tuple(DEFAULT_GRASP_POSE_BACKEND_ORDER)
    if mode in GRASP_BACKEND_MODES:
        return (mode,)
    choices = ", ".join(GRASP_BACKEND_MODES)
    raise ValueError(f"{GRASP_BACKEND_ENV_VAR} must be one of: {choices}")


def runtime_perception_rpc_timeout_s_from_env() -> float:
    """Return the bounded deadline for one remote perception RPC attempt."""

    raw = os.environ.get(
        PERCEPTION_RPC_TIMEOUT_ENV_VAR,
        str(DEFAULT_PERCEPTION_RPC_TIMEOUT_S),
    )
    try:
        timeout_s = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{PERCEPTION_RPC_TIMEOUT_ENV_VAR} must be a finite positive number"
        ) from exc
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"{PERCEPTION_RPC_TIMEOUT_ENV_VAR} must be a finite positive number")
    return timeout_s


@dataclass(frozen=True, slots=True)
class RuntimeCandidateCounts:
    """Host registration values that must match remote service metadata."""

    graspgenx_raw_pool_size: int = DEFAULT_GRASPGENX_RAW_POOL_SIZE
    anygrasp_raw_pool_size: int = 200
    anyplace_raw_pool_size: int = 96
    grasp_diversity_pool_size: int = 64
    anyplace_diversity_pool_size: int = 96
    grasp_full_plan_limit: int = 2
    anyplace_full_plan_limit: int = 2
    frozen_pair_grasp_branch_limit: int = 4
    frozen_pair_full_plan_limit: int = 2
    moveit_ik_seed_count: int = 8
    qualification_profile: str = "legacy"
    solver_profile: str = "auto"
    fast_beam_width: int = 2
    grasp_waves: tuple[int, ...] | str = DEFAULT_GRASP_WAVES
    placement_waves: tuple[int, ...] | str = (4, 8, 16, 32, 96)
    max_ik_concurrency: int = 8
    max_state_validity_concurrency: int = 8
    fast_ik_seed_count: int = 2
    recovery_ik_seed_count: int = 6
    fast_ik_timeout_ms: int = 50
    recovery_ik_timeout_ms: int = 200
    capability_map_id: str = ""

    def __post_init__(self) -> None:
        validated = CandidateFunnelConfig(
            graspgenx_raw_pool_size=self.graspgenx_raw_pool_size,
            anygrasp_raw_pool_size=self.anygrasp_raw_pool_size,
            anyplace_raw_pool_size=self.anyplace_raw_pool_size,
            grasp_diversity_pool_size=self.grasp_diversity_pool_size,
            anyplace_diversity_pool_size=self.anyplace_diversity_pool_size,
            grasp_full_plan_limit=self.grasp_full_plan_limit,
            anyplace_full_plan_limit=self.anyplace_full_plan_limit,
            frozen_pair_grasp_branch_limit=self.frozen_pair_grasp_branch_limit,
            frozen_pair_full_plan_limit=self.frozen_pair_full_plan_limit,
            moveit_ik_seed_count=self.moveit_ik_seed_count,
            qualification_profile=self.qualification_profile,
            solver_profile=self.solver_profile,
            fast_beam_width=self.fast_beam_width,
            grasp_waves=self.grasp_waves,
            placement_waves=self.placement_waves,
            max_ik_concurrency=self.max_ik_concurrency,
            max_state_validity_concurrency=self.max_state_validity_concurrency,
            fast_ik_seed_count=self.fast_ik_seed_count,
            recovery_ik_seed_count=self.recovery_ik_seed_count,
            fast_ik_timeout_ms=self.fast_ik_timeout_ms,
            recovery_ik_timeout_ms=self.recovery_ik_timeout_ms,
            capability_map_id=self.capability_map_id,
        )
        for name in (
            "graspgenx_raw_pool_size",
            "anygrasp_raw_pool_size",
            "anyplace_raw_pool_size",
            "grasp_diversity_pool_size",
            "anyplace_diversity_pool_size",
            "grasp_full_plan_limit",
            "anyplace_full_plan_limit",
            "moveit_ik_seed_count",
            "frozen_pair_grasp_branch_limit",
            "frozen_pair_full_plan_limit",
            "qualification_profile",
            "solver_profile",
            "fast_beam_width",
            "grasp_waves",
            "placement_waves",
            "max_ik_concurrency",
            "max_state_validity_concurrency",
            "fast_ik_seed_count",
            "recovery_ik_seed_count",
            "fast_ik_timeout_ms",
            "recovery_ik_timeout_ms",
            "capability_map_id",
        ):
            object.__setattr__(self, name, getattr(validated, name))


def runtime_candidate_counts_from_env() -> RuntimeCandidateCounts:
    return RuntimeCandidateCounts(
        graspgenx_raw_pool_size=os.environ.get(
            "OPENETA_GRASPGENX_RAW_POOL_SIZE", DEFAULT_GRASPGENX_RAW_POOL_SIZE
        ),
        anygrasp_raw_pool_size=os.environ.get("OPENETA_ANYGRASP_RAW_POOL_SIZE", 200),
        anyplace_raw_pool_size=os.environ.get("OPENETA_ANYPLACE_RAW_POOL_SIZE", 96),
        grasp_diversity_pool_size=os.environ.get("OPENETA_GRASP_DIVERSITY_POOL_SIZE", 64),
        anyplace_diversity_pool_size=os.environ.get("OPENETA_ANYPLACE_DIVERSITY_POOL_SIZE", 96),
        grasp_full_plan_limit=os.environ.get("OPENETA_GRASP_FULL_PLAN_LIMIT", 2),
        anyplace_full_plan_limit=os.environ.get("OPENETA_ANYPLACE_FULL_PLAN_LIMIT", 2),
        frozen_pair_grasp_branch_limit=os.environ.get("OPENETA_FROZEN_PAIR_GRASP_BRANCH_LIMIT", 4),
        frozen_pair_full_plan_limit=os.environ.get("OPENETA_FROZEN_PAIR_FULL_PLAN_LIMIT", 2),
        moveit_ik_seed_count=os.environ.get("OPENETA_MOVEIT_IK_SEED_COUNT", 8),
        qualification_profile=os.environ.get("OPENETA_QUALIFICATION_PROFILE", "legacy"),
        solver_profile=os.environ.get("OPENETA_QUALIFICATION_SOLVER_PROFILE", "auto"),
        fast_beam_width=os.environ.get("OPENETA_QUALIFICATION_BEAM_WIDTH", 2),
        grasp_waves=os.environ.get(
            "OPENETA_QUALIFICATION_GRASP_WAVES",
            ",".join(str(value) for value in DEFAULT_GRASP_WAVES),
        ),
        placement_waves=os.environ.get("OPENETA_QUALIFICATION_PLACEMENT_WAVES", "4,8,16,32,96"),
        max_ik_concurrency=os.environ.get("OPENETA_QUALIFICATION_MAX_IK_CONCURRENCY", 8),
        max_state_validity_concurrency=os.environ.get(
            "OPENETA_QUALIFICATION_MAX_STATE_VALIDITY_CONCURRENCY", 8
        ),
        fast_ik_seed_count=os.environ.get("OPENETA_QUALIFICATION_FAST_SEEDS", 2),
        recovery_ik_seed_count=os.environ.get("OPENETA_QUALIFICATION_RECOVERY_SEEDS", 6),
        fast_ik_timeout_ms=os.environ.get("OPENETA_QUALIFICATION_FAST_IK_TIMEOUT_MS", 50),
        recovery_ik_timeout_ms=os.environ.get("OPENETA_QUALIFICATION_RECOVERY_IK_TIMEOUT_MS", 200),
        capability_map_id=os.environ.get("OPENETA_CAPABILITY_MAP_ID", ""),
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
    grasp_backend_order: tuple[str, ...] = field(
        default_factory=runtime_grasp_backend_order_from_env
    )
    simulator_transport: SimulatorMcpTransport | None = None
    simulator_proxy_config: SimulatorMcpToolProxyConfig | None = None
    web_access_config: WebAccessConfig | None = None
    mcp_response_callback: SimulatorMcpResponseCallback | None = None
    allow_outside_sandbox: bool = False
    approve_outside_sandbox: Callable[[ToolExecutionContext, str], bool] | None = None
    human_action_approval: ApprovalCallback | None = None
    calibration_approval: PublicationApproval | None = None
    skill_approval: SkillApproval | None = None
    supervision_policy_provider: Callable[[], SupervisionPolicy] | None = None
    pre_safety_checks: dict[str, str] = field(default_factory=dict)
    tool_listeners: tuple[ToolEventListener, ...] = ()
    max_validation_retries: int = 2
    tool_profile: str = field(default_factory=resolve_tool_profile)


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
        sam3_url=configured.sam3_url or loader("openeta-sam3", aliases=("sam3",)),
        depth_prior_url=configured.depth_prior_url
        or loader(
            "openeta-depth-prior",
            aliases=("depth-prior", "depth_prior", "unidepth"),
        ),
        anygrasp_url=configured.anygrasp_url or loader("openeta-anygrasp", aliases=("anygrasp",)),
        anyplace_url=configured.anyplace_url or loader("openeta-anyplace", aliases=("anyplace",)),
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
            qualification_cache=qualification_cache,
        )
        internal_candidate_compilers["placement"] = build_compile_placement_seed_handler(
            workspace.grasp_profile_path,
            qualification_cache=qualification_cache,
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

    policy_provider = config.supervision_policy_provider or (lambda: config.supervision_policy)
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

    bind_configured_web_tool_handlers(
        tools,
        config=config.web_access_config,
        provider_config=config.provider,
    )
    sam3_selection_parent_context = Sam3SelectionParentContext()
    sam3_selection_reviewer = _build_sam3_selection_reviewer(
        config.backend_factory,
        parent_context=sam3_selection_parent_context,
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
        grasp_backend_order=config.grasp_backend_order,
        internal_candidate_compilers=internal_candidate_compilers,
        selection_reviewer=sam3_selection_reviewer,
    )
    active_proxy = (
        SimulatorMcpToolProxy(
            transport=config.simulator_transport,
            config=simulator_proxy_config,
        )
        if config.simulator_transport is not None
        else None
    )
    tools.bind_handler(
        "active_observe",
        build_active_observe_handler(
            artifact_root=artifact_root,
            candidate_qualifier=qualifier,
            simulator_proxy=active_proxy,
            sam3_handler=tools.bound_handler("sam3"),
            move_spec=tools.get("move_to"),
            observe_spec=tools.get("observe"),
            sam3_spec=tools.get("sam3"),
        ),
        replace=True,
        authority=ENVIRONMENT_AUTHORITY,
    )
    apply_tool_profile(tools, config.tool_profile)

    planner = ToolCallingPlanner(
        config.backend_factory(),
        max_validation_retries=config.max_validation_retries,
        context_config=PlannerContextConfig(
            context_window_tokens=config.provider.context_window_tokens,
            token_estimator_model=config.provider.model,
        ),
        sam3_selection_reviewer=sam3_selection_reviewer,
        sam3_selection_parent_context=sam3_selection_parent_context,
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


def _build_sam3_selection_reviewer(
    backend_factory: BackendFactory,
    *,
    parent_context: Sam3SelectionParentContext | None = None,
) -> Callable[[JsonDict], JsonDict]:
    """Build one bounded reviewer shared by inline and fallback selection."""

    backend = backend_factory(
        max_tokens=SAM3_SELECTION_REVIEW_MAX_OUTPUT_TOKENS,
        max_vision_images=2,
        timeout_s=SAM3_SELECTION_REVIEW_TIMEOUT_S,
        max_attempts=1,
    )
    return BackendSam3SelectionReviewer(
        backend,
        max_attempts=SAM3_SELECTION_REVIEW_MAX_ATTEMPTS,
        parent_context=parent_context,
    ).review


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
    grasp_backend_order: tuple[str, ...] = DEFAULT_GRASP_POSE_BACKEND_ORDER,
    internal_candidate_compilers: Mapping[str, ToolHandler] | None = None,
    selection_reviewer: Callable[[JsonDict], JsonDict] | None = None,
    perception_rpc_timeout_s: float | None = None,
) -> DepthPriorPrefetchCoordinator | None:
    counts = candidate_counts or runtime_candidate_counts_from_env()
    rpc_timeout_s = (
        runtime_perception_rpc_timeout_s_from_env()
        if perception_rpc_timeout_s is None
        else float(perception_rpc_timeout_s)
    )
    if not math.isfinite(rpc_timeout_s) or rpc_timeout_s <= 0:
        raise ValueError("perception_rpc_timeout_s must be finite and positive")
    rpc_timeout_kwargs = (
        {}
        if rpc_timeout_s == DEFAULT_PERCEPTION_RPC_TIMEOUT_S
        else {"timeout_seconds": rpc_timeout_s}
    )
    frozen_pair_coordinator = (
        _FrozenGoalPairCoordinator(
            candidate_qualifier,
            grasp_branch_limit=counts.frozen_pair_grasp_branch_limit,
        )
        if candidate_qualifier is not None
        else None
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
            build_sse_depth_prior_mcp_estimator(
                url=endpoints.depth_prior_url,
                **rpc_timeout_kwargs,
            ),
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
    if endpoints.sam3_url:
        tools.bind_handler(
            "sam3",
            build_sam3_handler(
                build_sse_sam3_mcp_segmenter(
                    url=endpoints.sam3_url,
                    **rpc_timeout_kwargs,
                ),
                segment_points=build_sse_sam3_mcp_segmenter(
                    url=endpoints.sam3_url,
                    tool_name="segment_points",
                    **rpc_timeout_kwargs,
                ),
                selection_reviewer=selection_reviewer,
                depth_prior_prefetch=(
                    depth_prefetch.prefetch_for_sam3 if depth_prefetch is not None else None
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
                build_sse_molmopoint_mcp_pointer(
                    url=endpoints.molmopoint_url,
                    **rpc_timeout_kwargs,
                ),
                output_root=artifact_root / "molmopoint_results",
            ),
            replace=True,
        )
    if endpoints.anyplace_url:
        anyplace_handler = build_anyplace_handler(
            build_sse_anyplace_mcp_placer(
                url=endpoints.anyplace_url,
                **rpc_timeout_kwargs,
            ),
            output_root=artifact_root / "anyplace_results",
            expected_raw_pool_size=counts.anyplace_raw_pool_size,
            pre_inference=(
                lambda context, request: (
                    _prepare_postattachment_frozen_goals(
                        context,
                        request,
                        coordinator=frozen_pair_coordinator,
                    )
                    if frozen_pair_coordinator is not None
                    else None
                )
            ),
        )
        if candidate_qualifier is not None:
            anyplace_handler = _qualifying_handler(
                anyplace_handler,
                candidate_qualifier,
                purpose="placement",
                frozen_pair_coordinator=frozen_pair_coordinator,
                candidate_compiler=(internal_candidate_compilers or {}).get("placement"),
            )
        tools.bind_handler(
            "anyplace",
            anyplace_handler,
            replace=True,
        )

    grasp_backends = {}
    if endpoints.anygrasp_url:
        grasp_backends["anygrasp"] = build_anygrasp_handler(
            build_sse_anygrasp_mcp_grasper(
                url=endpoints.anygrasp_url,
                **rpc_timeout_kwargs,
            ),
            output_root=artifact_root / "anygrasp_results",
            expected_raw_pool_size=counts.anygrasp_raw_pool_size,
        )
    if endpoints.graspgenx_url:
        list_grippers = build_sse_graspgenx_mcp_gripper_lister(
            url=endpoints.graspgenx_url,
            **rpc_timeout_kwargs,
        )
        grasp_backends["graspgenx"] = build_graspgenx_handler(
            build_sse_graspgenx_mcp_predictor(
                url=endpoints.graspgenx_url,
                **rpc_timeout_kwargs,
            ),
            list_grippers,
            output_root=artifact_root / "graspgenx_results",
            expected_raw_pool_size=counts.graspgenx_raw_pool_size,
        )
    # Resolve the configured Contact-GraspNet endpoint for startup/discovery,
    # but keep the backend disabled until its planner-facing contract is
    # explicitly re-enabled for the target deployment.
    if endpoints.contact_graspnet_url:
        build_sse_contact_graspnet_mcp_predictor(
            url=endpoints.contact_graspnet_url,
            **rpc_timeout_kwargs,
        )
    # Contact-GraspNet is temporarily disabled for the simulator drawer track.
    # Keep its endpoint/configuration and implementation available for a later
    # re-enable, but do not expose it as an executable grasp backend here.
    if grasp_backends:
        grasp_handler = build_grasp_pose_estimate_handler(
            grasp_backends,
            backend_order=grasp_backend_order,
            graspgenx_gripper_name="robotiq_2f_85",
        )
        if candidate_qualifier is not None:
            grasp_handler = _qualifying_handler(
                grasp_handler,
                candidate_qualifier,
                purpose="grasp",
                frozen_pair_coordinator=frozen_pair_coordinator,
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
    names = {str(item.get("name") or "") for item in tools_value or [] if isinstance(item, dict)}
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
        frozen_pair_full_plan_limit=counts.frozen_pair_full_plan_limit,
        ik_seed_count=counts.moveit_ik_seed_count,
        qualification_profile=counts.qualification_profile,
        solver_profile=counts.solver_profile,
        beam_width=counts.fast_beam_width,
        grasp_waves=counts.grasp_waves,
        placement_waves=counts.placement_waves,
        max_ik_concurrency=counts.max_ik_concurrency,
        max_state_validity_concurrency=counts.max_state_validity_concurrency,
        fast_seed_count=counts.fast_ik_seed_count,
        recovery_seed_count=counts.recovery_ik_seed_count,
        fast_ik_timeout_s=counts.fast_ik_timeout_ms / 1000.0,
        recovery_ik_timeout_s=counts.recovery_ik_timeout_ms / 1000.0,
        capability_map_id=counts.capability_map_id,
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
    ) -> JsonDict:
        if purpose == "placement":
            predicted_attachment = candidate.get("predicted_attachment_transform")
            physical_rebase = candidate.get("frozen_object_motion_rebase")
            if isinstance(predicted_attachment, Mapping) and not isinstance(
                physical_rebase, Mapping
            ):
                # Before the gripper has closed, the attachment is derived
                # from the model contact and the measured point-cloud object
                # frame.  A frozen goal may also carry a scene-bound physical
                # collision-body goal from an earlier legality pass.  Mixing
                # that physical goal with the point-cloud attachment shifts
                # the release terminal by the centroid/body-frame offset.
                # Compile in the model frame here; the legality binder below
                # applies the one scene-derived support correction to both the
                # object motion and the release EEF terminal before IK/L5.
                world_object_goal = (
                    candidate.get("model_pointcloud_object_goal_pose")
                    or candidate.get("world_object_goal_pose")
                    or candidate.get("object_goal_pose")
                )
                compiled_candidate = dict(candidate)
                if isinstance(world_object_goal, Mapping):
                    compiled_goal = dict(world_object_goal)
                    compiled_candidate["world_object_goal_pose"] = compiled_goal
                    compiled_candidate["object_goal_pose"] = dict(compiled_goal)
                    compiled_candidate["qualification_object_goal_source"] = (
                        "model_pointcloud_goal_with_predicted_attachment"
                    )
                attachment_transform: object = dict(predicted_attachment)
            else:
                world_object_goal = candidate.get("world_object_goal_pose") or candidate.get(
                    "object_goal_pose"
                )
                compiled_candidate = dict(candidate)
                if isinstance(world_object_goal, Mapping):
                    compiled_goal = dict(world_object_goal)
                    compiled_candidate["world_object_goal_pose"] = compiled_goal
                    compiled_candidate["object_goal_pose"] = dict(compiled_goal)
                    compiled_candidate["qualification_object_goal_source"] = (
                        "physical_goal_with_measured_attachment"
                    )
                attachment_transform = (
                    dict(predicted_attachment)
                    if isinstance(predicted_attachment, Mapping)
                    else source.get("attachment_transform")
                )
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
                    json.dumps(attachment_transform, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
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
            compiled_pose_chain = [dict(compiled["release_pose"])]
            stages = [
                {
                    **_qualification_pose("release", compiled["release_pose"]),
                    "scene_transition": "virtual_detach",
                    # Releasing changes both the planning scene and the hand
                    # geometry.  Validate the exact planned arm endpoint with
                    # the gripper fully open after the virtual detach so that
                    # a goal next to a bin wall cannot pass qualification and
                    # then fail during the irreversible physical release.
                    "qualification_post_transition_gripper_state": "open",
                },
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
                # Freeze the embodiment aperture with the same calibration
                # snapshot as the model-to-EEF transform.  The private cheap
                # legality layer uses it only to rank unchanged model poses.
                "max_gripper_width_m": float(profile["max_gripper_width_m"]),
            }
            compiled = compile_grasp_seed(
                parameters,
                profile=profile,
                profile_sha256=profile_sha256,
            )
            compiled_pose_chain = qualification_grasp_pose_chain(compiled)
            stages = [_qualification_pose("contact", compiled_pose_chain[0])]
            stages[0]["scene_transition"] = "virtual_attach"
            # A contact pose is executable only if the complete Robotiq close
            # sweep clears the static workcell at the exact arm endpoint.
            # Target/touch-link contact remains request-locally allowed; table,
            # fixture, camera, and arm collisions still reject the candidate.
            stages[0]["qualification_terminal_gripper_state"] = "closing_sweep"
        return {
            "qualification_stages": stages,
            "compile_parameters": {
                **parameters,
                "qualification_profile_sha256": profile_sha256,
                "qualified_compiled_pose_sha256": hashlib.sha256(
                    json.dumps(
                        compiled_pose_chain,
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
        """Freeze calibration once for one qualification batch."""

        if purpose not in {"grasp", "placement"}:
            raise ValueError("candidate qualification purpose is invalid")
        profile_bytes = profile_path.read_bytes()
        profile = json.loads(profile_bytes.decode("utf-8"))
        profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()

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
            quat = [
                0.25 * scale,
                (m[0][1] + m[1][0]) / scale,
                (m[0][2] + m[2][0]) / scale,
                (m[2][1] - m[1][2]) / scale,
            ]
        elif index == 1:
            scale = (1.0 + m[1][1] - m[0][0] - m[2][2]) ** 0.5 * 2.0
            quat = [
                (m[0][1] + m[1][0]) / scale,
                0.25 * scale,
                (m[1][2] + m[2][1]) / scale,
                (m[0][2] - m[2][0]) / scale,
            ]
        else:
            scale = (1.0 + m[2][2] - m[0][0] - m[1][1]) ** 0.5 * 2.0
            quat = [
                (m[0][2] + m[2][0]) / scale,
                (m[1][2] + m[2][1]) / scale,
                0.25 * scale,
                (m[1][0] - m[0][1]) / scale,
            ]
    result["quat_xyzw"] = quat
    return result


def _qualification_infrastructure_reason(result: ToolResult) -> str:
    """Return a private qualification infrastructure failure, if present."""

    details = result.details
    stop_reason = str(details.get("qualification_stop_reason") or "")
    evidence = details.get("qualification_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    if stop_reason == "infrastructure_error" or evidence.get("infrastructure_error") is True:
        counts = details.get("rejection_reason_counts")
        if isinstance(counts, Mapping) and counts.get("qualification_rpc_error"):
            return "qualification_rpc_error"
        return str(evidence.get("stop_reason") or stop_reason or "infrastructure_error")
    return ""


def _qualification_infrastructure_failure(result: ToolResult) -> ToolResult:
    """Fail the tool without relabelling an infrastructure fault as unreachable."""

    reason = _qualification_infrastructure_reason(result) or "infrastructure_error"
    details = json.loads(json.dumps(result.details))
    details.update(
        {
            "reason": "qualification_infrastructure_error",
            "infrastructure_error": True,
            "qualification_infrastructure_reason": reason,
            "execution_started": False,
        }
    )
    return ToolResult(
        False,
        f"MoveIt qualification infrastructure failed: {reason}",
        details,
    )


def _restore_frozen_model_motion_for_predicted_pair(pair: JsonDict) -> None:
    """Rebind one cached physical goal to a new predicted grasp attachment.

    The goal-legality cache stores the PlanningScene collision-body pose and
    retires the active point-cloud motion so post-attachment execution cannot
    accidentally apply it twice.  A *new* grasp branch is different: its
    predicted ``T_eef_object`` is expressed in the original point-cloud object
    frame, so pair qualification must temporarily replay the original rigid
    world motion.  The legality binder then applies that same motion to the
    physical collision body and attaches the actual PlanningScene object.

    A frontier rebased after real object motion already uses the physical
    frame end-to-end and must never restore this stale model transform.
    """

    if isinstance(pair.get("frozen_object_motion_rebase"), Mapping):
        return
    binding = pair.get("frozen_goal_frame_binding")
    model_motion = pair.get("model_object_motion_world_transform")
    if not (
        isinstance(binding, Mapping)
        and binding.get("physical_collision_goal") is True
        and isinstance(model_motion, Mapping)
        and isinstance(pair.get("predicted_attachment_transform"), Mapping)
        and isinstance(pair.get("frozen_contact_pose"), Mapping)
    ):
        return
    pair["object_motion_world_transform"] = json.loads(json.dumps(model_motion))
    pair["physical_scene_attachment_required"] = True
    pair["physical_scene_attachment_source"] = "cached_collision_goal_with_replayed_model_motion"


@dataclass(slots=True)
class _FrozenGoalPairCoordinator:
    """Host-private bounded look-ahead used only to rank executable grasps."""

    qualifier: MoveItCandidateQualifier
    grasp_branch_limit: int = 4
    object_goals: list[JsonDict] = field(default_factory=list)
    object_current_pose: JsonDict | None = None
    scene_epoch: int = -1
    planning_scene_revision: int = -1
    qualified_goals_by_grasp: dict[str, list[JsonDict]] = field(default_factory=dict)
    consumed_attachment_bindings: set[str] = field(default_factory=set)
    attachment_exposed_goal_ids: dict[str, set[str]] = field(default_factory=dict)
    attachment_prepared_exclusions: dict[str, frozenset[str]] = field(default_factory=dict)
    attachment_frontier_generations: dict[str, int] = field(default_factory=dict)
    active_attachment_binding: str = ""
    source_model_raw_candidate_count: int = 0
    source_candidate_image_ref: str = ""
    source_candidate_artifacts: list[JsonDict] = field(default_factory=list)
    source_binding: JsonDict = field(default_factory=dict)
    grasp_frontier_candidates: list[JsonDict] = field(default_factory=list)
    grasp_candidate_catalog: dict[str, JsonDict] = field(default_factory=dict)
    grasp_frontier_template: JsonDict = field(default_factory=dict)
    grasp_frontier_scene_epoch: int = -1
    grasp_frontier_planning_scene_revision: int = -1
    grasp_frontier_generation: int = 0
    physically_rejected_grasp_ids: set[str] = field(default_factory=set)

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
                "frozen placement goal binding evidence is incomplete",
                {"reason": "frozen_goal_binding_missing", "execution_started": False},
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
            for goal in goals:
                object_goal = goal.get("object_goal_pose")
                if isinstance(object_goal, Mapping):
                    goal["world_object_goal_pose"] = dict(object_goal)
        except Exception as exc:  # noqa: BLE001 - fail closed at frame boundary.
            return ToolResult(
                False,
                f"frozen placement goal binding failed: {exc}",
                {"reason": "frozen_goal_binding_invalid", "execution_started": False},
            )
        self.object_goals = goals
        self.object_current_pose = dict(current_pose)
        self.scene_epoch = scene_epoch
        self.planning_scene_revision = planning_scene_revision
        self.qualified_goals_by_grasp.clear()
        self.consumed_attachment_bindings.clear()
        self.attachment_exposed_goal_ids.clear()
        self.attachment_prepared_exclusions.clear()
        self.attachment_frontier_generations.clear()
        self.active_attachment_binding = ""
        self.grasp_frontier_candidates.clear()
        self.grasp_candidate_catalog.clear()
        self.grasp_frontier_template.clear()
        self.grasp_frontier_scene_epoch = -1
        self.grasp_frontier_planning_scene_revision = -1
        self.grasp_frontier_generation = 0
        self.physically_rejected_grasp_ids.clear()
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
        self.source_candidate_artifacts = (
            [
                json.loads(json.dumps(artifact))
                for artifact in artifacts
                if isinstance(artifact, Mapping)
            ]
            if isinstance(artifacts, list)
            else []
        )
        self.source_binding = {
            key: json.loads(json.dumps(source[key]))
            for key in (
                "object_observation",
                "placement_observation",
                "object_camera_to_placement_camera",
                "placement_camera_to_world",
                "placement_camera_extrinsics",
            )
            if key in source
        }
        # Raw goals remain host-private.  They are not executable placement
        # candidates and are never inserted into the placement qualification cache.
        result.details["placement_candidates"] = []
        result.details["candidate_count"] = 0
        result.details["selection_required"] = False
        result.details["frozen_goal_pool_ready"] = True
        result.details["frozen_goal_pool_count"] = len(goals)
        result.details["execution_started"] = False
        result.content = (
            f"Retained {len(goals)} host-private object goals for bounded "
            "model-goal grasp-place qualification."
        )
        return result

    def update_grasp_frontier(
        self,
        provider_result: ToolResult,
        qualified_result: ToolResult,
        *,
        scene_epoch: int,
        planning_scene_revision: int,
    ) -> None:
        """Retain only the unvisited tail of one frozen provider result."""

        raw = provider_result.details.get("grasp_candidates")
        raw = raw if isinstance(raw, list) else []
        raw_by_id = {
            str(candidate.get("id") or ""): json.loads(json.dumps(candidate))
            for candidate in raw
            if isinstance(candidate, Mapping) and str(candidate.get("id") or "")
        }
        self.grasp_candidate_catalog.update(raw_by_id)
        frontier_ids: list[str] = []
        artifact_rows_authoritative = False
        artifact = qualified_result.details.get("qualification_artifact")
        artifact_path = artifact.get("path") if isinstance(artifact, Mapping) else None
        if isinstance(artifact_path, str):
            try:
                payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            rows = payload.get("results") if isinstance(payload, Mapping) else None
            if isinstance(rows, list):
                artifact_rows_authoritative = True
                frontier_ids = [
                    str(row.get("candidate_id") or "")
                    for row in rows
                    if isinstance(row, Mapping)
                    and row.get("verdict") == "NOT_EVALUATED"
                    and str(row.get("candidate_id") or "") in raw_by_id
                ]
        if (
            not artifact_rows_authoritative
            and qualified_result.details.get("qualification_profile") == "fast_v3"
            and str(qualified_result.details.get("qualification_stop_reason") or "").startswith(
                "complete_l5_pass_found"
            )
        ):
            selected_ids = {
                str(candidate.get("id") or "")
                for candidate in qualified_result.details.get("grasp_candidates", [])
                if isinstance(candidate, Mapping)
            }
            # Artifact-free test/integration adapters cannot distinguish a
            # hard rejection from an untouched tail.  Preserve recall in that
            # degraded evidence mode; the next deterministic qualification
            # call will re-prove every retained entry before exposure.
            frontier_ids = [
                candidate_id for candidate_id in raw_by_id if candidate_id not in selected_ids
            ]
        self.grasp_frontier_candidates = [
            raw_by_id[candidate_id] for candidate_id in frontier_ids if candidate_id in raw_by_id
        ]
        self.grasp_frontier_template = {
            key: json.loads(json.dumps(value))
            for key, value in provider_result.details.items()
            if key != "grasp_candidates"
        }
        self.grasp_frontier_scene_epoch = scene_epoch
        self.grasp_frontier_planning_scene_revision = planning_scene_revision
        self.grasp_frontier_generation += 1
        qualified_result.details.update(
            {
                "frozen_grasp_frontier_remaining_count": len(self.grasp_frontier_candidates),
                "frozen_grasp_frontier_generation": self.grasp_frontier_generation,
                "frozen_grasp_frontier_model_inference_invoked": False,
            }
        )

    def prioritize_grasp_frontier_for_parent(self, candidate_id: str) -> int:
        """Move unchanged model-native siblings of one grasp to the front.

        GraspGenX centering-reserve candidates may name either the exact
        backend parent or a set of compatible parents for the same approach
        family.  Sharing any compatible parent is therefore equally strong
        model evidence that two unchanged poses belong to one useful local
        neighbourhood; no geometric pose is synthesized or modified here.
        """

        parent = self.grasp_candidate_catalog.get(str(candidate_id))
        parent_backend_index = parent.get("backend_index") if isinstance(parent, Mapping) else None
        parent_alignment = (
            parent.get("target_closing_alignment") if isinstance(parent, Mapping) else None
        )
        parent_compatible = (
            parent_alignment.get("compatible_parent_backend_indices")
            if isinstance(parent_alignment, Mapping)
            else None
        )
        parent_family = (
            {
                int(value)
                for value in parent_compatible
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            }
            if isinstance(parent_compatible, list)
            else set()
        )
        if not parent_family and (
            isinstance(parent_backend_index, bool)
            or not isinstance(parent_backend_index, int)
            or parent_backend_index < 0
        ):
            return 0
        preferred: list[JsonDict] = []
        remaining: list[JsonDict] = []
        for raw_candidate in self.grasp_frontier_candidates:
            candidate = json.loads(json.dumps(raw_candidate))
            alignment = candidate.get("target_closing_alignment")
            parents = (
                alignment.get("compatible_parent_backend_indices")
                if isinstance(alignment, Mapping)
                else None
            )
            candidate_family = (
                {
                    int(value)
                    for value in parents
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0
                }
                if isinstance(parents, list)
                else set()
            )
            direct_parent = (
                isinstance(parent_backend_index, int)
                and not isinstance(parent_backend_index, bool)
                and parent_backend_index in candidate_family
            )
            shared_parent = bool(parent_family.intersection(candidate_family))
            if direct_parent or shared_parent:
                candidate["frozen_frontier_parent_priority"] = True
                candidate["frozen_frontier_parent_candidate_id"] = str(candidate_id)
                candidate["frozen_frontier_parent_priority_basis"] = (
                    "direct_backend_parent" if direct_parent else "shared_model_centering_parent"
                )
                preferred.append(candidate)
            else:
                candidate.pop("frozen_frontier_parent_priority", None)
                candidate.pop("frozen_frontier_parent_candidate_id", None)
                candidate.pop("frozen_frontier_parent_priority_basis", None)
                remaining.append(candidate)
        self.grasp_frontier_candidates = [*preferred, *remaining]
        return len(preferred)

    def prepare_grasp_frontier_expansion(
        self,
        *,
        scene_epoch: int,
        planning_scene_revision: int,
    ) -> ToolResult:
        """Materialize the frozen unvisited tail without invoking a model."""

        if not self.grasp_frontier_candidates:
            return ToolResult(
                False,
                "The frozen grasp frontier is exhausted.",
                {
                    "reason": "frozen_grasp_frontier_exhausted",
                    "model_inference_invoked": False,
                    "execution_started": False,
                },
            )
        if planning_scene_revision != self.grasp_frontier_planning_scene_revision:
            return ToolResult(
                False,
                "The frozen grasp frontier no longer matches the PlanningScene revision.",
                {
                    "reason": "frozen_grasp_frontier_scene_revision_changed",
                    "source_planning_scene_revision": (self.grasp_frontier_planning_scene_revision),
                    "planning_scene_revision": planning_scene_revision,
                    "model_inference_invoked": False,
                    "execution_started": False,
                },
            )
        # A failed plan with execution_started=false may advance the runtime's
        # bookkeeping epoch while leaving the physical PlanningScene revision
        # unchanged.  The frontier obligation is emitted only for that safe
        # case; rebind the frozen goal packet to the current bookkeeping epoch
        # before the new grasp/pair proof is compiled.
        self.scene_epoch = scene_epoch
        details = json.loads(json.dumps(self.grasp_frontier_template))
        details.update(
            {
                "grasp_candidates": json.loads(json.dumps(self.grasp_frontier_candidates)),
                "candidate_count": len(self.grasp_frontier_candidates),
                "generated_candidate_count": len(self.grasp_frontier_candidates),
                "scene_epoch": scene_epoch,
                "scene_revision": planning_scene_revision,
                "frozen_grasp_frontier_expansion": True,
                "frozen_grasp_frontier_generation": self.grasp_frontier_generation,
                "model_inference_invoked": False,
                "execution_started": False,
            }
        )
        return ToolResult(
            True,
            (
                f"Prepared {len(self.grasp_frontier_candidates)} frozen grasp "
                "candidates for the next qualification wave without model inference."
            ),
            details,
        )

    def rebase_grasp_frontier_from_target_pose_sync(
        self,
        receipt: Mapping[str, object],
        *,
        scene_epoch: int,
        planning_scene_revision: int,
        failed_candidate_id: str = "",
    ) -> ToolResult:
        """Rebind all unconsumed model grasps after proven target motion.

        A physical close can move a detached object.  That invalidates both
        the untouched provider tail and already-qualified backups because all
        of their IK/L5 proofs were bound to the prior pose.  Rebuild the
        frontier from the frozen provider catalog, exclude every physically
        attempted candidate, apply the measured rigid transform, and let the
        ordinary qualifier prove the candidates again under the new scene.
        """

        sync = receipt.get("planning_scene_target_pose_sync")
        sync = sync if isinstance(sync, Mapping) else {}
        detachable = receipt.get("detachable_joint")
        detachable = detachable if isinstance(detachable, Mapping) else {}
        world_before = sync.get("world_ids_before")
        world_after = sync.get("world_ids_after")
        attached_before = sync.get("attached_ids_before")
        attached_after = sync.get("attached_ids_after")
        source_revision = sync.get("source_revision")
        target_revision = sync.get("revision")
        source_pose = sync.get("source_target_pose")
        target_pose = sync.get("target_pose")
        source = self.grasp_frontier_template.get("source")
        source = source if isinstance(source, Mapping) else {}
        camera_extrinsics = source.get("camera_extrinsics")
        safe_binding = (
            sync.get("schema_version") == "openeta.planning_scene_target_pose_sync.v1"
            and sync.get("operation") == "update_world_target"
            and sync.get("topology_unchanged") is True
            and sync.get("static_world_unchanged") is True
            and isinstance(world_before, list)
            and world_before == world_after
            and attached_before == attached_after == []
            and detachable.get("state") == "detached"
            and isinstance(source_revision, int)
            and not isinstance(source_revision, bool)
            and source_revision == self.grasp_frontier_planning_scene_revision
            and isinstance(target_revision, int)
            and not isinstance(target_revision, bool)
            and target_revision == planning_scene_revision
            and isinstance(source_pose, Mapping)
            and isinstance(target_pose, Mapping)
            and isinstance(camera_extrinsics, Mapping)
        )
        if not safe_binding:
            return ToolResult(
                False,
                "Frozen grasp rebind lacks an unchanged static-scene proof.",
                {
                    "reason": "frozen_grasp_frontier_rebase_proof_missing",
                    "model_inference_invoked": False,
                    "execution_started": False,
                },
            )
        rejected_id = str(failed_candidate_id or "").strip()
        if rejected_id:
            self.physically_rejected_grasp_ids.add(rejected_id)
        source_candidates = [
            json.loads(json.dumps(candidate))
            for candidate_id, candidate in self.grasp_candidate_catalog.items()
            if candidate_id not in self.physically_rejected_grasp_ids
        ]
        if not source_candidates:
            return ToolResult(
                False,
                "The frozen grasp frontier is exhausted after physical failures.",
                {
                    "reason": "frozen_grasp_frontier_exhausted",
                    "physically_rejected_candidate_ids": sorted(
                        self.physically_rejected_grasp_ids
                    ),
                    "model_inference_invoked": False,
                    "execution_started": False,
                },
            )
        try:
            rebased = [
                rebase_camera_grasp_candidate_for_object_motion(
                    candidate,
                    camera_extrinsics=camera_extrinsics,
                    source_object_pose=source_pose,
                    target_object_pose=target_pose,
                )
                for candidate in source_candidates
            ]
        except (TypeError, ValueError) as exc:
            return ToolResult(
                False,
                f"Frozen grasp rebind failed: {exc}",
                {
                    "reason": "frozen_grasp_frontier_rebase_invalid",
                    "model_inference_invoked": False,
                    "execution_started": False,
                },
            )

        rebase_evidence = {
            "schema_version": "openeta.frozen_object_motion_rebase.v1",
            "source_planning_scene_revision": source_revision,
            "planning_scene_revision": planning_scene_revision,
            "translation_delta_m": sync.get("translation_delta_m"),
            "rotation_delta_rad": sync.get("rotation_delta_rad"),
            "candidate_count": len(rebased),
            "physically_rejected_candidate_ids": sorted(
                self.physically_rejected_grasp_ids
            ),
            "model_inference_invoked": False,
            "static_world_sha256": sync.get("static_world_sha256_after"),
        }
        for candidate in rebased:
            candidate["frozen_object_motion_rebase"] = json.loads(json.dumps(rebase_evidence))
        self.grasp_frontier_candidates = rebased
        self.grasp_candidate_catalog.update(
            {
                str(candidate.get("id") or ""): json.loads(
                    json.dumps(candidate)
                )
                for candidate in rebased
                if str(candidate.get("id") or "")
            }
        )
        self.grasp_frontier_scene_epoch = scene_epoch
        self.grasp_frontier_planning_scene_revision = planning_scene_revision
        self.scene_epoch = scene_epoch
        self.planning_scene_revision = planning_scene_revision
        self.object_current_pose = json.loads(json.dumps(target_pose))
        self.qualified_goals_by_grasp.clear()
        self.grasp_frontier_template["source"] = json.loads(json.dumps(source))
        self.grasp_frontier_template["frozen_object_motion_rebase"] = json.loads(
            json.dumps(rebase_evidence)
        )
        return ToolResult(
            True,
            "Rebased the frozen grasp frontier from measured rigid object motion.",
            {
                "frozen_grasp_frontier_rebased": True,
                "candidate_count": len(rebased),
                "source_planning_scene_revision": source_revision,
                "planning_scene_revision": planning_scene_revision,
                "model_inference_invoked": False,
                "execution_started": False,
            },
        )

    @staticmethod
    def _qualification_result_rows(result: ToolResult) -> list[JsonDict]:
        """Load host-private rows needed to cache goal-legality bindings."""

        artifact = result.details.get("qualification_artifact")
        artifact_path = artifact.get("path") if isinstance(artifact, Mapping) else None
        if isinstance(artifact_path, str):
            try:
                path = Path(artifact_path)
                if path.is_file() and path.stat().st_size <= 64 * 1024 * 1024:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    rows = payload.get("results") if isinstance(payload, Mapping) else None
                    if isinstance(rows, list):
                        return [dict(row) for row in rows if isinstance(row, Mapping)]
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        evidence = result.details.get("qualification_evidence")
        rows = evidence.get("results") if isinstance(evidence, Mapping) else None
        return (
            [dict(row) for row in rows if isinstance(row, Mapping)]
            if isinstance(rows, list)
            else []
        )

    @staticmethod
    def _bind_physical_collision_goal(
        goal: Mapping[str, object],
        *,
        goal_id: str,
        collision_goal: object,
    ) -> JsonDict:
        frozen_goal = json.loads(json.dumps(goal))
        if not isinstance(collision_goal, Mapping):
            return frozen_goal
        model_goal = (
            frozen_goal.get("model_pointcloud_object_goal_pose")
            or frozen_goal.get("world_object_goal_pose")
            or frozen_goal.get("object_goal_pose")
        )
        if isinstance(model_goal, Mapping):
            frozen_goal["model_pointcloud_object_goal_pose"] = dict(model_goal)
        model_motion = frozen_goal.pop("object_motion_world_transform", None)
        if isinstance(model_motion, Mapping) and not isinstance(
            frozen_goal.get("model_object_motion_world_transform"), Mapping
        ):
            frozen_goal["model_object_motion_world_transform"] = dict(model_motion)
        frozen_goal["world_object_goal_pose"] = dict(collision_goal)
        frozen_goal["object_goal_pose"] = dict(collision_goal)
        frozen_goal["frozen_goal_frame_binding"] = {
            "schema_version": "openeta.frozen_physical_object_goal.v1",
            "source": "frozen_goal_legality",
            "source_object_goal_id": goal_id,
            "physical_collision_goal": True,
        }
        return frozen_goal

    def _cache_goal_legality_frontier(
        self,
        joint: ToolResult,
        *,
        pairs: Sequence[Mapping[str, object]],
    ) -> JsonDict:
        """Cache one physical object goal per legal AnyPlace target."""

        pair_goal_ids = {
            str(pair.get("id") or ""): str(pair.get("source_object_goal_id") or "")
            for pair in pairs
            if str(pair.get("id") or "") and str(pair.get("source_object_goal_id") or "")
        }
        legality_by_goal: dict[str, Mapping[str, object]] = {}
        for row in self._qualification_result_rows(joint):
            legality = row.get("goal_legality")
            if not isinstance(legality, Mapping):
                continue
            goal_id = str(legality.get("goal_id") or "") or pair_goal_ids.get(
                str(row.get("candidate_id") or ""), ""
            )
            if goal_id:
                legality_by_goal.setdefault(goal_id, legality)
        expected_ids = {
            str(goal.get("id") or "") for goal in self.object_goals if str(goal.get("id") or "")
        }
        complete = bool(expected_ids) and expected_ids.issubset(legality_by_goal)
        screened: list[JsonDict] = []
        pass_count = 0
        reject_count = 0
        for goal in self.object_goals:
            goal_id = str(goal.get("id") or "")
            legality = legality_by_goal.get(goal_id)
            verdict = str(legality.get("verdict") or "") if legality else ""
            if verdict == "PASS":
                pass_count += 1
            elif verdict == "FAIL":
                reject_count += 1
            if complete and verdict != "PASS":
                continue
            checks = legality.get("checks") if isinstance(legality, Mapping) else None
            binding = checks.get("object_frame_binding") if isinstance(checks, Mapping) else None
            collision_goal = (
                binding.get("collision_goal_pose") if isinstance(binding, Mapping) else None
            )
            screened.append(
                self._bind_physical_collision_goal(
                    goal,
                    goal_id=goal_id,
                    collision_goal=collision_goal,
                )
            )
        # Only a complete first-layer result may permanently remove hard
        # rejects. In artifact-free/degraded adapters, preserve the raw pool
        # and let the measured-attachment qualifier prove it again.
        self.object_goals = screened
        return {
            "frozen_goal_legality_screen_complete": complete,
            "frozen_goal_legality_evidence_count": len(legality_by_goal),
            "frozen_goal_legality_pass_count": pass_count,
            "frozen_goal_legality_reject_count": reject_count,
            "frozen_goal_legality_frontier_count": len(screened),
        }

    def filter_grasps(
        self,
        result: ToolResult,
        *,
        scene_epoch: int,
        planning_scene_revision: int,
        source: Mapping[str, object],
    ) -> ToolResult:
        """Find a complete primary and distinct-grasp backup when available.

        Grasp inference is immutable by this point.  ``fast_v3`` may stop its
        first grasp wave after two diverse L5 passes, yet attachment-aware
        grasp/place qualification can still reject either branch.  Retain one
        complete primary plus one distinct-grasp backup before execution when
        the frozen pool can prove both.  If the complete pool cannot supply a
        backup, preserve the proven primary as an explicit redundancy-degraded
        fallback.  Model inference is never repeated while this search runs.
        """

        matching_goal_pool = (
            isinstance(self.object_current_pose, Mapping)
            and self.scene_epoch == scene_epoch
            and self.planning_scene_revision == planning_scene_revision
        )
        valid_goal_pool = bool(self.object_goals) and matching_goal_pool
        fast_frontier = getattr(self.qualifier, "qualification_profile", "legacy") == "fast_v3"
        # The grasp qualifier supplies a two-branch outer beam.  Pair
        # qualification must preserve a distinct-grasp backup so one
        # stochastic physical contact failure advances the already-proven
        # queue instead of triggering a 192-pair recovery qualification.
        primary_target = 1
        backup_target = 1
        target = primary_target + backup_target
        if matching_goal_pool and not self.object_goals:
            result.details.update(
                {
                    "frozen_pair_stop_reason": "frozen_goal_pool_exhausted",
                    "frozen_pair_full_plan_pass_count": 0,
                    "frozen_goal_legality_frontier_count": 0,
                }
            )
            return self._replace_grasps(
                result,
                [],
                {},
                scene_epoch,
                planning_scene_revision,
            )
        if not fast_frontier or not valid_goal_pool:
            return self._filter_grasp_batch(
                result,
                scene_epoch=scene_epoch,
                planning_scene_revision=planning_scene_revision,
                source=source,
            )

        aggregate = result
        current = result
        retained: dict[str, JsonDict] = {}
        retained_cache: dict[str, JsonDict] = {}
        retained_proofs: dict[str, Mapping[str, object]] = {}
        retained_goals: dict[str, list[JsonDict]] = {}
        pair_artifacts: list[JsonDict] = []
        grasp_artifacts: list[JsonDict] = []
        expansion_count = 0
        pair_totals = {
            "frozen_pair_count": 0,
            "frozen_pair_lookahead_grasp_count": 0,
            "frozen_pair_workspace_pass_count": 0,
            "frozen_pair_endpoint_evaluated_count": 0,
            "frozen_pair_endpoint_not_evaluated_count": 0,
            "frozen_pair_endpoint_pass_count": 0,
            "frozen_pair_full_plan_submitted_count": 0,
            "frozen_pair_full_plan_pass_count": 0,
        }
        reserve_activated = False
        deferred_count = 0
        goal_pool_exhausted = False
        backup_parent_priority_count = 0

        def retained_quality_order() -> list[str]:
            """Rank complete branches across every frozen-frontier batch."""

            return sorted(
                retained,
                key=lambda grasp_id: (
                    *candidate_physical_quality_key(
                        retained_proofs.get(grasp_id, retained[grasp_id])
                    ),
                    grasp_id,
                ),
            )

        while True:
            batch_input_grasps = current.details.get("grasp_candidates")
            batch_input_grasps = (
                [json.loads(json.dumps(candidate)) for candidate in batch_input_grasps]
                if isinstance(batch_input_grasps, list)
                else []
            )
            needed = max(1, target - len(retained))
            filtered = self._filter_grasp_batch(
                current,
                scene_epoch=scene_epoch,
                planning_scene_revision=planning_scene_revision,
                source=source,
                # Each grasp batch may opportunistically prove two different
                # branches inside its current wave, but one complete pair is
                # enough to advance the outer best-first frontier.  Requiring
                # two from the same batch made an unplaceable secondary grasp
                # exhaust all 96 goals before the next model-native sibling
                # could be tried.
                l5_pass_target=1,
            )
            if not filtered.success:
                return filtered
            for key in pair_totals:
                value = filtered.details.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    pair_totals[key] += value
            reserve_activated = reserve_activated or (
                filtered.details.get("frozen_pair_reserve_activated") is True
            )
            pair_artifact = filtered.details.get("frozen_pair_qualification_artifact")
            if isinstance(pair_artifact, Mapping):
                pair_artifacts.append(json.loads(json.dumps(pair_artifact)))

            if (
                filtered.details.get("frozen_goal_legality_screen_complete") is True
                and not self.object_goals
            ):
                goal_pool_exhausted = True
                break

            batch_goals = self.qualified_goals_by_grasp
            batch_grasps = filtered.details.get("grasp_candidates")
            batch_grasps = batch_grasps if isinstance(batch_grasps, list) else []
            for grasp in batch_grasps:
                if not isinstance(grasp, Mapping):
                    continue
                grasp_id = str(grasp.get("id") or "")
                if not grasp_id or grasp_id in retained:
                    continue
                entry = self.qualifier.cache.resolve(
                    purpose="grasp",
                    candidate_id=grasp_id,
                    scene_epoch=scene_epoch,
                    planning_scene_revision=planning_scene_revision,
                )
                if not isinstance(entry, Mapping):
                    continue
                retained[grasp_id] = json.loads(json.dumps(grasp))
                cached_candidate = entry.get("candidate")
                if isinstance(cached_candidate, Mapping):
                    retained_cache[grasp_id] = json.loads(json.dumps(cached_candidate))
                proof = entry.get("proof")
                if isinstance(proof, Mapping):
                    retained_proofs[grasp_id] = json.loads(json.dumps(proof))
                goals = batch_goals.get(grasp_id)
                if isinstance(goals, list):
                    retained_goals[grasp_id] = json.loads(json.dumps(goals))

            if len(retained) >= target:
                selected_ids = set(retained_quality_order()[:target])
                known_frontier_ids = {
                    str(candidate.get("id") or "")
                    for candidate in self.grasp_frontier_candidates
                    if isinstance(candidate, Mapping)
                }
                deferred = [
                    candidate
                    for candidate in batch_input_grasps
                    if isinstance(candidate, Mapping)
                    and str(candidate.get("id") or "")
                    and str(candidate.get("id") or "") not in selected_ids
                    and str(candidate.get("id") or "") not in known_frontier_ids
                ]
                if deferred:
                    self.grasp_frontier_candidates = [
                        *deferred,
                        *self.grasp_frontier_candidates,
                    ]
                    deferred_count += len(deferred)
                break
            if retained:
                primary_id = next(iter(retained))
                backup_parent_priority_count = max(
                    backup_parent_priority_count,
                    self.prioritize_grasp_frontier_for_parent(primary_id),
                )
            if not self.grasp_frontier_candidates:
                break

            previous_frontier_ids = tuple(
                str(candidate.get("id") or "")
                for candidate in self.grasp_frontier_candidates
                if isinstance(candidate, Mapping)
            )
            expansion = self.prepare_grasp_frontier_expansion(
                scene_epoch=scene_epoch,
                planning_scene_revision=planning_scene_revision,
            )
            if not expansion.success:
                return expansion
            provider_snapshot = ToolResult(
                True,
                expansion.content,
                json.loads(json.dumps(expansion.details)),
            )
            needed = max(1, target - len(retained))
            current = self.qualifier.qualify_result(
                expansion,
                purpose="grasp",
                scene_epoch=scene_epoch,
                planning_scene_revision=planning_scene_revision,
                source=source,
                l5_pass_target=needed,
                l5_min_pass_target=needed,
            )
            if _qualification_infrastructure_reason(current):
                return _qualification_infrastructure_failure(current)
            grasp_artifact = current.details.get("qualification_artifact")
            if isinstance(grasp_artifact, Mapping):
                grasp_artifacts.append(json.loads(json.dumps(grasp_artifact)))
            self.update_grasp_frontier(
                provider_snapshot,
                current,
                scene_epoch=scene_epoch,
                planning_scene_revision=planning_scene_revision,
            )
            expansion_count += 1
            next_frontier_ids = tuple(
                str(candidate.get("id") or "")
                for candidate in self.grasp_frontier_candidates
                if isinstance(candidate, Mapping)
            )
            if previous_frontier_ids == next_frontier_ids:
                return ToolResult(
                    False,
                    "Frozen grasp frontier made no deterministic progress.",
                    {
                        "reason": "frozen_grasp_frontier_no_progress",
                        "model_inference_invoked": False,
                        "execution_started": False,
                    },
                )

        final_grasps = [retained[grasp_id] for grasp_id in retained_quality_order()[:target]]
        for frontier_rank, grasp in enumerate(final_grasps):
            grasp["grasp_place_frontier_quality_rank"] = frontier_rank
        final_ids = [str(grasp.get("id") or "") for grasp in final_grasps]
        self.qualified_goals_by_grasp = {
            grasp_id: retained_goals[grasp_id]
            for grasp_id in final_ids
            if grasp_id in retained_goals
        }
        details = aggregate.details
        details.update(pair_totals)
        details.update(
            {
                "frozen_pair_grasp_branch_limit": self.grasp_branch_limit,
                "frozen_pair_primary_grasp_count": min(primary_target, len(final_grasps)),
                "frozen_pair_reserve_grasp_count": max(0, len(final_grasps) - primary_target),
                "frozen_pair_reserve_activated": reserve_activated,
                "frozen_pair_execution_target": primary_target,
                "frozen_pair_backup_target": backup_target,
                "frozen_pair_backup_required": True,
                "frozen_pair_backup_ready": len(final_grasps) >= target,
                "frozen_pair_deferred_grasp_count": deferred_count,
                "frozen_pair_recovery_policy": ("resume_frozen_frontier_after_execution_failure"),
                "frozen_pair_frontier_expansion_count": expansion_count,
                "frozen_pair_backup_parent_priority_count": (backup_parent_priority_count),
                "frozen_grasp_frontier_remaining_count": len(self.grasp_frontier_candidates),
                "frozen_grasp_frontier_generation": self.grasp_frontier_generation,
                "frozen_grasp_frontier_model_inference_invoked": False,
                "frozen_pair_stop_reason": (
                    "frozen_goal_pool_exhausted"
                    if goal_pool_exhausted
                    else "complete_pair_with_backup_found"
                    if len(final_grasps) >= target
                    else "complete_pair_found_redundancy_degraded"
                    if final_grasps
                    else "frozen_grasp_frontier_exhausted"
                ),
                "ranking": "grasp_place_physical_quality",
            }
        )
        if pair_artifacts:
            details["frozen_pair_qualification_artifact"] = pair_artifacts[0]
            details["frozen_pair_qualification_artifacts"] = pair_artifacts
        if grasp_artifacts:
            details["frozen_grasp_frontier_qualification_artifacts"] = grasp_artifacts
        artifacts = details.setdefault("artifacts", [])
        if isinstance(artifacts, list):
            known_paths = {
                str(item.get("path") or "") for item in artifacts if isinstance(item, Mapping)
            }
            for artifact in [*grasp_artifacts, *pair_artifacts]:
                path = str(artifact.get("path") or "")
                if path and path not in known_paths:
                    artifacts.append(artifact)
                    known_paths.add(path)
        return self._replace_grasps(
            aggregate,
            final_grasps,
            {
                grasp_id: retained_proofs[grasp_id]
                for grasp_id in final_ids
                if grasp_id in retained_proofs
            },
            scene_epoch,
            planning_scene_revision,
            cache_grasps=[
                retained_cache[grasp_id] for grasp_id in final_ids if grasp_id in retained_cache
            ],
        )

    def _filter_grasp_batch(
        self,
        result: ToolResult,
        *,
        scene_epoch: int,
        planning_scene_revision: int,
        source: Mapping[str, object],
        l5_pass_target: int | None = None,
    ) -> ToolResult:
        grasps = result.details.get("grasp_candidates")
        if not isinstance(grasps, list) or not grasps:
            return result
        matching_goal_pool = (
            isinstance(self.object_current_pose, Mapping)
            and self.scene_epoch == scene_epoch
            and self.planning_scene_revision == planning_scene_revision
        )
        if not matching_goal_pool:
            # A stale or absent pool cannot influence grasp qualification.
            return result
        if not self.object_goals:
            result.details.update(
                {
                    "frozen_pair_stop_reason": "frozen_goal_pool_exhausted",
                    "frozen_pair_full_plan_pass_count": 0,
                    "frozen_goal_legality_frontier_count": 0,
                }
            )
            return self._replace_grasps(
                result,
                [],
                {},
                scene_epoch,
                planning_scene_revision,
            )

        retained_entries: dict[str, JsonDict] = {}
        per_grasp_pairs: list[list[JsonDict]] = []
        # Preserve the complete current AnyPlace pool through compilation and
        # conservative structural screening.  Preselecting a handful by
        # farthest-first SE(3) overrepresents extreme object rotations and can
        # discard every attachment-aware reachable goal before MoveIt sees it.
        # L3/L4 then traverse the round-robin pair order progressively until
        # the two-slot plan-only capacity is filled or the batch is exhausted.
        current_goals = [dict(goal) for goal in self.object_goals]
        for grasp_index, grasp in enumerate(grasps[: self.grasp_branch_limit]):
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
            contact_state = (
                stages[-1].get("end_joint_state") if isinstance(stages[-1], Mapping) else None
            )
            if not isinstance(contact, Mapping) or not isinstance(contact_state, Mapping):
                continue
            retained_entries[grasp_id] = dict(entry)
            grasp_equivalence_id = grasp_symmetry_family_id(grasp)
            predicted_attachment = predicted_attachment_from_grasp(
                contact_pose=contact,
                object_current_pose=self.object_current_pose,
            )
            pairs: list[JsonDict] = []
            for goal in current_goals:
                pair = dict(goal)
                goal_id = str(goal.get("id") or "goal")
                pair["id"] = f"frozen_pair_{grasp_id}_{goal_id}"
                pair["source_grasp_id"] = grasp_id
                pair["source_grasp_equivalence_id"] = grasp_equivalence_id
                pair["source_grasp_symmetry_equivalent"] = bool(grasp.get("symmetry_parent_id"))
                pair["frozen_pair_batch_index"] = grasp_index // 2
                pair["frozen_pair_batch_role"] = "primary" if grasp_index < 2 else "reserve"
                pair["source_object_goal_id"] = goal_id
                pair["frozen_contact_pose"] = dict(contact)
                pair["predicted_attachment_transform"] = dict(predicted_attachment)
                _restore_frozen_model_motion_for_predicted_pair(pair)
                alignment = grasp.get("target_closing_alignment")
                if isinstance(alignment, Mapping):
                    pair["target_closing_alignment"] = json.loads(json.dumps(alignment))
                score = grasp.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    pair["score"] = float(score)
                physical_rebase = grasp.get("frozen_object_motion_rebase")
                if isinstance(physical_rebase, Mapping):
                    pair["frozen_object_motion_rebase"] = json.loads(json.dumps(physical_rebase))
                pair["qualification_start_joint_state"] = dict(contact_state)
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
                "bounded frozen grasp-goal pair qualification",
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
            qualification_mode="frozen_pair",
            l5_pass_target=max(2, l5_pass_target or 1),
            l5_min_pass_target=l5_pass_target or 1,
        )
        if _qualification_infrastructure_reason(joint):
            return _qualification_infrastructure_failure(joint)
        passed_pairs = joint.details.get("placement_candidates")
        passed_pairs = passed_pairs if isinstance(passed_pairs, list) else []
        goal_legality_summary = self._cache_goal_legality_frontier(
            joint,
            pairs=pairs,
        )
        result.details.update(goal_legality_summary)
        pass_count: dict[str, int] = {}
        goal_ids: dict[str, list[str]] = {}
        physical_rank_by_grasp: dict[str, int] = {}
        goal_lookup = {
            str(goal.get("id") or ""): goal
            for goal in self.object_goals
            if isinstance(goal, Mapping) and str(goal.get("id") or "")
        }
        qualified_goals: dict[str, list[JsonDict]] = {}
        for pair_rank, pair in enumerate(passed_pairs):
            if not isinstance(pair, Mapping):
                continue
            grasp_id = str(pair.get("source_grasp_id") or "")
            if not grasp_id:
                continue
            physical_rank_by_grasp.setdefault(grasp_id, pair_rank)
            pass_count[grasp_id] = pass_count.get(grasp_id, 0) + 1
            goal_id = str(pair.get("source_object_goal_id") or "")
            goal_ids.setdefault(grasp_id, []).append(goal_id)
            original = goal_lookup.get(goal_id)
            if not isinstance(original, Mapping):
                continue
            frozen_goal = json.loads(json.dumps(original))
            seed_evidence = pair.get(SAME_RUN_QUALIFICATION_SEED_FIELD)
            if isinstance(seed_evidence, Mapping):
                frozen_goal[SAME_RUN_QUALIFICATION_SEED_FIELD] = json.loads(
                    json.dumps(seed_evidence)
                )
            physical_goal = pair.get("qualified_world_collision_object_goal_pose")
            if isinstance(physical_goal, Mapping):
                frozen_goal = self._bind_physical_collision_goal(
                    frozen_goal,
                    goal_id=goal_id,
                    collision_goal=physical_goal,
                )
                if isinstance(seed_evidence, Mapping):
                    frozen_goal[SAME_RUN_QUALIFICATION_SEED_FIELD] = json.loads(
                        json.dumps(seed_evidence)
                    )
            qualified_goals.setdefault(grasp_id, []).append(frozen_goal)
        self.qualified_goals_by_grasp = qualified_goals
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
            annotated["grasp_place_physical_quality_rank"] = physical_rank_by_grasp[grasp_id]
            retained.append(annotated)
            cached_candidate = entry.get("candidate")
            if isinstance(cached_candidate, Mapping):
                cache_grasps.append(dict(cached_candidate))
            if isinstance(entry.get("proof"), Mapping):
                proofs[grasp_id] = entry["proof"]
        result.details["frozen_pair_count"] = len(pairs)
        result.details["frozen_pair_grasp_branch_limit"] = self.grasp_branch_limit
        result.details["frozen_pair_lookahead_grasp_count"] = len(retained_entries)
        result.details["frozen_pair_primary_grasp_count"] = min(2, len(retained_entries))
        result.details["frozen_pair_reserve_grasp_count"] = max(0, len(retained_entries) - 2)
        qualification_waves = joint.details.get("qualification_waves")
        qualification_waves = qualification_waves if isinstance(qualification_waves, list) else []
        result.details["frozen_pair_reserve_activated"] = any(
            isinstance(wave, Mapping) and wave.get("frozen_pair_batch_index") == 1
            for wave in qualification_waves
        )
        result.details["frozen_pair_workspace_pass_count"] = joint.details.get(
            "workspace_pass_count", 0
        )
        result.details["frozen_pair_endpoint_evaluated_count"] = joint.details.get(
            "endpoint_evaluated_count", 0
        )
        result.details["frozen_pair_endpoint_not_evaluated_count"] = joint.details.get(
            "endpoint_not_evaluated_count", 0
        )
        result.details["frozen_pair_endpoint_pass_count"] = joint.details.get(
            "endpoint_pass_count", 0
        )
        result.details["frozen_pair_full_plan_submitted_count"] = joint.details.get(
            "full_plan_submitted_count", 0
        )
        result.details["frozen_pair_full_plan_pass_count"] = joint.details.get(
            "full_plan_pass_count", 0
        )
        if retained:
            retained.sort(
                key=lambda grasp: (
                    *parallel_gripper_centering_quality(grasp),
                    int(grasp.get("grasp_place_physical_quality_rank", 1_000_000)),
                )
            )
            result.details["ranking"] = (
                "parallel_gripper_centering_then_grasp_place_physical_quality"
            )
        if joint.details.get("qualification_artifact"):
            result.details["frozen_pair_qualification_artifact"] = joint.details[
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
        resume_frontier: bool = False,
        excluded_goal_ids: Sequence[str] = (),
    ) -> ToolResult | None:
        """Prepare one measured-attachment frontier without model inference."""

        priority_goals = self.qualified_goals_by_grasp.get(source_grasp_id)
        if not priority_goals:
            return None
        binding = self.attachment_binding(
            source_grasp_id=source_grasp_id,
            attachment_transform=attachment_transform,
        )
        excluded = {str(goal_id) for goal_id in excluded_goal_ids if str(goal_id)}
        if binding in self.consumed_attachment_bindings:
            exposed = self.attachment_exposed_goal_ids.get(binding, set())
            previous = self.attachment_prepared_exclusions.get(binding, frozenset())
            if not (
                resume_frontier
                and excluded
                and excluded.issubset(exposed)
                and previous.issubset(excluded)
                and frozenset(excluded) != previous
            ):
                return None
            generation = self.attachment_frontier_generations.get(binding, 0) + 1
        else:
            if resume_frontier or excluded:
                return None
            self.consumed_attachment_bindings.add(binding)
            self.attachment_exposed_goal_ids[binding] = set()
            generation = 0
        self.attachment_prepared_exclusions[binding] = frozenset(excluded)
        self.attachment_frontier_generations[binding] = generation
        self.active_attachment_binding = binding
        # A predicted attachment can make a goal look executable before the
        # grasp, then fail after the measured attachment transform is known.
        # Keep those L5-PASS goals at the head for the common fast path, but
        # continue from the complete frozen AnyPlace pool on failure.  The
        # post-attachment qualifier repeats goal/pair legality and advances
        # through its small deterministic waves until the first complete L5
        # PASS; merely materializing this frontier does not plan all 96 goals.
        frozen: list[JsonDict] = []
        prioritized_ids: set[str] = set()
        included_priority_count = 0
        for goal in priority_goals:
            goal_id = str(goal.get("id") or "")
            if goal_id and goal_id in excluded:
                continue
            frozen.append(json.loads(json.dumps(goal)))
            included_priority_count += 1
            if goal_id:
                prioritized_ids.add(goal_id)
        for goal in self.object_goals:
            goal_id = str(goal.get("id") or "")
            if goal_id and (goal_id in prioritized_ids or goal_id in excluded):
                continue
            frozen.append(json.loads(json.dumps(goal)))
        artifacts: list[JsonDict] = []
        for source_artifact in self.source_candidate_artifacts:
            artifact = json.loads(json.dumps(source_artifact))
            artifact["provenance"] = "frozen_anyplace_goal_pool"
            artifact["reused_for_measured_attachment_requalification"] = True
            artifact["anyplace_model_inference_invoked"] = False
            artifacts.append(artifact)
        metadata = {
            "candidate_source": "frozen_anyplace_goal_frontier",
            "priority_pair_pass_goal_count": included_priority_count,
            "frozen_frontier_goal_count": len(frozen),
            "frozen_frontier_excluded_goal_count": len(excluded),
            "frozen_frontier_generation": generation,
            "source_model_raw_candidate_count": self.source_model_raw_candidate_count,
            "anyplace_model_inference_invoked": False,
        }
        return ToolResult(
            True,
            (
                f"Prepared {len(frozen)} frozen object goals, with "
                f"{included_priority_count} predicted pair-PASS goals first, for "
                f"measured-attachment frontier generation {generation} without "
                "AnyPlace inference."
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
                "frozen_goal_requalification": True,
                "frozen_goal_count": len(frozen),
                "frozen_goal_priority_count": included_priority_count,
                "frozen_goal_frontier_count": len(frozen),
                "frozen_goal_excluded_count": len(excluded),
                "frozen_goal_total_eligible_count": len(frozen) + len(excluded),
                "frozen_goal_frontier_generation": generation,
                "frozen_goal_frontier_resume": resume_frontier,
                "discarded_postattach_model_candidate_count": 0,
                "anyplace_model_inference_invoked": False,
                "execution_started": False,
            },
        )

    def record_attachment_qualification(self, result: ToolResult) -> None:
        """Record only goals actually exposed by a measured-attachment round."""

        binding = self.active_attachment_binding
        if not binding or result.details.get("frozen_goal_requalification") is not True:
            return
        candidates = result.details.get("placement_candidates")
        if not isinstance(candidates, list):
            return
        exposed = self.attachment_exposed_goal_ids.setdefault(binding, set())
        exposed.update(
            str(candidate.get("id") or "")
            for candidate in candidates
            if isinstance(candidate, Mapping) and str(candidate.get("id") or "")
        )
        result.details["frozen_goal_exposed_count"] = len(exposed)

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
        return (
            self.attachment_binding(
                source_grasp_id=source_grasp_id,
                attachment_transform=attachment_transform,
            )
            in self.consumed_attachment_bindings
        )

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
        result.details["frozen_pair_qualified_grasp_count"] = len(grasps)
        return result


def _prepare_postattachment_frozen_goals(
    context: ToolExecutionContext,
    request: JsonDict,
    *,
    coordinator: _FrozenGoalPairCoordinator,
) -> ToolResult | None:
    """Short-circuit inference with the selected grasp's frozen goal frontier."""

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
    full_proof = attachment_gate.get("attachment_proof")
    attachment_transform = (
        full_proof.get("attachment_transform") if isinstance(full_proof, Mapping) else None
    )
    if not isinstance(attachment_transform, Mapping):
        return ToolResult(
            False,
            "measured attachment transform is required for frozen-goal requalification",
            {
                "reason": "frozen_goal_attachment_transform_missing",
                "execution_started": False,
            },
        )
    compiled_source_grasp = execution.get("compiled_grasp")
    source_grasp_id = _active_source_grasp_id(
        memory,
        compiled_source_grasp=compiled_source_grasp,
    )
    if not source_grasp_id:
        return ToolResult(
            False,
            "source grasp identity is required for frozen-goal requalification",
            {
                "reason": "frozen_goal_source_grasp_missing",
                "execution_started": False,
            },
        )
    if not coordinator.has_frozen_goals(source_grasp_id):
        return ToolResult(
            False,
            "no frozen AnyPlace goals are bound to the attached grasp",
            {
                "reason": "frozen_goal_pool_missing",
                "execution_started": False,
            },
        )
    scene_revision = request.get("scene_revision")
    if not isinstance(scene_revision, int) or isinstance(scene_revision, bool):
        return ToolResult(
            False,
            "scene revision is required for frozen-goal requalification",
            {
                "reason": "frozen_goal_scene_revision_missing",
                "execution_started": False,
            },
        )
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
    resume_frontier = request.get("resume_frozen_goal_frontier") is True
    if (
        coordinator.attachment_binding_consumed(
            source_grasp_id=source_grasp_id,
            attachment_transform=attachment_transform,
        )
        and not resume_frontier
    ):
        return ToolResult(
            False,
            "The measured-attachment frozen-goal pool was already consumed.",
            {
                "reason": "frozen_goal_requalification_already_consumed",
                "execution_started": False,
            },
        )
    raw_excluded_goal_ids = request.get("excluded_frozen_goal_ids", [])
    if not isinstance(raw_excluded_goal_ids, list) or any(
        not isinstance(goal_id, str) or not goal_id for goal_id in raw_excluded_goal_ids
    ):
        return ToolResult(
            False,
            "frozen placement frontier exclusions are invalid",
            {
                "reason": "frozen_goal_frontier_exclusions_invalid",
                "execution_started": False,
            },
        )
    placement_observation = request.get("placement_observation")
    if isinstance(placement_observation, Mapping):
        source = {
            "object_observation": json.loads(json.dumps(request.get("object_observation"))),
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
    elif request.get("reuse_frozen_goal_pool") is True and coordinator.source_binding:
        source = json.loads(json.dumps(coordinator.source_binding))
    else:
        return None
    return coordinator.prepare_frozen_goal_requalification(
        source_grasp_id=source_grasp_id,
        attachment_transform=attachment_transform,
        source=source,
        scene_revision=scene_revision,
        resume_frontier=resume_frontier,
        excluded_goal_ids=raw_excluded_goal_ids,
    )


def _qualifying_handler(
    handler: ToolHandler,
    qualifier: MoveItCandidateQualifier,
    *,
    purpose: str,
    frozen_pair_coordinator: _FrozenGoalPairCoordinator | None = None,
    candidate_compiler: ToolHandler | None = None,
) -> ToolHandler:
    """Apply private MoveIt qualification before a result reaches memory/VLM."""

    def qualified(context: ToolExecutionContext) -> ToolResult:
        supervision = context.metadata.get("supervision_context")
        memory = supervision.get("memory") if isinstance(supervision, dict) else None
        placement_policy = (
            memory.get("placement_candidate_policy") if isinstance(memory, dict) else None
        )
        if (
            purpose == "placement"
            and isinstance(placement_policy, dict)
            and placement_policy.get("status") == "stopped_requires_human"
        ):
            # Qualification is bounded.  Once the frozen pool is exhausted
            # and its bounded recovery is unavailable, fresh model
            # samples cannot safely alter the current attached state.
            return ToolResult(
                False,
                "CURRENT_GRASP_PLACE_INFEASIBLE: placement recovery is blocked pending human intervention.",
                {
                    "reason": "CURRENT_GRASP_PLACE_INFEASIBLE",
                    "execution_started": False,
                },
            )
        observation_metadata = (
            context.observation.metadata if context.observation is not None else {}
        )
        frozen_frontier_requested = (
            purpose == "grasp"
            and frozen_pair_coordinator is not None
            and context.parameters.get("mode") == "frozen_frontier"
            and context.parameters.get("model_inference") is False
        )
        if frozen_frontier_requested:
            frontier_scene_epoch = (
                memory.get("scene_epoch")
                if isinstance(memory, Mapping)
                and isinstance(memory.get("scene_epoch"), int)
                and not isinstance(memory.get("scene_epoch"), bool)
                else observation_metadata.get("scene_epoch", 0)
            )
            frontier_revision = context.parameters.get("scene_revision")
            if not isinstance(frontier_revision, int) or isinstance(frontier_revision, bool):
                frontier_revision = observation_metadata.get(
                    "planning_scene_revision", frontier_scene_epoch
                )
            observed_revision = observation_metadata.get("planning_scene_revision")
            recovery = memory.get("grasp_recovery") if isinstance(memory, Mapping) else None
            failed_candidate_id = (
                str(recovery.get("candidate_id") or "")
                if isinstance(recovery, Mapping)
                else ""
            )
            if (
                isinstance(observed_revision, int)
                and not isinstance(observed_revision, bool)
                and isinstance(frontier_revision, int)
                and not isinstance(frontier_revision, bool)
                and observed_revision != frontier_revision
            ):
                latest_receipt = (
                    memory.get("planning_scene_target_pose_sync")
                    if isinstance(memory, Mapping)
                    else None
                )
                rebase = frozen_pair_coordinator.rebase_grasp_frontier_from_target_pose_sync(
                    latest_receipt if isinstance(latest_receipt, Mapping) else {},
                    scene_epoch=(
                        frontier_scene_epoch
                        if isinstance(frontier_scene_epoch, int)
                        and not isinstance(frontier_scene_epoch, bool)
                        else 0
                    ),
                    planning_scene_revision=observed_revision,
                    failed_candidate_id=failed_candidate_id,
                )
                if not rebase.success:
                    return ToolResult(
                        False,
                        (
                            "The frozen grasp frontier no longer matches the "
                            "observed PlanningScene revision."
                        ),
                        {
                            "reason": "frozen_grasp_frontier_scene_revision_changed",
                            "source_planning_scene_revision": frontier_revision,
                            "planning_scene_revision": observed_revision,
                            "rebase_reason": rebase.details.get("reason"),
                            "model_inference_invoked": False,
                            "execution_started": False,
                        },
                    )
                frontier_revision = observed_revision
            preferred_parent_count = (
                frozen_pair_coordinator.prioritize_grasp_frontier_for_parent(failed_candidate_id)
                if failed_candidate_id
                else 0
            )
            result = frozen_pair_coordinator.prepare_grasp_frontier_expansion(
                scene_epoch=(
                    frontier_scene_epoch
                    if isinstance(frontier_scene_epoch, int)
                    and not isinstance(frontier_scene_epoch, bool)
                    else 0
                ),
                planning_scene_revision=(
                    frontier_revision
                    if isinstance(frontier_revision, int)
                    and not isinstance(frontier_revision, bool)
                    else 0
                ),
            )
            if result.success:
                result.details.update(
                    {
                        "frozen_frontier_failed_parent_candidate_id": (failed_candidate_id),
                        "frozen_frontier_preferred_parent_variant_count": (preferred_parent_count),
                    }
                )
        else:
            result = handler(context)
        if not result.success:
            return result
        provider_result_snapshot = (
            ToolResult(
                True,
                result.content,
                json.loads(json.dumps(result.details)),
            )
            if purpose == "grasp"
            and frozen_pair_coordinator is not None
            and frozen_pair_coordinator.object_goals
            else None
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
        source.setdefault(
            "provider",
            str(result.details.get("provider") or result.details.get("backend") or context.name),
        )
        source.setdefault(
            "provider_version",
            str(
                result.details.get("provider_version")
                or result.details.get("model_version")
                or result.details.get("checkpoint_sha256")
                or "unknown"
            ),
        )
        if context.observation is not None:
            source["start_joint_state"] = context.observation.robot.to_dict()
            frame_id = str(
                source.get("camera_frame_id") or result.details.get("camera_frame_id") or ""
            )
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
                # A moving wrist camera must retain the exact extrinsics of the
                # RGB-D packet used by the model.  Frozen-frontier retries update
                # only the robot start state; replacing these extrinsics with the
                # post-contact wrist transform would move every raw candidate.
                source.setdefault("camera_extrinsics", dict(camera.extrinsics))
        if purpose == "placement":
            attachment_gate = memory.get("attachment_gate") if isinstance(memory, dict) else None
            attachment_proof = (
                attachment_gate.get("attachment_proof")
                if isinstance(attachment_gate, dict)
                else None
            )
            attachment_transform = (
                attachment_proof.get("attachment_transform")
                if isinstance(attachment_proof, dict)
                else None
            )
            if not isinstance(attachment_transform, dict) and frozen_pair_coordinator is not None:
                return frozen_pair_coordinator.retain_goal_pool(
                    result,
                    source=source,
                    scene_epoch=scene_epoch,
                    planning_scene_revision=revision_value,
                )
            grasp_execution = memory.get("grasp_execution") if isinstance(memory, dict) else None
            compiled_source_grasp = (
                grasp_execution.get("compiled_grasp") if isinstance(grasp_execution, dict) else None
            )
            frozen_goal_requalification = result.details.get("frozen_goal_requalification") is True
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
                and (already_world_goals or isinstance(placement_extrinsics, dict))
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
                source["frozen_goal_requalification"] = True
            if isinstance(compiled_source_grasp, dict):
                source["source_grasp_compiled"] = dict(compiled_source_grasp)
        grasp_lookahead_target = (
            2
            if purpose == "grasp"
            and frozen_pair_coordinator is not None
            and frozen_pair_coordinator.object_goals
            else None
        )
        grasp_minimum_target = 2 if grasp_lookahead_target is not None else None
        qualified_result = qualifier.qualify_result(
            result,
            purpose=purpose,
            scene_epoch=scene_epoch,
            planning_scene_revision=revision_value,
            source=source,
            l5_pass_target=grasp_lookahead_target,
            l5_min_pass_target=grasp_minimum_target,
        )
        if _qualification_infrastructure_reason(qualified_result):
            return _qualification_infrastructure_failure(qualified_result)
        if (
            purpose == "placement"
            and frozen_goal_requalification
            and frozen_pair_coordinator is not None
        ):
            frozen_pair_coordinator.record_attachment_qualification(qualified_result)
        if purpose == "grasp" and frozen_pair_coordinator is not None:
            if provider_result_snapshot is not None:
                provider_result_snapshot.details["source"] = json.loads(json.dumps(source))
                frozen_pair_coordinator.update_grasp_frontier(
                    provider_result_snapshot,
                    qualified_result,
                    scene_epoch=scene_epoch,
                    planning_scene_revision=revision_value,
                )
            qualified_result = frozen_pair_coordinator.filter_grasps(
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
        "openeta.compiled_placement_seed.v3"
        if purpose == "placement"
        else "openeta.compiled_grasp_seed.v2"
    )
    events: list[JsonDict] = []
    for queue_position, selected in enumerate(candidates):
        candidate_id = str(selected.get("id") or "") if isinstance(selected, Mapping) else ""
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
            failure_details = compiled.details if isinstance(compiled, ToolResult) else {}
            return ToolResult(
                False,
                "host failed to compile a qualified candidate",
                {
                    **result.details,
                    "reason": "host_candidate_compilation_failed",
                    "candidate_id": candidate_id,
                    "queue_position": queue_position,
                    "compilation_diagnostics": failure_details.get("diagnostics", []),
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
        for pose_key in ("contact_pose",):
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
            "source": "scripted_tui"
            if policy.profile == SupervisionProfile.SCRIPTED_TUI
            else "runtime_policy",
            "reason": "Scripted TUI permits session-local registry changes."
            if policy.profile == SupervisionProfile.SCRIPTED_TUI
            else "Standard profile permits session-local registry changes.",
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
