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
    build_wrist_alignment_handler,
    compile_grasp_seed,
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
from tools.candidate_config import DEFAULT_CANDIDATE_COUNT, candidate_count


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

    graspgenx: int = DEFAULT_CANDIDATE_COUNT
    anygrasp: int = DEFAULT_CANDIDATE_COUNT
    anyplace: int = DEFAULT_CANDIDATE_COUNT

    def __post_init__(self) -> None:
        object.__setattr__(self, "graspgenx", candidate_count(self.graspgenx))
        object.__setattr__(self, "anygrasp", candidate_count(self.anygrasp))
        object.__setattr__(self, "anyplace", candidate_count(self.anyplace))


def runtime_candidate_counts_from_env() -> RuntimeCandidateCounts:
    return RuntimeCandidateCounts(
        graspgenx=candidate_count(
            os.environ.get("OPENETA_GRASPGENX_MAX_CANDIDATES", DEFAULT_CANDIDATE_COUNT)
        ),
        anygrasp=candidate_count(
            os.environ.get("OPENETA_ANYGRASP_MAX_CANDIDATES", DEFAULT_CANDIDATE_COUNT)
        ),
        anyplace=candidate_count(
            os.environ.get("OPENETA_ANYPLACE_CANDIDATE_COUNT", DEFAULT_CANDIDATE_COUNT)
        ),
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
    )
    tools.bind_handler(
        "compile_grasp_seed",
        build_compile_grasp_seed_handler(
            workspace.grasp_profile_path,
            strategy_root=workspace.grasp_strategy_root,
            qualification_cache=qualification_cache if qualifier is not None else None,
        ),
        replace=True,
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
) -> DepthPriorPrefetchCoordinator | None:
    counts = candidate_counts or runtime_candidate_counts_from_env()
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
            expected_candidate_count=counts.anyplace,
        )
        if candidate_qualifier is not None:
            anyplace_handler = _qualifying_handler(
                anyplace_handler, candidate_qualifier, purpose="placement"
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
            expected_candidate_count=counts.anygrasp,
        )
    if endpoints.graspgenx_url:
        list_grippers = build_sse_graspgenx_mcp_gripper_lister(
            url=endpoints.graspgenx_url
        )
        grasp_backends["graspgenx"] = build_graspgenx_handler(
            build_sse_graspgenx_mcp_predictor(url=endpoints.graspgenx_url),
            list_grippers,
            output_root=artifact_root / "graspgenx_results",
            expected_candidate_count=counts.graspgenx,
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
            backend_order=("graspgenx", "anygrasp", "contact_graspnet"),
            graspgenx_gripper_name="robotiq_2f_85",
        )
        if candidate_qualifier is not None:
            grasp_handler = _qualifying_handler(
                grasp_handler, candidate_qualifier, purpose="grasp"
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
    return MoveItCandidateQualifier(
        private_qualification_rpc(
            transport,
            handle_provider=lambda: proxy_config.handle,
            session_id_provider=lambda: proxy_config.session_id,
        ),
        cache=cache,
        artifact_root=artifact_root,
        compile_candidate=compile_candidate,
    )


def _candidate_qualification_compiler(
    workspace: SessionWorkspace,
) -> Callable:
    """Compile immutable candidates to the exact world poses sent to MoveIt."""

    profile_path = Path(workspace.grasp_profile_path)

    def compile_one(
        candidate: Mapping[str, object],
        purpose: str,
        source: Mapping[str, object],
        scene_epoch: int,
        planning_scene_revision: int,
    ) -> JsonDict:
        profile_bytes = profile_path.read_bytes()
        profile = json.loads(profile_bytes.decode("utf-8"))
        profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
        extrinsics = source.get("camera_extrinsics")
        if not isinstance(extrinsics, dict):
            raise ValueError("camera extrinsics unavailable for qualification")
        if purpose == "placement":
            selected_grasp = source.get("selected_grasp")
            selected_grasp = selected_grasp if isinstance(selected_grasp, dict) else {}
            source_grasp = selected_grasp.get("candidate")
            if not isinstance(source_grasp, dict):
                raise ValueError("source grasp unavailable for placement qualification")
            parameters: JsonDict = {
                "purpose": "placement",
                "placement_candidate_id": str(candidate.get("id") or ""),
                "placement_candidate": dict(candidate),
                "source_grasp": dict(source_grasp),
                "camera_extrinsics": dict(extrinsics),
                "camera_frame_id": str(source.get("camera_frame_id") or ""),
                "scene_epoch": scene_epoch,
                "scene_revision": planning_scene_revision,
            }
            compiled = compile_grasp_seed(
                parameters,
                profile=profile,
                profile_sha256=profile_sha256,
            )
            compiled_pose_chain = [
                dict(compiled["hover_pose"]),
                dict(compiled["release_pose"]),
            ]
            stages = [
                _qualification_pose("hover", compiled["hover_pose"]),
                _qualification_pose("release", compiled["release_pose"]),
            ]
        else:
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
                strategies=load_grasp_strategies(Path(workspace.grasp_strategy_root)),
            )
            compiled_pose_chain = [dict(compiled["hover_pose"])]
            stages = [_qualification_pose("hover", compiled["hover_pose"])]
            precontact = compiled.get("precontact_pose")
            if isinstance(precontact, dict):
                compiled_pose_chain.append(dict(precontact))
                stages.append(_qualification_pose("precontact", precontact))
            compiled_pose_chain.append(dict(compiled["contact_pose"]))
            stages.append(_qualification_pose("contact", compiled["contact_pose"]))
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
            },
        }

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


def _qualifying_handler(
    handler: ToolHandler,
    qualifier: MoveItCandidateQualifier,
    *,
    purpose: str,
) -> ToolHandler:
    """Apply private MoveIt qualification before a result reaches memory/VLM."""

    def qualified(context: ToolExecutionContext) -> ToolResult:
        result = handler(context)
        if not result.success:
            return result
        observation_metadata = (
            context.observation.metadata if context.observation is not None else {}
        )
        scene_epoch_value = observation_metadata.get(
            "scene_epoch",
            result.details.get("scene_epoch", context.parameters.get("scene_epoch", 0)),
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
        return qualifier.qualify_result(
            result,
            purpose=purpose,
            scene_epoch=scene_epoch,
            planning_scene_revision=revision_value,
            source=source,
        )

    return qualified


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
