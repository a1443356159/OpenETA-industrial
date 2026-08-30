"""MCP-only simulator tool proxy for OpenETA runtime."""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import uuid4

from adapter.protocol import EnvAction, EnvObservation, JsonDict, RobotState, StepResult
from agent.runtime.artifact_paths import artifact_session_id
from agent.runtime.image_artifacts import (
    DEFAULT_MCP_IMAGE_OUTPUT_ROOT,
    materialize_mcp_images,
)
from agent.runtime.response_artifacts import (
    DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT,
    build_motion_summary,
    build_observation_snapshot,
    build_observation_summary,
    build_response_reference,
    materialize_json_response,
)
from agent.runtime.text_artifacts import (
    DEFAULT_MAX_INLINE_TEXT_CHARS,
    DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT,
)
from agent.tools.registry import (
    ENVIRONMENT_AUTHORITY,
    ToolExecutionContext,
    ToolHandler,
    ToolRegistry,
    ToolResult,
    make_tool_result,
    make_tool_result_details,
)


DEFAULT_SIMULATOR_MCP_TOOL_NAMES = (
    "create_simulator_env",
    "close_simulator_env",
    "observe",
    "configure_work_order",
    "move_to",
    "follow_eef_trajectory",
    "gripper_control",
)

DEFAULT_SIMULATOR_IMAGE_WIDTH = 512
DEFAULT_SIMULATOR_IMAGE_HEIGHT = 512
DEFAULT_MCP_SSE_READ_TIMEOUT_S = 300.0
MCP_SSE_TIMEOUT_GRACE_S = 5.0
DEFAULT_READ_ONLY_MCP_MAX_ATTEMPTS = 2
DEFAULT_READ_ONLY_MCP_HEALTH_TIMEOUT_S = 5.0
READ_ONLY_MCP_RETRY_RECEIPT_SCHEMA_VERSION = "openeta.read_only_mcp_retry.v1"
ENVIRONMENT_RECEIPT_SCHEMA_VERSION = "openeta.environment_receipt.v1"
SIMULATOR_STARTUP_RETRY_METADATA_KEY = "_openeta_simulator_startup_retry_attempt"

# Controller-produced evidence that must remain available after the raw MCP
# response is materialized as an artifact.  The simulator proxy is an
# environment authority, so copying this bounded set into its host-stamped
# receipt preserves the control-plane proof without trusting agent-authored
# parameters or requiring runtime code to reopen an artifact from disk.
CONTROL_RECEIPT_FIELDS = (
    "ok",
    "error_code",
    "moveit_error_code",
    "failure_class",
    "candidate_rejection",
    "motion_outcome",
    "execution_started",
    "request_fingerprint",
    "planning_scene_revision",
    "scene_revision",
    "stalled",
    "reached_goal",
    "terminal_status",
    "terminal_status_code",
    # Compact terminal diagnostics used to distinguish an empty planning
    # rejection from a completed-but-missed controller trajectory.  The full
    # joint-state proof remains in the host-only observation snapshot.
    "planned_point_count",
    "position_error_m",
    "orientation_error_rad",
    # A failed pre-close arm motion may safely continue only from a causal,
    # stationary state proved by the simulator adapter.  This compact record
    # contains no agent-authored pose and must survive response artifacting.
    "current_state_restart",
    "detachable_joint",
    "attachment_transform",
    "physical_verification",
    "child_link_proof",
    "placement_verification",
    # Multi-object sorting changes the authoritative target binding and
    # assignment in-place.  Preserve the bounded transition proof so memory
    # can associate each native release with its assignment and validate the
    # next target instead of falling back to the legacy singleton target.
    "native_target_binding",
    "work_order",
    "multi_sort_progress",
    "next_assignment_planning_scene_revision",
    # A native attach can be acknowledged before a later controller-owned
    # planning-scene or pose-read step fails.  These bounded fields distinguish
    # that infrastructure failure from an ordinary unreachable candidate and
    # prove whether the native/planning-scene rollback completed.
    "infrastructure_error",
    "attach_acked_before_rollback",
    "native_state_snapshot",
    # A release can fail only after the irreversible native detach and
    # physical open have already completed.  Preserve this controller-owned
    # boolean so AgentMemory can stop replaying open instead of treating the
    # artifact-truncated response as a reversible tool failure.
    "gripper_open_executed",
    # The release transition is atomic but spans four independently proven
    # environment events.  Keep the bounded ordered evidence in the trusted
    # host receipt so AgentMemory can consume the transition without reopening
    # the raw response artifact from disk.
    "release_sequence",
    # A failed physical close may resynchronize the detached target pose before
    # the frozen frontier resumes.  This controller-authored proof is likewise
    # needed by the host state machine, not by the model context.
    "planning_scene_target_pose_sync",
    "planning_scene_rollback",
)

SIMULATOR_CONTROL_MCP_TOOL_NAMES = (
    "move_to",
    "follow_eef_trajectory",
    "gripper_control",
)

DEFAULT_SIMULATOR_MCP_TOOL_MAP = {
    "create_simulator_env": "create_env",
    "close_simulator_env": "close_env",
    "observe": "render_env",
    "configure_work_order": "configure_work_order",
    "move_to": "move_to",
}


def mcp_server_url_from_endpoint(url: str) -> str:
    """Return the browser/API base URL for an MCP HTTP endpoint."""

    endpoint = str(url or "").strip().rstrip("/")
    for suffix in ("/sse", "/mcp"):
        if endpoint.endswith(suffix):
            return endpoint[: -len(suffix)]
    return endpoint


def mcp_server_url_from_transport(transport: object) -> str:
    """Best-effort server URL extraction from a configured MCP transport."""

    url = getattr(transport, "url", "")
    if not isinstance(url, str):
        return ""
    return mcp_server_url_from_endpoint(url)


def mcp_dashboard_url(server_url: str, session_id: object) -> str:
    """Return the simulator dashboard URL for a session when enough data exists."""

    session = str(session_id or "").strip()
    base = str(server_url or "").strip().rstrip("/")
    if not base or not session:
        return ""
    return f"{base}/session/{session}"


def _mcp_request_descriptor(mcp_tool: str, arguments: Mapping[str, object]) -> JsonDict:
    """Return the local, immutable descriptor for one MCP RPC attempt.

    The descriptor is evidence only: it is never added to the remote MCP
    arguments.  A matching id is embedded in the materialized response and
    environment receipt so formal TUI acceptance can prove the full chain.
    """

    return {
        "request_id": uuid4().hex,
        "tool": mcp_tool,
        "arguments": dict(arguments),
    }


class SimulatorMcpTransport(Protocol):
    """Synchronous MCP tool transport used by simulator tool proxies."""

    def list_tools(self, *, timeout_s: float | None = None) -> JsonDict:
        """List simulator MCP tools and return compact JSON metadata."""
        ...

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        """Call one simulator MCP tool and return its JSON payload."""
        ...


class SimulatorMcpTransportError(RuntimeError):
    """Typed transport failure that preserves a concrete nested SDK error."""

    def __init__(self, operation: str, cause: BaseException) -> None:
        primary = _primary_transport_exception(cause)
        if _is_transport_timeout(cause):
            code = "simulator_mcp_transport_timeout"
        elif _is_transient_mcp_transport_error(cause):
            code = "simulator_mcp_transport_connection_lost"
        else:
            code = "simulator_mcp_call_failed"
        message = str(primary).strip()
        detail = type(primary).__name__
        if message and message != detail:
            detail = f"{detail}: {message}"
        super().__init__(f"{operation} failed: {detail}")
        self.code = code
        self.operation = operation
        self.cause_type = type(primary).__name__


READ_ONLY_MCP_TRANSIENT_ERROR_CODES = frozenset(
    {
        "simulator_mcp_transport_timeout",
        "simulator_mcp_transport_connection_lost",
    }
)


def _call_mcp_tool_with_wall_timeout(
    transport: SimulatorMcpTransport,
    name: str,
    arguments: JsonDict,
    *,
    transport_timeout_s: float | None,
    wall_timeout_s: float,
) -> JsonDict:
    """Bound one read-only acknowledgement even if SDK cancellation stalls.

    Some MCP transports use a structured task group whose context cleanup can
    keep ``asyncio.wait_for`` blocked after its deadline.  The qualification
    request is read-only and binding-idempotent, so its first attempt may run to
    completion in a daemon thread while the caller health-checks and retrieves
    the same proof through the bounded retry.
    """

    completed = threading.Event()
    outcome: list[tuple[bool, object]] = []

    def invoke() -> None:
        try:
            outcome.append(
                (
                    True,
                    transport.call_tool(
                        name,
                        dict(arguments),
                        timeout_s=transport_timeout_s,
                    ),
                )
            )
        except BaseException as exc:  # noqa: BLE001 - cross-thread propagation.
            outcome.append((False, exc))
        finally:
            completed.set()

    threading.Thread(
        target=invoke,
        name=f"openeta-read-only-mcp-{name}",
        daemon=True,
    ).start()
    if not completed.wait(timeout=wall_timeout_s):
        raise SimulatorMcpTransportError(
            f"call_tool:{name}",
            TimeoutError(
                f"read-only MCP acknowledgement exceeded {wall_timeout_s:.3f}s"
            ),
        )
    succeeded, value = outcome[0]
    if not succeeded:
        if isinstance(value, BaseException):
            raise value
        raise RuntimeError("read-only MCP worker returned invalid failure evidence")
    if not isinstance(value, dict):
        raise TypeError("read-only MCP tool returned a non-object response")
    return value


def call_read_only_mcp_tool_with_retry(
    transport: SimulatorMcpTransport,
    name: str,
    arguments: JsonDict,
    *,
    timeout_s: float | None = None,
    first_attempt_timeout_s: float | None = None,
    max_attempts: int = DEFAULT_READ_ONLY_MCP_MAX_ATTEMPTS,
    health_timeout_s: float = DEFAULT_READ_ONLY_MCP_HEALTH_TIMEOUT_S,
) -> JsonDict:
    """Call a read-only MCP tool with one bounded, health-gated retry.

    An HTTP transport can lose the tool acknowledgement after a backend has
    already completed its read-only work.  Retrying inside the host keeps that
    transport incident out of the planner transcript and does not consume a
    second TUI/model turn.  This helper is intentionally not used for
    simulator mutation tools because their execution outcome may be unknown.
    """

    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError("max_attempts must be an integer")
    attempts = int(max_attempts)
    if attempts not in (1, 2):
        raise ValueError("max_attempts must be 1 or 2")
    health_timeout = float(health_timeout_s)
    if not math.isfinite(health_timeout) or health_timeout <= 0:
        raise ValueError("health_timeout_s must be finite and positive")
    first_timeout: float | None = None
    if first_attempt_timeout_s is not None:
        first_timeout = float(first_attempt_timeout_s)
        if not math.isfinite(first_timeout) or first_timeout <= 0:
            raise ValueError(
                "first_attempt_timeout_s must be finite and positive"
            )
    effective_first_timeout_s = (
        min(float(timeout_s), first_timeout)
        if first_timeout is not None and timeout_s is not None and timeout_s > 0
        else first_timeout
    )

    first_failure: SimulatorMcpTransportError | None = None
    first_failure_elapsed_s = 0.0
    started = time.monotonic()
    for attempt in range(1, attempts + 1):
        attempt_timeout_s = timeout_s
        try:
            if attempt == 1 and effective_first_timeout_s is not None:
                response = _call_mcp_tool_with_wall_timeout(
                    transport,
                    name,
                    arguments,
                    # Let the in-flight read-only work retain its complete
                    # logical deadline.  Only the caller's acknowledgement is
                    # bounded; the daemon attempt may still populate the
                    # binding cache for the retry.
                    transport_timeout_s=timeout_s,
                    wall_timeout_s=effective_first_timeout_s,
                )
            else:
                response = transport.call_tool(
                    name,
                    dict(arguments),
                    timeout_s=attempt_timeout_s,
                )
        except SimulatorMcpTransportError as exc:
            if (
                attempt >= attempts
                or exc.code not in READ_ONLY_MCP_TRANSIENT_ERROR_CODES
            ):
                raise
            first_failure = exc
            first_failure_elapsed_s = time.monotonic() - started
            # A retry is permitted only after a fresh MCP session proves the
            # server is reachable and still advertises the exact tool.
            listing = transport.list_tools(
                timeout_s=min(
                    health_timeout,
                    (
                        float(timeout_s)
                        if timeout_s is not None and timeout_s > 0
                        else health_timeout
                    ),
                )
            )
            raw_tools = listing.get("tools") if isinstance(listing, Mapping) else None
            advertised = {
                str(item.get("name") or "")
                for item in raw_tools or []
                if isinstance(item, Mapping)
            }
            if name not in advertised:
                raise SimulatorMcpTransportError(
                    f"health_check:{name}",
                    RuntimeError(f"read-only MCP service no longer advertises {name}"),
                ) from exc
            continue

        if first_failure is None:
            return response
        payload = dict(response)
        payload["_openeta_transport_retry"] = {
            "schema_version": READ_ONLY_MCP_RETRY_RECEIPT_SCHEMA_VERSION,
            "attempt_count": attempt,
            "retry_count": attempt - 1,
            "health_check": "passed",
            "tool": name,
            "first_failure_code": first_failure.code,
            "first_failure_type": first_failure.cause_type,
            "first_failure_elapsed_s": round(first_failure_elapsed_s, 6),
        }
        if first_timeout is not None:
            payload["_openeta_transport_retry"].update(
                {
                    "first_attempt_timeout_s": effective_first_timeout_s,
                    "retry_timeout_s": timeout_s,
                }
            )
        return payload

    raise RuntimeError("read-only MCP retry loop exhausted")  # pragma: no cover


SimulatorMcpResponseCallback = Callable[[str, JsonDict, JsonDict], None]


@dataclass(slots=True)
class SimulatorMcpToolProxyConfig:
    """Configuration shared by simulator MCP tool proxy handlers."""

    session_id: str = ""
    handle: str = ""
    timeout_s: float = 120.0
    tool_name_map: Mapping[str, str] = field(default_factory=dict)
    materialize_images: bool = True
    image_output_root: str | Path = DEFAULT_MCP_IMAGE_OUTPUT_ROOT
    image_bundle_id: str = ""
    materialize_text: bool = True
    text_output_root: str | Path = DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT
    response_output_root: str | Path = DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT
    max_inline_text_chars: int = DEFAULT_MAX_INLINE_TEXT_CHARS
    forward_grasp_candidate_orientation: bool = False
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(slots=True)
class SimulatorMcpEpisodeConfig:
    """Configuration for one MCP-backed simulator episode."""

    env_id: str
    render_mode: str = "rgb_array"
    seed: int = 0
    image_width: int | None = DEFAULT_SIMULATOR_IMAGE_WIDTH
    image_height: int | None = DEFAULT_SIMULATOR_IMAGE_HEIGHT
    session_id: str = ""
    artifact_session_id: str = ""
    handle: str = ""
    timeout_s: float = 120.0
    image_output_root: str | Path = DEFAULT_MCP_IMAGE_OUTPUT_ROOT
    startup_attempts: int = 2
    startup_retry_delay_s: float = 0.5


class SimulatorMcpEpisodeEnvironment:
    """EpisodeEnvironment backed by a remote simulator MCP server.

    Control tools are executed by ``SimulatorMcpToolProxy`` during
    ``OpenEtaAgentRuntime.act()``. The episode environment owns env lifecycle
    and turns post-tool feedback into the next ``EnvObservation``.
    """

    def __init__(
        self,
        *,
        transport: SimulatorMcpTransport,
        config: SimulatorMcpEpisodeConfig,
        tool_proxy_config: SimulatorMcpToolProxyConfig | None = None,
    ) -> None:
        self.transport = transport
        self.config = config
        self.tool_proxy_config = tool_proxy_config or SimulatorMcpToolProxyConfig(
            session_id=config.session_id,
            handle=config.handle,
            timeout_s=config.timeout_s,
            image_output_root=config.image_output_root,
        )
        self.task = ""
        self.create_result: JsonDict = {}
        self.last_payload: JsonDict = {}
        self.startup_attempt_count = 0
        self.execution_id = ""
        self.agent_session_id = ""
        self._close_lock = threading.Lock()
        self._artifact_sequence = 0
        self._artifact_instance_id = uuid4().hex[:10]

    def reset(self, *, task: str, metadata: JsonDict | None = None) -> EnvObservation:
        self.task = task
        if isinstance(metadata, dict):
            self.execution_id = str(metadata.get("execution_id") or "")
            self.agent_session_id = str(metadata.get("agent_session_id") or "")
            self.config.artifact_session_id = str(
                metadata.get("agent_session_id") or self.config.artifact_session_id or ""
            ).strip()
        owns_environment = not self.config.handle
        attempts = max(1, self.config.startup_attempts if owns_environment else 1)
        for attempt in range(1, attempts + 1):
            self.startup_attempt_count = attempt
            try:
                if not self.config.handle:
                    self._create_env(task)
                payload = self._reset_env()
                break
            except Exception as exc:  # noqa: BLE001 - transient MCP failures may be grouped.
                if attempt >= attempts or not _is_transient_startup_error(exc):
                    raise
                if self.config.handle:
                    close_simulator_mcp_env(
                        self.transport,
                        handle=self.config.handle,
                        session_id=self.config.session_id,
                        timeout_s=min(self.config.timeout_s, 30.0),
                    )
                self.config.handle = ""
                self.tool_proxy_config.handle = ""
                if self.config.startup_retry_delay_s > 0:
                    time.sleep(self.config.startup_retry_delay_s)
        else:  # pragma: no cover - loop either returns payload or raises.
            raise RuntimeError("simulator MCP startup attempts exhausted")
        return self._observation_from_payload(payload, metadata=metadata)

    def _create_env(self, task: str) -> None:
        create_args: JsonDict = {
            "env_id": self.config.env_id,
            "render_mode": self.config.render_mode,
            "seed": self.config.seed,
            "task": task,
        }
        if self.config.image_width is not None:
            create_args["image_width"] = self.config.image_width
        if self.config.image_height is not None:
            create_args["image_height"] = self.config.image_height
        if self.config.session_id:
            create_args["session_id"] = self.config.session_id
        self.create_result = self.transport.call_tool(
            "create_env", create_args, timeout_s=self.config.timeout_s
        )
        _raise_if_mcp_error(self.create_result, tool_name="create_env")
        self.config.session_id = str(self.create_result.get("session_id") or self.config.session_id)
        self.config.handle = str(self.create_result.get("handle") or "")
        if not self.config.handle:
            raise RuntimeError("create_env did not return a simulator handle")
        self._sync_tool_proxy_config()

    def _reset_env(self) -> JsonDict:
        reset_args: JsonDict = {"handle": self.config.handle, "seed": self.config.seed}
        if self.config.session_id:
            reset_args["session_id"] = self.config.session_id
        payload = self.transport.call_tool("reset_env", reset_args, timeout_s=self.config.timeout_s)
        _raise_if_mcp_error(payload, tool_name="reset_env")
        return payload

    def step(self, action: EnvAction) -> StepResult:
        render_args: JsonDict = {"handle": self.config.handle}
        if self.config.session_id:
            render_args["session_id"] = self.config.session_id
        attempts = max(1, self.config.startup_attempts)
        for attempt in range(1, attempts + 1):
            try:
                payload = self.transport.call_tool(
                    "render_env",
                    render_args,
                    timeout_s=self.config.timeout_s,
                )
                _raise_if_mcp_error(payload, tool_name="render_env")
                break
            except Exception as exc:  # noqa: BLE001 - transient MCP failures may be grouped.
                if attempt >= attempts or not _is_transient_startup_error(exc):
                    raise
                if self.config.startup_retry_delay_s > 0:
                    time.sleep(self.config.startup_retry_delay_s)
        observation = self._observation_from_payload(
            payload,
            metadata={
                "previous_action": _summarize_mcp_action(action),
                "source": type(self).__name__,
            },
        )
        remote_termination_reason = _latest_action_termination_reason(action)
        info = {
            "environment": type(self).__name__,
            "env_id": self.config.env_id,
            "session_id": self.config.session_id,
            "handle": self.config.handle,
            "previous_action": _summarize_mcp_action(action),
        }
        if remote_termination_reason:
            info.update(
                {
                    "termination_source": "simulator_mcp",
                    "termination_reason": remote_termination_reason,
                }
            )
        reward = _latest_action_reward(action, payload)
        terminated = _latest_action_flag(action, payload, "terminated") or bool(
            remote_termination_reason
        )
        truncated = _latest_action_flag(action, payload, "truncated")
        receipt = {
            "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
            "receipt_id": uuid4().hex,
            "backend": "simulator_mcp_episode_environment",
            "agent_tool": "environment_step",
            "remote_tool": "render_env",
            "execution_id": self.execution_id,
            "agent_session_id": self.agent_session_id,
            "simulator_session_id": self.config.session_id,
            "handle": self.config.handle,
            "timestamp_s": time.time(),
            "reward_present": ("reward" in payload or _latest_action_receipt_has_reward(action)),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "observation_fresh": True,
        }
        info.update(
            {
                "environment_receipt_trusted": True,
                "official_reward": receipt["reward_present"],
                "environment_receipt": receipt,
            }
        )
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> JsonDict:
        with self._close_lock:
            handle = self.config.handle
            session_id = self.config.session_id
            if not handle:
                return {"ok": True, "skipped": True}
            self.config.handle = ""
            self.tool_proxy_config.handle = ""
        return close_simulator_mcp_env(
            self.transport,
            handle=handle,
            session_id=session_id,
            timeout_s=min(self.config.timeout_s, 30.0),
        )

    def _sync_tool_proxy_config(self) -> None:
        self.tool_proxy_config.session_id = self.config.session_id
        self.tool_proxy_config.handle = self.config.handle
        self.tool_proxy_config.timeout_s = self.config.timeout_s
        self.tool_proxy_config.image_output_root = self.config.image_output_root
        if not self.tool_proxy_config.image_bundle_id:
            self.tool_proxy_config.image_bundle_id = (
                self.config.session_id or self.config.handle or self.config.env_id
            )

    def _observation_from_payload(
        self,
        payload: JsonDict,
        *,
        metadata: JsonDict | None = None,
    ) -> EnvObservation:
        bundle = materialize_mcp_images(
            payload,
            output_root=self.config.image_output_root,
            bundle_id=self._next_observation_bundle_id(),
            session_id=self.config.artifact_session_id,
        )
        scrubbed = _with_anygrasp_camera_intrinsics(bundle.payload)
        self.last_payload = scrubbed
        observation_payload = _extract_observation_payload(scrubbed)
        assigned_task = observation_payload.get("task") or observation_payload.get(
            "task_description"
        )
        if isinstance(assigned_task, str) and assigned_task.strip():
            self.task = assigned_task.strip()
        else:
            assigned_task = ""
        observation = EnvObservation.from_dict(observation_payload, task=self.task)
        merged_metadata: JsonDict = {
            **observation.metadata,
            "source": type(self).__name__,
            "env_id": self.config.env_id,
            "session_id": self.config.session_id,
            "handle": self.config.handle,
            "create_env": self.create_result,
            "startup_attempt_count": self.startup_attempt_count,
        }
        if assigned_task:
            merged_metadata["assigned_task"] = self.task
            merged_metadata["assigned_task_source"] = "simulator_observation"
        if bundle.images:
            merged_metadata["image_artifacts"] = [image.to_dict() for image in bundle.images]
        merged_metadata.update(dict(metadata or {}))
        observation.metadata = merged_metadata
        return observation

    def _next_observation_bundle_id(self) -> str:
        self._artifact_sequence += 1
        base = self.config.session_id or self.config.handle or self.config.env_id
        return f"{base}-{self._artifact_instance_id}-{self._artifact_sequence:04d}-observation"


class SimulatorMcpToolProxy:
    """Tool handler that forwards OpenETA AgentTools to simulator MCP tools."""

    def __init__(
        self,
        *,
        transport: SimulatorMcpTransport,
        config: SimulatorMcpToolProxyConfig | None = None,
    ) -> None:
        self.transport = transport
        self.config = config or SimulatorMcpToolProxyConfig()
        self._artifact_sequence = 0
        self._artifact_instance_id = uuid4().hex[:10]

    def handler_for(self, tool_name: str) -> ToolHandler:
        def handler(context: ToolExecutionContext) -> ToolResult:
            return self.call(context, tool_name=tool_name)

        return handler

    def call(self, context: ToolExecutionContext, *, tool_name: str | None = None) -> ToolResult:
        agent_tool = tool_name or context.name
        try:
            mcp_tool, arguments = self._mcp_call(context, agent_tool=agent_tool)
        except Exception as exc:  # noqa: BLE001 - validation must stay structured.
            return ToolResult(
                False,
                content=f"Simulator MCP proxy could not build arguments for {agent_tool}: {exc}",
                details=make_tool_result_details(
                    context.spec,
                    context.parameters,
                    success=False,
                    diagnostics=[
                        {
                            "code": "simulator_mcp_argument_error",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                ),
            )

        mcp_request = _mcp_request_descriptor(mcp_tool, arguments)

        try:
            raw_response = self.transport.call_tool(
                mcp_tool,
                arguments,
                timeout_s=self.config.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - tool failures must stay structured.
            transport_timeout = _is_transport_timeout(exc)
            transport_connection_lost = _is_transient_mcp_transport_error(exc)
            transport_unknown = context.spec.effect.value == "world_mutating" and (
                transport_timeout or transport_connection_lost
            )
            return ToolResult(
                False,
                content=f"Simulator MCP tool failed: {mcp_tool}: {exc}",
                details=make_tool_result_details(
                    context.spec,
                    context.parameters,
                    success=False,
                    outputs={
                        "mcp": {
                            "tool": mcp_tool,
                            "agent_tool": agent_tool,
                            "session_id": arguments.get("session_id", ""),
                            "handle": arguments.get("handle", ""),
                            "request": mcp_request,
                        },
                        "motion_outcome": "unknown" if transport_unknown else "failed",
                        "reconciliation_required": transport_unknown,
                    },
                    diagnostics=[
                        {
                            "code": (
                                "simulator_mcp_transport_timeout"
                                if transport_timeout
                                else "simulator_mcp_transport_connection_lost"
                                if transport_unknown
                                else "simulator_mcp_call_failed"
                            ),
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                ),
            )

        success = _response_success(raw_response)
        incomplete_motion_receipt = (
            success
            and agent_tool in {"move_to", "follow_eef_trajectory"}
            and _move_response_lacks_completion_receipt(raw_response)
        )
        if incomplete_motion_receipt:
            success = False
        normalized = self._normalize_response(
            raw_response,
            agent_tool=agent_tool,
            mcp_tool=mcp_tool,
            artifact_session_id=artifact_session_id(context.metadata),
            execution_metadata=context.metadata,
            mcp_request=mcp_request,
        )
        if agent_tool == "move_to" and _is_anyplace_pose(context.parameters):
            normalized["outputs"]["mcp"]["target_orientation_mode"] = "preserve_current"
        elif agent_tool == "move_to" and _is_ranked_grasp_candidate_pose(context.parameters):
            normalized["outputs"]["mcp"]["target_orientation_mode"] = (
                "graspnet_to_panda_eef"
                if self.config.forward_grasp_candidate_orientation
                else "preserve_current"
            )
        response_unknown = incomplete_motion_receipt or (
            not success
            and context.spec.effect.value == "world_mutating"
            and (
                _response_reports_unknown_action_outcome(raw_response)
                or _response_lost_action_receipt(raw_response)
            )
        )
        if response_unknown:
            normalized["outputs"].update(
                {
                    "motion_outcome": "unknown",
                    "reconciliation_required": True,
                }
            )
            diagnostics = [
                {
                    "code": (
                        "simulator_mcp_motion_receipt_incomplete"
                        if incomplete_motion_receipt
                        else "simulator_mcp_action_receipt_unavailable"
                    ),
                    "message": _brief_response_error(raw_response),
                    "candidate_rejection": False,
                    "failure_class": "action_outcome_unknown",
                }
            ]
        else:
            diagnostics = _response_diagnostics(raw_response) if not success else []
        return ToolResult(
            success,
            content=_response_content(
                normalized["outputs"]["response"],
                mcp_tool=mcp_tool,
                success=success,
            ),
            details=make_tool_result_details(
                context.spec,
                context.parameters,
                success=success,
                outputs=normalized["outputs"],
                artifacts=normalized["artifacts"],
                state_delta=normalized["state_delta"],
                environment_receipt=normalized["environment_receipt"],
                diagnostics=diagnostics,
            ),
        )

    def _mcp_call(
        self,
        context: ToolExecutionContext,
        *,
        agent_tool: str,
    ) -> tuple[str, JsonDict]:
        if agent_tool == "gripper_control":
            binary_position = self._binary_gripper_position(context.parameters)
            if agent_tool in self.config.tool_name_map:
                return self.config.tool_name_map[agent_tool], self._with_session(
                    {"position": binary_position}
                )
            return self._gripper_tool_name(binary_position), self._with_session({})
        if agent_tool in self.config.tool_name_map:
            return self.config.tool_name_map[agent_tool], self._with_session(
                dict(context.parameters)
            )
        if agent_tool == "observe":
            return self._mcp_tool_name(agent_tool), self._with_session({})
        if agent_tool == "move_to":
            return self._mcp_tool_name(agent_tool), self._move_to_arguments(context.parameters)
        if agent_tool == "follow_eef_trajectory":
            return self._mcp_tool_name(agent_tool), self._with_session(dict(context.parameters))
        return self._mcp_tool_name(agent_tool), self._with_session(dict(context.parameters))

    def _mcp_tool_name(self, agent_tool: str) -> str:
        if agent_tool in self.config.tool_name_map:
            return self.config.tool_name_map[agent_tool]
        return DEFAULT_SIMULATOR_MCP_TOOL_MAP.get(agent_tool, agent_tool)

    def _with_session(self, arguments: JsonDict) -> JsonDict:
        if self.config.handle:
            arguments.setdefault("handle", self.config.handle)
        if self.config.session_id:
            arguments.setdefault("session_id", self.config.session_id)
        if not arguments.get("handle"):
            raise ValueError(
                "No active simulator MCP environment handle is bound. "
                "Create/reset a simulator environment before calling control tools."
            )
        return arguments

    def _move_to_arguments(self, parameters: JsonDict) -> JsonDict:
        x, y, z = _extract_xyz(parameters, tool_name="move_to")
        arguments: JsonDict = {"x": x, "y": y, "z": z}
        # ``move_to`` is encoded for the generic simulator transport as
        # x/y/z plus an orientation.  Keep the remaining, non-kinematic pose
        # identity in a separate envelope so a backend can bind the exact
        # host-compiled terminal to its qualification proof. The transport
        # owns x/y/z/quaternion and never lets this envelope alter motion.
        pose = (
            parameters.get("target_pose")
            or parameters.get("pose")
            or parameters.get("eef_pose")
        )
        if isinstance(pose, dict):
            provenance = {
                key: value
                for key, value in pose.items()
                if key
                not in {
                    "x",
                    "y",
                    "z",
                    "xyz",
                    "position",
                    "translation_xyz",
                    "quat_xyzw",
                    "euler_xyz_deg",
                }
            }
            if provenance:
                arguments["motion_provenance"] = provenance
        if "speed" in parameters:
            raise ValueError(
                "move_to `speed` is unsupported by the simulator MCP; "
                "use num_steps/tolerance or omit speed."
            )
        is_anyplace_pose = _is_anyplace_pose(parameters)
        is_grasp_candidate = _is_ranked_grasp_candidate_pose(parameters)
        if is_anyplace_pose:
            raise ValueError(
                "Raw AnyPlace poses are not executable; use only EEF poses from the "
                "host-owned qualified-candidate compilation event."
            )
        elif is_grasp_candidate and self.config.forward_grasp_candidate_orientation:
            arguments.update(
                _extract_graspnet_panda_orientation_arguments(
                    parameters,
                    tool_name="move_to",
                )
            )
        elif not is_grasp_candidate:
            arguments.update(_extract_orientation_arguments(parameters, tool_name="move_to"))
        for key in ("handle", "session_id"):
            if key in parameters:
                arguments[key] = parameters[key]
        for key in ("num_steps", "tolerance", "ori_tolerance", "enable_collision_check",
                    "velocity_scaling", "acceleration_scaling"):
            if key in parameters:
                arguments[key] = parameters[key]
        return self._with_session(arguments)

    def _binary_gripper_position(self, parameters: JsonDict) -> int:
        position = parameters.get("position")
        if position is None:
            position = parameters.get("open")
        if position is None:
            raise ValueError("gripper_control requires `position` or `open`.")
        # The actuator contract is intentionally type-strict. JSON `true`,
        # `false`, `0.0`, and `1.0` are not integer commands even though
        # Python normally compares them equal to 0/1.
        if type(position) is not int or position not in (0, 1):
            raise ValueError("gripper_control position must be exactly 0 or 1.")
        return position

    def _gripper_tool_name(self, binary_position: int) -> str:
        return "gripper_open" if binary_position == 1 else "gripper_close"

    def _normalize_response(
        self,
        response: JsonDict,
        *,
        agent_tool: str,
        mcp_tool: str,
        artifact_session_id: str = "",
        execution_metadata: JsonDict | None = None,
        mcp_request: JsonDict | None = None,
        receipt_session_id: str | None = None,
        receipt_handle: str | None = None,
    ) -> JsonDict:
        request = dict(mcp_request or _mcp_request_descriptor(mcp_tool, {}))
        request_id = str(request.get("request_id") or "").strip()
        if not request_id:
            raise ValueError("MCP request descriptor requires a request_id.")
        session_id = (
            self.config.session_id if receipt_session_id is None else receipt_session_id
        )
        handle = self.config.handle if receipt_handle is None else receipt_handle
        payload = dict(response)
        bundle_id = self._next_artifact_bundle_id(mcp_tool)
        artifacts: list[JsonDict] = []
        if self.config.materialize_images:
            bundle = materialize_mcp_images(
                payload,
                output_root=self.config.image_output_root,
                bundle_id=bundle_id,
                session_id=artifact_session_id,
            )
            payload = bundle.payload
            artifacts = [image.to_dict() for image in bundle.images]
        payload = _with_anygrasp_camera_intrinsics(payload)
        observation_snapshot = build_observation_snapshot(
            payload,
            image_artifacts=artifacts,
        )
        environment_receipt = _build_environment_receipt(
            payload,
            observation_snapshot=observation_snapshot,
            agent_tool=agent_tool,
            mcp_tool=mcp_tool,
            simulator_session_id=session_id,
            handle=handle,
            execution_metadata=execution_metadata,
            mcp_request_id=request_id,
        )
        response_artifact = materialize_json_response(
            payload,
            output_root=self.config.response_output_root,
            bundle_id=bundle_id,
            name=f"{mcp_tool}-response",
            session_id=artifact_session_id,
        )
        response_ref = build_response_reference(
            payload,
            response_artifact,
            image_artifacts=artifacts,
        )
        response_evidence: JsonDict = {
            **response_ref,
            "request_id": request_id,
            "tool": mcp_tool,
            "session_id": session_id,
            "handle": handle,
        }
        artifacts.append(response_artifact.to_dict())

        outputs: JsonDict = {
            "mcp": {
                "tool": mcp_tool,
                "agent_tool": agent_tool,
                "session_id": session_id,
                "handle": handle,
                "request": request,
                "response": response_evidence,
            },
            "response": response_evidence,
            "mcp_calls": [
                {
                    "request": request,
                    "response": response_evidence,
                    "environment_receipt": environment_receipt,
                }
            ],
        }
        for key in ("observation_summary", "motion_summary"):
            summary = response_ref.get(key)
            if isinstance(summary, dict):
                outputs[key] = summary
        return {
            "outputs": outputs,
            "artifacts": artifacts,
            "state_delta": _state_delta_from_response(payload),
            "environment_receipt": environment_receipt,
        }

    def _next_artifact_bundle_id(self, mcp_tool: str) -> str:
        self._artifact_sequence += 1
        base = self.config.image_bundle_id or self.config.session_id or "simulator-mcp"
        return f"{base}-{self._artifact_instance_id}-{self._artifact_sequence:04d}-{mcp_tool}"


class SimulatorEnvironmentCreator:
    """Create and reset one simulator environment through a stable AgentTool."""

    def __init__(
        self,
        *,
        transport: SimulatorMcpTransport,
        config: SimulatorMcpToolProxyConfig | None = None,
        response_callback: SimulatorMcpResponseCallback | None = None,
    ) -> None:
        self.transport = transport
        self.config = config or SimulatorMcpToolProxyConfig()
        self.response_callback = response_callback
        self.proxy = SimulatorMcpToolProxy(transport=transport, config=self.config)

    def handler(self, context: ToolExecutionContext) -> ToolResult:
        startup_retry_attempt = int(
            context.metadata.get(SIMULATOR_STARTUP_RETRY_METADATA_KEY, 0) or 0
        )
        env_id = str(context.parameters.get("env_id") or "").strip()
        if not env_id:
            return self._failure(
                context,
                content="create_simulator_env requires a non-empty env_id.",
                diagnostics=[{"code": "missing_env_id"}],
            )
        with self.config.lifecycle_lock:
            active_handle = self.config.handle
        if active_handle:
            return self._failure(
                context,
                content=(
                    "A simulator environment is already active. Call "
                    "close_simulator_env before creating another one."
                ),
                diagnostics=[
                    {
                        "code": "simulator_environment_already_active",
                        "handle": active_handle,
                    }
                ],
            )

        try:
            create_args = self._create_arguments(context, env_id=env_id)
        except (TypeError, ValueError) as exc:
            return self._failure(
                context,
                content=f"create_simulator_env parameters are invalid: {exc}",
                diagnostics=[
                    {
                        "code": "simulator_mcp_argument_error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        create_request = _mcp_request_descriptor("create_env", create_args)
        try:
            create_response = self.transport.call_tool(
                "create_env",
                create_args,
                timeout_s=self.config.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - transport errors stay structured.
            return self._transport_failure(context, "create_env", exc)

        if _context_execution_cancelled(context):
            abandoned_handle = str(create_response.get("handle") or "").strip()
            abandoned_session_id = str(
                create_response.get("session_id") or create_args.get("session_id") or ""
            ).strip()
            if abandoned_handle:
                self._close_abandoned_environment(
                    handle=abandoned_handle,
                    session_id=abandoned_session_id,
                )
            return self._failure(
                context,
                content="Simulator environment creation was cancelled and cleaned up.",
                diagnostics=[{"code": "execution_cancelled", "abandoned": True}],
            )

        create_normalized = self.proxy._normalize_response(  # noqa: SLF001
            create_response,
            agent_tool="create_simulator_env",
            mcp_tool="create_env",
            artifact_session_id=artifact_session_id(context.metadata),
            execution_metadata=context.metadata,
            mcp_request=create_request,
            receipt_session_id=str(
                create_response.get("session_id") or create_args.get("session_id") or ""
            ).strip(),
            receipt_handle=str(create_response.get("handle") or "").strip(),
        )
        create_ref = create_normalized["outputs"]["response"]
        self._notify("create_env", create_args, create_ref)
        if not _response_success(create_response):
            return self._failure(
                context,
                content=_response_content(create_ref, mcp_tool="create_env", success=False),
                outputs={
                    "mcp": create_normalized["outputs"]["mcp"],
                    "create_response": create_ref,
                },
                artifacts=create_normalized["artifacts"],
                diagnostics=_response_diagnostics(create_response),
            )

        handle = str(create_response.get("handle") or "").strip()
        session_id = str(
            create_response.get("session_id") or create_args.get("session_id") or ""
        ).strip()
        if not handle:
            return self._failure(
                context,
                content="Simulator create_env succeeded without returning a handle.",
                outputs={"create_response": create_ref},
                artifacts=create_normalized["artifacts"],
                diagnostics=[{"code": "create_env_missing_handle"}],
            )
        if _context_execution_cancelled(context):
            self._close_abandoned_environment(handle=handle, session_id=session_id)
            return self._failure(
                context,
                content="Simulator environment creation was cancelled and cleaned up.",
                diagnostics=[{"code": "execution_cancelled", "abandoned": True}],
            )

        with self.config.lifecycle_lock:
            self.config.handle = handle
            self.config.session_id = session_id
            self.config.image_bundle_id = session_id or handle
        reset_args: JsonDict = {"handle": handle, "seed": create_args["seed"]}
        if session_id:
            reset_args["session_id"] = session_id
        reset_request = _mcp_request_descriptor("reset_env", reset_args)
        try:
            reset_response = self.transport.call_tool(
                "reset_env",
                reset_args,
                timeout_s=self.config.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - transport errors stay structured.
            return self._transport_failure(
                context,
                "reset_env",
                exc,
                outputs={"create_response": create_ref},
                artifacts=create_normalized["artifacts"],
            )
        if _context_execution_cancelled(context):
            with self.config.lifecycle_lock:
                owns_handle = self.config.handle == handle
                if owns_handle:
                    self.config.handle = ""
            if owns_handle:
                self._close_abandoned_environment(handle=handle, session_id=session_id)
            return self._failure(
                context,
                content="Simulator environment reset was cancelled and cleaned up.",
                diagnostics=[{"code": "execution_cancelled", "abandoned": True}],
            )

        reset_normalized = self.proxy._normalize_response(  # noqa: SLF001
            reset_response,
            agent_tool="create_simulator_env",
            mcp_tool="reset_env",
            artifact_session_id=artifact_session_id(context.metadata),
            execution_metadata=context.metadata,
            mcp_request=reset_request,
        )
        reset_ref = reset_normalized["outputs"]["response"]
        self._notify("reset_env", reset_args, reset_ref)
        success = _response_success(reset_response)
        reset_failure_cleanup: JsonDict | None = None
        if not success:
            # A failed reset is terminal for this freshly created handle. Roll
            # it back inside the host tool so the planner does not spend extra
            # turns trying observe/create against a half-started environment.
            with self.config.lifecycle_lock:
                owns_handle = self.config.handle == handle
                if owns_handle:
                    self.config.handle = ""
            if owns_handle:
                cleanup_response = self._close_abandoned_environment(
                    handle=handle,
                    session_id=session_id,
                )
                reset_failure_cleanup = {
                    "attempted": True,
                    "success": _response_success(cleanup_response),
                    "handle": handle,
                    "response": cleanup_response,
                }
        assigned_task = _response_assigned_task(reset_ref)
        server_url = mcp_server_url_from_transport(self.transport)
        dashboard_url = mcp_dashboard_url(server_url, session_id)
        environment: JsonDict = {
            "env_id": env_id,
            "handle": handle,
            "session_id": session_id,
        }
        display_name = str(create_response.get("name") or "").strip()
        if display_name:
            environment["display_name"] = display_name
        control_spec = create_response.get("control_spec")
        if isinstance(control_spec, dict) and control_spec:
            environment["control_spec"] = dict(control_spec)
        if assigned_task:
            environment["assigned_task"] = assigned_task
        if server_url:
            environment["mcp_server_url"] = server_url
        if dashboard_url:
            environment["dashboard_url"] = dashboard_url
        outputs: JsonDict = {
            "mcp": {
                "tool": "create_env",
                "auto_reset_tool": "reset_env",
                "handle": handle,
                "session_id": session_id,
                "request": create_normalized["outputs"]["mcp"]["request"],
                "response": create_normalized["outputs"]["mcp"]["response"],
            },
            "mcp_calls": [
                *create_normalized["outputs"]["mcp_calls"],
                *reset_normalized["outputs"]["mcp_calls"],
            ],
            "environment": environment,
            "create_response": create_ref,
            "initial_observation": reset_ref,
        }
        if reset_failure_cleanup is not None:
            outputs["reset_failure_cleanup"] = reset_failure_cleanup
        if assigned_task:
            outputs["assigned_task"] = assigned_task
        for key in ("observation_summary",):
            summary = reset_normalized["outputs"].get(key)
            if isinstance(summary, dict):
                outputs[key] = summary
        result = ToolResult(
            success,
            content=(
                (
                    (f"Simulator environment created and reset. Assigned task: {assigned_task}")
                    if assigned_task
                    else "Simulator environment created and reset."
                )
                if success
                else _response_content(reset_ref, mcp_tool="reset_env", success=False)
            ),
            details=make_tool_result_details(
                context.spec,
                context.parameters,
                success=success,
                outputs=outputs,
                artifacts=[
                    *create_normalized["artifacts"],
                    *reset_normalized["artifacts"],
                ],
                state_delta={
                    **reset_normalized["state_delta"],
                    "simulator_environment": (
                        {
                            "handle": "",
                            "session_id": "",
                            "status": "closed_after_reset_failure",
                        }
                        if reset_failure_cleanup
                        and reset_failure_cleanup["success"] is True
                        else environment
                    ),
                },
                environment_receipt={
                    **reset_normalized["environment_receipt"],
                    "simulator_session_id": session_id,
                    "handle": handle,
                    "environment_closed": bool(
                        reset_failure_cleanup
                        and reset_failure_cleanup["success"] is True
                    ),
                },
                diagnostics=(
                    []
                    if success
                    else [
                        *_response_diagnostics(reset_response),
                        {
                            "code": "simulator_reset_failure_cleanup",
                            "attempted": reset_failure_cleanup is not None,
                            "success": bool(
                                reset_failure_cleanup
                                and reset_failure_cleanup["success"] is True
                            ),
                        },
                    ]
                ),
            ),
        )
        retryable_detach_readiness_failure = bool(
            not success
            and startup_retry_attempt == 0
            and reset_failure_cleanup
            and reset_failure_cleanup.get("success") is True
            and "NATIVE_GRASP_DETACH_ACK_MISSING"
            in json.dumps(reset_response, sort_keys=True)
        )
        if not retryable_detach_readiness_failure:
            return result

        # The failed handle has been positively closed, so a second isolated
        # create/reset is a safe host-level infrastructure retry.  Keep it in
        # this single planner tool call: another VLM turn cannot improve a
        # transient stock DetachableJoint readiness race and only inflates
        # latency/context.  The metadata marker bounds recursion to one retry.
        previous_marker = context.metadata.get(SIMULATOR_STARTUP_RETRY_METADATA_KEY)
        context.metadata[SIMULATOR_STARTUP_RETRY_METADATA_KEY] = 1
        try:
            retried = self.handler(context)
        finally:
            if previous_marker is None:
                context.metadata.pop(SIMULATOR_STARTUP_RETRY_METADATA_KEY, None)
            else:
                context.metadata[SIMULATOR_STARTUP_RETRY_METADATA_KEY] = (
                    previous_marker
                )

        retry_outputs = retried.details.setdefault("outputs", {})
        second_calls = list(retry_outputs.get("mcp_calls") or [])
        first_calls = [
            *create_normalized["outputs"]["mcp_calls"],
            *reset_normalized["outputs"]["mcp_calls"],
        ]
        retry_outputs["mcp_calls"] = [*first_calls, *second_calls]
        retry_outputs["startup_retry"] = {
            "schema_version": "openeta.simulator_startup_retry.v1",
            "attempt_count": 2,
            "retry_count": 1,
            "reason": "NATIVE_GRASP_DETACH_ACK_MISSING",
            "first_environment_closed": True,
            "first_create_response": create_ref,
            "first_reset_response": reset_ref,
            "first_cleanup": reset_failure_cleanup,
            "final_success": retried.success,
        }
        retry_artifacts = retried.details.setdefault("artifacts", [])
        retried.details["artifacts"] = [
            *create_normalized["artifacts"],
            *reset_normalized["artifacts"],
            *retry_artifacts,
        ]
        retry_receipt = retried.details.get("environment_receipt")
        if isinstance(retry_receipt, dict):
            retry_receipt.update(
                {
                    "startup_attempt_count": 2,
                    "startup_retry_count": 1,
                    "startup_retry_reason": (
                        "NATIVE_GRASP_DETACH_ACK_MISSING"
                    ),
                }
            )
        return retried

    def _create_arguments(
        self,
        context: ToolExecutionContext,
        *,
        env_id: str,
    ) -> JsonDict:
        parameters = context.parameters
        args: JsonDict = {
            "env_id": env_id,
            "render_mode": str(parameters.get("render_mode") or "rgb_array"),
            "seed": _required_integer(parameters.get("seed", 0), name="seed"),
            "image_width": _positive_integer(
                parameters.get("image_width") or DEFAULT_SIMULATOR_IMAGE_WIDTH,
                name="image_width",
            ),
            "image_height": _positive_integer(
                parameters.get("image_height") or DEFAULT_SIMULATOR_IMAGE_HEIGHT,
                name="image_height",
            ),
        }
        task = parameters.get("task")
        if isinstance(task, str) and task:
            args["task"] = task
        session_id = parameters.get("session_id") or self.config.session_id
        if isinstance(session_id, str) and session_id:
            args["session_id"] = session_id
        if "include_objects" in parameters:
            include_objects = parameters["include_objects"]
            if not isinstance(include_objects, bool):
                raise TypeError("include_objects must be a boolean")
            args["include_objects"] = include_objects
        return args

    def _notify(self, name: str, arguments: JsonDict, response: JsonDict) -> None:
        if self.response_callback is not None:
            self.response_callback(name, arguments, response)

    def _close_abandoned_environment(
        self, *, handle: str, session_id: str
    ) -> JsonDict:
        result = close_simulator_mcp_env(
            self.transport,
            handle=handle,
            session_id=session_id,
            timeout_s=min(self.config.timeout_s, 30.0),
        )
        if _response_success(result):
            with self.config.lifecycle_lock:
                if not self.config.handle:
                    self.config.session_id = ""
                    self.config.image_bundle_id = ""
            return result
        with self.config.lifecycle_lock:
            if not self.config.handle:
                self.config.handle = handle
                self.config.session_id = session_id
                self.config.image_bundle_id = session_id or handle
        return result

    def _transport_failure(
        self,
        context: ToolExecutionContext,
        mcp_tool: str,
        exc: Exception,
        *,
        outputs: JsonDict | None = None,
        artifacts: list[JsonDict] | None = None,
    ) -> ToolResult:
        return self._failure(
            context,
            content=f"Simulator MCP tool failed: {mcp_tool}: {exc}",
            outputs=outputs,
            artifacts=artifacts,
            diagnostics=[
                {
                    "code": "simulator_mcp_call_failed",
                    "tool": mcp_tool,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            ],
        )

    @staticmethod
    def _failure(
        context: ToolExecutionContext,
        *,
        content: str,
        outputs: JsonDict | None = None,
        artifacts: list[JsonDict] | None = None,
        diagnostics: list[JsonDict] | None = None,
    ) -> ToolResult:
        return ToolResult(
            False,
            content=content,
            details=make_tool_result_details(
                context.spec,
                context.parameters,
                success=False,
                outputs=outputs,
                artifacts=artifacts,
                diagnostics=diagnostics,
            ),
        )


class SimulatorEnvironmentCloser:
    """Close the one active simulator environment through a stable AgentTool."""

    def __init__(
        self,
        *,
        transport: SimulatorMcpTransport,
        config: SimulatorMcpToolProxyConfig,
        response_callback: SimulatorMcpResponseCallback | None = None,
    ) -> None:
        self.transport = transport
        self.config = config
        self.response_callback = response_callback
        self.proxy = SimulatorMcpToolProxy(transport=transport, config=config)

    def handler(self, context: ToolExecutionContext) -> ToolResult:
        with self.config.lifecycle_lock:
            handle = self.config.handle
            session_id = self.config.session_id
            if handle:
                self.config.handle = ""
        if not handle:
            return make_tool_result(
                context,
                success=True,
                content="No active simulator environment to close.",
                outputs={"closed": False, "skipped": True},
                environment_receipt={
                    "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
                    "receipt_id": uuid4().hex,
                    "backend": "simulator_mcp",
                    "agent_tool": "close_simulator_env",
                    "remote_tool": "close_env",
                    "simulator_session_id": session_id,
                    "handle": "",
                    "timestamp_s": time.time(),
                    "reward_present": False,
                    "observation_fresh": False,
                    "environment_closed": True,
                },
            )
        arguments: JsonDict = {"handle": handle}
        if session_id:
            arguments["session_id"] = session_id
        mcp_request = _mcp_request_descriptor("close_env", arguments)
        try:
            response = self.transport.call_tool(
                "close_env",
                arguments,
                timeout_s=min(self.config.timeout_s, 30.0),
            )
        except Exception as exc:  # noqa: BLE001 - lifecycle failures stay structured.
            with self.config.lifecycle_lock:
                if not self.config.handle:
                    self.config.handle = handle
            return make_tool_result(
                context,
                success=False,
                content=f"Simulator MCP tool failed: close_env: {exc}",
                outputs={"handle": handle, "session_id": session_id},
                diagnostics=[
                    {
                        "code": "simulator_mcp_call_failed",
                        "tool": "close_env",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        success = _response_success(response)
        if not success:
            with self.config.lifecycle_lock:
                if not self.config.handle:
                    self.config.handle = handle
        elif self.response_callback is not None:
            self.response_callback("close_env", arguments, response)
        normalized = self.proxy._normalize_response(  # noqa: SLF001
            response,
            agent_tool="close_simulator_env",
            mcp_tool="close_env",
            artifact_session_id=artifact_session_id(context.metadata),
            execution_metadata=context.metadata,
            mcp_request=mcp_request,
            receipt_session_id=session_id,
            receipt_handle=handle,
        )
        return make_tool_result(
            context,
            success=success,
            content=(
                "Simulator environment closed."
                if success
                else _response_content(response, mcp_tool="close_env", success=False)
            ),
            outputs={
                **normalized["outputs"],
                "closed": success,
                "environment": {"handle": handle, "session_id": session_id},
            },
            artifacts=normalized["artifacts"],
            state_delta={
                "simulator_environment": {
                    "handle": handle,
                    "session_id": session_id,
                    "status": "closed" if success else "close_failed",
                }
            },
            environment_receipt={
                **normalized["environment_receipt"],
                "environment_closed": success,
            },
            diagnostics=[] if success else _response_diagnostics(response),
        )


def _with_anygrasp_camera_intrinsics(payload: JsonDict) -> JsonDict:
    """Add metric depth scale to generic and legacy camera intrinsics."""

    enriched = json.loads(json.dumps(payload))
    _enrich_anygrasp_camera_intrinsics(enriched)
    return enriched if isinstance(enriched, dict) else dict(payload)


def _enrich_anygrasp_camera_intrinsics(value: Any) -> None:
    if isinstance(value, dict):
        _enrich_camera_dict(value)
        for item in value.values():
            _enrich_anygrasp_camera_intrinsics(item)
    elif isinstance(value, list):
        for item in value:
            _enrich_anygrasp_camera_intrinsics(item)


def _enrich_camera_dict(camera: JsonDict) -> None:
    intrinsics = camera.get("intrinsics")
    if not isinstance(intrinsics, dict):
        return
    has_camera_payload = any(
        isinstance(camera.get(key), str) and camera.get(key)
        for key in ("rgb_path", "depth_path", "image_path", "rgb_ref", "depth_ref")
    )
    if not has_camera_payload:
        return
    normalized_intrinsics = dict(intrinsics)
    scale = _camera_depth_scale(camera, intrinsics)
    if scale is not None:
        normalized_intrinsics["scale"] = scale
    camera["intrinsics"] = normalized_intrinsics
    camera.setdefault("anygrasp_intrinsics", dict(normalized_intrinsics))


def _camera_depth_scale(camera: JsonDict, intrinsics: JsonDict) -> float | None:
    for key in ("scale", "depth_scale"):
        parsed = _positive_float(intrinsics.get(key))
        if parsed is not None:
            return parsed
    for key in ("depth_scale", "scale"):
        parsed = _positive_float(camera.get(key))
        if parsed is not None:
            return parsed
    depth_path = camera.get("depth_path")
    if isinstance(depth_path, str) and depth_path.lower().endswith(".png"):
        return 1000.0
    return None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _required_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = _required_integer(value, name=name)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


@dataclass(slots=True)
class StdioSimulatorMcpTransport:
    """Synchronous stdio MCP transport for local simulator-server launches."""

    command: str
    args: Sequence[str] = ()
    cwd: str | Path | None = None

    def list_tools(self, *, timeout_s: float | None = None) -> JsonDict:
        return asyncio.run(
            _list_stdio_mcp_tools(
                command=self.command,
                args=list(self.args),
                cwd=str(self.cwd) if self.cwd is not None else None,
                timeout_s=timeout_s,
            )
        )

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        return asyncio.run(
            _call_stdio_mcp_tool(
                command=self.command,
                args=list(self.args),
                cwd=str(self.cwd) if self.cwd is not None else None,
                tool_name=name,
                arguments=arguments,
                timeout_s=timeout_s,
            )
        )


@dataclass(slots=True)
class SseSimulatorMcpTransport:
    """Synchronous SSE MCP transport for an already-running simulator server."""

    url: str = "http://localhost:8765/sse"

    def list_tools(self, *, timeout_s: float | None = None) -> JsonDict:
        try:
            with _temporary_no_proxy_for_url(self.url):
                return asyncio.run(
                    _with_optional_timeout(
                        _list_sse_mcp_tools(
                            url=self.url,
                            timeout_s=timeout_s,
                        ),
                        timeout_s=timeout_s,
                    )
                )
        except SimulatorMcpTransportError:
            raise
        except Exception as exc:
            raise SimulatorMcpTransportError("list_tools", exc) from exc

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        try:
            with _temporary_no_proxy_for_url(self.url):
                return asyncio.run(
                    _with_optional_timeout(
                        _call_sse_mcp_tool(
                            url=self.url,
                            tool_name=name,
                            arguments=arguments,
                            timeout_s=timeout_s,
                        ),
                        timeout_s=timeout_s,
                    )
                )
        except SimulatorMcpTransportError:
            raise
        except Exception as exc:
            raise SimulatorMcpTransportError(f"call_tool:{name}", exc) from exc


@dataclass(slots=True)
class StreamableHttpSimulatorMcpTransport:
    """Synchronous Streamable HTTP transport for a simulator MCP server."""

    url: str = "http://localhost:8765/mcp"

    def list_tools(self, *, timeout_s: float | None = None) -> JsonDict:
        try:
            with _temporary_no_proxy_for_url(self.url):
                return asyncio.run(
                    _with_optional_timeout(
                        _list_streamable_http_mcp_tools(
                            url=self.url,
                            timeout_s=timeout_s,
                        ),
                        timeout_s=timeout_s,
                    )
                )
        except SimulatorMcpTransportError:
            raise
        except Exception as exc:
            raise SimulatorMcpTransportError("list_tools", exc) from exc

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        try:
            with _temporary_no_proxy_for_url(self.url):
                return asyncio.run(
                    _with_optional_timeout(
                        _call_streamable_http_mcp_tool(
                            url=self.url,
                            tool_name=name,
                            arguments=arguments,
                            timeout_s=timeout_s,
                        ),
                        timeout_s=timeout_s,
                    )
                )
        except SimulatorMcpTransportError:
            raise
        except Exception as exc:
            raise SimulatorMcpTransportError(f"call_tool:{name}", exc) from exc


def simulator_mcp_transport_for_url(url: str) -> SimulatorMcpTransport:
    """Select the standard transport encoded by one MCP endpoint URL."""

    endpoint_path = urlparse(str(url or "")).path.rstrip("/")
    if endpoint_path.endswith("/mcp"):
        return StreamableHttpSimulatorMcpTransport(url)
    return SseSimulatorMcpTransport(url)


def bind_simulator_mcp_tool_handlers(
    tools: ToolRegistry,
    *,
    transport: SimulatorMcpTransport,
    config: SimulatorMcpToolProxyConfig | None = None,
    tool_names: Sequence[str] = DEFAULT_SIMULATOR_MCP_TOOL_NAMES,
    response_callback: SimulatorMcpResponseCallback | None = None,
    replace: bool = False,
) -> ToolRegistry:
    """Bind simulator-owned AgentTools to MCP proxy handlers."""

    shared_config = config or SimulatorMcpToolProxyConfig()
    proxy = SimulatorMcpToolProxy(transport=transport, config=shared_config)
    creator = SimulatorEnvironmentCreator(
        transport=transport,
        config=shared_config,
        response_callback=response_callback,
    )
    closer = SimulatorEnvironmentCloser(
        transport=transport,
        config=shared_config,
        response_callback=response_callback,
    )
    for name in tool_names:
        tools.get(name)
        if tools.can_execute(name) and not replace:
            continue
        if name == "create_simulator_env":
            handler = creator.handler
        elif name == "close_simulator_env":
            handler = closer.handler
        else:
            handler = proxy.handler_for(name)
        tools.bind_handler(
            name,
            handler,
            replace=replace,
            authority=ENVIRONMENT_AUTHORITY,
        )
    return tools


def close_environment_mcp_env(
    transport: SimulatorMcpTransport,
    *,
    handle: str,
    session_id: str = "",
    timeout_s: float | None = 30.0,
) -> JsonDict:
    """Best-effort cleanup for a remote MCP-managed environment.

    Any code path that creates an MCP env for tests or smoke runs must call
    ``close_env`` in a ``finally`` block. This helper keeps cleanup failures
    structured so the original test failure is not masked by a secondary close
    exception.
    """

    arguments: JsonDict = {"handle": handle}
    if session_id:
        arguments["session_id"] = session_id
    try:
        result = transport.call_tool("close_env", arguments, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - cleanup must be best-effort.
        return {
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "handle": handle,
            "session_id": session_id,
        }
    if not isinstance(result, dict):
        return {
            "ok": False,
            "error": f"close_env returned {type(result).__name__}",
            "handle": handle,
            "session_id": session_id,
        }
    return result


def close_simulator_mcp_env(
    transport: SimulatorMcpTransport,
    *,
    handle: str,
    session_id: str = "",
    timeout_s: float | None = 30.0,
) -> JsonDict:
    """Backward-compatible simulator-specific name for MCP environment cleanup."""

    return close_environment_mcp_env(
        transport,
        handle=handle,
        session_id=session_id,
        timeout_s=timeout_s,
    )


async def _call_stdio_mcp_tool(
    *,
    command: str,
    args: list[str],
    cwd: str | None,
    tool_name: str,
    arguments: JsonDict,
    timeout_s: float | None,
) -> JsonDict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args, cwd=cwd)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=_timeout_delta(timeout_s),
            )
    payload = _parse_mcp_tool_result(result)
    return payload


async def _list_stdio_mcp_tools(
    *,
    command: str,
    args: list[str],
    cwd: str | None,
    timeout_s: float | None,
) -> JsonDict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args, cwd=cwd)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
    return _parse_mcp_tools_result(result)


async def _call_sse_mcp_tool(
    *,
    url: str,
    tool_name: str,
    arguments: JsonDict,
    timeout_s: float | None,
) -> JsonDict:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(
        url,
        sse_read_timeout=_sse_read_timeout_s(timeout_s),
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=_timeout_delta(timeout_s),
            )
    payload = _parse_mcp_tool_result(result)
    return payload


async def _list_sse_mcp_tools(
    *,
    url: str,
    timeout_s: float | None,
) -> JsonDict:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(
        url,
        sse_read_timeout=_sse_read_timeout_s(timeout_s),
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
    return _parse_mcp_tools_result(result)


async def _call_streamable_http_mcp_tool(
    *,
    url: str,
    tool_name: str,
    arguments: JsonDict,
    timeout_s: float | None,
) -> JsonDict:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=_streamable_http_timeout(timeout_s)
    ) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (
            read,
            write,
            _get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    tool_name,
                    arguments,
                    read_timeout_seconds=_timeout_delta(timeout_s),
                )
    return _parse_mcp_tool_result(result)


async def _list_streamable_http_mcp_tools(
    *,
    url: str,
    timeout_s: float | None,
) -> JsonDict:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=_streamable_http_timeout(timeout_s)
    ) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (
            read,
            write,
            _get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
    return _parse_mcp_tools_result(result)


def _streamable_http_timeout(timeout_s: float | None) -> Any:
    import httpx

    return httpx.Timeout(30.0, read=_sse_read_timeout_s(timeout_s))


async def _with_optional_timeout(coro: Any, *, timeout_s: float | None) -> Any:
    if timeout_s is None or timeout_s <= 0:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout_s)


@contextmanager
def _temporary_no_proxy_for_url(url: str):
    """Bypass local HTTP proxies for the target MCP host during one call."""

    entries = _no_proxy_entries_for_url(url)
    if not entries:
        yield
        return
    old_values = {key: os.environ.get(key) for key in ("NO_PROXY", "no_proxy")}
    try:
        merged = _merge_no_proxy_entries(old_values["NO_PROXY"], entries)
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _no_proxy_entries_for_url(url: str) -> list[str]:
    parsed = urlparse(str(url or ""))
    host = parsed.hostname
    if not host:
        return []
    entries = [host]
    if parsed.port is not None:
        entries.append(f"{host}:{parsed.port}")
    return entries


def _merge_no_proxy_entries(existing: str | None, entries: Sequence[str]) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*(existing or "").split(","), *entries]:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return ",".join(merged)


def _parse_mcp_tools_result(result: Any) -> JsonDict:
    if isinstance(result, Mapping):
        raw_tools = result.get("tools", [])
    elif hasattr(result, "model_dump"):
        dumped = result.model_dump()
        raw_tools = dumped.get("tools", []) if isinstance(dumped, Mapping) else []
    else:
        raw_tools = getattr(result, "tools", [])
    if not isinstance(raw_tools, (list, tuple)):
        raw_tools = []
    tools = [_mcp_tool_to_dict(tool) for tool in raw_tools]
    return {"tools": tools, "tool_count": len(tools)}


def _mcp_tool_to_dict(tool: Any) -> JsonDict:
    if isinstance(tool, Mapping):
        payload = dict(tool)
    elif hasattr(tool, "model_dump"):
        dumped = tool.model_dump()
        payload = dict(dumped) if isinstance(dumped, Mapping) else {}
    else:
        payload = {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", ""),
        }
        for attr in ("inputSchema", "input_schema"):
            value = getattr(tool, attr, None)
            if isinstance(value, Mapping):
                payload[attr] = dict(value)
    input_schema = payload.get("inputSchema")
    if input_schema is None:
        input_schema = payload.get("input_schema")
    normalized: JsonDict = {
        "name": str(payload.get("name") or ""),
        "description": str(payload.get("description") or ""),
    }
    if isinstance(input_schema, Mapping):
        normalized["input_schema"] = dict(input_schema)
    return normalized


def _parse_mcp_tool_result(result: Any) -> JsonDict:
    is_error = _mcp_result_is_error(result)
    content_items: Any = []
    payload: JsonDict | None = None
    if isinstance(result, Mapping):
        content_items = result.get("content", [])
        if any(
            key in result
            for key in ("isError", "is_error", "structuredContent", "structured_content")
        ):
            for key in ("structuredContent", "structured_content"):
                structured = result.get(key)
                if isinstance(structured, Mapping):
                    payload = dict(structured)
                    break
            if payload is None:
                payload = _parse_mcp_content_items(content_items)
        else:
            payload = dict(result)

    if payload is None:
        for attr in ("structuredContent", "structured_content"):
            structured = getattr(result, attr, None)
            if isinstance(structured, Mapping):
                payload = dict(structured)
                break

    if payload is None and hasattr(result, "model_dump"):
        dumped = result.model_dump()
        if isinstance(dumped, Mapping):
            content_items = dumped.get("content", [])
            for key in ("structuredContent", "structured_content"):
                structured = dumped.get(key)
                if isinstance(structured, Mapping):
                    payload = dict(structured)
                    break
            if payload is None:
                payload = _parse_mcp_content_items(content_items)

    if payload is None:
        content_items = getattr(result, "content", []) or content_items
        payload = _parse_mcp_content_items(content_items)

    text_content = "\n".join(_mcp_content_texts(content_items)).strip()
    if payload is None:
        text = str(result)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            payload = decoded

    if payload is None:
        message = text_content or "Simulator MCP tool returned an invalid response."
        payload = {
            "success": False,
            "error": message,
            "content": message,
            "failure_class": _mcp_error_failure_class(message, is_error=is_error),
            "candidate_rejection": False,
            "details": {"raw_result_type": type(result).__name__},
        }

    if is_error:
        payload["success"] = False
        error_message = str(payload.get("error") or text_content or "").strip()
        if error_message:
            payload.setdefault("error", error_message)
            payload.setdefault("content", error_message)
        payload.setdefault(
            "failure_class",
            _mcp_error_failure_class(error_message, is_error=True),
        )
        payload.setdefault("candidate_rejection", False)
        details = payload.get("details")
        normalized_details = dict(details) if isinstance(details, Mapping) else {}
        normalized_details.setdefault("raw_result_type", type(result).__name__)
        normalized_details["mcp_is_error"] = True
        payload["details"] = normalized_details
    return payload


def _parse_mcp_content_items(items: Any) -> JsonDict | None:
    if not isinstance(items, (list, tuple)):
        return None
    for item in items:
        if isinstance(item, Mapping):
            if isinstance(item.get("json"), Mapping):
                return dict(item["json"])
            if isinstance(item.get("data"), Mapping):
                return dict(item["data"])
            text = item.get("text", "")
        else:
            if isinstance(getattr(item, "json", None), Mapping):
                return dict(getattr(item, "json"))
            if isinstance(getattr(item, "data", None), Mapping):
                return dict(getattr(item, "data"))
            text = getattr(item, "text", "")
        if isinstance(text, Mapping):
            return dict(text)
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _mcp_content_texts(items: Any) -> list[str]:
    if not isinstance(items, (list, tuple)):
        return []
    texts: list[str] = []
    for item in items:
        text = item.get("text", "") if isinstance(item, Mapping) else getattr(item, "text", "")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _mcp_result_is_error(result: Any) -> bool:
    if isinstance(result, Mapping):
        return result.get("isError") is True or result.get("is_error") is True
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        return True
    if hasattr(result, "model_dump"):
        dumped = result.model_dump()
        if isinstance(dumped, Mapping):
            return dumped.get("isError") is True or dumped.get("is_error") is True
    return False


def _mcp_error_failure_class(message: str, *, is_error: bool) -> str:
    normalized = message.lower()
    missing_patterns = (
        "unknown tool",
        "tool not found",
        "no tool named",
        "method not found",
    )
    if any(pattern in normalized for pattern in missing_patterns) or (
        "tool" in normalized and "not found" in normalized
    ):
        return "remote_capability_missing"
    return "mcp_tool_error" if is_error else "invalid_mcp_response"


def _response_success(response: JsonDict) -> bool:
    if response.get("success") is False:
        return False
    if response.get("ok") is False:
        return False
    return "error" not in response


def _context_execution_cancelled(context: ToolExecutionContext) -> bool:
    event = context.metadata.get("_cancel_event")
    return bool(event is not None and callable(getattr(event, "is_set", None)) and event.is_set())


def _response_content(response: JsonDict, *, mcp_tool: str, success: bool) -> str:
    content = response.get("content")
    if isinstance(content, str) and content.strip():
        return content if len(content) <= 500 else content[:500].rstrip()
    if not success:
        return str(response.get("error") or f"Simulator MCP tool failed: {mcp_tool}")
    response_path = response.get("response_path")
    if isinstance(response_path, str) and response_path:
        return f"Simulator MCP tool executed: {mcp_tool}; response saved to {response_path}"
    return f"Simulator MCP tool executed: {mcp_tool}"


def _response_assigned_task(response: JsonDict) -> str:
    observation = response.get("observation_summary")
    if not isinstance(observation, dict):
        return ""
    task = observation.get("task")
    return task.strip() if isinstance(task, str) else ""


def _brief_response_error(response: JsonDict) -> str:
    collision = response.get("collision")
    if _collision_rejects_motion(collision):
        return str(collision.get("message") or "Simulator motion collided before reaching target.")
    motion = build_motion_summary(response)
    if motion.get("reached_target") is False:
        return "Simulator motion did not reach the requested target."
    message = str(response.get("error") or response.get("content") or "")
    if len(message) > 500:
        return message[:500].rstrip()
    return message


def _iter_exception_chain(exc: BaseException) -> Iterable[BaseException]:
    """Yield nested exceptions, including Python 3.11 exception groups."""

    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        grouped = getattr(current, "exceptions", ())
        if isinstance(grouped, tuple):
            pending.extend(item for item in grouped if isinstance(item, BaseException))
        if isinstance(current.__cause__, BaseException):
            pending.append(current.__cause__)
        if isinstance(current.__context__, BaseException):
            pending.append(current.__context__)


def _primary_transport_exception(exc: BaseException) -> BaseException:
    chain = list(_iter_exception_chain(exc))
    leaves = [item for item in chain if not getattr(item, "exceptions", ())]
    for item in leaves:
        if _is_transport_timeout(item) or _is_transient_mcp_transport_error(item):
            return item
    return leaves[0] if leaves else exc


def _is_transport_timeout(exc: BaseException) -> bool:
    return any(
        "timeout" in type(item).__name__.lower()
        or "timed out" in str(item).lower()
        or "read timeout" in str(item).lower()
        for item in _iter_exception_chain(exc)
    )


def _is_transient_mcp_transport_error(exc: BaseException) -> bool:
    type_markers = (
        "brokenresourceerror",
        "connecterror",
        "connectionreseterror",
        "endofstream",
        "readerror",
        "remoteprotocolerror",
    )
    message_markers = (
        "connection refused",
        "connection reset",
        "incomplete chunked read",
        "peer closed connection",
        "server disconnected",
        "unexpected eof",
    )
    for item in _iter_exception_chain(exc):
        name = type(item).__name__.lower()
        message = str(item).lower()
        if "unsupportedprotocol" in name or "missing protocol" in message:
            continue
        if any(marker in name for marker in type_markers):
            return True
        if any(marker in message for marker in message_markers):
            return True
    return False


def _response_lost_action_receipt(response: JsonDict) -> bool:
    """Return true when a mutating call may have run but its receipt was lost."""

    # A structured controller outcome is safe to classify even when it is a
    # rejection.  A bare remote error is not: the HTTP/MCP handler may have
    # failed while serializing the observation after the physical action had
    # already completed.  Treat that boundary as outcome-unknown so the host
    # performs its bounded observe/reconcile path without another planner turn.
    if any(
        key in response
        for key in (
            "execution_started",
            "motion_outcome",
            "reached_goal",
            "terminal_status",
        )
    ):
        return False
    message = str(response.get("error") or response.get("content") or "").lower()
    return bool(message)


def _response_reports_unknown_action_outcome(response: JsonDict) -> bool:
    """Recognize a structured controller result that requires reconciliation."""

    return (
        response.get("reconciliation_required") is True
        or str(response.get("motion_outcome") or "").lower() == "unknown"
        or (
            "execution_started" in response
            and response.get("execution_started") is None
        )
    )


def _move_response_lacks_completion_receipt(response: JsonDict) -> bool:
    """Detect a controller receipt that ran steps but omitted its terminal pose."""

    motion = build_motion_summary(response)
    if motion.get("reached_target") in {True, False}:
        return False
    try:
        steps = int(motion.get("steps_executed") or response.get("steps_executed") or 0)
    except (TypeError, ValueError):
        return False
    if steps <= 0:
        return False
    end = motion.get("end")
    end_xyz = end.get("xyz") if isinstance(end, dict) else None
    return not (
        isinstance(end_xyz, list | tuple)
        and len(end_xyz) >= 3
        and all(
            isinstance(value, int | float) and math.isfinite(float(value)) for value in end_xyz[:3]
        )
    )


def _response_diagnostics(response: JsonDict) -> list[JsonDict]:
    failure_class = str(response.get("failure_class") or "").strip()
    candidate_rejection = response.get("candidate_rejection") is True
    collision = response.get("collision")
    if _collision_rejects_motion(collision):
        return [
            {
                "code": failure_class or "simulator_mcp_collision",
                "message": _brief_response_error(response),
                "collision": dict(collision),
                "candidate_rejection": candidate_rejection,
                "failure_class": failure_class,
            }
        ]
    motion = build_motion_summary(response)
    if motion.get("reached_target") is False:
        return [
            {
                "code": failure_class or "simulator_mcp_target_not_reached",
                "message": _brief_response_error(response),
                "motion_summary": motion,
                "candidate_rejection": candidate_rejection,
                "failure_class": failure_class,
            }
        ]
    return [
        {
            "code": failure_class or "simulator_mcp_error",
            "message": _brief_response_error(response),
            "candidate_rejection": candidate_rejection,
            "failure_class": failure_class,
        }
    ]


def _collision_rejects_motion(collision: object) -> bool:
    if not isinstance(collision, dict) or collision.get("detected") is not True:
        return False
    return collision.get("new_or_worsened") is not False


def _state_delta_from_response(response: JsonDict) -> JsonDict:
    delta: JsonDict = {}
    if "reward" in response:
        delta["reward"] = response.get("reward")
    if "terminated" in response:
        delta["terminated"] = response.get("terminated")
    if "truncated" in response:
        delta["truncated"] = response.get("truncated")
    observation = build_observation_summary(response)
    if observation:
        delta["observation"] = observation
    motion = build_motion_summary(response)
    if motion:
        delta["motion"] = motion
    return delta


def _build_environment_receipt(
    response: JsonDict,
    *,
    observation_snapshot: JsonDict,
    agent_tool: str,
    mcp_tool: str,
    simulator_session_id: str,
    handle: str,
    execution_metadata: JsonDict | None,
    mcp_request_id: str,
) -> JsonDict:
    metadata = dict(execution_metadata or {})
    receipt: JsonDict = {
        "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
        "receipt_id": uuid4().hex,
        "backend": "simulator_mcp",
        "agent_tool": agent_tool,
        "remote_tool": mcp_tool,
        "mcp_request_id": mcp_request_id,
        "execution_id": str(metadata.get("execution_id") or ""),
        "agent_session_id": str(metadata.get("session_id") or ""),
        "simulator_session_id": simulator_session_id,
        "handle": handle,
        "timestamp_s": time.time(),
        "reward_present": "reward" in response,
        "observation_fresh": bool(observation_snapshot),
    }
    for key in ("reward", "terminated", "truncated", "scene_epoch"):
        if key in response:
            receipt[key] = response.get(key)
    for key in CONTROL_RECEIPT_FIELDS:
        if key in response:
            receipt[key] = response.get(key)
    motion = build_motion_summary(response)
    if motion:
        receipt["motion"] = motion
    if observation_snapshot:
        receipt["observation_snapshot"] = observation_snapshot
    return receipt


def _extract_orientation_arguments(
    parameters: JsonDict,
    *,
    tool_name: str,
) -> JsonDict:
    pose = parameters.get("target_pose") or parameters.get("pose") or parameters.get("eef_pose")
    if not isinstance(pose, dict):
        return {}

    direct = [pose.get(axis) for axis in ("roll", "pitch", "yaw")]
    if any(value is not None for value in direct):
        if not all(isinstance(value, int | float) for value in direct):
            raise ValueError("move_to orientation requires roll, pitch, and yaw together.")
        return {
            "roll": float(direct[0]),
            "pitch": float(direct[1]),
            "yaw": float(direct[2]),
        }

    euler = pose.get("euler_xyz_deg")
    if euler is not None:
        if not _finite_numeric_sequence(euler, length=3):
            raise ValueError(
                f"{tool_name} target_pose.euler_xyz_deg must contain 3 finite numbers."
            )
        return {axis: float(euler[idx]) for idx, axis in enumerate(("roll", "pitch", "yaw"))}

    quaternion = pose.get("quat_xyzw")
    if quaternion is not None:
        if not _finite_numeric_sequence(quaternion, length=4):
            raise ValueError(
                f"{tool_name} target_pose.quat_xyzw must contain 4 finite numbers."
            )
        qx, qy, qz, qw = [float(value) for value in quaternion]
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-9:
            raise ValueError(f"{tool_name} target_pose.quat_xyzw must be non-zero.")
        qx, qy, qz, qw = [value / norm for value in (qx, qy, qz, qw)]
        sin_roll_cos_pitch = 2.0 * (qw * qx + qy * qz)
        cos_roll_cos_pitch = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
        sin_pitch = 2.0 * (qw * qy - qz * qx)
        pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
        sin_yaw_cos_pitch = 2.0 * (qw * qz + qx * qy)
        cos_yaw_cos_pitch = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
        return {
            "roll": math.degrees(roll),
            "pitch": math.degrees(pitch),
            "yaw": math.degrees(yaw),
        }

    rotation = pose.get("rotation_matrix")
    if rotation is None:
        return {}
    matrix = _finite_rotation_matrix(rotation)
    if matrix is None:
        raise ValueError(f"{tool_name} target_pose.rotation_matrix must be a finite 3x3 matrix.")
    roll, pitch, yaw = _rotation_matrix_to_xyz_intrinsic_degrees(matrix)
    return {"roll": roll, "pitch": pitch, "yaw": yaw}


_GRASP_ESTIMATOR_PROVENANCE = frozenset(
    {"grasp_pose_estimate", "anygrasp", "contact_graspnet", "graspgenx"}
)


def _is_ranked_grasp_candidate_pose(parameters: JsonDict) -> bool:
    """Return whether target_pose carries normalized grasp-candidate provenance.

    GraspNet-family rotation matrices describe a grasp frame, not the simulator
    controller's EEF frame. The compatibility default preserves the current EEF
    orientation until a deployment explicitly enables its calibrated mapping.

    ``grasp_pose_estimate`` re-ids backend candidates (``gpe-*``) and records
    provenance in ``source_tool``/``source_backend``/``grasp_frame`` rather than
    ``source_model``; those markers must also count as grasp candidates so their
    grasp-frame orientation is never forwarded as a raw EEF orientation.
    """

    pose = parameters.get("target_pose") or parameters.get("pose")
    if not isinstance(pose, dict):
        return False
    if pose.get("compiled_eef_pose") is True:
        return False
    candidate_id = str(pose.get("id") or pose.get("candidate_id") or "").strip()
    source_model = str(pose.get("source_model") or "").strip().lower()
    if source_model in {"anygrasp", "contact_graspnet"}:
        return True
    if candidate_id.startswith("grasp_") and any(
        key in pose for key in ("rank", "backend_index", "score", "gripper_tip_position_xyz")
    ):
        return True
    if str(pose.get("grasp_frame") or "").strip().lower() == "graspnet":
        return True
    source_tool = str(pose.get("source_tool") or "").strip().lower()
    source_backend = str(pose.get("source_backend") or "").strip().lower()
    return (
        source_tool in _GRASP_ESTIMATOR_PROVENANCE
        or source_backend in _GRASP_ESTIMATOR_PROVENANCE
    )


def _is_anyplace_pose(parameters: JsonDict) -> bool:
    pose = parameters.get("target_pose") or parameters.get("pose")
    if not isinstance(pose, dict):
        return False
    if pose.get("compiled_eef_pose") is True:
        return False
    candidate_id = str(
        pose.get("id")
        or pose.get("candidate_id")
        or pose.get("placement_candidate_id")
        or ""
    ).strip()
    source_tool = str(pose.get("source_tool") or "").strip().lower()
    return (
        source_tool == "anyplace"
        or "object_placement_transform" in pose
        or ("object_goal_pose" in pose and pose.get("compiled_eef_pose") is not True)
        or candidate_id.startswith("placement_")
    )


def _extract_graspnet_panda_orientation_arguments(
    parameters: JsonDict,
    *,
    tool_name: str,
) -> JsonDict:
    """Map a world GraspNet grasp frame to robosuite Panda EEF axes."""

    pose = parameters.get("target_pose") or parameters.get("pose")
    if not isinstance(pose, dict):
        return {}
    rotation = pose.get("rotation_matrix")
    if rotation is None:
        return {}
    grasp_matrix = _finite_rotation_matrix(rotation)
    if grasp_matrix is None:
        raise ValueError(f"{tool_name} target_pose.rotation_matrix must be a finite 3x3 matrix.")

    # GraspNet: x=approach, y=closing, z=binormal. Panda EEF:
    # x=closing, y=binormal, z=approach.
    eef_matrix = [[row[1], row[2], row[0]] for row in grasp_matrix]
    roll, pitch, yaw = _rotation_matrix_to_xyz_intrinsic_degrees(eef_matrix)
    return {"roll": roll, "pitch": pitch, "yaw": yaw}


def _rotation_matrix_to_xyz_intrinsic_degrees(
    matrix: list[list[float]],
) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1e-8:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        yaw = 0.0
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _finite_rotation_matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    rows: list[list[float]] = []
    for row in value:
        if not _finite_numeric_sequence(row, length=3):
            return None
        rows.append([float(item) for item in row])
    return rows


def _finite_numeric_sequence(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) == length
        and all(isinstance(item, int | float) and math.isfinite(float(item)) for item in value)
    )


def _extract_observation_payload(payload: JsonDict) -> JsonDict:
    observation = payload.get("observation")
    if isinstance(observation, dict):
        return observation
    if any(key in payload for key in ("cameras", "robot", "proprio", "objects")):
        return payload
    if any(key in payload for key in ("rgb_path", "rgb_ref", "image_path", "image_ref")):
        return {"cameras": [{"frame_id": "render", **payload}]}
    return {"cameras": [], "robot": RobotState().to_dict(), "metadata": {"raw_payload": payload}}


def _raise_if_mcp_error(payload: JsonDict, *, tool_name: str) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{tool_name} returned {type(payload).__name__}")
    if payload.get("success") is False or payload.get("ok") is False or "error" in payload:
        raise RuntimeError(str(payload.get("error") or f"{tool_name} failed"))


def _is_transient_startup_error(exc: BaseException) -> bool:
    if _is_transport_timeout(exc) or _is_transient_mcp_transport_error(exc):
        return True
    return any(
        marker in str(item).lower()
        for item in _iter_exception_chain(exc)
        for marker in ("unknown handle", "handle not found")
    )


def _summarize_mcp_action(action: EnvAction) -> JsonDict:
    request = action.command.get("request", {})
    return {
        "action_type": action.action_type,
        "request_kind": request.get("kind"),
        "request_name": request.get("name"),
        "status": action.command.get("status"),
        "tool_calls": [
            {
                "name": call.get("name"),
                "status": call.get("status"),
                "result_content": _truncate_action_text((call.get("result") or {}).get("content"))
                if isinstance(call.get("result"), dict)
                else None,
            }
            for call in action.command.get("tool_calls", [])
            if isinstance(call, dict)
        ],
    }


def _truncate_action_text(value: object, *, max_chars: int = 300) -> object:
    if not isinstance(value, str):
        return value
    return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"


def _latest_action_reward(action: EnvAction, payload: JsonDict) -> float:
    if "reward" in payload:
        reward = payload.get("reward")
        if (
            isinstance(reward, int | float)
            and not isinstance(reward, bool)
            and math.isfinite(float(reward))
        ):
            return float(reward)
        return 0.0
    receipt = _latest_trusted_action_environment_receipt(action)
    reward = receipt.get("reward")
    if (
        receipt.get("reward_present") is True
        and isinstance(reward, int | float)
        and not isinstance(reward, bool)
        and math.isfinite(float(reward))
    ):
        return float(reward)
    return 0.0


def _latest_action_flag(action: EnvAction, payload: JsonDict, key: str) -> bool:
    if key in payload:
        value = payload.get(key)
        return value if isinstance(value, bool) else False
    value = _latest_trusted_action_environment_receipt(action).get(key)
    return value if isinstance(value, bool) else False


def _latest_action_receipt_has_reward(action: EnvAction) -> bool:
    return _latest_trusted_action_environment_receipt(action).get("reward_present") is True


def _latest_trusted_action_environment_receipt(action: EnvAction) -> JsonDict:
    calls = action.command.get("tool_calls")
    if not isinstance(calls, list):
        return {}
    for call in reversed(calls):
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        details = result.get("details") if isinstance(result, dict) else None
        if not isinstance(details, dict):
            continue
        provenance = details.get("host_provenance")
        receipt = details.get("environment_receipt")
        if (
            isinstance(provenance, dict)
            and provenance.get("authority") == ENVIRONMENT_AUTHORITY
            and isinstance(receipt, dict)
            and receipt.get("schema_version") == ENVIRONMENT_RECEIPT_SCHEMA_VERSION
        ):
            return receipt
    return {}


def _latest_action_termination_reason(action: EnvAction) -> str:
    """Recognize an explicit remote episode-terminal error in the last tool result."""

    markers = (
        "executing action in terminated episode",
        "episode is terminated",
        "episode already terminated",
    )
    for call in reversed(action.command.get("tool_calls", [])):
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if not isinstance(result, dict) or result.get("success") is not False:
            continue
        messages = [result.get("content")]
        details = result.get("details")
        diagnostics = details.get("diagnostics") if isinstance(details, dict) else None
        messages.extend(item.get("message") for item in diagnostics or [] if isinstance(item, dict))
        if any(marker in str(message or "").lower() for marker in markers for message in messages):
            return "remote_episode_terminated"
    return ""


def _extract_xyz(
    parameters: JsonDict,
    *,
    tool_name: str,
) -> tuple[float, float, float]:
    pose = (
        parameters.get("target_pose")
        or parameters.get("pose")
        or parameters.get("eef_pose")
        or parameters
    )
    if isinstance(pose, dict):
        frame = str(pose.get("frame") or "").strip().lower()
        if frame and frame != "world":
            raise ValueError(f"{tool_name} target_pose.frame must be 'world'.")
        xyz = pose.get("xyz") or pose.get("position")
        if xyz is None:
            xyz = pose.get("translation_xyz")
        if xyz is None and all(axis in pose for axis in ("x", "y", "z")):
            xyz = [pose["x"], pose["y"], pose["z"]]
    else:
        xyz = pose
    if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
        raise ValueError(f"{tool_name} requires target_pose.xyz or x/y/z.")
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _timeout_delta(timeout_s: float | None) -> timedelta | None:
    if timeout_s is None:
        return None
    return timedelta(seconds=timeout_s)


def _sse_read_timeout_s(timeout_s: float | None) -> float:
    """Keep the transport stream alive through the caller-owned deadline."""

    if timeout_s is None or timeout_s <= 0:
        return DEFAULT_MCP_SSE_READ_TIMEOUT_S
    return max(DEFAULT_MCP_SSE_READ_TIMEOUT_S, float(timeout_s) + MCP_SSE_TIMEOUT_GRACE_S)
