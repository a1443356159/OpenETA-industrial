"""In-process MCP transport for the M1 oracle environment.

This is a contract test transport, not a second MCP server.  It mirrors the
existing simulator tool names so OpenETA's own
``SimulatorMcpEpisodeEnvironment`` can exercise lifecycle and cleanup without
requiring a Gazebo installation.  A real deployment replaces this transport
with the existing SSE/stdio MCP transport.
"""

from __future__ import annotations

from uuid import uuid4

from adapter.protocol import EnvObservation

from .config import GazeboConfig
from .lifecycle import GazeboEnvironment
from .live import GazeboLiveSession


class GazeboOracleMcpTransport:
    """Small synchronous transport implementing the simulator MCP contract."""

    def __init__(self, *, config: GazeboConfig | None = None) -> None:
        self.config = config
        self._environments: dict[str, GazeboEnvironment] = {}
        self.close_calls = 0

    def list_tools(self, *, timeout_s: float | None = None) -> dict:
        del timeout_s
        return {
            "tools": [
                {"name": "create_env"}, {"name": "reset_env"},
                {"name": "render_env"}, {"name": "close_env"},
            ]
        }

    def call_tool(self, name: str, arguments: dict, *, timeout_s: float | None = None) -> dict:
        del timeout_s
        if name == "create_env":
            handle = uuid4().hex[:12]
            env = GazeboEnvironment(config=self.config, task=str(arguments.get("task") or ""),
                                    seed=int(arguments.get("seed", 0)))
            self._environments[handle] = env
            env.create()
            return {"ok": True, "handle": handle, "session_id": str(arguments.get("session_id") or "oracle")}
        handle = str(arguments.get("handle") or "")
        env = self._environments.get(handle)
        if name in {"reset_env", "render_env"} and env is None:
            return {"ok": False, "error": f"Unknown handle: {handle}"}
        if name == "reset_env":
            observation = env.reset(seed=arguments.get("seed"))
            return _observation_payload(observation)
        if name == "render_env":
            return _observation_payload(env.observe())
        if name == "close_env":
            self.close_calls += 1
            if env is not None:
                env.close()
                self._environments.pop(handle, None)
            return {"ok": True, "handle": handle}
        return {"ok": False, "error": f"Unsupported MCP tool: {name}"}

    @property
    def active_handles(self) -> tuple[str, ...]:
        return tuple(self._environments)


def _observation_payload(observation: EnvObservation) -> dict:
    """Return the payload shape consumed by the existing MCP episode bridge."""

    return {"ok": True, **observation.to_dict()}


class GazeboLiveMcpTransport:
    """MCP transport facade for configured :class:`GazeboLiveSession` objects."""

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory
        self._sessions: dict[str, GazeboLiveSession] = {}

    def list_tools(self, *, timeout_s: float | None = None) -> dict:
        del timeout_s
        return {"tools": [{"name": name} for name in ("create_env", "reset_env", "render_env", "close_env")]}

    def call_tool(self, name: str, arguments: dict, *, timeout_s: float | None = None) -> dict:
        del timeout_s
        if name == "create_env":
            handle = uuid4().hex[:12]
            session = self.session_factory(
                task=str(arguments.get("task") or ""), seed=int(arguments.get("seed", 0))
            )
            self._sessions[handle] = session
            try:
                session.create()
            except Exception:
                self._sessions.pop(handle, None)
                session.close()
                raise
            return {"ok": True, "handle": handle,
                    "session_id": str(arguments.get("session_id") or "gazebo-live")}
        handle = str(arguments.get("handle") or "")
        session = self._sessions.get(handle)
        if name in {"reset_env", "render_env"} and session is None:
            return {"ok": False, "error": f"Unknown handle: {handle}"}
        if name == "reset_env":
            return _observation_payload(session.reset(seed=arguments.get("seed")))
        if name == "render_env":
            return _observation_payload(session.observe())
        if name == "close_env":
            if session is not None:
                session.close()
                self._sessions.pop(handle, None)
            return {"ok": True, "handle": handle}
        return {"ok": False, "error": f"Unsupported MCP tool: {name}"}

    @property
    def active_handles(self) -> tuple[str, ...]:
        return tuple(self._sessions)
