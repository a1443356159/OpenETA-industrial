from __future__ import annotations

from types import SimpleNamespace

from extensions.gazebo import GazeboLiveMcpTransport


class FakeSession:
    def __init__(self, task: str, seed: int) -> None:
        self.task, self.seed, self.closed = task, seed, False

    def create(self):
        return SimpleNamespace(to_dict=lambda: {"task": self.task, "cameras": [], "robot": {}, "metadata": {}})

    def reset(self, *, seed=None):
        self.seed = seed if seed is not None else self.seed
        return self.create()

    def observe(self):
        return self.create()

    def close(self):
        self.closed = True


def test_live_transport_uses_existing_mcp_tool_names_and_cleanup() -> None:
    made = []

    def factory(*, task, seed):
        session = FakeSession(task, seed)
        made.append(session)
        return session

    transport = GazeboLiveMcpTransport(factory)
    created = transport.call_tool("create_env", {"task": "inspect", "seed": 4})
    handle = created["handle"]
    assert transport.call_tool("reset_env", {"handle": handle, "seed": 9})["ok"]
    assert transport.call_tool("render_env", {"handle": handle})["ok"]
    assert transport.call_tool("close_env", {"handle": handle})["ok"]
    assert made[0].closed and transport.active_handles == ()

