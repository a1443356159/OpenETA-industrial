"""Wire-level transport coverage for the M0--M4 simulator MCP server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_listener(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise AssertionError(f"simulator MCP server exited early: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError("simulator MCP server did not become ready")


def _text_payload(result) -> dict:
    assert result.isError is False
    for item in result.content:
        if getattr(item, "type", None) == "text":
            return json.loads(item.text)
    raise AssertionError("MCP tool response did not contain text JSON")


async def _list_and_call_legacy_sse(url: str) -> tuple[set[str], dict]:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("list_active_envs", {})
    return {tool.name for tool in tools.tools}, _text_payload(result)


async def _list_and_call_streamable_http(url: str) -> tuple[set[str], dict]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("list_active_envs", {})
    return {tool.name for tool in tools.tools}, _text_payload(result)


def test_simulator_mcp_supports_standard_http_and_legacy_sse_read_only_lifecycle() -> None:
    """Both advertised paths complete initialize → tools/list → a read-only call.

    The M0--M4 scripted TUI continues to use `/sse`; `/mcp` is exercised here
    as the preferred Streamable HTTP endpoint without creating an environment
    or starting Gazebo.
    """

    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sim.mcp_server.server",
            "--transport",
            "sse",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_listener(port, process)
        base_url = f"http://127.0.0.1:{port}"
        legacy_tools, legacy_payload = asyncio.run(
            _list_and_call_legacy_sse(f"{base_url}/sse")
        )
        modern_tools, modern_payload = asyncio.run(
            _list_and_call_streamable_http(f"{base_url}/mcp")
        )
        assert "list_active_envs" in legacy_tools
        assert modern_tools == legacy_tools
        assert legacy_payload["count"] == 0
        assert modern_payload["count"] == 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
