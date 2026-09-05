"""Wire-level transport coverage for the simulator MCP server."""

from __future__ import annotations

import asyncio
from http.client import HTTPConnection, HTTPException
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from agent.tools.sim_mcp import StreamableHttpSimulatorMcpTransport


ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_application(port: int, process: subprocess.Popen[str]) -> None:
    """Wait for an ASGI response rather than a pre-bound TCP handshake."""

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise AssertionError(f"simulator MCP server exited early: {stderr}")
        try:
            connection = HTTPConnection("127.0.0.1", port, timeout=0.25)
            try:
                connection.request("GET", "/__openeta_mcp_ready__")
                response = connection.getresponse()
                response.read()
                if response.status == 404:
                    return
            finally:
                connection.close()
        except (OSError, HTTPException):
            pass
        time.sleep(0.05)
    raise AssertionError("simulator MCP application did not become ready")


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

    The normal scripted TUI uses the preferred `/mcp` endpoint. Legacy `/sse`
    remains covered for compatible deployments, without creating an
    environment or starting Gazebo.
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
        _wait_for_application(port, process)
        base_url = f"http://127.0.0.1:{port}"
        legacy_tools, legacy_payload = asyncio.run(
            _list_and_call_legacy_sse(f"{base_url}/sse")
        )
        modern_tools, modern_payload = asyncio.run(
            _list_and_call_streamable_http(f"{base_url}/mcp")
        )
        transport = StreamableHttpSimulatorMcpTransport(f"{base_url}/mcp")
        transport_tools = transport.list_tools(timeout_s=10.0)
        transport_payload = transport.call_tool(
            "list_active_envs",
            {},
            timeout_s=10.0,
        )
        assert "list_active_envs" in legacy_tools
        assert modern_tools == legacy_tools
        assert {item["name"] for item in transport_tools["tools"]} == modern_tools
        assert legacy_payload["count"] == 0
        assert modern_payload["count"] == 0
        assert transport_payload["count"] == 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def test_simulator_mcp_accepts_a_prebound_loopback_listener() -> None:
    """The acceptance launcher can hand a held port to Uvicorn without a race."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.set_inheritable(True)
    port = int(listener.getsockname()[1])
    fd = listener.fileno()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sim.mcp_server.server",
            "--transport",
            "sse",
            "--port",
            str(port),
            "--fd",
            str(fd),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        pass_fds=(fd,),
    )
    listener.close()
    try:
        _wait_for_application(port, process)
        transport = StreamableHttpSimulatorMcpTransport(
            f"http://127.0.0.1:{port}/mcp"
        )
        tools = transport.list_tools(timeout_s=10.0)
        assert any(item["name"] == "list_active_envs" for item in tools["tools"])
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
