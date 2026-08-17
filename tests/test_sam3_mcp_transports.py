"""Wire-level MCP transport coverage for the SAM3 service."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools" / "sam3_mcp_server.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_health(url: str, process: subprocess.Popen[str]) -> dict:
    deadline = time.monotonic() + 10.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise AssertionError(f"SAM3 MCP server exited early: {stderr}")
        try:
            with urlopen(url, timeout=0.5) as response:  # noqa: S310 - loopback test server
                return json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"SAM3 MCP health endpoint did not become ready: {last_error}")


async def _list_and_call_legacy_sse(url: str) -> tuple[set[str], dict]:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool(
                "segment",
                {"image_base64": "", "prompt": "test object"},
            )
    return {tool.name for tool in tools.tools}, json.loads(result.content[0].text)


async def _list_and_call_streamable_http(url: str) -> tuple[set[str], dict]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool(
                "segment",
                {"image_base64": "", "prompt": "test object"},
            )
    return {tool.name for tool in tools.tools}, json.loads(result.content[0].text)


def test_sam3_dual_server_supports_standard_http_and_legacy_sse() -> None:
    """Verify initialize → tools/list → tools/call on both advertised paths."""

    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            str(SERVER),
            "--transport",
            "dual",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        health = _wait_for_health(f"{base_url}/", process)
        assert health == {
            "ok": True,
            "server": "sam3",
            "mcp": {
                "primary_transport": "streamable-http",
                "endpoint": "/mcp",
                "legacy_sse_endpoint": "/sse",
            },
        }

        legacy_tools, legacy_payload = asyncio.run(
            _list_and_call_legacy_sse(f"{base_url}/sse")
        )
        modern_tools, modern_payload = asyncio.run(
            _list_and_call_streamable_http(f"{base_url}/mcp")
        )
        assert legacy_tools == modern_tools == {"segment", "segment_points"}
        assert legacy_payload["success"] is False
        assert modern_payload == legacy_payload
        assert modern_payload["details"]["reason"] == "missing_image"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
