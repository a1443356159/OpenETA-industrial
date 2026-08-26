#!/usr/bin/env python
"""AnyPlace MCP server for OpenETA."""

from __future__ import annotations

import argparse
import threading
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from tools.anyplace_core import (  # noqa: E402
    DEFAULT_INFERENCE_SEED,
    AnyPlaceBackend,
)
from tools.candidate_config import (  # noqa: E402
    DEFAULT_ANYPLACE_RAW_POOL_SIZE,
    argparse_raw_pool_size,
)


mcp = FastMCP("anyplace", log_level="WARNING")
_BACKEND: AnyPlaceBackend | None = None
_PREDICT_LOCK = threading.Lock()


@mcp.tool()
def predict_placement(
    object_observation: dict[str, Any] | None = None,
    placement_observation: dict[str, Any] | None = None,
    object_camera_to_placement_camera: list[list[float]] | None = None,
    placement_camera_to_world: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Predict object placement transforms from two independent observations.

    ``object_observation`` contains aligned RGB, depth, ``object_mask``, and
    intrinsics. ``placement_observation`` independently contains aligned RGB,
    depth, ``placement_region_mask``, and intrinsics. They may come from
    different cameras or times. The host supplies the calibrated rigid
    transform between their OpenCV camera frames and the calibrated
    placement-camera-to-world transform used to gravity-align official model
    inputs; no grasp candidate is accepted by this service.

    Args:
        object_observation: Independent object RGB-D/mask packet.
        placement_observation: Independent target-region RGB-D/mask packet.
        object_camera_to_placement_camera: Calibrated row-major 4x4 transform.
        placement_camera_to_world: Calibrated OpenCV-camera-to-world row-major 4x4 transform.

    Example:
        {
            "object_observation": {"rgb": {"format": "png", "base64": "..."}, "depth": {}, "object_mask": {}, "intrinsics": {}},
            "placement_observation": {"rgb": {"format": "png", "base64": "..."}, "depth": {}, "placement_region_mask": {}, "intrinsics": {}},
            "object_camera_to_placement_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            "placement_camera_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        }

    The tool returns the configured reserve of placement candidates in backend order. Each
    contains only ``object_placement_transform``. It does not accept a grasp,
    return an EEF pose, choose a best candidate, or execute motion. Do
    not materialize input base64 payloads in planner, memory, or action logs.
    """

    if _BACKEND is None:
        return {
            "success": False,
            "content": "AnyPlace placement prediction failed: backend not configured.",
            "details": {
                "tool": "anyplace",
                "backend": "anyplace_mcp",
                "model": "anyplace_multitask",
                "frame": "placement_camera",
                "camera_frame": "opencv",
                "candidate_count": 0,
                "placement_candidates": [],
                "reason": "model_load_failed",
                "metadata": {},
            },
        }
    with _PREDICT_LOCK:
        try:
            return _BACKEND.predict_placement(
                object_observation=object_observation,
                placement_observation=placement_observation,
                object_camera_to_placement_camera=object_camera_to_placement_camera,
                placement_camera_to_world=placement_camera_to_world,
            )
        finally:
            _release_cuda_cache()


def _release_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def health_payload() -> dict[str, Any]:
    backend = _BACKEND
    return {
        "ok": backend is not None,
        "server": "anyplace",
        "deterministic": True,
        "inference_seed": (
            DEFAULT_INFERENCE_SEED if backend is None else int(backend.seed)
        ),
        "raw_pool_size": 0 if backend is None else int(backend.raw_pool_size),
        "returned_candidate_count": (
            0 if backend is None else int(backend.last_returned_candidate_count)
        ),
    }


def main() -> int:
    global _BACKEND
    parser = argparse.ArgumentParser(description="OpenETA AnyPlace MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--anyplace-root", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument(
        "--inference-seed", type=int, default=DEFAULT_INFERENCE_SEED
    )
    parser.add_argument(
        "--raw-pool-size",
        type=argparse_raw_pool_size(placement=True),
        default=DEFAULT_ANYPLACE_RAW_POOL_SIZE,
    )
    args = parser.parse_args()

    anyplace_root = Path(args.anyplace_root)
    config_path = Path(args.config_path)
    if not anyplace_root.exists():
        parser.error(f"--anyplace-root does not exist: {anyplace_root}")
    if not config_path.exists():
        parser.error(f"--config-path does not exist: {config_path}")

    try:
        _BACKEND = AnyPlaceBackend(
            anyplace_root=anyplace_root,
            config_path=config_path,
            seed=args.inference_seed,
            raw_pool_size=args.raw_pool_size,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return 0

    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health(_request):
        return JSONResponse(health_payload())

    health_app = Starlette(routes=[Route("/", health, methods=["GET"])])
    sse_transport = SseServerTransport("/sse/messages/")

    async def combined(scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            if path == "/sse" and scope["method"] == "GET":
                async with sse_transport.connect_sse(scope, receive, send) as streams:
                    await mcp._mcp_server.run(
                        streams[0],
                        streams[1],
                        mcp._mcp_server.create_initialization_options(),
                    )
                return
            if path.startswith("/sse/messages/") and scope["method"] == "POST":
                await sse_transport.handle_post_message(scope, receive, send)
                return
        await health_app(scope, receive, send)

    print(f"\n  AnyPlace MCP SSE: http://{args.host}:{args.port}/sse")
    print(f"  Health:           http://{args.host}:{args.port}/")
    uvicorn.run(combined, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
