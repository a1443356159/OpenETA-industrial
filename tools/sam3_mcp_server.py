#!/usr/bin/env python
"""SAM3 MCP server for OpenETA."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from tools.sam3_core import segment_image_points, segment_image_prompt


mcp = FastMCP("sam3", log_level="WARNING")


@mcp.tool()
def segment(
    image_base64: str,
    prompt: str,
    *,
    image_format: str = "png",
    confidence_threshold: float = 0.5,
) -> dict:
    """Segment one image with a text prompt.

    Use this MCP tool when a caller has RGB image bytes and needs object
    masks/boxes for an open-vocabulary phrase. The MCP wire contract uses
    base64 payloads, not local file paths and not OpenETA artifact refs.

    Args:
        image_base64: Base64-encoded source image bytes.
        prompt: Short text phrase for the target object or concept.
        image_format: Source image encoding such as png, jpg, or jpeg.
        confidence_threshold: Detection confidence threshold.

    Example:
        {
            "image_base64": "<base64-encoded png bytes>",
            "image_format": "png",
            "prompt": "black shoe",
            "confidence_threshold": 0.5
        }

    Detections are returned with ranking=score_descending, a zero-based rank,
    and backend_index preserving the model's original output position. The
    response may include mask and overlay image payloads as base64. Clients
    should materialize those payloads into local temporary files, then pass file
    refs to downstream tools instead of reading or logging the base64 directly.
    """

    return segment_image_prompt(
        image_base64=image_base64,
        prompt=prompt,
        image_format=image_format,
        confidence_threshold=confidence_threshold,
    )


@mcp.tool()
def segment_points(
    image_base64: str,
    points: list[dict],
    *,
    image_format: str = "png",
) -> dict:
    """Segment one image from foreground and background pixel points.

    Use this MCP tool when a caller already has point coordinates on an RGB
    image and needs SAM3 instance masks. The MCP wire contract uses base64 image
    bytes, not local file paths and not OpenETA artifact refs.

    Args:
        image_base64: Base64-encoded source PNG or JPEG image bytes.
        points: One to 64 point objects with finite x/y pixel coordinates and a
            label. Coordinates use the original image resolution, origin at the
            top-left, x increasing rightward, and y increasing downward. label=1
            is a foreground point and label=0 is a background point. At least
            one foreground point is required. Normalized coordinates are not
            accepted, and out-of-bounds coordinates fail the whole request.
        image_format: Source image encoding such as png, jpg, or jpeg.

    Example:
        {
            "image_base64": "<base64-encoded jpeg bytes>",
            "image_format": "jpeg",
            "points": [
                {"x": 466.0, "y": 480.0, "label": 1},
                {"x": 520.0, "y": 480.0, "label": 0}
            ]
        }

    The tool always requests SAM3 multimask output and atomically returns
    exactly three candidates ranked by score. score is SAM3's predicted mask
    quality, not a calibrated probability; callers should not assume the
    highest-scoring mask is necessarily the complete semantic object. Each
    bbox_xyxy uses half-open pixel bounds [x_min, y_min, x_max, y_max], suitable
    for direct NumPy/Pillow slicing. Each candidate has its own base64 mask and
    overlay. Clients should materialize those payloads into local temporary
    files, then pass file refs to downstream tools instead of reading or logging
    the base64 directly.
    """

    return segment_image_points(
        image_base64=image_base64,
        points=points,
        image_format=image_format,
    )


def build_dual_http_app():
    """Serve standard MCP HTTP and legacy SSE compatibility endpoints.

    Streamable HTTP at ``/mcp`` is the current MCP HTTP transport.  The
    separate ``/sse`` and ``/messages/`` routes stay only for existing OpenETA
    M5 callers that still use the superseded HTTP+SSE transport.  Both route
    sets are constructed through FastMCP's public ASGI APIs; this server does
    not hand-roll JSON-RPC or access its private low-level server.
    """

    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    # Calling streamable_http_app creates FastMCP's public session manager.
    # Its lifespan is not retained when its routes are combined below, so this
    # outer application owns the manager explicitly.
    streamable_app = mcp.streamable_http_app()
    legacy_sse_app = mcp.sse_app()

    async def health(_request):
        return JSONResponse(
            {
                "ok": True,
                "server": "sam3",
                "mcp": {
                    "primary_transport": "streamable-http",
                    "endpoint": "/mcp",
                    "legacy_sse_endpoint": "/sse",
                },
            }
        )

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with mcp.session_manager.run():
            yield

    return Starlette(
        routes=[
            Route("/", health, methods=["GET"]),
            *legacy_sse_app.routes,
            *streamable_app.routes,
        ],
        lifespan=lifespan,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenETA SAM3 MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http", "dual"),
        default="stdio",
        help=(
            "MCP transport: stdio, legacy SSE, standard Streamable HTTP, or "
            "dual (/mcp plus legacy /sse compatibility)."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return 0

    # Standalone FastMCP modes read network settings from this public object.
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    if args.transport == "sse":
        print(f"\n  SAM3 MCP legacy SSE: http://{args.host}:{args.port}/sse")
        mcp.run(transport="sse")
        return 0
    if args.transport == "streamable-http":
        print(f"\n  SAM3 MCP Streamable HTTP: http://{args.host}:{args.port}/mcp")
        mcp.run(transport="streamable-http")
        return 0

    import uvicorn

    print(f"\n  SAM3 MCP Streamable HTTP: http://{args.host}:{args.port}/mcp")
    print(f"  SAM3 MCP legacy SSE:     http://{args.host}:{args.port}/sse")
    print(f"  Health:                   http://{args.host}:{args.port}/")
    uvicorn.run(build_dual_http_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
