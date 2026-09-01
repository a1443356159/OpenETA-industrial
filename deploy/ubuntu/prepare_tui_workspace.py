#!/usr/bin/env python3
"""Create the container-owned MCP registry used by the interactive TUI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def mcp_registry(*, sim_port: int, sam3_port: int, anyplace_port: int, graspgenx_port: int) -> dict:
    def endpoint(port: int) -> dict[str, str]:
        return {"url": f"http://127.0.0.1:{port}/sse"}

    return {
        "mcpServers": {
            "openeta-sim": endpoint(sim_port),
            "openeta-sam3": endpoint(sam3_port),
            "openeta-anyplace": endpoint(anyplace_port),
            "openeta-graspgenx": endpoint(graspgenx_port),
        }
    }


def write_registry(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim-port", type=int, default=8765)
    parser.add_argument("--sam3-port", type=int, default=8773)
    parser.add_argument("--anyplace-port", type=int, default=8775)
    parser.add_argument("--graspgenx-port", type=int, default=8778)
    args = parser.parse_args()
    ports = (args.sim_port, args.sam3_port, args.anyplace_port, args.graspgenx_port)
    if any(port < 1 or port > 65535 for port in ports):
        parser.error("MCP ports must be in [1, 65535]")
    write_registry(
        args.output.resolve(),
        mcp_registry(
            sim_port=args.sim_port,
            sam3_port=args.sam3_port,
            anyplace_port=args.anyplace_port,
            graspgenx_port=args.graspgenx_port,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
