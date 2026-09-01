#!/usr/bin/env python3
"""Emit shell-quoted provider exports from a Docker secret."""

from __future__ import annotations

from pathlib import Path
import shlex
import sys

from agent.backends.provider_config import read_env_file


ALLOWED_PREFIXES = (
    "OPENETA_LLM_",
    "OPENETA_OBJECT_MEMORY_",
    "OPENETA_WEB_",
)


def provider_values(path: Path) -> dict[str, str]:
    return {
        key: value
        for key, value in read_env_file(path).items()
        if key.startswith(ALLOWED_PREFIXES)
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: load_provider_env.py FILE", file=sys.stderr)
        return 2
    values = provider_values(Path(sys.argv[1]))
    for key in sorted(values):
        value = values[key]
        if "\0" in key or "\0" in value or "=" in key:
            raise ValueError(f"invalid provider environment key: {key!r}")
        print("export " + shlex.quote(f"{key}={value}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
