#!/usr/bin/env python3
"""Generate the fixed-seed 2M-sample sparse capability map for one robot hash.

The kinematics plugin is an explicit ``module:factory`` callable.  Its factory
receives the parsed CLI namespace and must return an object exposing
``forward_kinematics(joints) -> (xyz, quat_xyzw)`` and ``jacobian(joints)``.
Keeping that adapter external lets the same audited generator work with a ROS
MoveIt model loader or an offline Pinocchio loader without coupling either to
the OpenETA runtime.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from agent.runtime.capability_map import (
    DEFAULT_CAPABILITY_SAMPLE_COUNT,
    DEFAULT_ORIENTATION_NEIGHBORHOOD_DEG,
    DEFAULT_POSITION_RESOLUTION_M,
    DEFAULT_SOBOL_SEED,
    generate_sparse_capability_map,
    robot_model_hash,
)


def _floats(value: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one float")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--planning-group", required=True)
    parser.add_argument("--tcp", required=True)
    parser.add_argument("--gripper", required=True)
    parser.add_argument("--joint-lower", type=_floats, required=True)
    parser.add_argument("--joint-upper", type=_floats, required=True)
    parser.add_argument(
        "--kinematics-plugin",
        default="agent.runtime.urdf_jacobian:capability_map_plugin",
    )
    parser.add_argument("--sample-count", type=int, default=DEFAULT_CAPABILITY_SAMPLE_COUNT)
    parser.add_argument("--sobol-seed", type=int, default=DEFAULT_SOBOL_SEED)
    parser.add_argument(
        "--position-resolution-m", type=float, default=DEFAULT_POSITION_RESOLUTION_M
    )
    parser.add_argument(
        "--orientation-neighborhood-deg",
        type=float,
        default=DEFAULT_ORIENTATION_NEIGHBORHOOD_DEG,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_plugin(specification: str, args: argparse.Namespace) -> Any:
    module_name, separator, factory_name = specification.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("kinematics plugin must use module:factory syntax")
    factory = getattr(importlib.import_module(module_name), factory_name)
    plugin = factory(args)
    if not callable(getattr(plugin, "forward_kinematics", None)) or not callable(
        getattr(plugin, "jacobian", None)
    ):
        raise TypeError("kinematics plugin lacks FK/Jacobian callbacks")
    return plugin


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.joint_lower) != len(args.joint_upper):
        raise SystemExit("joint lower/upper lengths differ")
    model_hash = robot_model_hash(
        urdf=args.urdf.read_bytes(),
        srdf=args.srdf.read_bytes(),
        planning_group=args.planning_group,
        tcp=args.tcp,
        gripper=args.gripper,
    )
    plugin = _load_plugin(args.kinematics_plugin, args)
    payload = generate_sparse_capability_map(
        robot_model_sha256=model_hash,
        joint_lower=args.joint_lower,
        joint_upper=args.joint_upper,
        forward_kinematics=plugin.forward_kinematics,
        jacobian=plugin.jacobian,
        sample_count=args.sample_count,
        sobol_seed=args.sobol_seed,
        position_resolution_m=args.position_resolution_m,
        orientation_neighborhood_deg=args.orientation_neighborhood_deg,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"map_id": payload["map_id"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
