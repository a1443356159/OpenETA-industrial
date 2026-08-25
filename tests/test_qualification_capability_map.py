from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from agent.runtime.capability_map import (
    SparseCapabilityMap,
    generate_sparse_capability_map,
    robot_model_hash,
    sobol_batches,
)


def _canonical_payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_robot_hash_changes_with_tcp_or_gripper():
    base = robot_model_hash(
        urdf="urdf",
        srdf="srdf",
        planning_group="arm",
        tcp="tool0",
        gripper="robotiq",
    )
    assert base != robot_model_hash(
        urdf="urdf",
        srdf="srdf",
        planning_group="arm",
        tcp="tool1",
        gripper="robotiq",
    )
    assert base != robot_model_hash(
        urdf="urdf",
        srdf="srdf",
        planning_group="arm",
        tcp="tool0",
        gripper="other",
    )


def test_sobol_batches_are_fixed_seed_and_streamed():
    first = np.concatenate(list(sobol_batches(3, 9, seed=7, batch_size=4)))
    second = np.concatenate(list(sobol_batches(3, 9, seed=7, batch_size=5)))
    different = np.concatenate(list(sobol_batches(3, 9, seed=8, batch_size=9)))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.shape == (9, 3)
    assert np.all((0.0 <= first) & (first < 1.0))


def test_empty_capability_cell_is_low_confidence_not_unreachable():
    payload = generate_sparse_capability_map(
        robot_model_sha256="robot",
        joint_lower=[-1.0],
        joint_upper=[1.0],
        forward_kinematics=lambda joints: ([0.4, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]),
        jacobian=lambda joints: np.eye(1),
        sample_count=8,
        sobol_seed=1,
    )
    capability = SparseCapabilityMap.from_dict(
        payload,
        expected_map_id=payload["map_id"],
        expected_robot_model_sha256="robot",
    )

    missing = capability.lookup(
        {"xyz": [5.0, 5.0, 5.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]}
    )

    assert missing.confidence == 0.0
    assert missing.reachable_density == 0.0


def test_capability_density_is_relative_to_complete_sobol_sample_count():
    payload = generate_sparse_capability_map(
        robot_model_sha256="robot",
        joint_lower=[-1.0],
        joint_upper=[1.0],
        forward_kinematics=lambda joints: (
            [0.0 if joints[0] < 0.0 else 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ),
        jacobian=lambda _joints: [[1.0]],
        sample_count=2,
        sobol_seed=1,
    )

    densities = sorted(
        entry["reachable_density"]
        for entries in payload["cells"].values()
        for entry in entries
    )

    assert densities == [0.5, 0.5]


def test_capability_map_rejects_cross_robot_reuse():
    payload = generate_sparse_capability_map(
        robot_model_sha256="old-robot",
        joint_lower=[-1.0],
        joint_upper=[1.0],
        forward_kinematics=lambda joints: ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        jacobian=lambda joints: np.eye(1),
        sample_count=1,
    )

    with pytest.raises(ValueError, match="robot/TCP/gripper"):
        SparseCapabilityMap.from_dict(
            payload, expected_robot_model_sha256="new-robot"
        )


def test_capability_map_rejects_nonfinite_cell_quality():
    payload = generate_sparse_capability_map(
        robot_model_sha256="robot",
        joint_lower=[-1.0],
        joint_upper=[1.0],
        forward_kinematics=lambda _joints: (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ),
        jacobian=lambda _joints: [[1.0]],
        sample_count=1,
    )
    cell = next(iter(payload["cells"].values()))[0]
    cell["min_singular_value"] = float("nan")
    payload.pop("map_id")
    payload["map_id"] = _canonical_payload_hash(payload)

    with pytest.raises(ValueError, match="cell metrics"):
        SparseCapabilityMap.from_dict(payload)


def test_capability_map_generation_rejects_nonfinite_jacobian():
    with pytest.raises(ValueError, match="Jacobian callback"):
        generate_sparse_capability_map(
            robot_model_sha256="robot",
            joint_lower=[-1.0],
            joint_upper=[1.0],
            forward_kinematics=lambda _joints: (
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ),
            jacobian=lambda _joints: [[float("nan")]],
            sample_count=1,
        )


def test_robot_hash_canonicalizes_equivalent_xml_but_binds_tcp_and_gripper():
    first = robot_model_hash(
        urdf='<robot name="r"><link name="base" /></robot>',
        srdf='<robot name="r"></robot>',
        planning_group="arm",
        tcp="tool0",
        gripper="robotiq",
    )
    equivalent = robot_model_hash(
        urdf='<robot name="r">\n  <link name="base"/>\n</robot>',
        srdf='<robot name="r"/>',
        planning_group="arm",
        tcp="tool0",
        gripper="robotiq",
    )
    changed_tcp = robot_model_hash(
        urdf='<robot name="r"><link name="base"/></robot>',
        srdf='<robot name="r"/>',
        planning_group="arm",
        tcp="new_tool",
        gripper="robotiq",
    )

    assert first == equivalent
    assert changed_tcp != first
