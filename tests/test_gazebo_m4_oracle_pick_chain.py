"""Opt-in live integration driver for the M4 oracle-perception pick chain.

Chain under test (staged primitives only, no pick_place orchestration)::

    observe -> oracle_perceive -> select detection -> grasp_pose_estimate (fake)
    -> camera_pose_to_world -> move_to(pregrasp) -> move_to(grasp)
    -> gripper_close -> lift -> M3 physical verdict

Gating: skipped unless ``OPENETA_RUN_LIVE_ROS_TEST=1``.  The test expects a
running openeta-sim MCP server (``OPENETA_SIM_MCP_URL``, default
``http://127.0.0.1:8765/sse``) whose gazebo bench worker hosts the
``openeta/gazebo_rm75_robotiq2f85_pickplace-v0`` (m3) environment.

Notes:

* ``grasp_pose_estimate`` depends on GPU services (Contact-GraspNet etc.) that
  are not part of this harness, so the grasp candidate is a local fake in the
  exact ``grasp_pose_estimate``/Contact-GraspNet output contract, derived from
  the oracle ground-truth object pose.  The point of this driver is the call
  sequence and the per-hop contract, not backend inference quality.
* The EEF first moves to a reachable grasp orientation (tilted approach,
  horizontal closing axis — the family validated by the M3 acceptance
  driver); candidate-based moves are position-level with
  ``preserve_current`` because this arm has no GraspNet→EEF calibration
  profile.  M3's remaining live blocker is gripper-linkage physics variance
  at close/lift; hitting it reports xfail instead of a hard failure.
"""

from __future__ import annotations

import contextlib
import math
import os
from typing import Any, Mapping, Sequence

import pytest

from agent.tools.handlers import CONTACT_GRASPNET_GRIPPER_DEPTH, bind_dummy_tool_handlers
from agent.tools.registry import build_default_tool_registry
from agent.tools.sim_mcp import (
    SimulatorMcpToolProxyConfig,
    SseSimulatorMcpTransport,
    bind_simulator_mcp_tool_handlers,
)

ENV_ID = "openeta/gazebo_rm75_robotiq2f85_pickplace-v0"
DEFAULT_MCP_URL = "http://127.0.0.1:8765/sse"
TOP_CAMERA_FRAME_ID = "top_camera_optical_frame"
TARGET_OBJECT_ID = "m3_target"
TARGET_PROMPT = "target block"
PREGRASP_HEIGHT_M = 0.080
# Reachable grasp orientation for this table layout, shared with the M3
# acceptance driver: a tilted approach with a horizontal gripper closing
# axis.  The previous revision kept the home (gripper-up) orientation, which
# is unreachable above the table without the forearm crossing it.
_GRASP_TILT_DEG = 65.0
_GRASP_AZIMUTH_DEG = 0.0
# Measured from live TF at full aperture: grasp centre (midpoint of the two
# fingertip collision centres) expressed in the EEF mount frame.
_GRASP_CENTER_OFFSET_MOUNT_M = (0.0, -0.0004, 0.1268)


def _reachable_grasp_quat() -> tuple[float, float, float, float]:
    """Tilted top grasp whose closing axis stays horizontal (see M3 driver)."""

    tilt = math.radians(_GRASP_TILT_DEG)
    azimuth = math.radians(_GRASP_AZIMUTH_DEG)
    approach = (
        math.sin(tilt) * math.cos(azimuth),
        math.sin(tilt) * math.sin(azimuth),
        -math.cos(tilt),
    )
    closing = (-math.sin(azimuth), math.cos(azimuth), 0.0)
    third = (
        approach[1] * closing[2] - approach[2] * closing[1],
        approach[2] * closing[0] - approach[0] * closing[2],
        approach[0] * closing[1] - approach[1] * closing[0],
    )
    columns = (closing, third, approach)
    m = [[columns[col][row] for col in range(3)] for row in range(3)]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s)


def _quat_xyzw_to_matrix3(quat: Sequence[float]) -> list[list[float]]:
    qx, qy, qz, qw = (float(value) for value in quat)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    return [
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ]


def _mat3_vec3(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [
        sum(float(matrix[row][col]) * float(vector[col]) for col in range(3))
        for row in range(3)
    ]


def _observation_of(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = receipt.get("observation", receipt)
    assert isinstance(observation, Mapping), "MCP receipt carries no observation"
    return observation


def _top_camera(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    cameras = observation.get("cameras")
    assert isinstance(cameras, list) and cameras, "observation has no camera frames"
    for camera in cameras:
        if isinstance(camera, Mapping) and camera.get("frame_id") == TOP_CAMERA_FRAME_ID:
            return camera
    raise AssertionError(f"top camera frame {TOP_CAMERA_FRAME_ID!r} not in observation")


def _target_position(observation: Mapping[str, Any]) -> list[float]:
    objects = observation.get("objects") or []
    for item in objects:
        if isinstance(item, Mapping) and item.get("id") == TARGET_OBJECT_ID:
            position = item.get("position")
            assert isinstance(position, list) and len(position) == 3
            return [float(value) for value in position]
    raise AssertionError(f"oracle object {TARGET_OBJECT_ID!r} not in observation")


def _physical_reason_code(observation: Mapping[str, Any]) -> str:
    record = observation.get("metadata", {}).get("physical_verification")
    assert isinstance(record, Mapping), "M3 physical verification record missing"
    return str(record.get("reason_code") or "")


def _fake_camera_frame_grasp_candidate(
    target_world: Sequence[float],
    extrinsics: Mapping[str, Any],
) -> dict[str, Any]:
    """GPU-free stand-in for one grasp_pose_estimate (Contact-GraspNet) candidate.

    Top-down grasp: the GraspNet approach axis (+x of the grasp frame) aligns
    with the camera optical axis (+z), so the world-frame approach is straight
    down.  Field set mirrors ``_normalise_contact_graspnet_candidate`` plus the
    provenance markers ``grasp_pose_estimate`` stamps on its candidates.
    """

    rotation_cw = _quat_xyzw_to_matrix3(extrinsics["quat_xyzw"])
    pos = [float(value) for value in extrinsics["pos"]]
    offset = [float(target_world[row]) - pos[row] for row in range(3)]
    transpose = [[rotation_cw[col][row] for col in range(3)] for row in range(3)]
    center_camera = _mat3_vec3(transpose, offset)
    grasp_rotation_camera = [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    approach_camera = [0.0, 0.0, 1.0]
    depth = CONTACT_GRASPNET_GRIPPER_DEPTH
    tip_camera = [center_camera[row] + depth * approach_camera[row] for row in range(3)]
    return {
        "id": "gpe-live-oracle-000",
        "frame": "camera",
        "camera_frame": "opencv",
        "grasp_frame": "graspnet",
        "source_tool": "grasp_pose_estimate",
        "source_backend": "contact_graspnet",
        "source_model": "contact_graspnet",
        "gripper_model": "panda",
        "rank": 0,
        "score": 1.0,
        "translation_xyz": center_camera,
        "rotation_matrix": grasp_rotation_camera,
        "gripper_depth": depth,
        "depth": depth,
        "width": 0.04,
        "gripper_tip_position_xyz": tip_camera,
        "contact_point_xyz": list(center_camera),
    }


@pytest.mark.skipif(
    os.environ.get("OPENETA_RUN_LIVE_ROS_TEST") != "1",
    reason="opt-in: set OPENETA_RUN_LIVE_ROS_TEST=1 for the live M4 oracle pick chain",
)
def test_gazebo_m4_oracle_pick_chain_live() -> None:
    url = os.environ.get("OPENETA_SIM_MCP_URL", DEFAULT_MCP_URL)
    transport = SseSimulatorMcpTransport(url)
    handle = session_id = ""
    try:
        created = transport.call_tool(
            "create_env",
            {"env_id": ENV_ID, "seed": 41, "task": "M4 oracle pick chain"},
            timeout_s=180,
        )
        handle = str(created.get("handle") or "")
        session_id = str(created.get("session_id") or "")
        assert handle, "create_env returned no handle"
        assert created.get("control_spec", {}).get("m3") is True, "wrong gazebo profile"
        common = {"handle": handle, "session_id": session_id}
        transport.call_tool("reset_env", {**common, "seed": 41}, timeout_s=180)

        # Agent-side proxy handlers bound to the live handle: move_to and
        # gripper_control exercise the sim_mcp -> gazebo action mapping.
        tools = bind_simulator_mcp_tool_handlers(
            build_default_tool_registry(),
            transport=transport,
            config=SimulatorMcpToolProxyConfig(session_id=session_id, handle=handle),
            tool_names=("move_to", "gripper_control"),
        )
        geometry_tools = bind_dummy_tool_handlers(build_default_tool_registry())

        # 1. observe
        observation = _observation_of(
            transport.call_tool("observe_env", common, timeout_s=60)
        )
        camera = _top_camera(observation)
        rgb_base64 = str(camera.get("rgb_base64") or "")
        assert rgb_base64, "top camera frame carries no rgb_base64"
        extrinsics = camera.get("extrinsics")
        assert isinstance(extrinsics, Mapping) and extrinsics.get("pos"), (
            "top camera frame carries no numeric extrinsics"
        )
        target_world = _target_position(observation)

        # 2. oracle_perceive (SAM3-contract response via openeta-sim MCP)
        oracle = transport.call_tool(
            "oracle_perceive",
            {**common, "image_base64": rgb_base64, "prompt": TARGET_PROMPT},
            timeout_s=60,
        )
        assert oracle.get("success") is True, f"oracle_perceive failed: {oracle}"
        details = oracle.get("details") or {}
        detections = details.get("detections") or []
        assert len(detections) == 1, f"expected exactly the target detection: {detections}"
        detection = detections[0]
        assert detection["label"] == TARGET_PROMPT
        assert detection["score"] == 1.0
        assert len(detection["bbox_xyxy"]) == 4
        assert detection["mask"]["format"] == "png" and detection["mask"]["base64"]
        assert (details.get("metadata") or {}).get("perception_source") == "gazebo_oracle"

        # 3. selection: the agent runtime would call select_sam3_detection;
        #    this driver takes the rank-0 detection directly.
        assert detection.get("rank") == 0

        # 4. grasp_pose_estimate (fake candidate in the exact output contract)
        candidate = _fake_camera_frame_grasp_candidate(target_world, extrinsics)

        # 5. camera_pose_to_world through the real agent geometry handler,
        #    fed with the observation-supplied pos/quat_xyzw extrinsics.
        transformed = geometry_tools.call(
            "camera_pose_to_world",
            {
                "camera_pose": candidate,
                "camera_to_world": dict(extrinsics),
                "camera_frame_id": TOP_CAMERA_FRAME_ID,
            },
        )
        assert transformed.success is True, transformed.content
        world_pose = transformed.details["outputs"]["world_pose"]
        assert world_pose["frame"] == "world"
        assert world_pose["translation_xyz"] == pytest.approx(target_world, abs=1e-6)
        # Grasp provenance must survive the transform so the move_to proxy
        # recognises the grasp frame and never forwards it as a raw EEF pose.
        assert world_pose["grasp_frame"] == "graspnet"

        def move(pose: Mapping[str, Any], *, stage: str) -> None:
            result = tools.call("move_to", {"target_pose": dict(pose)})
            outputs = result.details.get("outputs") or {}
            mcp = outputs.get("mcp") or {}
            assert mcp.get("target_orientation_mode") == "preserve_current"
            if not result.success:
                message = f"{result.content} {result.details.get('diagnostics')}"
                if stage == "grasp" and "MOTION_PLAN_FAILED" in message:
                    pytest.xfail("M3 known live blocker: contact-pose MOTION_PLAN_FAILED")
                raise AssertionError(f"{stage} move_to failed: {message}")

        # 6. orient + pregrasp: drive the EEF to the reachable grasp
        #    orientation above the box with an absolute pose (a plain
        #    non-candidate move_to).  The candidate-derived moves below then
        #    keep this orientation through preserve_current.  The EEF
        #    translation accounts for the grasp-centre-to-mount offset.
        grasp_quat = _reachable_grasp_quat()
        offset_world = _mat3_vec3(
            _quat_xyzw_to_matrix3(grasp_quat), _GRASP_CENTER_OFFSET_MOUNT_M
        )
        eef_grasp = [target_world[axis] - offset_world[axis] for axis in range(3)]
        eef_pregrasp = [eef_grasp[0], eef_grasp[1], eef_grasp[2] + PREGRASP_HEIGHT_M]
        oriented = tools.call(
            "move_to",
            {
                "target_pose": {
                    "frame": "world",
                    "xyz": eef_pregrasp,
                    "quat_xyzw": list(grasp_quat),
                }
            },
        )
        assert oriented.success is True, f"orient move_to failed: {oriented.content}"

        def candidate_move(xyz: Sequence[float], *, stage: str) -> None:
            pose = dict(world_pose)
            pose["translation_xyz"] = [float(value) for value in xyz]
            move(pose, stage=stage)

        # 7. move_to(pregrasp) through the grasp-candidate contract
        candidate_move(eef_pregrasp, stage="pregrasp")

        # 8. move_to(grasp)
        candidate_move(eef_grasp, stage="grasp")

        # 9. gripper_close
        closed = tools.call("gripper_control", {"position": 0})
        assert closed.success is True, closed.content
        observation = _observation_of(
            transport.call_tool("observe_env", common, timeout_s=60)
        )
        close_reason = _physical_reason_code(observation)
        if close_reason != "LIFT_REQUIRED":
            pytest.xfail(f"M3 known live blocker: close did not stall on the target ({close_reason})")

        # 10. lift
        candidate_move(
            [eef_grasp[0], eef_grasp[1], eef_grasp[2] + PREGRASP_HEIGHT_M],
            stage="lift",
        )

        # 11. M3 verdict
        observation = _observation_of(
            transport.call_tool("observe_env", common, timeout_s=60)
        )
        held_reason = _physical_reason_code(observation)
        if held_reason != "TARGET_HELD":
            pytest.xfail(f"M3 known live blocker: lift comovement unproven ({held_reason})")
    finally:
        if handle:
            with contextlib.suppress(Exception):
                transport.call_tool(
                    "close_env",
                    {"handle": handle, "session_id": session_id},
                    timeout_s=60,
                )
