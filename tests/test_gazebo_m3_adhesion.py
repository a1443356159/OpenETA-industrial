from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from extensions.gazebo.adhesion import GazeboM3AdhesionControl
from extensions.gazebo.m3 import (
    AdhesionReceipt,
    AdhesionState,
    M3Verifier,
    ObjectState,
    PhysicsSnapshot,
    Pose,
    ReasonCode,
    coerce_adhesion_receipt,
)
from extensions.gazebo.profiles import gazebo_profile


def _snapshot(stamp: float = 10.0) -> PhysicsSnapshot:
    def object_state(object_id: str, xyz: tuple[float, float, float]) -> ObjectState:
        return ObjectState(
            object_id=object_id,
            name=object_id,
            label=object_id,
            role="target" if object_id == "m3_target" else "distractor",
            pose=Pose(xyz, (0, 0, 0, 1)),
            linear_velocity=(0, 0, 0),
            angular_velocity=(0, 0, 0),
            support="m3_table",
            timestamp_s=stamp,
        )

    streams = ("joint_state", "tf", "rgb", "depth", "odometry_target", "odometry_distractor")
    return PhysicsSnapshot(
        timestamp_s=stamp,
        received_monotonic_s=1.0,
        eef_pose=Pose((0.28, -0.10, 0.55), (0, 0, 0, 1)),
        aperture_m=0.04,
        objects=(
            object_state("m3_target", (0.28, -0.10, 0.43)),
            object_state("m3_distractor", (0.28, 0.12, 0.44)),
        ),
        stream_timestamps_s=tuple((stream, stamp) for stream in streams),
    )


def test_receipt_requires_plugin_identity_and_numeric_receipt_id() -> None:
    receipt = coerce_adhesion_receipt(
        {
            "phase": "CAPTURED",
            "model_name": "m3_target",
            "receipt_id": 42,
            "window_id": 9,
        }
    )
    assert receipt is not None and receipt.capture_accepted
    assert receipt.receipt_id == "42"
    assert receipt.window_id == "9"

    rejected = M3Verifier().verify(
        _snapshot(),
        action_type="gripper_close",
        action_timestamp_s=9.0,
        adhesion_receipt={"phase": "CAPTURED", "model_name": "m3_target"},
    )
    assert rejected.reason_code is ReasonCode.CONTACT_REJECTED


def test_contact_rejection_reasons_never_fall_back_to_geometry() -> None:
    for reason in ("NO_CONTACT", "LEFT_ONLY", "MIXED_OBJECTS", "STALE_CONTACT", "UNKNOWN_ENTITY"):
        record = M3Verifier().verify(
            _snapshot(),
            action_type="gripper_close",
            action_timestamp_s=9.0,
            adhesion_receipt={"phase": "REJECTED", "reason": reason},
        )
        assert (record.verdict.value, record.reason_code) == ("FAIL", ReasonCode.CONTACT_REJECTED)
        assert record.evidence["adhesion_capture"]["reason"] == reason


def test_native_capture_does_not_use_gripper_stall_as_grasp_evidence() -> None:
    record = M3Verifier().verify(
        _snapshot(),
        action_type="gripper_close",
        action_timestamp_s=9.0,
        adhesion_receipt={
            "phase": "CAPTURED",
            "model_name": "m3_target",
            "receipt_id": 1,
            "window_id": 1,
        },
    )
    assert record.reason_code is ReasonCode.LIFT_REQUIRED


def test_adhesion_control_decodes_plugin_schema_and_polls_pending(monkeypatch) -> None:
    encoded = json.dumps(
        {
            "schema": "openeta.m3.adhesion.v1",
            "phase": "CAPTURED",
            "model_name": "m3_target",
            "receipt_id": 3,
            "window_id": 2,
            "reason": "BILATERAL_CONTACT",
        }
    )
    assert GazeboM3AdhesionControl._decode_reply(f"data: {json.dumps(encoded)}")["phase"] == "CAPTURED"

    control = GazeboM3AdhesionControl(timeout_ms=100)
    replies = iter(
        (
            AdhesionReceipt(AdhesionState.ARMED, window_id="2"),
            AdhesionReceipt(AdhesionState.CAPTURED, "m3_target", "3", "2"),
        )
    )
    monkeypatch.setattr(control, "_request", lambda _endpoint: next(replies))
    monkeypatch.setattr("extensions.gazebo.adhesion.time.sleep", lambda _seconds: None)
    captured = control.capture()
    assert captured.capture_accepted and captured.candidate_id == "m3_target"


def test_direct_runtime_has_no_tf_mesh_or_geometric_contact_fallback() -> None:
    source = (
        Path(__file__).parents[1] / "extensions" / "gazebo" / "direct_env.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "tf_buffer",
        "fingertip_collision_bounds_m",
        "PadSnapshot",
        "confirm_pad_contact",
        "select_attachment_object",
    ):
        assert forbidden not in source
    assert "adhesion.arm_contact_window()" in source
    assert "adhesion.capture()" in source
    assert "adhesion.release()" in source


class _Adhesion:
    def __init__(self) -> None:
        self.events: list[str] = []

    def arm_contact_window(self) -> AdhesionReceipt:
        self.events.append("arm")
        return AdhesionReceipt(AdhesionState.ARMED, window_id="1")

    def capture(self) -> AdhesionReceipt:
        self.events.append("capture")
        return AdhesionReceipt(AdhesionState.CAPTURED, "m3_target", "1", "1")


def test_direct_close_arms_then_closes_then_captures_without_tf_fallback() -> None:
    pytest.importorskip("gymnasium")
    from extensions.gazebo.direct_env import GazeboDirectEnv

    adhesion = _Adhesion()
    snapshot = _snapshot()

    class _Runtime:
        deployment = SimpleNamespace()
        physics_source = SimpleNamespace(capture=lambda **_kwargs: snapshot, planning_scene=None)

        def execute(self, _action):
            adhesion.events.append("close")
            return (
                EnvObservation(
                    task="t",
                    cameras=[CameraFrame("top", [[[0, 0, 0]]], [[1.0]], timestamp_s=10.0)],
                    robot=RobotState(
                        end_effector_pose={"xyz": [0.28, -0.1, 0.55], "quat_xyzw": [0, 0, 0, 1]},
                        gripper_state={"aperture_m": 0.04},
                        metadata={"joint_state_timestamp_s": 10.0, "tf_timestamp_s": 10.0},
                    ),
                ),
                {"ok": True, "action_completed_ros_time_s": 9.0},
            )

        def close(self):
            pass

    runtime = _Runtime()
    runtime.adhesion = adhesion
    env = GazeboDirectEnv(profile=gazebo_profile("m3_pickplace"), runtime=runtime)
    observation, _, _, _, info = env.step({"action_type": "gripper_close"})
    assert adhesion.events == ["arm", "close", "capture"]
    assert observation["metadata"]["grasp_mechanism"] == "bilateral_contact_adhesion_v1"
    assert info["_openeta_receipt"]["physical_verification"]["reason_code"] == "LIFT_REQUIRED"
    assert info["_openeta_receipt"]["adhesion"]["receipt_id"] == "1"
