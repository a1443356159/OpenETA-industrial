from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapter.protocol import EnvObservation, RobotState
from extensions.gazebo.direct_env import GazeboDirectEnv
from extensions.gazebo.native_grasp import (
    ChildLinkProof,
    NativeContactSample,
    NativeGraspVerifier,
    NativePickPlaceConfig,
    ReasonCode,
    Verdict,
    confirm_native_bilateral_contact,
)
from extensions.gazebo.profiles import STRUCTURED_RECEIPT


def _sample(side: str, stamp: float) -> NativeContactSample:
    return NativeContactSample(
        side=side,
        timestamp_s=stamp,
        received_monotonic_s=20.0,
        collision_names=(
            f"rm75::robotiq_85_{side}_finger_tip_link",
            "target_object::target_link::target_collision",
        ),
    )


def _accepted_gate():
    return confirm_native_bilateral_contact(
        [
            *(_sample("left", stamp) for stamp in (10.01, 10.07, 10.12)),
            *(_sample("right", stamp) for stamp in (10.02, 10.08, 10.13)),
        ],
        close_completed_sim_time_s=10.0,
        now_monotonic_s=20.1,
    )


def test_native_contact_and_attach_ack_directly_prove_grasp() -> None:
    gate = _accepted_gate()
    assert gate.accepted
    assert gate.reason_code is ReasonCode.CONTACT_TARGET_CONFIRMED

    record = NativeGraspVerifier().close_result(gate, attach_acked=True)

    assert record.verdict is Verdict.PASS
    assert record.reason_code is ReasonCode.ATTACHMENT_CONFIRMED
    assert record.grasp_confirmed is True
    assert record.evidence["proof_boundary"] == (
        "bilateral_native_contact_and_attach_ack"
    )


def test_native_contact_rejects_distractor_and_preclose_samples() -> None:
    distractor = confirm_native_bilateral_contact(
        [
            NativeContactSample(
                "left",
                10.01,
                20.0,
                ("left_tip", "distractor_object::distractor_link"),
            ),
            *(_sample("left", stamp) for stamp in (10.07, 10.12)),
            *(_sample("right", stamp) for stamp in (10.02, 10.08, 10.13)),
        ],
        close_completed_sim_time_s=10.0,
        now_monotonic_s=20.1,
    )
    preclose = confirm_native_bilateral_contact(
        [
            *(_sample("left", stamp) for stamp in (9.90, 10.01, 10.12)),
            *(_sample("right", stamp) for stamp in (10.02, 10.08, 10.13)),
        ],
        close_completed_sim_time_s=10.0,
        now_monotonic_s=20.1,
    )

    assert distractor.reason_code is ReasonCode.CONTACT_DISTRACTOR
    assert preclose.reason_code is ReasonCode.CONTACT_SAMPLE_BEFORE_CLOSE


def test_transport_retention_has_no_direction_or_distance_threshold() -> None:
    verifier = NativeGraspVerifier()
    verifier.close_result(_accepted_gate(), attach_acked=True)

    moved_down = verifier.prove_retention(ChildLinkProof(0.43, 0.39, 0.003))
    excessive_drift = verifier.prove_retention(ChildLinkProof(0.43, 0.80, 0.011))

    assert moved_down.verdict is Verdict.PASS
    assert moved_down.reason_code is ReasonCode.TARGET_HELD
    assert moved_down.evidence["vertical_displacement_m"] == pytest.approx(-0.04)
    assert excessive_drift.verdict is Verdict.FAIL
    assert excessive_drift.reason_code is ReasonCode.RELATIVE_POSE_DRIFT


def test_retention_without_attach_ack_fails_closed() -> None:
    record = NativeGraspVerifier().prove_retention(
        ChildLinkProof(0.43, 0.43, 0.0)
    )

    assert record.verdict is Verdict.FAIL
    assert record.reason_code is ReasonCode.ATTACH_ACK_MISSING


def _failed_close_env():
    config = NativePickPlaceConfig()
    synchronized: list[tuple[object, object]] = []
    observation = EnvObservation(
        task="pick and place",
        cameras=[],
        robot=RobotState(),
    )

    class Attachment:
        state = "detached"

        @staticmethod
        def native_target_mount_poses():
            return (
                SimpleNamespace(
                    xyz=(0.31, -0.08, 0.43),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                SimpleNamespace(
                    xyz=(0.30, -0.08, 0.55),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            )

    class Controller:
        planning_scene = SimpleNamespace(revision=7)

        def sync_planning_scene_target_pose(
            self, _config, *, target_xyz, target_quat_xyzw
        ):
            synchronized.append((target_xyz, target_quat_xyzw))
            self.planning_scene.revision = 8
            return 8

    runtime = SimpleNamespace(
        attachment=Attachment(),
        controller=Controller(),
        scene_revision=7,
        observe=lambda: observation,
        execute=lambda _action: (observation, {"ok": True}),
    )
    env = object.__new__(GazeboDirectEnv)
    env.runtime = runtime
    env.profile = SimpleNamespace(
        model_config=config,
        cameras=(),
        capabilities={STRUCTURED_RECEIPT},
    )
    env._native_grasp_config = config
    env._native_grasp_verifier = NativeGraspVerifier(config)
    env._native_grasp_verifier.close_result(
        confirm_native_bilateral_contact(
            [], close_completed_sim_time_s=10.0, now_monotonic_s=20.0
        ),
        attach_acked=False,
    )
    env._native_grasp_transport_locked = True
    env._attachment_transform = None
    return env, synchronized


def test_failed_close_forbids_motion_and_requires_exact_reopen() -> None:
    env, synchronized = _failed_close_env()

    _, _, _, _, blocked = env.step(
        {"action_type": "move_to", "target_pose": {"frame": "world", "xyz": [0, 0, 0]}}
    )
    raw, _, _, _, reopened = env.step({"action_type": "gripper_open"})

    blocked_receipt = blocked["_openeta_receipt"]
    reopen_receipt = reopened["_openeta_receipt"]
    assert blocked_receipt["ok"] is False
    assert blocked_receipt["error_code"] == ReasonCode.ATTACH_ACK_MISSING.value
    assert blocked_receipt["native_grasp_recovery_diagnostic"][
        "required_recovery"
    ] == "gripper_open"
    assert reopen_receipt["ok"] is True
    assert reopen_receipt["planning_scene_revision"] == 8
    assert synchronized == [
        ((0.31, -0.08, 0.43), (0.0, 0.0, 0.0, 1.0))
    ]
    assert env._native_grasp_transport_locked is False
    assert raw["metadata"]["planning_scene_revision"] == 8
