from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapter.protocol import EnvObservation, RobotState
from extensions.gazebo.direct_env import (
    GazeboDirectEnv,
    _detached_target_motion_audit,
)
from extensions.gazebo import direct_env as gazebo_direct_env
from extensions.gazebo.native_grasp import (
    ChildLinkProof,
    NativeContactSample,
    NativeGraspVerifier,
    NativePickPlaceConfig,
    PlacementPoseSample,
    ReasonCode,
    Verdict,
    confirm_native_bilateral_contact,
)
from extensions.gazebo.profiles import STRUCTURED_RECEIPT
from extensions.gazebo.planning_scene import PlanningSceneError
from extensions.gazebo.robotiq_kinematics import AttachedTransportReliefUnavailable


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
        verification_window_started_sim_time_s=10.0,
        now_monotonic_s=20.1,
    )


@pytest.mark.parametrize(
    ("error", "attach_acked", "candidate_rejection", "infrastructure_error", "failure_class"),
    [
        (
            AttachedTransportReliefUnavailable("insufficient relief"),
            True,
            True,
            False,
            "attached_transport_relief_unavailable",
        ),
        (
            RuntimeError("NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"),
            True,
            False,
            True,
            "post_attach_infrastructure_failure",
        ),
        (
            RuntimeError("ATTACH_FAILED"),
            False,
            False,
            False,
            "native_attach_unacknowledged",
        ),
    ],
)
def test_native_close_failure_classification_distinguishes_candidate_geometry(
    error: Exception,
    attach_acked: bool,
    candidate_rejection: bool,
    infrastructure_error: bool,
    failure_class: str,
) -> None:
    assert gazebo_direct_env._native_close_failure_classification(
        error,
        attach_acked=attach_acked,
    ) == (candidate_rejection, infrastructure_error, failure_class)


def test_detached_target_motion_audit_uses_authoritative_compound_radius() -> None:
    source = {
        "shape": "compound",
        "size_xyz": [0.215, 0.062, 0.030],
        "pose_xyz": [0.30, 0.10, 0.42],
        "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "primitives": [
            {
                "shape": "box",
                "size_xyz": [0.165, 0.025, 0.026],
                "pose_xyz": [-0.025, 0.0, 0.0],
                "pose_rpy": [0.0, 0.0, 0.0],
            },
            {
                "shape": "box",
                "size_xyz": [0.055, 0.062, 0.030],
                "pose_xyz": [0.080, 0.0, 0.0],
                "pose_rpy": [0.0, 0.0, 0.0],
            },
        ],
    }

    stationary = _detached_target_motion_audit(
        source_spec=source,
        before_xyz=(0.30, 0.10, 0.42),
        before_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        after_xyz=(0.3002, 0.10, 0.42),
        after_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        physical_tolerance_m=0.001,
    )
    displaced = _detached_target_motion_audit(
        source_spec=source,
        before_xyz=(0.30, 0.10, 0.42),
        before_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        after_xyz=(0.30, 0.10, 0.42),
        after_quat_xyzw=(0.0, 0.0, 0.01, 0.99995),
        physical_tolerance_m=0.001,
    )

    assert stationary["valid"] is True
    assert stationary["target_geometry_radius_m"] > 0.10
    assert displaced["valid"] is False
    assert displaced["failure_phase"] == "moveit_contact_approach"
    assert displaced["maximum_surface_displacement_m"] > 0.002


def test_contact_move_rejects_native_target_displacement_before_close() -> None:
    config = NativePickPlaceConfig()
    observation = EnvObservation(task="pick and place", cameras=[], robot=RobotState())

    class Attachment:
        state = "detached"
        reads = 0

        def native_target_mount_poses_with_retry(self, *, max_attempts):
            assert max_attempts == 2
            self.reads += 1
            xyz = (0.30, 0.10, 0.42) if self.reads == 1 else (0.312, 0.10, 0.42)
            return (
                SimpleNamespace(xyz=xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0)),
                SimpleNamespace(
                    xyz=(0.0, 0.0, 0.6),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                1,
            )

    class Controller:
        planning_scene = SimpleNamespace(
            revision=7,
            world_ids={"work_table", config.target_id},
            attached_ids=set(),
            world_specs={
                "work_table": {
                    "shape": "box",
                    "size_xyz": [1.0, 1.0, 0.05],
                    "pose_xyz": [0.0, 0.0, -0.025],
                    "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                config.target_id: {
                    "shape": "box",
                    "size_xyz": [0.20, 0.05, 0.03],
                    "pose_xyz": [0.30, 0.10, 0.42],
                    "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        )

        def sync_planning_scene_target_pose(
            self,
            _config,
            *,
            target_xyz,
            target_quat_xyzw,
            allow_target_touch=False,
        ):
            assert allow_target_touch is True
            self.planning_scene.world_specs[config.target_id] = {
                **self.planning_scene.world_specs[config.target_id],
                "pose_xyz": list(target_xyz),
                "pose_quat_xyzw": list(target_quat_xyzw),
            }
            self.planning_scene.revision = 8
            return 8

    controller = Controller()
    runtime = SimpleNamespace(
        attachment=Attachment(),
        controller=controller,
        scene_revision=7,
        observe=lambda: observation,
        execute=lambda _action: (
            observation,
            {
                "ok": True,
                "execution_started": True,
                "motion_outcome": "completed",
                "reached_target": True,
            },
        ),
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
    env._native_grasp_transport_locked = False
    env._attachment_transform = None

    _, _, _, _, result = env.step(
        {
            "action_type": "move_to",
            "target_pose": {
                "frame": "world",
                "xyz": [0.30, 0.10, 0.42],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "grasp_stage": "contact",
            },
        }
    )

    receipt = result["_openeta_receipt"]
    assert receipt["ok"] is False
    assert receipt["error_code"] == "GRASP_CONTACT_TARGET_DISPLACED"
    assert receipt["candidate_rejection"] is True
    assert receipt["infrastructure_error"] is False
    assert receipt["detached_target_motion_audit"]["valid"] is False
    assert receipt["planning_scene_target_pose_sync"]["revision"] == 8
    assert receipt["detachable_joint"]["state"] == "detached"


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


def test_invalid_measured_attachment_state_preserves_attach_evidence() -> None:
    verifier = NativeGraspVerifier()

    record = verifier.attachment_state_rejected(
        _accepted_gate(),
        detail=(
            "planning-scene current state is invalid; "
            "collision_pairs=[['blue_bin_wall_left', 'target_object']]"
        ),
    )

    assert record.verdict is Verdict.FAIL
    assert record.reason_code is ReasonCode.ATTACHMENT_STATE_INVALID
    assert record.grasp_confirmed is False
    assert record.evidence["native_attach_acked_before_rollback"] is True
    assert verifier.phase == "attachment_rejected"
    assert verifier.attached is False


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
        verification_window_started_sim_time_s=10.0,
        now_monotonic_s=20.1,
    )
    preclose = confirm_native_bilateral_contact(
        [
            *(_sample("left", stamp) for stamp in (9.90, 10.01, 10.12)),
            *(_sample("right", stamp) for stamp in (10.02, 10.08, 10.13)),
        ],
        verification_window_started_sim_time_s=10.0,
        now_monotonic_s=20.1,
    )

    assert distractor.reason_code is ReasonCode.CONTACT_DISTRACTOR
    assert preclose.reason_code is ReasonCode.CONTACT_SAMPLE_BEFORE_WINDOW


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
        planning_scene = SimpleNamespace(
            revision=7,
            world_ids={"work_table", "target_object"},
            attached_ids=set(),
            world_specs={
                "work_table": {
                    "shape": "box",
                    "pose_xyz": [0.40, 0.0, 0.38],
                    "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "target_object": {
                    "shape": "box",
                    "pose_xyz": [0.28, -0.10, 0.43],
                    "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        )

        def sync_planning_scene_target_pose(
            self,
            _config,
            *,
            target_xyz,
            target_quat_xyzw,
            allow_target_touch=False,
        ):
            assert allow_target_touch is True
            synchronized.append((target_xyz, target_quat_xyzw))
            self.planning_scene.world_specs["target_object"]["pose_xyz"] = list(
                target_xyz
            )
            self.planning_scene.world_specs["target_object"][
                "pose_quat_xyzw"
            ] = list(target_quat_xyzw)
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
            [],
            verification_window_started_sim_time_s=10.0,
            now_monotonic_s=20.0,
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
    assert reopen_receipt["planning_scene_target_pose_sync"][
        "topology_unchanged"
    ] is True
    assert reopen_receipt["planning_scene_target_pose_sync"][
        "static_world_unchanged"
    ] is True
    assert synchronized == [
        ((0.31, -0.08, 0.43), (0.0, 0.0, 0.0, 1.0))
    ]
    assert env._native_grasp_transport_locked is False
    assert raw["metadata"]["planning_scene_revision"] == 8


def test_post_attach_pose_retry_exhaustion_preserves_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = NativePickPlaceConfig()
    observation = EnvObservation(task="pick and place", cameras=[], robot=RobotState())

    class ContactWindow:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def arm():
            return None

        @staticmethod
        def evaluate(**_kwargs):
            return _accepted_gate()

        @staticmethod
        def prove_contact_clearance(**_kwargs):
            return {
                "schema_version": "openeta.native_pad_clearance.v1",
                "cleared": True,
            }

        @staticmethod
        def close():
            return None

    class Attachment:
        state = "detached"
        _last_native_pose_read_attempt_count = 0

        def attach(self):
            self.state = "attached"

        def native_target_mount_poses_with_retry(self, *, max_attempts):
            assert max_attempts == 2
            self._last_native_pose_read_attempt_count = 2
            raise gazebo_direct_env.GazeboProcessError(
                "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
            )

        @staticmethod
        def native_target_mount_poses():
            raise gazebo_direct_env.GazeboProcessError(
                "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
            )

        def ensure_detached(self, *, require_ack):
            assert require_ack is True
            self.state = "detached"

    attachment = Attachment()
    controller = SimpleNamespace(
        planning_scene=SimpleNamespace(revision=7, attached_ids=set())
    )
    runtime = SimpleNamespace(
        attachment=attachment,
        controller=controller,
        scene_revision=7,
        observe=lambda: observation,
        execute=lambda _action: (observation, {"ok": True}),
    )
    env = object.__new__(GazeboDirectEnv)
    env.runtime = runtime
    env.deployment = SimpleNamespace(
        gz_executable="gz", process_environment={}
    )
    env.profile = SimpleNamespace(
        model_config=config,
        cameras=(),
        capabilities={STRUCTURED_RECEIPT},
    )
    env._native_grasp_config = config
    env._native_grasp_verifier = NativeGraspVerifier(config)
    env._native_grasp_transport_locked = False
    env._attachment_transform = None
    monkeypatch.setattr(gazebo_direct_env, "GazeboNativeContactWindow", ContactWindow)

    _, _, _, _, result = env.step({"action_type": "gripper_close"})

    receipt = result["_openeta_receipt"]
    assert receipt["ok"] is False
    assert receipt["error_code"] == "NATIVE_GRASP_CHILD_LINK_STATE_UNAVAILABLE"
    assert receipt["infrastructure_error"] is True
    assert receipt["attach_acked_before_rollback"] is True
    assert receipt["native_state_snapshot"] == {
        "post_attach_attempt_count": 2,
        "baseline_attempt_count": 0,
        "maximum_attempts_per_read": 2,
        "retry_exhausted": True,
    }
    assert receipt["detachable_joint"]["state"] == "detached"
    assert attachment.state == "detached"


def test_measured_attachment_collision_is_candidate_rejection_then_pose_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = NativePickPlaceConfig()
    observation = EnvObservation(task="pick and place", cameras=[], robot=RobotState())

    class ContactWindow:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def arm():
            return None

        @staticmethod
        def evaluate(**_kwargs):
            return _accepted_gate()

        @staticmethod
        def prove_contact_clearance(**_kwargs):
            return {
                "schema_version": "openeta.native_pad_clearance.v1",
                "cleared": True,
            }

        @staticmethod
        def close():
            return None

    class Attachment:
        state = "detached"

        def attach(self):
            self.state = "attached"

        @staticmethod
        def native_target_mount_poses():
            return (
                SimpleNamespace(
                    xyz=(0.291, -0.151, 0.430),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                SimpleNamespace(
                    xyz=(0.300, -0.080, 0.550),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            )

        def ensure_detached(self, *, require_ack):
            assert require_ack is True
            self.state = "detached"

    class Controller:
        planning_scene = SimpleNamespace(
            revision=7,
            world_ids={"work_table", "blue_bin_wall_left", "target_object"},
            attached_ids=set(),
            world_specs={
                "work_table": {
                    "shape": "box",
                    "pose_xyz": [0.40, 0.0, 0.38],
                    "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "blue_bin_wall_left": {
                    "shape": "box",
                    "pose_xyz": [0.30, -0.16, 0.48],
                    "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "target_object": {
                    "shape": "box",
                    "pose_xyz": [0.28, -0.10, 0.43],
                    "pose_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        )

        @staticmethod
        def establish_attached_transport_hold():
            return {
                "schema_version": "openeta.attached_transport_hold.v1",
                "actuator_model": "single_common_driver",
                "object_environment_collision_enabled": True,
                "minimum_proven_relief_rad": 0.02,
                "measured_common_before_rad": 0.60,
                "measured_common_after_rad": 0.57,
                "action_completed_ros_time_s": 12.0,
            }

        def sync_planning_scene_attach(self, *_args, **_kwargs):
            self.planning_scene.world_ids.remove("target_object")
            self.planning_scene.attached_ids.add("target_object")
            self.planning_scene.revision = 8
            raise PlanningSceneError(
                "planning-scene current state is invalid; "
                "collision_pairs=[['blue_bin_wall_left', 'target_object']]"
            )

        def sync_planning_scene_detach(
            self, _config, *, target_xyz, target_quat_xyzw
        ):
            self.planning_scene.attached_ids.clear()
            self.planning_scene.world_ids.add("target_object")
            self.planning_scene.world_specs["target_object"] = {
                "shape": "box",
                "pose_xyz": list(target_xyz),
                "pose_quat_xyzw": list(target_quat_xyzw),
            }
            self.planning_scene.revision = 9
            return 9

        def sync_planning_scene_target_pose(
            self,
            _config,
            *,
            target_xyz,
            target_quat_xyzw,
            allow_target_touch=False,
        ):
            self.planning_scene.world_specs["target_object"]["pose_xyz"] = list(
                target_xyz
            )
            self.planning_scene.world_specs["target_object"][
                "pose_quat_xyzw"
            ] = list(target_quat_xyzw)
            self.planning_scene.revision = 10
            return 10

    attachment = Attachment()
    controller = Controller()
    runtime = SimpleNamespace(
        attachment=attachment,
        controller=controller,
        scene_revision=7,
        observe=lambda: observation,
        execute=lambda _action: (observation, {"ok": True}),
    )
    env = object.__new__(GazeboDirectEnv)
    env.runtime = runtime
    env.deployment = SimpleNamespace(gz_executable="gz", process_environment={})
    env.profile = SimpleNamespace(
        model_config=config,
        cameras=(),
        capabilities={STRUCTURED_RECEIPT},
    )
    env._native_grasp_config = config
    env._native_grasp_verifier = NativeGraspVerifier(config)
    env._native_grasp_transport_locked = False
    env._attachment_transform = None
    monkeypatch.setattr(gazebo_direct_env, "GazeboNativeContactWindow", ContactWindow)

    _, _, _, _, close_result = env.step({"action_type": "gripper_close"})

    close_receipt = close_result["_openeta_receipt"]
    assert close_receipt["ok"] is False
    assert close_receipt["infrastructure_error"] is False
    assert close_receipt["candidate_rejection"] is True
    assert close_receipt["failure_class"] == "measured_attachment_collision"
    assert close_receipt["attach_acked_before_rollback"] is True
    assert close_receipt["attached_transport_hold"][
        "object_environment_collision_enabled"
    ] is True
    assert close_receipt["attached_transport_hold"]["schema_version"] == (
        "openeta.attached_transport_hold.v2"
    )
    assert close_receipt["error_code"] == ReasonCode.ATTACHMENT_STATE_INVALID.value
    assert close_receipt["planning_scene_rollback"] == {
        "state": "detached",
        "revision": 9,
    }
    rollback_sync = close_receipt["planning_scene_target_pose_sync"]
    assert rollback_sync["source_revision"] == 7
    assert rollback_sync["revision"] == 9
    assert rollback_sync["target_id"] == "target_object"
    assert rollback_sync["topology_unchanged"] is True
    assert rollback_sync["static_world_unchanged"] is True
    assert rollback_sync["execution_started"] is True
    assert close_receipt["physical_verification"]["reason_code"] == (
        ReasonCode.ATTACHMENT_STATE_INVALID.value
    )
    assert attachment.state == "detached"

    _, _, _, _, open_result = env.step({"action_type": "gripper_open"})

    open_receipt = open_result["_openeta_receipt"]
    assert open_receipt["ok"] is True
    assert open_receipt["planning_scene_revision"] == 10
    assert open_receipt["planning_scene_target_pose_sync"]["target_id"] == (
        "target_object"
    )
    assert open_receipt["planning_scene_target_pose_sync"][
        "static_world_unchanged"
    ] is True
    assert env._native_grasp_transport_locked is False


def _attached_release_env(*, detach_fails: bool = False):
    config = NativePickPlaceConfig()
    released_xyz = (
        config.destination_center_xy[0],
        config.destination_center_xy[1],
        config.destination_support_z_m + config.target_size_m[2] / 2.0,
    )
    events: list[str] = []
    observation = EnvObservation(
        task="pick and place",
        cameras=[],
        robot=RobotState(),
    )

    class Attachment:
        state = "attached"

        @staticmethod
        def native_target_mount_poses():
            return (
                SimpleNamespace(
                    xyz=released_xyz,
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                SimpleNamespace(
                    xyz=(released_xyz[0], released_xyz[1], released_xyz[2] + 0.12),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            )

        def ensure_detached(self, *, require_ack):
            assert require_ack is True
            events.append("native_detach_ack")
            if detach_fails:
                raise RuntimeError("detach ACK unavailable")
            self.state = "detached"

        @staticmethod
        def sample_detached_target_poses(*, duration_s, interval_s):
            assert duration_s >= 0.5
            assert interval_s > 0.0
            events.append("sample_released_target")
            return [
                PlacementPoseSample(
                    monotonic_s=1.0,
                    xyz=released_xyz,
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                PlacementPoseSample(
                    monotonic_s=1.2,
                    xyz=released_xyz,
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                PlacementPoseSample(
                    monotonic_s=1.6,
                    xyz=released_xyz,
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            ]

    class Controller:
        planning_scene = SimpleNamespace(revision=7)

        def sync_planning_scene_detach(
            self, _config, *, target_xyz, target_quat_xyzw
        ):
            assert target_xyz == released_xyz
            assert target_quat_xyzw == (0.0, 0.0, 0.0, 1.0)
            events.append("planning_scene_detach_ack")
            self.planning_scene.revision = 8
            return 8

        def sync_planning_scene_target_pose(
            self,
            _config,
            *,
            target_xyz,
            target_quat_xyzw,
            allow_target_touch=False,
        ):
            assert target_xyz == released_xyz
            assert target_quat_xyzw == (0.0, 0.0, 0.0, 1.0)
            assert allow_target_touch is True
            events.append("planning_scene_pose_sync_ack")
            self.planning_scene.revision = 9
            return 9

    attachment = Attachment()

    def execute(action):
        assert action == {"action_type": "gripper_open"}
        events.append("gripper_open")
        return observation, {"ok": True}

    runtime = SimpleNamespace(
        attachment=attachment,
        controller=Controller(),
        scene_revision=7,
        observe=lambda: observation,
        execute=execute,
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
    env._native_grasp_verifier.close_result(_accepted_gate(), attach_acked=True)
    env._native_grasp_transport_locked = False
    env._attachment_transform = {
        "schema_version": "openeta.attachment_transform.v1"
    }
    return env, events


def test_release_ack_triggers_open_while_planning_scene_detach_runs() -> None:
    env, events = _attached_release_env()

    raw, _, _, _, result = env.step({"action_type": "gripper_open"})

    receipt = result["_openeta_receipt"]
    assert receipt["ok"] is True
    assert events[0] == "native_detach_ack"
    assert set(events[1:3]) == {"gripper_open", "planning_scene_detach_ack"}
    assert events[3:] == [
        "sample_released_target",
        "planning_scene_pose_sync_ack",
    ]
    assert [item["event"] for item in receipt["release_sequence"]] == [
        "native_detach_ack",
        "planning_scene_detach_ack",
        "gripper_open_completed",
        "released_target_pose_sync_ack",
    ]
    assert receipt["gripper_open_executed"] is True
    assert receipt["release_coordination"] == {
        "schema_version": "openeta.native_release_coordination.v1",
        "mode": "detach_confirmation_triggers_open",
        "native_detach_confirmed_before_open_dispatch": True,
        "planning_scene_sync_concurrent_with_open_dispatch": True,
    }
    assert receipt["placement_verification"]["verdict"] == "PASS"
    assert raw["metadata"]["planning_scene_revision"] == 9


def test_irreversible_release_proof_survives_later_scene_sync_failure() -> None:
    env, events = _attached_release_env()

    def fail_pose_sync(
        _config,
        *,
        target_xyz,
        target_quat_xyzw,
        allow_target_touch=False,
    ):
        del _config, target_xyz, target_quat_xyzw, allow_target_touch
        events.append("planning_scene_pose_sync_failed")
        # TimeoutError without a message reproduces the empty error code that
        # previously hid the actual post-release failure stage.
        raise TimeoutError

    env.runtime.controller.sync_planning_scene_target_pose = fail_pose_sync

    _, _, _, _, result = env.step({"action_type": "gripper_open"})

    receipt = result["_openeta_receipt"]
    assert receipt["ok"] is False
    assert receipt["gripper_open_executed"] is True
    assert receipt["infrastructure_error"] is True
    assert receipt["post_release_failure_stage"] == "released_target_pose_sync"
    assert receipt["error_code"] == "TimeoutError"
    assert receipt["error_type"] == "TimeoutError"
    assert receipt["placement_verification"]["verdict"] == "PASS"
    assert [item["event"] for item in receipt["release_sequence"]] == [
        "native_detach_ack",
        "planning_scene_detach_ack",
        "gripper_open_completed",
    ]
    assert events[0] == "native_detach_ack"
    assert set(events[1:3]) == {"gripper_open", "planning_scene_detach_ack"}
    assert events[3:] == [
        "sample_released_target",
        "planning_scene_pose_sync_failed",
    ]


def test_release_detach_failure_forbids_gripper_open() -> None:
    env, events = _attached_release_env(detach_fails=True)

    _, _, _, _, result = env.step({"action_type": "gripper_open"})

    receipt = result["_openeta_receipt"]
    assert receipt["ok"] is False
    assert receipt["gripper_open_executed"] is False
    assert events == ["native_detach_ack"]
