from __future__ import annotations

from types import MethodType, SimpleNamespace

from extensions.gazebo.m2 import M2ControlResult
from extensions.gazebo.m3 import (
    ContactState,
    M3Config,
    ObjectState,
    PhysicsSnapshot,
    Pose,
    ReasonCode,
)
from extensions.gazebo.ros_control import gripper_action_success
from extensions.gazebo import worker


def _snapshot(stamp: float = 12.0) -> PhysicsSnapshot:
    target = ObjectState(
        object_id="m3_target",
        name="m3_target",
        label="target block",
        role="target",
        pose=Pose((0.28, -0.10, 0.43), (0.0, 0.0, 0.0, 1.0)),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        support="m3_table",
        timestamp_s=stamp,
    )
    distractor = ObjectState(
        object_id="m3_distractor",
        name="m3_distractor",
        label="distractor cylinder",
        role="distractor",
        pose=Pose((0.28, 0.12, 0.44), (0.0, 0.0, 0.0, 1.0)),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        support=None,
        timestamp_s=stamp,
    )
    streams = (
        "joint_state", "tf", "rgb", "depth", "contact_left", "contact_right",
        "contact_target", "odometry_target", "odometry_distractor",
    )
    return PhysicsSnapshot(
        timestamp_s=stamp,
        received_monotonic_s=100.0,
        eef_pose=Pose((0.0, 0.0, 0.916), (0.0, 0.0, 0.0, 1.0)),
        aperture_m=0.085,
        contacts=ContactState(
            left_object_ids=(), right_object_ids=(), target_support_ids=("m3_table",),
            left_durations_s=(), right_durations_s=(),
            timestamps_s=(("left", stamp), ("right", stamp), ("target", stamp)),
        ),
        objects=(target, distractor),
        stream_timestamps_s=tuple((name, stamp) for name in streams),
        gripper_stalled=False,
        gripper_reached_goal=True,
    )


class _Physics:
    def __init__(self):
        self.calls = []
        self.planning_scene = SimpleNamespace(attached=False)

    def capture(self, **kwargs):
        self.calls.append(kwargs)
        return _snapshot()


def _raw_observation():
    return {
        "task": "",
        "cameras": {
            "top": {"timestamp_s": 12.0},
            "wrist": {"timestamp_s": 12.0},
        },
        "robot": {
            "end_effector_pose": {"xyz": [0.0, 0.0, 0.916], "quat_xyzw": [0, 0, 0, 1]},
            "gripper_state": {"aperture_m": 0.085},
            "metadata": {"joint_state_timestamp_s": 12.0, "tf_timestamp_s": 12.0},
        },
        "objects": [],
        "metadata": {},
    }


def test_m3_merge_adds_truth_objects_contacts_and_one_shared_verification_record() -> None:
    environment = object.__new__(worker.GazeboM3WorkerEnv)
    environment._m3_config = M3Config()
    environment._verifier = worker.M3Verifier(environment._m3_config)
    environment._physics_source = _Physics()
    environment._last_snapshot = None
    environment._latest = None

    observation, snapshot = environment._merge_physics(_raw_observation())

    assert snapshot is not None
    assert [item["id"] for item in observation["objects"]] == [
        "m3_target", "m3_distractor",
    ]
    gripper = observation["robot"]["gripper_state"]
    assert gripper["contact_left"] is False and gripper["contact_right"] is False
    assert gripper["grasp_confirmed"] is False
    physical = observation["metadata"]["physical_verification"]
    assert physical["schema_version"] == "m3_physical_verification_v1"
    assert physical["reason_code"] == ReasonCode.READY.value


def test_m3_step_receipt_reuses_the_observation_physical_record_and_fresh_barrier() -> None:
    environment = object.__new__(worker.GazeboM3WorkerEnv)
    environment._m3_config = M3Config()
    environment._verifier = worker.M3Verifier(environment._m3_config)
    environment._physics_source = _Physics()
    environment._last_snapshot = None
    environment._latest = None
    environment.controller = SimpleNamespace(
        execute=lambda action: M2ControlResult(
            True,
            payload={
                "action_completed_ros_time_s": 11.0,
                "reached_goal": True,
                "stalled": False,
            },
        )
    )
    environment._ensure_controller = MethodType(lambda self: self.controller, environment)
    environment.refresh_observation = MethodType(
        lambda self, **kwargs: _raw_observation(), environment
    )

    # Call the production logic with its direct super-observation seam patched
    # at the M2 class boundary.
    original = worker.GazeboM2WorkerEnv.refresh_observation
    try:
        worker.GazeboM2WorkerEnv.refresh_observation = lambda self, **kwargs: _raw_observation()
        observation, _, _, _, info = environment.step({"action_type": "move_to"})
    finally:
        worker.GazeboM2WorkerEnv.refresh_observation = original

    assert info["observation"] is observation
    assert info["physical_verification"] is observation["metadata"]["physical_verification"]
    assert environment._physics_source.calls[0]["min_timestamp_s"] == 11.0


def test_m3_stall_success_does_not_change_m2_reached_goal_policy() -> None:
    assert gripper_action_success(reached_goal=False, stalled=True, allow_stalling=True)
    assert not gripper_action_success(reached_goal=False, stalled=True, allow_stalling=False)
    assert gripper_action_success(reached_goal=True, stalled=False, allow_stalling=False)
