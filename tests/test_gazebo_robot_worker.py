from __future__ import annotations

import threading

import numpy as np

from adapter.protocol import EnvObservation

from extensions.gazebo.direct_env import GazeboDirectEnv, build_gazebo_control_spec
from extensions.gazebo.profiles import CONTROL, STRUCTURED_RECEIPT, gazebo_profile
from sim import bench_worker


def test_robot_profiles_use_the_single_direct_env_and_structured_receipt_capability() -> None:
    for name in ("rm75_robotiq2f85_control",):
        profile = gazebo_profile(name)
        assert CONTROL in profile.capabilities
        assert STRUCTURED_RECEIPT in profile.capabilities
        assert GazeboDirectEnv.__name__ == "GazeboDirectEnv"


def test_robot_control_spec_advertises_validated_relative_motion_only_for_control_profile() -> None:
    control_spec = build_gazebo_control_spec(gazebo_profile("rm75_robotiq2f85_control"))

    guidance = control_spec["validated_relative_motion"]
    assert guidance["reference"] == "first_fresh_end_effector_pose_after_reset"
    assert guidance["orientation"] == "preserve_observed"
    assert guidance["targets"] == [
        {"name": "vertical_low", "xyz_offset_m": [0.0, 0.0, -0.040]},
        {"name": "vertical_high", "xyz_offset_m": [0.0, 0.0, -0.020]},
    ]
    assert "validated_relative_motion" not in build_gazebo_control_spec(gazebo_profile("rm75_robotiq2f85_pickplace"))


def test_generic_worker_codec_restores_internal_receipt_without_observation_clobber() -> None:
    observation = {"task": "", "cameras": {}, "robot": {}, "objects": [], "metadata": {"model_id": "control"}}

    class Environment:
        def step(self, _action):
            return observation, 0.0, False, False, {
                "_openeta_receipt": {"ok": True, "debug": np.zeros((1,)).tolist()}
            }

    payload = bench_worker._step_with_image(Environment(), {}, render=False)
    assert payload["ok"] is True
    assert payload["observation"]["metadata"]["model_id"] == "control"
    assert "_openeta_receipt" not in payload["info"]


def test_structured_receipt_raw_observation_never_clobbers_the_mcp_observation() -> None:
    # GazeboDirectEnv anchors the raw unified observation (numpy arrays,
    # cameras keyed by frame_id) inside its receipt.  The wire contract keeps
    # the StepResult-converted MCP observation at the top level.
    raw_cameras = {
        "top_camera_optical_frame": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
    }
    observation = {"task": "", "cameras": {}, "robot": {}, "objects": [], "metadata": {}}
    receipt_observation = {"task": "", "cameras": raw_cameras, "robot": {}, "objects": []}

    class Environment:
        def step(self, _action):
            return observation, 0.0, False, False, {
                "_openeta_receipt": {"ok": True, "observation": receipt_observation}
            }

    payload = bench_worker._step_with_image(Environment(), {}, render=False)
    assert payload["ok"] is True
    assert isinstance(payload["observation"]["cameras"], list)
    assert payload["observation"]["cameras"] == []


def test_worker_numpy_rgbd_fast_path_preserves_mcp_camera_contract() -> None:
    observation = {
        "task": "inspect",
        "cameras": {
            "top": {
                "rgb": np.array(
                    [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                    dtype=np.uint8,
                ),
                "depth": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32),
                "intrinsics": {"fx": 100.0, "fy": 100.0},
                "extrinsics": {
                    "camera_frame": "opencv",
                    "normalized_from": "gazebo_ros",
                },
                "timestamp_s": 12.5,
                "role": "scene_primary",
            }
        },
        "robot": {},
        "objects": [],
        "metadata": {"scene_epoch": 3},
    }

    expected = EnvObservation.from_dict(observation).to_mcp_dict()
    actual = bench_worker._env_obs_to_mcp(observation)

    assert actual == expected


def test_worker_numpy_rgbd_fast_path_skips_nested_list_conversion(
    monkeypatch,
) -> None:
    import adapter.protocol as protocol

    def reject_list_materialization(_value):
        raise AssertionError("array-backed camera must not materialize nested lists")

    monkeypatch.setattr(protocol, "_to_int_list3d", reject_list_materialization)
    monkeypatch.setattr(protocol, "_to_float_list2d", reject_list_materialization)
    observation = {
        "task": "inspect",
        "cameras": {
            "top": {
                "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
                "depth": np.ones((2, 3), dtype=np.float32),
                "intrinsics": {},
                "extrinsics": {},
            }
        },
        "robot": {},
        "objects": [],
        "metadata": {},
    }

    payload = bench_worker._env_obs_to_mcp(observation)

    assert payload["cameras"][0]["width"] == 3
    assert payload["cameras"][0]["height"] == 2
    assert payload["cameras"][0]["rgb_base64"]
    assert payload["cameras"][0]["depth_base64"]


def test_dashboard_fresh_observation_waits_for_atomic_physical_step() -> None:
    handle = "fresh-observation-lock-test"
    step_entered = threading.Event()
    release_step = threading.Event()
    observe_entered = threading.Event()
    result = {
        "task": "",
        "cameras": {},
        "robot": {},
        "objects": [],
        "metadata": {},
    }

    class Environment:
        openeta_capabilities = frozenset({"fresh_observation"})

        def step(self, _action):
            step_entered.set()
            assert release_step.wait(timeout=2.0)
            return result, 0.0, False, False, {}

        def observe(self):
            observe_entered.set()
            return result

    environment = Environment()
    bench_worker._envs[handle] = environment
    step_thread = threading.Thread(
        target=bench_worker._step_with_image,
        args=(environment, {}),
        kwargs={"handle": handle, "render": False},
    )
    observe_thread = threading.Thread(
        target=bench_worker._observe_with_image,
        args=(environment,),
        kwargs={"handle": handle},
    )
    try:
        step_thread.start()
        assert step_entered.wait(timeout=1.0)
        observe_thread.start()
        assert not observe_entered.wait(timeout=0.1)
        release_step.set()
        step_thread.join(timeout=2.0)
        observe_thread.join(timeout=2.0)
        assert not step_thread.is_alive()
        assert not observe_thread.is_alive()
        assert observe_entered.is_set()
    finally:
        release_step.set()
        step_thread.join(timeout=2.0)
        observe_thread.join(timeout=2.0)
        bench_worker._envs.pop(handle, None)
        bench_worker._last_obs.pop(handle, None)
        bench_worker._done_handles.discard(handle)
        with bench_worker._obs_locks_guard:
            bench_worker._obs_locks.pop(handle, None)
