#!/usr/bin/env python3
"""Run the real RM75 DetachableJoint hard gate through the production launch.

Unlike the historical static-model diagnostic, this row never teleports the
robot root.  It starts ``m3_gazebo_pickplace.launch.py`` in detachable mode
and drives the real ``FollowJointTrajectory`` arm controller.  `Pose_V`
relative pose is the only pass/fail evidence; plugin state messages are kept
only as request-ACK diagnostics.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from run import (
    MAX_ATTACHED_ROTATION_RAD,
    MAX_ATTACHED_TRANSLATION_M,
    MIN_DETACHED_ROTATION_RAD,
    MIN_DETACHED_SEPARATION_M,
    ReproError,
    _pose_delta,
    _quat_normalized,
    _relative_pose,
)


HERE = Path(__file__).resolve().parent
WORLD_NAME = "m3_rm75_robotiq2f85_pickplace"
ROBOT = "rm75_robotiq_2f85_pickplace_sim_v1"
ARM_JOINTS = tuple(f"joint_{index}" for index in range(1, 8))
# joint_7 is coaxial with the production gripper mount: a 0.35 rad coordinate
# change produces only millimetres of mount motion.  The probe must move the
# actual detachable parent visibly in world Pose_V, so use the base joint.
PROBE_JOINT = "joint_1"
PARENT_MOTION_DELTA_RAD = 0.35
OBJECTS = {
    "target": {"model": "m3_target", "link": "target_link"},
    "distractor": {"model": "m3_distractor", "link": "distractor_link"},
}


def _field(block: str, name: str, *, default: float) -> float:
    found = re.search(rf"\b{name}:\s*([-+0-9.eE]+)", block)
    return float(found.group(1)) if found else default


class Rm75Repro:
    def __init__(
        self, *, cycles: int, output_dir: Path, parent_link: str,
        run_contact_negative: bool = True, fixed_root: bool = True,
    ) -> None:
        if parent_link not in {"gripper_mount_link", "link_7"}:
            raise ValueError("parent link must be gripper_mount_link or link_7")
        self.cycles = cycles
        self.output_dir = output_dir
        self.parent_link = parent_link
        self.run_contact_negative = run_contact_negative
        self.fixed_root = fixed_root
        self.gz = shutil.which("gz")
        self.ros2 = shutil.which("ros2")
        self.partition = f"openeta_detachable_rm75_{parent_link}_{os.getpid()}"
        self.environment = dict(os.environ, GZ_PARTITION=self.partition, ROS2CLI_NO_DAEMON="1")
        self.process: subprocess.Popen[str] | None = None
        self.diagnostics: list[dict[str, Any]] = []
        self._ros: Any | None = None
        self._arm_home: dict[str, float] | None = None
        self._contact_controller: Any | None = None

    def _run_gz(self, *arguments: str, timeout_s: float = 30.0) -> subprocess.CompletedProcess[str]:
        if not self.gz:
            raise ReproError("gz is unavailable; source /opt/ros/jazzy/setup.bash first")
        try:
            return subprocess.run(
                [self.gz, *arguments], capture_output=True, text=True,
                timeout=timeout_s, check=False, env=self.environment,
            )
        except subprocess.TimeoutExpired as exc:
            # Cold Gazebo Transport discovery can outlast an individual CLI
            # request.  The bounded stack wait below retries instead of
            # mistaking that transient discovery delay for a launch failure.
            return subprocess.CompletedProcess(
                [self.gz, *arguments], 124,
                stdout=(
                    exc.stdout.decode(errors="replace")
                    if isinstance(exc.stdout, bytes)
                    else exc.stdout or ""
                ),
                stderr=(
                    exc.stderr.decode(errors="replace")
                    if isinstance(exc.stderr, bytes)
                    else exc.stderr or "timeout"
                ),
            )

    def _publish(self, label: str, action: str) -> None:
        result = self._run_gz(
            "topic", "-t", f"/m3/detachable_joint/{label}/{action}",
            "-m", "gz.msgs.Empty", "-p", "",
        )
        if result.returncode != 0:
            raise ReproError(f"{action} publish failed for {label}: {result.stderr[-800:]}")

    def _place_object_clear_of_gripper(self, model: str) -> None:
        """Place only the payload at a collision-free clear-pad setup pose.

        This never teleports the RM75.  A DetachableJoint preserves the
        relative pose at attach time, so attaching a payload that is still on
        the table introduces a second table constraint and cannot isolate the
        detachable-joint lifecycle.  The object is parked 0.20 m above the
        current mount while the real arm remains untouched.
        """

        parent = self._world_poses()[self.parent_link]["position"]
        position = (parent[0], parent[1], parent[2] + 0.20)
        result = self._run_gz(
            "service", "-s", f"/world/{WORLD_NAME}/set_pose",
            "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req",
            (
                f'name: "{model}", position: {{x: {position[0]}, y: {position[1]}, '
                f'z: {position[2]}}}, orientation: {{w: 1.0}}'
            ),
        )
        if result.returncode != 0 or "data: true" not in result.stdout.lower():
            raise ReproError(f"failed to place clear payload {model}: {result.stderr[-800:]}")

    def _request_clear_attach(self, label: str) -> None:
        # Publish immediately after the placement service.  State messages are
        # diagnostics only; waiting for an echo before attach would let the
        # free payload fall back into a table/robot collision.
        self._publish(label, "attach")
        time.sleep(0.25)
        self.diagnostics.append({
            "label": label,
            "action": "attach",
            "setup": "clear_payload_requested",
        })

    def _state_diagnostic(self, label: str, action: str) -> None:
        """Capture a request ACK, deliberately excluding it from the verdict."""

        if not self.gz:
            return
        topic = f"/m3/detachable_joint/{label}/state"
        echo = subprocess.Popen(
            [self.gz, "topic", "-e", "-n", "1", "-t", topic],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.environment,
        )
        try:
            time.sleep(0.25)
            self._publish(label, action)
            stdout, stderr = echo.communicate(timeout=4.0)
            self.diagnostics.append({"label": label, "action": action, "state_topic": stdout.strip(), "stderr": stderr.strip()})
        except (ReproError, subprocess.TimeoutExpired) as exc:
            self.diagnostics.append({"label": label, "action": action, "state_topic_error": str(exc)})
        finally:
            if echo.poll() is None:
                echo.kill()
                echo.wait()

    def _world_poses(self) -> dict[str, dict[str, tuple[float, ...]]]:
        result = self._run_gz("topic", "-e", "-n", "1", "-t", f"/world/{WORLD_NAME}/pose/info", timeout_s=10.0)
        if result.returncode != 0 or not result.stdout.strip():
            raise ReproError(f"Pose_V read failed: {result.stderr[-800:]}")
        blocks: list[str] = []
        current: list[str] | None = None
        depth = 0
        for line in result.stdout.splitlines():
            if current is None:
                if line.strip() == "pose {":
                    current, depth = [line], 1
                continue
            current.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0:
                blocks.append("\n".join(current))
                current = None
        poses: dict[str, dict[str, tuple[float, ...]]] = {}
        for block in blocks:
            match = re.search(r'\bname:\s*"([^"]+)"', block)
            position = re.search(r"position\s*\{(.*?)\}", block, re.DOTALL)
            orientation = re.search(r"orientation\s*\{(.*?)\}", block, re.DOTALL)
            if not match or not position or not orientation:
                continue
            poses[match.group(1)] = {
                "position": tuple(_field(position.group(1), axis, default=0.0) for axis in ("x", "y", "z")),
                "orientation": _quat_normalized(tuple(_field(orientation.group(1), axis, default=1.0 if axis == "w" else 0.0) for axis in ("x", "y", "z", "w"))),
            }
        required = {self.parent_link, *(item["link"] for item in OBJECTS.values())}
        missing = required - set(poses)
        if missing:
            raise ReproError(f"Pose_V missing production RM75 entities: {sorted(missing)}")
        return poses

    def _wait_for_stack(self) -> None:
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                log = self.output_dir / "rm75-production-launch.log"
                detail = log.read_text(encoding="utf-8", errors="replace")[-800:] if log.exists() else ""
                raise ReproError(f"production M3 launch exited before Gazebo was ready: {detail}")
            models = self._run_gz("model", "--list", timeout_s=5.0)
            if ROBOT in models.stdout and all(item["model"] in models.stdout for item in OBJECTS.values()):
                return
            time.sleep(0.25)
        raise ReproError("production M3 launch did not expose RM75 and both objects")

    def _start(self) -> None:
        if not self.gz or not self.ros2:
            raise ReproError("ROS 2 Jazzy and Gazebo Sim are required")
        self.process = subprocess.Popen(
            [
                self.ros2, "launch", "openeta_rm75_robotiq2f85_sim",
                "m3_gazebo_pickplace.launch.py", "attachment_mode:=detachable",
                f"detachable_fixed_root:={'true' if self.fixed_root else 'false'}",
                f"detachable_parent_link:={self.parent_link}",
            ],
            stdout=(self.output_dir / "rm75-production-launch.log").open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT, text=True, start_new_session=True, env=self.environment,
        )
        self._wait_for_stack()
        self._start_ros_trajectory_client()

    def _start_ros_trajectory_client(self) -> None:
        try:
            import rclpy
            from action_msgs.msg import GoalStatus
            from builtin_interfaces.msg import Duration
            from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
            from rclpy.action import ActionClient
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectoryPoint
        except ImportError as exc:
            raise ReproError("rclpy/control_msgs unavailable for FollowJointTrajectory") from exc
        rclpy.init(args=None)
        node = rclpy.create_node("openeta_rm75_detachable_repro")
        arm = ActionClient(node, FollowJointTrajectory, "/rm_group_controller/follow_joint_trajectory")
        gripper = ActionClient(node, ParallelGripperCommand, "/parallel_gripper_controller/gripper_cmd")
        if not arm.wait_for_server(timeout_sec=30.0) or not gripper.wait_for_server(timeout_sec=30.0):
            node.destroy_node()
            rclpy.shutdown()
            raise ReproError("production trajectory or gripper action server is unavailable")
        latest_positions: dict[str, float] = {}

        def on_joint_state(message: JointState) -> None:
            latest_positions.update(
                {
                    name: float(position)
                    for name, position in zip(message.name, message.position)
                    if name in ARM_JOINTS
                }
            )

        joint_states = node.create_subscription(JointState, "/joint_states", on_joint_state, 10)
        self._ros = {
            "rclpy": rclpy, "node": node, "arm": arm, "gripper": gripper,
            "trajectory_type": FollowJointTrajectory, "point_type": JointTrajectoryPoint,
            "duration_type": Duration, "goal_status": GoalStatus,
            "gripper_type": ParallelGripperCommand, "joint_states": joint_states,
            "latest_positions": latest_positions,
        }
        self._arm_home = self._read_arm_positions()

    def _await(self, future: Any, timeout_s: float) -> Any:
        assert self._ros is not None
        rclpy, node = self._ros["rclpy"], self._ros["node"]
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
        if not future.done() or future.exception() is not None:
            raise ReproError("ROS action did not return a terminal result")
        return future.result()

    def _read_arm_positions(self) -> dict[str, float]:
        """Read the spawned RM75 pose instead of inventing a zero pose.

        The production model starts with non-zero joints (notably ``joint_5``).
        Sending an all-zero trajectory makes ros2_control hit its joint limit
        and adds accidental robot/object contacts to the detachable probe.
        """

        assert self._ros is not None
        deadline = time.monotonic() + 20.0
        latest = self._ros["latest_positions"]
        while time.monotonic() < deadline:
            if all(joint in latest for joint in ARM_JOINTS):
                return {joint: latest[joint] for joint in ARM_JOINTS}
            self._ros["rclpy"].spin_once(self._ros["node"], timeout_sec=0.25)
        missing = [joint for joint in ARM_JOINTS if joint not in latest]
        raise ReproError(f"joint_states missing RM75 joints: {missing}")

    def _move_parent(self, probe_joint_delta: float) -> float:
        """Move only the real RM75 probe joint relative to its home pose.

        Action ``SUCCEEDED`` is insufficient for this repro.  Record the
        measured joint displacement too, otherwise a controller-side clamp
        could be mistaken for a detachable-joint failure.
        """

        assert self._ros is not None
        if self._arm_home is None:
            raise ReproError("RM75 home joint state was not captured")
        goal = self._ros["trajectory_type"].Goal()
        point = self._ros["point_type"]()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        point.positions = [
            self._arm_home[joint] + (float(probe_joint_delta) if joint == PROBE_JOINT else 0.0)
            for joint in ARM_JOINTS
        ]
        point.time_from_start = self._ros["duration_type"](sec=3)
        goal.trajectory.points = [point]
        handle = self._await(self._ros["arm"].send_goal_async(goal), 10.0)
        if not handle.accepted:
            raise ReproError("FollowJointTrajectory goal rejected")
        wrapped = self._await(handle.get_result_async(), 20.0)
        if (
            int(wrapped.status) != int(self._ros["goal_status"].STATUS_SUCCEEDED)
            or int(wrapped.result.error_code)
            != int(self._ros["trajectory_type"].Result.SUCCESSFUL)
        ):
            raise ReproError("FollowJointTrajectory parent motion failed")
        settle_deadline = time.monotonic() + 10.0
        actual = self._read_arm_positions()[PROBE_JOINT] - self._arm_home[PROBE_JOINT]
        while abs(actual - probe_joint_delta) > 0.025 and time.monotonic() < settle_deadline:
            self._ros["rclpy"].spin_once(self._ros["node"], timeout_sec=0.25)
            actual = self._ros["latest_positions"].get(PROBE_JOINT, actual + self._arm_home[PROBE_JOINT])
            actual -= self._arm_home[PROBE_JOINT]
        if abs(actual - probe_joint_delta) > 0.025:
            raise ReproError(
                "FollowJointTrajectory did not physically reach "
                f"{PROBE_JOINT} delta {probe_joint_delta:.3f} rad "
                f"(measured {actual:.3f} rad)"
            )
        return actual

    def _open_clear_pads(self) -> None:
        assert self._ros is not None
        goal = self._ros["gripper_type"].Goal()
        goal.command.name = ["gripper_left_finger_joint"]
        goal.command.position = [0.15]
        handle = self._await(self._ros["gripper"].send_goal_async(goal), 10.0)
        if not handle.accepted:
            raise ReproError("production gripper backoff goal rejected")
        wrapped = self._await(handle.get_result_async(), 20.0)
        if int(wrapped.status) != int(self._ros["goal_status"].STATUS_SUCCEEDED):
            raise ReproError("production gripper did not clear pads")

    def _contact_reattach_negative(self) -> dict[str, Any]:
        """Exercise the prohibited in-contact reattach against real M3 motion.

        This is a negative row only: an ``attached`` topic response can never
        make it pass.  The MoveIt contact approach reuses the production
        controller, while the following parent probe remains a real
        FollowJointTrajectory move.
        """

        from extensions.gazebo.m3 import M3Config, quaternion_rotate
        from extensions.gazebo.ros_control import RosM2ControllerFactory
        from extensions.gazebo.ros2_ws.m3_pickplace_acceptance import _grasp_orientation

        config = M3Config()
        controller = RosM2ControllerFactory(readiness_timeout_s=45.0).create(config)
        self._contact_controller = controller
        try:
            poses = self._world_poses()
            target = poses[OBJECTS["target"]["link"]]["position"]
            open_receipt = controller.execute({"action_type": "gripper_open", "timeout_s": 45.0}).to_dict()
            if not open_receipt.get("ok"):
                raise ReproError("contact-negative could not open production gripper")
            # Measure the frozen fingertip centres from the same live TF used
            # by the runtime.  This avoids reviving the removed static
            # static-contact approximation.
            from rclpy.time import Time
            mesh_dir = config.gripper_asset_root / "meshes" / "collision" / "2f_85"
            centres = []
            for link in config.fingertip_links:
                side = "left" if "left" in link else "right"
                transform = controller.runtime.state_source.tf_buffer.lookup_transform(
                    config.mount_child, link, Time()
                ).transform
                from extensions.gazebo.m3 import fingertip_collision_center_m

                local = fingertip_collision_center_m(mesh_dir / f"{side}_finger_tip.stl")
                rotation = (
                    transform.rotation.x, transform.rotation.y,
                    transform.rotation.z, transform.rotation.w,
                )
                translated = quaternion_rotate(rotation, local)
                centres.append(tuple(
                    float((transform.translation.x, transform.translation.y, transform.translation.z)[index])
                    + translated[index]
                    for index in range(3)
                ))
            offset = tuple(sum(item[index] for item in centres) / 2 for index in range(3))
            orientation = _grasp_orientation(65.0, 0.0)
            rotated = quaternion_rotate(orientation, offset)
            contact_pose = {
                "xyz": [
                    target[0] - rotated[0], target[1] - rotated[1],
                    target[2] + 0.009 - rotated[2],
                ],
                "quat_xyzw": list(orientation),
            }
            move = controller.execute({
                "action_type": "move_to", "target_pose": contact_pose,
                "timeout_s": 90.0, "position_tolerance_m": 0.001,
                "orientation_tolerance_rad": 0.01,
                "max_velocity_scaling_factor": 0.1,
                "max_acceleration_scaling_factor": 0.1,
            }).to_dict()
            if not move.get("ok"):
                raise ReproError("contact-negative production contact motion failed")
            close = controller.execute({"action_type": "gripper_close", "timeout_s": 45.0}).to_dict()
            if not close.get("ok"):
                raise ReproError("contact-negative production close failed")
            before = _relative_pose(
                self._world_poses()[self.parent_link],
                self._world_poses()[OBJECTS["target"]["link"]],
            )
            self._state_diagnostic("target", "attach")
            self._move_parent(PARENT_MOTION_DELTA_RAD)
            after = _relative_pose(
                self._world_poses()[self.parent_link],
                self._world_poses()[OBJECTS["target"]["link"]],
            )
            translation, rotation = _pose_delta(before, after)
            self._state_diagnostic("target", "detach")
            self._move_parent(0.0)
            self._open_clear_pads()
            return {
                "finger_pad_overlap_pose": True,
                "attached_translation_drift_m": translation,
                "attached_rotation_drift_rad": rotation,
                "not_accepted_as_proof": True,
            }
        finally:
            controller.close()
            self._contact_controller = None

    def _stop(self) -> None:
        if self._contact_controller is not None:
            with contextlib.suppress(Exception):
                self._contact_controller.close()
            self._contact_controller = None
        if self._ros is not None:
            with contextlib.suppress(Exception):
                self._ros["node"].destroy_node()
                self._ros["rclpy"].shutdown()
            self._ros = None
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=12.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
        finally:
            self.process = None

    def run(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "case": "rm75_robotiq_production",
            "parent_link": self.parent_link,
            "detachable_fixed_root": self.fixed_root,
            "partition": self.partition,
            "pose_source": f"/world/{WORLD_NAME}/pose/info (gz.msgs.Pose_V)",
            "parent_motion": {
                "interface": "FollowJointTrajectory",
                "joint": PROBE_JOINT,
                "delta_rad": PARENT_MOTION_DELTA_RAD,
            },
            "criteria": {
                "attached_relative_translation_m_lt": MAX_ATTACHED_TRANSLATION_M,
                "attached_relative_rotation_rad_lt": MAX_ATTACHED_ROTATION_RAD,
                "detached_relative_separation_m_gt": MIN_DETACHED_SEPARATION_M,
                "detached_relative_rotation_rad_gt": MIN_DETACHED_ROTATION_RAD,
            },
            "cycles": [],
        }
        try:
            self._start()
            for label, item in OBJECTS.items():
                child = item["link"]
                model = item["model"]
                for index in range(self.cycles):
                    self._state_diagnostic(label, "detach")
                    self._open_clear_pads()
                    self._place_object_clear_of_gripper(model)
                    placed = self._world_poses()
                    before = _relative_pose(placed[self.parent_link], placed[child])
                    self._request_clear_attach(label)
                    parent_before_motion = self._world_poses()[self.parent_link]
                    measured_forward_delta = self._move_parent(PARENT_MOTION_DELTA_RAD)
                    parent_after_motion = self._world_poses()[self.parent_link]
                    parent_translation, parent_rotation = _pose_delta(
                        parent_before_motion, parent_after_motion
                    )
                    if (
                        parent_translation <= MIN_DETACHED_SEPARATION_M
                        and parent_rotation <= MIN_DETACHED_ROTATION_RAD
                    ):
                        raise ReproError(
                            "FollowJointTrajectory changed its joint state but did not "
                            "move the detachable parent in world Pose_V"
                        )
                    after = _relative_pose(
                        self._world_poses()[self.parent_link], self._world_poses()[child]
                    )
                    translation, rotation = _pose_delta(before, after)
                    self._state_diagnostic(label, "detach")
                    measured_return_delta = self._move_parent(0.0)
                    time.sleep(0.5)
                    detached = _relative_pose(
                        self._world_poses()[self.parent_link], self._world_poses()[child]
                    )
                    separation, detached_rotation = _pose_delta(after, detached)
                    report["cycles"].append({
                        "object": model,
                        "child_link": child,
                        "index": index + 1,
                        "clear_pad_attach": True,
                        "object_setup": "0.20m above current gripper mount",
                        f"measured_{PROBE_JOINT}_forward_delta_rad": measured_forward_delta,
                        f"measured_{PROBE_JOINT}_return_delta_rad": measured_return_delta,
                        "parent_world_translation_m": parent_translation,
                        "parent_world_rotation_rad": parent_rotation,
                        "attached_translation_drift_m": translation,
                        "attached_rotation_drift_rad": rotation,
                        "detached_relative_separation_m": separation,
                        "detached_relative_rotation_rad": detached_rotation,
                        "passed": (
                            translation < MAX_ATTACHED_TRANSLATION_M
                            and rotation < MAX_ATTACHED_ROTATION_RAD
                            and (
                                separation > MIN_DETACHED_SEPARATION_M
                                or detached_rotation > MIN_DETACHED_ROTATION_RAD
                            )
                        ),
                    })
            # The formal default keeps the contact row.  A focused positive
            # reproduction can skip it and run that deliberately disruptive
            # row from a separate fresh launch.
            if self.run_contact_negative:
                report["contact_reattach_negative"] = self._contact_reattach_negative()
            report["passed"] = bool(report["cycles"]) and all(item["passed"] for item in report["cycles"])
        except Exception as exc:
            report["passed"] = False
            report["error_type"] = type(exc).__name__
            report["error"] = str(exc)
        finally:
            self._stop()
        report["state_topic_diagnostics"] = self.diagnostics
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--parent-link", choices=("gripper_mount_link", "link_7"), default="gripper_mount_link")
    parser.add_argument("--skip-contact-negative", action="store_true")
    parser.add_argument("--unfixed-root", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)
    if arguments.cycles < 3:
        parser.error("--cycles must be at least 3 for the M3 hard gate")
    output_dir = arguments.output_dir or Path(tempfile.mkdtemp(prefix="openeta-detachable-rm75-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = Rm75Repro(
        cycles=arguments.cycles,
        output_dir=output_dir,
        parent_link=arguments.parent_link,
        run_contact_negative=not arguments.skip_contact_negative,
        fixed_root=not arguments.unfixed_root,
    ).run()
    path = output_dir / "rm75_summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **report}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
