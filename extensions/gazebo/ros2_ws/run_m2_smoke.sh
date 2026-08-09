#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENETA_SKIP_ROSDEP=1 "${SCRIPT_DIR}/build.sh"
ROS2_BIN="${OPENETA_ROS2_BIN:-$(command -v ros2 || true)}"
if [[ -z "${ROS2_BIN}" ]]; then echo "ROS_NOT_READY: ros2 is not on PATH" >&2; exit 3; fi
ROS_PREFIX="$(cd -- "$(dirname -- "${ROS2_BIN}")/.." && pwd)"
set +u
if [[ -r "${ROS_PREFIX}/setup.bash" ]]; then source "${ROS_PREFIX}/setup.bash"; fi
source "${SCRIPT_DIR}/install/setup.bash"
set -u
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${M2_PID:-}" ]] && kill -0 "${M2_PID}" 2>/dev/null; then
    kill -TERM -- "-${M2_PID}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "${M2_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${M2_PID}" 2>/dev/null || true
    wait "${M2_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
setsid ros2 launch openeta_rm75_parallel_sim m2_gazebo_moveit.launch.py >"${SCRIPT_DIR}/m2-smoke.log" 2>&1 &
M2_PID=$!
for _ in $(seq 1 90); do
  if ros2 action list 2>/dev/null | grep -qx '/move_action' && ros2 action list 2>/dev/null | grep -qx '/parallel_gripper_controller/gripper_cmd' && ros2 control list_controllers 2>/dev/null | grep -Eq '^joint_state_broadcaster[[:space:]].*active' && ros2 control list_controllers 2>/dev/null | grep -Eq '^rm_group_controller[[:space:]].*active' && ros2 control list_controllers 2>/dev/null | grep -Eq '^parallel_gripper_controller[[:space:]].*active'; then
    topics="$(ros2 topic list)"
    for topic in /joint_states /tf /tf_static \
      /openeta_rgbd/image /openeta_rgbd/depth_image /openeta_rgbd/camera_info \
      /openeta_wrist_rgbd/image /openeta_wrist_rgbd/depth_image /openeta_wrist_rgbd/camera_info; do
      grep -qx "${topic}" <<<"${topics}"
    done
    sample="$(timeout 8 ros2 topic echo /joint_states sensor_msgs/msg/JointState --once)"
    for index in 1 2 3 4 5 6 7; do grep -q "joint_${index}" <<<"${sample}"; done
    tf_sample="$(timeout 8 ros2 run tf2_ros tf2_echo base_link link_7 2>/dev/null || true)"
    grep -q 'Translation:' <<<"${tf_sample}"
    timeout 8 ros2 topic echo /openeta_rgbd/camera_info sensor_msgs/msg/CameraInfo --once >/dev/null
    timeout 8 ros2 topic echo /openeta_wrist_rgbd/camera_info sensor_msgs/msg/CameraInfo --once >/dev/null
    echo "M2 smoke readiness passed"
    exit 0
  fi
  kill -0 "${M2_PID}" 2>/dev/null || { tail -100 "${SCRIPT_DIR}/m2-smoke.log" >&2; exit 4; }
  sleep 1
done
echo "ROS_NOT_READY: action servers did not become ready" >&2
exit 5
