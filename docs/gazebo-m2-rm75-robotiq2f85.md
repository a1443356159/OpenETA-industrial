# Gazebo M2 RM75 + Robotiq 2F-85

The canonical user-facing name is **Gazebo 仿真环境**. The
machine-facing environment ID remains `openeta/gazebo_rm75_robotiq2f85-v0`,
and `Robotiq 2F-85` refers only to the gripper/profile. User interfaces and
operator instructions use the canonical display name above.

The recommended M2 profile is `openeta/gazebo_rm75_robotiq2f85-v0` with model
`rm75_robotiq_2f85_sim_v1`.  It uses the BSD-3-Clause PickNik Robotics asset
closure frozen at commit `2c047340aeb2440f7a60e429264221aab9658707`.

OpenETA exposes only binary gripper commands (`gripper_open` and
`gripper_close`, or `gripper_control(position=1|0)`).  Internally the active
`gripper_left_finger_joint` is commanded in radians and a deterministic FK
calibration maps it to aperture: closed 0 m, active opening 0.0425 m, and
maximum aperture 0.085 m. Gazebo Harmonic does not propagate the complete
2F-85 linkage reliably from one imported URDF mimic command, so the
simulation-only action adapter expands the active-joint command to all six
vendor multiplier targets and drives six Gazebo position systems. Planner and
MCP callers still use the standard one-joint
`control_msgs/action/ParallelGripperCommand` endpoint; no Gazebo command topic
is exposed in the planner-facing schema.

The `link_7 -> gripper_mount_link` plate is explicitly a simulation fixture
parameter for RM75; it is not presented as a real Robotiq/UR adapter.  The
legacy `openeta/gazebo_rm75_parallel-v0` 70 mm fixture remains available for
backward compatibility.

The profile publishes two RGB-D views.  `top_camera_optical_frame` is the
parameterized overhead scene camera (edit the `<pose>` in
`worlds/m2_rm75_robotiq2f85.sdf`, expressed as `x y z roll pitch yaw`).
`wrist_camera_optical_frame` is attached through the V description's fixed
`link_7 -> camera_rolink -> camera_link` chain and follows the end effector.
Its bracket and camera-body meshes, inertias and default offsets come directly
from the supplied RM75-6FB-V package. `camera_rojoint` is fixed at its default
mounting pose and has no controller. Their ROS topics are:

```text
/openeta_rgbd/{image,depth_image,camera_info}          # top / scene_primary
/openeta_wrist_rgbd/{image,depth_image,camera_info}    # wrist / wrist
```

The OpenETA observation contains both camera frames; `gripper_mount_link` is
only the mechanical gripper mounting link and is not used as a camera label.

Install and validate the reproducible Ubuntu 24.04/Jazzy environment using
[`openeta-m0-m2-runtime.md`](openeta-m0-m2-runtime.md). The complete launch
starts Gazebo Harmonic, spawns `rm75_robotiq_2f85_sim_v1`, activates the three
ros2_control controllers in dependency order, then starts MoveIt and the RGB-D
bridge with simulation time. Arm trajectories use collision-enabled MoveIt
planning at 30% of the declared joint velocity and acceleration limits. Run
the unified live acceptance check with:

```bash
bash extensions/gazebo/ros2_ws/run_m2_robotiq2f85_smoke.sh
```

The command selects and locks an unused ROS domain in `100..199`, an isolated
Gazebo partition, and a locked MCP port. It builds both M2 profiles, runs the
offline contracts, exercises direct ROS actions, then executes the real SSE
MCP `create -> reset -> gripper -> move_to -> observe -> close` lifecycle. It
also verifies normal, startup-failure, action-failure, and signal cleanup
paths. Only process groups created by the command are terminated.

The 2026-08-09 local checkpoint used commit
`9bc2a2c67c3881b8c687182de341cc1a8bf7c503`: direct motion stayed within
3.46 mm and 0.054 rad, the maximum measured mimic error was 0.0166 rad, and
the MCP unreachable target returned `MOTION_PLAN_FAILED`. The ignored JSON
evidence is written under `.cache/reports/`; the checkpoint run is
`m2-robotiq2f85-acceptance-20260808T180318Z-74215.json` (the filename is UTC).
These results are development evidence and do not constitute formal M2
acceptance.
