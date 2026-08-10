# Gazebo M2 RM75 + Robotiq 2F-85

The canonical user-facing name is **Gazebo 仿真环境**. The
machine-facing environment ID remains `openeta/gazebo_rm75_robotiq2f85-v0`,
and `Robotiq 2F-85` refers only to the gripper/profile. User interfaces and
operator instructions use the canonical display name above.

The sole M2 profile is `openeta/gazebo_rm75_robotiq2f85-v0` with model
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
parameter for RM75; it is not presented as a real Robotiq/UR adapter.

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

The command selects and locks an unused ROS domain in `80..101`, an isolated
Gazebo partition, and a locked MCP port. Candidate availability is established
with two direct rclpy graph samples (never a ros2cli daemon); an unavailable
ROS or Gazebo observation is recorded as `inconclusive`, not clean. The JSON
report is immutable after finalization. It builds both M2 profiles, runs the
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
acceptance. Any legacy report that was manually finalized after its original
run is untrusted cleanup evidence and is not a release gate.

## Formal acceptance

**M2 formal acceptance PASSED on 2026-08-10** (WSL2 host, report
`.cache/reports/m2-robotiq2f85-acceptance-20260810T194732Z-542318.json`,
`overall_status=passed`). All gates passed: `ros_workspace_build`,
`m2_runtime_check`, `offline_contract_regression`, `direct_live`,
`mcp_live`, `repository_regression`, and `isolation_cleanup`. Direct motion
stayed within 3.73 mm and 0.069 rad over the ten z-round moves; the SSE MCP
lifecycle (`create -> reset -> observe -> move_to x2 -> observe -> close`)
passed with `backend=gazebo`.

The 2026-08-10 session fixed four live blockers to get there: the UnifiedEnv
backend name regression (`gazebodirectenv` -> `gazebo`), the missing gazebo
observation normaliser (MCP observations lost their cameras), the structured
receipt's raw observation clobbering the MCP-wire observation, and the Jazzy
`ros2 control` readiness probe leaking a detached ros2-daemon into the
acceptance process group (cleanup gate). Two robustness fixes landed as well:
`num_planning_attempts=1 -> 3` (MoveIt executes the shortest plan, avoiding
random joint-space windup onto joint limits) and SRDF `FourBar` exemptions for
`finger_link <-> inner_knuckle_link` (the six independent Gazebo position
systems can briefly desynchronise the closed linkage and report a spurious
self-contact that previously aborted planning with `START_STATE_IN_COLLISION`).
