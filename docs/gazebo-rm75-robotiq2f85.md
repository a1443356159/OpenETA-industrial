# Gazebo RM75 + Robotiq 2F-85

The canonical user-facing name is **Gazebo 仿真环境**. The
machine-facing environment ID remains `openeta/gazebo_rm75_robotiq2f85-v0`,
and `Robotiq 2F-85` refers only to the gripper/profile. User interfaces and
operator instructions use the canonical display name above.

The sole robot-control profile is `openeta/gazebo_rm75_robotiq2f85-v0` with model
`rm75_robotiq_2f85_sim_v1`.  It uses the BSD-3-Clause PickNik Robotics asset
closure frozen at commit `2c047340aeb2440f7a60e429264221aab9658707`.

OpenETA exposes only binary gripper commands (`gripper_open` and
`gripper_close`, or `gripper_control(position=1|0)`).  Internally the active
`gripper_left_finger_joint` is commanded in radians and a deterministic FK
calibration maps it to aperture: closed 0 m, active opening 0.0425 m, and
maximum aperture 0.085 m. Gazebo Harmonic does not propagate the complete
2F-85 linkage reliably from one imported URDF mimic command, so the
simulation-only action adapter expands the active-joint command to all six
vendor multiplier targets and drives six Gazebo position systems from one
common closure coordinate. It never freezes or advances one side independently:
one-pad contact is a normal intermediate state, bounded common preload and
reduced common motion provide simulated compliance, and bilateral sustained
contact admits attach. Planner and MCP callers still use the standard one-joint
`control_msgs/action/ParallelGripperCommand` endpoint; no Gazebo command topic
is exposed in the planner-facing schema.

The `link_7 -> gripper_mount_link` plate is explicitly a simulation fixture
parameter for RM75; it is not presented as a real Robotiq/UR adapter.

The profile publishes two RGB-D views.  `top_camera_optical_frame` is the
parameterized overhead scene camera (edit the `<pose>` in
`worlds/rm75_robotiq2f85.sdf`, expressed as `x y z roll pitch yaw`).
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

The checked-in world source is
`extensions/gazebo/ros2_ws/src/openeta_rm75_robotiq2f85_sim/worlds/rm75_robotiq2f85_pickplace.sdf`.
The task-neutral `multi_normal` binding is selected by
`extensions/gazebo/ros2_ws/src/openeta_rm75_robotiq2f85_sim/config/acceptance_scenes.json`.
At launch, `acceptance_scene_world.py` compiles one authoritative collision
contract from that world and materializes both Gazebo collision geometry and
MoveIt `CollisionObject` payloads. Visual-only alignment is not accepted as
collision evidence; geometry identity and readback hashes are recorded in the
PlanningScene receipt.

The complete launch starts Gazebo Harmonic, spawns
`rm75_robotiq_2f85_sim_v1`, activates the three ros2_control controllers in
dependency order, then starts MoveIt and the RGB-D bridge with simulation
time. Arm trajectories use collision-enabled MoveIt planning at 30% of the
declared joint velocity and acceleration limits. DART
(`gz-physics-dartsim-plugin`) is the stable default physics engine; Bullet
remains an explicit development override through `OPENETA_GZ_PHYSICS_ENGINE`
and is not the validated release default. Run the final task-neutral
release acceptance through the `multi_normal` entry point:

```bash
scripts/run_multi_normal_gazebo_acceptance.sh --operator-mode human_tui
```

Follow the complete [human TUI procedure](multi-normal-tui-reproduction.md). The separate `normal`
scenario remains useful for control-chain smoke and development checks, but is not the final
multi-object release scene.

The runner allocates an unused ROS domain, isolated Gazebo partition and MCP
port, records immutable environment and cleanup receipts, and terminates only
process groups proven to belong to that run. See
[`gazebo-normal-acceptance.md`](gazebo-normal-acceptance.md) for the complete
model, planning and physical PASS contract.

When the operator GUI is enabled, the runner passes the same partition to
`scripts/run_gazebo_gpu_gui.sh`. That launcher waits for the current Gazebo
server before opening a VirtualGL/OGRE2 client on the VNC display, so an older
server or an early empty GUI cannot become the visible authority.
