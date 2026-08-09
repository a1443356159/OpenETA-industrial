# Gazebo M2 RM75 parallel-gripper profile

Environment ID: `openeta/gazebo_rm75_parallel-v0`; model ID:
`rm75_parallel_gripper_sim_v1`.

This profile uses the supplied RM75-6FB-V arm chain (`base_link` to `link_7`, MoveIt group
`rm_group`) and mounts a simulation-only parallel two-finger fixture at
`gripper_mount_link`. Its active travel is 35 mm and total aperture is 70 mm.
It is not claimed to reproduce EG2-4C2 or soft-finger production geometry.

The repository-owned launch provides Gazebo Sim, `robot_state_publisher`, model
spawn, `gz_ros2_control/GazeboSimSystem`, the
`gz_ros2_control::GazeboSimROS2ControlPlugin`, joint-state/arm/parallel-gripper
controllers, MoveIt `move_group`, RGB-D topics, JointState, and TF. The Python
adapter delays all ROS imports until creation, then constructs the documented
`MoveGroup` and `ParallelGripperCommand` action clients and fresh JointState/TF
state source. Missing dependencies fail closed. Production code has no
external asset-root or factory override.

The closed V description lives in `extensions/gazebo/assets/rm75_6fb_v_vendor`
and is installed by `openeta_rm75_v_description`. `asset_manifest.json` records
source provenance and a
SHA-256 digest for every payload. Run the only supported build and smoke entry
points from the repository root:

```bash
bash extensions/gazebo/ros2_ws/build.sh
bash extensions/gazebo/ros2_ws/run_m2_smoke.sh
```

The build performs the offline asset preflight before invoking colcon. ROS 2
Jazzy, Gazebo Sim, MoveIt and ros2_control remain target-system dependencies.

`move_to` targets `gripper_mount_link` in `base_link`; the configured fixed
mount is inverted before forming the `link_7` MoveGroup constraint. Only exact
integer gripper commands (`0` closed, `1` open) are accepted. `reached_goal`
means controller joint-target
completion only and is never evidence of grasp/contact. Object grasp, lift,
placement, and contact success remain M3 work.
