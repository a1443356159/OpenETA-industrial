# Gazebo adapter design

`GazeboDirectEnv` and `GazeboRuntime` own live RGB-D observation,
RM75/Robotiq motion control, and guarded native-contact pick/place. The
production chain is:

`MCP/SSE → dedicated Gazebo bench worker → UnifiedEnv → GazeboDirectEnv → GazeboRuntime → ROS 2 / Gazebo Sim`

Native-contact pick/place has one grasp mechanism only: the stock Gazebo Sim 8
`gz::sim::systems::DetachableJoint` fixed joint from `gripper_mount_link` to
`target_object/target_link`. The launch starts paused. Runtime must receive a
fresh `detached` state ACK before unpausing or starting controller readiness.
Every reset repeats pause → reset → detached ACK → object restore → unpause.

`gripper_close` arms both native Gazebo fingertip contact streams before the
real command. Attach is issued only when each stream has at least three fresh,
post-close samples covering 100 ms and every sample identifies only
`target_object`. Unknown, mixed, stale, single-sided, or distractor contacts fail
closed. An attach ACK alone is not a grasp verdict. Bilateral target contact
plus that ACK directly proves the grasp and freezes the measured attachment
transform. No artificial lift, hover, or displacement threshold is part of
the grasp verdict.

There is no force injection, compliance, gravity compensation, kinematic
following, geometric/TF/distance admission, or alternate physics path. A
missing state ACK or invalid attachment state reports an explicit native-grasp
error and stops the chain.

## Oracle boundary

`oracle_perceive` remains simulator-truth projection, marked
`perception_source="gazebo_oracle"`. M4's fake grasp candidate is contractual
input-shape evidence only; it does not claim visual reasoning and cannot
bypass native-contact, ACK, or child-link proof gates.

Motion-control gripper safeguards and articulated-handle assets are independent
of native grasping. The registered profile names are `rgbd_observation`,
`rm75_robotiq2f85_control`, and `rm75_robotiq2f85_pickplace`; the native-grasp
receipt schema is `openeta.gazebo.native_grasp.v1`.

## Acceptance status

Formal M0–M4 evidence must be captured through the real PTY TUI → MCP/SSE →
Gazebo chain. Scripted approvals are labelled `scripted_tui`; they are never
reported as human approval. A final pass additionally requires the remote
clean-clone run and its isolation/cleanup evidence. No such remote result is
claimed by this document.
