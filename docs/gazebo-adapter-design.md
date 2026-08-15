# Gazebo adapter design

`GazeboDirectEnv` and `GazeboRuntime` own M1 live RGB-D, M2 RM75/Robotiq
control, and M3's guarded pick/place profile. The production chain is:

`MCP/SSE → dedicated Gazebo bench worker → UnifiedEnv → GazeboDirectEnv → GazeboRuntime → ROS 2 / Gazebo Sim`

M3 has one grasp mechanism only: the stock Gazebo Sim 8
`gz::sim::systems::DetachableJoint` fixed joint from `gripper_mount_link` to
`m3_target/target_link`. The M3 launch starts paused. Runtime must receive a
fresh `detached` state ACK before unpausing or starting controller readiness.
Every reset repeats pause → reset → detached ACK → object restore → unpause.

`gripper_close` arms both native Gazebo fingertip contact streams before the
real command. Attach is issued only when each stream has at least three fresh,
post-close samples covering 100 ms and every sample identifies only
`m3_target`. Unknown, mixed, stale, single-sided, or distractor contacts fail
closed. An attach ACK permits transport but is not a grasp verdict. M3 passes
only after native Gazebo child-link state proves at least 80 mm target lift
and no more than 10 mm capture-relative translation.

There is no force injection, compliance, gravity compensation, kinematic
following, geometric/TF/distance admission, or alternate physics path. A
missing state ACK or unreadable DART child-link state reports an explicit M3
error and stops the chain.

## Oracle boundary

`oracle_perceive` remains simulator-truth projection, marked
`perception_source="gazebo_oracle"`. M4's fake grasp candidate is contractual
input-shape evidence only; it does not claim visual reasoning and cannot
bypass M3's native-contact, ACK, or child-link proof gates.

M2 gripper safeguards and articulated-handle assets are independent of M3.
The registered profile names are `m1`, `m2_robotiq2f85`, and `m3_pickplace`;
the M3 receipt schema is `openeta.m3.detachable_joint.v1`.

## Acceptance status

Formal M0–M4 evidence must be captured through the real PTY TUI → MCP/SSE →
Gazebo chain. Scripted approvals are labelled `scripted_tui`; they are never
reported as human approval. A final pass additionally requires the remote
clean-clone run and its isolation/cleanup evidence. No such remote result is
claimed by this document.
