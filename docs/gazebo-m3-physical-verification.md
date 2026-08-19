# M3 native-contact DetachableJoint verification

M3 uses one stock Gazebo Sim 8 fixed `DetachableJoint`: parent
`gripper_mount_link`, child `m3_target/target_link`, with
`/m3/detachable_joint/target/{attach,detach,state}`.

Gazebo initializes this plugin attached. The M3 launch intentionally omits
`-r`; runtime holds the world paused, sends `detach`, listens for
`data: "detached"`, and only then resumes. Reset repeats the same ACK gate.
Attach and release similarly subscribe before publishing and require
`data: "attached"` or `data: "detached"`. `gripper_open` completes before a
release detach request.

Before a real close, M3 arms `/m3/contacts/left_pad` and
`/m3/contacts/right_pad`, both native `gz.msgs.Contacts` streams. Attach is
permitted only if each stream supplies three samples after close completion,
spanning at least 100 ms, all fresh, all unambiguously naming `m3_target`.
Unknown, mixed, stale, distractor, or one-sided evidence is rejection.

An ACK proves plugin state only. The M3 lift proof reads Gazebo's native
`/world/rm75_robotiq2f85_pickplace/pose/info` link state for
`target_link` and `gripper_mount_link`: target child-link lift must be at
least 80 mm and capture-relative translation at most 10 mm. DART/state
incompatibility reports `M3_DART_UNSUPPORTED` or a child-link/ACK error; it
does not select another mechanism.

Historical reports for removed experiments are diagnostic only. This document
does not claim a remote formal acceptance result.
