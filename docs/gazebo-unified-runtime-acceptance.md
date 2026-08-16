# Gazebo unified runtime acceptance

M1 and M2 use the profile-driven runtime for launch, cameras, controller
readiness, fresh observations and cleanup. M3 adds only the guarded stock
DetachableJoint path: paused preflight detach ACK, reset detach ACK, native
dual-pad contact admission, attach ACK and native child-link physical proof.

M3 is fail-closed on contact, state ACK, DART, or proof errors. It has no
compatibility mechanism. M4 Oracle data is marked simulator truth and does
not relax these requirements.

Formal M0–M4 acceptance is PTY TUI → MCP/SSE → Gazebo, with per-case ROS
domain, Gazebo partition, loopback port, process-group logs and cleanup
evidence. This source tree does not assert a completed remote acceptance.

## Current control-layer evidence

On 2026-08-16, SHA `c202f51` completed a remote, no-provider M2 control
diagnostic from a fresh hash-verified detached clone. A clean system-Python
venv passed the ROS ABI preflight and workspace build; real MCP/SSE then
performed create/reset, dual RGB-D observations, gripper open/close/open,
two completed MoveIt motions, and close. The isolated ROS domain, Gazebo
partition, loopback port, and owned worker cleanup were checked afterward.

This is useful control-layer evidence only. It did not invoke a planner or
provider, did not use the formal PTY TUI, and could not use the required
remote `origin` clean clone because of `problem.md` P-007. It therefore does
not mark M0–M4 formal acceptance complete.
