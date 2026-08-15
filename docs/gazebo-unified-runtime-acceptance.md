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
