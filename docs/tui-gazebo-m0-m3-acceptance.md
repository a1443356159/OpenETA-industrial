# TUI Gazebo acceptance scope

The formal path is a real `openeta_cli` PTY TUI connected to MCP/SSE and then
to the isolated Gazebo worker. M0–M4 evidence consists of the PTY transcript,
MCP request/response trace, tool receipt and cleanup record for each case.

M3 requires the stock DetachableJoint native-contact/ACK/child-link chain;
M4 records `gazebo_oracle` and fake-candidate provenance while using that same
chain. This includes the executed `oracle_perceive` MCP call: its request,
case-local materialized response, and receipt must correlate. Scripted
approvals are labelled `scripted_tui`, never `human_gated`.

Remote formal runs use a SHA-specific clean clone and stop immediately after
any failed predecessor. The clone never supplies an untracked `.venv`: the
approved remote plan creates a separate
`OPENETA_CLOUD_ACCEPTANCE_ROOT/venvs/<SHA>` environment from an explicitly
selected system Python (`/usr/bin/python3` or `/usr/bin/python3.12`), without
`--system-site-packages`. It bootstraps pip/setuptools/wheel, installs declared
project dependencies, checks application imports and the sourced Jazzy
`rclpy`/`rosgraph_msgs.msg.Clock` ABI, then passes that venv executable to the
TUI. The plan requires a safe branch ref and uses a HTTP/1.1 `--depth 1
--branch <branch>` clone before checking out the requested SHA detached.

At present the remote GitHub origin clone is blocked by the recorded TLS/SSH
transport issue in `problem.md` P-007. A hash-verified bundle clone was used
only to diagnose the no-provider M2 control chain; it is neither a formal
origin clone nor a PTY/TUI result. This document makes no claim that a remote
formal run has passed.
