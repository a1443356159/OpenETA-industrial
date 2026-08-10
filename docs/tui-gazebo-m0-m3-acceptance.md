# TUI Gazebo M0–M3 acceptance

The acceptance entry point is:

```bash
bash scripts/run_tui_gazebo_acceptance.sh \
  --direct-gates /absolute/path/to/direct-gates.json
```

The shell entry point only sources ROS 2 Jazzy and the repository overlay. The
Python coordinator allocates a new `ROS_DOMAIN_ID`, `GZ_PARTITION`, localhost
port, run directory, MCP process group, `.mcp.json`, and real TUI session for
every case. It sets `OPENETA_SUPERVISION_PROFILE=human_gated`; the operator
must approve every create/reset, motion, gripper, and close prompt in the TUI.
It never edits the user's MCP configuration.

M2 and M3 formal runs require a reviewed direct-gate JSON. The minimal shape
is:

```json
{
  "m2": {
    "status": "passed",
    "collision_checked": true,
    "targets": {"A": {}, "B": {}, "unreachable": {}}
  },
  "m3": {
    "status": "passed",
    "passes": 5,
    "attempts": 5,
    "collision_checked": true,
    "targets": {}
  }
}
```

The concrete target objects must be the exact collision-checked poses emitted
by the direct fixture gates. The verifier rejects submitted M2 poses not equal
to a frozen target.

Each case directory contains the terminal transcript, full session
`trace.jsonl`, response/image artifacts, MCP log, PID/PGID record, cleanup
result, and a hashed environment receipt. The receipt captures ROS domain,
Gazebo partition, RMW implementation, overlay, WSL version, git head/dirty
summary, port, and pre-existing long-lived processes. Cleanup only signals the
recorded process group and checks that its port and newly-created
ROS/Gazebo/worker processes are gone. Domains 42 and 100 are never allocated.

The verifier reads structured tool results and receipts, never the Planner's
natural-language conclusion. Its immutable final report uses schema
`openeta.tui_gazebo_acceptance.v1` and keeps `backend_chain_status` separate
from `planner_autonomy_status`. A Planner failure cannot overwrite a backend
gate. Formal M0→M3 execution stops at the first failed or blocked backend gate.

Exit codes are `0` for all deterministic gates passed, `1` for deterministic
failure, `2` for blocked/inconclusive infrastructure, and `130` for operator
interrupt. On WSL2, an M1 topic-discovery timeout is reported as blocked and
stops the formal chain; do not reuse domains 42/100, the ros2cli daemon, or an
old ROS graph, and do not substitute synthetic observations. As of 2026-08-10
the two M1 live tests (`tests/test_gazebo_rgbd_bridge.py`,
`tests/test_gazebo_world_control.py`) pass on the WSL2 host (7/7 consecutive
runs): the historical timeout was reproduced as cold-start discovery exceeding
the old 15 s readiness budget under a leftover CPU-bound `gz sim`, and
`GazeboProcess.wait_for_topics` now defaults to 30 s.

For evidence review without starting processes:

```bash
.venv/bin/python scripts/tui_gazebo_acceptance.py \
  --run-root /absolute/path/to/run \
  --direct-gates /absolute/path/to/direct-gates.json \
  --verify-only
```

`--prepare-only` creates isolated case directories and operator instruction
files without starting MCP or TUI. Acceptance artifacts live under
`.cache/reports/` by default and are not committed. Milestone status should be
updated only after the coordinator exits 0 and the report is complete.
