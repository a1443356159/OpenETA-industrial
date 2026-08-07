# Project memory

## 2026-08-08 — M0/M1 baseline

- `plan.md` is the authoritative implementation plan. The first assignment is
  repository audit, adapter design, and a minimal Gazebo create/reset/observe/
  close backend; SAM3, MoveIt, and physical manipulation remain deferred.
- Existing OpenETA contracts are documented in `docs/architecture.md`,
  `docs/agent-action-pipeline.md`, `adapter/protocol.py`, `sim/unified_env.py`,
  and the MCP lifecycle in `sim/mcp_server/server.py`.
- Added `docs/gazebo-integration-audit.md` and
  `docs/gazebo-adapter-design.md`.
- Added dependency-light oracle adapter under `extensions/gazebo/`. It emits
  `EnvObservation` with metric RGB-D, explicit OpenCV camera conventions, and
  `provenance=gazebo_oracle`; reset is deterministic for a given config/seed
  and close is idempotent.
- M1 adapter is intentionally read-only. `step()` rejects control until M2;
  it does not claim Gazebo or ROS 2 physical execution.
- Validation: `PYTHONPATH=. pytest -q tests/test_gazebo_lifecycle.py` passes
  (4 tests). Broader simulator tests could not collect in the current shell
  because optional dependency `gymnasium` is not installed.

## Open questions / blockers

- A real Gazebo process/ROS 2 transport and MCP registry entry require the
  deployment environment and must be implemented only after its documented
  process/topic contract is available.
