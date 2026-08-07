# Task status

Source of truth: `plan.md` (M0/M1 first assignment). Keep this file updated
when work advances; do not start later milestones early.

## M0 — upstream audit

- [x] Inspect architecture, action pipeline, unified observation, adapter
      protocol, and simulator MCP lifecycle.
- [x] Record extension points and deviations in
      `docs/gazebo-integration-audit.md`.
- [ ] Run full upstream baseline in an environment with declared dependencies.
  - Current shell lacks both `gymnasium` and `uv`; retain this item open until
    a declared project environment is available.

## M1 — Gazebo read-only observation

- [x] Specify lifecycle, ROS/camera mapping, reset, cleanup, errors, and tests
      in `docs/gazebo-adapter-design.md`.
- [x] Add deterministic oracle create/reset/observe/close adapter.
- [x] Emit OpenETA `EnvObservation` with metric depth and explicit frame tags.
- [x] Add lifecycle/serialization/provenance tests.
- [ ] Connect the adapter to a real Gazebo/ROS 2 process through the existing
      MCP worker boundary (requires documented deployment transport).
- [ ] Verify an end-to-end OpenETA MCP episode and resource cleanup.

## Deferred by plan

- [ ] M2 robot control / MoveIt.
- [ ] M3 physical grasp and placement verification.
- [ ] M4 oracle pick/place.
- [ ] M5+ SAM3 integration and industrial benchmark.
