# Gazebo industrial benchmark

> **Status:** Offline/reference contract, not the final `multi_normal` release gate and not a live
> benchmark result. Its AnyGrasp requirements belong only to this optional benchmark definition.

`examples/gazebo_industrial_benchmark_v0.json` is an offline scene and metric
contract only. It must not be reported as a live benchmark. A live benchmark starts only
after normal acceptance passes with real SAM3, licensed official AnyGrasp and official AnyPlace, compiled
placement calibration, planning-scene revisions, and deterministic recovery.

Any live benchmark result must preserve the bilateral native-contact gate,
plugin attach ACK, measured attachment transform, exact model terminal poses,
TUI/MCP chain, and remote isolation evidence. Artificial lift/hover waypoints
and displacement thresholds are forbidden. Post-run simulator ground truth is
evaluation evidence, not visual reasoning or a planner input.

SAM3 fine-tuning is downstream of this live run and must be driven by its
measured perception errors, not by the offline manifest.
