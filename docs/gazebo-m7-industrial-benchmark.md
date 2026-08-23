# Gazebo M7 industrial benchmark

`examples/gazebo_industrial_benchmark_v0.json` is an offline scene and metric
contract only. It must not be reported as a live benchmark. Live M7 starts only
after M6 passes with real SAM3, licensed official AnyGrasp and official AnyPlace, compiled
placement calibration, planning-scene revisions, and deterministic recovery.

Any live benchmark result must preserve the native-contact gate, plugin ACK,
80 mm child-link lift and 10 mm capture-relative threshold, plus TUI/MCP and
remote isolation evidence. Oracle fields remain simulator truth, not visual
reasoning.

M8 SAM3 fine-tuning is downstream of this live run and must be driven by its
measured perception errors, not by the offline manifest.
