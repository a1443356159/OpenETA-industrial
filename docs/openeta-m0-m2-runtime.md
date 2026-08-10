# OpenETA M0-M2 reproducible runtime

> M2 live acceptance uses an evidence-only isolation observer: a private ROS
> home, localhost discovery range, and direct rclpy graph snapshots rather
> than ros2cli daemon output. Its safe candidate domain range is 80–101. ROS
> and Gazebo cleanup is reported as passed / failed / inconclusive; an
> inconclusive observation is never treated as a clean shutdown, and finalized
> acceptance reports are immutable.

The verification baseline is Ubuntu 24.04 amd64, Python 3.12, ROS 2 Jazzy and
Gazebo Harmonic. The M2 environment is
`openeta/gazebo_rm75_robotiq2f85-v0` (`rm75_robotiq_2f85_sim_v1`).
Its canonical user-facing name is **Gazebo 仿真环境**. `Robotiq
2F-85` identifies the installed gripper/profile only and is not used as the
environment's display name.

## Install and check

From a clean native Ubuntu 24.04 checkout:

```bash
bash scripts/setup_openeta_m2.sh
source config/runtime/m0_m2.env
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source extensions/gazebo/ros2_ws/install/setup.bash
bash scripts/check_openeta_m2.sh
```

The setup command supports `--check-only`, `--no-apt`, `--python-only`, and
`--ros-only`. It validates both vendored asset manifests, uses the committed
`uv.lock` with repository-local uv 0.8.13, runs rosdep, and rebuilds the colcon
overlay with `--symlink-install`. Exact Debian package versions and the actual
checkout commit are written to the ignored
`config/runtime/m0_m2.versions.local.yaml` report.

The checker emits stable failure codes including `PYTHON_NOT_READY`,
`ROS_NOT_READY`, `GAZEBO_NOT_READY`, `ROS_PACKAGE_MISSING`,
`MODEL_ASSET_NOT_FOUND`, and `WORKSPACE_NOT_BUILT`. A missing NVIDIA GPU is
informational: server-side headless rendering remains the portable default.

## Start MCP and smoke test

The base simulator has no LLM API, SAM, grasp-service, checkpoint, or external
MCP prerequisite:

```bash
python -m sim.mcp_server --port 8765
```

Connect an MCP client to `http://127.0.0.1:8765/sse`, then call
`create_env`, `reset_env`, `observe_env`, `gripper_open`, `move_to`,
`gripper_close`, `move_to`, `observe_env`, and `close_env`. Every mutating M2
action returns a fresh observation. Unknown, cancelled, timed-out, or
unreconciled outcomes fail closed. `close_env` owns and terminates the complete
ROS/Gazebo launch process group.

The unified ROS/Gazebo and MCP acceptance entry is:

```bash
bash extensions/gazebo/ros2_ws/run_m2_robotiq2f85_smoke.sh
```

It selects locked ROS domain, Gazebo partition, and MCP-port resources so an
existing M2 worker can remain live. It checks MoveIt and gripper action
servers, active controllers, JointState, TF, fresh dual RGB-D frames,
approximately 0/85 mm aperture, mimic synchronization, dynamically selected
collision-aware poses, the production SSE MCP lifecycle, fail-closed
unreachable motion, idempotent close, and exact process-group cleanup. The old
parallel-fixture smoke remains available as `run_m2_smoke.sh` for compatibility
only.

The 2026-08-09 local checkpoint passed `scripts/check_openeta_m2.sh`, 34 focused
offline contracts, both live layers, all cleanup checks, and the repository
regression (`1202 passed, 14 skipped`). It is development evidence and does not
constitute formal M2 acceptance. For this checkpoint, the two
optional BEHAVIOR/RoboCasa suites that require `torch` were excluded. The
machine-readable report is ignored by Git and written to `.cache/reports/`;
the checkpoint report is
`m2-robotiq2f85-acceptance-20260808T180318Z-74215.json` at commit
`9bc2a2c67c3881b8c687182de341cc1a8bf7c503`.

## Relocation and build outputs

`build/`, `install/`, and `log/` are not migration inputs. To test relocation,
copy the tracked checkout (without ignored files) to any temporary directory,
then run setup, check, build, and smoke there. Runtime paths are derived from
`config/runtime/m0_m2.env`; no `/home/<developer>` path or external asset
symlink is supported.

WSL2 is a development compatibility mode only. Use headless Gazebo and expect
WSLg, systemd, multicast/DDS, and GPU passthrough differences; it is not an
verification platform. SAM3, AnyGrasp, GraspGenX, AnyPlace, Contact-GraspNet,
MolmoPoint, UniDepth, Isaac/BEHAVIOR, RoboCasa, real-arm SDKs, and RealSense are
separate service or benchmark environments and are not installed by M0-M2.
