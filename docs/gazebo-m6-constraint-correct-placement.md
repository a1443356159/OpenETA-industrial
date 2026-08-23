# Gazebo M6: real GraspGenX/AnyPlace constraint-correct placement and recovery

M6 keeps the existing AgentTool surface. `move_to` remains one MoveIt request
that performs IK, collision-aware planning, and execution. There is no separate
IK preview, route planner, trajectory tool, fixed world wrist orientation, or
goal-region extension.

## Observed failure and semantics

The remote run returned a generic MoveIt failure for the old fixed-orientation
transport request. That proves only that the current joint state, current
planning scene, exact target, and tolerances did not produce a plan. A generic
numeric result such as `99999` is not evidence of a specific IK or collision
cause, and a failed request does not make a spatial coordinate permanently
unreachable.

Motion failure receipts include MoveIt code, planned point count,
`execution_started`, scene revision, and a fingerprint over start joints,
target, tolerances, and scene. A planning failure with
`execution_started=false` rejects only the active placement candidate. The same
fingerprint is not retried. Unknown execution outcome stops for reconciliation
or human handling.

## Perception and compilation boundary

Grasp and placement use independent observations. A fresh grasp RGB-D packet
feeds target SAM3 and GraspGenX for the `robotiq_2f_85` embodiment. The service
returns a host-only reserve pool of up to 200. After world-EEF compilation,
deterministic source-aware SE(3) diversity retains at most 64 candidates across
translation, approach, complete wrist rotation, score, and branch provenance.
The private `openeta.moveit_candidate_funnel.v2` submits at most four complete
plans. Every full-plan PASS is retained in one equal-status qualified queue;
there is no separate VLM exposure subset.

In a combined task, a pregrasp AnyPlace call first freezes up to 96 absolute
world object goals. The host screens at most four qualified grasp modes against
the complete goal batch, performs shallow compilation/structural checks for
every constructed pair, and progressively runs endpoint checks until the
four-slot plan-only capacity is filled or the batch is exhausted. This look-
ahead selects a grasp compatible with at least one object goal; it does not
authorize placement execution.

After close, Gazebo attach acknowledgement, and the unchanged M3 lift gate,
the host measures and freezes `T_eef_object_attached`. It then acquires a new
placement RGB-D packet, independently segments the attached object and target
region with SAM3, and calls AnyPlace. AnyPlace accepts those observations and
outputs only object goal poses `T_world_object_goal`; it does not accept
`selected_grasp`/`source_grasp_id` or output `place_grasp_pose`.

Only then does the host activate the stable head of the equal-status PASS queue
and internally compile it using the retained candidate id:

```json
{"placement_candidate_id":"placement_002"}
```

The host resolves that id from its qualification cache and computes
`T_world_eef_goal = T_world_object_goal * inverse(T_eef_object_attached)`. The
output is a world-frame EEF hover/release pair with the candidate's full
rotation. Any attachment-transform, pose, calibration, joint-state, scene-epoch,
or planning-scene-revision change invalidates the proof. Raw AnyPlace and grasp
estimator poses fail closed at the motion proxy.

Candidate accounting is explicit from `model_raw_candidate_count` and
`raw_candidate_count` through diversity, coordinate/TCP, workspace, pure IK,
collision IK, endpoint, full-plan, and qualification counts.
`generated_candidate_count` aliases the returned raw pool,
`submitted_candidate_count` aliases full-plan submissions, and
`candidate_count == full_plan_pass_count` counts the complete stored PASS queue;
there is no separate exposure subset.

### Camera-input selection

The observation may contain multiple calibrated RGB-D views. The runtime pairs
RGB, aligned depth, intrinsics, frame id, and available extrinsics into
`current_rgbd_views` without guessing across cameras. During perception turns,
the planner receives the corresponding current RGB images and selects a view
from visual evidence: correct target identity, useful pixel area, low
occlusion, and valid paired depth. `scene_primary` and `wrist_primary` describe
camera geometry; neither is a quality score. In particular, a wrist view at a
home pose may contain only the arm or background.

After a zero-PASS grasp batch, the host first requires a genuinely fresh RGB-D
packet. `grasp_view_selection_obligation` then constrains the VLM to one exact,
untried RGB path and the unchanged target prompt. Empty or visually rejected
masks consume only that passive view; an old mask is never reused. Active
camera motion remains available only through an existing host-generated,
IK/collision-checked obligation and pose. This logic is scene-independent and
does not add a tool.

Object and destination-region SAM3 selections for AnyPlace may come from
different complete calibrated views. Each mask remains bound to its own RGB,
depth, intrinsics, extrinsics, and frame; AnyPlace still receives the exact two
host-built observation packets.

### GraspGenX producer work

Four stochastic diffusion draws are preserved. The first invocation uses the
full GraspMoE union (diffusion plus deterministic OBB); the remaining three use
the official diffusion-only planner. This removes three repeated deterministic
OBB generations without reducing diffusion samples, OBB coverage, the returned
200-candidate reserve, scene collision filtering, or SE(3) diversity. Result
metadata reports the total draw count, one GraspMoE draw, three diffusion-only
draws, and the single-full-draw OBB policy so remote timing and candidate counts
remain auditable.

## Motion and scene constraints

M3 approach/capture/lift and its `0.0002 m / 0.002 rad` gate are unchanged.
After lift, placement uses:

- direct MoveIt planning to candidate release XY at least 100 mm above release;
- full compiled wrist rotation, without direction path constraints;
- `0.002 m / 0.05 rad` goal tolerances and `0.1` velocity/acceleration scaling;
- release 5 mm above AnyPlace's low reference, then open and detach.

The adapter applies and reads back a MoveIt planning scene. Reset contains the
table, distractor, and target. Only target contact with the two fingertip touch
links is allowed. After Gazebo attach ACK, the native target/mount state moves
the target from world collision object to an attached object on
`gripper_mount_link`; after detach ACK, the latest native target pose is added
back to world. Apply failure, readback mismatch, or set/attachment mismatch
marks the scene unavailable and blocks motion. The attached payload therefore
participates in table and distractor collision checking.

## Recovery and acceptance

Candidate rejection retains the current state and lets the host advance to the
next retained PASS candidate. Zero grasp PASS triggers a fresh grasp observation,
an untried complete view selection, and another ordinary GraspGenX cycle without
switching backends. After attachment, an independent placement observation and
SAM3 selection are still mandatory. The first placement round recompiles and
fully requalifies the frozen pregrasp object goals with the measured attachment
without model inference. Only zero PASS permits one real new-seed AnyPlace call
on that same independent observation; failed frozen goals are not merged into
the new batch. `execution_started=true`, UNKNOWN, or unsafe recovery stops for
human handling.

## Remote real-model deployment

GraspGenX and AnyPlace source, isolated environments, gripper assets, and
checkpoints belong under `/root/autodl-tmp/openeta-services/` on the approved
RTX 4090 service node, beside but isolated from the existing SAM3 service.
Nothing below `third_party/` in a developer checkout is an accepted model
deployment. GraspGenX and AnyPlace must use separate environments because their
official PyTorch stacks are incompatible.

The pinned revisions are GraspGenX `b9429097`, model repository `7c834043`,
gripper assets `19a03c00`, and AnyPlace `3049f78a`. The canonical working tree
is `/root/autodl-tmp/OpenETA-industrial`; model sources, environments, assets,
and checkpoints are deployed separately under
`/root/autodl-tmp/openeta-services/m6`. Checkpoint SHA-256 values and real MCP
smoke results must be recorded from that server. Local workstation artifacts
are never M6 evidence.

Live acceptance requires real SAM3, official GraspGenX checkpoints and
`robotiq_2f_85` gripper assets, official AnyPlace, a
main-VLM selection with no Oracle, native bilateral contact, attach ACK, at
least 80 mm lift, at most 10 mm relative drift, end-to-end candidate/calibration/
scene-revision provenance, fresh receipts, no repeated fingerprint, detach ACK,
and stable marked-zone placement. Any missing or incompatible GraspGenX,
AnyPlace, SAM3, MoveIt, or Gazebo dependency blocks live M6 rather than being
substituted with AnyGrasp, mocks, fixed candidates, or Oracle state.

## Portable launch and reproduction

`scripts/m6_gazebo_acceptance.py` is an acceptance/test coordinator, not a
production pick-place feature entry point.  Its canonical launcher is
`scripts/run_m6_gazebo_acceptance.sh`.  Always use the launcher for a live M6
case: a Python process cannot retroactively source ROS shell setup files, so a
direct `python scripts/m6_gazebo_acceptance.py` invocation can import the ROS
underlay while still hiding the checkout-specific simulation package.  That
misconfiguration previously surfaced only after the 90-second Gazebo startup
deadline.

For a fresh checkout on an Ubuntu 24.04/Jazzy host, build the overlay with the
same system Python used by ROS, then create the independent OpenETA application
environment:

```bash
source /opt/ros/jazzy/setup.bash
cd extensions/gazebo/ros2_ws
colcon build --symlink-install --cmake-args \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_EXECUTABLE=/usr/bin/python3
cd ../../..
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --no-build-isolation .
```

With SAM3, AnyPlace, and GraspGenX already serving their health/SSE endpoints,
run a case from the repository root:

```bash
scripts/run_m6_gazebo_acceptance.sh \
  --scenario normal \
  --run-root /absolute/path/to/acceptance-runs/normal-1 \
  --sam3-url http://127.0.0.1:8773/sse \
  --anyplace-url http://127.0.0.1:8875/sse \
  --graspgenx-url http://127.0.0.1:8878/sse
```

The wrapper selects `OPENETA_PYTHON_EXECUTABLE` when explicitly supplied,
otherwise the checkout's `.venv/bin/python`, sources the ROS underlay and
selected overlay, and verifies all of the following before allocating an
isolated ROS domain or starting Gazebo:

- CPython 3.12 and the declared application runtime imports;
- `rclpy` plus Jazzy generated message types in that interpreter;
- `ros2` and `gz` commands;
- `openeta_rm75_robotiq2f85_sim` resolving from the selected overlay.

Non-default installation locations are portable through absolute prefixes:

```bash
OPENETA_PYTHON_EXECUTABLE=/srv/openeta/venv/bin/python \
OPENETA_GAZEBO_SYSTEM_ROS_PREFIX=/opt/ros/jazzy \
OPENETA_GAZEBO_OVERLAY=/srv/openeta/ros2_ws/install \
scripts/run_m6_gazebo_acceptance.sh --scenario normal
```

The Python coordinator repeats the import, command, package-prefix, and
checkout/overlay-consistency preflight.  Accidental direct invocation therefore
now exits immediately with structured `openeta.gazebo_runtime_preflight.v1`
evidence.  `OPENETA_GAZEBO_OVERLAY_PACKAGE_UNAVAILABLE` means the overlay is
not sourced or not built; `OPENETA_GAZEBO_OVERLAY_PACKAGE_MISMATCH` means a
different checkout's overlay is winning `AMENT_PREFIX_PATH`.  Rebuild/source
the intended overlay instead of extending the Gazebo timeout.  `--verify-only`
remains usable without ROS because it reads existing evidence and starts no
environment.
