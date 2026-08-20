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
feeds target SAM3 and GraspGenX for the `robotiq_2f_85` embodiment. GraspGenX's
raw pool is reduced to ten formal candidates with deterministic source-aware
SE(3) MMR over translation, SO(3) geodesic angle, backend score, and branch
provenance. This does not rewrite poses or use robot IK. All ten formal
candidates then receive private MoveIt qualification, and only PASS candidates
are exposed to the main VLM.

After close, Gazebo attach acknowledgement, and the unchanged M3 lift gate,
the host measures and freezes `T_eef_object_attached`. It then acquires a new
placement RGB-D packet, independently segments the attached object and target
region with SAM3, and calls AnyPlace. AnyPlace accepts those observations and
outputs only object goal poses `T_world_object_goal`; it does not accept
`selected_grasp`/`source_grasp_id` or output `place_grasp_pose`.

Only then does the main VLM call `compile_placement_seed` with:

```json
{"placement_candidate_id":"placement_002"}
```

The host resolves that id from its qualification cache and computes
`T_world_eef_goal = T_world_object_goal * inverse(T_eef_object_attached)`. The
output is a world-frame EEF hover/release pair with the candidate's full
rotation. Any attachment-transform, pose, calibration, joint-state, scene-epoch,
or planning-scene-revision change invalidates the proof. Raw AnyPlace and grasp
estimator poses fail closed at the motion proxy.

Candidate accounting is explicit: `raw_candidate_count` is GraspGenX output
before diversity, `generated_candidate_count` is the ten-candidate formal pool,
`submitted_candidate_count` is the pool sent to MoveIt, and
`qualified_candidate_count`/`candidate_count` count PASS candidates exposed to
the VLM.

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

Candidate rejection retains the current state and asks the main VLM to select
another PASS candidate. Zero grasp PASS triggers a fresh grasp observation and
reruns GraspGenX without switching backends. Zero placement PASS keeps the
native attachment, acquires a new placement observation, resegments both
placement inputs, and reruns AnyPlace only. `execution_started=true`, UNKNOWN,
or unsafe recovery stops for human handling.

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
