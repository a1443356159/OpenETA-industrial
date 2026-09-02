---
name: active_vision
description: Guidance for pre-contact grasp-view and placement-region recovery.
version: v1
editable: true
task_patterns:
  - active observation
  - inspect occluded object
  - improve grasp view
  - recover placement region
  - 主动感知
  - 主动观察
  - 遮挡目标
allowed_tools:
  - observe
  - sam3
  - select_sam3_detection
  - active_observe
  - grasp_pose_estimate
---
# Active vision

Use `active_observe` before contact or attachment when the current target view
is inadequate. A selected, current SAM3 evidence ID enables quality refinement.
If bounded passive text segmentation and visual point grounding still produce
no mask, the host may bind a retained calibrated image point for one
semantic-search attempt. When the configured point service is unavailable, the
tool may instead use the configured provider once in a clean, low-token visual
grounding context; the returned point is still checked against current calibrated
depth. For `placement_region`, that point seeds SAM3 on the existing primary
scene RGB-D frame and the robot does not move. For `grasp_target`, the agent
decides that another view is useful and the
tool owns deterministic viewpoint generation, strict cheap legality, Beam-2
IK/state validity, two L5 plan-only proofs, bounded motion, fresh RGB-D
acquisition, and point-grounded target refresh. After motion it checks every
current calibrated RGB-D view in deterministic order. Nested point-prompt SAM3
masks are selected from aligned depth, the projected world target and the
ordinary grasp-quality gate; this refresh does not spend another semantic VLM
review for each view.

For grounded refinement, copy `target_evidence_id` from the selected
grasp-target SAM3 `result_id`. For semantic search, choose `active_observe`
with empty parameters when it is the current host-hydrated obligation; the
host supplies `semantic_target` and, when already available, `target_hint`. Use
`semantic_role=grasp_target`, `quality_profile=grasp_rgbd`, and normally leave
`max_motion_attempts=2`. Do not pass a pose, pixel, camera offset, waypoint, or
joint state. Observation poses are sensing states only and never modify a
model contact or release terminal.

When the host binds placement-region recovery, use the same empty-parameter
call. It supplies `semantic_role=placement_region`,
`quality_profile=placement_rgbd`, and `max_motion_attempts=0`. The isolated
visual point and the resulting SAM3 mask remain model- and geometry-driven;
the agent must not invent a bin coordinate or task-specific pixel.

A `reused` result means the existing calibrated RGB-D packet already passes the
quality gate and no motion/model call occurred. An `acquired` grasp result supplies a
new observation bundle after bounded motion; an acquired placement-region result
supplies a point-grounded mask from the unchanged scene view. Continue the normal
grasp or AnyPlace flow from that evidence. On `exhausted`, inspect the compact stop reason; do not repeat the
same fingerprint. In particular, `camera_self_occlusion_unusable` is a
mechanical/calibration defect that arm motion cannot repair. On
`infrastructure_error`, stop rather than recording the target as unreachable.

When MoveIt reports that every generated grasp is inconsistent with the
authoritative target geometry, follow the host-bound `active_observe` recovery.
Rebuild only the target mask and grasp frontier from the acquired view; keep an
unchanged frozen AnyPlace goal pool instead of rerunning placement inference.

Never call active vision after attachment, without either selected evidence or
a host-bound semantic-search obligation, or merely to obtain a prettier image. Preserve the
frozen target and scene epoch. Rerun high-level perception only when the scene
genuinely changed or the bounded active-view frontier was exhausted.
