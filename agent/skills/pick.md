---
name: pick
description: Guidance for acquiring a target object with atomic tools.
version: v1
editable: true
task_patterns:
  - pick <object>
  - grasp <object>
  - take <object>
  - 抓取 <object>
  - 抓起来 <object>
  - 拿起 <object>
allowed_tools:
  - observe
  - retrieve_asset_reference
  - sam3
  - select_sam3_detection
  - estimate_depth_prior
  - enhance_depth
  - grasp_pose_estimate
  - reject_sam3_detections
  - activate_final_grasp_candidate
  - compute_wrist_alignment
  - camera_pose_to_world
  - obstacle_avoidance
  - move_to
  - gripper_control
---
# Pick

Use as text guidance only, not an executable macro. Inspect each result.

## Recommended Tool Sequence

1. Call `observe` to get the complete current scene observation.
2. Extract the target phrase from the user task and normalize it to a concise
   English visual object phrase for `sam3`. For example:
   - "please pick up milk box" -> `milk box`
   - "把桌上的罐子抓起来" -> `can`
   - "拿起牛奶盒" -> `milk box`
   - "抓取方块" -> `cube`
3. Call `sam3` on the exact local RGB path from `current_camera_artifacts` with
   the normalized `prompt`, for example `milk box` or `can`.
   Do not pass a non-English user phrase directly to `sam3` if a clear English object name is available.
   If direct text segmentation is empty or clearly fails to identify an unusual
   simulator asset, and `retrieve_asset_reference` is executable, call it with
   the active simulator `environment`, the exact target asset name from the task
   as `target_object`, and the exact local original RGB `scene_image`. Object
   memory resolves this task phrase to a canonical asset. Do not add
   visual category words such as `can`, `bottle`, or `box` to object-memory
   lookup. Its isolated localizer returns original-image `positive_points` and an
   audit image. Call `sam3` on that exact `scene_image`; copy points unchanged.
   Use `roi_bbox_xyxy` only for the runtime's single bbox fallback after the
   selected point mask and one dense grasp attempt produce no candidates.
4. Stop after `sam3` and inspect its result before calling
   `grasp_pose_estimate`. The
   runtime does not pass outputs between dependent batched calls. For every
   non-empty result, including a single detection, the runtime creates a
   `selection_obligation` and attaches the original RGB plus a candidate contact
   sheet to the next VLM planner request. Inspect those images and call
   `select_sam3_detection` with the exact `sam3_result_id` and `detection_id`.
   Score ranks candidates but does not prove identity. Gather another view when uncertain.
5. For real-robot RGB-D or poor depth, call `estimate_depth_prior` when
   executable, then `enhance_depth` with the same camera's exact `rgb`, `depth`,
   and `intrinsics`. Pass prior paths and confidence semantics unchanged, plus
   available registration, timestamps, scene epoch, and calibration hash. If
   no prior tool exists, sensor-only enhancement is diagnostic. Use
   `candidate_depth_png` for grasp generation only when its quality flag allows
   it. Collision evidence must use `safety_depth_png` or the safety point cloud,
   never mono-filled geometry.
6. In a combined pick-place task, do not call `grasp_pose_estimate` until the
   selected target mask and selected destination-region mask have produced the
   host's private pregrasp AnyPlace goal pool. Keep the target-object selection
   authoritative after selecting the region; the host restores its exact mask
   for grasp estimation. This bounded look-ahead changes grasp ranking only and
   never authorizes placement execution.
7. Call `grasp_pose_estimate` with the exact
   `targeted_grasp_obligation.required_parameters`. The host joins:
   - `rgb`/`depth`: exact current artifact paths for the same camera.
   - `intrinsics`: same camera intrinsics with `fx`, `fy`, `cx`, `cy`, and `scale`.
   - `object_mask`: selected artifact with exact `mask_ref` and `source_image`;
     never pass a bare path or default to `detections[0]`.
   - `camera_frame_id` and `scene_epoch`: exact host provenance.
   Backend-specific options and fallback are host-owned. Do not call AnyGrasp,
   Contact-GraspNet, or GraspGenX directly.
   If candidate depth was used, follow the host-generated
   `grasp_sensor_safety_obligation`: `obstacle_avoidance` must return
   `clear=true` for the exact candidate, scene epoch, report, and sensor-only
   safety artifacts before any motion from the host-compiled queue head may run.
8. Read the normalized grasp candidate list. Candidate poses use the
   camera/OpenCV GraspNet convention and are sorted by backend-local score.
   Scores are not comparable across backends. The runtime records
   `grasp_candidate_policy`: rank 0 is the initial `active_candidate`; lower
   ranked candidates are fallbacks and are not activated ahead of the stable
   queue order. Candidates
   that passed full plan-only have equal qualification status; the host
   deterministically activates the stable queue head.
   For a configured coupled parallel gripper, the host may derive an auditable
   closing-axis-centered pose from the selected mask and aligned depth before
   qualification. This changes translation only along the candidate's local
   closing axis; it does not change approach, wrist rotation, depth, or width.
   Do not reconstruct, undo, rank, or directly execute that correction: use
   only the host-retained full-plan PASS candidate ID.
   When selecting the SAM3 mask, include truthful
   `target_geometry_family` (`upright_can`, `upright_bottle`, `boxed_item`,
   `bowl`, `apple`, `drawer_handle`, or `other`) only when visually clear. It is
   task evidence for strategy matching, not a calibration allowlist.
9. Candidate compilation is a host-owned transition, not an AgentTool. After
   qualification, require `host_candidate_compilation` with
   `execution_started=false`, the active qualified id, current scene epoch and
   PlanningScene revision before grasp motion. The host owns the complete
   candidate, matching camera calibration, TCP profile, strategy binding and
   proof hashes. Never send a normalized grasp pose to `camera_pose_to_world`
   or construct world EEF poses in the planner.
10. Follow `grasp_execution` one observed atomic edge at a time. The host opens only
   when the latched command is not already open. Hover is at least 0.15 m opposite
   world-frame `approach_world_xyz`, not unconditionally world `+Z`. At hover, use
   fresh matching wrist RGB-D to call `compute_wrist_alignment`; bounded feedback
   corrections must preserve world frame and candidate provenance. Accept only at contact.
11. After contact, execute binary `gripper_control position=0`; `0=closed`, `1=open`,
   and the command stays latched across motion. Its acknowledgement and observed
   openness do not prove attachment; a static post-close image is not evidence.
   Portable objects use the exact lift probe; PASS permits full lift. An articulated
   handle instead uses `prepare_attachment_probe` for a 5 cm linear/arc path, then
   `assess_attachment_probe`; PASS keeps its endpoint and UNKNOWN gets one refresh.
12. A simulator transport timeout means the action outcome is unknown, not failed.
    Observe the same handle and reconcile state before retry or a new action. A structured,
    candidate-linked rejection advances to the next candidate; calibration errors,
    unrelated failures, timeout, and interruption keep the current candidate active.
    For host-classified `perception_refinable` or `uncertain_review` exhaustion,
    follow `grasp_estimation_fallback_obligation` exactly: passive RGB-D views,
    one IK/collision-checked hover plus fresh wrist re-estimation, then another
    backend. Never invent a hover. Safety, IK, collision, wrong-target, malformed
    pose, stale scene, and invalid calibration rejection remain hard stops.

## Recovery Notes

- If exact-task `sam3` returns an empty mask and `retrieve_asset_reference` is
  executable, use reference localization before changing the prompt. Do not
  broaden an unusual asset name such as `alphabet soup` to `soup can`: that can
  segment another same-category instance. The point-prompt path may retry grasp
  estimation once in dense mode, then SAM3 once with bbox ROI attention.
- If `sam3` returns multiple plausible masks, resolve the target identity before
  grasp estimation; confidence rank alone is not semantic identity.
- After all views/backends, the host activates one highest-score refinable pose
  for a final attempt. Pre-hover wrist images do not count as the wrist retry.
- Do not advance the grasp queue for transport errors, missing calibration,
  malformed parameters, or unrelated gripper failures. Automatic fallback is
  limited to rejection explicitly linked to the active grasp pose.
- After a known close rejection, the exact compiled-hover withdrawal is not by
  itself permission for another grasp. Require the host receipt to confirm
  that the detached target's measured world pose was synchronized into a new
  PlanningScene revision; a failed or missing sync is a hard stop. A successful
  sync returns to the ordinary grasp flow, not a regrasp-specific mode.
- Never move from stale perception. Observe after every world-mutating tool call.
  Keep `scene_epoch` with artifact provenance; do not reuse old masks, depth, or poses.

For explicit robot/environment calibration or parameter discovery, use the
`embodiment_explore` skill outside the benchmark episode. This skill consumes
the resulting validated profile; it does not silently recalibrate one.
