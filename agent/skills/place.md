---
name: place
description: Guidance for placing a held object on or inside a target receptacle.
version: v1
editable: true
task_patterns:
  - place <object> on <target>
  - put <object> into <target>
  - place <object> in <target>
allowed_tools:
  - observe
  - sam3
  - select_sam3_detection
  - reject_sam3_detections
  - anyplace
  - move_to
  - gripper_control
  - close_simulator_env
---
# Place

Use this skill as text guidance only. Do not treat `place` as an executable
macro. After each tool result, inspect the returned observation or tool output
before choosing the next tool call.

## Recommended Tool Sequence

1. Before grasp estimation in a combined pick-place task, segment and select
   the target object, then segment and select the destination region from the
   same calibrated, unchanged RGB-D scene. Follow the host
   `placement_obligation` once to let AnyPlace retain a host-private object-goal
   pool. These are not executable placement candidates and are not shown to the
   VLM.
2. Call grasp estimation only after that pool is ready. The host first runs the
   normal complete grasp funnel, then performs a bounded look-ahead over at most
   four grasp PASS candidates and the current complete 96-goal pool. All
   constructed pairs pass through coordinate, workspace, pure-IK, collision-IK
   and endpoint screening; globally at most four grasp-goal
   pairs receive plan-only. Only grasps compatible with at least one placement
   goal remain in the qualified queue, and the host activates its stable head;
   this look-ahead does not prove the later real execution.
3. Complete the pickup using the host-activated grasp. After closing the gripper,
   require the native attach acknowledgement, fixed lift, and attachment PASS.
   Executable placement candidate selection, placement compilation, and motion
   remain blocked until this gate completes.
4. After attachment PASS, observe the placement scene independently. Segment
   the held object and the target region from placement observations and call
   `anyplace` with the exact host-provided object/placement observation packets.
   AnyPlace predicts only object goal poses; it never accepts a selected grasp
   or produces an EEF pose. Never run grasp estimation on the receptacle as a
   substitute for AnyPlace.
   The host freezes only the absolute object goals that passed the pregrasp
   look-ahead for the grasp that was actually executed. On the first
   post-attachment `anyplace` obligation, the host validates the independent
   observation packets, skips model inference, recompiles those goals using
   the measured attachment and current robot state, then reruns the complete
   qualification funnel. The earlier look-ahead trajectory is never reused as
   executable proof. Only a zero-PASS frozen requalification invokes one real
   new-seed AnyPlace inference on the same observation.
5. Every retained AnyPlace candidate has passed the full host funnel; failed,
   UNKNOWN and unsubmitted candidates are absent, and every PASS has equal
   qualification status. The host deterministically activates the stable queue
   head and compiles it as an internal transition. Require the immutable
   `host_candidate_compilation` event to bind the full object goal, measured
   attachment transform, current start state, calibration, scene epoch and
   PlanningScene revision with `execution_started=false`. Never send a raw
   AnyPlace pose to `camera_pose_to_world` or `move_to`.
6. Move directly to the compiled pre-place hover, then descend to the compiled
   release pose. Preserve its full wrist rotation and use the placement motion
   profile; MoveIt computes the joint path under the complete planning scene.
7. Inspect fresh evidence after transport. The earlier lift-probe
   PASS is stale after motion: continue only when the target is still co-located
   with the gripper and its source location remains vacant. If the target is
   visible elsewhere and the closed-gripper openness has collapsed to the empty
   threshold, follow the `attachment_lost` recovery action so the current grasp
   candidate is rejected before attempting another grasp.
8. Descend only to the compiled `placement_motion_guidance.release_pose`, whose
   profile-owned clearance is 5 mm above the AnyPlace low reference.
9. Call `gripper_control` with `position=1` only after the vertical placement
   motion succeeds and fresh evidence still supports attachment over the
   receptacle.
10. After opening, allow the configured natural settling observation horizon to
    complete. The verifier still judges only the final 0.5 seconds, requires
    drift <=5 mm, and preserves the height and full-footprint gates. Retreat
    with `move_to`, then call `observe` to verify the object was released
    in the intended place and check the official task reward. A successful
    gripper or retreat tool call is not placement success: require the returned
    placement verification itself to be PASS. If post-release verification is
    FAIL or UNKNOWN, do not claim completion or try another candidate after
    execution has started.
11. Once the final placement verification has been reported, close the active
    simulator environment exactly once with `close_simulator_env`, whether the
    verification was PASS, FAIL, or UNKNOWN. Do not leave an environment open
    merely because the physical task failed.

## Recovery Notes

- If the target receptacle or surface is ambiguous, call `ask_human` before
  moving.
- If measured-attachment requalification of the frozen pregrasp PASS goals has
  zero PASS candidates, keep the independent placement observation and run
  exactly one new-seed AnyPlace round. Qualify only that round's new pool,
  without merging it with or requalifying goals from the failed frozen set.
- If both rounds fail, treat the current attachment as
  `CURRENT_GRASP_PLACE_INFEASIBLE` and request human intervention. Do not alter
  the attachment or start another grasp cycle automatically.
- After a known grasp-close failure, execute the host-provided return to that
  grasp's compiled hover before observing or estimating another grasp. Never
  collect a recovery observation while the gripper remains at contact.
- If the target is occluded, observe from another camera or request a broader
  scene query before choosing a release pose.
- If MoveIt rejects a plan before execution starts, reject only that candidate
  and let the host advance to the next precompiled retained candidate. Never
  blindly retry the same request fingerprint or describe the coordinate as
  permanently unreachable.
- If the object remains in the gripper after opening, retry `gripper_control`
  once, observe, then ask for help or replan.
- Never release an object from stale perception. Observe again after every
  world-mutating tool call.

For explicit clearance or controller-profile discovery, use the
`embodiment_explore` skill outside the benchmark episode. Do not copy a
successful value from another robot or environment into this task.
