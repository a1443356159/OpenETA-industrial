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

This is decision guidance, not an executable macro. The planner explicitly
chooses every tool from fresh feedback. AnyPlace predicts object goal poses;
the host proves exact geometry and MoveIt owns robot paths.

## Normal flow

1. In a combined pick-place task, segment/select the target object and
   destination region once, each from a complete calibrated RGB-D packet. The
   host calls AnyPlace once and retains all 96 returned object goals. Do not
   expose them to the VLM as motion waypoints.
2. Before IK, the host evaluates each AnyPlace goal exactly once for finite
   SE(3), valid rotation, full footprint inside the placement region, legal
   support/height/bounds, no static-scene penetration, and mathematically
   certain workspace limits.
3. For each pair that reaches the current deep wave, the host derives the exact release EEF
   pose as `object_goal * inverse(measured_attachment)`. Pair
   legality checks that exact terminal state, attached object, gripper, static
   scene, and strict analytic reach bounds. This relatively expensive geometry
   is not computed for the untouched tail. Parallel-gripper-equivalent evidence
   may share a result but both candidate records remain.
4. Qualification alternates the two grasp branches and covers placement goals
   per branch in deterministic incremental `4 -> 8 -> 16 -> 32 -> 96` waves.
   A candidate that enters a wave proceeds immediately through pair legality,
   Beam-2, state validity, and L5 plan-only. Wave results merge at a stable
   barrier. Stop when a primary and distinct-grasp backup are proven; otherwise
   advance the frozen frontier without rerunning AnyPlace. No host waypoint is
   inserted.
5. After native contact+attach PASS, reuse the frozen AnyPlace goals and
   recompile them with the measured attachment and current PlanningScene
   revision. Do not rerun AnyPlace simply because a pair or later L5 plan fails.
   A new AnyPlace inference is allowed only after the cached 96-goal pool and
   bounded recovery budget are exhausted.
6. Execute one `move_to` from the current attached state to the exact compiled
   release EEF pose. Preserve its full orientation. There is no carry lift,
   pre-place hover, descend offset, rim clearance, release-height adjustment,
   adaptive near-target acceptance, or retreat.
7. Every attached MoveIt receipt must confirm the joint remains attached and
   relative drift stays within the native threshold. Loss of attachment before
   exact release is failure even if the object happens to be near or inside the
   receptacle.
8. At the exact release pose call `gripper_control position=1` once. Success
   requires detach ACK plus native stable placement: terminal drift, height,
   support, and full-footprint-in-zone checks must PASS. An empty gripper,
   proximity, a successful open, or reward alone is not placement proof.
9. When placement verification PASS is retained, close the simulator
   environment exactly once and report task completion. No post-release motion
   is part of the acceptance contract.
10. If MoveIt rejects before execution starts, reject only that pair and try
    the next already-qualified candidate, then continue the frozen pair
    frontier. Transport-unknown or service errors are infrastructure failures
    and must not be recorded as unreachable.

Never edit model goals, add offsets, retry an unchanged failed fingerprint, or
claim success without native state validity, complete MoveIt plan proof, and
stable in-zone release evidence.
