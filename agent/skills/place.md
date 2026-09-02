---
name: place
description: Closed-loop guidance for qualifying AnyPlace goals and releasing an attached object in a receptacle.
version: v2
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

This is compact decision guidance, not an executable macro. Choose every tool
from fresh feedback. AnyPlace predicts object goals; the host proves geometry
and MoveIt owns complete robot paths.

## Normal flow

1. In combined pick-place, select the target object and the flat support inside
   the receptacle once from calibrated RGB-D. Do not segment the whole bin, rim,
   or walls as its support. Call AnyPlace once and keep all 96 object goals
   private; they are not VLM waypoints.
2. The host checks each goal once for finite SE(3), valid rotation, legal
   support/height/footprint, static penetration, and certain workspace bounds.
   Only pairs entering the current small wave pay the deeper geometry cost.
3. For a flat support, derive the exact release EEF pose as
   `object_goal * inverse(measured_attachment)`. For a collision-backed
   container, AnyPlace still selects the destination and proves a settled pose,
   while the host derives a geometry-backed gravity-drop terminal at that
   destination that keeps the carried orientation. A flat support uses the
   configured drop. A container may prefer its lowest geometry-proven full
   exterior entry edge plus clearance; internal or suspended obstacles do not
   qualify. If that batch exhausts IK/L5, the host retries configured height
   under the same gates before expanding the frozen grasp frontier.
   Its Z is host evidence, not an agent waypoint; gravity owns settling after detach.
   Pair legality, Beam-2, state validity, and L5 plan-only must all pass in
   either case. Stable wave barriers preserve deterministic order; use the
   first complete pair.
4. After native attach PASS, recompile the frozen AnyPlace pool from the
   measured attachment and current PlanningScene. Do not rerun AnyPlace or the
   grasp model because one pair fails; continue the frozen pair/grasp frontiers.
5. Execute one `move_to` to the exact host-qualified release EEF pose with its
   full compiled orientation.
   There is no carry lift, hover, descent waypoint, agent-authored offset,
   near-target shortcut, or retreat. Every receipt must retain attachment and
   satisfy the native drift bound.
6. At the qualified release terminal, call `gripper_control position=1` once. PASS requires
   detach ACK and native stable placement. After Gazebo confirms physical
   detach and collision-filter state, opening may proceed while the host
   synchronizes the same detach into PlanningScene; the ordered receipt must
   prove both. A successful open, empty gripper, visual proximity, or reward
   alone is not proof.
7. On retained placement PASS, close the environment once. If planning rejects
   before execution, consume only that pair. If execution leaves a known safe
   state, follow the typed frozen-frontier recovery. Unknown transport state
   requires observation on the same handle. Infrastructure errors are never
   candidate failures.

Never hand-edit model goals, add agent-authored offsets, retry an unchanged failed fingerprint, or
claim success without native state validity, complete MoveIt plan proof, and
stable in-zone release evidence.
