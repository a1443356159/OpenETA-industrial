# DetachableJoint hard-gate reproductions

Run after sourcing the Jazzy and workspace overlays:

```bash
python extensions/gazebo/detachable_joint_repro/run.py --case all --cycles 3
```

The initial rows cover the required single-link positive control and a
non-canonical fixed terminal link in an articulated parent. `summary.json`
records only the `gz.msgs.Pose_V` relative-pose measurements as pass/fail
evidence. Plugin state-topic output is diagnostic and cannot pass a row.

The RM75 row starts the production M3 launch in `attachment_mode:=detachable`
and drives the real arm controller through `FollowJointTrajectory`; it never
teleports the robot parent. It measures three clear-pad
`detach → attach → 0.35 rad parent motion → detach` cycles for **each** target.
The production row uses `joint_1`, not wrist-roll `joint_7`: its world Pose_V
motion gives the detachable parent a measurable lever arm.
Before each attach it moves only the payload to a pose 0.20 m above the live
gripper mount, outside the open pads; the RM75 itself is moved exclusively by
`FollowJointTrajectory`.

```bash
python extensions/gazebo/detachable_joint_repro/run_rm75.py --cycles 3
```

For a focused positive-controller diagnosis, `--skip-contact-negative` omits
the deliberately disruptive contact row; the formal default includes it.

The plugin state topic is retained only as a request-ACK diagnostic. It never
passes the hard gate. For the RM75 row, the parent-to-**child-link** Pose_V is
the physical source; a matching object-model pose is corroboration only. The
`<1 mm / <0.5°` attached child-link drift and the post-detach Pose_V separation
must both hold. The detached pose must differ by either
`>50 mm` translation or `>0.05 rad` rotation. The in-contact reattach row remains a negative
diagnostic and is not accepted as physical evidence.
