# M4 Gazebo Oracle perception

`oracle_perceive` projects simulator truth into SAM3-shaped masks, boxes and
scores and explicitly marks `perception_source="gazebo_oracle"`. It is not
visual inference. The optional fake grasp candidate is likewise a
contract-shaped input fixture, not a predicted grasp.

M4 may route its candidate through M3 only. It must still obtain the real
dual native-contact gate and official DetachableJoint attach ACK at the exact
contact terminal; neither Oracle output nor a fake candidate can create a
joint or prove a grasp. Reports must state both the Oracle provenance and the
fake-candidate boundary. No lift waypoint or displacement threshold is used.

The repository contains implementation and offline contracts, not a claim
that the M4 remote formal acceptance has passed.

For a formal M4 case, the executed `oracle_perceive` call is also a simulator
MCP evidence item: its request descriptor, case-local materialized response,
and environment receipt share a request id. A missing or mismatched Oracle
MCP chain fails verification just as a missing control-tool chain does.
