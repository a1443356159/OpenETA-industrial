# Documentation map

This index separates normative release contracts from operator runbooks, implementation
designs, and historical or optional references. For the industrial Gazebo release, start with
the first table instead of reading the `docs/` directory alphabetically.

## Industrial `multi_normal` release

| Document | Status | Purpose |
| --- | --- | --- |
| [`multi-normal-tui-reproduction.md`](multi-normal-tui-reproduction.md) | Operator runbook | Canonical human TUI procedure, representative prompt, evidence checks, and task variants. |
| [`final-dev-validation-report.md`](final-dev-validation-report.md) | Release evidence | Current `final-dev` BaiLian Qwen `multi_normal` evidence, timing, and exact evidence roots. |
| [`gazebo-normal-acceptance.md`](gazebo-normal-acceptance.md) | Normative contract | Final task-neutral scene, model/geometry boundaries, physical proofs, recovery, and release gates. |
| [`openeta-qualification-funnel-v3.md`](openeta-qualification-funnel-v3.md) | Normative contract | `fast_v3` scheduling, Beam-2 IK, deterministic recovery, evidence, and promotion criteria. |
| [`gazebo-rm75-robotiq2f85.md`](gazebo-rm75-robotiq2f85.md) | Runtime profile | Robot, gripper, cameras, controllers, PlanningScene, and Gazebo launch behavior. |
| [`gazebo-adapter-design.md`](gazebo-adapter-design.md) | Implementation design | Ownership boundaries from MCP through the Gazebo runtime and native grasp mechanism. |
| [`gazebo-native-grasp-verification.md`](gazebo-native-grasp-verification.md) | Proof contract | Native bilateral contact, attach/detach acknowledgement, and measured attachment transform. |
| [`gazebo-unified-runtime-acceptance.md`](gazebo-unified-runtime-acceptance.md) | Runtime contract | Isolation, TUI provenance, observations, cleanup, and smoke-versus-agentic evidence. |
| [`gazebo-sam3-assets-and-deployment.md`](gazebo-sam3-assets-and-deployment.md) | Deployment runbook | Pinned SAM3 asset and isolated service deployment. |
| [`anygrasp-remote-deployment.md`](anygrasp-remote-deployment.md) | Optional runbook | Licensed AnyGrasp development backend. It is not part of the final `multi_normal` gate. |

The release scene does not contain a static work order. The operator's natural-language TUI
request supplies object order and destination bins. The release path uses GraspGenX, AnyPlace,
SAM3, MoveIt, and the `fast_v3` qualification profile. AnyGrasp remains available for separate
backend evaluation. The repository-wide rollback default may remain `legacy`; the release runner
selects `fast_v3` explicitly.

`3a70294` (2026-08-31) is a historical GPU-GUI validation baseline, not a
claim about the current release candidate. Current `final-dev` evidence and
source revisions for the shipped `multi_normal` flow are recorded separately
in the validation report. The operator runbook records the SSH/VNC tunnel,
health and VirtualGL checks, prompt, approvals, report queries, timings,
tokens, and evidence roots.

## Framework and interfaces

| Document | Purpose |
| --- | --- |
| [`architecture.md`](architecture.md) | Agent, Host, adapter, simulator, and MCP authority boundaries. |
| [`agent-action-pipeline.md`](agent-action-pipeline.md) | Tool schemas, AtomAction sequencing, observation obligations, and safety gates. |
| [`agent-framework-selection.md`](agent-framework-selection.md) | Planner/runtime framework decision record. |
| [`code-policy-runtime.md`](code-policy-runtime.md) | Bounded code execution and sandbox contracts. |
| [`env-registry-spec.md`](env-registry-spec.md) | Environment registration schema and validation rules. |
| [`env-backend-inventory.md`](env-backend-inventory.md) | Implemented backend inventory and provenance. |
| [`../sim/README.md`](../sim/README.md) | Simulation installation, MCP service, REST API, and Dashboard. |
| [`../agent/README.md`](../agent/README.md) | Agent runtime package and extension points. |
| [`../real/README.md`](../real/README.md) | Real-robot adapters, calibration, and safety boundaries. |

## Evaluation, evidence, and lifecycle

| Document | Status and purpose |
| --- | --- |
| [`rollout-data-contract.md`](rollout-data-contract.md) | Normative session, trace, artifact, and immutable-evidence layers. |
| [`task-playbooks.md`](task-playbooks.md) | Operator-authored task guidance without hidden execution. |
| [`parallel-simulator-evaluation.md`](parallel-simulator-evaluation.md) | General isolated simulator evaluation and experience promotion. |
| [`calibration-lifecycle.md`](calibration-lifecycle.md) | Calibration proposal, review, publication, and rollback. |
| [`grasp-strategy-lifecycle.md`](grasp-strategy-lifecycle.md) | Grasp strategy evidence and promotion lifecycle. |
| [`gazebo-industrial-benchmark.md`](gazebo-industrial-benchmark.md) | Offline/reference benchmark contract; not a live release result. |

## Source-of-truth order

When two descriptions appear to disagree, resolve them in this order:

1. executable schemas, profile configuration, and verifier code;
2. the normative acceptance or proof contract;
3. the operator/deployment runbook;
4. implementation design and inventory documents;
5. optional, benchmark, or historical reference material.

Acceptance artifacts under `.cache/reports/` are run evidence, not source documentation, and are
not committed. Documentation must not contain provider credentials, private keys, model licenses,
or unredacted operator secrets. A new release behavior should update its executable contract and
the corresponding normative document in the same change.
