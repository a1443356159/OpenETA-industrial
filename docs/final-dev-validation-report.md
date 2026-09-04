# `final-dev` Qwen validation report

> **Status:** Release-candidate evidence ledger. This document records only
> artifacts that have been read and verified. It is not a substitute for the
> final three-by-three BaiLian promotion gate below.

## Scope

The current source baseline is `7cbe9ef` (`final-dev`). It uses the task-neutral
RM75/Robotiq `multi_normal` world, `fast_v3`, GraspGenX with a frozen reserve
of 512 model candidates, AnyPlace with 96 object goals, MoveIt L5 plan-only
proofs, and the real PTY-based agentic TUI. The GUI is a case-owned observer;
it is never part of candidate selection.

The 512 reserve is deliberately the release default. Qualification uses small
deterministic waves and stops on the first complete proof. A 1024 reserve is
available as an explicit coverage experiment. It reuses GraspGenX's fixed
model-native inference draws, but increases host-side selection, collision
filtering and frozen-result serialization; physical evidence must therefore
establish its recall/latency trade-off before it becomes the default. Its
default ladder adds a 512-candidate wave before the terminal 1024 tail,
retaining lazy deep qualification rather than issuing one oversized final
wave.

## Current exact-revision `multi_normal` evidence

On 2026-09-04, the representative two-item work order completed twice in
fresh, isolated run roots at exact Git head
`7cbe9ef9e751322771a2d6aeaa80ac13722277b1`. Each run used the configured
official BaiLian-compatible Qwen Vision deployment, `qwen3-vl-flash`, the real
agentic PTY, GPU Gazebo GUI, GraspGenX, AnyPlace and `fast_v3`; neither used a
host macro nor an `ask_human` recovery.

| Work order | Evidence root | Result | Tool calls | Planner tokens |
| --- | --- | --- | ---: | ---: |
| yellow adjustable wrench → green parts bin; red hex bolt → blue parts bin | `/root/autodl-tmp/openeta-final-dev-qwen-validation/multi-normal-seedfix-a/` | PASS | 17 | 115,366 |
| same work order, fresh scene/run root | `/root/autodl-tmp/openeta-final-dev-qwen-validation/multi-normal-seedfix-b/` | PASS | 17 | 113,701 |

Both `environment-receipt.json` files record the exact Git head above. Their
`acceptance-report.json` files record `agentic_closed_loop`,
`host_dispatch_count=0`, `fast_v3`, GraspGenX, `status=passed` and no verifier
errors. Per assignment, SAM3, AnyPlace and GraspGenX each ran once; placement
reused the frozen AnyPlace pool rather than invoking another model inference.

This is the current stable representative release path. It is intentionally
not misrepresented as completion of the broader three-by-three promotion
matrix below: the remaining task classes and random-layout rows need their own
fresh evidence before a wider release claim.

## Verified recent evidence

All paths below are remote evidence roots, not repository fixtures. Their
reports show `status: passed`, `fast_v3`, a real planner provider, and no
residual case-owned Gazebo/MCP processes after lifecycle close.

| World / work order | Evidence root | Result | Tool calls |
| --- | --- | --- | ---: |
| `multi_normal`; bolt → blue bin, wrench → green bin | `/root/autodl-tmp/openeta-final-dev-fourbar-validation/multi-512-2-reversed-tfhistory-unique/` | PASS | 17 |
| `multi_normal`; wrench → blue bin, bolt → green bin | `/root/autodl-tmp/openeta-final-dev-fourbar-validation/multi-512-3-wrench-blue-bolt-green-packetfix/` | PASS | 17 |
| `multi_normal_random_12345`; bolt → green bin, wrench → blue bin | `/root/autodl-tmp/openeta-final-dev-fourbar-validation/multi-random-512-1-packetfix/` | PASS | 17 |

The last two runs also record clean case lifecycle close: the GUI, Gazebo
partition and MCP server were all retired without touching unrelated services.
The packet-fix run verifies that a successful `active_observe` keeps its own
fresh observation packet instead of forcing an immediately redundant passive
observation.

## Delivery hardening lineage

The historical artifacts above deliberately retain their source revision
(`187b4df`) rather than being relabelled as evidence for later code. The
current exact-revision runs in the preceding section cover the accumulated
delivery candidate, including the following hardening changes:

| Revision | Change | Local evidence |
| --- | --- | --- |
| `b78164c` | An opt-in 1024 GraspGenX reserve extends the small-wave ladder through 512 instead of creating one 768-candidate deep wave. | candidate scheduling/configuration tests and full repository suite |
| `b061c80`, `afaff5a` | The 1024 reserve is documented as a host-side coverage expansion, and a regression test proves it cannot add model-native GraspGenX draws. | targeted 512/1024 draw-contract test |
| `41dd0b9` | Provider `/models` discovery is advisory; the configured model's direct structured chat smoke remains the BaiLian compatibility gate. | provider-preflight tests and full repository suite |
| `35e67ec` | A returned, known gripper controller terminal result after native detach is retained as telemetry instead of forcing a false human recovery; rejections and unknown outcomes remain strict. | Gazebo controller/release tests; included in current physical runs |
| `7cbe9ef` | A recovery screen preserves the candidate's deterministic fast base seeds before adding six unique fixed supplements; mutable batch cache cannot displace the base branch. | funnel regression tests; included in current physical runs |

These revisions do not substitute for the remaining physical promotion matrix
below. Every new row must still retain its exact `HEAD` in the case receipt.

## Provider-only compatibility check

On 2026-09-04, the current repository's native provider-preflight was run
against the configured official BaiLian `qwen3-vl-flash` workspace endpoint
without starting Gazebo, MCP services, or a TUI. The endpoint advertised the
selected model; model discovery completed in about 157 ms and the direct
structured chat smoke completed in about 432 ms (about 589 ms total). A
separate one-image structured request also completed successfully through the
same OpenETA client path.

The same client then received the representative Chinese two-item work order
with the task-neutral manipulation catalog. It selected
`configure_work_order` and returned the exact ordered catalog assignments
`yellow wrench → green parts bin`, then `red hex bolt → blue parts bin`, with
no schema retry (about 2.2 s; 1,191 provider tokens). This is a narrow
agentic semantic check, not a substitution for perception, qualification, or
physical execution.

A host-bound `grasp_contact` decision was also checked: the model chose
`move_to` and returned `parameters={}`, correctly leaving the qualified pose
and trajectory to the immutable host binding (about 1.6 s; 1,183 provider
tokens). This guards against an otherwise costly failure mode in which a
planner tries to reconstruct geometry that it was never given.

For a broader non-physical work-order check, the same client parsed one real
single-item, one double-item, and one three-item `multi_normal` Chinese work
order against the full task-neutral catalog. All three returned the exact
ordered catalog assignments in one valid `configure_work_order` call:

| Class | Fixture | Result | Provider latency | Provider tokens |
| --- | --- | --- | ---: | ---: |
| Single | `screwdriver-green` | exact ordered assignment | 1.93 s | 1,069 |
| Double | `wrench-green-bolt-blue` | exact ordered assignments | 2.06 s | 1,142 |
| Multi | `three-tools-a` | exact ordered assignments | 2.64 s | 1,195 |

These are semantic-control checks only. They show that expanding a work order
does not require a second model repair turn in this client path; they do not
prove grasp, placement, collision, or physical-release success.

For deterministic-control evidence, the representative double-item Chinese
work order was submitted to the same endpoint ten independent times. All ten
responses selected `configure_work_order` with the same valid ordered catalog
assignments and required no schema repair. Provider latency was 1.92–3.01 s
(P50 2.31 s). This is deliberately scoped to the bounded semantic decision;
the physical promotion matrix still requires fresh full episodes.

This proves that the selected endpoint accepts the release client's current
OpenAI-compatible structured and visual request shapes. It is deliberately
**not** a physical-sort result, a throughput claim for full agent contexts, or
promotion evidence. Every physical case must still retain its own redacted
`provider-preflight.json` under its run root.

## Promotion matrix

The final delivery must be run with the official BaiLian Qwen Vision endpoint
and a clean Docker image. A row is promoted only after **three independent
PASS** runs, each with a fresh run root and a clean lifecycle receipt:

| Work order class | Contract fixture | Required proof |
| --- | --- | --- |
| Single physical sort | `screwdriver-green` (and at least one geometrically different single object) | 3 × PASS |
| Double physical sort | default `wrench-green-bolt-blue` plus order/bin variation | 3 × PASS |
| Multi physical sort | `three-tools-a` or `mixed-tools-b` | 3 × PASS |
| Generalization | `multi_normal_random_12345` with a non-duplicate work order | 3 × PASS |

Every promoted report must show the configured BaiLian model and redacted
endpoint provenance, `agentic_closed_loop`, `fast_v3`, one frozen model pool
per item, MoveIt state-validity/L5 evidence, native attach/detach evidence,
and clean lifecycle shutdown. A provider or transport outage remains an
infrastructure event and cannot be counted as a geometric failure.

## Reproduction

Use the Docker commands in
[Ubuntu Docker deployment](ubuntu-docker-deployment.md#百炼-qwen-plannervlm).
The default `agentic-normal` command runs a two-item `multi_normal` task;
select the single-, multi-item, or random variants explicitly as needed. For
interactive VNC/TUI reproduction, use
[the operator runbook](multi-normal-tui-reproduction.md).

The report is intentionally evidence-first: it leaves the BaiLian promotion
matrix unclaimed until the matching artifacts have been produced and checked.
