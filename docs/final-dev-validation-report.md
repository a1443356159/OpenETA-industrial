# `final-dev` Qwen validation report

> **Status:** Release-candidate evidence ledger. This document records only
> artifacts that have been read and verified. It is not a substitute for the
> final three-by-three BaiLian promotion gate below.

## Scope

The source baseline used by the verified remote runs was `187b4df` (`final-dev`). It uses the task-neutral
RM75/Robotiq `multi_normal` world, `fast_v3`, GraspGenX with a frozen reserve
of 512 model candidates, AnyPlace with 96 object goals, MoveIt L5 plan-only
proofs, and the real PTY-based agentic TUI. The GUI is a case-owned observer;
it is never part of candidate selection.

The 512 reserve is deliberately the release default. Qualification uses small
deterministic waves and stops on the first complete proof. A 1024 reserve is
available only as an explicit coverage experiment after a 512-pool miss; it
does not make the initial model generation free. Its default ladder adds a
512-candidate wave before the terminal 1024 tail, retaining lazy deep
qualification rather than issuing one oversized final wave.

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

## Local delivery hardening after those artifacts

The verified artifacts above deliberately retain their source revision
(`187b4df`) rather than being relabelled as evidence for later code. The
current `final-dev` delivery candidate also contains the following
unit-tested, but **not yet physically re-run**, hardening changes:

| Revision | Change | Local evidence |
| --- | --- | --- |
| `b78164c` | An opt-in 1024 GraspGenX reserve extends the small-wave ladder through 512 instead of creating one 768-candidate deep wave. | candidate scheduling/configuration tests and full repository suite |
| `41dd0b9` | Provider `/models` discovery is advisory; the configured model's direct structured chat smoke remains the BaiLian compatibility gate. | provider-preflight tests and full repository suite |

These revisions are delivery preparation, not a substitute for the physical
promotion matrix below. A new remote run must record its exact `HEAD` in its
case receipt before it can be added to the verified table.

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
