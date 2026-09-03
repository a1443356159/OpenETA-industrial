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
does not make the initial model generation free.

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
