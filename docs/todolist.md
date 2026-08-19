# OpenETA TODO List

This list tracks the near-term OpenETA agent work derived from the RFC.

Primary execution path: closed-loop `tool_call`.
Optional path: bounded `code_policy` as an atomic-tool backend only.

## Agent Brain

- [x] Implement a multi-step closed-loop episode runner.
  - Run `observe -> plan -> tool_call -> tool result -> memory update -> observe`.
  - Use independent runner-owned turn/tool-call/time/token budgets; let the
    agent or env/checker terminate the episode before resource exhaustion.
  - Expose the loop through the `uv run openeta` CLI and `/run`.

- [x] Define the agent-side tool handler adapter contract.
  - Standardize `ToolResult.details` shapes for perception, planning, safety,
    bookkeeping, and world-mutating tools.
  - Provide dummy handlers for `scene_detector`, `sam3`, `anygrasp`,
    `move_to`, and `gripper_control`.
  - Ensure tool results can be recorded into session trace and working memory.

- [x] Add minimal checker hooks without locking final sub-agent schema.
  - Add protocol-style placeholders for safety and failure checker backends.
  - Keep `safe_check` as a named `tool_call` capability.
  - Record checker outputs through `metadata` or `ToolResult.details` until the
    shared schema is reviewed.

- [x] Improve planner context assembly.
  - Include relevant markdown skill guidance in planner context.
  - Keep context bounded with skill metadata and memory summaries.
  - Preserve `skill_call` as guidance-only, not hidden execution.

## Skills

- [x] Add `place.md` as text guidance under `agent/skills/`.
  - Use atomic tools such as `observe`, `scene_detector`,
    `obstacle_avoidance`, `move_to`, and `gripper_control`.

- [x] Add `push.md`, `pull.md`, and `stack.md` skeleton guidance.
  - Keep them as editable markdown skills with frontmatter.
  - Do not introduce macro execution.

- [x] Add a skill selection smoke test.
  - Verify markdown frontmatter loading.
  - Verify selected skill content appears in planner-facing context.

## Memory

- [x] Add CLI visibility for local memory.
  - Add a slash command or tool trace view for facts, artifacts, skill notes,
    compact summary, and session path.

- [x] Implement a compact policy for long sessions.
  - Keep explicit `compact_memory`.
  - Add automatic compaction when planner context reaches the configured
    context-window threshold.
  - Resolve model context windows from provider metadata when available; keep
    manual `OPENETA_LLM_CONTEXT_WINDOW_TOKENS` as the reliable fallback.

- [x] Add an explicit promoted-memory workflow.
  - Keep `.openeta_memory/` as gitignored runtime state.
  - Write to `agent/memory/` only through an explicit reviewed action.
  - Added `memory_extract` skill for agent-driven working-memory extraction;
    reviewed promotion into `agent/memory/` is handled by explicit CLI command.
  - Add `/promote-memory` as the reviewed CLI action for writing promoted
    markdown entries under `agent/memory/`.

## CLI And Runtime UX

- [x] Make CLI `/run` show multi-step tool-call traces.
  - Show planner request, tool parameters, result, memory update, and next turn.
  - Require permission for world-mutating dummy commands.

- [x] Add a dry-run example for model-backed planning.
  - Use existing OpenAI-compatible backend config.
  - Avoid requiring a real simulator.
  - Add `examples/model_backed_planner_dry_run.py` for planner-only backend
    calls without simulator step or tool execution.

- [x] Add a resume or session-id display path.
  - Show current `session_id`.
  - Show JSONL trace path when `JsonMemoryStore` is attached.
  - Add `/session` and print session trace path with episode output.

## Simulator And Tool Integration Boundary

- [x] Keep real simulator handler integration behind the tool registry.
  - Do not let planner call raw simulator APIs directly.
  - Use simulator MCP tools through `SimulatorMcpToolProxy` as the narrow
    boundary.
  - Verified remote MCP tools include `move_to`, `gripper_open`, and
    `gripper_close`.

- [x] Remove low-level simulator state fields from the agent-side observation
  requirement.
  - IK, safety checks, controller expansion, action clipping, EE pose
    convention, and backend action details are simulator-side responsibilities.
  - Agent-facing MCP results should carry only planner/perception-relevant
    information, tool outcomes, and materialized image refs.
  - Full camera intrinsics/extrinsics, mandatory object lists, step index, and
    full-state `observe_env` are no longer required for control execution.
  - Current remote smoke uses `render_env` for fresh camera refresh.

- [ ] Coordinate with perception/control owners on real handlers.
  - [x] `sam3` via remote MCP `segment` and `segment_points`
  - [x] `anygrasp` via remote MCP `detect_grasps`
  - `obstacle_avoidance`
  - [x] `move_to` via MCP `move_to`
  - [x] `gripper_control` via MCP `gripper_open` / `gripper_close`

- [x] Run a real MCP simulator smoke episode.
  - Completed against remote metaworld env
    `openeta/metaworld_50_assembly-v3-v0`.
  - Executed `create_env -> reset -> render_env -> move_to -> gripper_open ->
    render_env -> close_env`.
  - Confirmed MCP base64 images are materialized to local refs before planner
    context.
  - Any future MCP smoke/integration test that calls `create_env` must call
    `close_env` in `finally`; use `close_simulator_mcp_env()` for best-effort
    cleanup.
  - Note: remote MCP does not currently expose `observe_env`; `render_env` is
    used as the observe substitute.

## Gazebo Milestones

- [x] M5: opt-in real upstream SAM3 SSE MCP control-only integration.
  - Runs only after M0–M4 control gates through
    `scripts/tui_gazebo_acceptance.py --control-only --include-m5 --sam3-url URL`.
  - Requires one real `segment` candidate and an explicit host-only
    `scripted_single_candidate` selection; zero/multiple/Oracle/fake candidates
    fail before M3 motion.
  - Strictly associates source RGB, depth, mask, frame id and numeric
    `camera_to_world` extrinsics; evidence is case-local and hash-linked.
  - The result is `control_only_real_sam3_no_planner_not_formal_tui`, not a
    formal remote/PTy/LLM acceptance or perception benchmark.
  - Details: `docs/gazebo-m5-sam3-perception.md`.

- [ ] M4: Oracle perception remains a SAM3-shaped simulator-only perception
  module. M4 candidates must use M3's native-contact DetachableJoint gate.
  - [x] `oracle_perceive` reuses the SAM3 handler/contract/selection flow end
    to end (agent tool -> sim MCP -> bench_worker -> pure geometric projection
    in `extensions/gazebo/oracle_perception.py`).
  - [x] `OPENETA_PERCEPTION_PROFILE=sam3|oracle` (default `sam3`); exactly one
    segmenter tool is exposed to the planner per profile, with provenance
    marked in response metadata, artifacts, and Working Memory facts.
  - [x] Offline tests: 39 new (worker 16+13, agent contract 10) + 5
    execution-link patch tests; full offline suite 1320 passed, 0 failed.
  - [ ] Manipulation/live acceptance: require the guarded DetachableJoint
    path and remote TUI/MCP evidence; historical results are not acceptance
    evidence.
  - Out of scope: wrist projection, pick_place NOT_READY gate, unified object
    summary schema (plan.md §17), SAM3-side changes, M6 fine-tuning.
  - Details: `docs/gazebo-m4-oracle-perception.md`.

## RFC And Review

- [x] Review section 5 schema with collaborators.
  - Confirmed `tool_call` and `response` as the sufficient top-level command
    classification; `noop` is not part of the planner-facing command surface.
  - Confirmed `skill_call` guidance-only semantics.
  - Confirmed `SkillSpec` markdown/frontmatter fields.
  - Confirmed checker hook direction before adding hard schemas.

Durable contract changes should continue to be synchronized to the RFC with
the branch, commit hash, changed files, validation commands, and any open
contract questions. This is an ongoing collaboration rule rather than a
one-time implementation TODO.

## Demo Acceptance Path

- [x] Add bounded parallel simulator evaluation.
  - `uv run openeta-batch --manifest ... --concurrency 10` runs independent
    model-planned episodes in a thread pool.
  - Keep each episode's world-mutating tool loop serial; isolate sessions,
    traces, artifacts, failures, and cleanup.
  - Keep a hard concurrency limit of 32 and default to 10.
  - Use `success / need_human / fail` lifecycle labels and report autonomous
    versus assisted success separately.
  - Persist `need_human` session/interaction ids, close the expiring simulator
    handle, and restart the same env/task/seed with preserved Agent memory after
    a human answer.
  - Define executable `fail` budgets: 50 concrete tool calls, 600 seconds per
    MCP environment episode, and 5,000,000 cumulative model tokens; report a
    structured failure reason and keep the planner-turn guard separate.
  - Actively interrupt deadline-exceeded turns, request thread-safe environment
    cleanup, reject late step commits, and estimate missing provider token usage
    through the shared TUI token-counting module.

- [x] Run the real agent workflow through `uv run openeta` with multi-step
  closed-loop decisions.
- [ ] Run a real RLinf-backed episode once sim/tool handlers are ready.
  - [x] Validate the remote LIBERO RGBD -> SAM3 -> AnyGrasp -> control/cleanup
    chain against `openeta/libero_libero_10_task0-v0`.
  - [ ] Complete a model-planned long-horizon LIBERO task.
- [ ] Demonstrate one safety block and successful replan.
- [ ] Demonstrate one failure detection and recovery.
- [x] Persist complete episode trace data.
  - `JsonMemoryStore` writes the session event stream to
    `.openeta_memory/sessions/<session_id>/trace.jsonl` and keeps session-local
    working state alongside it.
- [ ] Implement offline episode reconstruction/demo replay from persisted
  traces.
- [ ] Validate no schema fields are missing across a full episode.
