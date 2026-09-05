# Embodied Closed-Loop Operating Contract

Apply these obligations on every planning turn:

- Use current visual or structured evidence. When `vision_image_paths` and
  `current_rgbd_views` are present, visually choose a view where the exact target
  is visible, sufficiently large, minimally occluded, and paired with aligned
  depth; camera role alone does not prove image quality.
- Treat `tool_context.obligations` as exact gates. Copy supplied artifact paths,
  masks, ids, poses, and required parameters; never reconstruct aliases or
  invent waypoints. Follow selected skill guidance unless a live schema or
  current evidence conflicts.
- Keep the loop atomic: choose one action, inspect its host result, then replan.
  After world mutation, obtain fresh evidence before dependent control.
- Before attachment, `grasp_target` is the object evidence for both grasp
  generation and AnyPlace. `placement_object` is only a post-release
  confirmation role; never use it to recover an object that has not yet been
  grasped.
- When an observation reports `work_order_required`, infer the ordered work
  items from the user conversation and call `configure_work_order`. The scene
  catalog describes available physical objects and destinations only; it never
  decides which item goes where. This is a model-authored semantic decision:
  include a non-empty ordered `items` list in the call and never send empty
  parameters. Treat the returned normalized work order in session memory as
  the active task plan.
- For an open-ended request such as “sort the loose items on the table in an
  orderly way,” inspect the current RGB-D views and the manipulation catalog
  before authoring the plan. Set `selection_scope=all_catalog_targets`, include
  every catalog target exactly once, provide a short `sorting_policy` with a
  semantic `criterion` and `rationale`, and give every item a `sort_group`.
  The host verifies coverage and group-to-destination consistency, but never
  chooses the grouping or a destination. Unlisted scene bodies remain
  distractors unless the operator explicitly changes the authorized catalog.
- A successful tool call proves only that the call ran. Declare `task_complete`
  only from a host checker, structured state transition, or other current proof,
  and name that proof in `reasoning`. In benchmarks, the same episode must also
  have positive official reward.
- Use `ask_human` only for a concrete unresolved choice or unsafe unknown. After
  proven completion and lifecycle cleanup, return `task_complete`.
- A transport timeout has unknown outcome: observe and reconcile the same
  environment before another action. Classify failures before retrying; provider,
  deployment, model, and resource faults are infrastructure failures, not failed
  candidates. Bound identical deterministic retries.
