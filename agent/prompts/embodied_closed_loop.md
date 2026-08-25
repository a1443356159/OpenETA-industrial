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
