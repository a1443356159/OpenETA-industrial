# M7 工业基准（v0）：任务 manifest 与设计

## 当前结论

M7 第一步交付**任务 manifest 与 schema 设计**，不实现 runner。`examples/gazebo_industrial_benchmark_v0.json` 包含首批 5 类任务 × 2 seeds = 10 个 episode，字段风格与 `examples/parallel_libero_*.json` 完全一致，可直接被现有 `load_parallel_episode_manifest`（`agent/cli/batch_eval.py`）解析。评估不新建：复用 OpenETA rollout/evaluation 基础设施（plan.md §27、§31），物理成败真值复用 M3 verifier 的 `VerificationRecord`。

离线 manifest 校验测试通过；live 运行是 opt-in，且继承 M3 的 blocked 状态（夹爪连杆保真度问题，见 `docs/gazebo-m3-physical-verification.md`），M3 验收通过前本基准只提供 manifest 与指标定义，不产出正式结果。

## 任务类别（首批 5 类）

| task_class | 说明 | 对应 plan.md M7 变化轴 |
| --- | --- | --- |
| `single_grasp` | 单物体抓取并抬起（无放置） | 基线 |
| `pick_place` | 单物体抓取 + 放入标记区 | different placements（seed 间初始位姿变化） |
| `sort_to_bin` | 目标/干扰物身份区分，分拣入指定 bin | multiple object identities（seed1 交换目标身份） |
| `multi_sort` | 多物体分拣入多 bin | 多目标 + clutter（seed1 含 clutter 物体与 partial occlusion 声明） |
| `grasp_recovery` | 注入一次抓取失败，考察恢复 | failure injection（`grasp_pose_offset` / `drop_after_lift`） |

每类 2 个 seed；seed1 通过 `initial_xyz` 偏移提供放置变化。`occlusion`/`lighting` 轴在 v0 仅作 schema 声明（`"none"`/`"default"`/`"partial"`/`"low"`），物化依赖 world 模板扩展，属 follow-up。

## Manifest schema

顶层字段由 `ParallelEpisodeSpec.from_dict`（`agent/runtime/parallel.py`）定义，与 Libero manifest 相同：`episode_id`、`env_id`、`task`、`seed`、`max_turns`、`max_tool_calls`、`timeout_s`、`max_total_tokens`、`metadata`。runner 把 `metadata` 原样透传进 episode metadata，因此 Gazebo 专用参数全部放在 `metadata` 下，无需改动解析代码：

```text
metadata
├── suite: "gazebo_industrial"          # 与 Libero 的 suite 字段同义
├── benchmark_version: "v0"
├── task_class: 上表五类之一
├── split: "canary"                      # 沿用 Libero 的 split 约定
├── scene                                # 逻辑场景声明（物化由未来 runner 负责）
│   ├── objects[]: {id, label, role, kind, size_m, mass_kg, initial_xyz}
│   │       role ∈ target | distractor | clutter；kind ∈ box | cylinder
│   │       box size_m=[x,y,z]；cylinder size_m=[diameter,length]
│   ├── destinations[]: {id, label, center_xy, size_xy_m}
│   ├── goal: {target_id, destination_id}            # 单目标
│   │     或 {assignments: [{target_id, destination_id}, ...]}  # multi_sort
│   └── variation: {occlusion, lighting, clutter}    # v0 仅声明
└── failure_injection: null | {kind, ...params, max_injections}
        kind ∈ grasp_pose_offset | drop_after_lift
```

`env_id` 统一为 M3 环境 `openeta/gazebo_rm75_robotiq2f85_pickplace-v0`。物体几何/质量/位姿与放置区坐标的默认值逐项取自 `M3Config`（`extensions/gazebo/m3.py`）：目标 `0.04×0.04×0.06 m / 0.10 kg @ [0.28,-0.10,0.43]`，干扰物圆柱 `⌀0.05×0.08 m / 0.12 kg @ [0.28,0.12,0.44]`，放置区中心 `[0.48,-0.10]`、`0.12×0.12 m`。

`failure_injection` 语义（声明式，执行器属 runner follow-up）：

- `grasp_pose_offset`：首个抓取候选在执行前平移 `offset_m`，制造一次 `EMPTY_GRASP`/`TARGET_NOT_LIFTED`；
- `drop_after_lift`：首次验证持握后执行张爪→运动学 release，制造 `OBJECT_DROPPED`；
- `max_injections` 限定注入次数，防止 runner 把恢复循环本身当成注入。

## 指标采集：复用现有 rollout/eval 基础设施

不建第二评估器（plan.md §27、§31 红线）。分层指标的来源映射：

| plan.md §27 层 | 数据来源（现有） |
| --- | --- |
| Agent（task success / tool calls / planner turns / recovery count） | `ParallelEpisodeOutcome` + `EpisodeResult`（`agent/runtime/parallel.py`），`classify_episode_result` 判定 success/fail；recovery count 由 episode 内 M3 verdict 从 FAIL/UNKNOWN 回到 PASS 的次数统计 |
| Perception（recall / IoU / wrong-target） | SAM3 或 oracle_perceive 工具 artifact（M4 已统一契约），wrong-target 由 M3 `WRONG_OBJECT` reason code 佐证 |
| Grasp（候选生成 / 可达 / 物理成功） | 观测与动作回执中的 `metadata.physical_verification`（`m3_physical_verification_v1`），`TARGET_HELD`/`EMPTY_GRASP` 等 reason code |
| Placement（生成 / 物理成功） | 同上，place 阶段的 `TARGET_PLACED`/`OUTSIDE_DESTINATION`/`NOT_SETTLED` |
| Verification（TP/TN/false success/false failure/unknown） | **join** episode 最终状态与最终 `VerificationRecord`：episode success 且 verifier PASS = TP；episode fail 且 verifier FAIL = TN |

**false-success 率**是重点（plan.md §27）：定义为 episode 被 `classify_episode_result` 判为 success、但最终 M3 verifier verdict 非 PASS（FAIL 或 UNKNOWN）的比例。M3 的三态 verdict 与 fail-closed UNKNOWN 语义保证 planner 文本不能自证成功（plan.md §13），false success 只能在「环境 receipt 误报」路径上发生，因此该比率直接度量验证链路的诚实性。UNKNOWN 率单独报告，不并入 fail。

复现性（plan.md §31）：manifest 的 `seed` + scene 声明 + 现有 episode metadata（commit、batch_id 等由 parallel runner 自动注入）构成完整 provenance；Gazebo world 版本与 `grasp_mechanism=bilateral_contact_adhesion_v1` 已在 M3 观测 metadata 中标注，无需新增记录通道。

## 与 M3 验收报告的衔接

- 场景、物体、校验器、reason code 全部继承 M3（`docs/gazebo-m3-physical-verification.md`）；v0 manifest 不引入 M3 verifier 无法判定的新真值来源。
- M3 live benchmark 仅在同 SHA M0–M4 正式验收通过后产出正式结果。manifest 与离线校验可独立运行；真实抓取依据双垫接触吸附收据和 M3 verifier，而非旧式物理保持方案。
- `grasp_recovery` 类直接复用 M3 观察到的自然失败模式（`EMPTY_GRASP`、`OBJECT_DROPPED`）作为注入模型，注入是加速复现，不改变 verifier 语义。
- `multi_sort` seed1 的 `bin_b`、clutter 物体和 `occlusion/lighting` 变化超出 M3 固定场景，需要 world 模板参数化（spawn 额外实体、移动光源），属 follow-up；manifest 只声明逻辑场景，M3 场景 natively 支撑其余 9 个 episode。

## 测试与验收状态

- 离线新增 `tests/test_gazebo_m7_benchmark_manifest.py`：经 `load_parallel_episode_manifest` 解析、episode_id 唯一、5 类任务各 ≥2 episode、env_id 与 `M3_ENV_ID` 一致、scene/goal 引用完整性（goal 中的 id 均在 objects/destinations 中声明）、`grasp_recovery` 必带 `failure_injection`、无注入任务不得携带该字段。
- live 运行 opt-in：需 Gazebo/ROS 环境，不在本机启动；待 runner 实现后以 `OPENETA_RUN_LIVE_ROS_TEST=1` 门控，与 M3/M4 live 测试同约定。

## 明确不做

- runner / 场景物化代码（本任务只交付 manifest 与设计）；
- 新评估器、新轨迹记录器（plan.md §27/§31 禁止）；
- occlusion/lighting 的 world 模板实现（v0 仅 schema 声明）；
- SceneGraph（plan.md §1.5 禁止）；
- 对 `extensions/gazebo/` 现有模块与 M3/M4 测试的任何改动。
