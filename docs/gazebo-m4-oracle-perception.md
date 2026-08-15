# M4（修订版）Gazebo Oracle 感知模块

## 当前结论

M4 已按 2026-08-10 的用户决策修订落地，与 plan.md 原 M4（Full Oracle Pick/Place）有两点差异：

1. oracle 感知做成与 SAM3 **完全同形态**的感知模块——同一 MCP/工具契约、同一 selection flow、同一下游消费路径，按感知 profile 互换；plan.md §14 要求 oracle 不混入正式感知 profile，此处以显式 `OPENETA_PERCEPTION_PROFILE` 开关和三重 provenance 标注满足"preserve provenance"与"mark clearly as simulator-only"。
2. **不做** `pick_place` 编排工具（plan.md §22 的宏编排与 NOT_READY 门由用户免除）；执行使用分阶段 MoveIt 原语（`move_to` / `gripper_control`）。

目标环境：`openeta/gazebo_rm75_robotiq2f85_pickplace-v0`（M3 profile）。live
验收不再把 M3 失败转换为预期跳过：每个 seed 必须真实到达 `TARGET_HELD`，否则
整条 M4 链 fail-closed。

## 架构与数据通路

```text
agent oracle_perceive 工具
→ build_sam3_handler（复用，tool_name="oracle_perceive"）
→ build_oracle_perceive_segmenter（复用 simulator transport）
→ openeta-sim MCP oracle_perceive 工具（sim/mcp_server/server.py，
  经 worker_mgr._proxy_oracle_perceive 转发）
→ bench_worker 路由 POST /env/{handle}/oracle_perceive
→ extensions/gazebo/oracle_perception.py 纯几何投影核心
```

否决方案及理由：

- **独立 MCP server**：需要跨进程接入 ROS graph 获取物体真值与相机外参，部署与生命周期复杂度远超复用既有 worker 通路。
- **Gazebo segmentation camera**：需要同时修改 SDF、ros_gz bridge 和订阅路径三处，且产生的是类别级分割而非契约要求的实例 mask，改动更重。

## 投影核心语义

`extensions/gazebo/oracle_perception.py` 不依赖 ROS，输入为物体注册表、世界位姿、相机内外参和图像：

- box 取 8 个角点，cylinder 取两端圆盘各 16 个采样点；
- 经 `camera_to_world` 逆变换（OpenCV 相机系）进入相机系，针孔投影到像素平面；
- 取投影点凸包，用 PIL 填充多边形生成 `0/255` 单通道 mask，编码为 PNG base64；
- `bbox`/`area` 由 mask 重算，不直接使用几何投影值；
- 忽略遮挡——被挡住的部分也计入 mask（oracle 语义，非可见性估计）；
- prompt 对物体 id/name/label 做大小写不敏感的子串匹配；命中即 `score=1.0`；
- 响应逐字段对齐 `tools/sam3_core.py` 的 segment 契约，仅 `details.metadata` 增加标注，`metadata.perception_source="gazebo_oracle"`。

物体注册表由 M3Config duck-typing 构造：目标 box `0.04 × 0.04 × 0.06 m`，干扰物 cylinder `r=0.025 m, l=0.08 m`。M1/M2 profile 没有对应物体定义，返回 `oracle_unsupported_env`。

## 帧匹配与限制

worker 侧复用 bench_worker 既有 `_last_obs` 缓存，不新增观测通道。调用方传入的 `image_base64` 先按（尺寸 + 像素完全一致）匹配缓存帧；匹配不到时回退为尺寸唯一匹配，并在响应 `diagnostics` 标注 `frame_match="fallback_size"`。wrist 相机的外参（`tf_dynamic`）尚未数值化，对其请求显式失败 `ORACLE_FRAME_UNSUPPORTED`；wrist 投影支持是 follow-up。

## 感知 profile（plan.md §14 落地）

- `OPENETA_PERCEPTION_PROFILE=sam3|oracle`，默认 `sam3`，不设置时现状零影响。
- `build_default_tool_registry` 按 profile 只注册一个分割工具（`sam3` 或 `oracle_perceive`），两个工具**永不同时暴露**给 planner。
- provenance 三重标注：响应 `metadata.perception_source`、artifact 的 tool 字段、Working Memory fact 的 source（用实际工具名）。

## 上游最小补丁清单（plan.md §23）

| 文件 | 补丁 |
| --- | --- |
| `agent/runtime/memory.py` | `SELECTION_CAPTURE_TOOL_NAMES = {"sam3", "oracle_perceive"}`；`_capture_sam3_selection_state` 工具名匹配放宽，fact source 使用实际工具名 |
| `agent/tools/registry.py` | `oracle_perceive` ToolSpec；profile 常量与 `resolve_perception_profile` 解析 |
| `agent/tools/handlers.py` | `build_sam3_handler` / `_normalise_sam3_response` / `_build_sam3_selection_artifacts` 增加 `tool_name` 参数（默认 `"sam3"`，SAM3 行为不变）；新增 `build_oracle_perceive_segmenter`；`_parse_camera_extrinsics` 支持 pos + `quat_xyzw` 外参（Gazebo 观测外参形态） |
| `agent/tools/sim_mcp.py` | `_is_ranked_grasp_candidate_pose` 识别 `grasp_pose_estimate`/`graspgenx` provenance；抓取候选姿态默认 `preserve_current` 模式，防止把 GraspNet 抓取系姿态当 EEF RPY 静默误发 |
| `agent/runtime/runtime_assembly.py` / `planner.py` | profile 装配与系统提示 |
| `sim/bench_worker.py`、`sim/mcp_server/worker_mgr.py`、`sim/mcp_server/server.py` | worker 路由、proxy 转发、MCP 工具 |

SAM3 侧（`tools/sam3_core.py` 及其 MCP server）零改动。

## 分阶段执行链路核验结论（任务 A）

`move_to` / `gripper_control` 的参数映射与编排假设逐项兼容：位置、姿态约定、容差、二值 gripper 均一致。`world` 与 `base_link` 数值一致是部署级隐式约定（机器人 spawn 在原点），不在代码层断言。

已知限制：

- `plan_only` 预检不暴露到 MCP 面（有意设计，保持 MCP 面最小）。
- RM75/Robotiq 2F-85 没有 GraspNet→EEF 标定 profile（仅 Panda 有），因此 Gazebo 抓取保持位置级 `move_to` + `preserve_current`，不做姿态级抓取。
- memory 的 `grasp_candidate_gate` 对 `grasp_pose_estimate`/`anygrasp` 来源候选拦截 `camera_pose_to_world`（正式路径 `compile_grasp_seed` 需要标定 profile）；`contact_graspnet` 来源候选不受此限。

## 测试与验收状态

- 离线新增 39 项：worker 侧 29 项（`tests/test_gazebo_oracle_perception.py` 16 项 + `tests/test_gazebo_oracle_worker_route.py` 13 项）+ agent 契约 `tests/tools/test_oracle_perceive_tool.py` 10 项；另有执行链路补丁 5 项（`tests/test_simulator_mcp_proxy.py` 3 项、`tests/tools/test_camera_pose_transform_tool.py` 2 项），全部通过。
- 全量离线回归：`1320 passed, 14 skipped, 0 failed`。
- live：`tests/test_gazebo_m4_oracle_pick_chain.py`（`OPENETA_RUN_LIVE_ROS_TEST=1` 门控，默认 skip）完整链路为 `observe → oracle_perceive → select → fake grasp 候选（真值反推相机系，免 GPU）→ camera_pose_to_world → move_to(pregrasp/grasp) → gripper_close → lift → TARGET_HELD`。`OPENETA_M4_ORACLE_SEEDS` 可声明多个 seed；云端正式入口固定运行三个 seed。任何 motion、双垫接触或持物硬门失败都会失败，不再被预期跳过掩盖。

Oracle 只证明 simulator ground truth 的感知与工具契约；fake candidate 只证明
GraspNet-shaped 参数链路。两者不构成真实 SAM3 或 GraspNet 推理声明。

## 明确不做

- `pick_place` 编排工具与 NOT_READY 门（用户免除）；
- wrist 相机投影（follow-up）；
- plan.md §17 统一对象摘要 schema；
- SAM3 侧任何改动；
- M6 工业微调。
