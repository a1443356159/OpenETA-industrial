# M5 感知桥接：detections → 3D 对象摘要

## 当前结论

M5 的目标是 plan.md §M5 的验收口径——"same manipulation flow works with SAM3 perception"。M4 已交付与 SAM3 同契约的 oracle 感知（见 `docs/gazebo-m4-oracle-perception.md`），但 SAM3/oracle 的输出都是**像素级**（mask + bbox + score），而 planner 消费的对象摘要是 plan.md §17 的 **3D 级**条目。本文档对应本次交付的桥接模块：

- 新模块 `extensions/gazebo/perception_summary.py`（纯 Python，无 ROS import，numpy/PIL 懒加载，风格对齐 `oracle_perception.py`）；
- 新测试 `tests/test_gazebo_perception_summary.py`（14 项，全离线）；
- **不接线**到 agent 工具层/worker/MCP 面——接线属于后续 M5 任务；当前 M3/M4
  manipulation 仍须经过 M3 的原生接触、joint ACK 和 child-link 物理证明；Oracle
  离线契约不改变该边界。

## 管线定位

```text
top RGB-D observe（CameraFrame: rgb/depth/intrinsics/extrinsics）
→ SAM3 segment（OPENETA_PERCEPTION_PROFILE=sam3，detections: mask/bbox/score）
→ selection flow（既有，选中 detection 带 mask_ref + source_image）
→ perception_summary.summarize_detection   ← 本模块
→ plan.md §17 对象摘要条目（世界系 position）
→ Observation / Working Memory（后续接线）
```

Oracle profile 下同一模块同样可用（oracle 与 SAM3 同检测契约），这使 M5 的对比口径可以是：**同一桥接、同一下游，仅切换感知 profile**。

## 输入与输出 schema

输入：

- `detection`：选中的 SAM3/oracle detection。mask 取 `mask_ref`（单通道 PNG 路径，前景 >0）或内联 `mask={"format":"png","base64":...}`；`id`/`label`/`score` 分别映射到摘要字段；`source_image` 按 selection 契约接收但不做像素级复核——detection 与 CameraFrame 的帧对应关系由调用方保证（fresh-observation 语义；M4 中这类匹配在 worker 侧完成）。
- `camera_frame`：`adapter/protocol.py` 的 `CameraFrame`（dataclass 或等价 mapping，duck-typed），depth 为 metres 的 H×W 数组。

输出（plan.md §17 形态）：

```json
{
  "id": "det_0",
  "label": "target block",
  "confidence": 0.96,
  "position": [0.39, -0.01, 2.0],
  "visibility": "unknown",
  "source_camera": "top_camera_optical_frame",
  "provenance": "sam3_perception"
}
```

`build_object_summary` 把多个 detection 包成 §17 的 `{"objects": [...]}`。

字段口径：

- `position`：世界系 `[x, y, z]`（米）。无法诚实计算时为 `null`，并附 `position_error` 原因——不编造（§17："Do not invent precise orientation just to fill a schema / Use unknown / missing fields where appropriate"）。
- `visibility`：恒为 `"unknown"`。SAM3 mask 是可见区域本身，不包含遮挡状态信息；不做图像边缘截断之类的启发式猜测。
- `confidence`：直接取 detection 的 `score`；缺失/非数值时为 `null`。
- `source_camera`：CameraFrame 的 `frame_id`（稳定后端标识，不改写）。
- `provenance`：默认 `"sam3_perception"`，可参数化（如 oracle profile 复用时显式覆盖），延续 M4 的 provenance 标注要求。

## 几何语义

1. mask 解码为布尔数组；与 depth 分辨率不一致时按最近邻重采样到 depth 尺寸（与 agent 侧 selection overlay 的处理一致）。
2. 有效样本 = mask 内且深度有限且 >0；为空则 `position=null`（`no_valid_depth`）。
3. `z` = 有效深度**中位数**（对 mask 边缘混入背景深度 Robust）；`(u, v)` = 有效像素质心。
4. 针孔反投影：`x=(u-cx)·z/fx, y=(v-cy)·z/fy`。
5. `camera_to_world` 外参（OpenCV 相机系，pos + quat_xyzw）旋转平移到世界系，复用 `extensions/gazebo/m3.py` 的 `quaternion_rotate`。

`position_error` 取值：`mask_missing` / `mask_decode_failed` / `empty_mask` / `depth_missing` / `no_valid_depth` / `invalid_intrinsics` / `unsupported_extrinsics`。

## 与 oracle 的对比口径

| 维度 | oracle（M4） | SAM3 + 本模块（M5） |
| --- | --- | --- |
| 物体发现 | 真值注册表 + prompt 子串匹配 | SAM3 文本提示分割 |
| mask 来源 | 几何投影栅格化（忽略遮挡） | 模型推理（可见区域） |
| score | 恒 1.0 | 模型置信度 |
| 3D position | 真值直接可得（未走摘要） | mask 内深度中位数 + 反投影 |
| provenance | `gazebo_oracle` | `sam3_perception` |

M5 的 A/B 评测应保持桥接与下游不变、只切 `OPENETA_PERCEPTION_PROFILE`，position 精度差异即感知质量差异。真值仅用于评测（plan.md M5："Keep Gazebo ground truth only for evaluation"）。

已知系统误差来源（诚实声明）：mask 边缘像素的深度混合（中位数仅部分缓解）、质心对非对称 mask 的偏移、深度噪声。对抓取用途，position 是对象级参考点而非 6-DoF 位姿；朝向不编造（§17）。

## wrist 相机 tf_dynamic 限制

与 M4 oracle 相同的限制：wrist 相机外参在 profile 中是 `{"frame_transform": "tf_dynamic"}`（`extensions/gazebo/profiles.py`），尚未数值化。本模块对非 `camera_to_world` 外参显式返回 `position=null, position_error="unsupported_extrinsics"`，而非编造。wrist 外参数值化（TF 查询接入）是 follow-up，届时本模块无需改动——外参变为数值即自动支持。

## 测试与验收状态

- `tests/test_gazebo_perception_summary.py` 14 项全离线通过：手算反投影（恒定深度块、中位数选取、无效深度剔除）、外参旋转平推手算、depth 缺失、空 mask、mask 无有效深度、`tf_dynamic` 拒绝、无效内参、mask 缺失、内联 base64 mask、分辨率重采样、`CameraFrame` dataclass 输入、`build_object_summary` 聚合。
- 运行方式：`.venv/bin/python -m pytest tests/test_gazebo_perception_summary.py -q`，无需 live 环境（不起 Gazebo）。
- live 链路（SAM3 → 摘要 → 操控复跑）属于 M5 后续接线任务，本次不验收。

## 明确不做

- 不接 agent 工具层 / worker / MCP 面（后续 M5 任务）；
- wrist 相机外参数值化（follow-up）；
- 6-DoF 姿态估计、遮挡/visibility 启发式；
- SAM3 侧任何改动；
- M6 工业微调。
