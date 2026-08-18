# M5：真实 SAM3 感知到 M3 物理闭环

## 当前能力

M5 是一个严格 opt-in 的 control-only 集成验收。它复用 M3 的
`openeta/gazebo_rm75_robotiq2f85_pickplace-v0` 场景、固定轨迹和原生
`DetachableJoint`；不创建 world、URDF、软吸附、AnyGrasp 或 AnyPlace 流程。

运行：

```bash
scripts/tui_gazebo_acceptance.py --control-only --include-m5 --sam3-url "$OPENETA_SAM3_URL"
```

`--include-m5` 只能与 `--control-only` 一起使用，且必须给出一个已运行的
上游 SAM3 **legacy SSE** MCP endpoint（例如 `http://127.0.0.1:8773/sse`）。
验收脚本只发现其工具并调用一次 `segment`；它不启动、停止或配置该服务。

OpenETA 自带 SAM3 服务启动为 `dual`：标准 Streamable HTTP 在 `/mcp`，为 M5
兼容保留的旧 HTTP+SSE 在 `/sse`。当前 MCP 标准以 `/mcp` 为首选；新客户端应
使用它。M5 的首轮合约仍明确测试旧 `/sse`，所以不要把 `/mcp` 作为
`--sam3-url` 传入该验收命令。

SAM3 service 应使用隔离 Python 环境，并仅在该服务进程里注入具有官方模型访问
权限的 `HF_TOKEN`（或经哈希核验的本地官方 checkpoint/cache）。不要将 token 放进
仓库、命令行、`--sam3-url`、M5 evidence 或日志。若模型尚未获授权或无法下载，
M5 会以 `SAM3_INFERENCE_UNAVAILABLE` 阻塞，并且不会进入 M3 motion。

ModelScope checkpoint 的固定版本、SHA-256、离线 cache 布局和服务命令统一记录在
[Gazebo SAM3 资产与服务部署](gazebo-sam3-assets-and-deployment.md)，不要在各验收文档
中复制另一份权重清单。

同一 run 必须先严格通过 M0–M4；只有之后才创建 M5 的 M3 场景并执行以下链路：

```text
当前顶视 RGB-D observe（case-local artifacts）
→ 外部 SAM3 SSE MCP segment（真实文本分割）
→ 恰好一个候选
→ select_sam3_detection（host-only scripted_single_candidate）
→ 严格 RGB-D → m5_object_summary
→ 原有 M3 close / attach / lift / open / detach
```

它不调用 Planner Provider、LLM 或 PTY/TUI。成功报告的 scope 固定为
`control_only_real_sam3_no_planner_not_formal_tui`；这不是正式 TUI/远端验收，
也不证明视觉泛化能力或工业 benchmark 成绩。

## 受限感知 bridge

`extensions/gazebo/perception_summary.py` 保留通用的离线几何函数
`summarize_detection`，它可支持 mask resize。真实 M5 路径使用更严格的
`build_m5_object_summary`：

- 仅接受本次 observe 的 `scene_primary` 顶视 RGB、depth、intrinsics 和数值
  OpenCV `camera_to_world` 外参；wrist 的 `tf_dynamic` 不可用。
- 已选 detection 必须有 case-local `mask_ref`、`source_image` 和
  `source_frame_id`；RGB 路径和 frame id 必须与当前 observe 完全相同。
- RGB、depth、mask 都必须位于 case root 内，RGB/depth/mask 的尺寸必须完全一致。
  depth 仅从当前 uint16 PNG 按显式 scale 还原为 metres；M5 不重采样或猜测。
- mask 内有限、正 depth 的中位数与像素质心经针孔反投影和外参转换成世界系
  `position`。不能得到有限位置即失败。

输出的单物体 summary 带 `provenance="sam3_perception"`、源相机与 SAM3 score；
不编造 6-DoF 姿态或 visibility。

## 选择与物理门

M5 没有 LLM 选择。SAM3 必须返回恰好一个候选，仍通过既有
`select_sam3_detection` 合约记录为 `selection_source="scripted_single_candidate"`。
这是显式 host-only 选择，不是按 score 自动选择。

零候选、多候选、response 结构错误、Oracle、contractual fake candidate、帧或
artifact 不匹配都会在进入 M3 motion 前停止。通过感知门后，M3 的规则未变：
必须有双垫 native contact、attached ACK、至少 80 mm child-link lift、最多 10 mm
capture-relative translation 和 detached ACK。

## 证据与状态

每个 M5 case 生成 `m5-perception.json` 和 `m5-object-summary.json`。前者关联：

- observe MCP receipt、RGB/depth SHA-256、source frame/intrinsics/extrinsics；
- SAM3 tool catalog、脱敏 endpoint id、已脱敏 request/response、tool-result 和 mask；
- selection record、object summary 和 M3 response receipts；
- M3 动作后才记录的 `used_for_control=false` Gazebo 真值 evaluation metadata。

证据绝不写 URL query、凭据、图像 base64 或模型密钥。外部服务不可达、缺少
`segment`、模型加载或推理环境不可用标为 `blocked`；结构、候选或 RGB-D 合约错误
标为 `failed`。两种情况都禁止 M3 motion。

普通单元测试使用受控 MCP stub。真实 SAM3 集成应只在
`OPENETA_RUN_M5_LIVE=1` 并显式提供 `--sam3-url` 时启用。

运动工具的成功回执还必须通过动作后的新鲜 TF 验证。回执会给出
`position_error_m`、`orientation_error_rad` 及对应 verification tolerance；即使
MoveIt terminal result 为成功，实际位姿越界也必须返回
`MOTION_TARGET_NOT_REACHED`、`ok=false`、`reached_target=false`，不能进入下一物理门。

## 未覆盖范围

- AnyGrasp、AnyPlace、grasp/planner provider 与 M6 工业微调；
- wrist 外参数值化、6-DoF 位姿估计和遮挡启发式；
- 感知精度阈值、视觉 benchmark 或正式 PTY/TUI 验收。
