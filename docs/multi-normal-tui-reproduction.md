# `multi_normal` 人工 TUI 复现指南

> **文档状态：** `final-dev` 的 `multi_normal` 交付操作手册。当前发行范围只要求本文的
> 代表性双物件工单稳定通过；验收语义以
> [`gazebo-normal-acceptance.md`](gazebo-normal-acceptance.md) 为准，资格漏斗细节见
> [`openeta-qualification-funnel-v3.md`](openeta-qualification-funnel-v3.md)。

本文用于复现最终发行验收：人类操作员在同一个任务中，先把黄色活动扳手放入绿色零件箱，再把红色六角螺栓放入蓝色零件箱。物理场景始终是任务中立的 `multi_normal`；物品顺序和目标料箱来自操作员在 TUI 中输入的自然语言，不由场景静态注入。

## 验收边界

- 使用真实 OpenETA TUI、VLM、SAM3、GraspGenX、AnyPlace、MoveIt 和 Gazebo 链路。
- 使用 `fast_v3` 资格漏斗和 GraspGenX；本发行验收不启动 AnyGrasp。
- Gazebo GUI 在 GPU VNC 桌面上保持开启，Dashboard 仅用于观察。
- 每件物品只运行一次模型推理；物理失败优先从冻结候选前沿恢复。
- VLM 配置工单后，每件物品用一次宿主绑定的 SAM3 assignment 调用依次完成目标和料箱分割；两份 mask、选择审核、attempt ID 和 artifact 仍相互独立。当前 GPU 服务内部串行执行这两个 prompt，不以并发争抢显存。
- 人类只描述任务、处理真正的语义澄清，并确认世界状态变更；不输入末端位姿、偏移量、候选编号或工具调用顺序。
- Agent 保留正常工程闭环的决策自由；验收脚本不规定回合数、工具调用数、观察/重试次数或恢复顺序，也不使用跨观察、GraspGenX 或 AnyPlace 的全局失败熔断器。

## 1. 准备

从操作员电脑连接服务器。当前 VNC 只监听服务器 loopback，因此先建立 SSH 隧道，再让 RealVNC 连接本机 `127.0.0.1:5903`：

```bash
ssh -N -L 5903:127.0.0.1:5903 hhh
```

另开一个终端登录服务器，在 `final-dev` 工作树根目录执行。当前部署路径如下；若部署位置变化，只替换 `REPO`，不要把旧工作树加入 `PYTHONPATH`。

```bash
ssh hhh

# Override OPENETA_REPO if this server uses a different final-dev worktree.
export REPO="${OPENETA_REPO:-/root/autodl-tmp/openeta-industrial-workstation/source/worktrees/final-dev-qwen-validation}"
test -d "$REPO/.git" || { echo "not an OpenETA worktree: $REPO" >&2; exit 2; }
cd "$REPO"
git status --short --branch
git rev-parse --short HEAD

# Prefer the isolated runtime paired with this worktree.  A repository-local
# .venv remains supported for a conventional checkout, but never borrow a
# Python/ROS overlay from a different worktree.
if [[ -x "$REPO/.openeta_runtime/venv/bin/python" ]]; then
  export OPENETA_PYTHON_EXECUTABLE="$REPO/.openeta_runtime/venv/bin/python"
  export OPENETA_GAZEBO_OVERLAY="$REPO/.openeta_runtime/ros-overlay"
elif [[ -x "$REPO/.venv/bin/python" ]]; then
  export OPENETA_PYTHON_EXECUTABLE="$REPO/.venv/bin/python"
else
  echo "no OpenETA Python runtime found beneath $REPO" >&2
  exit 2
fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export DISPLAY=:3
export OPENETA_GAZEBO_DISPLAY=:3
export OPENETA_GAZEBO_OPERATOR_GUI=1
export OPENETA_GAZEBO_GUI_FPS=30

"$OPENETA_PYTHON_EXECUTABLE" scripts/openeta_mcp_services.py health sam3 \
  --host 127.0.0.1 --sam3-port 8773 --json
"$OPENETA_PYTHON_EXECUTABLE" scripts/openeta_mcp_services.py health graspgenx \
  --host 127.0.0.1 --graspgenx-port 8778 --json
"$OPENETA_PYTHON_EXECUTABLE" scripts/openeta_mcp_services.py health anyplace \
  --host 127.0.0.1 --anyplace-port 8775 --json

DISPLAY=:3 vglrun -d egl -c proxy glxinfo -B \
  | grep -E 'OpenGL vendor string|OpenGL renderer string'
```

三项健康检查都应返回 `"ok": true`，VirtualGL 检查应显示 `NVIDIA Corporation` 和 NVIDIA GPU。裸 `DISPLAY=:3 glxinfo` 可能显示 llvmpipe，因为 TigerVNC 只提供 2D framebuffer；Gazebo GUI 由 `vglrun` 重定向到 GPU，判断时以带 `vglrun` 的结果为准。用 RealVNC 连接上述 SSH 隧道，确认桌面可见；验收启动后 Gazebo GUI 会自动打开。Provider 凭据由部署目录中的受控配置加载，不要把密钥写入命令、Prompt 或报告。

最终百炼配置使用 OpenAI-compatible Vision 端点和 `qwen3-vl-flash`；Docker 部署中的
无密钥模板、权限设置和预检步骤见
[Ubuntu Docker 部署](ubuntu-docker-deployment.md#百炼-qwen-plannervlm)。

历史实现基线 `3a70294` 只用于追溯，不是当前发布证明。实际复现必须记录
`git rev-parse HEAD`，并使用验证报告中列出的 `final-dev` 发行谱系；工作树必须干净，且代码、ROS overlay 与受控服务来自同一谱系。当前稳定性证据和准确的运行提交见
[final-dev-validation-report.md](final-dev-validation-report.md)。

## 2. 启动人工 TUI

```bash
RUN_ROOT="$REPO/.cache/reports/multi-normal-human-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$RUN_ROOT"

scripts/run_multi_normal_gazebo_acceptance.sh \
  --operator-mode human_tui \
  --run-root "$RUN_ROOT"
```

启动器会完成 Python、ROS、Gazebo overlay、模型服务和 Provider 预检，创建隔离的 ROS domain、Gazebo partition、MCP 端口及报告目录；随后由宿主创建并 reset Gazebo、保存首帧，再连接本轮 GUI。GUI 的窗口和相机服务都就绪后，才进入真实交互式 TUI。因此操作员看到 `›` 提示符时，RealVNC 中已经能看到本轮场景。环境生命周期不占用智能体回合或 VLM token，物理场景已绑定，但抓取任务尚未配置。

TUI 必须运行在 PTY 中。按上面的方式先 `ssh hhh` 再执行脚本即可；若要从本机写成一条命令，必须使用 `ssh -tt hhh '…'`，不要通过无 PTY 的后台 SSH 或日志重定向启动。Dashboard 对应本轮启动器分配的 session；地址和环境身份记录在 `$RUN_ROOT/pick-place/human_tui/host-simulator-lifecycle.json`。Gazebo 窗口未出现时，保持脚本运行并检查：

```bash
tail -f "$RUN_ROOT/pick-place/human_tui/operator-gui.log"
```

不要手动启动一个没有本轮 `GZ_PARTITION` 的 `gz sim -g` 客户端。

## 3. 输入代表性任务

在 TUI 的 `›` 提示符处一次性粘贴下面这段自然语言并按 Enter：

```text
请先看清工作台。先把黄色活动扳手放进绿色零件箱，再把红色六角螺栓放进蓝色零件箱；其他物件不要动。第一件放好后继续第二件，全部完成后确认两件物品都已在对应料箱中放稳，再结束任务。如果当前视角不清楚，可以换一个观察角度确认。
```

这是代表性复现 Prompt，不是控制脚本。允许人类换一种自然表达，但正式复现时应保持以下语义不变：

1. 黄色活动扳手 → 绿色零件箱；
2. 红色六角螺栓 → 蓝色零件箱；
3. 顺序执行，不移动干扰物；
4. 第一件完成后继续同一环境中的第二件。

## 4. 操作期间

`human_tui` 只会为任务中的世界状态变更询问确认，例如机械臂运动和夹爪开合。Gazebo 的创建、reset 和关闭由启动脚本负责，不会出现在 TUI 工具列表、Prompt 或审批对话里。确认 TUI 中的动作与当前阶段一致，并在 Gazebo GUI 中没有明显异常后输入 `y`。感知、模型推理、IK、状态有效性检查和 L5 plan-only 不需要人工确认。

如果智能体确实提出语义问题，像普通操作员一样简短回答目标物品或料箱；不要给出候选编号、坐标或关节值。共享 GPU 负载可能使 TUI 一段时间没有新输出，看到模型或资格筛选仍在运行时不要重复提交任务。

一次 `grasp_pose_estimate` 会在内部完成模型生成、目标预绑定和多轮 MoveIt 资格筛选；因此工具总耗时可能超过单次资格 RPC 的确认窗口。只要 TUI 仍显示 `working`，就继续等待，不要再次粘贴 Prompt。若物理运动失败，正常恢复输出应类似：

```text
grasp_pose_estimate mode=frozen_frontier model_inference=False
```

这表示系统正在使用同一次模型输出的冻结候选前沿，不是重新运行 GraspGenX。资格响应在传输层丢失时，主机也会自动健康检查并按绑定哈希幂等重取；该恢复不增加 VLM 回合，不需要操作员干预。

运动失败后，控制器先等待因果上属于本次动作的状态样本稳定：稳定到达目标即成功；稳定但未到目标则记录实际关节状态，发出 `current_state_restart`，并从该状态重新资格化尚未访问的冻结候选。若失败的闭合动作推动了仍未 attach 的物体，系统会把 PlanningScene 同步到物体当前实测位姿，只刚性重基冻结抓取前沿；料箱中的物理放置目标保持世界坐标不变，旧的模型物体运动变换不会再次应用。

只有动作后状态仍无法被仿真回执或后续观测证明时才应出现 `ask_human`。此时像普通现场操作员一样依据 Gazebo 画面简短回答“夹爪已打开、机械臂已停止，可以从当前状态继续”或“现场状态不清楚，请重新观察”；不要代替系统给坐标、候选编号或跳过安全证明。实际工程不禁止重新观察或重新推理，验收只是在冻结证据仍有效时优先选择低成本恢复。

任务成功时，TUI 应显示任务完成；此时 Gazebo 仍保持可见，方便操作员核对最终画面。此后输入：

```text
/quit
```

不要在第二件物品完成前输入 `/quit`。TUI 进程退出后，启动器会关闭它预先创建的精确环境 handle，再关闭 GUI/MCP 并进行正式证据校验。不要另开终端手工关闭 Gazebo，否则会破坏本轮生命周期证据。

## 5. 判断 PASS

检查报告：

```bash
jq '{status,scenario,task_variant,operator_mode,tool_call_count,
     planner_mode:.planner_evidence.planner_mode,
     host_dispatch_count:.planner_evidence.host_dispatch_count,
     total_tokens:.planner_evidence.total_tokens}' \
  "$RUN_ROOT/acceptance-report.json"

jq '{mcp_group_exited,port_free,
     host_simulator_lifecycle,
     owned_process_residuals,owned_residual_groups,
     operator_gui:{started:.operator_gui.started,
                   group_exited:.operator_gui.group_exited,
                   lifecycle_ok:.operator_gui.lifecycle_ok},
     protected_ros_graphs_unchanged}' \
  "$RUN_ROOT/pick-place/human_tui/cleanup.json"
```

正式复现必须同时满足：

- `status` 为 `passed`；
- `scenario` 为 `multi_normal`；
- `operator_mode` 为 `human_tui`；
- `planner_mode` 为 `agentic_closed_loop`；
- `host_dispatch_count` 为 `0`；
- `host_simulator_lifecycle.closed` 和 `host_simulator_lifecycle.ok` 均为 `true`，且 `host-simulator-lifecycle.json` 证明先于 TUI 完成 reset、晚于 TUI 退出执行关闭；
- 两个物品各有一次 SAM3 assignment、一次 AnyPlace 和一次 GraspGenX 模型链；每个 SAM3 assignment 必须包含按 `grasp_target → placement_region` 排列的两份独立语义证据；
- 最终放置具有 MoveIt 状态有效性、L5 plan-only、原生 attach/detach，以及供 VLM 判断目标箱、正面/朝向和明显物理失败的因果 post-release RGB-D；释放工具不再阻塞等待固定时长的仿真落稳采样；多物体切换复用一次 Gazebo 位姿快照，并以一次原子 PlanningScene 事务同步已释放物体和下一目标；
- `cleanup.json` 中 `mcp_group_exited`、`port_free`、GUI lifecycle 和 protected ROS graph 检查均通过，且 owned residual 列表为空。

运行失败时保留整个 `RUN_ROOT`，不要覆盖或删除。可用相同参数加 `--verify-only` 重新读取证据；新的物理复测必须换一个新目录。

```bash
scripts/run_multi_normal_gazebo_acceptance.sh \
  --operator-mode human_tui \
  --run-root "$RUN_ROOT" \
  --verify-only
```

## 本交付的范围边界

`multi_normal` 本身不绑定固定措辞；操作员可用自然语言改写第 3 节 Prompt，但应保持
同一物件、目标料箱、顺序和“不移动其他物件”的语义。验收器的 `--task-variant`、随机
布局和更多物件组合仍是仓库的研发能力，不是本手册要求操作员运行的附加验证，也不会在
默认复现命令中启动。

不要为了证明额外场景而覆盖本轮 `RUN_ROOT` 或修改当前物理场景。若以后正式扩展交付
范围，应为新的自然语言任务和世界状态单独建立证据，而不是把它伪装成这份
`multi_normal` 复现记录的一部分。
