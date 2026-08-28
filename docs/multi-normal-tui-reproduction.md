# `multi_normal` 人工 TUI 复现指南

本文用于复现最终发行验收：人类操作员在同一个任务中，先把黄色活动扳手放入绿色零件箱，再把红色六角螺栓放入蓝色零件箱。物理场景始终是任务中立的 `multi_normal`；物品顺序和目标料箱来自操作员在 TUI 中输入的自然语言，不由场景静态注入。

## 验收边界

- 使用真实 OpenETA TUI、VLM、SAM3、GraspGenX、AnyPlace、MoveIt 和 Gazebo 链路。
- 使用 `fast_v3` 资格漏斗和 GraspGenX；本发行验收不启动 AnyGrasp。
- Gazebo GUI 在 GPU VNC 桌面上保持开启，Dashboard 仅用于观察。
- 每件物品只运行一次模型推理；物理失败优先从冻结候选前沿恢复。
- 人类只描述任务、处理真正的语义澄清，并确认世界状态变更；不输入末端位姿、偏移量、候选编号或工具调用顺序。

## 1. 准备

在服务器上的 `final-dev` 工作树根目录执行。确认工作树提交与待验收发行提交一致，并使用一个新的报告目录；不要复用旧目录。

```bash
git status --short --branch

export OPENETA_PYTHON_EXECUTABLE=/root/autodl-tmp/OpenETA-industrial/.venv/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
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
```

三项健康检查都应成功。用 RealVNC 连接服务器的 `5903` 端口，确认桌面可见；验收启动后 Gazebo GUI 会自动打开。Provider 凭据由部署目录中的受控配置加载，不要把密钥写入命令、Prompt 或报告。

## 2. 启动人工 TUI

```bash
RUN_ROOT=".cache/reports/multi-normal-human-$(date -u +%Y%m%dT%H%M%SZ)"

scripts/run_multi_normal_gazebo_acceptance.sh \
  --operator-mode human_tui \
  --run-root "$RUN_ROOT"
```

启动器会完成 Python、ROS、Gazebo overlay、模型服务和 Provider 预检，创建隔离的 ROS domain、Gazebo partition、MCP 端口及报告目录，然后进入真实交互式 TUI。此时物理场景已由运行器绑定，但抓取任务尚未配置。

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

`human_tui` 会在以下世界状态变更前询问确认：创建/关闭环境、机械臂运动和夹爪开合。确认 TUI 中的工具名与当前阶段一致，并在 Gazebo GUI 中没有明显异常后输入 `y`。感知、模型推理、IK、状态有效性检查和 L5 plan-only 不需要人工确认。

如果智能体确实提出语义问题，像普通操作员一样简短回答目标物品或料箱；不要给出候选编号、坐标或关节值。共享 GPU 负载可能使 TUI 一段时间没有新输出，看到模型或资格筛选仍在运行时不要重复提交任务。

任务成功时，TUI 应显示环境已终止。此后输入：

```text
/quit
```

不要在第二件物品完成前输入 `/quit`。退出后启动器才会完成进程清理和正式证据校验。

## 5. 判断 PASS

检查报告：

```bash
jq '{status,scenario,task_variant,operator_mode,tool_call_count,
     planner_mode:.planner_evidence.planner_mode,
     host_dispatch_count:.planner_evidence.host_dispatch_count,
     total_tokens:.planner_evidence.total_tokens}' \
  "$RUN_ROOT/acceptance-report.json"
```

正式复现必须同时满足：

- `status` 为 `passed`；
- `scenario` 为 `multi_normal`；
- `operator_mode` 为 `human_tui`；
- `planner_mode` 为 `agentic_closed_loop`；
- `host_dispatch_count` 为 `0`；
- 两个物品各有一次 SAM3、AnyPlace 和 GraspGenX 模型链证据；
- 最终放置具有 MoveIt 状态有效性、L5 plan-only、原生 attach/detach 和稳定入箱证明；
- `cleanup.json` 证明本轮拥有的 MCP、ROS/Gazebo 和 GUI 进程均已退出。

运行失败时保留整个 `RUN_ROOT`，不要覆盖或删除。可用相同参数加 `--verify-only` 重新读取证据；新的物理复测必须换一个新目录。

## 可接受的 Prompt 变化

`multi_normal` 并不绑定固定任务文本。保持同一物品顺序和料箱对应关系时，操作员可以自由改变措辞，不需要修改启动命令。若要正式验证另外三种顺序/料箱组合，在启动命令中选择相应的私有核对契约，并输入语义一致的自然语言 Prompt：

| `--task-variant` | 连续任务 |
| --- | --- |
| `wrench-blue-bolt-green` | 扳手 → 蓝箱，然后螺栓 → 绿箱 |
| `bolt-blue-wrench-green` | 螺栓 → 蓝箱，然后扳手 → 绿箱 |
| `bolt-green-wrench-blue` | 螺栓 → 绿箱，然后扳手 → 蓝箱 |

例如：

```bash
scripts/run_multi_normal_gazebo_acceptance.sh \
  --operator-mode human_tui \
  --task-variant bolt-green-wrench-blue \
  --run-root "$RUN_ROOT"
```

`--task-variant` 只告诉验收器如何核对人类请求，不会把任务写入物理场景或 Planner 上下文。实际智能体仍可接受场景内更多自然语言任务；若要把新的物品/料箱组合纳入正式自动核验，应新增任务契约，而不是复制或特化物理场景。
