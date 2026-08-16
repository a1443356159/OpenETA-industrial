# 问题台账（Gazebo M0–M4）

本台账记录控制层验收中真实遇到的环境、依赖和运行时问题。每一项都必须保留
失败原因、最小修复、回归证据和当前状态；**`fail-closed` 仅表示拒绝不可信结果，
不等于问题已解决或里程碑已通过。**

正式 PTY TUI → MCP/SSE → Gazebo M0–M4 验收仍另行计数。下列修复及后续运行均
不调用模型/provider API，除非某一条目明确说明。

| ID | 范围 | 现象与根因 | 持久修复 | 验证与状态 |
| --- | --- | --- | --- | --- |
| P-001 | 远端 M2 Gazebo worker | 使用 `/root/autodl-tmp/env/ros2_jazzy/bin/python` 建立、且带 `--system-site-packages` 的 venv 时，应用 Python 路径与 `/opt/ros/jazzy` 的原生 typesupport 混用；创建 ROS node 时会出现 `rosgraph_msgs` typesupport 的 undefined symbol。旧行为正确地 fail-closed，但不能完成 M2。 | 云端协调器只允许 `/usr/bin/python3` 或 `/usr/bin/python3.12` 建立**无** `--system-site-packages` 的 venv；worker 继续只信任 `/opt/ros/jazzy` 与本 clone overlay 的 ROS Python/LD/AMENT 路径；启动前显式导入 `rclpy` 与 `rosgraph_msgs.msg.Clock`。 | 本地 ABI 隔离和 runtime 回归已通过；新的远端隔离 M2 控制运行待执行。状态：**修复已实现，远端待验证**。 |
| P-002 | MCP server 依赖 | `sim.mcp_server.server` 直接导入 `starlette`、`uvicorn`，但它们以前只可能由宿主环境的传递依赖提供；清洁 venv 会在 worker/MCP 启动处缺模块。 | 在项目的正式 dependencies 中声明 `starlette>=0.27` 和 `uvicorn>=0.23`；启动包装器把它们与 `mcp` 纳入必需导入预检。 | 云端计划和本地回归已断言这些依赖与预检存在；干净 venv 的远端安装待与 P-001 一起验证。状态：**修复已实现，远端待验证**。 |
| P-003 | 远端 clean venv 安装 | `pip install --no-build-isolation .` 在新 venv 中可能因没有 `setuptools`/`wheel` 而无法构建项目。 | 在项目安装前，云端计划固定 bootstrap `pip>=24`、`setuptools>=68`、`wheel>=0.42`；不从系统 site-packages 借包。 | 云端计划单测覆盖 bootstrap 及 no-system-site venv；远端安装待验证。状态：**修复已实现，远端待验证**。 |
| P-004 | ROS 隔离 | 手工诊断曾使用 `ROS_DOMAIN_ID=255`；Jazzy 的合法值为 `0..232`，部署配置因此拒绝启动。 | 接受器只从 `80..101` 的非保护 domain 池分配；`GazeboDeploymentConfig` 继续对越界值 fail-closed。 | 分配器/部署配置已有回归；之后远端运行必须使用 case receipt 中分配的 domain。状态：**已解决**。 |
| P-005 | M1/M2 冷启动 | launch 父进程存活不等于 `/world/<name>/control` 服务可用；远端冷启动时过早 reset 会超时。 | 在任意 reset 前等待精确 control 服务出现，并以无状态 `WorldControl` ACK 探针确认就绪；超时保持 fail-closed。 | 本地隔离 M1 `wait_ready`→`reset_all` 实测通过，相关 runtime/process 回归通过。状态：**已解决（M2 继续受 P-001 前置约束）**。 |
| P-006 | Gazebo `gz` 启动 | 宿主 Ruby/Gem 环境可抢占 vendor `gz` wrapper 的 `#!/usr/bin/env ruby`，导致世界未创建或不稳定退出。 | Gazebo child environment 固定把 `/usr/bin` 放在 PATH 前，移除 Ruby/Gem/Bundle/rbenv/rvm 变量，同时保留 ROS、overlay、GZ transport 与渲染变量。 | 受污染环境的本地 M1 launch/control/reset 已通过；部署回归覆盖变量清理。状态：**已解决**。 |

## 重试纪律

1. 每个远端 case 使用独立 ROS domain、GZ partition、loopback port 和 case-local
   日志目录；保留 stdout、stderr、MCP response artifacts、trace 与 `cleanup.json`。
2. 出现失败时，先把错误码、日志路径、原因判断和修复写入本文件，再运行针对性
   本地回归；不得通过删除验证项、伪造 receipt 或启用软吸附/运动学回退来继续。
3. M0→M4 严格串行；前一控制门未通过，后一门标记 `not_run`。
4. 只有远端报告的相应控制门实际 `passed`，才能更新工程文档为“控制层已通过”；
   这仍不自动代表有 PTY/推理层的正式验收。
