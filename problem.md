# 问题台账（Gazebo M0–M4）

本台账记录控制层验收中真实遇到的环境、依赖和运行时问题。每一项都必须保留
失败原因、最小修复、回归证据和当前状态；**`fail-closed` 仅表示拒绝不可信结果，
不等于问题已解决或里程碑已通过。**

正式 PTY TUI → MCP/SSE → Gazebo M0–M4 验收仍另行计数。下列修复及后续运行均
不调用模型/provider API，除非某一条目明确说明。

| ID | 范围 | 现象与根因 | 持久修复 | 验证与状态 |
| --- | --- | --- | --- | --- |
| P-001 | 远端 M2 Gazebo worker | 使用 `/root/autodl-tmp/env/ros2_jazzy/bin/python` 建立、且带 `--system-site-packages` 的 venv 时，应用 Python 路径与 `/opt/ros/jazzy` 的原生 typesupport 混用；创建 ROS node 时会出现 `rosgraph_msgs` typesupport 的 undefined symbol。旧行为正确地 fail-closed，但不能完成 M2。 | 云端协调器只允许 `/usr/bin/python3` 或 `/usr/bin/python3.12` 建立**无** `--system-site-packages` 的 venv；worker 继续只信任 `/opt/ros/jazzy` 与本 clone overlay 的 ROS Python/LD/AMENT 路径；启动前显式导入 `rclpy` 与 `rosgraph_msgs.msg.Clock`。 | SHA `c202f51` 的远端干净 venv 已实际通过 ABI 导入、ROS workspace build、MCP create/reset 与真实 M2 控制动作；这是无 API 的控制诊断，不是正式 PTY 验收。状态：**已解决并远端复验**。 |
| P-002 | MCP server 依赖 | `sim.mcp_server.server` 直接导入 `starlette`、`uvicorn`，但它们以前只可能由宿主环境的传递依赖提供；清洁 venv 会在 worker/MCP 启动处缺模块。 | 在项目的正式 dependencies 中声明 `starlette>=0.27` 和 `uvicorn>=0.23`；启动包装器把它们与 `mcp` 纳入必需导入预检。 | 远端无 system-site venv 已完成项目安装、模块导入与真实 MCP server 启动。状态：**已解决并远端复验**。 |
| P-003 | 远端 clean venv 安装 | `pip install --no-build-isolation .` 在新 venv 中可能因没有 `setuptools`/`wheel` 而无法构建项目。 | 在项目安装前，云端计划固定 bootstrap `pip>=24`、`setuptools>=68`、`wheel>=0.42`；不从系统 site-packages 借包。 | 本地最小 clean venv 先复现缺 `setuptools`，bootstrap 后安装成功；远端同一路径也已成功。状态：**已解决并远端复验**。 |
| P-004 | ROS 隔离 | 手工诊断曾使用 `ROS_DOMAIN_ID=255`；Jazzy 的合法值为 `0..232`，部署配置因此拒绝启动。 | 接受器只从 `80..101` 的非保护 domain 池分配；`GazeboDeploymentConfig` 继续对越界值 fail-closed。 | 分配器/部署配置已有回归；之后远端运行必须使用 case receipt 中分配的 domain。状态：**已解决**。 |
| P-005 | M1/M2 冷启动 | launch 父进程存活不等于 `/world/<name>/control` 服务可用；远端冷启动时过早 reset 会超时。 | 在任意 reset 前等待精确 control 服务出现，并以无状态 `WorldControl` ACK 探针确认就绪；超时保持 fail-closed。 | 本地隔离 M1 `wait_ready`→`reset_all` 与远端 M2 create/reset 均已通过。状态：**已解决并远端复验**。 |
| P-006 | Gazebo `gz` 启动 | 宿主 Ruby/Gem 环境可抢占 vendor `gz` wrapper 的 `#!/usr/bin/env ruby`，导致世界未创建或不稳定退出。 | Gazebo child environment 固定把 `/usr/bin` 放在 PATH 前，移除 Ruby/Gem/Bundle/rbenv/rvm 变量，同时保留 ROS、overlay、GZ transport 与渲染变量。 | 受污染环境的本地 M1 launch/control/reset 已通过；部署回归覆盖变量清理。状态：**已解决**。 |
| P-007 | 远端正式 clone 前置条件 | 远端对 GitHub 的 HTTPS `git -c http.version=HTTP/1.1 ls-remote` 仍报 `gnutls_handshake() failed: The TLS connection was non-properly terminated`；SSH remote 又没有可用 deploy key（`Permission denied (publickey)`）。 | 未降低 TLS 校验，也未复用过期 `/root/OpenETA-industrial`。为诊断 M2，仅使用 SHA-256 已核验的本地 bundle 建立全新 detached clone，诊断后删除 bundle；这不是正式 clone 替代。正式修复需要远端 GitHub HTTPS 可用，或配置只读 deploy key。 | SHA `c202f51` 的 bundle detached clone 仅用于无 API 控制诊断并已清理上传 bundle。状态：**外部环境未解决；正式 origin clean-clone/PTy 验收仍阻塞**。 |
| P-008 | M3 DetachableJoint reset | 世界 control service 已就绪并不代表后续 robot spawn 的 stock DetachableJoint topic 已就绪；首次 detach 可丢失 ACK。另一个 stock 行为是：`model_only` reset 后 joint 已处于 detached，重复 detach 没有状态转换，因此不会产生可验证的新 ACK。 | 启动时先等待 attach/detach/state 三个官方 transport endpoint，再 listener-first 请求 detach。M3 后续 reset 不再把无状态的 `model_only` 误当成新 joint reset：它销毁并重建独立的 paused world，使每次 reset 都从 stock attached 状态得到一次真实 detached ACK 后才 unpause。close 对已知 detached 状态不伪造第二次 ACK；unknown/attached 仍要求真 ACK。 | 本地 Gazebo Sim 8.11：endpoint triplet 后初始 detach ACK 通过；真实 MCP M3 `create_env -> reset_env` 返回无错误，worker 无 traceback、无残留。单元覆盖 endpoint gate、暂停顺序与第二次 reset 的重建。状态：**本地已解决；远端待复验**。 |

## 重试纪律

1. 每个远端 case 使用独立 ROS domain、GZ partition、loopback port 和 case-local
   日志目录；保留 stdout、stderr、MCP response artifacts、trace 与 `cleanup.json`。
2. 出现失败时，先把错误码、日志路径、原因判断和修复写入本文件，再运行针对性
   本地回归；不得通过删除验证项、伪造 receipt 或启用软吸附/运动学回退来继续。
3. M0→M4 严格串行；前一控制门未通过，后一门标记 `not_run`。
4. 只有远端报告的相应控制门实际 `passed`，才能更新工程文档为“控制层已通过”；
   这仍不自动代表有 PTY/推理层的正式验收。
