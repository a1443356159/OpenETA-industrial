# `final-dev` `multi_normal` 交付记录

> **交付范围：** 百炼 Qwen 驱动的双物件 `multi_normal` 连续分拣。本文是该范围的
> 发行证据，不把随机布局、额外物件类别或历史 benchmark 误写为本次交付的前置条件。

## 运行配置

- 场景：任务中立的 RM75 / Robotiq `multi_normal`；操作员在 TUI 中用自然语言给出
  物品顺序和目标料箱。
- Planner/VLM：百炼 OpenAI-compatible Vision API，`qwen3-vl-flash`。
- 感知与动作：SAM3、GraspGenX、AnyPlace、MoveIt 和 GPU Gazebo GUI。
- 资格化：`fast_v3`，GraspGenX 每件物品冻结一个 512 候选的模型结果池，按小波次
  深度贯通；512 是候选储备，不代表 512 次同步 IK 或 L5。
- 回退边界：AnyGrasp 不进入此交付链；GUI 和 Dashboard 只供操作员观察，不参与候选
  选择或验收决策。

运行 A、B 使用代码提交 `7cbe9ef9e751322771a2d6aeaa80ac13722277b1`。运行 C 使用的
运行时源码现由 `1ebe5e7` 提交；该提交在运行结束后从相同的两个运行时文件创建，所以 C 的
环境回执仍记录提交前的 `9155541`，而不是伪造新的回执哈希。`1ebe5e7` 仅保留刚完成就绪门槛
的原生关节/TF 样本，并包含对应回归测试；它不改变场景、物件、候选、VLM 或控制轨迹。
`46429fa` 的成功回执上下文去重也已随 C 得到一次新的真实物理 PASS。`0f678b4` 与本文档提交
只改变交付材料，不改变运行代码。

运行 D、E 使用相同运行时基线，并加入随后提交为 `2317881` 的共享主机 ROS domain 分配修复：
先通过既有的双样本 ROS 图预检在 `102--232` 中寻找空闲 domain，旧 `80--101` 范围保留为
回退。两次回执均选中 domain 102；因为提交是在两轮结束后创建，回执中的 Git HEAD 仍是
`5b2ca98`，不应将它误读为缺少该运行时修改。

## 稳定性证据

在相同的代表性工单、全新隔离 run root 和真实 PTY 中完成五次独立连续运行：

> 先把黄色活动扳手放进绿色零件箱，再把红色六角螺栓放进蓝色零件箱；其他物件不要动。

| 运行 | 证据根目录 | 结果 | TUI episode | 工具调用 | Planner tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| A | `/root/autodl-tmp/openeta-final-dev-qwen-validation/multi-normal-seedfix-a/` | PASS | — | 17 | 115,366 |
| B | `/root/autodl-tmp/openeta-final-dev-qwen-validation/multi-normal-seedfix-b/` | PASS | — | 17 | 113,701 |
| C | `/root/autodl-tmp/openeta-final-dev-qwen-validation/multi-normal-context-elision-d-20260905T0032Z/` | PASS | 435 s | 17 | 100,951 |
| D | `/root/autodl-tmp/openeta-final-dev-qwen-validation/multi-normal-baseline-stability-1-20260905T025522Z/` | PASS | 316 s | 17 | 100,059 |
| E | `/root/autodl-tmp/openeta-final-dev-qwen-validation/multi-normal-baseline-stability-2-20260905T030330Z/` | PASS | 350 s | 17 | 99,933 |

五份 `acceptance-report.json` 都记录：

- `status=passed`、`scenario=multi_normal`、`agentic_closed_loop`；
- `host_dispatch_count=0`、`fast_v3`、GraspGenX，且 verifier errors 为空；
- 每个任务物件各运行一次 SAM3 assignment、一次 GraspGenX 和一次 AnyPlace 模型链；
  后续资格化复用冻结候选池，不重新推理；
- 没有 `ask_human` 恢复。

其中 C 在此前出现一次 `TF_TIMEOUT` 的相同共享服务器条件下重新创建隔离环境并通过，说明
当前 reset 时序修复没有把基础设施问题降格为候选失败。A、B 提供独立复现基线，C 为当前
`final-dev` 运行时源码的物理桥接证据。D、E 则在旧 domain 池已拥挤的共享服务器上，
通过新的隔离 domain 选择完整通过。更广的任务组合和随机世界仍可作为后续研发评估，
但不是当前版本的验收门槛，也不会由默认复现流程触发。

## 已纳入的可靠性修复

| 提交 | 修复 | 对 `multi_normal` 的作用 |
| --- | --- | --- |
| `35e67ec` | 原生 detach 已完成时，已知夹爪控制器终态失败只保留为 telemetry；拒绝和未知结果仍严格失败。 | 避免 Gazebo 控制器已知终态造成无意义的人工恢复。 |
| `7cbe9ef` | 恢复层先保留候选的固定快速基础种子，再补六个固定恢复种子；批次缓存不能替换基础分支。 | 相同模型候选不会因进入恢复波次而丢掉此前可行的确定性 IK 分支。 |
| `1ebe5e7` | 原生拾放 reset 新建控制器后保留其 readiness 门槛已证明的同步关节/TF 样本；非物理 model reset 仍清空旧样本。 | 避免静止机器人等待一条多余的 TF 发布而把环境就绪问题报为 `TF_TIMEOUT`。 |

本次没有为当前样本加入坐标、物体名称或候选编号特判；冻结模型输出、确定性种子和
MoveIt 状态/L5 证明仍是通用机制。

## 观察到的时延

服务器在两轮期间有共享负载，以下数值用于定位优化方向，而非独占性能承诺：

| 指标 | 运行 A | 运行 B | 运行 C |
| --- | ---: | ---: | ---: |
| TUI episode 墙钟 | 436 s | 349 s | 435 s |
| 两次 `grasp_pose_estimate`（GraspGenX 与资格化） | 103 s | 74 s | 125 s |
| 四段 MoveIt/Gazebo `move_to` | 145 s | 111 s | 127 s |
| 四次原生 `gripper_control` | 86 s | 79 s | 89 s |
| AnyPlace 模型/冻结池调用 | 14.5 s | 14.6 s | 18.6 s |

主要时延来自物理仿真轨迹和夹爪执行，不是 AnyPlace 的冻结池。当前使用已通过跟踪误差
检验的保守控制器配置；不为追求一次测试的更短时间而改变速度/加速度或牺牲终态证明。

在运行 B 的可审计模型调用记录中，17 次已知成功的宿主工具回执共重复进入 Qwen 上下文
34,516 个字符。当前 `final-dev` 已将这类回执保留在 append-only 审计记录和结构化状态中，
但不再重复作为下一次模型请求的聊天消息；失败或未知回执仍完整发送，以保留恢复能力。
这是一项通用的上下文投影优化，不改变工具、候选、物理控制或验收条件。运行 C 的 provider
总 token 为 100,951；共享负载与模型生成存在自然波动，因此该单次差异不单独作为性能承诺，
但它确认上下文投影没有破坏真实闭环。会话和 Planner 投影单元测试也已通过。

## 复现与 Docker 交付

- 服务器人工 TUI、VNC 和 GPU GUI 的逐步操作见
  [multi-normal-tui-reproduction.md](multi-normal-tui-reproduction.md)。
- 可移植 Docker 镜像、百炼 provider secret、模型准备和默认双轮命令见
  [ubuntu-docker-deployment.md](ubuntu-docker-deployment.md)。

在当前 `final-dev` 上，`tests/test_ubuntu_docker_deployment.py` 与
`tests/test_hpc_deployment.py` 共 17 项静态部署契约测试通过：它们覆盖唯一的
CUDA/ROS 镜像定义、只读模型卷/可写状态卷、窄范围 provider secret、Qwen/GraspGenX
默认入口、HPC 复用同一 Dockerfile，以及本地凭据不会进入 build context。该检查不等同于
实际镜像构建；首次部署仍应在具备 Docker Engine、NVIDIA Container Toolkit、模型权重和
GPU 的目标主机上执行文档中的 `build`、`fetch-models` 与 `validate-assets`。

复现时应使用新 run root。默认 Docker `agentic-normal` 连续跑两轮：首轮失败即退出，
不会以第二轮成功掩盖故障。若仅需操作员演示一轮，可显式传入 `--runs 1`；这不改变
上述 A、B 双轮稳定性基线及 C 的当前运行时桥接证据。
