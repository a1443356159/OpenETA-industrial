# M3 Gazebo 真实接触抓放物理验证

## 当前结论

M3 的场景、正式 ROS/Gazebo 数据路径、MoveIt PlanningScene、无 ROS 依赖校验器、worker/MCP 路由和验收驱动已经实现。离线合同、SDF/URDF、ROS 包构建、实时 topic/controller/TF/PlanningScene 诊断已通过。

截至 2026-08-09，M3 **未完成正式验收**。真实接触探针尚未形成双指稳定夹持：首个固定候选可通过 MoveIt plan-only 并到达预抓取位姿，但接触位规划被拒绝；继续闭合时只有左指尖接触，夹爪到达约 `0.00059 m` 开口而非在 `32–48 mm` 阻挡区间 stall，目标仍由工作台支撑。因此校验器返回 `UNKNOWN/DATA_MISSING`，没有误报 `TARGET_HELD`。M2 也仍为“实现完成、正式验收待重验”，所以两个 milestone checkbox 均保持未完成。

## 实现边界

- 环境 ID：`openeta/gazebo_rm75_robotiq2f85_pickplace-v0`
- 模型 ID：`rm75_robotiq_2f85_pickplace_sim_v1`
- 使用现有 `create/reset/observe/move_to/gripper_open/gripper_close/close` 原子接口。
- Gazebo 中只使用碰撞、重力、摩擦、Contact sensor 和 OdometryPublisher。没有 detachable joint、目标固定关节、吸附命令、外力抓取插件或运动学附着。
- MoveIt AttachedCollisionObject 仅在 `TARGET_HELD` 得到物理证据后用于后续规划；它不会改变 Gazebo entity/joint 关系。
- M3 允许标准 gripper stall-success 回执：动作可以 `ok=true`，但必须同时报告 `stalled=true`、`reached_goal=false`；最终抓取成败只由 M3 verifier 判定。M2 的 reached-goal 行为不变。
- 任一 JointState、TF、RGB-D、Contact、Odometry、身份或动作后时间戳证据缺失/过期时，校验结果 fail-closed 为 `UNKNOWN`。

## 场景和物理参数

工作台尺寸为 `0.70 × 0.60 × 0.04 m`，台面 `z=0.40 m`。规划诊断证明计划中的 `x=0.35 m` 会让桌边穿过 RM75 的 `link_2/link_3` 碰撞体，因此采用最近的无碰撞中心 `x=0.40 m`；目标、干扰物和放置区坐标保持计划值。

| 项目 | 值 |
| --- | --- |
| 目标物 | `0.04 × 0.04 × 0.06 m`, `0.10 kg`, 初始中心 `[0.28, -0.10, 0.43]` |
| 干扰物 | 直径 `0.05 m`, 高 `0.08 m`, `0.12 kg`, 初始中心 `[0.28, 0.12, 0.44]` |
| 放置区 | 中心 `[0.48, -0.10]`, `0.12 × 0.12 m`, 纯视觉、无 collision |
| 目标摩擦 | ODE `mu=mu2=1.2` |
| 指尖摩擦 | `mu1=mu2=1.5` |
| 接触参数 | `kp=100000`, `kd=10` |
| 物理步长 | `0.001 s`, DART, 重力 `-9.81 m/s²` |
| Contact/Odometry | `100 Hz` 配置 |

## 可信数据和状态机

`extensions/gazebo/m3.py` 定义不可变输入、reason codes、PlanningScene 命令和状态机；`extensions/gazebo/ros_physics.py` 只负责把官方消息规范化为该输入。observation 增量提供：

- `objects[]` 的 Gazebo 世界位姿、速度、支撑、时间戳和 `provenance=gazebo_truth`；
- `robot.gripper_state` 的左右接触、对象身份、检测、抓持确认和滑移字段；
- `metadata.physical_verification` 的 `m3_physical_verification_v1` 记录；
- 每个动作回执中同一份 `physical_verification` 和完整 fresh observation。

Gazebo Harmonic 的 contact sensor 在无接触时不会发布空 `Contacts` 心跳。本机实时验证中，目标—桌面接触 topic 持续发布，但未接触的左右指尖 topic 没有消息。实现没有用 Odometry 或超时伪造“无接触”，而是保留缺失 stream 并返回 `UNKNOWN/DATA_MISSING`。这也意味着当前正式空抓、脱离支撑和掉落负例无法满足“动作后 Contact 必须 fresh”的 gate，是 M3 的独立上游语义阻塞。

## PlanningScene

reset 后加入工作台、目标和干扰物。完整 SRDF AllowedCollisionMatrix 会先从 `/get_planning_scene` 读取，再只增加目标物与两个指尖链路的对称允许项。禁止发送部分 ACM，因为实测 MoveIt 会把它作为替换矩阵，擦除既有相邻链路豁免并导致起始态自碰撞。

通过物理抬升后才发送 AttachedCollisionObject；释放前恢复 world object，释放后用 Gazebo 最终姿态更新。reset/close 清除三个 world object 和可能的 attached object。

## 验收驱动

运行：

```bash
bash extensions/gazebo/ros2_ws/run_m3_pickplace_acceptance.sh
```

驱动锁定 `ROS_DOMAIN_ID=100..199`、独立 `GZ_PARTITION` 和 MCP 端口，按自身 partition 精确清理进程组，不调用宽泛 `pkill`。报告写入忽略提交的 `.cache/reports/m3-pickplace-<timestamp>-<pid>.json`。

Direct 驱动从 live TF 和冻结 STL 包围盒计算指尖碰撞中心，以固定 pitch/yaw 顺序执行 MoveIt plan-only 并冻结首个候选；随后要求 5/5 正向流程、四个结构化负例、Gazebo joint inventory 前后一致。只有 Direct 全部通过才会启动真实 SSE MCP 的两轮正向流程和四个负例。任一物理 gate 失败，报告标为 `blocked`，脚本非零退出。

## 本机版本与上游依据

| 组件 | 已安装版本 |
| --- | --- |
| Gazebo Sim | `8.11.0` |
| ros_gz | `1.0.22-1noble.20260616.074726` |
| MoveIt | `2.12.4-1noble.20260617.161956` |
| ros2_control | `4.45.2-1noble.20260615.175135` |
| ros2_controllers | `4.40.1-1noble.20260616.074625` |

- Gazebo Harmonic sensors/contact：<https://gazebosim.org/docs/harmonic/sensors/>
- Gazebo Sim 8 Contact system：<https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1Contact.html>
- Gazebo Sim 8 OdometryPublisher：<https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1OdometryPublisher.html>
- ros_gz_bridge 正式消息映射：<https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_bridge/README.md>
- MoveIt PlanningScene/CollisionObject：<https://moveit.picknik.ai/main/doc/tutorials/planning_around_objects/planning_around_objects.html>
- MoveIt AllowedCollisionMatrix：<https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html>
- ros2_control Jazzy gripper stall：<https://control.ros.org/jazzy/doc/ros2_controllers/gripper_controllers/doc/userdoc.html>

## 下一步解除阻塞

1. 依据当前冻结资产重新生成不提前单侧碰撞、接触位仍可规划的夹持候选；不得放宽为几何近邻成功。
2. 对 Harmonic 无接触不发消息的行为寻找同版本官方、带显式空状态的接口。若没有正式接口，继续保持 `UNKNOWN`，不自定义协议或合成心跳。
3. 只有 Direct `5/5`、四负例、SSE `2/2`、清理 gate 都通过，且用户另行安排 M2 正式重验后，才更新 M2/M3 milestone checkbox。
