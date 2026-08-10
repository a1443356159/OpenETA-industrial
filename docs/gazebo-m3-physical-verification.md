# M3 Gazebo Robotiq 语义抓放物理验证

## 当前结论

M3 的场景、正式 ROS/Gazebo 数据路径、MoveIt PlanningScene、无 ROS 依赖校验器、worker/MCP 路由和验收驱动已经实现。离线合同、SDF/URDF、ROS 包构建、实时 topic/controller/TF/PlanningScene 诊断已通过。

截至 2026-08-10，M3 **未完成正式验收**。校验语义已经改为复用 Robotiq 官方对象检测结果：闭合提前受阻只产生待 lift 验证的候选，目标身份和最终成功由目标/干扰物 Odometry 相对 EEF 的共动证明。旧的双指 Contact 新鲜度门槛、传感器、bridge 和 observation 扩展均已删除。

最新正式运行已通过 ROS workspace build、M3/M2 离线合同（`86 passed`）和 M2 checkpoint 检查。四个固定候选都能完成 pregrasp plan/execute，但全部在接触位返回 `MOTION_PLAN_FAILED`；驱动按计划继续到最后一个候选并记录 `contact_execute` blocker，未执行闭合/lift，因此没有冻结候选、没有开始 Direct `5/5` 或 SSE MCP。正式报告的清理 gate 因 DDS/Gazebo discovery 仍短暂显示已退出的测试图而失败；隔离进程实际为空，稍后对同一 ROS domain 与 Gazebo partition 的独立复查均为空。milestone checkbox 保持未完成。

## 实现边界

- 环境 ID：`openeta/gazebo_rm75_robotiq2f85_pickplace-v0`
- 模型 ID：`rm75_robotiq_2f85_pickplace_sim_v1`
- 使用现有 `create/reset/observe/move_to/gripper_open/gripper_close/close` 原子接口。
- Gazebo 中只使用碰撞、重力、摩擦和 OdometryPublisher。没有 Contact sensor、detachable joint、目标固定关节、吸附命令、外力抓取插件或运动学附着。
- MoveIt AttachedCollisionObject 仅在 `TARGET_HELD` 得到物理证据后用于后续规划；它不会改变 Gazebo entity/joint 关系。
- M3 允许标准 gripper stall-success 回执：动作可以 `ok=true`，但必须同时报告 `stalled=true`、`reached_goal=false`；最终抓取成败只由 M3 verifier 判定。M2 的 reached-goal 行为不变。
- 任一 JointState、TF、RGB-D、目标/干扰物 Odometry 或动作后时间戳证据缺失/过期时，校验结果 fail-closed 为 `UNKNOWN`。

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
| Odometry | `100 Hz` 配置 |

## 可信数据和状态机

`extensions/gazebo/m3.py` 定义不可变输入、reason codes、PlanningScene 命令和状态机；`extensions/gazebo/ros_physics.py` 只负责把官方消息规范化为该输入。observation 增量提供：

- `objects[]` 的 Gazebo 世界位姿、速度、支撑、时间戳和 `provenance=gazebo_truth`；
- `robot.gripper_state` 的对象检测、抓持确认和滑移字段；
- `metadata.physical_verification` 的 `m3_physical_verification_v1` 记录；
- 每个动作回执中同一份 `physical_verification` 和完整 fresh observation。

闭合状态严格映射 Robotiq 语义：`stalled=true && reached_goal=false && aperture>6 mm` 返回 `UNKNOWN/LIFT_REQUIRED`；`reached_goal=true && stalled=false && aperture≤6 mm` 返回 `FAIL/EMPTY_GRASP`；缺失、矛盾或越界状态保持 `UNKNOWN`。闭合时同时保存目标和干扰物的世界位姿及相对 EEF 位姿。`80 mm` lift 后，仅目标满足上升 `≥60 mm`、离桌、相对漂移 `≤10 mm/0.15 rad` 才返回 `TARGET_HELD`；仅干扰物满足返回 `WRONG_OBJECT`，两者满足返回 `IDENTITY_INCOMPLETE`，均不满足返回 `TARGET_NOT_LIFTED`。

`objects[].support` 由 Odometry、已知包围盒和桌面几何计算。掉落通过失去相对共动并下降或落桌判定；放置要求目标底面距桌面 `≤5 mm`、完整位于桌面范围、速度低于既有阈值并稳定 `1 s`。这些判定不读取或推断 Contact。

## PlanningScene

reset 后加入工作台、目标和干扰物。完整 SRDF AllowedCollisionMatrix 会先从 `/get_planning_scene` 读取，再只增加目标物与两个指尖链路的对称允许项。禁止发送部分 ACM，因为实测 MoveIt 会把它作为替换矩阵，擦除既有相邻链路豁免并导致起始态自碰撞。

通过物理抬升后才发送 AttachedCollisionObject；释放前恢复 world object，释放后用 Gazebo 最终姿态更新。reset/close 清除三个 world object 和可能的 attached object。

## 验收驱动

运行：

```bash
bash extensions/gazebo/ros2_ws/run_m3_pickplace_acceptance.sh
```

驱动锁定 `ROS_DOMAIN_ID=80..101`、独立 `GZ_PARTITION` 和 MCP 端口，按自身 partition 精确清理进程组，不调用宽泛 `pkill`。候选域须无既有 ros2cli daemon，且由短生命周期 rclpy Context 连续两次直接 graph 观测为空；探针不使用 ros2cli daemon。报告写入忽略提交的 `.cache/reports/m3-pickplace-<timestamp>-<pid>.json`，终结后不可再次写入。

清理证据为三态：`passed` 表示所有查询成功且资源满足预期，`failed` 表示确认有残留，`inconclusive` 表示 ROS 或 Gazebo graph 查询不可用。后者绝不伪作通过；其退出码为 10（确认残留为 9，报告参数/重复终结错误为 11）。

Direct 驱动从 live TF 和冻结 STL 包围盒计算指尖碰撞中心。四个固定 pitch/yaw 候选依次单独 reset，并逐个验证 pregrasp plan/execute、接触位 execute、闭合受阻和 lift 共动；只冻结首个完整通过者，失败阶段写入 blocker。随后要求 5/5 正向流程、四个结构化负例、Gazebo joint inventory 前后一致。抓错负例也必须在闭合后 lift，再按共动物体返回 `WRONG_OBJECT`。只有 Direct 全部通过才会启动真实 SSE MCP 的两轮正向流程和四个负例。

## 本机版本与上游依据

| 组件 | 已安装版本 |
| --- | --- |
| Gazebo Sim | `8.11.0` |
| ros_gz | `1.0.22-1noble.20260616.074726` |
| MoveIt | `2.12.4-1noble.20260617.161956` |
| ros2_control | `4.45.2-1noble.20260615.175135` |
| ros2_controllers | `4.40.1-1noble.20260616.074625` |

- Gazebo Sim 8 OdometryPublisher：<https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1OdometryPublisher.html>
- ros_gz_bridge 正式消息映射：<https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_bridge/README.md>
- MoveIt PlanningScene/CollisionObject：<https://moveit.picknik.ai/main/doc/tutorials/planning_around_objects/planning_around_objects.html>
- MoveIt AllowedCollisionMatrix：<https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html>
- ros2_control Jazzy gripper stall：<https://control.ros.org/jazzy/doc/ros2_controllers/gripper_controllers/doc/userdoc.html>

## 下一步解除阻塞

1. 运行四候选完整物理探针，冻结首个能形成闭合受阻和目标共动的姿态；四个均失败时保留逐阶段 blocker，不引入新控制器或插件。
2. 只有 Direct `5/5`、四负例、SSE `2/2`、清理 gate 都通过，且用户另行安排 M2 正式重验后，才更新 M2/M3 milestone checkbox。
