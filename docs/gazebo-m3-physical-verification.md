# M3 Gazebo Robotiq 语义抓放物理验证

## 当前结论

M3 的场景、正式 ROS/Gazebo 数据路径、MoveIt PlanningScene、无 ROS 依赖校验器、worker/MCP 路由和验收驱动已经实现。离线合同、SDF/URDF、ROS 包构建、实时 topic/controller/TF/PlanningScene 诊断已通过。

截至 2026-08-10，M3 **未完成正式验收**。校验语义已经改为复用 Robotiq 官方对象检测结果：闭合提前受阻只产生待 lift 验证的候选，目标身份和最终成功由目标/干扰物 Odometry 相对 EEF 的共动证明。旧的双指 Contact 新鲜度门槛、传感器、bridge 和 observation 扩展均已删除。

2026-08-10 的 live 排查解除了最初的规划层阻塞，并把剩余阻塞精确定位到抓取物理。已确认的事实链（均有探针/报告证据）：

1. **原阻塞（接触位 `MOTION_PLAN_FAILED`）根因是 PlanningScene ACM 过窄**：四个固定候选的接触位目标都被 MoveIt 判为 `GOAL_STATE_INVALID`（-27）。逐对豁免实验证明阻塞对是**工作台与夹爪远端连杆**（指尖/指节/掌关节），而非目标物。`M3PlanningSceneModel.initialize` 现在额外放行 `table_touch_links`（八个远端连杆）与桌面、以及 `grasp_touch_links` 与目标/干扰物——持握态的 `/check_state_validity` 实测接触对为 `finger_link`/`inner_knuckle_link ↔ m3_target`，且世界对象 ACM 行在 reset 清除周期中会被裁剪，因此所有豁免必须在 initialize 内重放。
2. **原四候选姿态族（`_q_euler(π, ±60°, yaw)`）物理上不可能完成抓取**：该族的闭合轴随 pitch 倾斜 60°，实测接触下降过程中低位指尖在抓心到达前 ~5 cm 处就把盒子推倒（目标被推走 5 cm 并翻倒）。驱动改用水平闭合轴姿态族 `_grasp_orientation(tilt, azimuth)`；全网格扫掠（tilt 15–90° × 方位角 24 点）证明机械臂在该桌型布局下只能到达 tilt≈55–75°、azimuth≈0° 附近的窄带，候选更新为 `(65,0) (70,0) (75,0) (60,15)`。`(65,0)` 实测接触零扰动、闭合稳定 stall（aperture≈41 mm > 6 mm 门槛）。
3. **夹爪适配器两个物理修复**：stall 成功后保持当前位置（原先继续向全闭合命令推压，把已夹住的盒子挤出）——闭合后目标保持在位已验证；闭合/张开改为 1.5 s 斜坡（降低首触冲击）。
4. **2026-08-10/11 第二轮（本结论的现行状态）**：两轮正式运行（`.cache/reports/m3-pickplace-20260810T223312Z-564675.json`、`...T224016Z-566616.json`）确认规划层已收敛——pregrasp/contact 规划 8 次候选尝试 7 次成功（plan 重试 3 次消除了 99999 采样抖动；reset 现在通过 `return_home` 关节回零 + `set_pose` 显式还原目标/干扰物，因为实测 gz-sim 8.11 的 `model_only` reset **既不还原自由物体位姿也不还原机械臂关节**）。剩余阻塞是**夹爪连杆的仿真保真度**：六个独立位置系统驱动的 2F-85 闭环连杆在闭合时不确定性失步——实测垫面（TF+STL 测量）在早期卡死（假 stall，aperture≈8.1 cm）、全闭但不收敛（横向错开 2-4 cm，`EMPTY_GRASP` 而盒未动）、持握后零力滑脱（lift 时 `rose=0`）与过压弹出之间随机分布。挤压参数扫掠（stall 保持偏移 0/0.01/0.02/0.025/0.03/0.05/0.08 rad，逐关节保持）未找到稳定窗口：正式两轮中闭合结果分别为 `LIFT_REQUIRED`（随后 lift 因盒被挤出后目标位姿垃圾而失败）与 3×`EMPTY_GRASP`。持握成功率估计 ≤1/3，不满足 5/5 正轮的验收门槛。

清理 gate 方面，分区探针获得与 domain 探针相同的有界重试（WSL2 上 Gazebo Transport 发现滞后于进程退出），不再把单纯的发现滞后误报为残留。已知缺口：Direct 驱动进程内启动的 launch 栈与驱动同进程组，驱动异常退出时清理路径按设计不信号自身组，孤立的 gz/bridge/move_group 需按 partition 手动回收（本轮两轮正式运行后均如此清理）。

## Detachable-joint fallback（用户 2026-08-10 解禁）

plan-M3.plan 曾显式禁止 detachable joint 等附着机制；用户 2026-08-10 拍板将其解禁为**受控 fallback**。机制与语义边界：

- **机制**：`gz::sim::systems::DetachableJoint` 插件（两个实例，`gripper_mount_link ↔ m3_target` 与 `↔ m3_distractor`）仍由 xacro 参数控制，但 detachable launch 会将生产 xacro 渲染/转换为临时 SDF，验证两实例后仅添加 `world → base_link` 固定根；`self_collide` 显式为 `false`。physics launch 继续以原 URDF 字符串 spawn。OpenETA 侧由 `GazeboDetachableJointControl`（`process.py`）维护 `UNKNOWN/DETACHED/ATTACHED` 请求 ACK 状态；该状态不是物理成功证据。每个 attach ACK 还会记录真正 child link 的 Pose_V 相对基线；只有该 link 共动与既有 Odometry 同时通过，MoveIt `AttachedCollisionObject` 才可参与后续**规划**碰撞语义。
- **开关**：`GazeboDeploymentConfig.m3_attachment_mode` 由 `OPENETA_M3_ATTACHMENT_MODE` 读取，`physics`（默认）完全不加载插件、不创建执行器——现状行为零影响；验收报告与观测 metadata 均标注 `attachment_mode`。
- **诚实语义（红线）**：attach 只在 close 产生 `LIFT_REQUIRED` 后，双侧冻结指垫 STL/TF 在 action barrier 后持续至少 0.10 s（≥3 个样本）同时满足双侧间隙、切向重叠、法向方向、低速和唯一候选时触发。随后必须回退到 `0.15 rad` 并再次确认双侧净空；生命周期为 `DETACHED → CONTACT_CONFIRMED → ATTACH_ACKED_UNPROVEN → HELD_PROVEN`。`gripper_open` 严格先 detach ACK、再实际 open、再 fresh observation、最后 PlanningScene release。验证器的 lift 共动、三态 verdict、六场景判定仍只读取 Odometry，attach 本身不构成证据。
- **离线合同**：`tests/test_gazebo_m3_attachment.py`（门逻辑：无 stall 不 attach、空位不 attach、错物选 distractor、模式默认/校验、runtime 执行器开关）。

**验收结果（2026-08-11，detachable 模式三轮正式运行）**：
`...T034841Z-576217.json`、`...T035724Z-578522.json`、`...T040543Z-580628.json`，均 `blocked`。几何门按设计工作：run 2 中 `(65,0)` 的 close 弹出盒子后，门正确判定垫区无物体而**未** attach（target 已被弹到 0.82 m 外）。三轮中闭合结果分布：stall 2/9、空抓 5/9、弹出 1/9、pregrasp 规划失败 2/12。即使 fallback 就位，**触点事件本身（垫面命中 4 cm 盒面）的可靠性不足**——六根失步使垫面落点逐次漂移 1-2 cm，与盒体 ±2 cm 净空同量级。因此六场景未跑通，milestone 保持 blocked。

**历史排查结论已撤销（2026-08-15）**：此前把 `self_collide=true` 加到真实 RM75 的 URDF 导入模型，触发了 DART mesh 碰撞断言；未固定的机器人根又使跨模型固定关节随真实关节运动漂移。`gz model -j` 和插件 state topic 只显示请求/组件信息，不能证明物理刚性约束。detachable fallback 现在只在专用临时 SDF 中使用 `world → base_link` 固定根，显式保持 `self_collide=false`；physics 默认路径完全不变。真实 RM75 硬门改以生产 launch、`FollowJointTrajectory` 和 `Pose_V` 共动测量执行。

**实现后隔离（2026-08-15，非正式验收）**：临时 SDF 的生产 launch 已成功 spawn，真实 `rm_group_controller` 以 `FollowJointTrajectory` 驱动 `joint_1` 的实测 `0.348–0.350 rad` 轨迹，且 `gripper_mount_link` 的世界姿态相应变化约 `0.350 rad`。同一安装上的 DART 控制组均连续 3/3 通过：单链、固定根动态关节链、以及“动态关节 → 固定末端”的 RM75 同构拓扑。故 `DetachableJoint`、固定根和固定末端都不是一般性阻塞。

**真实 RM75 仍 fail-closed**：固定根下，`gripper_mount_link` 与真正渲染为 `link_7` 的插件父链接对照均出现相同的不一致。模型 `m3_target` 的 Pose_V 在 attach/detach 后仍随父链接；实际被 joint 指定的 `target_link` 却在 attach 期间相对父链接变化约 `0.350 rad`。即模型实体与 child link 的 Pose_V 对同一个物理 joint 给出互相矛盾的结果，state topic 同时仍回 `detached`。去掉世界固定根会使这两个证据部分变化，却立即产生 `0.17 rad / 6 mm` 级 attached 漂移。因而当前 `gz-sim 8.11 / gz-physics 7.6 DART` 与完整 RM75/ros2_control 图的跨模型实体同步不可作为物理证据；fallback 必须继续拒绝验收，不能依 topic、`gz model -j`、模型 pose 或链接 pose 的任一种单独放行。报告在 `.cache/detachable-dynamic-current2/`、`.cache/detachable-dynamic-fixed-terminal/`、`.cache/rm75-detachable-repro-positive/`、`.cache/rm75-detachable-repro-unfixed-root/` 和 `.cache/rm75-detachable-repro-link-posev/`；均为未清洁工作树的诊断，不能替代同 SHA clean-source 正式报告。

**修复边界**：仓库侧已将 production SDF、真实 FJT、父链接/固定根对照和 model/link 双口径的矛盾复现固定下来；运行时也会在 child-link 共动缺失时将原有的 `TARGET_HELD` 回退为既有 `TARGET_NOT_LIFTED`，随后 detach。真正修复需要 Gazebo 上游在该完整 DART skeleton 上修正 detach 后的跨模型 link/model pose 同步（或提供经此 repro 验证的 vendor 更新）；在此之前不引入自定义吸附插件，也不把任何 ACK 扩写为物理成功。

## 统一驱动攻关（2026-08-11，用户拍板方向 ①）

- **四连杆闭式解**（`extensions/gazebo/robotiq_kinematics.py`，纯 Python）：A/B/C 枢轴取 vendor URDF 精确值，耦合点 D 由碰撞网格镗孔圆拟合（std≤0.55 mm），|BD| 取零位精确闭环。数值结论：**vendor 的常量 mimic 与精确解全行程闭环误差 <0.03 mm**——六关节目标本就几何一致，失步不是目标不一致导致的。单测 `tests/test_gazebo_robotiq_kinematics.py`（闭环误差、端点开度与 FK 表一致、单调性、限位）。适配器 `drive_mode=four_bar|multiplier`：M3 launch 用 four_bar，M2 launch 固定 multiplier（其 mimic 合同 0.035 rad 容差与四活塞偏差叠加会越界）。
- **适配器斜坡改 lead-limit**（`max_lead_rad=0.06`）：目标最多领先实测 0.06 rad——消除均匀慢速斜坡的卡死假 stall（8.1 cm 假 stall 根因是 blocked-pause 死锁），同时避免全行程过压。两段速保留。
- **MoveIt attach 修复**：attach diff 里显式 world REMOVE 与 attach 自动移除世界对象重复，导致 `ApplyPlanningScene` 返回 `success=False`——去掉显式 REMOVE 后 attach/release/clear 全链路 live 验证通过。
- **接触位 +9 mm 偏置**（驱动常量 `GRASP_CENTER_Z_BIAS_M`）：相机实况显示中心高度接触让盒子被垫面从下方挤出；上移后垫面咬在盒体上三分之一。探针最好成绩 **8/8 连续干净持握**（contact_moved=0、stall、attach、lift `TARGET_HELD`）。
- **逐 reset 抓心校准**：实测闭合中线相对盒心偏移以每轮 3-10 mm 累积（连杆失步漂移），`GazeboRuntime.reset` 现在在每次 reset 于接触孔径（0.41 rad）处重测抓心偏移（`grasp_center_offset_m`），驱动每个候选/正轮/负例 reset 后取新值。
- **reset 确定性与串行化**：`set_pose` 加重试；还原为 detach→回零→set_pose 单段（曾用双段：第一段会在持盒 reset 时把盒子实体化进闭合指间弹射，2026-08-11 下午改序，见 detachable 验收记录）；M3 run 脚本 direct 驱动改 setsid，堵上同组 launch 泄漏缺口；探针包装器按 partition 强制清场。
- **剩余失效模式（当前状态）**：近 boot 间方差仍大——同一构建在不同 boot 下探针分布从 8/8 到 1/8 摆动，正式运行（physics：`...T065939Z`、`...T071311Z`；detachable：`...T073755Z`、`...T091952Z`）均止步于闭合命中率（候选冻结成功率约 50-75%，单轮成功率更高但 5/5 连过仍不足）。最新两轮 detachable 首次走通 **lift→TARGET_HELD→transport→lower→release** 全链：`...T091952Z-628664.json` 止于放置（附着体 touch_links 过窄致 transport/lower 规划 99999）；`...T092725Z-630483.json`（touch_links=`grasp_touch_links` + 目标/干扰物↔桌面 ACM 修复后）正轮 1 走到 transport 仍遇一次 99999 目标采样抖动，随后序列为 OBJECT_DROPPED 收场。规划采样抖动（OMPL/KDL 目标采样在高负载下偶发 30s 无解）是当前最后一层非物理阻塞。
- **M2 复验通过**：`m2-robotiq2f85-acceptance-20260811T073214Z-611375.json`（统一驱动适配器 + multiplier 模式），七 gate 全绿。

### 收官轮攻关（2026-08-11 下午，physics 三轮 + detachable 八轮正式运行）

- **规划层 99999 根因与修复**：扫掠探针（plan_only ×12/姿态）证明 pregrasp/contact/lower 全 12/12（0.03-0.05 s），而 transport（tilt 65°、+80 mm、mount x≈0.355）**系统性 0/12**（每次耗尽 8 s 预算）——不是随机抖动。修复：`config/kinematics.yaml` 的 `kinematics_solver_timeout: 0.05 → 0.2`（WSL2 负载下 KDL 50 ms 解不完导致目标采样全灭）；搬运高度 `CARRY_HEIGHT_M` 定为 `0.060`（+80 mm 不可采样；+30 mm 可规划但吊盒/指尖擦桌甩盒；高度扫掠 +40/50/60/70 mm 全 6/6，+60 mm p50≈0.85 s 为安全边际最高点）。此后 99999 未再出现。
- **抓取关键 move 目标域收紧**（驱动常量 `GRASP_MOVE_PARAMS`：1 mm / 0.01 rad，轨迹缩放 0.1）：实测同一接触位姿两轮执行落点相差 ~4 mm / ~3°（默认 2 mm/0.05 rad 目标域各吃满所致），足以翻转边际抓取的成败；收紧后接触位姿复现到 ~1 mm / ~0.6°。慢速缩放同时消除 OMPL 关节空间路径的腕部急转对持握/附着载荷的扰动。`make_move_group_goal` 与 MCP `move_to` 新增 scaling 透传（默认 0.3 不变，M2 契约不受影响）。
- **抓取物理调参链**（每步均有探针/正式报告证据）：① 桌面 mu 1.0→1.5——首触垫面在 stall 判定前的 1 s 窗口内持续推盒 7-11 mm；② 适配器**逐侧 stall 冻结**——原实现只盯主动关节速度，先接触一侧的指垫在 lead-limit 下持续推压造成"携带"（盒在闭合中位移最高 31 mm），改为左右两侧各自监测冻结（`stall_timeout` 0.3 s，`side_moved` 门防启动假 stall）；③ `stall_hold_extra_rad=0.03`——纯冻结把各关节保持于实测位，PID 误差归零即**加持力为零**，盒子只是被"关笼"而非"夹紧"，lift 打破静摩擦即滑脱；④ 盒 mu 1.2→2.0、指尖 mu 1.5→2.0——可达抓取带 tilt 60-75° 使垫面相对盒面倾斜 ~25°，接触为垫缘线接触（铰链），高摩擦抑制绕铰链翻转。探针最好成绩 3/3 连续干净 TARGET_HELD。
- **physics 模式残余失效**（`...T121707Z`、`...T122641Z`、`...T123942Z`，均 blocked）：单次 close→lift 成功率约 50-75%，失败签名为 lift 中盒沿垫面**侧向滑脱或绕垫缘翻转 20-40°**（相对漂移超 0.15 rad 容差），距 Direct 5/5 连过门槛仍不足。正式轮中观察到失败集中于 reset 后的第二轮抓取，但探针三轮连测（含首轮即失败样本）证明这是边际抓取的统计涨落而非 reset 状态污染。
- **GPU 渲染评估（WSL2）**：本机 RTX 4060 Laptop；默认 GL 栈为 Mesa llvmpipe（纯软件渲染，RGB-D 相机负载使 sim 低于实时，是既有时序压力背景）。`GALLIUM_DRIVER=d3d12 glxinfo -B` 实测显示 "D3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)"——Mesa dzn 路径可启用 GPU 加速。**未用于正式验收**：观测契约（相机分辨率/频率/时间戳语义）不变性优先，且规划/物理阻塞与渲染端无关；验收脚本在 GPU 可用时记录 `INFO RENDERER NVIDIA_GPU_AVAILABLE`。GPU 卸载作为后续可选优化记录。
- **当前唯一未试手段**：垫面反馈闭环闭合（按 TF 实测两垫间距收敛到目标宽度并维持微压力，取代开环位置斜坡）——直接消除"冻结点随机→加持力随机"的残余方差；detachable 方向需先做最小复现确认 dartsim 关节创建（见上节结论）。

**下一步（当前范围）**：仅保留 detachable 隔离路径。先以本仓库的 RM75 Pose_V hard gate 验证 Gazebo vendor 更新或上游补丁；在 model/link 两口径连续通过 3× target、3× distractor 的 attach/detach 周期前，fallback 一律 fail-closed。physics 抓取与 M2 live 复验不属于本轮。

## 实现边界

- 环境 ID：`openeta/gazebo_rm75_robotiq2f85_pickplace-v0`
- 模型 ID：`rm75_robotiq_2f85_pickplace_sim_v1`
- 使用现有 `create/reset/observe/move_to/gripper_open/gripper_close/close` 原子接口。
- physics 模式只使用碰撞、重力、摩擦和 OdometryPublisher；没有 Contact sensor、吸附命令、外力抓取插件或运动学附着。detachable 模式是受控 fallback：它只在经过双指垫几何门、0.15 rad 回退净空检查后发送 DetachableJoint 请求，并且仍须由下一次 lift/transport 的 Odometry 与真实 child-link Pose_V 共动共同产生 `TARGET_HELD`。
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
| 目标摩擦 | ODE `mu=mu2=2.0`（1.2 起调：垫面倾斜线接触铰链抑制） |
| 桌面摩擦 | ODE `mu=mu2=1.5`（1.0 起调：防滑工作台面，抑制闭合期推盒） |
| 指尖摩擦 | `mu1=mu2=2.0`（1.5 起调，同上铰链抑制） |
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

reset 后加入工作台、目标和干扰物。完整 SRDF AllowedCollisionMatrix 会先从 `/get_planning_scene` 读取，再对称放行三组实测必要接触对：目标物/干扰物与 `grasp_touch_links`（两指尖+两指节+两掌关节+两内掌关节——持握态实测接触对）、工作台与 `table_touch_links`（同一远端集合——接触位规划实测需要）。禁止发送部分 ACM，因为实测 MoveIt 会把它作为替换矩阵，擦除既有相邻链路豁免并导致起始态自碰撞。世界对象的 ACM 行在 reset 的 clear 周期中被裁剪，因此豁免必须在每次 initialize 重放。

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

1. 解决 2F-85 闭环连杆的 Gazebo 保真度（当前唯一卡点）。收官轮已把失效收敛到 lift 持握边际（侧滑/翻转），下一步即**垫面反馈闭合**（按 TF 实测两垫间距收敛到目标宽度并维持微压力）；备选：gz 侧用关节约束/闭环耦合替代六独立位置驱动；或软化目标接触参数（kp=1e5 对 100 g 轻物过硬）。任选其一都需保持 M2 的 reached-goal 契约与既有验收证据不破（需要时重跑 M2 正式验收）。
2. 连杆闭合可靠后，用 `/tmp` 探针（`m3_visual_probe.py` 模式）量化闭合成功率 ≥90% 再跑正式验收。
3. 只有 Direct `5/5`、四负例、SSE `2/2`、清理 gate 都通过，才更新 M3 milestone checkbox。M2 正式验收已于 2026-08-10 通过（见 `docs/gazebo-m2-rm75-robotiq2f85.md`）；注意收官轮改动了适配器/launch/kinematics.yaml，M2 live 复验待执行。
