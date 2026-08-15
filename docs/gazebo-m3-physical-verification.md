# M3 双垫接触吸附验证

## 结论与边界

M3 的唯一抓取路径是 `bilateral_contact_adhesion_v1`：Gazebo 原生接触
传感器先证明左右真实指垫同时接触同一个已知物体；随后仓库自有
`M3AdhesionSystem` 以确定性运动学方式携带该物体。Gazebo 物理引擎只
用于接触判定和释放后的落稳验证，**不**用于维持抓持。

这不是摩擦夹持、也不是真空物理模型。它的可信性来自可追溯的双垫接
触与独立的抬升/身份硬门；持物行为本身是明确标记的运动学吸附。M3
没有可选抓持后端或运行时机制开关。

## 接触与吸附协议

M3 world 在左右 fingertip 的碰撞体上声明 Gazebo 原生 `contact` sensor，
并加载 `gz-sim-contact-system`。运行时不使用 TF、碰撞网格、OBB、距离
或夹爪 stall 作为接触替代证据。

`gripper_close` 后的放行条件固定为：

- 每一侧至少三条传感器样本；
- 每侧样本时间跨度至少 100 ms，最新样本相对动作完成时间新鲜；
- 两侧都命中同一个 `m3_target` 或 `m3_distractor`；
- 无接触、单侧、混合物体、未知碰撞实体或陈旧样本均拒绝。

`M3AdhesionSystem` 私有状态机为
`arm_contact_window → capture → release`。capture 时插件再次验证双垫
窗口、保存对象身份与相对 `gripper_mount_link` 位姿；持物的每个仿真步
都按该位姿跟随，同时禁用该对象重力/碰撞并清零速度。`gripper_open`
先实际张开、再 release；release 恢复碰撞和动力学、零初速度并等待物理
落稳。

MCP 原子接口不变。close 是接触寻址闭合，双垫合格即吸附；open 先张开
再释放。所有 M3 observation 和 action receipt 都标注
`grasp_mechanism=bilateral_contact_adhesion_v1`。

## 验证语义与 PlanningScene

插件 capture 不是 `TARGET_HELD` 的充分条件。M3 verifier 仍要求：

- 插件吸附收据与目标身份一致；
- target 相对 EEF 位姿稳定；
- target 实际抬升并离开支撑面；
- distractor 没有共动；
- 所有 JointState、TF、RGB-D、接触和对象状态证据新鲜。

任何缺失、歧义或矛盾证据均 fail-closed。只有上述硬门通过后，MoveIt 才
添加 target 的 AttachedCollisionObject；它只服务后续规划，绝不改变
Gazebo 的持物机制。网格中心计算仅用于规划接近位姿，绝不用于接触判定。

reset 和 close 都清除吸附/PlanningScene 状态。Gazebo joint topology 在
抓取前后必须不变。

## 正式验收

M3 Direct 必须完成 5/5 拾放；MCP 必须完成 2/2。正例必须依次产生
`LIFT_REQUIRED`、`TARGET_HELD`、`TARGET_PLACED`。下列负例必须 fail-closed：

- 空抓；
- 单侧或两侧命中不同物体；
- 吸附/抬升错误物体；
- 空中释放；
- 目标区外释放。

M4 使用同一 M3 硬门。它的 Oracle 感知和合约化 fake grasp candidate 仅
验证工具链，不宣称真实 SAM3 或 GraspNet 推理。

非交互式云端正式入口为：

```bash
OPENETA_CLOUD_ACCEPTANCE_ROOT=/data/openeta-cloud-acceptance \
  bash scripts/run_cloud_m0_m4_acceptance.sh
```

入口在数据盘为一个已由 `origin` 引用的 SHA 创建干净 detached clone，
只 build 一次，随后串行运行 M0–M4。它为每个 live 段独立分配 ROS
domain、Gazebo partition 和 MCP port；仅清理携带本次唯一 partition 的
进程组。不可变总报告、每个 milestone JSON、build/launch/MCP/stdout 日志
及清理证据都写在该 clone 的
`.cache/cloud-m0-m4-<UTC>-<SHA>/` 下。前置 gate 未通过时后续 milestone
不会运行，尤其 M4 不会被标记为通过。

## 本地验证

在不运行 live 任务时，至少执行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_gazebo_m3_profile.py tests/test_gazebo_m3_source.py \
  tests/test_gazebo_m3_verifier.py tests/test_gazebo_m3_adhesion.py \
  tests/test_gazebo_m3_worker.py \
  tests/test_gazebo_m3_acceptance.py
.venv/bin/python -m compileall -q extensions/gazebo scripts
bash -n extensions/gazebo/ros2_ws/run_m2_robotiq2f85_smoke.sh \
  extensions/gazebo/ros2_ws/run_m3_pickplace_acceptance.sh \
  scripts/run_cloud_m0_m4_acceptance.sh
git diff --check
```

正式结论只以同 SHA 的云端报告为准；本地或历史诊断输出不是正式验收
证据。
