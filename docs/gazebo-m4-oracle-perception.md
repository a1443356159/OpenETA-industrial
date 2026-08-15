# M4 Gazebo Oracle 感知模块

`oracle_perceive` 是与 SAM3 输出形态兼容的、模拟器真值投影工具。它从 worker
缓存的物体姿态和相机内外参生成 mask、bbox 与 score，并以
`perception_source="gazebo_oracle"` 标记来源。它不运行 ROS 或 Gazebo 推理，
也不声称真实 SAM3 视觉能力。

物体注册表保留 M3 场景的 box/cylinder 尺寸，供离线感知契约使用；这不是 M3
可执行场景。可使用 contract-shaped fake grasp candidate 验证参数格式和调用链，
但它不是真实 grasp 生成或抓取确认。

当前 M4 操控、lift、hold、place 与 live/cloud 验收均 fail-closed：创建对应
环境会返回 `DETACHABLE_JOINT_UNIMPLEMENTED_OR_UNAPPROVED`。在获得
DetachableJoint 的设计确认和实现前，不存在 M4 正式验收入口，也不得把 Oracle
或 fake candidate 的测试结果报告为物理操控成功。
