# Gazebo unified runtime acceptance

M1 和 M2 仍使用 profile 驱动的统一运行时：运行时负责 ROS 2 launch、相机、
控制器、fresh-observation 和清理边界。M2 的夹爪保护及 articulated-handle
资产不属于 M3 撤销范围，继续按各自契约维护。

M3/M4 操控当前不在该运行时的可用能力集合中。M3 profile 的构造即返回
`DETACHABLE_JOINT_UNIMPLEMENTED_OR_UNAPPROVED`；不会启动 Gazebo、ROS、
MCP、规划附着或任何兼容回退。M4 的 Oracle 感知模块可继续做像素级契约
测试，但不构成真实视觉推理或可执行抓取。

以往的 M3/M4 cloud、Direct 和 MCP 结果仅可用于定位被删除方案的问题，不能
作为正式验收证据。后续正式验收的调用链、PTY transcript 和远端隔离规则须在
获得 DetachableJoint 设计确认后重新定义。
