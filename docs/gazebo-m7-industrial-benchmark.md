# Gazebo M7 industrial benchmark（离线 manifest）

`examples/gazebo_industrial_benchmark_v0.json` 是场景和指标的离线声明；它可被
解析和静态校验，但当前不代表可执行的操控 benchmark。

M3/M4 已 fail-closed，原因是 approved native DetachableJoint 尚未实现。因而
manifest 中的目标尺寸、seed、场景和 Oracle 相关字段只能支持离线工具与契约
测试，不能生成物理抓取成功率、放置成功率或正式 M0–M4 结果。

历史报告是撤销机制的诊断证据，不得用于证明 benchmark、M3 或 M4 通过。待
获得设计确认并实现新的原生 joint 路径后，需重新定义 live runner、证据链和
验收口径。
