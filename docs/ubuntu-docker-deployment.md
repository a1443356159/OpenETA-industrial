# OpenETA Ubuntu Docker 部署

普通单机 Docker 运行层位于 `deploy/ubuntu/`；Slurm/Apptainer 适配层位于
`deploy/HPC/`。两者共用 `deploy/ubuntu/Dockerfile` 这一份 Ubuntu、CUDA、ROS、
Gazebo、MoveIt 和模型服务依赖定义，不维护两套容易漂移的镜像。

容器内包含四个隔离环境：

| 环境 | 路径 | 用途 |
| --- | --- | --- |
| OpenETA | `/opt/openeta/venvs/openeta` | Python 3.12、TUI、ROS/MoveIt 控制链 |
| SAM3 | `/opt/openeta/venvs/sam3` | SAM3 MCP 服务 |
| AnyPlace | `/opt/openeta/venvs/anyplace` | Python 3.10、Torch 1.13/cu117 放置服务 |
| GraspGenX | `/opt/openeta/venvs/graspgenx` | GraspGenX 抓取服务 |

AnyGrasp 不在普通容器的默认启动链中。当前 normal 使用
`SAM3 + AnyPlace + GraspGenX + fast_v3 + MoveIt/Gazebo`。

## 主机要求

- Linux x86-64；推荐 Ubuntu 24.04。
- Docker Engine 和 `docker compose` v2。
- NVIDIA 驱动与 NVIDIA Container Toolkit；主机无需安装 CUDA、ROS 或 Python 依赖。
- 完整镜像和模型需要较大的磁盘空间。模型不进入镜像，也不进入 Git。

先验证 GPU 容器通路：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## 首次构建与模型准备

在仓库根目录执行：

```bash
deploy/ubuntu/openeta.sh build
deploy/ubuntu/openeta.sh fetch-models
deploy/ubuntu/openeta.sh validate-assets
```

默认目录为：

```text
.cache/docker/
├── models/   # 只读挂载到容器；固定 revision 的权重和 gripper assets
└── state/    # HOME、cache、ROS 日志、TUI memory 和验收证据
```

`validate-assets` 会校验所有模型和上游源码 revision，并在 `state/` 中建立可写的
SAM3 Hugging Face cache 视图。因此运行时模型目录保持只读，容器不能改写 checkpoint。
如需使用其他磁盘，先显式设置两个绝对路径：

```bash
export OPENETA_MODEL_ROOT=/data/openeta/models
export OPENETA_DOCKER_STATE_ROOT=/data/openeta/state
deploy/ubuntu/openeta.sh fetch-models
```

## 常用入口

```bash
# 容器 shell
deploy/ubuntu/openeta.sh shell

# OpenETA 交互式 TUI（自动启动 simulator、SAM3、AnyPlace、GraspGenX MCP）
deploy/ubuntu/openeta.sh tui

# 无 Planner/VLM 的 normal 控制链，默认连续两轮
deploy/ubuntu/openeta.sh smoke-normal

# 带 Planner/VLM 的最终双物件 `multi_normal`，默认连续两轮
deploy/ubuntu/openeta.sh agentic-normal

# 容器内测试
deploy/ubuntu/openeta.sh test tests/test_hpc_deployment.py -q
```

每组验收证据位于 `.cache/docker/state/runs/<profile>-<timestamp>/`。第一轮失败会
立即返回非零，不用第二轮重试掩盖故障。可显式修改轮数和场景：

```bash
deploy/ubuntu/openeta.sh smoke-normal --runs 1 --scenario normal
```

`agentic-normal` 的默认场景是最终的任务中立 `multi_normal` 双物件分拣；
`smoke-normal` 仍只支持单物件 `normal`，不能替代带 VLM 的验收。Docker 入口也支持
同一物理场景内的单、双和三物件自然语言工单。`--task-variant` 仅给验收器选择私有
核对契约，实际任务语义仍来自 Planner 在 TUI 中读取的工单：

```bash
# 单物件：蓝黑色螺丝刀 → 绿色料箱
deploy/ubuntu/openeta.sh agentic-normal --runs 2 \
  --scenario multi_normal --task-variant screwdriver-green

# 默认双物件：黄色活动扳手 → 绿色料箱；红色六角螺栓 → 蓝色料箱
deploy/ubuntu/openeta.sh agentic-normal --runs 2

# 三物件：螺丝刀、银色梅花扳手、蓝柄钢丝钳依次分拣
deploy/ubuntu/openeta.sh agentic-normal --runs 2 \
  --scenario multi_normal --task-variant three-tools-a

# 固定 seed 的随机布局；仍使用同一权威场景编译器和碰撞模型
deploy/ubuntu/openeta.sh agentic-normal --runs 2 \
  --scenario multi_normal_random_12345 --task-variant mixed-tools-b
```

Planner/VLM 配置默认从仓库内已忽略的 `.env` 以 Docker secret 只读挂载；它不会作为
Compose `env_file` 或 Compose 自身的插值配置读取，也不会出现在容器 inspect 配置中。
也可指定独立文件：

```bash
export OPENETA_PROVIDER_ENV_FILE=/secure/openeta-provider.env
deploy/ubuntu/openeta.sh agentic-normal
```

`.env`、`apikey.md`、本机 MCP registry、checkpoint、license 和运行证据均被
`.dockerignore` 排除，不会进入 build context 或镜像层。不要把 API key 写入
`compose.yaml` 或 Dockerfile。

### 百炼 Qwen Planner/VLM

最终部署使用百炼 OpenAI-compatible Vision 接口时，创建一个仅由部署用户可读的
provider secret。以下示例使用北京业务空间端点；将 `WORKSPACE_ID` 和密钥替换为
控制台实际值，**不要**把密钥提交到 Git：

```bash
umask 077
cat > /secure/openeta-bailian-qwen.env <<'EOF'
OPENETA_LLM_PROVIDER=openai-compatible
OPENETA_LLM_MODEL=qwen3-vl-flash
OPENETA_LLM_API_BASE=https://WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
OPENETA_LLM_API_KEY=replace-with-bailian-api-key
OPENETA_LLM_ENABLE_VISION=true
OPENETA_LLM_THINKING_MODE=disabled
OPENETA_LLM_TIMEOUT_S=60
OPENETA_LLM_MAX_ATTEMPTS=3
OPENETA_LLM_RETRY_BACKOFF_S=0.5
OPENETA_LLM_MAX_TOKENS=512
EOF
chmod 600 /secure/openeta-bailian-qwen.env
export OPENETA_PROVIDER_ENV_FILE=/secure/openeta-bailian-qwen.env
deploy/ubuntu/openeta.sh config
```

`qwen3-vl-flash` is the recommended low-latency visual Planner profile here:
it supports image input, function calling and structured output through the
OpenAI-compatible Vision API. The official endpoint form and model capability
are documented by [Alibaba Cloud Model Studio](https://help.aliyun.com/zh/model-studio/qwen-vl-compatible-with-openai)
and [the qwen3-vl-flash model page](https://help.aliyun.com/zh/model-studio/qwen3-vl-flash).
The OpenETA provider preflight records the endpoint and model with the key
redacted. Its `/models` lookup is diagnostic only because workspace-scoped
OpenAI-compatible endpoints may omit it; a direct structured chat request to
the configured model is the compatibility gate. A provider or service failure
is an infrastructure failure, never a candidate rejection.

The final Docker profile freezes 512 GraspGenX candidates once and consumes
them in the `fast_v3` small-wave funnel. It does not issue 512 eager IK or L5
requests. Set `OPENETA_GRASPGENX_RAW_POOL_SIZE=1024` only for a measured
coverage experiment after a 512-pool miss; it is not the release default. The
default ladder then extends through 512 before the final 1024 tail, so the
larger reserve remains lazy at the IK/L5 layers.

## GPU GUI / VNC 转发

默认采用 headless EGL/OGRE2。需要把独立 Gazebo client 或其他 Qt 窗口显示到主机已有的
X11/VNC 桌面时，使用 `--gui`。例如 VNC 桌面为 `:3`：

```bash
export DISPLAY=:3
export OPENETA_XAUTHORITY="$HOME/.Xauthority"
deploy/ubuntu/openeta.sh --gui shell
```

GUI 模式只额外传入当前 X11 socket、只读 Xauthority 和 NVIDIA
`graphics,display` capability；它不会启动新的 VNC server。主机应先创建并连接目标
桌面。formal normal 的权威 Gazebo worker 仍采用独立 headless server，并通过返回的
Dashboard URL 观察；`--gui` 不会把 GUI 进程塞入验收关键执行器。容器不使用
`--privileged`。

## 开发模式与不可变运行

默认命令执行镜像中 `/opt/openeta/src` 的不可变源码。需要即时验证当前工作树的 Python
修改时加 `--dev`：

```bash
deploy/ubuntu/openeta.sh --dev test tests/test_openeta_cli.py -q
deploy/ubuntu/openeta.sh --gui --dev smoke-normal --runs 1
```

当前 checkout 会以只读方式挂到 `/workspace/openeta`；HOME、memory 和证据仍写入
state volume。ROS overlay 仍使用构建进镜像的版本，因此修改 URDF、launch、controller
或其他 ROS workspace 文件后必须重新 `build`。正式验收和发布证据不要使用 `--dev`。

## 运行边界

- Compose 使用 host network，以便动态 Dashboard 端口、ROS 2 discovery 和本机浏览器
  直接工作；默认 `ROS_LOCALHOST_ONLY=1`，不会发现外部机器人。
- 接真实 ROS 网络时显式设置 `OPENETA_ROS_LOCALHOST_ONLY=0`，并按设备权限单独增加最小
  `device`/group 映射；不要直接改成 privileged。
- 通过 `OPENETA_NVIDIA_VISIBLE_DEVICES=<index>` 固定 GPU。
- `OPENETA_SHM_SIZE` 默认 `8gb`，供 PyTorch、相机帧和 ROS 进程共享内存使用。
- 同一主机同时运行多个实例时，必须用 `OPENETA_MCP_PORT`、`OPENETA_SAM3_PORT`、
  `OPENETA_ANYPLACE_PORT`、`OPENETA_GRASPGENX_PORT` 以及 ROS/Gazebo domain 分配不同值。

HPC/Slurm 的 OCI digest、SIF 导入和 sbatch 流程见
[HPC 容器部署](hpc-container-deployment.md)。
