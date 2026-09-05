# `final-dev` 迁移与复现交付包

本文是评委或新机器管理员的最短可复现路径。它交付的是一个固定 Git revision 的源码包、
一个可重建的 NVIDIA/ROS/Gazebo Docker 镜像，以及可校验的模型资产；模型、API 密钥、
运行缓存和验收证据不会混入源码包或镜像层。

运行行为、物理证明与发行范围仍以
[Gazebo normal 验收契约](gazebo-normal-acceptance.md) 为准；人工操作步骤见
[`multi_normal` TUI 复现指南](multi-normal-tui-reproduction.md)。

## 交付物边界

从已检出的 `final-dev` 工作树生成源码包：

```bash
git switch final-dev
git pull --ff-only origin final-dev
scripts/package_final_dev_bundle.sh
```

默认产物是 `dist/openeta-final-dev-<commit>.tar.gz` 与同目录的 `.sha256` 文件。它由
`git archive` 生成，只含该提交跟踪的源码、Dockerfile、部署脚本、场景资产和文档；不会
包含 `.env`、`apikey.md`、模型权重、`.cache/`、`.openeta_memory/`、ROS build/log 或本机
虚拟环境。接收方先校验再解包：

```bash
sha256sum -c openeta-final-dev-*.tar.gz.sha256
tar -xzf openeta-final-dev-*.tar.gz
cd openeta-final-dev-*
```

也可以直接 clone 后固定记录的 commit；两种方式使用同一份 Docker 定义。不要把 provider
密钥或模型目录打进源码包。

## 目标主机要求

- Linux x86-64，推荐 Ubuntu 24.04；Docker Engine 与 Docker Compose v2。
- NVIDIA 驱动和 NVIDIA Container Toolkit，至少一张可用于 CUDA、EGL/OGRE2 的 NVIDIA GPU。
- 足够的磁盘空间存放镜像、模型和可写运行状态；模型不进入 Git 或 Docker 镜像。
- 出网仅在首次镜像构建与权重下载时需要。运行验收时模型可离线挂载并由哈希校验。

先确认 Docker 可以看到 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Docker 构建、模型与 Provider

在源码根目录，选择一个容量充足的本地磁盘保存权重和状态：

```bash
export OPENETA_MODEL_ROOT=/data/openeta/models
export OPENETA_DOCKER_STATE_ROOT=/data/openeta/state
mkdir -p "$OPENETA_MODEL_ROOT" "$OPENETA_DOCKER_STATE_ROOT"

deploy/ubuntu/openeta.sh build
deploy/ubuntu/openeta.sh fetch-models
deploy/ubuntu/openeta.sh validate-assets
```

`fetch-models` 下载固定的 SAM3、AnyPlace、GraspGenX 和 Robotiq 资产；
`validate-assets` 校验 SHA-256 和上游 revision，并把模型卷保持为只读。若任一项失败，
不要绕过检查或混用不同 revision 的权重。

再建立仅部署用户可读的 provider 文件。下例只展示字段，不应提交真实值：

```bash
umask 077
cat > /secure/openeta-provider.env <<'EOF'
OPENETA_LLM_PROVIDER=openai-compatible
OPENETA_LLM_MODEL=qwen3-vl-flash
OPENETA_LLM_API_BASE=https://YOUR_WORKSPACE.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
OPENETA_LLM_API_KEY=replace-with-secret
OPENETA_LLM_ENABLE_VISION=true
OPENETA_LLM_THINKING_MODE=disabled
OPENETA_LLM_TIMEOUT_S=60
OPENETA_LLM_MAX_ATTEMPTS=3
OPENETA_LLM_MAX_TOKENS=512
EOF
chmod 600 /secure/openeta-provider.env
export OPENETA_PROVIDER_ENV_FILE=/secure/openeta-provider.env
deploy/ubuntu/openeta.sh config
```

Compose 将它作为只读 Docker secret 交给 TUI 子进程；密钥不会进入 image、`docker inspect` 的
环境变量或验收报告。其他 OpenAI-compatible 多模态 provider 也可使用相同字段。

## 评委复现命令

默认正式验证是带 VLM 的 `multi_normal`，连续两次独立 case；第一轮失败立即返回非零：

```bash
deploy/ubuntu/openeta.sh agentic-normal
```

单轮演示可显式写出，避免误认为它等价于稳定性验证：

```bash
deploy/ubuntu/openeta.sh agentic-normal --runs 1
```

无 VLM 控制链 smoke test 与正式 VLM 验收不同，只用于检查基础设施：

```bash
deploy/ubuntu/openeta.sh smoke-normal --runs 2
```

每轮的 `acceptance-report.json`、清理证据、trace、模型资产清单及计时文件在
`$OPENETA_DOCKER_STATE_ROOT/runs/`。PASS 必须同时满足报告 `status=passed`、
`agentic_closed_loop`、`fast_v3`、GraspGenX、以及 host-owned Gazebo 生命周期清理通过。

## 人工 TUI 与连续工作单

启动 TUI 时，Gazebo 环境由启动器先创建；它不是 Agent 工具。操作员在 TUI 中输入自然语言
工作单，例如：

```text
请先把黄色活动扳手放进绿色零件箱，再把红色六角螺栓放进蓝色零件箱。其他物件不要动。
```

该工作单完成后，不关闭 TUI 或 Gazebo，直接输入下一张任务，例如：

```text
把银色扳手也放进绿色料箱。
```

完成的工单与其原始 TUI 用户消息绑定：同一条消息不会被重复配置；只有新的操作员消息可以
在同一物理工作单元中创建下一张工单。对象和料箱由 live view 与 manipulation catalog 决定，
不依赖 Object Memory Bank。宽泛的命令（如“对桌子上的物品进行分拣”）保留给 VLM 根据当前
可见物品和 catalog 自主构造有序工作单，不由场景注入静态任务。

若要直接演示这种开放式整理，而非代表性固定验收工单，使用任务中立入口：

```bash
deploy/ubuntu/openeta.sh --gui open-sort
```

它会启动同一套模型服务和 GPU Gazebo GUI；操作员在 TUI 中输入开放任务。退出后生成的
`operator-session-report.json` 中的 `work_order_outcome: completed` 汇总本轮 trace 的
宿主 `multi_sort_progress`；它不是固定工单验收 PASS。

为 CI 或研发验证真实持续 TUI 行为，可传递一个 JSON 数组；这只是 driver 将自然语言逐条
送进同一 PTY，不会替 Agent 决策：

```bash
export OPENETA_SCRIPTED_TUI_FOLLOW_UP_TASKS='["把银色扳手也放进绿色料箱"]'
deploy/ubuntu/openeta.sh agentic-normal --runs 1
```

容器中的默认验收 verifier 针对单张代表性发行工作单；连续工作单回归应读取完整 trace 与
`scripted-tui-driver.json`，确认 `completed_episode_count`、同一 host lifecycle 与每张 VLM
工单的物理回执，而不要把额外工作单伪装成原固定工单的通过记录。

## 可选 GPU GUI

评委若需同时展示 Gazebo GUI，先准备一个本机 X11/VNC 桌面，再传入当前 display：

```bash
export DISPLAY=:3
export OPENETA_XAUTHORITY="$HOME/.Xauthority"
deploy/ubuntu/openeta.sh --gui agentic-normal --runs 1
```

GUI 是独立的、GPU 渲染的观察客户端，不参与 Agent 决策或验收。没有 GUI 的 headless EGL/OGRE2
路径仍是正式可复现链路。
