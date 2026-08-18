# Gazebo SAM3 资产与服务部署

本文记录 M5 使用的第三方模型资产、隔离运行环境和可复现校验方式。它是部署清单，
不把 checkpoint 纳入 Git，也不改变 OpenETA 的 Tool API、环境路由或验收门禁。

## 已验证资产

2026-08-18 在远端 RTX 4090 节点验证了以下固定版本：

| 资产 | 来源与版本 | 完整性 |
| --- | --- | --- |
| SAM3 源码 | `facebookresearch/sam3` commit `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da` | 独立安装在 SAM3 service venv |
| SAM3 模型仓库 | [ModelScope `facebook/sam3`](https://www.modelscope.cn/models/facebook/sam3) commit `96f3e1b404ba14f2cfac60ee6ae87c269a7b7923` | Git commit 固定 |
| `sam3.pt` | 上述 ModelScope 仓库的 Git LFS 对象，3,450,062,241 bytes | SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |

这些资产仍受各自上游许可和使用条款约束。ModelScope 是本次部署的下载来源；表中的
哈希是 OpenETA 实测资产校验值，不表示 OpenETA 重新分发或重新授权模型。

## 远端目录布局

已验证节点使用下列布局。服务和模型放在仓库之外，避免误提交大文件：

```text
/root/autodl-tmp/openeta-services/sam3/
├── source/                         # 固定版本的官方 SAM3 源码
├── venv/                           # Python 3.12 隔离服务环境
├── modelscope-facebook-sam3/       # ModelScope Git/LFS 模型仓库
│   ├── config.json
│   └── sam3.pt
└── hf/hub/models--facebook--sam3/  # 离线兼容缓存，不复制 checkpoint
    ├── refs/main
    └── snapshots/<revision>/
        ├── config.json -> ModelScope checkout
        └── sam3.pt -> ModelScope checkout
```

应用 venv 与 SAM3 service venv 必须分离。后者已验证为 PyTorch `2.10.0+cu128`、
CUDA 12.8 和 RTX 4090；模型可在 `cuda:0` 加载并完成真实文本分割。加载后实际服务
显存占用约 5.6 GiB，部署时仍应为瞬时峰值和 Gazebo 留出余量。

## 下载与校验

只拉取 M5 所需的 `sam3.pt`，避免同时下载内容重复的 `model.safetensors`：

```bash
export SAM3_SERVICE_ROOT=/root/autodl-tmp/openeta-services/sam3
export SAM3_MODEL_DIR="$SAM3_SERVICE_ROOT/modelscope-facebook-sam3"

GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://www.modelscope.cn/facebook/sam3.git "$SAM3_MODEL_DIR"
git -C "$SAM3_MODEL_DIR" checkout \
  96f3e1b404ba14f2cfac60ee6ae87c269a7b7923
git -C "$SAM3_MODEL_DIR" lfs pull \
  --include="sam3.pt" --exclude="model.safetensors"
sha256sum "$SAM3_MODEL_DIR/sam3.pt"
```

只有大小和 SHA-256 都与资产表一致时才继续。不要把 HF token、SSH 密码或其他凭据
写入仓库、命令行历史、服务日志和验收 evidence；已经在聊天或日志中暴露的 token
应撤销并轮换。

## 隔离服务环境

源码和 CUDA 依赖只安装到 SAM3 专用 venv。以下版本组合已在远端验证；不同 CUDA
驱动环境应选择 PyTorch 官方提供的匹配 wheel，但仍须保持服务与 OpenETA 应用环境
隔离：

```bash
export SAM3_SOURCE_DIR="$SAM3_SERVICE_ROOT/source"

git clone https://github.com/facebookresearch/sam3.git "$SAM3_SOURCE_DIR"
git -C "$SAM3_SOURCE_DIR" checkout \
  8f0b7f4d4e7eda2ed606ebde6702c93359ad01da

python3.12 -m venv "$SAM3_SERVICE_ROOT/venv"
"$SAM3_SERVICE_ROOT/venv/bin/python" -m pip install --upgrade pip
"$SAM3_SERVICE_ROOT/venv/bin/python" -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$SAM3_SERVICE_ROOT/venv/bin/python" -m pip install \
  -r tools/requirements-sam3.txt
"$SAM3_SERVICE_ROOT/venv/bin/python" -m pip install -e "$SAM3_SOURCE_DIR"

"$SAM3_SERVICE_ROOT/venv/bin/python" -c \
  'import sam3, torch; assert torch.cuda.is_available(); print(torch.__version__)'
```

`tools/requirements-sam3.txt` 把 `setuptools` 保持在 `<81`，因为当前固定版本 SAM3
仍导入 `pkg_resources`。不要把该兼容依赖扩散到默认应用 venv。

## 离线缓存桥接

当前官方 SAM3 builder 通过 Hugging Face Hub 的 `facebook/sam3` repo id 查找
`config.json` 和 `sam3.pt`。ModelScope checkout 可按 Hugging Face cache 目录约定
只读桥接，无需复制 3.45 GB checkpoint：

```bash
export SAM3_HF_HOME="$SAM3_SERVICE_ROOT/hf"
export SAM3_REVISION=96f3e1b404ba14f2cfac60ee6ae87c269a7b7923
export SAM3_SNAPSHOT="$SAM3_HF_HOME/hub/models--facebook--sam3/snapshots/$SAM3_REVISION"

mkdir -p "$SAM3_SNAPSHOT" \
  "$SAM3_HF_HOME/hub/models--facebook--sam3/refs"
ln -sfn "$SAM3_MODEL_DIR/config.json" "$SAM3_SNAPSHOT/config.json"
ln -sfn "$SAM3_MODEL_DIR/sam3.pt" "$SAM3_SNAPSHOT/sam3.pt"
printf '%s' "$SAM3_REVISION" > \
  "$SAM3_HF_HOME/hub/models--facebook--sam3/refs/main"
```

`refs/main` 不应带尾随换行；已验证环境的 `huggingface_hub` 会把其完整内容当作
snapshot 名。启动前应设置 `HF_HUB_OFFLINE=1`，以确认推理只依赖本地固定资产。
这是部署兼容层，不是新的产品配置字段，也不改变模型 repo id。

## 启动与检查

在 OpenETA 仓库根目录启动隔离服务：

```bash
export HF_HUB_OFFLINE=1
export OPENETA_SAM3_URL=http://127.0.0.1:8773/sse

.venv/bin/python scripts/openeta_mcp_services.py start sam3 \
  --host 127.0.0.1 \
  --sam3-port 8773 \
  --state-dir "$SAM3_SERVICE_ROOT/state" \
  --sam3-python "$SAM3_SERVICE_ROOT/venv/bin/python" \
  --sam3-hf-home "$SAM3_HF_HOME"

.venv/bin/python scripts/openeta_mcp_services.py health sam3 \
  --host 127.0.0.1 --sam3-port 8773 \
  --state-dir "$SAM3_SERVICE_ROOT/state"
.venv/bin/python scripts/openeta_mcp_services.py smoke sam3 \
  --host 127.0.0.1 --sam3-port 8773 \
  --state-dir "$SAM3_SERVICE_ROOT/state"
```

服务同时提供首选的 Streamable HTTP `/mcp` 和 M5 当前使用的 legacy SSE `/sse`。
health/smoke 只证明服务与工具目录可达；至少一次真实 `segment` 推理成功，才能证明
模型资产、CUDA 和 processor 已经拉通。

## 2026-08-18 控制层复验

OpenETA commit `10a56e1cd10a6208505f8c0369d2b00a193c3331` 在上述资产上完成了
M0–M5 串行 control-only 运行。报告位于远端：

```text
/root/autodl-tmp/openeta-services/acceptance-runs/
  10a56e1-m0-m5-control-20260818/control-acceptance-report.json
```

报告为 `overall_status=passed`，scope 为
`control_only_real_sam3_no_planner_not_formal_tui`；未调用 Planner Provider，
`formal_tui_acceptance=not_run`。M5 使用文本 `red rectangular block`，SAM3 返回唯一
候选（score `0.50390625`），随后完成选择、双垫原生接触、attach、99.85 mm lift、
0.34 mm 相对漂移、open 和 detach。所有 M0–M5 case 均记录空 ROS graph、空 Gazebo
partition、释放端口和无 owned process residual。

该结果证明真实 SAM3 MCP 感知到 Gazebo 原子控制的链路和回执，不是正式 PTY/TUI
验收，也不证明通用视觉精度。正式范围和证据要求见
[M5 感知闭环](gazebo-m5-sam3-perception.md)与
[统一运行时验收](gazebo-unified-runtime-acceptance.md)。
