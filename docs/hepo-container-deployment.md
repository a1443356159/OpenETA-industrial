# OpenETA 的通用 Slurm 容器部署

`hepo` 是本方案的首个落地点，但 `deploy/hepo/` 中的运行脚本遵循通用 Slurm
边界，不写死用户、共享目录、partition、account、QoS、GPU 型号或容器运行时：

1. 有 Docker 能力的外部 CI 从 `Dockerfile` 构建不可变 OCI 镜像并发布到 GHCR；
2. 集群在一次 Slurm allocation 中用 Apptainer 或 Singularity 无特权地转换为 SIF；
3. ROS、Gazebo、MoveIt 和模型服务只在 Slurm 分配的 GPU 计算节点内运行；
4. 模型权重、运行证据和 workspace 位于集群共享存储，不写入镜像；
5. 站点特有的 partition、GRES、account、QoS、constraint 和 memory 只在提交时注入。

这也适用于登录节点没有 Docker/Podman、sudo 或 subordinate UID/GID 的集群。
Docker 是镜像定义与构建格式，SIF 是 Slurm 节点上的正式运行格式。

镜像使用 NVIDIA PyTorch 25.03（Ubuntu 24.04、CUDA 12.8）作为基础，并包含
四个相互隔离的环境：

| 环境 | 路径 | 用途 |
| --- | --- | --- |
| OpenETA | `/opt/openeta/venvs/openeta` | Python 3.12、ROS Jazzy ABI、验收控制链 |
| SAM3 | `/opt/openeta/venvs/sam3` | SAM3 与 MCP server |
| AnyPlace | `/opt/openeta/venvs/anyplace` | Python 3.10、Torch 1.13/cu117、放置 MCP |
| GraspGenX | `/opt/openeta/venvs/graspgenx` | NGC Torch、GraspGenX 与抓取 MCP |

模型服务仍由 `scripts/openeta_mcp_services.py` 管理并使用正式 MCP 接口；各服务
不共享其 venv 中新增的 site-packages。依赖 GPU/ROS ABI 的基础包只读复用镜像层。

## 共享目录契约

管理员或用户先选择一个所有目标计算节点都可见的绝对路径，后文记为
`SLURM_DEPLOY_ROOT`：

```text
SLURM_DEPLOY_ROOT/
├── images/       # OCI 转换后的不可变 SIF 与 current 软链接
├── models/       # 固定 revision 的权重和 gripper assets
├── runs/         # Slurm 日志、两轮验收证据、MCP 日志
└── workspace/    # 与镜像 revision 完全一致的 Git checkout
```

不要把 checkpoint、token 或 license 放进 Git 或 Docker build context。根目录的
`.dockerignore` 会排除当前已有的 AnyGrasp checkpoint/license 文件。

## 构建 OCI 镜像

`.github/workflows/hepo-container.yml` 只发布
`ghcr.io/<owner>/openeta-slurm:sha-<full-commit>`。正式候选通过不可变 tag 触发：

```bash
git tag openeta-container-<release-id> <commit>
git push origin openeta-container-<release-id>
```

workflow artifact 会记录 OCI digest 和源码 revision。部署必须使用 digest，不能只用
tag，也不能使用 `latest`。workflow 同时发布 maximal provenance；由于完整 ROS/CUDA
依赖树生成的 SPDX attestation 超过 GitHub 40 MiB artifact 限制，不附加 registry
SBOM。源码、基础镜像、第三方 revision 和 Python 依赖仍在 Dockerfile 中固定并由
OCI digest 封存。workflow 合入默认分支后也可手动触发。

## 在 Slurm allocation 中导入 SIF

通用调用如下；方括号部分由集群策略决定：

```bash
srun [--partition=...] [--account=...] --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --time=01:00:00 \
  bash deploy/hepo/import_oci_image.sh \
    ghcr.io/<owner>/openeta-slurm@sha256:<digest> \
    <SLURM_DEPLOY_ROOT>/images/openeta-slurm-<commit>.sif \
    <SLURM_DEPLOY_ROOT>/images/openeta-slurm-current.sif
```

导入脚本自动选择 `apptainer` 或 `singularity`，在节点本地临时目录完成转换，验证
`inspect` 后才原子地发布版本化 SIF 和 `current` 软链接。它拒绝可漂移的 OCI 引用、
拒绝覆盖已有 SIF，也拒绝默认在登录节点执行。大镜像层下载若被 registry 瞬时中断，
脚本默认在同一节点缓存中重试三次，并从第二次起使用 HTTP/1.1，避免重复丢弃已完成的
layer；重试次数可通过 `OPENETA_IMAGE_IMPORT_ATTEMPTS` 调整。

## 固定模型资产

在能够访问模型 registry 的节点执行：

```bash
bash deploy/hepo/fetch_models.sh <SLURM_DEPLOY_ROOT>
```

脚本只获取 smoke 所需资产并校验每个关键文件的 SHA-256：

- SAM3 ModelScope commit `96f3e1b...`；
- AnyPlace dataset revision `669f1b0...` 的 checkpoint 包，只使用 multitask checkpoint；
- GraspGenX model revision `7c83404...`；
- gripper descriptions revision `19a03c0...` 中的 `robotiq_2f_85`。

容器启动后，`prepare_assets.py` 再次验证源码 revision 和模型 hash，并建立 SAM3
离线 cache 视图。任何不匹配都作为基础设施错误终止验收。

## 提交两轮 smoke_normal

先把 `workspace` checkout 到镜像 label 中记录的精确 commit。通用提交入口为：

```bash
OPENETA_SLURM_PARTITION=<partition> \
OPENETA_SLURM_GRES=<site-gres> \
bash deploy/hepo/submit_smoke_normal.sh <SLURM_DEPLOY_ROOT>
```

可选变量包括 `OPENETA_SLURM_ACCOUNT`、`OPENETA_SLURM_QOS`、
`OPENETA_SLURM_CONSTRAINT`、`OPENETA_SLURM_MEMORY`、`OPENETA_SLURM_CPUS` 和
`OPENETA_SLURM_TIME`。默认不请求 memory TRES，适配未配置 Slurm memory accounting
的站点。默认 GRES 为 `gpu:1`；需要固定 GPU 型号时由站点覆盖。

job 会校验镜像、workspace 和模型，启动三个隔离 MCP 服务，并以
`normal + smoke_normal + fast_v3 + graspgenx` 连续运行两次。第一轮失败会立即停止并
保留诊断证据，不以第二次重跑掩盖故障。正式通过要求两份
`acceptance-report.json` 都为 `status=passed`，且 smoke profile 的 planner/VLM
调用与 token 均为零。

## hepo 实例

hepo 当前使用共享根 `/home/yyy/openeta-hepo`，运行分区 `hepnodes`，正式 GPU GRES
为 `gpu:L40:1`，并且不请求 memory TRES：

```bash
OPENETA_SLURM_PARTITION=hepnodes \
OPENETA_SLURM_GRES=gpu:L40:1 \
bash deploy/hepo/submit_smoke_normal.sh /home/yyy/openeta-hepo
```

这些值只属于 hepo 的提交实例，不进入通用 sbatch job。结果位于
`/home/yyy/openeta-hepo/runs/smoke-normal-<job-id>/`。
