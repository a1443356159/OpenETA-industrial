FROM nvcr.io/nvidia/pytorch:25.03-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG ANYPLACE_COMMIT=3049f78ad226ba0d9e54c63e2ca7ad7bbcfaa45e
ARG GRASPGENX_COMMIT=b9429097728cb1c430dd78b92edf17ba318aad03
ARG OPENETA_REVISION=unknown
ARG SAM3_COMMIT=8f0b7f4d4e7eda2ed606ebde6702c93359ad01da
ARG UV_VERSION=0.8.13

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=jazzy \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/openeta:/opt/openeta/src \
    QT_QPA_PLATFORM=offscreen \
    GZ_SIM_RENDER_ENGINE_SERVER=ogre2 \
    UV_PYTHON_INSTALL_DIR=/opt/openeta/python \
    OPENETA_CONTAINER_ROOT=/opt/openeta \
    OPENETA_IMAGE_REVISION=${OPENETA_REVISION} \
    OPENETA_MODEL_ROOT=/srv/openeta/models \
    OPENETA_RUN_ROOT=/srv/openeta/runs

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    gnupg \
    libegl1 \
    libfcl-dev \
    libgl1 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libglu1-mesa-dev \
    libglx-mesa0 \
    libgomp1 \
    libosmesa6-dev \
    libsm6 \
    libspatialindex-dev \
    libxext6 \
    libxrender1 \
    locales \
    mesa-utils \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    rsync \
    software-properties-common \
    time \
    unzip \
    xauth \
    xvfb && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions \
      python3-rosdep \
      ros-jazzy-controller-manager \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-moveit \
      ros-jazzy-pick-ik \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-ros-base \
      ros-jazzy-ros-gz \
      ros-jazzy-ros2-controllers \
      ros-jazzy-trac-ik-kinematics-plugin \
      ros-jazzy-xacro

RUN python3 -m pip install "uv==${UV_VERSION}" && \
    mkdir -p /opt/openeta/third_party /opt/openeta/venvs /opt/openeta/config

RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ac-rad/anyplace.git \
      /opt/openeta/third_party/anyplace && \
    git -C /opt/openeta/third_party/anyplace checkout "${ANYPLACE_COMMIT}" && \
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/GraspGenX.git \
      /opt/openeta/third_party/GraspGenX && \
    git -C /opt/openeta/third_party/GraspGenX checkout "${GRASPGENX_COMMIT}" && \
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/facebookresearch/sam3.git \
      /opt/openeta/third_party/sam3 && \
    git -C /opt/openeta/third_party/sam3 checkout "${SAM3_COMMIT}"

COPY deploy/hepo/requirements-anyplace-runtime.txt /tmp/openeta-requirements/
COPY deploy/hepo/requirements-graspgenx-runtime.txt /tmp/openeta-requirements/
COPY tools/requirements-anyplace.txt /tmp/openeta-requirements/
COPY tools/requirements-graspgenx.txt /tmp/openeta-requirements/
COPY tools/requirements-sam3.txt /tmp/openeta-requirements/

RUN uv python install 3.10.16 && \
    uv venv --seed --python 3.10.16 /opt/openeta/venvs/anyplace && \
    /opt/openeta/venvs/anyplace/bin/python -m pip install \
      torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
      --extra-index-url https://download.pytorch.org/whl/cu117 && \
    /opt/openeta/venvs/anyplace/bin/python -m pip install \
      torch-scatter torch-cluster \
      -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-index && \
    /opt/openeta/venvs/anyplace/bin/python -m pip install \
      -r /tmp/openeta-requirements/requirements-anyplace-runtime.txt \
      -r /tmp/openeta-requirements/requirements-anyplace.txt && \
    /opt/openeta/venvs/anyplace/bin/python -m pip install --no-deps \
      -e /opt/openeta/third_party/anyplace

RUN python3 -m venv --system-site-packages /opt/openeta/venvs/graspgenx && \
    /opt/openeta/venvs/graspgenx/bin/python -m pip install --upgrade \
      pip "setuptools>=45,<78" wheel && \
    /opt/openeta/venvs/graspgenx/bin/python -m pip install \
      -r /tmp/openeta-requirements/requirements-graspgenx-runtime.txt \
      -r /tmp/openeta-requirements/requirements-graspgenx.txt && \
    /opt/openeta/venvs/graspgenx/bin/python -m pip install --no-deps \
      -e /opt/openeta/third_party/GraspGenX

RUN python3 -m venv --system-site-packages /opt/openeta/venvs/sam3 && \
    /opt/openeta/venvs/sam3/bin/python -m pip install --upgrade \
      pip "setuptools>=68,<81" wheel && \
    /opt/openeta/venvs/sam3/bin/python -m pip install \
      -r /tmp/openeta-requirements/requirements-sam3.txt && \
    /opt/openeta/venvs/sam3/bin/python -m pip install \
      -e /opt/openeta/third_party/sam3

COPY . /opt/openeta/src

RUN python3 -m venv --system-site-packages /opt/openeta/venvs/openeta && \
    /opt/openeta/venvs/openeta/bin/python -m pip install --upgrade pip wheel && \
    /opt/openeta/venvs/openeta/bin/python -m pip install -e "/opt/openeta/src[dev]" && \
    cp \
      /opt/openeta/third_party/anyplace/anyplace/config/full_eval_cfgs/anyplace_eval/stacking/anyplace_diffusion_molmocrop_multitask.yaml \
      /opt/openeta/config/anyplace-normal.yaml && \
    sed -i \
      's#\$PATH_TO_CHECKPOINT#/srv/openeta/models/anyplace/release-669f1b0ebcbe2ae3a72970ff31e911e8af73b2d6/anyplace_ckpts/anyplace_multitask/model.pth#g' \
      /opt/openeta/config/anyplace-normal.yaml

RUN source /opt/ros/jazzy/setup.bash && \
    cd /opt/openeta/src/extensions/gazebo/ros2_ws && \
    colcon build --event-handlers console_direct+

RUN /opt/openeta/venvs/openeta/bin/python -c \
      "import gymnasium, mcp, numpy, prompt_toolkit, pytest, starlette, uvicorn" && \
    /opt/openeta/venvs/anyplace/bin/python -c \
      "import mcp, torch, torch_cluster, torch_scatter; assert torch.version.cuda == '11.7'" && \
    /opt/openeta/venvs/graspgenx/bin/python -c \
      "from packaging.version import Version; import graspgenx, torch; assert Version(torch.__version__.split('+')[0]) < Version('2.7')" && \
    /opt/openeta/venvs/sam3/bin/python -c "import sam3, torch"

WORKDIR /workspace/openeta

LABEL org.opencontainers.image.source="https://github.com/a1443356159/OpenETA-industrial" \
      org.opencontainers.image.revision="${OPENETA_REVISION}" \
      org.opencontainers.image.description="OpenETA ROS Jazzy, Gazebo, MoveIt and isolated MCP service venvs for Slurm"

CMD ["bash"]
