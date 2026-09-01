#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 DEPLOY_ROOT" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
DEPLOY_ROOT="$(realpath -m -- "$1")"
MODEL_ROOT="${OPENETA_MODEL_ROOT:-${DEPLOY_ROOT}/models}"

download_checked() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  local expected_size="$4"
  mkdir -p "$(dirname -- "${destination}")"
  if [[ ! -f "${destination}" ]]; then
    curl -fL --retry 5 --continue-at - --output "${destination}.partial" "${url}"
    [[ "$(stat -c %s "${destination}.partial")" == "${expected_size}" ]]
    echo "${expected_sha}  ${destination}.partial" | sha256sum --check --status
    mv "${destination}.partial" "${destination}"
  fi
  echo "${expected_sha}  ${destination}" | sha256sum --check
}

clone_lfs_checked() {
  local url="$1"
  local destination="$2"
  local revision="$3"
  mkdir -p "$(dirname -- "${destination}")"
  if [[ ! -e "${destination}" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "${url}" "${destination}"
    git -C "${destination}" checkout "${revision}"
  fi
  [[ -d "${destination}/.git" ]] || {
    echo "refusing incomplete model checkout: ${destination}" >&2
    exit 2
  }
  [[ "$(git -C "${destination}" rev-parse HEAD)" == "${revision}" ]] || {
    echo "model checkout revision mismatch: ${destination}" >&2
    exit 2
  }
  git -C "${destination}" lfs pull
}

sam3_root="${MODEL_ROOT}/sam3/96f3e1b404ba14f2cfac60ee6ae87c269a7b7923"
download_checked \
  "https://www.modelscope.cn/models/facebook/sam3/resolve/96f3e1b404ba14f2cfac60ee6ae87c269a7b7923/sam3.pt" \
  "${sam3_root}/sam3.pt" \
  "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e" \
  "3450062241"
download_checked \
  "https://www.modelscope.cn/models/facebook/sam3/resolve/96f3e1b404ba14f2cfac60ee6ae87c269a7b7923/config.json" \
  "${sam3_root}/config.json" \
  "4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f" \
  "25843"

anyplace_root="${MODEL_ROOT}/anyplace"
download_checked \
  "https://huggingface.co/datasets/yuchiallanzhao/anyplace/resolve/669f1b0ebcbe2ae3a72970ff31e911e8af73b2d6/anyplace_ckpts.zip" \
  "${anyplace_root}/anyplace_ckpts.zip" \
  "edb3ec855ff556d4e39287abe7db3c9390d65c69307b9f7fda8255b5ab925ea0" \
  "371478276"
anyplace_release="${anyplace_root}/release-669f1b0ebcbe2ae3a72970ff31e911e8af73b2d6"
if [[ ! -f "${anyplace_release}/anyplace_ckpts/anyplace_multitask/model.pth" ]]; then
  if unzip -Z1 "${anyplace_root}/anyplace_ckpts.zip" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
    echo "unsafe path in AnyPlace checkpoint archive" >&2
    exit 2
  fi
  mkdir -p "${anyplace_release}"
  unzip -q -n "${anyplace_root}/anyplace_ckpts.zip" -d "${anyplace_release}"
fi
echo \
  "d3d33f0a279633c25f252960a208d4b4447a756f0cff8e94be0faadc20dc5be5  ${anyplace_release}/anyplace_ckpts/anyplace_multitask/model.pth" \
  | sha256sum --check

graspgenx_root="${MODEL_ROOT}/graspgenx"
clone_lfs_checked \
  "https://huggingface.co/adithyamurali/GraspGenXModel" \
  "${graspgenx_root}/GraspGenXModel" \
  "7c834043c11a11417e31d6d5ea9355801e40a2c1"
clone_lfs_checked \
  "https://huggingface.co/datasets/adithyamurali/gripper_descriptions" \
  "${graspgenx_root}/gripper_descriptions" \
  "19a03c00d19aeaf052d0f6801f0041982d676e8a"

sha256sum --check <<EOF
8b55f31cdb8340a573b4df27b027c15cff326bd6debcb389bf631d2aaab7ac44  ${graspgenx_root}/GraspGenXModel/release/gen/epoch_736.pth
cbf3f3bdb2e4c03fca8486ed24de0e6a8a859e6bd22bce2f1434a610335abd3e  ${graspgenx_root}/GraspGenXModel/release/dis/epoch_1056.pth
098a69c968b05dc0f712b26c7043cf888290e08e1b67a1778e7bfa4825163165  ${graspgenx_root}/gripper_descriptions/gripper_descriptions/assets/x_grippers/robotiq_2f_85/config.json
39bb45ebe636d11b20eb171cae453c5fd0f5901e35dab1a536d2e8e5eb2728ef  ${graspgenx_root}/gripper_descriptions/gripper_descriptions/assets/x_grippers/robotiq_2f_85/gripper.urdf
cc5d6d867c9f77a61d1659cb37df8270aead11aeb3cd348fd38719938ad1e0d8  ${graspgenx_root}/gripper_descriptions/gripper_descriptions/assets/x_grippers/robotiq_2f_85/points.json
EOF
