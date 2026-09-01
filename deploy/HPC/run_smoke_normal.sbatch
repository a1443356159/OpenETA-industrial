#!/usr/bin/env bash
#SBATCH --job-name=openeta-smoke-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

: "${OPENETA_SLURM_ROOT:?OPENETA_SLURM_ROOT must name the shared deployment root}"
DEPLOY_ROOT="$(realpath -m -- "${OPENETA_SLURM_ROOT}")"
IMAGE_PATH="${OPENETA_SLURM_IMAGE:-${DEPLOY_ROOT}/images/openeta-slurm-current.sif}"
WORKSPACE="${OPENETA_SLURM_WORKSPACE:-${DEPLOY_ROOT}/workspace}"
MODEL_ROOT="${OPENETA_MODEL_ROOT:-${DEPLOY_ROOT}/models}"
JOB_RUN_ROOT="${DEPLOY_ROOT}/runs/smoke-normal-${SLURM_JOB_ID}"
CONTAINER_ENTRYPOINT="/opt/openeta/src/deploy/HPC/container_smoke_normal.sh"

if [[ "${OPENETA_SLURM_DIAGNOSTIC_WORKSPACE_ENTRYPOINT:-0}" == 1 ]]; then
  # Explicitly dirty/workspace-bound diagnostics are never formal image
  # evidence. The default always executes the immutable in-image entrypoint.
  CONTAINER_ENTRYPOINT="/workspace/openeta/deploy/HPC/container_smoke_normal.sh"
  echo "diagnostic workspace entrypoint enabled; result is not formal image evidence" >&2
fi

[[ -f "${IMAGE_PATH}" ]] || { echo "missing SIF image: ${IMAGE_PATH}" >&2; exit 2; }
[[ -d "${WORKSPACE}/.git" ]] || { echo "missing workspace: ${WORKSPACE}" >&2; exit 2; }
[[ -d "${MODEL_ROOT}" ]] || { echo "missing model root: ${MODEL_ROOT}" >&2; exit 2; }
mkdir -p "${JOB_RUN_ROOT}"

allocated_cuda="${CUDA_VISIBLE_DEVICES:-0}"
container_runtime="${OPENETA_CONTAINER_RUNTIME:-}"
if [[ -z "${container_runtime}" ]]; then
  container_runtime="$(command -v apptainer || command -v singularity || true)"
fi
[[ -x "${container_runtime}" ]] || {
  echo "neither Apptainer nor Singularity is available on the allocated node" >&2
  exit 2
}

exec "${container_runtime}" exec \
  --nv \
  --contain \
  --cleanenv \
  --writable-tmpfs \
  --bind "${WORKSPACE}:/workspace/openeta" \
  --bind "${MODEL_ROOT}:/srv/openeta/models:ro" \
  --bind "${JOB_RUN_ROOT}:/srv/openeta/runs" \
  --env "CUDA_VISIBLE_DEVICES=${allocated_cuda}" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  --pwd /workspace/openeta \
  "${IMAGE_PATH}" \
  "${CONTAINER_ENTRYPOINT}" /srv/openeta/runs
