#!/usr/bin/env bash
set -euo pipefail

usage() {
  # Site policy is injected by this HPC wrapper, never by the Ubuntu image.
  cat >&2 <<'EOF'
usage: submit_smoke_normal.sh SLURM_DEPLOY_ROOT

Optional environment:
  OPENETA_SLURM_PARTITION, OPENETA_SLURM_ACCOUNT, OPENETA_SLURM_QOS
  OPENETA_SLURM_GRES (default gpu:1), OPENETA_SLURM_CONSTRAINT
  OPENETA_SLURM_CPUS (default 12), OPENETA_SLURM_TIME (default 02:00:00)
  OPENETA_SLURM_MEMORY (unset by default), OPENETA_SLURM_WORKSPACE
  OPENETA_SLURM_IMAGE, OPENETA_MODEL_ROOT, OPENETA_CONTAINER_RUNTIME
EOF
  exit 2
}

[[ $# -eq 1 ]] || usage
command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is unavailable" >&2
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(realpath -m -- "$1")"
[[ -d "${DEPLOY_ROOT}" ]] || {
  echo "deployment root does not exist: ${DEPLOY_ROOT}" >&2
  exit 2
}
mkdir -p "${DEPLOY_ROOT}/runs"

cpus="${OPENETA_SLURM_CPUS:-12}"
wall_time="${OPENETA_SLURM_TIME:-02:00:00}"
gres="${OPENETA_SLURM_GRES:-gpu:1}"
[[ "${cpus}" =~ ^[1-9][0-9]*$ ]] || {
  echo "OPENETA_SLURM_CPUS must be a positive integer" >&2
  exit 2
}
[[ -n "${wall_time}" && -n "${gres}" ]] || {
  echo "OPENETA_SLURM_TIME and OPENETA_SLURM_GRES must be non-empty" >&2
  exit 2
}

sbatch_args=(
  --parsable
  --nodes=1
  --ntasks=1
  "--cpus-per-task=${cpus}"
  "--gres=${gres}"
  "--time=${wall_time}"
  "--output=${DEPLOY_ROOT}/runs/slurm-%j.out"
  "--export=ALL,OPENETA_SLURM_ROOT=${DEPLOY_ROOT}"
)

for pair in \
  "partition:OPENETA_SLURM_PARTITION" \
  "account:OPENETA_SLURM_ACCOUNT" \
  "qos:OPENETA_SLURM_QOS" \
  "constraint:OPENETA_SLURM_CONSTRAINT" \
  "mem:OPENETA_SLURM_MEMORY"; do
  option="${pair%%:*}"
  variable="${pair#*:}"
  value="${!variable:-}"
  if [[ -n "${value}" ]]; then
    sbatch_args+=("--${option}=${value}")
  fi
done

exec sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_smoke_normal.sbatch"
