#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 OCI_DIGEST_REFERENCE TARGET_SIF [CURRENT_SYMLINK]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage
[[ -n "${SLURM_JOB_ID:-}" || "${OPENETA_ALLOW_NON_SLURM_IMPORT:-0}" == 1 ]] || {
  echo "image import must run inside a Slurm allocation" >&2
  exit 2
}

image_reference="$1"
target_sif="$(realpath -m -- "$2")"
current_link="${3:-}"
[[ "${image_reference}" == *@sha256:* ]] || {
  echo "OCI image must be pinned by sha256 digest" >&2
  exit 2
}
[[ ! -e "${target_sif}" ]] || {
  echo "refusing to overwrite existing SIF: ${target_sif}" >&2
  exit 2
}

container_runtime="${OPENETA_CONTAINER_RUNTIME:-}"
if [[ -z "${container_runtime}" ]]; then
  container_runtime="$(command -v apptainer || command -v singularity || true)"
fi
[[ -x "${container_runtime}" ]] || {
  echo "neither Apptainer nor Singularity is available" >&2
  exit 2
}

import_attempts="${OPENETA_IMAGE_IMPORT_ATTEMPTS:-3}"
retry_delay_seconds="${OPENETA_IMAGE_IMPORT_RETRY_DELAY_SECONDS:-5}"
[[ "${import_attempts}" =~ ^[1-9]$ ]] || {
  echo "OPENETA_IMAGE_IMPORT_ATTEMPTS must be an integer from 1 to 9" >&2
  exit 2
}
[[ "${retry_delay_seconds}" =~ ^[0-9]+$ ]] || {
  echo "OPENETA_IMAGE_IMPORT_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
  exit 2
}
(( retry_delay_seconds <= 60 )) || {
  echo "OPENETA_IMAGE_IMPORT_RETRY_DELAY_SECONDS must not exceed 60" >&2
  exit 2
}

scratch_parent="${SLURM_TMPDIR:-/tmp}"
scratch_dir="$(mktemp -d "${scratch_parent%/}/openeta-image-import.XXXXXX")"
partial_sif=""
cleanup() {
  if [[ -n "${partial_sif}" && -e "${partial_sif}" ]]; then
    rm -f -- "${partial_sif}"
  fi
  rm -rf -- "${scratch_dir}"
}
trap cleanup EXIT INT TERM

mkdir -p "$(dirname -- "${target_sif}")"
local_sif="${scratch_dir}/openeta.sif"
export APPTAINER_CACHEDIR="${scratch_dir}/cache"
export SINGULARITY_CACHEDIR="${scratch_dir}/cache"
build_succeeded=0
for ((attempt = 1; attempt <= import_attempts; attempt++)); do
  rm -f -- "${local_sif}"
  if (( attempt == 1 )); then
    if "${container_runtime}" build "${local_sif}" "docker://${image_reference}"; then
      build_succeeded=1
      break
    fi
  else
    # Some registries intermittently reset large HTTP/2 layer streams. Keep
    # the successful layer cache and use HTTP/1.1 for deterministic retries.
    retry_godebug="${GODEBUG:-}"
    retry_godebug="${retry_godebug:+${retry_godebug},}http2client=0"
    echo "retrying OCI import ${attempt}/${import_attempts} with HTTP/1.1" >&2
    if GODEBUG="${retry_godebug}" \
      "${container_runtime}" build "${local_sif}" "docker://${image_reference}"; then
      build_succeeded=1
      break
    fi
  fi

  if (( attempt < import_attempts )); then
    sleep_for=$((retry_delay_seconds * attempt))
    echo "OCI import attempt ${attempt}/${import_attempts} failed; retrying in ${sleep_for}s" >&2
    sleep "${sleep_for}"
  fi
done
(( build_succeeded == 1 )) || {
  echo "OCI import failed after ${import_attempts} attempts" >&2
  exit 1
}
"${container_runtime}" inspect "${local_sif}" >/dev/null

partial_sif="${target_sif}.partial.${SLURM_JOB_ID:-local}"
cp --reflink=auto -- "${local_sif}" "${partial_sif}"
mv -- "${partial_sif}" "${target_sif}"
sha256sum -- "${target_sif}"
"${container_runtime}" inspect --json "${target_sif}"

if [[ -n "${current_link}" ]]; then
  current_directory="$(realpath -m -- "$(dirname -- "${current_link}")")"
  current_link="${current_directory}/$(basename -- "${current_link}")"
  [[ "${current_link}" != "${target_sif}" ]] || {
    echo "CURRENT_SYMLINK must differ from TARGET_SIF" >&2
    exit 2
  }
  [[ ! -e "${current_link}" || -L "${current_link}" ]] || {
    echo "refusing to replace non-symlink current path: ${current_link}" >&2
    exit 2
  }
  mkdir -p "$(dirname -- "${current_link}")"
  temporary_link="${current_link}.partial.${SLURM_JOB_ID:-local}"
  ln -s -- "${target_sif}" "${temporary_link}"
  mv -T -- "${temporary_link}" "${current_link}"
fi
