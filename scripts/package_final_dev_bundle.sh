#!/usr/bin/env bash
# Create a portable, credential-free source bundle for a pinned OpenETA revision.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/package_final_dev_bundle.sh [--revision REV] [--output ARCHIVE]

Creates a gzip-compressed Git source archive plus a SHA-256 sidecar. The archive
contains only files tracked at REV: no model weights, provider credentials,
runtime caches, rollout evidence, or local virtual environments are included.

Default output: dist/openeta-final-dev-<short-revision>.tar.gz
EOF
  exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
revision="HEAD"
output=""

while (( $# > 0 )); do
  case "$1" in
    --revision)
      (( $# >= 2 )) || usage
      revision="$2"
      shift 2
      ;;
    --output)
      (( $# >= 2 )) || usage
      output="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      ;;
  esac
done

git -C "${repo_root}" rev-parse --is-inside-work-tree >/dev/null
commit="$(git -C "${repo_root}" rev-parse --verify "${revision}^{commit}")"
short_commit="$(git -C "${repo_root}" rev-parse --short=12 "${commit}")"
bundle_root="openeta-final-dev-${short_commit}"

if [[ -z "${output}" ]]; then
  output="${repo_root}/dist/${bundle_root}.tar.gz"
fi
output_dir="$(dirname -- "${output}")"
output_name="$(basename -- "${output}")"
mkdir -p "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd)"
output="${output_dir}/${output_name}"
checksum="${output}.sha256"

if [[ -e "${output}" || -e "${checksum}" ]]; then
  echo "refusing to overwrite existing bundle or checksum: ${output}" >&2
  exit 2
fi

temporary="$(mktemp "${output_dir}/.${output_name}.XXXXXX")"
trap 'rm -f -- "${temporary}"' EXIT
git -C "${repo_root}" archive --format=tar --prefix="${bundle_root}/" "${commit}" \
  | gzip -n > "${temporary}"
mv -- "${temporary}" "${output}"
trap - EXIT

(
  cd -- "${output_dir}"
  sha256sum "${output_name}" > "${output_name}.sha256"
)

printf 'OPENETA_SOURCE_BUNDLE_OK revision=%s archive=%s checksum=%s\n' \
  "${commit}" "${output}" "${checksum}"
