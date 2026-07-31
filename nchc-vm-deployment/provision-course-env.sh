#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "Run this provisioning script as root after a student VM is deployed." >&2
  exit 1
fi

target_user=${1:-ubuntu}
target_home=$(getent passwd "$target_user" | cut -d: -f6)
[[ -n "$target_home" && -d "$target_home" ]] || {
  echo "Target user or home directory not found: $target_user" >&2
  exit 1
}

NCHC_RAP_BASE_URL=https://portal.genai.nchc.org.tw/api/v1
RAP_NEMOTRON_3_SUPER_MODEL=NVIDIA-Nemotron-3-Super-120B-A12B
RAP_NEMOTRON_3_ULTRA_MODEL=NVIDIA-Nemotron-3-Ultra-550B-A55B
RAP_GEMMA_26B_MODEL=gemma-4-26B-A4B-it
RAP_GEMMA_31B_MODEL=gemma-4-31B-it
NVIDIA_NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_NIM_MODEL=nvidia/nemotron-3-ultra-550b-a55b

: "${RAP_GEMMA_26B_API_KEY:?RAP_GEMMA_26B_API_KEY must be injected}"
: "${RAP_GEMMA_31B_API_KEY:?RAP_GEMMA_31B_API_KEY must be injected}"

nemotron_key=${RAP_NEMOTRON_3_ULTRA_API_KEY:-${RAP_NENOTRON_3_ULTRA_API_KEY:-}}
[[ -n "$nemotron_key" ]] || {
  echo "Inject RAP_NEMOTRON_3_ULTRA_API_KEY (or the legacy NENOTRON spelling)." >&2
  exit 1
}

for secret_value in "$nemotron_key" "$RAP_GEMMA_26B_API_KEY" "$RAP_GEMMA_31B_API_KEY" "${NVIDIA_NIM_API_KEY:-}"; do
  [[ "$secret_value" != *$'\n'* && "$secret_value" != *' '* ]] || {
    echo "API keys must not contain spaces or newlines." >&2
    exit 1
  }
done

jupyter_token=${JUPYTER_TOKEN:-$(openssl rand -hex 24)}
config_dir="$target_home/.config/nchc-cudaq-course"
course_env_file="$config_dir/course.env"
temporary_env=$(mktemp /run/nchc-cudaq-course-env.XXXXXX)
trap 'rm -f "$temporary_env"' EXIT
umask 077

{
  printf 'COMPOSE_PROJECT_NAME=%s\n' 'nchc-cudaq-course'
  printf 'COURSE_IMAGE=%s\n' 'nchc/cudaq-agentic-coding:2026-07-17'
  printf 'COURSE_REPO_PATH=%s\n' "$target_home/cudaq-agentic-coding"
  printf 'COURSE_SOURCE_PATH=%s\n' '/opt/nchc-cudaq-course/source'
  printf 'COURSE_EXPECTED_GPU_NAME=%s\n' 'H200'
  printf 'JUPYTER_PORT=%s\n' '8888'
  printf 'JUPYTER_BIND_ADDRESS=%s\n' '0.0.0.0'
  printf 'JUPYTER_PUBLIC_HOST=%s\n' "${JUPYTER_PUBLIC_HOST:-}"
  printf 'JUPYTER_TOKEN=%s\n' "$jupyter_token"
  printf 'NCHC_RAP_BASE_URL=%s\n' "$NCHC_RAP_BASE_URL"
  printf 'RAP_NEMOTRON_3_SUPER_MODEL=%s\n' "$RAP_NEMOTRON_3_SUPER_MODEL"
  printf 'RAP_NEMOTRON_3_ULTRA_MODEL=%s\n' "$RAP_NEMOTRON_3_ULTRA_MODEL"
  printf 'RAP_NEMOTRON_3_ULTRA_API_KEY=%s\n' "$nemotron_key"
  printf 'RAP_NENOTRON_3_ULTRA_API_KEY=%s\n' "$nemotron_key"
  printf 'RAP_GEMMA_26B_MODEL=%s\n' "$RAP_GEMMA_26B_MODEL"
  printf 'RAP_GEMMA_26B_API_KEY=%s\n' "$RAP_GEMMA_26B_API_KEY"
  printf 'RAP_GEMMA_31B_MODEL=%s\n' "$RAP_GEMMA_31B_MODEL"
  printf 'RAP_GEMMA_31B_API_KEY=%s\n' "$RAP_GEMMA_31B_API_KEY"
  printf 'NVIDIA_NIM_BASE_URL=%s\n' "$NVIDIA_NIM_BASE_URL"
  printf 'NVIDIA_NIM_MODEL=%s\n' "$NVIDIA_NIM_MODEL"
  printf 'NVIDIA_NIM_API_KEY=%s\n' "${NVIDIA_NIM_API_KEY:-}"
} > "$temporary_env"

install -d -m 0700 -o "$target_user" -g "$target_user" "$config_dir"
install -m 0600 -o "$target_user" -g "$target_user" "$temporary_env" "$course_env_file"
unset nemotron_key jupyter_token

echo "Provisioned course env for $target_user at $course_env_file (mode 600)."
echo "No secret value was printed. Do not capture this VM back into the clean image."
