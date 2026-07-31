#!/usr/bin/env bash
set -euo pipefail

kit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
default_config_root=${XDG_CONFIG_HOME:-$HOME/.config}
course_env_file=${1:-$default_config_root/nchc-cudaq-course/course.env}
public_host=${JUPYTER_PUBLIC_HOST:-}

if [[ -e "$course_env_file" ]]; then
  if [[ -z "$public_host" ]]; then
    public_host=$(sed -n 's/^JUPYTER_PUBLIC_HOST=//p' "$course_env_file" | tail -n 1)
  fi
  backup_file="${course_env_file}.backup.$(date +%Y%m%d-%H%M%S)"
  mv "$course_env_file" "$backup_file"
  echo "Existing course env moved to: $backup_file"
fi

mkdir -p "$(dirname "$course_env_file")"
umask 077

prompt_secret() {
  local prompt=$1
  local value
  read -r -s -p "$prompt (input hidden; Enter to skip): " value
  echo >&2
  if [[ "$value" == *$'\n'* || "$value" == *' '* ]]; then
    echo "Secrets must not contain spaces or newlines." >&2
    exit 1
  fi
  printf '%s' "$value"
}

course_repo_path=/home/ubuntu/cudaq-agentic-coding
rap_base_url=https://portal.genai.nchc.org.tw/api/v1
nemotron_super_model=NVIDIA-Nemotron-3-Super-120B-A12B
nemotron_ultra_model=NVIDIA-Nemotron-3-Ultra-550B-A55B
gemma26_model=gemma-4-26B-A4B-it
gemma31_model=gemma-4-31B-it
nim_base_url=https://integrate.api.nvidia.com/v1
nim_model=nvidia/nemotron-3-ultra-550b-a55b

echo "Using fixed NCHC classroom endpoints and model IDs."
echo "Only API keys are requested; Jupyter token is generated automatically."
echo

nemotron_key=$(prompt_secret "RAP Nemotron 3 Super/Ultra shared API key")
gemma26_key=$(prompt_secret "RAP Gemma 26B API key")
gemma31_key=$(prompt_secret "RAP Gemma 31B API key")
nim_key=$(prompt_secret "NVIDIA NIM API key")

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required to generate the Jupyter token." >&2
  exit 1
fi
jupyter_token=$(openssl rand -hex 24)

{
  printf 'COMPOSE_PROJECT_NAME=%s\n' 'nchc-cudaq-course'
  printf 'COURSE_IMAGE=%s\n' 'nchc/cudaq-agentic-coding:2026-07-17'
  printf 'COURSE_REPO_PATH=%s\n' "$course_repo_path"
  printf 'COURSE_SOURCE_PATH=%s\n' '/opt/nchc-cudaq-course/source'
  printf 'COURSE_EXPECTED_GPU_NAME=%s\n' 'H200'
  printf 'JUPYTER_PORT=%s\n' '8888'
  printf 'JUPYTER_BIND_ADDRESS=%s\n' '0.0.0.0'
  printf 'JUPYTER_PUBLIC_HOST=%s\n' "$public_host"
  printf 'JUPYTER_TOKEN=%s\n' "$jupyter_token"
  printf 'NCHC_RAP_BASE_URL=%s\n' "$rap_base_url"
  printf 'RAP_NEMOTRON_3_SUPER_MODEL=%s\n' "$nemotron_super_model"
  printf 'RAP_NEMOTRON_3_ULTRA_MODEL=%s\n' "$nemotron_ultra_model"
  printf 'RAP_NEMOTRON_3_ULTRA_API_KEY=%s\n' "$nemotron_key"
  printf 'RAP_GEMMA_26B_MODEL=%s\n' "$gemma26_model"
  printf 'RAP_GEMMA_26B_API_KEY=%s\n' "$gemma26_key"
  printf 'RAP_GEMMA_31B_MODEL=%s\n' "$gemma31_model"
  printf 'RAP_GEMMA_31B_API_KEY=%s\n' "$gemma31_key"
  printf 'NVIDIA_NIM_BASE_URL=%s\n' "$nim_base_url"
  printf 'NVIDIA_NIM_MODEL=%s\n' "$nim_model"
  printf 'NVIDIA_NIM_API_KEY=%s\n' "$nim_key"
} > "$course_env_file"

chmod 600 "$course_env_file"
unset nemotron_key gemma26_key gemma31_key nim_key jupyter_token public_host

echo
echo "Course env created with mode 600: $course_env_file"
echo "Keep it outside Git and inject it after VM deployment; do not bake it into the image."
echo "Start with: COURSE_ENV_FILE='$course_env_file' '$kit_dir/start-lab.sh'"
