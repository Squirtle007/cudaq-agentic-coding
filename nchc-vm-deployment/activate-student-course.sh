#!/usr/bin/env bash
set -euo pipefail

install_root=/opt/nchc-cudaq-course
canonical_kit=$install_root/kit
activity_env=$install_root/activity/activity-keys.env
script_path=$(readlink -f "${BASH_SOURCE[0]}")

# Run the exact copy selected by the student. This lets an updated script that
# replaces ~/cudaq-course-kit/activate-student-course.sh work on an older VM
# image whose root-owned canonical copy has not changed.
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  [[ -f "$script_path" ]] || {
    echo "Course activation script is missing: $script_path" >&2
    exit 1
  }
  exec sudo -- "$script_path"
fi

target_user=${SUDO_USER:-ubuntu}
target_home=$(getent passwd "$target_user" | cut -d: -f6)
target_group=$(id -gn "$target_user" 2>/dev/null || true)
[[ -n "$target_home" && -d "$target_home" && -n "$target_group" ]] || {
  echo "Student account or home directory not found: $target_user" >&2
  exit 1
}
[[ "$target_user" != root ]] || {
  echo "Run this script from a student account, not directly as root." >&2
  exit 1
}

command -v flock >/dev/null 2>&1 || {
  echo "flock is required to activate two student environments safely." >&2
  exit 1
}
exec 9>/run/nchc-cudaq-course-activate.lock
flock 9

[[ -f "$activity_env" ]] || {
  echo "Activity credentials are missing from this VM image." >&2
  echo "Contact the instructor; do not enter keys into notebooks or chat." >&2
  exit 1
}
[[ $(stat -c '%a %U:%G' "$activity_env") == "600 root:root" ]] || {
  echo "Activity credential bundle must be mode 600 and owned by root:root." >&2
  exit 1
}
[[ -d "$install_root/source" && -d "$canonical_kit" ]] || {
  echo "The course source or VM kit is missing under $install_root." >&2
  exit 1
}

cat <<'NOTICE'
============================================================
NCHC 課程活動 API key 啟用通知

這些共用 key 只供本次課程使用，活動結束後會失效並撤銷。
未來自行使用本教材時，請向 NCHC RAP／NVIDIA 申請自己的 key，
再執行 ~/cudaq-course-kit/configure-course-env.sh 進行設定。
請勿將活動 key、course.env 或 Jupyter token 貼到 Notebook、聊天或 GitHub。
============================================================
NOTICE

set -a
# Root-owned file created by prepare-activity-image-credentials.sh.
# shellcheck disable=SC1090
source "$activity_env"
set +a

nemotron_key=${RAP_NEMOTRON_3_ULTRA_API_KEY:-${RAP_NENOTRON_3_ULTRA_API_KEY:-}}
[[ -n "$nemotron_key" ]] || {
  echo "The activity credential bundle is missing the RAP Nemotron key." >&2
  exit 1
}
: "${RAP_GEMMA_26B_API_KEY:?The activity credential bundle is missing the RAP Gemma 26B key}"
: "${RAP_GEMMA_31B_API_KEY:?The activity credential bundle is missing the RAP Gemma 31B key}"
for secret_value in \
  "$nemotron_key" \
  "$RAP_GEMMA_26B_API_KEY" \
  "$RAP_GEMMA_31B_API_KEY" \
  "${NVIDIA_NIM_API_KEY:-}"; do
  [[ "$secret_value" != *$'\n'* && "$secret_value" != *' '* ]] || {
    echo "API keys must not contain spaces or newlines." >&2
    exit 1
  }
done

project_is_running() {
  local project_name=$1
  docker ps \
    --filter "label=com.docker.compose.project=$project_name" \
    --filter "label=com.docker.compose.service=lab" \
    --format '{{.ID}}' 2>/dev/null | grep -q .
}

primary_project=nchc-cudaq-course
primary_env=$target_home/.config/nchc-cudaq-course/course.env
secondary_project=nchc-cudaq-course-student2
secondary_env=$target_home/.config/nchc-cudaq-course-student2/course.env

if [[ -f "$secondary_env" ]]; then
  instance_number=2
elif project_is_running "$secondary_project"; then
  echo "A second JupyterLab container exists, but its course env is missing: $secondary_env" >&2
  echo "Ask the instructor to recover the second student's course env; a new token was not generated." >&2
  exit 1
elif [[ -f "$primary_env" ]] && project_is_running "$primary_project"; then
  instance_number=2
else
  instance_number=1
fi

if [[ $instance_number -eq 1 ]]; then
  instance_label="學員 1 / Student 1"
  project_name=$primary_project
  jupyter_port=8888
  course_env_file=$primary_env
  course_repo=$target_home/cudaq-agentic-coding
  course_kit=$target_home/cudaq-course-kit
else
  instance_label="學員 2 / Student 2"
  project_name=$secondary_project
  jupyter_port=8889
  course_env_file=$secondary_env
  course_repo=$target_home/cudaq-agentic-coding-student2
  course_kit=$target_home/cudaq-course-kit-student2
fi

instance_was_running=0
if project_is_running "$project_name"; then
  instance_was_running=1
fi

copy_course_directory() {
  local source_dir=$1
  local destination_dir=$2
  local expected_file=$3

  if [[ -e "$destination_dir" ]]; then
    [[ -d "$destination_dir" && -f "$destination_dir/$expected_file" ]] || {
      echo "Refusing to overwrite an incomplete course directory: $destination_dir" >&2
      exit 1
    }
    return
  fi

  cp -a -- "$source_dir" "$destination_dir"
  chown -R "$target_user:$target_group" "$destination_dir"
}

copy_course_directory "$install_root/source" "$course_repo" 00_notebook.ipynb
copy_course_directory "$canonical_kit" "$course_kit" compose.yaml

# Keep the downloaded activation script in the second kit as well. The source
# and destination can be identical on a later Student 2 rerun.
if [[ "$script_path" != "$course_kit/activate-student-course.sh" ]]; then
  install -m 0755 -o "$target_user" -g "$target_group" \
    "$script_path" "$course_kit/activate-student-course.sh"
fi
find "$course_kit" -maxdepth 1 -type f -name '*.sh' -exec chmod 0755 {} +

install_instance_wrapper() {
  local helper_name=$1
  local helper_path=$course_kit/$helper_name
  local base_name=.nchc-course-base-$helper_name
  local base_path=$course_kit/$base_name

  [[ -f "$helper_path" || -f "$base_path" ]] || {
    echo "Course helper is missing: $helper_path" >&2
    exit 1
  }
  if [[ ! -f "$base_path" ]]; then
    mv -- "$helper_path" "$base_path"
  fi
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -euo pipefail'
    printf '%s\n' 'kit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)'
    printf '%s\n' 'IFS= read -r COURSE_ENV_FILE < "$kit_dir/.course-env-file"'
    printf '%s\n' 'export COURSE_ENV_FILE'
    printf 'exec "$kit_dir/%s" "$@"\n' "$base_name"
  } > "$helper_path"
  chown "$target_user:$target_group" "$helper_path" "$base_path"
  chmod 0755 "$helper_path" "$base_path"
}

if [[ $instance_number -eq 2 ]]; then
  printf '%s\n' "$course_env_file" > "$course_kit/.course-env-file"
  chown "$target_user:$target_group" "$course_kit/.course-env-file"
  chmod 0644 "$course_kit/.course-env-file"
  for helper_name in \
    start-lab.sh \
    stop-lab.sh \
    lab-logs.sh \
    verify-environment.sh \
    course-reset.sh \
    reset-manual-validation.sh; do
    install_instance_wrapper "$helper_name"
  done

  # Older VM images accepted only the unsuffixed workspace in course-reset.sh.
  # Extend the preserved base helper in place so replacing this activation
  # script alone also makes Student 2 resets safe and usable.
  secondary_reset_base=$course_kit/.nchc-course-base-course-reset.sh
  if grep -Fq '  /home/*/cudaq-agentic-coding) ;;' "$secondary_reset_base"; then
    sed -i \
      's@  /home/\*/cudaq-agentic-coding) ;;@  /home/*/cudaq-agentic-coding|/home/*/cudaq-agentic-coding-student2) ;;@' \
      "$secondary_reset_base"
  fi
fi

new_environment=0
if [[ -f "$course_env_file" ]]; then
  expected_owner=$target_user:$target_group
  [[ $(stat -c '%a %U:%G' "$course_env_file") == "600 $expected_owner" ]] || {
    echo "Course env must be mode 600 and owned by $expected_owner: $course_env_file" >&2
    exit 1
  }
  for expected_setting in \
    "COMPOSE_PROJECT_NAME=$project_name" \
    "COURSE_REPO_PATH=$course_repo" \
    "COURSE_SOURCE_PATH=$install_root/source" \
    "JUPYTER_PORT=$jupyter_port" \
    'JUPYTER_BIND_ADDRESS=0.0.0.0'; do
    grep -Fxq "$expected_setting" "$course_env_file" || {
      setting_name=${expected_setting%%=*}
      echo "Existing course env has an unsafe or unexpected $setting_name: $course_env_file" >&2
      exit 1
    }
  done
  grep -Eq '^JUPYTER_TOKEN=.+$' "$course_env_file" || {
    echo "Existing course env has no Jupyter token: $course_env_file" >&2
    exit 1
  }
else
  config_dir=$(dirname "$course_env_file")
  temporary_env=$(mktemp /run/nchc-cudaq-course-env.XXXXXX)
  trap 'rm -f "$temporary_env"' EXIT
  umask 077
  jupyter_token=$(openssl rand -hex 24)
  {
    printf 'COMPOSE_PROJECT_NAME=%s\n' "$project_name"
    printf 'COURSE_IMAGE=%s\n' 'nchc/cudaq-agentic-coding:2026-07-17'
    printf 'COURSE_REPO_PATH=%s\n' "$course_repo"
    printf 'COURSE_SOURCE_PATH=%s\n' "$install_root/source"
    printf 'COURSE_EXPECTED_GPU_NAME=%s\n' 'H200'
    printf 'JUPYTER_PORT=%s\n' "$jupyter_port"
    printf 'JUPYTER_BIND_ADDRESS=%s\n' '0.0.0.0'
    printf 'JUPYTER_PUBLIC_HOST=%s\n' "${JUPYTER_PUBLIC_HOST:-}"
    printf 'JUPYTER_TOKEN=%s\n' "$jupyter_token"
    printf 'NCHC_RAP_BASE_URL=%s\n' 'https://portal.genai.nchc.org.tw/api/v1'
    printf 'RAP_NEMOTRON_3_SUPER_MODEL=%s\n' 'NVIDIA-Nemotron-3-Super-120B-A12B'
    printf 'RAP_NEMOTRON_3_ULTRA_MODEL=%s\n' 'NVIDIA-Nemotron-3-Ultra-550B-A55B'
    printf 'RAP_NEMOTRON_3_ULTRA_API_KEY=%s\n' "$nemotron_key"
    printf 'RAP_NENOTRON_3_ULTRA_API_KEY=%s\n' "$nemotron_key"
    printf 'RAP_GEMMA_26B_MODEL=%s\n' 'gemma-4-26B-A4B-it'
    printf 'RAP_GEMMA_26B_API_KEY=%s\n' "$RAP_GEMMA_26B_API_KEY"
    printf 'RAP_GEMMA_31B_MODEL=%s\n' 'gemma-4-31B-it'
    printf 'RAP_GEMMA_31B_API_KEY=%s\n' "$RAP_GEMMA_31B_API_KEY"
    printf 'NVIDIA_NIM_BASE_URL=%s\n' 'https://integrate.api.nvidia.com/v1'
    printf 'NVIDIA_NIM_MODEL=%s\n' 'nvidia/nemotron-3-ultra-550b-a55b'
    printf 'NVIDIA_NIM_API_KEY=%s\n' "${NVIDIA_NIM_API_KEY:-}"
  } > "$temporary_env"
  install -d -m 0700 -o "$target_user" -g "$target_group" "$config_dir"
  install -m 0600 -o "$target_user" -g "$target_group" \
    "$temporary_env" "$course_env_file"
  unset jupyter_token
  new_environment=1
  echo "Provisioned $instance_label course env at $course_env_file (mode 600)."
  echo "No secret value was printed."
fi

unset nemotron_key secret_value RAP_NEMOTRON_3_ULTRA_API_KEY \
  RAP_NENOTRON_3_ULTRA_API_KEY RAP_GEMMA_26B_API_KEY \
  RAP_GEMMA_31B_API_KEY NVIDIA_NIM_API_KEY

echo
echo "Selected: $instance_label — isolated repo, runtime, Compose project, port, and token."
echo "Host workspace: $course_repo"
echo "Jupyter port: $jupyter_port"

if [[ $new_environment -eq 1 || $instance_was_running -eq 0 ]]; then
  sudo -u "$target_user" -H "$course_kit/verify-environment.sh"
else
  echo "Running $instance_label environment found; keeping its current Jupyter token."
fi
sudo -u "$target_user" -H "$course_kit/start-lab.sh"

echo
echo "$instance_label 課程環境已啟用。"
echo "請從上方「★ 下一步 / NEXT STEP ★」框開啟包含 token 的完整 JupyterLab 連結。"
echo "學員 1 使用 8888；學員 2 使用 8889，兩者的教材、runtime、Compose project 與 token 互相隔離。"
echo "兩個 container 仍共用同一張 GPU，請避免同時執行長時間或高記憶體 Notebook。"
echo "若 NCHC RAP 當天變慢，可在 OpenCode 執行 /connect 連接 OpenCode Zen，再從 /models 選擇 opencode/nemotron-3-ultra-free；Free 模型仍需個人 Zen API key。"
echo "活動結束後共用 API key 會失效；日後請改用自己的 key。"
