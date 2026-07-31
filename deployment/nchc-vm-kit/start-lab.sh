#!/usr/bin/env bash
set -euo pipefail

kit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
default_config_root=${XDG_CONFIG_HOME:-$HOME/.config}
course_env_file=${COURSE_ENV_FILE:-$default_config_root/nchc-cudaq-course/course.env}
runtime_dir="$kit_dir/.runtime"

if [[ ! -f "$course_env_file" ]]; then
  echo "Course env not found: $course_env_file" >&2
  echo "Instructor: run $kit_dir/configure-course-env.sh" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker Engine and Docker Compose are required in the prebuilt VM." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$course_env_file"
set +a

: "${COURSE_IMAGE:?COURSE_IMAGE is missing from the course env file}"
: "${COURSE_REPO_PATH:?COURSE_REPO_PATH is missing from the course env file}"
: "${JUPYTER_TOKEN:?JUPYTER_TOKEN is missing from the course env file}"

if [[ ! -d "$COURSE_REPO_PATH" || ! -f "$COURSE_REPO_PATH/00_notebook.ipynb" ]]; then
  echo "Course repository is missing or incomplete: $COURSE_REPO_PATH" >&2
  echo "Run course-reset.sh or contact the instructor." >&2
  exit 1
fi

if ! docker image inspect "$COURSE_IMAGE" >/dev/null 2>&1; then
  echo "Prebuilt course container image is missing: $COURSE_IMAGE" >&2
  echo "Students should not build or pull it; contact the instructor." >&2
  exit 1
fi

export RUNTIME_DIR="$runtime_dir"
mkdir -p \
  "$runtime_dir/jupyter" \
  "$runtime_dir/opencode-data" \
  "$runtime_dir/opencode-cache" \
  "$runtime_dir/opencode-skills"

# The container runs as the CUDA-Q image's cudaq user (currently uid/gid 1001).
# Keep the bind-mounted runtime directories writable for OpenCode/Jupyter data.
chmod 700 "$runtime_dir"
cudaq_user=$(docker run --rm --entrypoint sh "$COURSE_IMAGE" -c \
  'printf "%s:%s\n" "$(id -u cudaq)" "$(id -g cudaq)"' 2>/dev/null || true)
course_owner=$(stat -c '%u:%g' "$COURSE_REPO_PATH")
if [[ "$cudaq_user" =~ ^[0-9]+:[0-9]+$ && "$course_owner" =~ ^[0-9]+:[0-9]+$ ]]; then
  cudaq_uid=${cudaq_user%%:*}
  course_owner_uid=${course_owner%%:*}
  docker run --rm --user 0:0 \
    --env CUDAQ_UID="$cudaq_uid" \
    --env COURSE_OWNER_UID="$course_owner_uid" \
    --volume "$COURSE_REPO_PATH:/course" \
    --entrypoint bash \
    "$COURSE_IMAGE" -lc '
      set -euo pipefail
      setfacl -R -m "u:${COURSE_OWNER_UID}:rwX,u:${CUDAQ_UID}:rwX,m::rwX" /course
      find /course -type d -exec setfacl \
        -m "d:u:${COURSE_OWNER_UID}:rwx,d:u:${CUDAQ_UID}:rwx,d:m::rwx" {} +
    '
else
  echo "Unable to determine course owner and cudaq UID; course files may not be writable." >&2
fi
if [[ "$cudaq_user" =~ ^[0-9]+:[0-9]+$ ]]; then
  docker run --rm --user 0:0 \
    --env CUDAQ_USER="$cudaq_user" \
    --volume "$runtime_dir:/runtime" \
    --entrypoint bash \
    "$COURSE_IMAGE" -lc '
      set -euo pipefail
      chown -R "$CUDAQ_USER" /runtime/jupyter /runtime/opencode-data /runtime/opencode-cache /runtime/opencode-skills
      chmod 700 /runtime/jupyter /runtime/opencode-data /runtime/opencode-cache /runtime/opencode-skills
    '
else
  chmod 777 "$runtime_dir/jupyter" "$runtime_dir/opencode-data" "$runtime_dir/opencode-cache" "$runtime_dir/opencode-skills"
fi
if [[ -e "$runtime_dir/opencode.json" && "$course_owner" =~ ^[0-9]+:[0-9]+$ ]]; then
  docker run --rm --user 0:0 \
    --volume "$runtime_dir:/runtime" \
    --entrypoint chown \
    "$COURSE_IMAGE" \
    "$course_owner" /runtime/opencode.json
fi
"$kit_dir/render-opencode-config.sh" "$course_env_file" "$runtime_dir/opencode.json"
if [[ "$cudaq_user" =~ ^[0-9]+:[0-9]+$ ]]; then
  docker run --rm --user 0:0 \
    --volume "$runtime_dir:/runtime" \
    --entrypoint chown \
    "$COURSE_IMAGE" \
    "$cudaq_user" /runtime/opencode.json
else
  chmod 644 "$runtime_dir/opencode.json"
fi

docker compose \
  --project-directory "$kit_dir" \
  --env-file "$course_env_file" \
  -f "$kit_dir/compose.yaml" \
  up -d --no-build

jupyter_port=${JUPYTER_PORT:-8888}
public_host=${JUPYTER_PUBLIC_HOST:-}
public_host_source=configured

is_ipv4() {
  local value=$1
  [[ "$value" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
  local IFS=.
  local -a octets
  read -r -a octets <<< "$value"
  [[ ${#octets[@]} -eq 4 ]] || return 1
  local octet
  for octet in "${octets[@]}"; do
    ((10#$octet >= 0 && 10#$octet <= 255)) || return 1
  done
}

if [[ -z "$public_host" ]] && command -v curl >/dev/null 2>&1; then
  ipify_ip=$(curl -4 -fsS --connect-timeout 2 --max-time 4 https://api.ipify.org 2>/dev/null |
    tr -d '[:space:]' || true)
  aws_ip=$(curl -4 -fsS --connect-timeout 2 --max-time 4 https://checkip.amazonaws.com 2>/dev/null |
    tr -d '[:space:]' || true)
  if [[ -n "$ipify_ip" && "$ipify_ip" == "$aws_ip" ]] && is_ipv4 "$ipify_ip"; then
    public_host=$ipify_ip
    public_host_source=public_consensus
  fi
fi
if [[ -z "$public_host" ]] && command -v ip >/dev/null 2>&1; then
  public_host=$(ip -4 route get 1.1.1.1 2>/dev/null |
    awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}' || true)
  [[ -n "$public_host" ]] && public_host_source=local_route
fi
if [[ -z "$public_host" ]] && command -v hostname >/dev/null 2>&1; then
  public_host=$(hostname -I 2>/dev/null |
    awk '{for (i = 1; i <= NF; i++) if ($i !~ /^127\./ && $i !~ /:/) {print $i; exit}}' || true)
  [[ -n "$public_host" ]] && public_host_source=local_address
fi
public_host=${public_host:-VM_IP}
[[ "$public_host" == VM_IP ]] && public_host_source=placeholder
encoded_token=$(jq -rn --arg token "$JUPYTER_TOKEN" '$token | @uri')
direct_url="http://${public_host}:${jupyter_port}/lab?token=${encoded_token}"
tunnel_url="http://localhost:${jupyter_port}/lab?token=${encoded_token}"

echo
echo "JupyterLab is starting on ${JUPYTER_BIND_ADDRESS:-0.0.0.0}:${jupyter_port}."
echo
if [[ -t 1 ]]; then
  banner_heading=$'\033[1;32m'
  banner_url=$'\033[1;36m'
  banner_reset=$'\033[0m'
else
  banner_heading=
  banner_url=
  banner_reset=
fi
echo "================================================================================"
printf '%b\n' "${banner_heading}★ 下一步 / NEXT STEP ★${banner_reset}"
echo
printf '%b\n' "${banner_heading}請在你的筆電瀏覽器開啟以下完整 JupyterLab 登入連結：${banner_reset}"
printf '%b\n' "${banner_heading}OPEN THIS COMPLETE LOGIN LINK FROM YOUR LAPTOP:${banner_reset}"
echo
printf '  %b%s%b\n' "$banner_url" "$direct_url" "$banner_reset"
echo
echo "請點擊連結，或從 http 開始完整複製整行；不要只複製 token。"
echo "Click it, or copy the entire line beginning with http into your laptop browser."
echo "================================================================================"
echo
echo "Jupyter token:"
echo "  $JUPYTER_TOKEN"
echo
case "$public_host_source" in
  public_consensus)
    echo "Public IP auto-detected by two HTTPS services: $public_host"
    ;;
  local_route|local_address|placeholder)
    echo "Only a local/placeholder address could be detected: $public_host"
    echo "Set JUPYTER_PUBLIC_HOST in $course_env_file if your laptop uses a different IP/DNS."
    ;;
esac
echo "Security: this link contains the Jupyter token. Do not share it or paste it into chat."
echo "Backup SSH tunnel (SSH port 3322):"
echo "  ssh -p 3322 -L ${jupyter_port}:127.0.0.1:${jupyter_port} ubuntu@${public_host}"
echo "  Then visit: $tunnel_url"
echo "The token is also stored in: $course_env_file"
