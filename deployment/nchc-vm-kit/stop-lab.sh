#!/usr/bin/env bash
set -euo pipefail

kit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
default_config_root=${XDG_CONFIG_HOME:-$HOME/.config}
course_env_file=${COURSE_ENV_FILE:-$default_config_root/nchc-cudaq-course/course.env}
export RUNTIME_DIR="$kit_dir/.runtime"

docker compose \
  --project-directory "$kit_dir" \
  --env-file "$course_env_file" \
  -f "$kit_dir/compose.yaml" \
  down
