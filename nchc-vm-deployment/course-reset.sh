#!/usr/bin/env bash
set -euo pipefail

default_config_root=${XDG_CONFIG_HOME:-$HOME/.config}
course_env_file=${COURSE_ENV_FILE:-$default_config_root/nchc-cudaq-course/course.env}

if [[ ! -f "$course_env_file" ]]; then
  echo "Course env not found: $course_env_file" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$course_env_file"
set +a

: "${COURSE_REPO_PATH:?COURSE_REPO_PATH is missing}"
: "${COURSE_SOURCE_PATH:?COURSE_SOURCE_PATH is missing}"

case "$COURSE_REPO_PATH" in
  /home/*/cudaq-agentic-coding|/home/*/cudaq-agentic-coding-student2) ;;
  *) echo "Refusing unexpected reset target: $COURSE_REPO_PATH" >&2; exit 1 ;;
esac

[[ "$COURSE_SOURCE_PATH" == "/opt/nchc-cudaq-course/source" ]] || {
  echo "Refusing unexpected source path: $COURSE_SOURCE_PATH" >&2
  exit 1
}
[[ -f "$COURSE_SOURCE_PATH/00_notebook.ipynb" ]] || {
  echo "Clean source snapshot is incomplete: $COURSE_SOURCE_PATH" >&2
  exit 1
}

backup_root="$HOME/course-backups"
backup_path="$backup_root/cudaq-agentic-coding-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_root"

if [[ -e "$COURSE_REPO_PATH" ]]; then
  mv "$COURSE_REPO_PATH" "$backup_path"
  echo "Current work moved to: $backup_path"
fi

cp -a "$COURSE_SOURCE_PATH" "$COURSE_REPO_PATH"
echo "Course restored to: $COURSE_REPO_PATH"
echo "The backup remains recoverable under: $backup_root"
