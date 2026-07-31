#!/usr/bin/env bash
set -euo pipefail

kit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
backup_root="$HOME/manual-validation-backups"
backup_dir="$backup_root/manual-reset-$(date +%Y%m%d-%H%M%S-%N)"

install -d -m 700 "$backup_dir"

echo "Stopping JupyterLab..."
"$kit_dir/stop-lab.sh"

move_to_backup() {
  local source_path=$1
  local backup_relative=${2:-$(basename "$source_path")}
  local destination="$backup_dir/$backup_relative"
  if [[ -e "$source_path" || -L "$source_path" ]]; then
    install -d -m 700 "$(dirname "$destination")"
    mv -- "$source_path" "$destination"
    echo "Moved to backup: $source_path"
  fi
}

move_to_backup "$kit_dir/.runtime" "runtime"
move_to_backup "$HOME/opencode-manual-validation" "opencode-manual-validation"
move_to_backup "$HOME/manual-acceptance.log" "manual-acceptance.log"

# Course prompts can install either the student-authored optimization skill or
# NVIDIA's cudaq-guide globally. Move only these known course skills; never
# remove unrelated personal/system skills.
for skill_name in cudaq-gpu-opt-skill cudaq-guide; do
  move_to_backup \
    "$HOME/.config/opencode/skills/$skill_name" \
    "agent-skills/opencode/$skill_name"
  move_to_backup \
    "$HOME/.codex/skills/$skill_name" \
    "agent-skills/codex/$skill_name"
  move_to_backup \
    "$HOME/.claude/skills/$skill_name" \
    "agent-skills/claude/$skill_name"
done

echo
echo "Restoring the clean course workspace..."
"$kit_dir/course-reset.sh"

default_config_root=${XDG_CONFIG_HOME:-$HOME/.config}
course_env_file=${COURSE_ENV_FILE:-$default_config_root/nchc-cudaq-course/course.env}
set -a
# shellcheck disable=SC1090
source "$course_env_file"
set +a

reset_failed=0
if [[ -e "$kit_dir/.runtime" ]]; then
  echo "FAIL  Runtime directory still exists: $kit_dir/.runtime" >&2
  reset_failed=1
fi
if ! diff -qr "$COURSE_SOURCE_PATH" "$COURSE_REPO_PATH" >/dev/null; then
  echo "FAIL  Restored course workspace differs from the clean source snapshot." >&2
  reset_failed=1
fi
for skill_name in cudaq-gpu-opt-skill cudaq-guide; do
  for skill_path in \
    "$HOME/.config/opencode/skills/$skill_name" \
    "$HOME/.codex/skills/$skill_name" \
    "$HOME/.claude/skills/$skill_name"; do
    if [[ -e "$skill_path" || -L "$skill_path" ]]; then
      echo "FAIL  Course skill still active: $skill_path" >&2
      reset_failed=1
    fi
  done
done
if (( reset_failed != 0 )); then
  echo "Reset verification failed; backups remain available at: $backup_dir" >&2
  exit 1
fi

echo
echo "Manual validation state has been reset."
echo "PASS  Active workspace matches the clean source snapshot."
echo "PASS  Jupyter/OpenCode runtime, stats, cache, logs, and installed course skills are inactive."
echo "Runtime/log backup: $backup_dir"
echo "Course workspace backup: $HOME/course-backups"
echo "The course env, API keys, and Jupyter token were preserved."
echo
echo "Run the manual validation again with:"
echo "  cd $kit_dir"
echo "  ./verify-environment.sh 2>&1 | tee $HOME/manual-acceptance.log"
echo "  ./start-lab.sh"
