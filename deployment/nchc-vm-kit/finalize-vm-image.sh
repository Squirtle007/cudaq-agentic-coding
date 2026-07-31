#!/usr/bin/env bash
set -euo pipefail

canonical_script=/opt/nchc-cudaq-course/kit/finalize-vm-image.sh
target_user=ubuntu
target_home=/home/ubuntu
kit_dir=/home/ubuntu/cudaq-course-kit
course_env_dir=$target_home/.config/nchc-cudaq-course
course_env_file=$course_env_dir/course.env
activity_env=/opt/nchc-cudaq-course/activity/activity-keys.env
course_image=nchc/cudaq-agentic-coding:2026-07-17

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  [[ -x "$canonical_script" ]] || {
    echo "Canonical finalization script is missing: $canonical_script" >&2
    exit 1
  }
  exec sudo "$canonical_script"
fi

delete_tree() {
  local path=$1
  case "$path" in
    /home/ubuntu/build-bootcamp|\
    /home/ubuntu/.anaconda|\
    /home/ubuntu/.cache|\
    /home/ubuntu/.codex|\
    /home/ubuntu/.conda|\
    /home/ubuntu/.config/nchc-cudaq-course|\
    /home/ubuntu/.config/gh|\
    /home/ubuntu/.config/opencode|\
    /home/ubuntu/.docker|\
    /home/ubuntu/.ipython|\
    /home/ubuntu/.jupyter|\
    /home/ubuntu/.npm|\
    /home/ubuntu/.nv|\
    /home/ubuntu/.claude|\
    /home/ubuntu/.local/share/jupyter|\
    /home/ubuntu/.local/share/opencode|\
    /home/ubuntu/.local/share/opentui|\
    /home/ubuntu/.local/state|\
    /home/ubuntu/manual-validation-backups|\
    /home/ubuntu/course-backups|\
    /home/ubuntu/opencode-manual-validation|\
    /home/ubuntu/cudaq-course-kit/.runtime|\
    /opt/nchc-cudaq-course/kit/.runtime|\
    /home/ubuntu/.config/opencode/skills/cudaq-gpu-opt-skill|\
    /home/ubuntu/.config/opencode/skills/cudaq-guide|\
    /home/ubuntu/.codex/skills/cudaq-gpu-opt-skill|\
    /home/ubuntu/.codex/skills/cudaq-guide|\
    /home/ubuntu/.claude/skills/cudaq-gpu-opt-skill|\
    /home/ubuntu/.claude/skills/cudaq-guide|\
    /opt/nchc-cudaq-course/kit.backup.*|\
    /opt/nchc-cudaq-course/source.backup.*|\
    /home/ubuntu/cudaq-course-kit.backup.*)
      ;;
    *)
      echo "Refusing to delete unexpected path: $path" >&2
      exit 1
      ;;
  esac

  if [[ -d "$path" ]]; then
    find "$path" -mindepth 1 -delete
    rmdir "$path"
    echo "Removed: $path"
  elif [[ -e "$path" || -L "$path" ]]; then
    rm -f -- "$path"
    echo "Removed: $path"
  fi
}

echo "Stopping the course service..."
sudo -u "$target_user" -H "$kit_dir/stop-lab.sh"
echo "Restoring the student workspace from the clean source snapshot..."
sudo -u "$target_user" -H "$kit_dir/course-reset.sh"

if [[ -f "$course_env_file" ]]; then
  "$kit_dir/prepare-activity-image-credentials.sh" \
    "$course_env_file" "$activity_env"
fi

[[ -f "$activity_env" ]] || {
  echo "Activity credential bundle is missing: $activity_env" >&2
  exit 1
}
[[ $(stat -c '%a %U:%G' "$activity_env") == "600 root:root" ]] || {
  echo "Activity credential bundle must be 600 root:root." >&2
  exit 1
}
echo "PASS  Authorized activity credential bundle: 600 root:root"

if [[ -d "$course_env_dir" ]]; then
  find "$course_env_dir" -maxdepth 1 -type f -name 'course.env*' -delete
fi
delete_tree "$course_env_dir"
delete_tree "$kit_dir/.runtime"
delete_tree "/opt/nchc-cudaq-course/kit/.runtime"
delete_tree "$target_home/opencode-manual-validation"
delete_tree "$target_home/manual-validation-backups"
delete_tree "$target_home/course-backups"

while IFS= read -r -d '' path; do
  delete_tree "$path"
done < <(
  find "$target_home" -maxdepth 1 -type d \
    -name 'cudaq-course-kit.backup.*' -print0
)
while IFS= read -r -d '' path; do
  delete_tree "$path"
done < <(
  find /opt/nchc-cudaq-course -maxdepth 1 -type d \
    \( -name 'kit.backup.*' -o -name 'source.backup.*' \) -print0
)

for skill_path in \
  "$target_home/.config/opencode/skills/cudaq-gpu-opt-skill" \
  "$target_home/.config/opencode/skills/cudaq-guide" \
  "$target_home/.codex/skills/cudaq-gpu-opt-skill" \
  "$target_home/.codex/skills/cudaq-guide" \
  "$target_home/.claude/skills/cudaq-gpu-opt-skill" \
  "$target_home/.claude/skills/cudaq-guide"; do
  delete_tree "$skill_path"
done

for disposable_user_path in \
  "$target_home/.anaconda" \
  "$target_home/.cache" \
  "$target_home/.codex" \
  "$target_home/.conda" \
  "$target_home/.config/gh" \
  "$target_home/.config/opencode" \
  "$target_home/.docker" \
  "$target_home/.ipython" \
  "$target_home/.jupyter" \
  "$target_home/.npm" \
  "$target_home/.nv" \
  "$target_home/.claude" \
  "$target_home/.local/share/jupyter" \
  "$target_home/.local/share/opencode" \
  "$target_home/.local/share/opentui" \
  "$target_home/.local/state"; do
  delete_tree "$disposable_user_path"
done

rm -f -- \
  "$target_home/manual-acceptance.log" \
  "$target_home/.bash_history" \
  "$target_home/.gitconfig" \
  "$target_home/.zsh_history" \
  "$target_home/.python_history" \
  "$target_home/.lesshst" \
  "$target_home/.viminfo" \
  "$target_home/.sudo_as_admin_successful" \
  "$target_home/.wget-hsts"

rm -f -- "$target_home/.local/bin/codex"

delete_tree "$target_home/build-bootcamp"

test ! -e "$course_env_file"
test ! -e "$kit_dir/.runtime"
test ! -e /opt/nchc-cudaq-course/kit/.runtime
test -x /opt/nchc-cudaq-course/kit/activate-student-course.sh
test -d /opt/nchc-cudaq-course/source
test -d "$target_home/cudaq-agentic-coding"
diff -qr /opt/nchc-cudaq-course/source \
  "$target_home/cudaq-agentic-coding" >/dev/null
docker image inspect "$course_image" >/dev/null
test -z "$(docker ps -q)"

if grep -RIlE \
  --exclude="*.pdf" --exclude="*.zip" \
  '(sk-[A-Za-z0-9_-]{12,}|nvapi-[A-Za-z0-9_-]{12,}|gho_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,})' \
  /opt/nchc-cudaq-course/source \
  /opt/nchc-cudaq-course/kit \
  "$target_home/cudaq-course-kit" \
  "$target_home/cudaq-agentic-coding" >/dev/null; then
  echo "FAIL  Possible credential remains outside the authorized activity bundle." >&2
  exit 1
fi
echo "PASS  No credential pattern outside the authorized activity bundle"

if docker history --no-trunc "$course_image" |
  grep -Eq '(sk-[A-Za-z0-9_-]{12,}|nvapi-[A-Za-z0-9_-]{12,}|gho_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,})'; then
  echo "FAIL  Possible credential pattern in Docker history." >&2
  exit 1
fi
echo "PASS  No credential pattern in Docker history"
echo "PASS  No running container"
echo
echo "VM image finalization complete. Do not start the Lab again."
echo "Hand the powered-off VM to NCHC for image capture."
