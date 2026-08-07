#!/usr/bin/env bash
set -euo pipefail

activation_script=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/activate-student-course.sh
fixture=$(mktemp -d /tmp/nchc-activate-test.XXXXXX)
cleanup() {
  case "$fixture" in
    /tmp/nchc-activate-test.*) rm -rf -- "$fixture" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture" >&2 ;;
  esac
}
trap cleanup EXIT

mkdir -p \
  "$fixture/bin" \
  "$fixture/home/student" \
  "$fixture/opt/nchc-cudaq-course/activity" \
  "$fixture/opt/nchc-cudaq-course/kit" \
  "$fixture/opt/nchc-cudaq-course/source" \
  "$fixture/run"
cp "$activation_script" "$fixture/activate-student-course.sh"
chmod 0755 "$fixture/activate-student-course.sh"
: > "$fixture/opt/nchc-cudaq-course/source/00_notebook.ipynb"
: > "$fixture/opt/nchc-cudaq-course/kit/compose.yaml"

cat > "$fixture/opt/nchc-cudaq-course/activity/activity-keys.env" <<'EOF'
RAP_NEMOTRON_3_ULTRA_API_KEY=test-nemotron-key
RAP_GEMMA_26B_API_KEY=test-gemma-26b-key
RAP_GEMMA_31B_API_KEY=test-gemma-31b-key
NVIDIA_NIM_API_KEY=
EOF
chmod 0600 "$fixture/opt/nchc-cudaq-course/activity/activity-keys.env"

cat > "$fixture/opt/nchc-cudaq-course/kit/verify-environment.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
env_file=${COURSE_ENV_FILE:-$HOME/.config/nchc-cudaq-course/course.env}
source "$env_file"
test -d "$COURSE_REPO_PATH"
printf 'verify:%s\n' "$COMPOSE_PROJECT_NAME" >> /run/verification.log
EOF

cat > "$fixture/opt/nchc-cudaq-course/kit/start-lab.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
env_file=${COURSE_ENV_FILE:-$HOME/.config/nchc-cudaq-course/course.env}
source "$env_file"
touch "/run/${COMPOSE_PROJECT_NAME}.running"
printf 'start:%s:%s\n' "$COMPOSE_PROJECT_NAME" "$JUPYTER_PORT" >> /run/start.log
printf 'http://VM_IP:%s/lab?token=%s\n' "$JUPYTER_PORT" "$JUPYTER_TOKEN"
EOF

cat > "$fixture/opt/nchc-cudaq-course/kit/stop-lab.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat > "$fixture/opt/nchc-cudaq-course/kit/lab-logs.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

# This intentionally models the older image's Student 1-only reset guard. The
# activation script must patch the preserved Student 2 helper at runtime.
cat > "$fixture/opt/nchc-cudaq-course/kit/course-reset.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
env_file=${COURSE_ENV_FILE:-$HOME/.config/nchc-cudaq-course/course.env}
source "$env_file"
case "$COURSE_REPO_PATH" in
  /home/*/cudaq-agentic-coding) ;;
  *) exit 1 ;;
esac
EOF

cat > "$fixture/opt/nchc-cudaq-course/kit/reset-manual-validation.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod 0755 "$fixture/opt/nchc-cudaq-course/kit/"*.sh

cat > "$fixture/bin/getent" <<'EOF'
#!/usr/bin/env bash
if [[ ${1:-} == passwd && ${2:-} == student ]]; then
  echo 'student:x:1000:1000:Student:/home/student:/bin/bash'
  exit 0
fi
exec /usr/bin/getent "$@"
EOF

cat > "$fixture/bin/id" <<'EOF'
#!/usr/bin/env bash
if [[ ${1:-} == -gn && ${2:-} == student ]]; then
  echo student
  exit 0
fi
exec /usr/bin/id "$@"
EOF

cat > "$fixture/bin/chown" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat > "$fixture/bin/install" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
args=()
while (($#)); do
  case "$1" in
    -o|-g) shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
exec /usr/bin/install "${args[@]}"
EOF

cat > "$fixture/bin/stat" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
last_argument=${!#}
case "$last_argument" in
  /opt/nchc-cudaq-course/activity/activity-keys.env)
    echo '600 root:root'
    ;;
  /home/student/.config/*/course.env)
    echo '600 student:student'
    ;;
  *)
    exec /usr/bin/stat "$@"
    ;;
esac
EOF

cat > "$fixture/bin/sudo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == -u ]]; then shift 2; fi
if [[ ${1:-} == -H ]]; then shift; fi
export HOME=/home/student USER=student
exec "$@"
EOF

cat > "$fixture/bin/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} == ps ]] || exit 1
project=
while (($#)); do
  if [[ $1 == --filter && ${2:-} == label=com.docker.compose.project=* ]]; then
    project=${2#label=com.docker.compose.project=}
  fi
  shift
done
[[ -n "$project" && -f "/run/${project}.running" ]] && echo mock-container-id
EOF
chmod 0755 "$fixture/bin/"*

cat > "$fixture/run-test.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

mount --bind "$FIXTURE/opt" /opt
mount --bind "$FIXTURE/home" /home
mount --bind "$FIXTURE/run" /run
export PATH="$FIXTURE/bin:/usr/bin:/bin"
export HOME=/home/student SUDO_USER=student USER=student

"$FIXTURE/activate-student-course.sh" > /run/first.out
"$FIXTURE/activate-student-course.sh" > /run/second.out
"$FIXTURE/activate-student-course.sh" > /run/third.out
rm -f /run/nchc-cudaq-course-student2.running
"$FIXTURE/activate-student-course.sh" > /run/fourth.out

primary_env=/home/student/.config/nchc-cudaq-course/course.env
secondary_env=/home/student/.config/nchc-cudaq-course-student2/course.env
source "$primary_env"
primary_token=$JUPYTER_TOKEN
test "$COMPOSE_PROJECT_NAME" = nchc-cudaq-course
test "$JUPYTER_PORT" = 8888
test "$COURSE_REPO_PATH" = /home/student/cudaq-agentic-coding

source "$secondary_env"
secondary_token=$JUPYTER_TOKEN
test "$COMPOSE_PROJECT_NAME" = nchc-cudaq-course-student2
test "$JUPYTER_PORT" = 8889
test "$COURSE_REPO_PATH" = /home/student/cudaq-agentic-coding-student2
test "$primary_token" != "$secondary_token"

test -d /home/student/cudaq-agentic-coding
test -d /home/student/cudaq-agentic-coding-student2
test -d /home/student/cudaq-course-kit
test -d /home/student/cudaq-course-kit-student2
test "$(wc -l < /run/verification.log)" -eq 3
test "$(wc -l < /run/start.log)" -eq 4
grep -Fq "http://VM_IP:8889/lab?token=$secondary_token" /run/third.out
grep -Fq 'Running 學員 2 / Student 2 environment found; keeping its current Jupyter token.' /run/third.out
grep -Fq "http://VM_IP:8889/lab?token=$secondary_token" /run/fourth.out
grep -Fq '/home/*/cudaq-agentic-coding-student2' \
  /home/student/cudaq-course-kit-student2/.nchc-course-base-course-reset.sh

# Direct Student 2 helper use must select the secondary env without an export.
/home/student/cudaq-course-kit-student2/start-lab.sh > /run/direct-secondary.out
grep -Fq "http://VM_IP:8889/lab?token=$secondary_token" /run/direct-secondary.out
EOF
chmod 0755 "$fixture/run-test.sh"

if ! FIXTURE="$fixture" unshare -Ur -m "$fixture/run-test.sh"; then
  echo "FAIL: isolated activation integration test" >&2
  exit 1
fi

echo "PASS: first activation created Student 1 on 8888"
echo "PASS: second activation created isolated Student 2 on 8889"
echo "PASS: repeated activation preserved Student 2 token and printed its URL"
echo "PASS: stopped Student 2 was re-verified and restarted with the same token"
echo "PASS: Student 2 helper wrapper selected its own course env"
