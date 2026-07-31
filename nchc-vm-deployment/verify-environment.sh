#!/usr/bin/env bash
set -euo pipefail

kit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
default_config_root=${XDG_CONFIG_HOME:-$HOME/.config}
course_env_file=${COURSE_ENV_FILE:-$default_config_root/nchc-cudaq-course/course.env}

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1" >&2; exit 1; }

echo "== Host checks =="
[[ $(uname -m) == "x86_64" ]] || fail "Expected x86_64, found $(uname -m)"
pass "Architecture: x86_64"

[[ -f "$course_env_file" ]] || fail "Course env file not found: $course_env_file"
[[ $(stat -c '%a' "$course_env_file") == "600" ]] || fail "Course env file must have mode 600"
pass "Course env permissions: 600"

set -a
# shellcheck disable=SC1090
source "$course_env_file"
set +a

[[ ${RAP_NEMOTRON_3_SUPER_MODEL:-} == "NVIDIA-Nemotron-3-Super-120B-A12B" ]] ||
  fail "RAP_NEMOTRON_3_SUPER_MODEL must use the validated classroom model ID"
[[ ${RAP_NEMOTRON_3_ULTRA_MODEL:-} == "NVIDIA-Nemotron-3-Ultra-550B-A55B" ]] ||
  fail "RAP_NEMOTRON_3_ULTRA_MODEL must use the validated classroom model ID"
pass "RAP Nemotron 3 Super default and Ultra fallback model IDs"

[[ ${JUPYTER_BIND_ADDRESS:-0.0.0.0} == "0.0.0.0" ]] ||
  fail "JUPYTER_BIND_ADDRESS must be 0.0.0.0 for direct classroom access"
pass "Jupyter publish address: 0.0.0.0:${JUPYTER_PORT:-8888}"

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not installed"
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
expected_gpu_name=${COURSE_EXPECTED_GPU_NAME-H200}
if [[ -n "$expected_gpu_name" && "$gpu_name" != *"$expected_gpu_name"* ]]; then
  fail "Expected GPU matching '$expected_gpu_name', found: $gpu_name"
fi
pass "GPU: $gpu_name"

command -v docker >/dev/null 2>&1 || fail "Docker is not installed"
if ! docker version >/dev/null 2>&1; then
  if id -nG "${USER:-$(id -un)}" | tr " " "\n" | grep -qx docker; then
    fail "Docker group membership is configured, but this login session is stale. Exit and reconnect over SSH, then rerun this script."
  fi
  fail "Docker daemon is unavailable or ${USER:-$(id -un)} is not in the docker group"
fi
docker compose version >/dev/null 2>&1 || fail "Docker Compose plugin is unavailable"
pass "Docker Engine and Compose"

: "${COURSE_IMAGE:?COURSE_IMAGE is missing}"
: "${COURSE_REPO_PATH:?COURSE_REPO_PATH is missing}"

docker image inspect "$COURSE_IMAGE" >/dev/null 2>&1 || fail "Course image is not preloaded: $COURSE_IMAGE"
pass "Preloaded image: $COURSE_IMAGE"

required_files=(
  "NCHC_SOURCE_REVISION"
  "README.md"
  "AGENTS.md"
  "_intro_cudaq.ipynb"
  "_intro_Ising_Calibration.ipynb"
  "00_notebook.ipynb"
  "cudaq-doc.md"
  "helpers.py"
  "my_code.py"
  "requirements.txt"
  "data/taiwan_map_xy.json"
)
for relative_path in "${required_files[@]}"; do
  [[ -f "$COURSE_REPO_PATH/$relative_path" ]] || fail "Missing course file: $relative_path"
done
pass "Course repository files"
pass "Course source identity: $(tr '\n' ' ' < "$COURSE_REPO_PATH/NCHC_SOURCE_REVISION")"

docker run --rm \
  --volume "$COURSE_REPO_PATH:/workspace/cudaq-agentic-coding:ro" \
  --volume "$kit_dir/validate-course-notebooks.py:/tmp/validate-course-notebooks.py:ro" \
  --entrypoint python \
  "$COURSE_IMAGE" \
  /tmp/validate-course-notebooks.py
pass "Notebook JSON, Python syntax, kernel metadata, saved execution state, and error outputs"

echo
echo "== Container checks =="
docker run --rm --gpus all --entrypoint bash "$COURSE_IMAGE" -lc '
  set -euo pipefail
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  python -c "import cudaq; print(\"CUDA-Q import: PASS\")"
  nvq++ --version | head -n 1
  jupyter lab --version
  python -c "import jupyter_ai; print(\"Jupyter AI import: PASS\")"
  opencode --version
  python -c "import qiskit; print(\"Qiskit\", qiskit.__version__)"
  python /opt/nchc-cudaq-course/smoke-test-cudaq.py
'
pass "CUDA-Q CPU/GPU smoke tests"

echo
echo "All required checks passed. Start the lab with: $kit_dir/start-lab.sh"
