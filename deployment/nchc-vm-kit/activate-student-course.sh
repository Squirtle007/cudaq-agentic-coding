#!/usr/bin/env bash
set -euo pipefail

canonical_script=/opt/nchc-cudaq-course/kit/activate-student-course.sh
activity_env=/opt/nchc-cudaq-course/activity/activity-keys.env

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  [[ -x "$canonical_script" ]] || {
    echo "Course activation script is missing from the VM image: $canonical_script" >&2
    exit 1
  }
  exec sudo "$canonical_script"
fi

target_user=${SUDO_USER:-ubuntu}
if [[ "$target_user" != ubuntu ]]; then
  echo "This classroom image expects the student account to be ubuntu, found: $target_user" >&2
  exit 1
fi

[[ -f "$activity_env" ]] || {
  echo "Activity credentials are missing from this VM image." >&2
  echo "Contact the instructor; do not enter keys into notebooks or chat." >&2
  exit 1
}
[[ $(stat -c '%a %U:%G' "$activity_env") == "600 root:root" ]] || {
  echo "Activity credential bundle must be mode 600 and owned by root:root." >&2
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

/opt/nchc-cudaq-course/kit/provision-course-env.sh "$target_user"
unset RAP_NEMOTRON_3_ULTRA_API_KEY RAP_NENOTRON_3_ULTRA_API_KEY \
  RAP_GEMMA_26B_API_KEY RAP_GEMMA_31B_API_KEY NVIDIA_NIM_API_KEY

sudo -u "$target_user" -H /home/ubuntu/cudaq-course-kit/verify-environment.sh
sudo -u "$target_user" -H /home/ubuntu/cudaq-course-kit/start-lab.sh

echo
echo "課程環境已啟用並完成驗證。"
echo "現在請從筆電開啟上方「★ 下一步 / NEXT STEP ★」框內的完整 JupyterLab 登入連結。"
echo "若直接連線不可用，可用 SSH 3322 port 建立 tunnel 作為備案。"
echo "若 NCHC RAP 當天變慢，可在 OpenCode 執行 /connect 連接 OpenCode Zen，再從 /models 選擇 opencode/nemotron-3-ultra-free；Free 模型仍需個人 Zen API key。"
echo "活動結束後共用 API key 會失效；日後請改用自己的 key。"
