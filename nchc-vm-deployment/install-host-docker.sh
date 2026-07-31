#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "Run this script as the normal Ubuntu user; it invokes sudo when needed." >&2
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  echo "Unsupported system: /etc/os-release is missing." >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || { echo "This guide supports Ubuntu only." >&2; exit 1; }
[[ "${VERSION_ID:-}" == "22.04" || "${VERSION_ID:-}" == "24.04" ]] || {
  echo "Expected Ubuntu 22.04 or 24.04; found ${VERSION_ID:-unknown}." >&2
  exit 1
}
[[ $(uname -m) == "x86_64" ]] || { echo "Expected x86_64." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "NVIDIA driver must be installed and nvidia-smi must work before this script." >&2
  exit 1
}

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg patch unzip

if ! command -v docker >/dev/null 2>&1; then
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc

  docker_codename=${UBUNTU_CODENAME:-$VERSION_CODENAME}
  docker_arch=$(dpkg --print-architecture)
  docker_source="deb [arch=${docker_arch} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${docker_codename} stable"
  printf '%s\n' "$docker_source" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER"

echo
echo "Host runtime installed. Log out and back in before running Docker without sudo."
echo "After re-login, validate with the commands in instructor-image-guide.zh-TW.md."
