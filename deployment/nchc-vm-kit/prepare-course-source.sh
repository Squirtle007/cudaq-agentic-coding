#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/cudaq-agentic-coding-main.zip" >&2
  exit 2
fi

archive_path=$1
kit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
install_root=/opt/nchc-cudaq-course
source_path=$install_root/source
kit_path=$install_root/kit
source_revision=${COURSE_SOURCE_REVISION:-unknown}

[[ -f "$archive_path" ]] || { echo "Archive not found: $archive_path" >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "unzip is required." >&2; exit 1; }
command -v patch >/dev/null 2>&1 || { echo "patch is required." >&2; exit 1; }

staging_dir=$(mktemp -d /tmp/nchc-cudaq-course.XXXXXX)
cleanup_staging() {
  case "$staging_dir" in
    /tmp/nchc-cudaq-course.*) rm -rf "$staging_dir" ;;
    *) echo "Refusing to remove unexpected staging path: $staging_dir" >&2 ;;
  esac
}
trap cleanup_staging EXIT
unzip -q "$archive_path" -d "$staging_dir"

repo_root=$(find "$staging_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)
[[ -n "$repo_root" && -f "$repo_root/00_notebook.ipynb" && -f "$repo_root/helpers.py" ]] || {
  echo "Archive does not contain the expected course repository." >&2
  exit 1
}

patch --dry-run -d "$repo_root" -p1 < "$kit_dir/course-repo.patch" >/dev/null
patch -d "$repo_root" -p1 < "$kit_dir/course-repo.patch"

# Upstream OpenCode setup examples may contain key-shaped placeholders.
# Replace only those example patterns with a non-secret sentinel so VM image
# credential audits do not flag them and students cannot mistake them for keys.
for example_file in \
  "$repo_root/opencode-setup/README.md" \
  "$repo_root/opencode-setup/opencode.json"; do
  if [[ -f "$example_file" ]]; then
    sed -E -i \
      's/(sk-|nvapi-)[A-Za-z0-9_-]{12,}/YOUR_NVIDIA_API_KEY/g' \
      "$example_file"
  fi
done

archive_sha256=$(sha256sum "$archive_path" | awk '{print $1}')
{
  printf 'upstream_commit=%s\n' "$source_revision"
  printf 'archive_sha256=%s\n' "$archive_sha256"
} > "$repo_root/NCHC_SOURCE_REVISION"

timestamp=$(date +%Y%m%d-%H%M%S)
if sudo test -e "$source_path"; then
  sudo mv "$source_path" "$install_root/source.backup.$timestamp"
fi
if sudo test -e "$kit_path"; then
  sudo mv "$kit_path" "$install_root/kit.backup.$timestamp"
fi

sudo install -d -m 0755 "$install_root"
sudo cp -a "$repo_root" "$source_path"
sudo cp -a "$kit_dir" "$kit_path"
# Never promote a running Lab runtime into the root-owned formal kit snapshot.
if sudo test -d "$kit_path/.runtime"; then
  sudo find "$kit_path/.runtime" -mindepth 1 -delete
  sudo rmdir "$kit_path/.runtime"
fi
sudo chown -R root:root "$source_path" "$kit_path"
sudo chmod -R a+rX "$source_path" "$kit_path"

echo "Installed clean course source: $source_path"
echo "Installed VM kit: $kit_path"
echo "Course source revision: $source_revision"
echo "Course archive SHA-256: $archive_sha256"
echo "Existing installations, if any, were moved to timestamped backups under $install_root."
