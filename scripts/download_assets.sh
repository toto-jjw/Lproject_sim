#!/usr/bin/env bash
set -euo pipefail

# Download Lproject_sim assets from Google Drive
# Assets: Textures, Terrains, USD_Assets, Ephemeris (~3GB)

FOLDER_ID="1H0jR8eA9DT5elJcx0JrdOfkLZXecnYl0"
ASSETS_DIR="$(cd "$(dirname "$0")/.." && pwd)/assets"

echo "==> Checking dependencies..."
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found." && exit 1
fi

if ! python3 -c "import gdown" &>/dev/null; then
    echo "==> Installing gdown..."
    pip install -q gdown
fi

echo "==> Downloading assets to: $ASSETS_DIR"
mkdir -p "$ASSETS_DIR"

python3 - <<EOF
import gdown, os, shutil

folder_id = "$FOLDER_ID"
assets_dir = "$ASSETS_DIR"
tmp_dir = assets_dir + "/_tmp_download"

print(f"Downloading from Google Drive folder: {folder_id}")
gdown.download_folder(id=folder_id, output=tmp_dir, quiet=False, use_cookies=False)

# Move subfolders into assets/
for item in os.listdir(tmp_dir):
    src = os.path.join(tmp_dir, item)
    dst = os.path.join(assets_dir, item)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    print(f"  Moved: {item}")

os.rmdir(tmp_dir)
print("Done.")
EOF

echo ""
echo "==> Assets downloaded successfully!"
echo "    $(du -sh "$ASSETS_DIR" | cut -f1)  $ASSETS_DIR"
