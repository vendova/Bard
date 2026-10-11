#!/usr/bin/env bash
# Runs once when the Codespace/devcontainer is created.
# Ensures ffmpeg is present (the feature usually installs it, but we verify
# and fall back to apt) and installs the CPU-only Python requirements.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> ffmpeg missing — installing via apt"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg
fi

echo "==> ffmpeg: $(command -v ffmpeg || echo MISSING)"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements-render.txt

cat <<'EOF'

Setup complete. Start the app with:

  source .venv/bin/activate
  python src/gui.py

EOF
