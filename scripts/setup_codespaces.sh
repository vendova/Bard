#!/usr/bin/env bash
# BeatSync Engine - Codespaces / Linux CPU setup.
#
# Installs the CPU-only Python deps, the llama.cpp Linux backend, the
# Qwen3-VL GGUF model, and ffmpeg so Qwen semantic tagging works without
# a GPU. Re-run this after a Codespace rebuild (bin/ is gitignored).
#
# Usage:  bash scripts/setup_codespaces.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# llama.cpp release used for the Linux CPU backend. Bump as needed.
LLAMA_BUILD="${LLAMA_BUILD:-b11541}"
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz"

MODEL_DIR="$ROOT_DIR/bin/models"
LLAMA_DIR="$ROOT_DIR/bin/llama-bin-linux"
MODEL_FILE="$MODEL_DIR/Qwen3VL-2B-Instruct-Q8_0.gguf"
MMPROJ_FILE="$MODEL_DIR/mmproj-Qwen3VL-2B-Instruct-F16.gguf"

echo "==> System packages (ffmpeg)"
if ! command -v ffmpeg >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends ffmpeg
fi

echo "==> Python venv + CPU-only requirements"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements-render.txt

echo "==> llama.cpp Linux backend ($LLAMA_BUILD)"
mkdir -p "$LLAMA_DIR"
if [ ! -x "$LLAMA_DIR/llama-server" ]; then
  tmp_tar="$(mktemp -d)/llama.tar.gz"
  curl -L --fail -o "$tmp_tar" "$LLAMA_URL"
  tmp_extract="$(mktemp -d)"
  tar xzf "$tmp_tar" -C "$tmp_extract"
  # Release extracts to a versioned folder; copy its contents into bin/llama-bin-linux.
  cp -a "$tmp_extract"/*/. "$LLAMA_DIR/"
  chmod +x "$LLAMA_DIR"/llama-server "$LLAMA_DIR"/llama-cli "$LLAMA_DIR"/llama-mtmd-cli
fi

echo "==> Qwen3-VL GGUF models (~2.6 GB)"
mkdir -p "$MODEL_DIR"
[ -f "$MODEL_FILE" ] || curl -L --fail -o "$MODEL_FILE" \
  "https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3-VL-2B-Instruct-Q8_0.gguf"
[ -f "$MMPROJ_FILE" ] || curl -L --fail -o "$MMPROJ_FILE" \
  "https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-F16.gguf"

cat <<'EOF'

Setup complete. Start the app with:

  source .venv/bin/activate
  python src/gui.py

Qwen3-VL semantic tagging is enabled automatically once bin/llama-bin-linux
and bin/models are present. On a low-RAM Codespace, cap Qwen memory first:

  export BEATSYNC_QWEN_LLAMA_SLOTS=2
  export BEATSYNC_QWEN_LLAMA_CTX=4096

The UI defaults to CPU H.264 (no NVENC GPU in Codespaces).
EOF
