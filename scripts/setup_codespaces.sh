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

Storage / size controls (intermediate clips are now bounded-quality rather
than lossless, so renders stay small and assembly stays fast):
  export BEATSYNC_PLAYBACK_CRF=20        # higher = smaller/faster clips (18-23 typical)
  export BEATSYNC_CPU_PRESET=veryfast    # clip encode speed/quality
  export BEATSYNC_CLIP_MAX_WIDTH=1920    # downscale source wider than this
  export BEATSYNC_CLIP_MAX_HEIGHT=1080
  export BEATSYNC_CLIP_MAX_MB=200        # per-clip size ceiling (0 = off)
  export BEATSYNC_MAX_OUTPUT_MB=200      # final output cap (2x input by default)

Draft workflow (fast low-quality preview, then a full-quality final):
  # Preview pass: small proxy clips + a manifest of the exact edit
  BEATSYNC_DRAFT=1 python src/gui.py     # or tick "Draft Mode" in the UI
  # Tune the edit, then rebuild the identical cut at full quality:
  python scripts/render_final_from_draft.py --latest

Draft knobs (proxy quality for the preview pass only):
  export BEATSYNC_DRAFT_CRF=30           # higher = faster/smaller preview
  export BEATSYNC_DRAFT_PRESET=ultrafast
  export BEATSYNC_DRAFT_MAX_WIDTH=960
  export BEATSYNC_DRAFT_MAX_HEIGHT=540

Lyrics + clip-diversity controls:
  export BEATSYNC_LYRICS_SCALE=1.3       # lyric size multiplier (1.0-2.5)
  export BEATSYNC_MAX_OUTPUT_SECONDS=    # optional hard cap on output length
  # By default the render adapts to the footage: if the music is longer than
  # the total source video, the music is trimmed to the footage length so no
  # clip is repeated to pad the runtime. Source moments are never reused
  # back-to-back and are spread round-robin across the whole video.

YouTube link source (paste a URL in the "YouTube Link" box):
  # Downloads the video and builds up to 20 refreshed clips. It prefers shots
  # without baked-in text overlays (crops the text band if needed) and, by
  # default, keeps only a few clips with audio and forces the rest silent so
  # no background music leaks in. Needs yt-dlp + ffmpeg (installed above).
  export BEATSYNC_YOUTUBE_KEEP_AUDIO=1        # 0 = every clip silent
  export BEATSYNC_YOUTUBE_MAX_AUDIO_FRACTION=0.25  # share of clips that keep audio
  export BEATSYNC_YOUTUBE_CLIP_SECONDS=2.6    # target seconds per clip
  # Datacenter/Codespaces IPs are often bot-blocked by YouTube. Export cookies
  # from a logged-in browser and point yt-dlp at them:
  export BEATSYNC_YT_COOKIES=/path/to/cookies.txt
  # or: export BEATSYNC_YT_COOKIES_FROM_BROWSER=chrome
EOF
