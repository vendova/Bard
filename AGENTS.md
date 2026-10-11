# AGENTS.md — BeatSync-Engine

Repository-specific notes for AI agents working in this repo.

## Environment (Linux / GitHub Codespaces, CPU-only)

- **FFmpeg is required** for every render (probe, scene analysis, encode).
  A Codespace built from the *default* image ships ffmpeg in the system
  image, but a rebuild can drop it — the symptom is
  `FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'`
  (and `'ffmpeg'`) mid-render, and a 0-scene / no-output result.
- `src/media_paths.py::resolve_ffmpeg()` resolves the binaries in order:
  `BEATSYNC_FFMPEG_PATH`/`BEATSYNC_FFPROBE_PATH` → `bin/ffmpeg/` bundle
  (gitignored) → `PATH` → best-effort `apt-get install ffmpeg`.
  Both `src/logger.py` and `src/ffmpeg_processing.py` use it, so the pair
  always matches. `bin/` is gitignored, so the portable bundle is optional.
- **Preferred Codespaces setup:** open in a devcontainer
  (`.devcontainer/devcontainer.json` installs ffmpeg + Python deps and
  forwards port 12000). Manual setup: `bash scripts/setup_codespaces.sh`.

## Run the server

```bash
source .venv/bin/activate
PORT=12000 GRADIO_SERVER_NAME=0.0.0.0 GRADIO_ANALYTICS_ENABLED=False \
  nohup python -u src/gui.py > /tmp/beatsync_gui.log 2>&1 &
```

Health: `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:12000/`

## Feature notes

- **Lyric video link** (Lyrics section): paste a download link to a
  black-background lyric clip. It is blended over the edited clips with a
  `lighten` effect and, when no audio file is uploaded, its own audio is
  extracted (`ffmpeg_processing.extract_audio_track`) and used as the music
  track. See `overlay_video_lighten(..., audio_mode='overlay'|'base')`.
- **Multi-link input** (`youtube_source.parse_links` +
  `prepare_clips_from_sources`): downloads each link in turn and spreads a
  ~20-clip budget across successful sources.

## Conventions

- Keep changes minimal; the render pipeline is CPU-only (libx264).
- Two-pass encode must keep a matching `-pix_fmt` on both passes (see the
  0-byte-output fix) — `_run_pass()` raises on a failed pass.
