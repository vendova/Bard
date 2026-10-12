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
- **Cinematic looks** (`src/look_presets.py`): named colour grades compiled to
  FFmpeg filters by `build_look_filters()` (colorbalance/eq/vibrance/curves/
  noise/gblur+blend halation/lut3d). The grade is prepended to each clip's `-vf`
  chain via `EffectsConfig.look_name` in `video_effects.build_clip_effect_filters`
  — importantly *outside* the `cfg.enabled` guard, so picking a look applies it
  even with per-clip effects off (same pattern as transitions). Filters are
  probed with `post_ops.has_filter()` and skipped if unsupported. Optional
  `.cube` LUTs live in `assets/luts/`. A few looks carry an off-by-default
  `letterbox` field that pads bars (changes aspect — keep it opt-in).
- **Theme presets** (`src/theme_presets.py`): bundle font combo + look +
  transition preset + effect defaults. `gui.apply_theme` fills the pickers on
  dropdown change; the resolved `theme_name`/`look_name` are threaded through
  `process_video → _render_video_job → _process_video_impl → create_music_video`
  and persisted in the job record + draft manifest. `scripts/render_final_from_draft.py`
  rebuilds `EffectsConfig(look_name=...)` from the manifest so a replayed draft
  keeps its grade.
- **Lyric animations** (`src/lyrics_engine.py`): `_apply_ass_animation` also
  supports `karaoke_word` (word-level `\k` highlight, takes `primary_color`) plus
  `bounce`, `glow`, and `wave`. Font combos carry the `animation` string.

- **Transitions vs. effects master toggle:** the "✨ Enable Visual
  Effects" checkbox only gates the *per-clip* effects (gradient, vignette,
  zoom punch, shake, flash, color boost, glitch, mirror, slow motion). The
  "🔀 Transition Type" dropdown is independent — `needs_transitions()` in
  `video_effects.py` must NOT require `cfg.enabled`, or selecting a
  transition without the master toggle silently falls through to the
  hard-cut concat branch in `video_processor.create_music_video`.
- **Smart Sync transition budget** (`_place_transitions` in `smart_sync.py`):
  Smart Sync never puts a transition on every cut. Cuts are scored by
  `transition_importance` (energy jumps, drop entries, section changes, strong
  hits, plus a bonus for beat-grid bar/phrase anchors); the best cuts are placed
  first, then a beat-locked minimum gap — short in intense passages, wide in
  calm ones (`_min_transition_gap`) — and a rolling cap on heavy dip
  transitions (`fadeblack`/`fadewhite`, `_DIP_MAX_PER_WINDOW` per
  `_DIP_WINDOW_SECONDS`) keep the effect a highlight. An empty string in the
  transition list means hard cut.
- **Smart Sync precedence** (`apply_smart_sync_preferences` in
  `video_processor.py`): Smart Sync alone picks a transition *only where a cut
  is a musical landmark* and hard-cuts the rest (see `transition_importance` /
  `choose_transition_for_segment` in `smart_sync.py`; an empty string in the
  transition list means hard cut). Explicit user choices win — an explicitly
  chosen Transition Type is applied to every cut (uniform, overriding the
  selective shuffle), then a chosen preset, and
  `restrict_overrides_to_user_effects` in `smart_sync.py` keeps Smart Sync from
  injecting effects the user left off (with the caveat that Smart Sync alone,
  master toggle on and no specific effect selected, still drives its own
  dynamic effects).
- **Mixed transitions + hard cuts** (`smart_transitions_ffmpeg` in
  `video_effects.py`): each boundary is either an `xfade` (named transition) or
  a `concat` (hard cut). Every source is first normalised with `fps=<fps>` and
  each join stage ends with `fps=<fps>` too — without that, `concat` drops the
  frame rate and `xfade` refuses inputs whose rates differ
  ("inputs needs to be a constant frame rate"). Keep the `fps` normalisation on
  both sides of every join when touching this filter graph.

## Forensics (`src/forensics`)

Offline analysis of a finished video — "how was this edit made?" — with no
external service. Two independent halves, deliberately kept honest about what
each signal can and cannot prove:

- `containers.py` — a from-scratch ISO-BMFF box/atom parser (no dependencies).
  Reads `ftyp` brands, `udta`/`meta` tags (incl. iTunes `©too`), `stss`
  keyframes, `stts`/`stsz`/`stco`, and `elst` edit lists.
- `fingerprint.py` — attributes the writing software from brands + metadata
  (`ftyp` brand = strong, generic tags = weak), reporting confidence.
- `reencode.py` — GOP length stats and splice evidence. Note: a **uniform GOP
  is normal for any single-pass encode**, not proof of a re-encode; irregular
  GOPs suggest a concat/assembly. An `elst` with a large non-zero `media_time`
  marks a stream-copy trim; a trim that was re-encoded re-zeroes it, so absence
  is not evidence of no edit.
- `scenes.py` — shot detection; uses `scenedetect` when installed (content or
  adaptive), else ffmpeg `select=gt(scene,…)`. `probe_media` wraps ffprobe.
- `report.py` — `analyze_file()` / `format_report()` and a CLI
  (`python -m forensics.report <file> [--json]`).
- `gui_panel.py` — the Gradio glue (kept UI-free in the core). Surfaced as the
  "🔬 Forensics" accordion. The import is wrapped in try/except so a broken
  optional dep never blocks the app.

System tools (`exiftool`, `mediainfo`) and the `scenedetect` Python extra
(`requirements-forensics.txt`) are optional; the report degrades gracefully.

Tests are in `tests/` (pytest, ffmpeg-generated fixtures via `conftest.py`).

## Post-processing (`src/post_ops.py`)

Optional, off-by-default render-path operations surfaced as the
**🎚️ Smoothing, Clipping & Timeline** accordion:

- **Smoothing** — `stabilize_video` (two-pass `vid.stab`), `interpolate_fps`
  (`minterpolate`), and an isolated neural `interpolate_rife` that only runs
  when torch + a Practical-RIFE checkout (`BEATSYNC_RIFE_PATH`) exist. Probe
  filters with `has_filter()`; never assume a filter is present.
- **Clipping** — `detect_silences` / `active_segments` / `trim_to_active_segments`
  remove dead space from source clips before editing (keeps A/V in sync).
- **Syncing** — `estimate_offset` cross-correlates two sources' RMS envelopes
  (normalised overlap correlation; FFT magnitude is biased toward long overlaps,
  so don't revert to raw FFT). `smart_sync.snap_to_scene_boundary` +
  `assign_camera_groups` add scene-aligned starts and multicam awareness.
- **Interchange** — `write_edl` / `write_cutlist_json` / `write_otio`.
  OpenTimelineIO is optional (`requirements-postops.txt`); EDL/JSON always work.

GUI wiring lives in `src/gui.py`: preprocessing (2a/3a) runs in
`_process_video_impl` after source resolution; smoothing (1a/1b) runs right
after the web-optimise pass; interchange (4) is requested via
`create_music_video(export_timeline=True, timeline_dir=…)`.

## Conventions

- Keep changes minimal; the render pipeline is CPU-only (libx264).
- Two-pass encode must keep a matching `-pix_fmt` on both passes (see the
  0-byte-output fix) — `_run_pass()` raises on a failed pass.
