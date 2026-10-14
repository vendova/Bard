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

- **Speech-to-text (Lyrics "Generate LRC")**: `src/lyrics_asr.py`. Backends in
  priority order: `faster_whisper` (CTranslate2, no PyTorch — the recommended
  one), `openai-whisper`, `vosk`. Audio is decoded via FFmpeg (never PyAV — see
  `_decode_pcm`) and fed to Whisper as a float32 numpy array. The UI exposes a
  "Speech-to-text model" dropdown (`lyrics_asr.MODEL_CHOICES`): tiny/base/
  small/medium/large-v3 (default)/large-v3-turbo plus community HuggingFace
  fine-tunes. HF repos in the Transformers (PyTorch) format are converted to
  CTranslate2 on first use (`_resolve_faster_whisper_model`, cache under
  `BEATSYNC_ASR_CACHE`, default `~/.cache/beatsync_asr`) and need `torch` +
  `transformers` (CPU torch only; not installed by default). Conversion uses
  `load_as_float16=True, low_cpu_mem_usage=True` — a plain large-v2 conversion
  OOMs a 15 GB CPU box. Env overrides: `BEATSYNC_WHISPER_MODEL`,
  `BEATSYNC_WHISPER_COMPUTE`.
- **LRC line splitting**: Whisper emits long segments (a whole verse, or a
  single blob for sung audio), so `lyrics_asr` transcribes with
  `word_timestamps=True` and regroups words into singable lines at natural
  pauses (`_words_to_lines`; tunable via `BEATSYNC_LRC_LINE_GAP`,
  `BEATSYNC_LRC_LINE_MAX_DUR`, `BEATSYNC_LRC_LINE_MAX_CHARS`). VAD
  `min_silence_duration_ms` is lowered from the 2 s default
  (`BEATSYNC_LRC_VAD_SILENCE_MS`), `hallucination_silence_threshold=2.0` plus
  `_filter_implausible_lines` drop the date/time text Whisper invents over
  trailing silence.
- **Lyric video link** (Lyrics section): paste a download link to a
  black-background lyric clip. It is blended over the edited clips with a
  `lighten` effect and, when no audio file is uploaded, its own audio is
  extracted (`ffmpeg_processing.extract_audio_track`) and used as the music
  track. See `overlay_video_lighten(..., audio_mode='overlay'|'base')`.
- **Multi-link input** (`youtube_source.parse_links` +
  `prepare_clips_from_sources`): downloads each link in turn and spreads a
  ~20-clip budget across successful sources.
- **Whole-song render** (Smooth section → "Render the whole song"): when on,
  `create_music_video(..., whole_song=True)` skips the footage-length crop in
  `_resolve_duration_cap` so the output keeps the full audio length and reuses
  source moments (`_ensure_candidate_coverage` tops up the Smart Sync candidate
  pool). Off by default; an explicit `BEATSYNC_MAX_OUTPUT_SECONDS` still caps.
- **Lyrics: LRC first, burn optional** (Lyrics section): "📝 Generate LRC"
  calls `gui.generate_lrc_output` → `lyrics_engine.build_lyrics_artifact`,
  which detects the script/language (`transliteration.detect_language`),
  parses pasted LRC or auto-times plain text, and writes a `.lrc` shown in the
  UI + offered as a download. Burning captions onto the video is a separate
  opt-in checkbox (`burn_lyrics`), default off; it is threaded through
  `process_video`/`_render_video_job`/`_process_video_impl`/`regenerate_video`
  and persisted for resume.
- **Indic + more scripts** (`transliteration.py`): detection/transliteration
  now covers Bengali, Gurmukhi, Gujarati, Odia, Sinhala (plus the existing
  Devanagari/Telugu/Tamil/Malayalam/Kannada) and Han. `detect_language()`
  returns script/language/indic flags; `INDIC_SCRIPTS` lists the Indic set.
- **Optional ASR** (`lyrics_asr.py`): when the lyrics box is empty and a
  backend is installed (faster-whisper → whisper → vosk, see
  `requirements-lyrics.txt`), the audio is transcribed into timed lines.
  Nothing is bundled; without a backend the UI reports a clear note.
- **Shot pacing** (`auto_mode.AutoWaveConfig.min_shot_seconds`, GUI
  "Minimum shot length"): a floor (default 2.0s) applied on top of the
  energy-based min-intervals in `analyze_beats_auto` via
  `dataclasses.replace`, so shots breathe instead of flickering. It is part
  of the beat-cache key (`render_jobs.beat_cache_key`) — changing it forces a
  re-plan.
- **Neural beat detection** (`src/beat_models/`, GUI "🎵 Neural beat detection"
  in the Smooth accordion, default **off**): optional replacement for the
  librosa beat grid, built for dense percussive/Indic material.
  - `tcn.py` vendors `finnvoorhees/beat-detection-tcn` (log-mel → TCN →
    beat-activation peak-pick); `refiner.py` vendors
    `livechord-music/livechord-beat-refiner` (98-channel CQT/chroma/onset/RMS
    stack + beat/downbeat one-hot grid → Transformer → refined beat **and
    downbeat** grid). Weights download from the HF Hub on first use into
    `BEATSYNC_BEAT_MODEL_CACHE`; offline boxes point `BEATSYNC_BEAT_MODEL_DIR`
    / `BEATSYNC_BEAT_REFINER_DIR` at pre-downloaded dirs.
  - Nothing here imports torch at module load. `beat_models.torch_available()`
    gates the GUI checkbox; `gui.NEURAL_BEATS_AVAILABLE` probes it once at
    import. `beat_models.detect_beat_grid()` never raises — on any failure it
    returns `backend="unavailable"` and the caller keeps librosa.
  - `auto_mode._detect_neural_beats` resamples the already-decoded buffer to
    22050 Hz and calls the package with `y=` (no second decode). When it
    succeeds, `analyze_beats_auto(enable_neural_beats=True)` swaps in the
    neural beat times and passes the real downbeats to
    `stage2_features.analyze_wave_features(downbeat_times=...)`, which snaps
    bar anchors onto them (phrase anchors every `phrase_beats` downbeats) and
    only falls back to the 4/4 every-Nth-beat heuristic when there are none.
    `beat_info["neural_beats"]` reports `{enabled, used, backend, refined, notes}`.
  - **The refiner caps its input at `MAX_FRAMES` (~15 min)**. A refined grid
    that only covers a prefix is rejected and the full-length TCN grid is kept
    (`detect_beat_grid` compares `covered_seconds` vs `input_seconds`), so a
    long song never silently loses its tail.
  - `neural_beats` is threaded like the other pacing knobs: bound in
    `_shared_inputs` **after** `burn_lyrics_enabled` (order must match the
    signatures), passed through `process_video → _render_video_job →
    _process_video_impl`, restored on resume from the job record, and included
    in `render_jobs.beat_cache_key` so toggling it re-plans.
  - Optional dep in `requirements-beats.txt` (torch/huggingface_hub/
    safetensors); it is **not** in `requirements-render.txt` because the Render
    free plan is 512 MB. `tests/test_beat_models.py` builds random-weight
    checkpoints so the real load/forward path is exercised without network.
- **Beat-sync** (`auto_mode.stage7_snap`, GUI "Beat-sync strength",
  `onset_snap_strength` default 1.0): after cut selection, `snap_cuts_to_onsets`
  nudges every cut onto the nearest real transient (`detect_onsets`) within
  `onset_snap_tolerance` (0.09s × strength), re-applying the no-flicker gap.
  This is what makes cuts land on the *audible* hit rather than the quantised
  grid. Stage 2 also exposes `is_accent`/`onset` in `beat_info["rhythm_data"]`.
  - **Motion-on-beat**: `_build_segment_profiles` adds `anchor_time` /
    `anchor_offset` (the strongest beat inside each segment); `_materialize_clip`
    (both `stage6_av_planner` and `smart_sync`) aligns a drop/build/rhythm clip's
    source `peak_time` onto that accent via `beat_alignment`.
  - **Accent flash**: `smart_effect_overrides` sets `flash_offset`, and
    `build_clip_effect_filters` flashes at that time instead of at t=0.
  - Both `onset_snap_strength` and `min_shot_seconds` are part of
    `beat_cache_key` and the persisted job record (resume restores them).
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
