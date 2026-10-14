# lyric_style — parameterised lyric-video edit template

A repeatable, parameterised pipeline that reproduces the **style, structure,
timing and typography** of a short-form romantic lyric edit:

- emotional close-ups,
- a **hard cut on every musical bar**,
- a **blur-and-flash entrance** on every shot,
- a **two-layer lyric caption** (white calligraphic script over a bold
  condensed all-caps key word),
- a small watermark of your handle.

It ships **no third-party footage, music or lyrics**. Everything is either
generated as self-authored test assets or supplied by you and gated through a
licence manifest (Section 1.1). The reference *text* is never copied.

## Pipeline

| Stage | Module | Output |
|-------|--------|--------|
| 2 Ingest | `src/ingest.py` | `assets/footage_index.json` (cuts + scored windows) |
| 3 Audio | `src/analyze_audio.py` | `*.track_analysis.json` (BPM, beats, loudness) |
| 4 Plan | `src/plan.py` | `out/reports/plan_*.json` + printed timeline |
| 5 Render | `src/render.py` | `out/renders/*.mp4` |
| 6 QC | `src/qc.py` | `out/reports/*_qc.json` + `*_qc_sheet.jpg` |
| Batch | `src/make_batch.py` | `out/renders/manifest.csv` |
| Fonts | `src/font_match.py` | `font_match_sheet.png` (pick the pair) |

## Quick start (CPU-only)

```bash
cd lyric_style
python ../-venv/bin/pip install -r requirements.txt   # ffmpeg + ffprobe required

bash scripts/fetch_fonts.sh            # OFL fonts -> assets/fonts
bash scripts/fetch_models.sh           # YuNet face model -> assets/models
bash scripts/make_synthetic_assets.sh  # self-authored test footage + music

# Licence gate: register every asset before it can be used.
# (synthetic assets are pre-registered in assets/licence_manifest.csv)

PYTHONPATH=src python src/ingest.py
PYTHONPATH=src python src/analyze_audio.py
PYTHONPATH=src python src/plan.py \
  --track-analysis assets/music/synth_125bpm.track_analysis.json \
  --captions assets/captions/example.captions --layout reference
PYTHONPATH=src python src/render.py \
  --plan out/reports/plan_reference_20240601.json \
  --music assets/music/synth_125bpm.wav \
  --captions assets/captions/example.captions
PYTHONPATH=src python src/qc.py \
  --plan out/reports/plan_reference_20240601.json \
  --video out/renders/synth_125bpm_reference_20240601.mp4
```

## Fonts

`fonts.script` / `fonts.key` in `config.yaml` must be set before rendering. Run
`src/font_match.py` to render a contact sheet of every candidate pair, choose
one, and set it. Batch rendering refuses to start with an unset pair.

## Assets and rights

Every footage clip, music track and caption file the renderer touches must have
a row in `assets/licence_manifest.csv` with a licence and `allowed=yes`. If an
asset is missing a row or is marked `allowed=no`, the pipeline refuses to
render it and says which asset and why. Do not place lyrics you do not hold the
rights to in `assets/captions/`.

## Determinism

Every random choice derives from `seed` in `config.yaml`; the same config,
assets and seed produce the same edit. Each variant's full tuple is logged to
`out/renders/manifest.csv`.
