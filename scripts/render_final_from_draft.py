#!/usr/bin/env python3
"""Render a full-quality final video from a draft manifest.

A draft pass (BEATSYNC_DRAFT=1) encodes fast, low-quality proxy clips and
writes a manifest capturing the whole edit: the exact cut list, transitions,
per-clip effect overrides, frame-locked timing and, most importantly, the
variation seed. Replaying the manifest re-runs the *identical* edit at the
configured (full) quality, so editing is fast but the final output is not
degraded by the draft.

Usage:
    python scripts/render_final_from_draft.py input/drafts/manifest_<job>.json
    python scripts/render_final_from_draft.py --latest
    python scripts/render_final_from_draft.py input/drafts/manifest_<job>.json \
        --output output/my_video_final.mp4
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _latest_manifest() -> str:
    pattern = os.path.join(ROOT, "input", "drafts", "manifest_*.json")
    matches = glob.glob(pattern)
    if not matches:
        raise SystemExit(f"No draft manifests found in {os.path.dirname(pattern)}")
    return max(matches, key=os.path.getmtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", help="Path to a draft manifest JSON")
    parser.add_argument("--latest", action="store_true",
                        help="Use the most recently written manifest")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file (default: output/<audio>_final.mp4)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel clip workers (default: library default)")
    args = parser.parse_args()

    if args.latest or not args.manifest:
        manifest_path = _latest_manifest()
    else:
        manifest_path = args.manifest
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"Manifest not found: {manifest_path}")

    import json
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    # Replay from the manifest, but drop the draft flag so clips encode at
    # full configuration quality.
    os.environ.pop("BEATSYNC_DRAFT", None)

    from video_processor import create_music_video

    audio_file = manifest["audio_file"]
    video_files = manifest["video_files"]
    beat_times = manifest.get("selected_beats") or []
    if not beat_times:
        raise SystemExit("Manifest has no 'selected_beats'; cannot rebuild the cut timeline")
    if not os.path.isfile(audio_file):
        raise SystemExit(f"Audio file missing: {audio_file}")
    missing = [v for v in video_files if not os.path.isfile(v)]
    if missing:
        raise SystemExit(f"Source video(s) missing: {missing}")

    output = args.output
    if not output:
        name = os.path.splitext(os.path.basename(audio_file))[0]
        out_dir = os.path.join(ROOT, "output")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, f"{name}_final.mp4")

    print(f"🎬 Replaying draft {os.path.basename(manifest_path)} at full quality")
    print(f"   Seed: {manifest.get('seed')}  |  "
          f"Clips: {len(manifest.get('planned_clips') or [])}  |  "
          f"FPS: {manifest.get('fps')}")
    print(f"   Output: {output}")

    result = create_music_video(
        audio_file=audio_file,
        video_files=video_files,
        beat_times=beat_times,
        output_file=output,
        start_time=float(manifest.get("start_time") or 0.0),
        end_time=manifest.get("end_time"),
        max_workers=args.workers,
        beat_info={"times": beat_times},
        lossless_mode=False,
        use_gpu=False,
        gpu_encoder="none",
        fps=manifest.get("fps"),
        smart_sync=bool(manifest.get("smart_sync", False)),
        transition_preset_name=manifest.get("transition_preset_name") or "",
        draft_manifest=manifest_path,
    )
    print(f"✅ Final video: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
