"""Shared test helpers: synthetic media generation with ffmpeg."""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from media_paths import resolve_ffmpeg

FFMPEG, FFPROBE = resolve_ffmpeg()
HAS_FFMPEG = bool(FFMPEG)
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")


def make_video(path, gop=24, duration=4, size="320x240", rate=24):
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=size={size}:rate={rate}:duration={duration}",
         "-c:v", "libx264", "-g", str(gop), "-pix_fmt", "yuv420p", str(path)],
        check=True)


def concat_videos(out, *sources):
    listfile = os.path.join(os.path.dirname(str(out)), "concat_list.txt")
    with open(listfile, "w") as fh:
        for src in sources:
            fh.write(f"file {src}\n")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", listfile, "-c", "copy", str(out)], check=True)
