"""Tests for the minimum-shot-length pacing floor in Auto Mode."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from media_paths import resolve_ffmpeg
from auto_mode import AutoWaveConfig, analyze_beats_auto

FFMPEG, _ = resolve_ffmpeg()
requires_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and Path(FFMPEG).exists()), reason="ffmpeg not available"
)


def _click_track(path, duration=30.0, bpm=120):
    """A dense percussive click track so beat detection finds many beats."""
    interval = 60.0 / bpm
    filt = (f"aevalsrc=0.7*sin(2*PI*90*t)*exp(-35*mod(t\\,{interval}))"
            f":s=44100:d={duration}")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", filt,
         "-c:a", "pcm_s16le", str(path)], check=True)


def test_config_default_floor_is_two_seconds():
    assert AutoWaveConfig().min_shot_seconds == 2.0


@requires_ffmpeg
def test_pacing_floor_enforces_minimum_shot(tmp_path):
    audio = tmp_path / "click.wav"
    _click_track(audio)

    sel, _ = analyze_beats_auto(
        str(audio), video_files=[], enable_video_analysis=False,
        enable_qwen_semantics=False, min_shot_seconds=2.0)

    gaps = np.diff(sel)
    assert gaps.size > 0
    # Every shot must respect the floor (frame quantisation may add a hair).
    assert gaps.min() >= 2.0 - 1e-6, f"min shot {gaps.min():.3f}s < 2.0s"


@requires_ffmpeg
def test_floor_creates_fewer_cuts_than_default(tmp_path):
    audio = tmp_path / "click.wav"
    _click_track(audio)

    dense, _ = analyze_beats_auto(
        str(audio), video_files=[], enable_video_analysis=False,
        enable_qwen_semantics=False, min_shot_seconds=0.5)
    calm, _ = analyze_beats_auto(
        str(audio), video_files=[], enable_video_analysis=False,
        enable_qwen_semantics=False, min_shot_seconds=2.0)

    assert len(calm) < len(dense)
