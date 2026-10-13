"""Tests for beat-sync: onset snapping and motion-on-accent alignment."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from media_paths import resolve_ffmpeg
from auto_mode import AutoWaveConfig, analyze_beats_auto
from auto_mode.stage7_snap import snap_cuts_to_onsets
from auto_mode.stage6_av_planner import _build_segment_profiles, _materialize_clip

FFMPEG, _ = resolve_ffmpeg()
requires_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and Path(FFMPEG).exists()), reason="ffmpeg not available"
)


def _click_track(path, duration=30.0, bpm=120):
    interval = 60.0 / bpm
    filt = (f"aevalsrc=0.7*sin(2*PI*90*t)*exp(-35*mod(t\\,{interval}))"
            f":s=44100:d={duration}")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", filt,
         "-c:a", "pcm_s16le", str(path)], check=True)


def test_snap_moves_only_cuts_near_an_onset():
    cfg = AutoWaveConfig()
    onsets = np.array([1.0, 2.0, 3.0])
    selected = np.array([1.02, 2.30, 3.0])

    snapped = snap_cuts_to_onsets(selected, onsets, audio_duration=10.0,
                                  cfg=cfg, strength=1.0)

    # 1.02 is within 0.09s of 1.0 -> snapped; 2.30 is far -> kept; 3.0 exact.
    assert np.isclose(snapped[0], 1.0)
    assert np.isclose(snapped[1], 2.30)
    assert np.isclose(snapped[2], 3.0)


def test_zero_strength_disables_snapping():
    cfg = AutoWaveConfig()
    onsets = np.array([1.0, 2.0, 3.0])
    selected = np.array([1.02, 2.30, 3.0])

    out = snap_cuts_to_onsets(selected, onsets, 10.0, cfg, strength=0.0)
    assert np.allclose(np.sort(selected), out)


def test_snap_never_creates_too_short_shots():
    cfg = AutoWaveConfig()
    onsets = np.array([1.0, 1.02, 1.04, 2.0])
    selected = np.array([1.0, 1.03, 2.0])
    out = snap_cuts_to_onsets(selected, onsets, 10.0, cfg, strength=1.0)
    gaps = np.diff(out)
    assert gaps.size == 0 or gaps.min() >= cfg.peak_energy_min_interval - 1e-9


def test_segment_profiles_expose_accent_anchor():
    beat_info = {
        "times": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        "energy_profile": {"wave": np.array([0.2, 0.3, 0.9, 0.4, 0.5, 0.6, 0.3]),
                           "arc": np.linspace(0, 1, 7)},
        "rhythm_data": {
            "impact_strength": np.array([0.1, 0.2, 0.95, 0.3, 0.4, 0.5, 0.2]),
            "combined_strength": np.array([0.2, 0.3, 0.8, 0.4, 0.5, 0.5, 0.3]),
            "novelty_strength": np.array([0.1, 0.2, 0.7, 0.3, 0.4, 0.5, 0.2]),
        },
        "sections": [{"start": 0.0, "end": 3.5, "type": "drop", "energy": 0.8}],
    }
    profiles = _build_segment_profiles(
        np.array([0.0, 1.5, 3.0, 3.5]), np.array([1.5, 1.5, 0.5]), beat_info)

    first = profiles[0]
    # The strongest beat (1.0s, impact 0.95) sits inside the first segment.
    assert first["anchor_time"] is not None
    assert np.isclose(first["anchor_time"], 1.0)
    assert first["start"] <= first["anchor_time"] <= first["end"]
    assert np.isclose(first["anchor_offset"], first["anchor_time"] - first["start"])
    assert first["anchor_impact"] > 0.9


def test_drop_clip_aligns_motion_peak_to_accent():
    profile = {
        "index": 0, "start": 0.0, "end": 2.0, "duration": 2.0,
        "target": "drop", "anchor_time": 1.4, "anchor_offset": 1.4,
        "anchor_impact": 0.9, "wave": 0.9, "impact": 0.9,
    }
    candidate = {
        "id": "c1", "video_file": "/tmp/x.mp4", "start": 0.0, "center": 5.0,
        "peak_time": 5.0, "video_duration": 20.0, "duration": 2.0,
        "tags": [], "quality_score": 0.8, "action_score": 0.8,
    }
    clip = _materialize_clip(candidate, profile, 0)

    assert clip["beat_alignment"] > 0.8
    # align = anchor_offset / duration = 1.4/2.0 = 0.7 -> start = peak - 0.7*dur.
    assert np.isclose(clip["start_time"], 5.0 - 0.7 * 2.0)
    assert np.isclose(clip["anchor_time"], 1.4)


@requires_ffmpeg
def test_beat_sync_increases_onset_lock(tmp_path):
    audio = tmp_path / "click.wav"
    _click_track(audio)

    _, off = analyze_beats_auto(
        str(audio), video_files=[], enable_video_analysis=False,
        enable_qwen_semantics=False, min_shot_seconds=2.0,
        onset_snap_strength=0.0)
    _, on = analyze_beats_auto(
        str(audio), video_files=[], enable_video_analysis=False,
        enable_qwen_semantics=False, min_shot_seconds=2.0,
        onset_snap_strength=1.0)

    assert on["beat_sync"]["onset_count"] > 0
    assert on["beat_sync"]["fraction_on_onset"] >= off["beat_sync"]["fraction_on_onset"]
    assert on["rhythm_data"]["is_accent"] is not None
