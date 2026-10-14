"""Tests for the whole-song render option (no trim-to-footage cap)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import video_processor as vp
from media_paths import resolve_ffmpeg

FFMPEG, FFPROBE = resolve_ffmpeg()
requires_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and Path(FFMPEG).exists()), reason="ffmpeg not available"
)


@pytest.fixture
def two_short_clips(tmp_path):
    from conftest import make_video
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    make_video(str(a), duration=4)
    make_video(str(b), duration=4)
    return [str(a), str(b)]


def test_whole_song_off_crops_to_footage(two_short_clips, monkeypatch):
    monkeypatch.delenv("BEATSYNC_MAX_OUTPUT_SECONDS", raising=False)
    cap = vp._resolve_duration_cap(two_short_clips, audio_duration=120.0,
                                   whole_song=False)
    assert cap is not None
    assert cap == pytest.approx(8.0, abs=0.5)


def test_whole_song_on_keeps_full_audio(two_short_clips, monkeypatch):
    monkeypatch.delenv("BEATSYNC_MAX_OUTPUT_SECONDS", raising=False)
    cap = vp._resolve_duration_cap(two_short_clips, audio_duration=120.0,
                                   whole_song=True)
    # No crop: full 120s music is kept (returns None = no cap applied).
    assert cap is None


def test_whole_song_still_honours_env_cap(two_short_clips, monkeypatch):
    monkeypatch.setenv("BEATSYNC_MAX_OUTPUT_SECONDS", "60")
    cap = vp._resolve_duration_cap(two_short_clips, audio_duration=120.0,
                                   whole_song=True)
    assert cap == pytest.approx(60.0, abs=0.5)


def test_whole_song_short_audio_is_untouched(two_short_clips, monkeypatch):
    monkeypatch.delenv("BEATSYNC_MAX_OUTPUT_SECONDS", raising=False)
    assert vp._resolve_duration_cap(two_short_clips, audio_duration=5.0,
                                    whole_song=False) is None
    assert vp._resolve_duration_cap(two_short_clips, audio_duration=5.0,
                                    whole_song=True) is None


def test_candidate_coverage_preserves_existing_and_tops_up(two_short_clips):
    existing = [{"id": f"c{i}", "video_file": two_short_clips[0],
                 "quality_score": 0.9} for i in range(3)]
    out = vp._ensure_candidate_coverage(two_short_clips, existing, segment_count=40)
    assert len(out) >= 40
    # Original candidates keep their ids and are untouched.
    ids = {c["id"] for c in out}
    assert {"c0", "c1", "c2"}.issubset(ids)
    assert out[0] is existing[0]
    # New windows carry the fields the planner relies on.
    new = [c for c in out if c["id"].startswith("cover_")]
    assert new
    for c in new:
        assert c["video_file"] in two_short_clips
        assert c["start"] >= 0
        assert c["end"] > c["start"]
        assert c["duration"] > 0


def test_candidate_coverage_noop_when_pool_is_big_enough(two_short_clips):
    existing = [{"id": f"c{i}", "video_file": two_short_clips[0]} for i in range(10)]
    out = vp._ensure_candidate_coverage(two_short_clips, existing, segment_count=5)
    assert out is existing
