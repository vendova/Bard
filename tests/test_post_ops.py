"""Tests for post_ops (smoothing / clipping / syncing / interchange)."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import post_ops as po
from media_paths import resolve_ffmpeg

FFMPEG, FFPROBE = resolve_ffmpeg()


def _have_ffmpeg() -> bool:
    return bool(FFMPEG and Path(FFMPEG).exists())


requires_ffmpeg = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")


def _has_otio() -> bool:
    try:
        import opentimelineio  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def tone_with_gap(tmp_path_factory):
    """2s tone, 2s silence, 2s tone."""
    out = tmp_path_factory.mktemp("audio") / "gap.wav"
    subprocess.run([
        FFMPEG, "-y", "-v", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=2",
        "-f", "lavfi", "-i", "sine=frequency=660:duration=2",
        "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1[a]",
        "-map", "[a]", str(out),
    ], check=True)
    return str(out)


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("video") / "tiny.mp4"
    subprocess.run([
        FFMPEG, "-y", "-v", "error", "-f", "lavfi",
        "-i", "testsrc=size=160x120:rate=24:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "24", str(out),
    ], check=True)
    return str(out)


# --- Interchange ---------------------------------------------------------

def test_write_edl_timecodes(tmp_path):
    edits = [
        po.Edit(reel="a.mp4", name="/x/a.mp4", start_src=1.0, end_src=2.5, start_rec=0.0),
        po.Edit(reel="b.mp4", name="/x/b.mp4", start_src=5.0, end_src=6.2, start_rec=1.5),
    ]
    path = po.write_edl(edits, str(tmp_path / "cut.edl"), fps=30)
    text = Path(path).read_text()
    assert text.startswith("TITLE: BEATSYNC")
    assert "FCM: NON-DROP FRAME" in text
    assert "00:00:01:00 00:00:02:15" in text   # src in/out for 1.0..2.5 @30
    assert "* FROM CLIP NAME: /x/a.mp4" in text


def test_cutlist_json_records_edits(tmp_path):
    edits = [po.Edit(reel="a.mp4", start_src=2.0, end_src=4.0, start_rec=0.0)]
    path = po.write_cutlist_json(edits, str(tmp_path / "cut.json"), fps=25, meta={"x": 1})
    import json
    data = json.loads(Path(path).read_text())
    assert data["fps"] == 25
    assert data["edit_count"] == 1
    assert data["edits"][0]["duration"] == 2.0
    assert data["meta"] == {"x": 1}


@pytest.mark.skipif(not _has_otio(), reason="opentimelineio not installed")
def test_otio_export_roundtrip(tmp_path):
    import opentimelineio as otio
    edits = [
        po.Edit(reel="a.mp4", name="/x/a.mp4", start_src=1.0, end_src=2.0),
        po.Edit(reel="b.mp4", name="/x/b.mp4", start_src=1.0, end_src=2.5, start_rec=1.0),
    ]
    path = po.write_otio(edits, str(tmp_path / "cut.otio"), fps=30)
    tl = otio.adapters.read_from_file(path)
    assert [c.name for c in tl.tracks[0]] == ["/x/a.mp4", "/x/b.mp4"]


def test_edits_from_plan_shapes():
    planned = [
        {"source_file": "/a.mp4", "start_time": 1.0, "duration": 2.0},
        {"path": "/b.mp4", "start": 3.0, "end_time": 5.5},
    ]
    edits = po.edits_from_plan(planned, fps=30)
    assert edits[0].start_src == 1.0 and edits[0].duration == 2.0
    assert edits[1].start_src == 3.0 and edits[1].duration == 2.5
    assert edits[1].start_rec == 2.0  # plays after the first clip


# --- Clipping ------------------------------------------------------------

@requires_ffmpeg
def test_detect_silences_and_active_segments(tone_with_gap):
    sils = po.detect_silences(tone_with_gap, noise_db=-35, min_duration=0.4)
    assert len(sils) == 1
    assert abs(sils[0].start - 2.0) < 0.2 and abs(sils[0].end - 4.0) < 0.2

    dur = po.probe_duration(tone_with_gap)
    segs = po.active_segments(dur, sils)
    assert segs[0][0] == 0.0 and abs(segs[0][1] - 2.0) < 0.2
    assert abs(segs[1][0] - 4.0) < 0.2 and abs(segs[1][1] - dur) < 0.2


def test_active_segments_no_silence():
    segs = po.active_segments(10.0, [])
    assert segs == [(0.0, 10.0)]


@requires_ffmpeg
def test_trim_to_active_segments_shortens_video(tone_with_gap, tmp_path):
    # A video whose audio has a 2s silent gap should shrink when trimmed.
    import subprocess as sp
    vid = tmp_path / "withgap.mp4"
    sp.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi",
            "-i", "testsrc=size=160x120:rate=25:duration=6",
            "-i", tone_with_gap, "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(vid)], check=True)
    sils = po.detect_silences(str(vid), noise_db=-35, min_duration=0.4)
    dur = po.probe_duration(str(vid))
    segs = po.active_segments(dur, sils)
    out = tmp_path / "trimmed.mp4"
    assert po.trim_to_active_segments(str(vid), str(out), segs)
    assert po.probe_duration(str(out)) < dur - 1.0


# --- Syncing -------------------------------------------------------------

@requires_ffmpeg
def test_estimate_offset_positive(tone_with_gap, tmp_path):
    delayed = tmp_path / "delayed.wav"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", tone_with_gap,
                    "-af", "adelay=500", str(delayed)], check=True)
    off = po.estimate_offset(tone_with_gap, str(delayed))
    assert abs(off - 0.5) <= 0.05


@requires_ffmpeg
def test_estimate_offset_negative(tone_with_gap, tmp_path):
    advanced = tmp_path / "advanced.wav"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", tone_with_gap,
                    "-af", "atrim=start=0.4,asetpts=PTS-STARTPTS", str(advanced)],
                   check=True)
    off = po.estimate_offset(tone_with_gap, str(advanced))
    assert abs(off + 0.4) <= 0.05


@requires_ffmpeg
def test_align_offset_applies_delay(tone_with_gap, tmp_path):
    dst = tmp_path / "aligned.m4a"
    ok = po.align_offset(str(dst), tone_with_gap, 0.5)
    assert ok and Path(dst).stat().st_size > 0


# --- Smoothing -----------------------------------------------------------

def test_filter_probe_does_not_crash():
    # Whatever this build ships, the probes must return a bool, not raise.
    assert isinstance(po.has_filter("minterpolate"), bool)
    assert po.rife_available() in (True, False)


@requires_ffmpeg
def test_interpolate_doubles_fps(tiny_video, tmp_path):
    if not po.has_filter("minterpolate"):
        pytest.skip("minterpolate unavailable")
    dst = tmp_path / "sixty.mp4"
    po.interpolate_fps(tiny_video, str(dst), 48, quality="draft")
    assert abs(po.probe_fps(str(dst)) - 48) < 1.0


@requires_ffmpeg
def test_stabilize_produces_output(tiny_video, tmp_path):
    if not (po.has_filter("vidstabdetect") and po.has_filter("vidstabtransform")):
        pytest.skip("vid.stab unavailable")
    dst = tmp_path / "stab.mp4"
    assert po.stabilize_video(tiny_video, str(dst), quality="draft")
    assert Path(dst).stat().st_size > 0


# --- Glue ----------------------------------------------------------------

@requires_ffmpeg
def test_postprocess_chain_stabilize_then_interpolate(tiny_video, tmp_path):
    if not po.has_filter("minterpolate"):
        pytest.skip("minterpolate unavailable")
    cfg = po.PostOps(stabilize=po.has_filter("vidstabtransform"),
                     interpolate=True, interpolate_fps=48)
    dst = tmp_path / "out.mp4"
    po.postprocess(tiny_video, str(dst), cfg, quality="draft")
    assert Path(dst).stat().st_size > 0
    assert abs(po.probe_fps(str(dst)) - 48) < 2.0


def test_postprocess_noop_when_disabled(tiny_video):
    cfg = po.PostOps()
    assert po.postprocess(tiny_video, "/tmp/never_used.mp4", cfg) == tiny_video
