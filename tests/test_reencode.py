"""GOP structure and re-encoding / splice evidence."""
from conftest import requires_ffmpeg, make_video, concat_videos

from forensics.containers import parse_container
from forensics.reencode import gop_stats, analyze_reencode


@requires_ffmpeg
def test_uniform_gop(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p, gop=24, duration=4)
    stats = gop_stats(parse_container(p.read_bytes()))
    assert stats.gop_lengths
    assert stats.uniform
    assert abs(stats.gop_mean - 24) <= 1


@requires_ffmpeg
def test_irregular_gop_after_concat(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    make_video(a, gop=24, duration=2)
    make_video(b, gop=6, duration=2)
    out = tmp_path / "cat.mp4"
    concat_videos(out, a, b)
    stats = gop_stats(parse_container(out.read_bytes()))
    assert not stats.uniform


@requires_ffmpeg
def test_reencode_evidence_reports_structure(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p, gop=24, duration=4)
    info = parse_container(p.read_bytes())
    ev = analyze_reencode(info)
    assert ev.gop.keyframe_count > 1
    assert any("GOP" in f for f in ev.findings)


@requires_ffmpeg
def test_midgop_cut_flagged_as_splice(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p, gop=48, duration=4)  # long GOP: one keyframe in the middle
    info = parse_container(p.read_bytes())
    # A cut at 1.0s (frame ~25) does not sit on a keyframe -> splice evidence.
    ev = analyze_reencode(info, cut_times=[1.0], fps=24.0)
    assert 1.0 in ev.spliced_cuts
