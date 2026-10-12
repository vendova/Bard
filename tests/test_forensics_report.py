"""End-to-end forensic report: probe + scenes + diagnoses."""
import json

from conftest import requires_ffmpeg, make_video

from forensics.report import analyze_file, format_report, main
from forensics.scenes import probe_media, detect_scenes


@requires_ffmpeg
def test_probe_and_scenes(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p, gop=24, duration=3)
    probe = probe_media(str(p))
    assert abs(probe.fps - 24.0) < 0.01
    assert abs(probe.duration - 3.0) < 0.2
    assert probe.width == 320 and probe.height == 240
    scenes = detect_scenes(str(p))
    assert scenes.method


@requires_ffmpeg
def test_full_report_and_json(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p, gop=24, duration=3)
    report = analyze_file(str(p))
    assert "ffmpeg" in report.software.lower()
    assert report.probe["fps"] > 0
    assert report.gop["keyframe_count"] > 0
    assert report.diagnoses
    text = format_report(report)
    assert "Forensic report" in text
    assert json.loads(json.dumps(report.to_dict()))["software"] == report.software


@requires_ffmpeg
def test_main_cli_json(tmp_path, capsys):
    p = tmp_path / "a.mp4"
    make_video(p, gop=24, duration=2)
    rc = main([str(p), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"].endswith("a.mp4")
