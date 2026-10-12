"""Container/atom parsing and encoder fingerprinting."""
from conftest import requires_ffmpeg, make_video

from forensics.containers import (parse_container, parse_boxes, Box,
                                  edit_list_summary)
from forensics.fingerprint import fingerprint


@requires_ffmpeg
def test_parser_reads_box_tree(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p, gop=24)
    info = parse_container(p.read_bytes())
    assert info.major_brand
    assert info.box_count > 10
    assert "moov" in info.box_types and "stss" in info.box_types
    assert info.sample_count > 0
    assert info.keyframe_indices and min(info.keyframe_indices) >= 1


@requires_ffmpeg
def test_parser_handles_malformed_gracefully(tmp_path):
    p = tmp_path / "junk.bin"
    p.write_bytes(b"not a real mp4 file")
    info = parse_container(p.read_bytes())
    assert info.box_count == 0


@requires_ffmpeg
def test_fingerprint_detects_ffmpeg(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p, gop=24)
    fp = fingerprint(parse_container(p.read_bytes()))
    assert "ffmpeg" in fp.software.lower()
    assert fp.confidence == "high"
    assert any("too" in k for k in fp.editor_metadata)


@requires_ffmpeg
def test_benign_edit_list_is_not_a_trim(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p, gop=24)
    data = p.read_bytes()
    root = Box(type="root", offset=0, size=len(data), header_size=0,
               children=parse_boxes(data))
    has_el, is_trim = edit_list_summary(data, root)
    assert has_el and not is_trim


@requires_ffmpeg
def test_streamcopy_trim_leaves_edit_list_offset(tmp_path):
    import subprocess
    from conftest import FFMPEG
    src = tmp_path / "src.mp4"
    make_video(src, gop=48, duration=6)  # 3s falls mid-GOP, so not KF-aligned
    out = tmp_path / "trim.mp4"
    # A stream-copy cut that does not land on a keyframe keeps the media intact
    # and records a large media_time offset in the edit list: a real trim trace.
    subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", "3", "-i", str(src),
                    "-c", "copy", str(out)], check=True)
    data = out.read_bytes()
    root = Box(type="root", offset=0, size=len(data), header_size=0,
               children=parse_boxes(data))
    has_el, is_trim = edit_list_summary(data, root)
    assert has_el and is_trim
