"""Unified forensic report: fingerprint + GOP/re-encode + shot timeline.

Reads the file bytes for the container analysis and shells out to ffprobe/ffmpeg
for decoding, so it works with no Python extras beyond the standard library.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from .containers import parse_container
from .fingerprint import fingerprint, Fingerprint
from .reencode import analyze_reencode, ReencodeEvidence
from .scenes import probe_media, detect_scenes, MediaProbe, SceneResult


@dataclass
class ForensicReport:
    path: str = ""
    file_size: int = 0
    software: str = "unknown"
    confidence: str = "none"
    editor_metadata: dict = field(default_factory=dict)
    brands: List[str] = field(default_factory=list)
    probe: dict = field(default_factory=dict)
    gop: dict = field(default_factory=dict)
    has_edit_list: bool = False
    reencoded: Optional[bool] = None
    cut_count: int = 0
    cut_times: List[float] = field(default_factory=list)
    scene_method: str = ""
    diagnoses: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "file_size": self.file_size,
            "software": self.software,
            "confidence": self.confidence,
            "editor_metadata": self.editor_metadata,
            "brands": self.brands,
            "probe": self.probe,
            "gop": self.gop,
            "has_edit_list": self.has_edit_list,
            "reencoded": self.reencoded,
            "cut_count": self.cut_count,
            "cut_times": self.cut_times,
            "scene_method": self.scene_method,
            "diagnoses": self.diagnoses,
            "notes": self.notes,
        }


def _diagnose(fp: Fingerprint, ev: ReencodeEvidence, scenes: SceneResult,
              probe: MediaProbe) -> List[str]:
    # NOTE: ``reencoded``/``uniform`` reflect encoder GOP structure only; a
    # uniform GOP is the norm for any single-pass encode, not proof of a
    # re-encode. The report states the structure rather than overclaiming.
    out: List[str] = []
    out.append(f"Writer: {fp.software} (confidence: {fp.confidence})")
    if ev.reencoded:
        out.append("Uniform GOP: consistent with a single continuous encode.")
    if ev.gop.gop_lengths and not ev.gop.uniform:
        out.append("Irregular GOP: stream looks assembled from multiple encodes.")
    if ev.spliced_cuts:
        out.append(f"{len(ev.spliced_cuts)} cut(s) fall mid-GOP (splice/re-encode).")
    if scenes.cut_times:
        shot_count = len(scenes.cut_times) + 1
        avg = probe.duration / shot_count if probe.duration else 0
        out.append(f"{shot_count} shot(s) detected, ~{avg:.1f}s average.")
    return out


def analyze_file(
    path: str,
    *,
    cut_threshold: float = 0.4,
    run_scenes: bool = True,
) -> ForensicReport:
    """Produce a complete forensic report for *path*."""
    report = ForensicReport(path=path)
    with open(path, "rb") as fh:
        data = fh.read()
    report.file_size = len(data)

    info = parse_container(data)
    fp = fingerprint(info)
    report.software = fp.software
    report.confidence = fp.confidence
    report.editor_metadata = fp.editor_metadata
    report.brands = fp.brands
    report.notes.extend(fp.notes)

    probe = probe_media(path)
    report.probe = probe.to_dict()

    scenes = SceneResult()
    if run_scenes and probe.duration > 0:
        scenes = detect_scenes(path, threshold=cut_threshold)
        report.cut_count = len(scenes.cut_times)
        report.cut_times = [round(t, 3) for t in scenes.cut_times]
        report.scene_method = scenes.method

    ev = analyze_reencode(info, scenes.cut_times, probe.fps)
    report.gop = ev.to_dict()["gop"]
    report.has_edit_list = ev.has_edit_list
    report.reencoded = ev.reencoded

    report.diagnoses = _diagnose(fp, ev, scenes, probe)
    return report


def format_report(report: ForensicReport) -> str:
    """Render a report as readable text."""
    lines = [
        f"Forensic report: {os.path.basename(report.path)}",
        f"  File size      : {report.file_size:,} bytes",
        f"  Container      : major={report.brands[0] if report.brands else '?'} "
        f"compatible={report.brands[1:] or '[]'}",
        f"  Writer         : {report.software} (confidence: {report.confidence})",
    ]
    if report.editor_metadata:
        meta = ", ".join(f"{k}={v}" for k, v in report.editor_metadata.items())
        lines.append(f"  Editor metadata: {meta}")
    p = report.probe
    if p:
        lines.append(f"  Video          : {p.get('width')}x{p.get('height')} "
                     f"{p.get('codec')} @ {p.get('fps', 0):.3f} fps, "
                     f"{p.get('duration', 0):.2f}s")
    g = report.gop or {}
    if g:
        lines.append(f"  GOP            : mean={g.get('gop_mean')} std={g.get('gop_std')} "
                     f"keyframes={g.get('keyframe_count')} uniform={g.get('uniform')}")
    lines.append(f"  Edit list      : {report.has_edit_list}")
    gop_pattern = ("single continuous encode" if report.reencoded
                   else "assembled / irregular" if report.reencoded is False
                   else "unknown")
    lines.append(f"  GOP pattern    : {gop_pattern}")
    lines.append(f"  Shots detected : {report.cut_count} ({report.scene_method})")
    if report.diagnoses:
        lines.append("  Diagnoses      :")
        for d in report.diagnoses:
            lines.append(f"    - {d}")
    if report.notes:
        lines.append("  Notes          :")
        for n in report.notes:
            lines.append(f"    - {n}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m forensics.report",
        description="Forensic analysis of a finished video file.")
    parser.add_argument("path", help="video file to analyse")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="scene cut threshold (default 0.4)")
    parser.add_argument("--no-scenes", action="store_true",
                        help="skip shot detection (container/GOP only)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    report = analyze_file(args.path, cut_threshold=args.threshold,
                          run_scenes=not args.no_scenes)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
