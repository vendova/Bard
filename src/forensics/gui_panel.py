"""Gradio glue for the forensics package.

Kept separate from :mod:`forensics.report` so the analysis core stays UI-free.
"""
from __future__ import annotations

import os
from typing import Optional


def analyze_path(path: Optional[str], method: str = "auto",
                 threshold: float = 0.4) -> str:
    """Return a formatted forensic report for *path* (or a friendly error)."""
    if not path:
        return "_Choose a video file to analyse._"
    if not os.path.isfile(path):
        return f"⚠️ File not found: `{path}`"

    from .report import analyze_file, format_report

    try:
        report = analyze_file(path, cut_threshold=float(threshold),
                              run_scenes=(method != "container-only"))
    except Exception as exc:  # never let the UI crash on a bad file
        return f"⚠️ Could not analyse `{os.path.basename(path)}`: {exc}"

    if method in ("content", "adaptive"):
        from .scenes import detect_scenes
        scenes = detect_scenes(path, threshold=float(threshold), method=method)
        report.cut_count = len(scenes.cut_times)
        report.cut_times = [round(t, 3) for t in scenes.cut_times]
        report.scene_method = scenes.method

    return "```\n" + format_report(report) + "\n```"
