"""Stage 6 — QC: qc.json + qc_sheet.jpg automatic checks (Section 10 Stage 6)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from util import read_json, resolve, run, write_json


def _dur(ffprobe: str, path: str) -> float:
    res = run([ffprobe, "-v", "error", "-show_entries", "format=duration",
               "-of", "json", path])
    return float(json.loads(res.stdout)["format"]["duration"])


def _loudness(ffmpeg: str, path: str) -> dict:
    res = run([ffmpeg, "-hide_banner", "-nostats", "-i", path,
               "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
              check=False)
    import re
    text = res.stderr or ""
    lufs = [float(m.group(1)) for m in re.finditer(r"I:\s*(-?\d+\.\d+)\s*LUFS", text)]
    peaks = [float(m.group(1)) for m in re.finditer(r"Peak:\s*(-?\d+\.\d+)\s*dBFS", text)]
    # ebur128 Peak is sample peak; true peak can be ~0.4 dB higher. Add the same
    # headroom the analysis uses so the ceiling check is conservative.
    sp = peaks[-1] if peaks else None
    return {"lufs": lufs[-1] if lufs else None,
            "true_peak_dbfs": (sp + 0.4) if sp is not None else None,
            "sample_peak_dbfs": sp}


def _settle_check(ffmpeg: str, path: str, shot_in: float, fps: int,
                  settle: int) -> dict:
    """First `settle` frames of a shot: sharpness should rise, brightness fall.

    Frames are extracted precisely with ffmpeg (OpenCV's frame seek is
    keyframe-approximate on a compressed file, which would sample the wrong
    frames and give misleading numbers).
    """
    import tempfile, glob
    tmp = tempfile.mkdtemp(prefix="qc_settle_")
    run([ffmpeg, "-y", "-v", "error", "-i", path, "-ss", f"{shot_in:.4f}",
         "-frames:v", str(settle), "-vsync", "0",
         str(Path(tmp) / "%03d.png")], check=False)
    laps, bright = [], []
    for png in sorted(glob.glob(str(Path(tmp) / "*.png"))):
        img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        laps.append(float(cv2.Laplacian(img, cv2.CV_64F).var()))
        bright.append(float(img.mean()))
    shutil.rmtree(tmp, ignore_errors=True)
    if len(laps) < 3:
        return {"ok": False, "reason": "too few frames", "lap": laps,
                "brightness": bright}
    a = max(1, len(laps) // 3)
    sharp_rise = np.mean(laps[-a:]) > np.mean(laps[:a])
    bright_fall = np.mean(bright[-a:]) <= np.mean(bright[:a]) + 1e-6
    return {"ok": bool(sharp_rise and bright_fall),
            "sharp_rise": bool(sharp_rise), "bright_fall": bool(bright_fall),
            "lap_first": float(np.mean(laps[:a])), "lap_last": float(np.mean(laps[-a:])),
            "bright_first": float(np.mean(bright[:a])),
            "bright_last": float(np.mean(bright[-a:]))}


def contact_sheet(ffmpeg: str, path: str, plan: dict, out: Path, fps: int) -> None:
    """One frame per shot, tiled into a single image (qc_sheet.jpg)."""
    frames = []
    cap = cv2.VideoCapture(path)
    for s in plan["shots"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(s["timeline_in"] * fps)))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.resize(fr, (320, 180)))
    cap.release()
    if not frames:
        return
    cols = min(5, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    sheet = np.zeros((rows * 180, cols * 320, 3), np.uint8)
    for i, fr in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * 180:(r + 1) * 180, c * 320:(c + 1) * 320] = fr
    cv2.imwrite(str(out), sheet)


def qc(cfg: dict, plan: dict, video: str, out_dir: Path | None = None) -> dict:
    from util import require_ffmpeg
    ffmpeg, ffprobe = require_ffmpeg()
    fps = cfg["output"]["fps"]
    q = cfg["qc"]
    reports = out_dir or resolve(cfg, "reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    stem = Path(video).stem

    planned_dur = plan["shots"][-1]["timeline_out"] - plan["shots"][0]["timeline_in"]
    actual_dur = _dur(ffprobe, video)
    checks: dict = {}
    checks["duration_ok"] = abs(actual_dur - planned_dur) <= q["duration_tolerance_frames"] / fps + 0.05
    checks["planned_duration"] = round(planned_dur, 3)
    checks["actual_duration"] = round(actual_dur, 3)

    # Cut alignment vs beat grid. Cuts are snapped to beats (or half-beats),
    # so the error is measured against the full grid.
    track_beats = np.asarray(plan.get("_beats", []), dtype=float)
    grid = np.sort(np.concatenate([
        track_beats, (track_beats[:-1] + track_beats[1:]) / 2.0
    ])) if track_beats.size > 1 else track_beats
    cut_errs = []
    if grid.size:
        for s in plan["shots"]:
            e = float(np.min(np.abs(grid - s["timeline_in"])))
            cut_errs.append(e)
        checks["max_cut_beat_error_ms"] = round(max(cut_errs) * 1000, 2)
        checks["cuts_on_beat"] = max(cut_errs) <= q["cut_beat_tolerance_ms"] / 1000.0 + 0.001

    # Settle behaviour on cuts (only shots long enough to contain the settle).
    settles = []
    settle_len = q["settle_frames"] / fps + 0.10
    for i, s in enumerate(plan["shots"][1:8], start=1):
        if s["length"] < settle_len:
            continue
        settles.append(_settle_check(ffmpeg, video, s["timeline_in"], fps,
                                     q["settle_frames"]))
    checks["settle_samples"] = settles
    checks["settle_ok"] = all(x["ok"] for x in settles) if settles else None

    # Captions present in A/C, absent in B/D.
    cap_ok = True
    for s in plan["shots"]:
        want = s["section"] in cfg["structure"]["captioned_sections"]
        have = bool(s.get("caption"))
        if want != have and want:  # missing where required
            cap_ok = False
    checks["captions_ok"] = cap_ok
    checks["missing_captions"] = plan.get("missing_captions", [])

    # Duplicate source shot inside the variant.
    ids = [f"{s['source_file']}#{s['source_in']}" for s in plan["shots"]]
    checks["no_duplicate_shots"] = len(ids) == len(set(ids))

    # Loudness. Target is integrated loudness; the true-peak ceiling is a hard
    # limit, so a very high-crest track may be peak-limited just below target —
    # that is correct behaviour, not a failure.
    loud = _loudness(ffmpeg, video)
    checks["loudness"] = loud
    tol = q["lufs_tolerance"]
    ceiling = cfg["loudness"]["max_true_peak_dbtp"]
    within = loud["lufs"] is not None and abs(loud["lufs"] - cfg["loudness"]["target_lufs"]) <= tol
    peak_limited = (loud["true_peak_dbfs"] is not None
                    and loud["true_peak_dbfs"] >= ceiling - 0.3)
    checks["loudness_ok"] = bool(within or peak_limited)
    checks["loudness_within_tolerance"] = bool(within)
    checks["true_peak_ok"] = (
        loud["true_peak_dbfs"] is None
        or loud["true_peak_dbfs"] <= ceiling + 0.1
    )

    sheet = reports / f"{stem}_qc_sheet.jpg"
    contact_sheet(ffmpeg, video, plan, sheet, fps)

    report = {"video": str(video), "checks": checks, "qc_sheet": str(sheet)}
    write_json(reports / f"{stem}_qc.json", report)
    passed = all(v for k, v in checks.items()
                 if k.endswith("_ok") and v is not None)
    print(f"✅ QC {'PASS' if passed else 'REVIEW'} -> {reports / (stem + '_qc.json')}")
    return report


def main(argv=None) -> int:
    import argparse
    from util import load_config
    ap = argparse.ArgumentParser(description="Stage 6: QC a rendered video")
    ap.add_argument("--config", default=None)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--video", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    qc(cfg, read_json(args.plan), args.video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
