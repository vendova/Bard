"""Stage 2 — Ingest: probe footage, detect cuts, score candidate windows.

Produces footage_index.json: for every clip, its probed properties, its internal
scene cuts, and a ranked list of candidate windows (source in/out) with the
features used by the planner (Section 4).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from util import (MissingAssetError, gate_assets, read_json, resolve, run,
                  write_json)


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
def probe(path: str, ffprobe: str) -> dict:
    res = run([
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ])
    import json
    info = json.loads(res.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise MissingAssetError(f"{path}: no video stream")
    num, den = (v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den or 1)
    dur = float(info["format"].get("duration") or v.get("duration") or 0.0)
    return {
        "path": str(Path(path).resolve()),
        "duration": dur,
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": fps,
        "nb_frames": int(v.get("nb_frames") or round(dur * fps)),
    }


# --------------------------------------------------------------------------- #
# Scene cuts (PySceneDetect)
# --------------------------------------------------------------------------- #
def scene_cuts(path: str, threshold: float, min_len_s: float) -> list[float]:
    try:
        from scenedetect import detect, ContentDetector
    except Exception as exc:  # pragma: no cover
        raise MissingAssetError(f"scenedetect unavailable: {exc}") from exc
    scenes = detect(path, ContentDetector(threshold=threshold,
                                          min_scene_len=max(1, int(min_len_s * 30))))
    cuts = [s[0].get_seconds() for s in scenes]
    return sorted(set(round(c, 3) for c in cuts))


# --------------------------------------------------------------------------- #
# Face detection (YuNet primary, no Haar in OpenCV 5)
# --------------------------------------------------------------------------- #
class FaceDetector:
    def __init__(self, model_path: str | None):
        self._net = None
        self._size = None
        if model_path and Path(model_path).exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._net = cv2.FaceDetectorYN.create(model_path, "", (320, 320),
                                                      score_threshold=0.6)
            except Exception:
                self._net = None

    @property
    def available(self) -> bool:
        return self._net is not None

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        if self._net is None:
            return []
        h, w = frame_bgr.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._net.detect(frame_bgr)
        boxes = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = f[0], f[1], f[2], f[3]
                boxes.append((float(x), float(y), float(x + fw), float(y + fh)))
        return boxes


# --------------------------------------------------------------------------- #
# Per-window features
# --------------------------------------------------------------------------- #
def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _text_likelihood(frame_bgr: np.ndarray) -> float:
    """Heuristic: strong, high-frequency horizontal edge structure in bands.

    Burnt-in subtitles/logos produce dense horizontal edges with high contrast.
    Returns 0..1; higher => more likely to contain text.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    # Morphological close along x to merge glyph strokes into text lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    density = float(closed.mean()) / 255.0
    return min(1.0, density * 6.0)


def _frame_features(frame_bgr: np.ndarray, det: FaceDetector) -> dict:
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    boxes = det.detect(frame_bgr)
    if boxes:
        # Largest face dominates "subject emphasis".
        boxes_sorted = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                              reverse=True)
        x0, y0, x1, y1 = boxes_sorted[0]
        face_h = (y1 - y0) / h
        face_area = ((x1 - x0) * (y1 - y0)) / (w * h)
        face_cx = ((x0 + x1) / 2) / w
        face_cy = ((y0 + y1) / 2) / h
    else:
        face_h = face_area = 0.0
        face_cx = face_cy = 0.5
    return {
        "brightness": float(gray.mean()) / 255.0,
        "sharpness": _laplacian_var(gray),
        "text_likelihood": _text_likelihood(frame_bgr),
        "face_count": len(boxes),
        "face_height_frac": face_h,
        "face_area_frac": face_area,
        "face_cx": face_cx,
        "face_cy": face_cy,
    }


def _read_frames(path: str, times: list[float], max_width: int = 640):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise MissingAssetError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > max_width:
            s = max_width / w
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        out.append((t, frame))
    cap.release()
    return out


def _motion_energy(path: str, t0: float, t1: float) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)))
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return 0.0
    prev = cv2.cvtColor(cv2.resize(prev, (160, 90)), cv2.COLOR_BGR2GRAY)
    n = max(1, int((t1 - t0) * min(fps, 10)))
    diffs = []
    for _ in range(n):
        ok, cur = cap.read()
        if not ok:
            break
        cur = cv2.cvtColor(cv2.resize(cur, (160, 90)), cv2.COLOR_BGR2GRAY)
        diffs.append(float(np.abs(cur.astype(np.int16) - prev.astype(np.int16)).mean()))
        prev = cur
    cap.release()
    return float(np.mean(diffs)) if diffs else 0.0


def build_windows(prob: dict, cuts: list[float], det: FaceDetector,
                  win_s: float = 1.5, hop_s: float = 0.5,
                  trim_after_cut: tuple[float, float] = (0.10, 0.30)) -> list[dict]:
    """Slide a window over the clip, avoiding scene-cut boundaries."""
    path = prob["path"]
    dur = prob["duration"]
    boundaries = [0.0] + cuts + [dur]
    windows: list[dict] = []
    for bi in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[bi], boundaries[bi + 1]
        seg_len = seg_end - seg_start
        if seg_len < win_s:
            continue
        # Never start on the source's own cut frame (Section 4.7).
        offset = min(trim_after_cut[1], max(trim_after_cut[0], 0.1))
        start = seg_start + offset
        while start + win_s <= seg_end - 1e-6:
            mid = start + win_s / 2.0
            times = [start + win_s * 0.25, mid, start + win_s * 0.75]
            feats = [_frame_features(f, det) for _, f in _read_frames(path, times)]
            if not feats:
                start += hop_s
                continue
            agg = {
                "brightness": float(np.mean([f["brightness"] for f in feats])),
                "sharpness": float(np.mean([f["sharpness"] for f in feats])),
                "text_likelihood": float(np.mean([f["text_likelihood"] for f in feats])),
                "face_count": float(np.mean([f["face_count"] for f in feats])),
                "face_height_frac": float(np.mean([f["face_height_frac"] for f in feats])),
                "face_area_frac": float(np.mean([f["face_area_frac"] for f in feats])),
                "face_cx": float(np.mean([f["face_cx"] for f in feats])),
                "face_cy": float(np.mean([f["face_cy"] for f in feats])),
            }
            agg["motion"] = _motion_energy(path, start, start + win_s)
            agg["window_start"] = round(start, 3)
            agg["window_end"] = round(start + win_s, 3)
            windows.append(agg)
            start += hop_s
    return windows


# --------------------------------------------------------------------------- #
# Scoring (Section 4)
# --------------------------------------------------------------------------- #
def _norm(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def score_window(win: dict, section: str, weights: dict, energy_cfg: dict,
                 text_penalty: bool = True) -> float:
    face_size = _norm(win["face_height_frac"], 0.25, 0.55)
    face_vis = 1.0 if win["face_count"] >= 1 else 0.0
    sharp = _norm(win["sharpness"], 20, 400)
    b_lo, b_hi = energy_cfg["brightness"]
    brightness_fit = 1.0 - abs(win["brightness"] - (b_lo + b_hi) / 2) / 0.4
    brightness_fit = max(0.0, min(1.0, brightness_fit))
    want_motion = energy_cfg["motion"]
    m = win["motion"]
    if want_motion == "high":
        motion_fit = _norm(m, 2, 12)
    elif want_motion == "low":
        motion_fit = 1.0 - _norm(m, 2, 12)
    else:
        motion_fit = 1.0 - abs(_norm(m, 2, 12) - 0.5) * 2
    no_text = 1.0 - min(1.0, win["text_likelihood"])
    s = (
        weights["face_size"] * face_size
        + weights["face_visibility"] * face_vis
        + weights["sharpness"] * sharp
        + weights["brightness_fit"] * brightness_fit
        + weights["motion_energy"] * motion_fit
        + weights["no_text"] * no_text
    )
    if text_penalty and win["text_likelihood"] > 0.35:
        s *= 0.5
    return float(s)


def ingest(cfg: dict, ffprobe: str, footage_paths: list[str]) -> dict:
    models_dir = resolve(cfg, "models_dir")
    det = FaceDetector(str(models_dir / "face_detection_yunet.onnx"))
    if not det.available:
        raise MissingAssetError(
            "Face detector unavailable. Run `bash scripts/fetch_models.sh` "
            "(needs assets/models/face_detection_yunet.onnx)."
        )
    ig = cfg["ingest"]
    trim = tuple(cfg["selection"]["trim_after_source_cut_s"])
    index = {"clips": [], "face_detector": "yunet"}
    for path in footage_paths:
        prob = probe(path, ffprobe)
        cuts = scene_cuts(path, ig["scene_threshold"], ig["min_scene_len_s"])
        windows = build_windows(prob, cuts, det, trim_after_cut=trim)
        # Score for every section so the planner can pick per slot.
        for w in windows:
            w["scores"] = {
                sec: round(score_window(w, sec, cfg["selection"]["score_weights"],
                                        cfg["selection"]["section_energy"][sec],
                                        cfg["selection"]["reject_burnt_in_text"]), 4)
                for sec in cfg["selection"]["section_energy"]
            }
            w["has_text"] = w["text_likelihood"] > 0.35
        index["clips"].append({**prob, "scene_cuts": cuts, "windows": windows})
    return index


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Stage 2: ingest + index footage")
    ap.add_argument("--config", default=None)
    ap.add_argument("--footage", nargs="*", default=None,
                    help="explicit footage files (else scan assets/footage)")
    args = ap.parse_args(argv)

    from util import load_config, require_ffmpeg
    cfg = load_config(args.config)
    _, ffprobe = require_ffmpeg()
    fdir = resolve(cfg, "footage_dir")
    paths = args.footage or [
        str(p) for p in sorted(fdir.glob("*"))
        if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm", ".avi"}
    ]
    if not paths:
        print(f"No footage found in {fdir}. Add clips or pass --footage.")
        return 2
    # Licence gate — refuse unlicensed footage (Section 1.1).
    paths = gate_assets(cfg, paths, "footage")
    if not paths:
        print("All footage was rejected by the licence gate. Nothing to do.")
        return 3
    index = ingest(cfg, ffprobe, paths)
    out = resolve(cfg, "footage_dir").parent / "footage_index.json"
    write_json(out, index)
    n = sum(len(c["windows"]) for c in index["clips"])
    print(f"✅ Indexed {len(index['clips'])} clips, {n} candidate windows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
