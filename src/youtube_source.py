#!/usr/bin/env python3
"""YouTube -> 20 usable clips pipeline.

Given a YouTube URL this module:

1. Downloads the video (and gives every clip a music-free audio track).
2. Splits it into ~20 short clips, preferring scene cuts.
3. Scores each moment and drops windows that contain baked-in text overlays
   (captions, lower-thirds, logos) so lyric/text-free footage is preferred.
4. Optionally crops residual text bands and re-encodes each clip.

Audio policy
------------
Only a small fraction of clips keep their original audio, and those keep
dialogue / action sound effects only. Clips whose audio is dominated by
continuous background music are forced silent. This is a *lightweight* guard
(the repo has no source-separation model); set ``BEATSYNC_YOUTUBE_KEEP_AUDIO=0``
to make every clip silent, or install ``demucs`` and set
``BEATSYNC_YOUTUBE_DEMUCS=1`` for real music separation.

Everything is offline-testable: ``build_clips_from_video`` runs the exact same
pipeline against any local file.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from ffmpeg_processing import (
    FFMPEG_PATH,
    detect_video_scene_changes,
    get_video_duration,
    get_video_fps,
    get_video_resolution,
)


TARGET_CLIP_COUNT = 20
DEFAULT_CLIP_SECONDS = 2.6
MIN_CLIP_SECONDS = 1.6
MAX_CLIP_SECONDS = 4.5

# Text-overlay heuristic tuning.
_TEXT_EDGE_VAR_THRESHOLD = 48.0
_TEXT_LUMINANCE_THRESHOLD = 200.0
_TEXT_REGION_PAD = 0.06  # fraction of frame height cropped off a text band


@dataclass
class OverlayDetection:
    has_text: bool
    regions: List[tuple] = field(default_factory=list)  # (top, bottom) in height fraction
    score: float = 0.0


@dataclass
class ClipPlan:
    start: float
    end: float
    has_audio: bool
    overlay_free: bool
    audio_mode: str  # "keep" | "silent" | "demucs"
    video_path: str = ""


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _run(cmd: Sequence[str], log: Optional[Callable[[str], None]] = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0 and log:
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-4:])
        log(f"   ⚠️  ffmpeg failed ({proc.returncode}): {tail}")
    return proc


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _locate_ytdlp() -> Optional[Sequence[str]]:
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        return None
    import sys
    return [sys.executable, "-m", "yt_dlp"]


def is_youtube_url(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"(youtube\.com|youtu\.be)", text, re.IGNORECASE))


_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")


def is_video_url(text: str) -> bool:
    """True for a direct media link (path ends in a known video extension)."""
    if not text:
        return False
    path = urlparse(text.strip()).path
    return path.lower().endswith(_VIDEO_EXTENSIONS)


def _guess_filename(url: str, content_disposition: str | None = None) -> str:
    if content_disposition:
        m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", content_disposition, re.IGNORECASE)
        if m:
            name = unquote(m.group(1).strip())
            if name:
                root, ext = os.path.splitext(name)
                if ext.lower() not in _VIDEO_EXTENSIONS:
                    name = f"{root or 'source'}.mp4"
                return name
    path = unquote(urlparse(url).path)
    name = os.path.basename(path) or "download"
    root, ext = os.path.splitext(name)
    if ext.lower() not in _VIDEO_EXTENSIONS:
        ext = ".mp4"
    return f"source{ext}" if not root else f"{root}{ext}"


_PROBE_VIDEO_TYPES = ("video/", "application/octet-stream", "binary/octet-stream",
                      "application/mp4", "application/x-matroska")


def probe_content_type(url: str, log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Best-effort content-type of a URL (HEAD, then a ranged GET fallback)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, headers=headers, method=method)
            if method == "GET":
                req.add_header("Range", "bytes=0-0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                ctype = (resp.headers.get_content_type() or "").lower()
                if ctype:
                    return ctype
        except Exception:
            continue
    return None


def download_direct_url(url: str, dest_dir: str,
                        log: Optional[Callable[[str], None]] = None) -> str:
    """Download a direct media URL (no extraction needed) with a browser UA."""
    os.makedirs(dest_dir, exist_ok=True)
    req = urllib.request.Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "*/*",
    })
    with urllib.request.urlopen(req, timeout=120) as resp:
        ctype = (resp.headers.get_content_type() or "").lower()
        if ctype.startswith("text/") or ctype == "application/json":
            raise RuntimeError(
                f"URL returned {ctype or 'an HTML page'}, not a video file "
                "(it may be a landing page rather than a direct link)."
            )
        if ctype and not any(ctype.startswith(t) or ctype == t for t in _PROBE_VIDEO_TYPES):
            if log:
                log(f"   ⚠️  Unusual content-type {ctype}; saving anyway")
        out_path = os.path.join(
            dest_dir, _guess_filename(url, resp.headers.get("Content-Disposition"))
        )
        if log:
            log(f"   ⬇️  Downloading {os.path.basename(out_path)}…")
        with open(out_path, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("Direct download produced an empty file (check the URL).")
    return out_path


def download_media(url: str, dest_dir: str,
                   log: Optional[Callable[[str], None]] = None) -> str:
    """Save any supported link: direct video URL, downloader endpoint, or site page.

    Routing:
      * URL ending in a media extension -> fetch directly.
      * URL whose server reports a video/octet-stream content-type (e.g. a
        ``.../admin-ajax.php?action=download`` endpoint) -> fetch directly.
      * otherwise -> yt-dlp; if yt-dlp refuses due to an unusual extension,
        fall back to a direct fetch.
    """
    if is_video_url(url):
        return download_direct_url(url, dest_dir, log=log)

    ctype = probe_content_type(url, log=log)
    if ctype and any(ctype.startswith(t) or ctype == t for t in _PROBE_VIDEO_TYPES):
        if log:
            log(f"   🔎 Server reports {ctype} → direct download")
        return download_direct_url(url, dest_dir, log=log)

    try:
        return download_youtube(url, dest_dir, log=log)
    except RuntimeError as e:
        if "unusual and will be skipped" in str(e) or "unusual extension" in str(e):
            if log:
                log("   ↩️  yt-dlp skipped the file; retrying as a direct download")
            return download_direct_url(url, dest_dir, log=log)
        raise


def download_youtube(url: str, dest_dir: str,
                     log: Optional[Callable[[str], None]] = None) -> str:
    """Download ``url`` into ``dest_dir``; return the local video path.

    Datacenter IPs (CI sandboxes, some cloud hosts) are frequently hit by
    YouTube's "Sign in to confirm you're not a bot" check. If that happens,
    export cookies from a logged-in browser and point at them via
    ``BEATSYNC_YT_COOKIES=/path/cookies.txt`` (or
    ``BEATSYNC_YT_COOKIES_FROM_BROWSER=chrome``).
    """
    os.makedirs(dest_dir, exist_ok=True)
    ytdlp = _locate_ytdlp()
    if ytdlp is None:
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install yt-dlp  "
            "(or re-run scripts/setup_codespaces.sh)."
        )
    outtmpl = os.path.join(dest_dir, "source.%(ext)s")
    cmd = list(ytdlp) + [
        "--no-playlist",
        "--no-warnings",
        # ffmpeg as the downloader survives throttled m3u8 streams on many IPs.
        "--downloader", "ffmpeg",
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/best",
        "--merge-output-format", "mp4",
        "-o", outtmpl,
    ]
    cookies = os.environ.get("BEATSYNC_YT_COOKIES")
    if cookies and os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    browser = os.environ.get("BEATSYNC_YT_COOKIES_FROM_BROWSER")
    if browser:
        cmd += ["--cookies-from-browser", browser]
    extractor_args = os.environ.get("BEATSYNC_YT_EXTRACTOR_ARGS")
    if extractor_args:
        cmd += ["--extractor-args", extractor_args]
    cmd.append(url)

    if log:
        log("   ⬇️  Downloading YouTube source…")
    proc = _run(cmd, log)

    for name in sorted(os.listdir(dest_dir)):
        if name.startswith("source.") and name.rsplit(".", 1)[-1].lower() in {"mp4", "mkv", "webm", "mov"}:
            return os.path.join(dest_dir, name)

    tail = "\n".join((proc.stdout or "").strip().splitlines()[-3:])
    blob = (proc.stdout or "")
    blocked = (
        "Sign in to confirm" in blob
        or "not a bot" in blob
        or "403 Forbidden" in blob
        or "confirm you" in blob
        or "This video is unavailable" in blob
    )
    if blocked:
        raise RuntimeError(
            "The site refused the download (bot check / HTTP 403). This is common on "
            "datacenter or shared IPs. On a normal home/Codespace network it usually "
            "works; if not, export cookies from a logged-in browser and set "
            "BEATSYNC_YT_COOKIES=/path/cookies.txt (or BEATSYNC_YT_COOKIES_FROM_BROWSER=chrome). "
            "You can also download the video yourself and upload the clip(s) directly. "
            f"Details: {tail}"
        )
    raise RuntimeError(f"Download did not produce a video file. Details: {tail}")


# --------------------------------------------------------------------------- #
# Text-overlay detection
# --------------------------------------------------------------------------- #
def detect_text_overlay(frame_bgr: np.ndarray) -> OverlayDetection:
    """Cheap text-overlay detector using horizontal edge density + bright strokes.

    Baked-in captions/lower-thirds produce many thin, high-contrast horizontal
    edges concentrated in a horizontal band; natural footage does not. This runs
    at full speed on CPU and needs no OCR model.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return OverlayDetection(False)
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Horizontal Sobel -> text strokes light up strongly on the vertical edges.
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge = np.abs(sobel_x)

    # Keep only strong, bright strokes (text is usually light-on-dark).
    bright = gray > _TEXT_LUMINANCE_THRESHOLD
    strong_edge = (edge > _TEXT_EDGE_VAR_THRESHOLD).astype(np.float32) * bright.astype(np.float32)

    row_density = strong_edge.mean(axis=1)  # per-row stroke density
    total = float(row_density.mean())
    if total < 0.004:
        return OverlayDetection(False, [], total)

    # Find the densest band (text lives in a contiguous horizontal strip).
    window = max(8, int(h * 0.12))
    smoothed = np.convolve(row_density, np.ones(window) / window, mode="same")
    peak = int(np.argmax(smoothed))
    band_half = max(12, int(h * 0.10))
    top = max(0, peak - band_half)
    bottom = min(h, peak + band_half)
    band_mean = float(smoothed[peak])

    # Local concentration relative to the rest of the frame.
    ratio = band_mean / (total + 1e-6)
    has_text = band_mean > 0.010 and ratio > 1.6
    regions = [(top / h, bottom / h)] if has_text else []
    return OverlayDetection(has_text, regions, total)


def _sample_frames(video_path: str, fps: float, start: float, end: float,
                   count: int = 6) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    if not cap.isOpened():
        return frames
    try:
        duration = max(0.0, end - start)
        for i in range(count):
            t = start + duration * (i + 0.5) / count
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def measure_silence_ratio(video_path: str, start: float, end: float) -> float:
    """Fraction of the window that is silent (used to reject silent/BGM-only beds).

    Returns 1.0 when the window has no audio at all (treated as silence).
    """
    dur = max(0.0, end - start)
    if dur <= 0:
        return 1.0
    cmd = [
        FFMPEG_PATH, "-nostdin", "-hide_banner",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", video_path,
        "-vn", "-af", "silencedetect=noise=-35dB:d=0.25", "-f", "null", "-",
    ]
    proc = _run(cmd)
    out = proc.stdout or ""
    silence = 0.0
    for m in re.finditer(r"silence_duration:\s*([0-9.]+)", out):
        silence += float(m.group(1))
    if re.search(r"audio:\s*0 streams", out) or "does not contain any stream" in out:
        return 1.0
    return float(np.clip(silence / dur, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def _window_bounds(duration: float, boundaries: Sequence[float], target: int,
                   clip_len: float) -> List[tuple]:
    cuts = [0.0] + [b for b in boundaries if 0.5 < b < duration - 0.5] + [duration]
    wins: List[tuple] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = b - a
        if seg < MIN_CLIP_SECONDS:
            continue
        if seg > MAX_CLIP_SECONDS * 1.5:
            # Split long shots into evenly timed sub-windows.
            n = max(1, int(round(seg / clip_len)))
            step = seg / n
            for k in range(n):
                wins.append((a + k * step, a + (k + 1) * step))
        else:
            wins.append((a, b))
    if len(wins) >= target:
        return wins
    # Not enough scene cuts: fall back to a uniform grid.
    n = max(target, int(duration // clip_len) or target)
    step = duration / n
    return [(k * step, (k + 1) * step) for k in range(n)]


def build_clip_plan(
    video_path: str,
    target_count: int = TARGET_CLIP_COUNT,
    log: Optional[Callable[[str], None]] = None,
) -> List[ClipPlan]:
    """Choose ~``target_count`` windows, preferring scene cuts and text-free frames."""
    duration = max(0.0, float(get_video_duration(video_path)))
    fps = max(1.0, float(get_video_fps(video_path)))
    if duration <= 0:
        raise RuntimeError(f"Could not read video duration for {video_path}")

    clip_len = _env_float("BEATSYNC_YOUTUBE_CLIP_SECONDS", DEFAULT_CLIP_SECONDS)
    boundaries = detect_video_scene_changes(
        video_path, threshold=0.27, use_gpu=False, analysis_fps=5.0, analysis_width=320
    )
    windows = _window_bounds(duration, boundaries, target_count * 2, clip_len)
    if log:
        log(f"   🎬 {len(boundaries)} scene cuts → {len(windows)} candidate windows")

    # Score each window: illustration quality + overlay cleanliness.
    scored = []
    for (a, b) in windows:
        frames = _sample_frames(video_path, fps, a, b, count=6)
        if not frames:
            continue
        detections = [detect_text_overlay(f) for f in frames]
        text_frames = sum(1 for d in detections if d.has_text)
        overlay_free = text_frames == 0

        # Prefer lively, detailed frames (edge energy) and penalise text.
        detail = float(np.mean([
            cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
            for f in frames
        ]))
        score = detail / (1.0 + 4.0 * text_frames)
        regions = [r for d in detections if d.has_text for r in d.regions]
        scored.append((score, overlay_free, a, b, regions))

    scored.sort(key=lambda s: (s[1], s[0]), reverse=True)

    # Keep a spread: take overlay-free first, then fill gaps from the rest.
    chosen: List[tuple] = []
    for item in scored:
        if len(chosen) >= target_count:
            break
        if not any(abs(item[2] - c[2]) < clip_len * 0.6 for c in chosen):
            chosen.append(item)
    for item in scored:
        if len(chosen) >= target_count:
            break
        if item not in chosen:
            chosen.append(item)
    chosen = chosen[:target_count]

    keep_audio = _env_flag("BEATSYNC_YOUTUBE_KEEP_AUDIO", True)
    max_audio_frac = _env_float("BEATSYNC_YOUTUBE_MAX_AUDIO_FRACTION", 0.25)
    audio_budget = max(1, int(round(len(chosen) * max_audio_frac)))
    audio_used = 0
    demucs = _env_flag("BEATSYNC_YOUTUBE_DEMUCS", False)
    silence_threshold = _env_float("BEATSYNC_YOUTUBE_SILENCE_THRESHOLD", 0.9)

    plans: List[ClipPlan] = []
    for score, overlay_free, a, b, regions in chosen:
        mode = "silent"
        has_audio = False
        if keep_audio:
            silence = measure_silence_ratio(video_path, a, b)
            if silence >= silence_threshold:
                mode = "silent"
            elif audio_used < audio_budget:
                mode = "demucs" if demucs else "keep"
                has_audio = True
                audio_used += 1
            else:
                mode = "silent"
        plans.append(ClipPlan(
            start=a, end=b,
            has_audio=has_audio,
            overlay_free=overlay_free,
            audio_mode=mode,
        ))

    # Chronological order for reviewability.
    plans.sort(key=lambda p: p.start)
    return plans


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _probe_regions(video_path: str, start: float, end: float) -> List[tuple]:
    frames = _sample_frames(video_path, 1.0, start, end, count=5)
    regions: List[tuple] = []
    for f in frames:
        det = detect_text_overlay(f)
        regions.extend(det.regions)
    return regions


def _compute_text_crop(regions: Sequence[tuple]) -> tuple:
    """Turn overlay bands (top_frac, bottom_frac) into (top_crop, bottom_crop).

    Band coordinates are y positions in [0, 1] from the top of the frame; the
    returned values are the fraction of frame height to remove from the top and
    bottom so the band is excluded. Total removal is capped at half the frame.
    """
    top_crop = 0.0
    bottom_crop = 0.0
    for (t, b) in regions:
        center = (t + b) / 2.0
        if center > 0.5:
            # Band sits in the lower half — keep the area above it.
            bottom_crop = max(bottom_crop, (1.0 - t) + _TEXT_REGION_PAD)
        else:
            # Band sits in the upper half — keep the area below it.
            top_crop = max(top_crop, b + _TEXT_REGION_PAD)
    top_crop = min(0.5, max(0.0, top_crop))
    bottom_crop = min(0.5, max(0.0, bottom_crop))
    if top_crop + bottom_crop > 0.5:
        scale = 0.5 / (top_crop + bottom_crop)
        top_crop *= scale
        bottom_crop *= scale
    return top_crop, bottom_crop


def _even(value: int) -> int:
    return value - (value % 2)


def _render_clip(
    video_path: str,
    plan: ClipPlan,
    out_path: str,
    crop_top_frac: float,
    crop_bottom_frac: float,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    dur = max(0.05, plan.end - plan.start)
    vf_parts = []
    keep_frac = 1.0 - crop_top_frac - crop_bottom_frac
    if 0.0 < keep_frac < 0.999:
        width, height = get_video_resolution(video_path) or (0, 0)
        crop_h = _even(int(height * keep_frac)) if height else 0
        if crop_h >= 16 and crop_h < _even(height):
            vf_parts.append(f"crop={_even(int(width))}:{crop_h}:0:{_even(int(height * crop_top_frac))}")
    vf = ",".join(vf_parts)

    cmd = [
        FFMPEG_PATH, "-nostdin", "-hide_banner", "-y",
        "-ss", f"{plan.start:.3f}", "-t", f"{dur:.3f}", "-i", video_path,
    ]
    if vf:
        cmd += ["-vf", vf]
    if plan.audio_mode == "keep":
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264", "-preset", os.environ.get("BEATSYNC_CPU_PRESET", "veryfast"),
        "-crf", str(int(_env_float("BEATSYNC_YOUTUBE_CRF", 20))),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out_path,
    ]
    _run(cmd, log)


def build_clips_from_video(
    video_path: str,
    out_dir: str,
    target_count: int = TARGET_CLIP_COUNT,
    crop_text: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Split ``video_path`` into up to ``target_count`` refreshed clips.

    Returns the clip paths (sorted). Best-effort: never raises for individual
    clip failures.
    """
    os.makedirs(out_dir, exist_ok=True)
    plans = build_clip_plan(video_path, target_count=target_count, log=log)
    if not plans:
        raise RuntimeError("No usable clip windows were found in the source video.")

    clips: List[str] = []
    for i, plan in enumerate(plans):
        crop_top = crop_bottom = 0.0
        if crop_text and not plan.overlay_free:
            regions = _probe_regions(video_path, plan.start, plan.end)
            crop_top, crop_bottom = _compute_text_crop(regions)

        out_path = os.path.join(out_dir, f"ytclip_{i:02d}.mp4")
        _render_clip(video_path, plan, out_path, crop_top, crop_bottom, log)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            plan.video_path = out_path
            clips.append(out_path)
            if log:
                tag = "🔇" if plan.audio_mode == "silent" else "🔊"
                if plan.overlay_free:
                    txt = "text-free"
                elif crop_top + crop_bottom > 0:
                    txt = f"cropped {int((crop_top + crop_bottom) * 100)}%"
                else:
                    txt = "text-kept"
                log(f"   ✂️  clip {i + 1:02d}/{len(plans)}  "
                    f"{plan.start:6.1f}-{plan.end:6.1f}s  {tag} {txt}")
    if log:
        kept = sum(1 for p in plans if p.audio_mode != "silent")
        free = sum(1 for p in plans if p.overlay_free)
        log(f"   ✅ {len(clips)} clips built  ({free} text-free, {kept} with audio)")
    return sorted(clips)


def parse_links(text: str) -> List[str]:
    """Split a blob of pasted text into individual http(s) links (deduped, in order)."""
    parts = re.split(r"[\s,;]+", text or "")
    links: List[str] = []
    seen = set()
    for p in parts:
        p = p.strip()
        if not p or p in seen:
            continue
        if p.lower().startswith("http://") or p.lower().startswith("https://"):
            links.append(p)
            seen.add(p)
    return links


def prepare_clips_from_sources(
    links: Sequence[str],
    session_dir: str,
    target_count: int = TARGET_CLIP_COUNT,
    log: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Download several links in turn, keep only usable footage, and build clips.

    Every link is downloaded; clips are then cut from all sources and the
    ``target_count`` best windows are kept, preferring shots with no baked-in
    text overlays and spreading the count across sources.
    """
    links = [l.strip() for l in (links or []) if l and l.strip()]
    if not links:
        raise RuntimeError("No links provided.")

    src_root = os.path.join(session_dir, "youtube")
    clips_root = os.path.join(session_dir, "youtube_clips")
    os.makedirs(clips_root, exist_ok=True)

    downloaded: List[tuple] = []
    for i, url in enumerate(links):
        if log:
            log(f"   🔗 Source {i + 1}/{len(links)}: {url}")
        try:
            video = download_media(url, os.path.join(src_root, f"src{i:02d}"), log=log)
        except Exception as e:
            if log:
                log(f"   ⚠️  Source {i + 1} skipped: {e}")
            continue
        if log:
            log(f"   📼 Source {i + 1}: {os.path.basename(video)} "
                f"({get_video_duration(video):.1f}s)")
        downloaded.append((i, video))

    if not downloaded:
        raise RuntimeError("None of the provided links could be downloaded.")

    # Spread the target clip count evenly across the sources that worked.
    n = len(downloaded)
    base, extra = divmod(target_count, n)
    per_source = [max(1, base + (1 if i < extra else 0)) for i in range(n)]

    clips: List[str] = []
    for (src_idx, video), want in zip(downloaded, per_source):
        sub = os.path.join(clips_root, f"src{src_idx:02d}")
        try:
            got = build_clips_from_video(video, sub, target_count=want, log=log)
        except Exception as e:
            if log:
                log(f"   ⚠️  Source {src_idx + 1} produced no clips: {e}")
            continue
        clips.extend(got)

    if not clips:
        raise RuntimeError("No usable clips could be built from the provided links.")
    if log:
        log(f"   ✅ {len(clips)} clips from {len(downloaded)} link(s)")
    return sorted(clips)


def prepare_youtube_clips(
    url: str,
    session_dir: str,
    target_count: int = TARGET_CLIP_COUNT,
    log: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Download ``url`` (direct link or yt-dlp site) and return ~``target_count`` clips."""
    return prepare_clips_from_sources([url], session_dir, target_count=target_count, log=log)
